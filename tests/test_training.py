"""DataLoaderと学習ループ（Stage I-A/I-B, Stage II）のテスト。"""

from __future__ import annotations

import io
import zipfile

import pytest
import torch
import yaml
from PIL import Image

from src.data import db as db_module
from src.data.dataset import MultiPatchDataset, PatchDataset, select_image_rows
from src.data.ingest import ingest
from src.data.repository import MetadataRepository
from src.experiment import builders
from src.experiment.config import Config
from src.train.train_stage1 import TrainerStage1
from src.train.train_stage2 import TrainerStage2


def _png(width=96, height=96, color=(120, 80, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def project(tmp_path):
    """GenImage風のZIPを1つ作り、ingest済みのDBとconfigを用意する。"""
    root = tmp_path / "Extreme Pro"
    root.mkdir()
    with zipfile.ZipFile(root / "genimage-stable-diffusion-v1-4.zip", "w") as zf:
        for i in range(12):
            zf.writestr(f"train/ai/ai_{i:03d}.png", _png(color=(200, 50, 50)))
            zf.writestr(f"train/nature/nat_{i:03d}.png", _png(color=(50, 200, 50)))
        for i in range(4):
            zf.writestr(f"val/ai/ai_{i:03d}.png", _png(color=(200, 50, 50)))
            zf.writestr(f"val/nature/nat_{i:03d}.png", _png(color=(50, 200, 50)))

    data = Config.load("configs/base.yaml").as_dict()
    data["paths"]["dataset_root"] = str(root)
    data["paths"]["db_path"] = str(tmp_path / "test.sqlite3")
    data["dataset"]["sources"] = [
        {"name": "SDv1.4", "path": "genimage-stable-diffusion-v1-4.zip", "category": "diffusion", "type": "auto"}
    ]
    # 小さなフィクスチャでも val split が空にならないようにする
    data["dataset"]["split"]["val_ratio_from_train"] = 0.5
    data["augmentation"]["jpeg"]["enabled"] = False
    data["model"]["conditional_unet"]["base_channels"] = 8
    data["train"].update(
        {"batch_size": 2, "num_workers": 0, "num_epochs": 1, "max_eval_steps": 1, "save_every_epoch": 1}
    )

    db_module.init_db(data["paths"]["db_path"])
    path = tmp_path / "test_base.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    ingest(Config.load(path), progress=False)
    return tmp_path, data


def _write_config(tmp_path, data, name, **overrides):
    merged = yaml.safe_load(yaml.safe_dump(data))
    for key, value in overrides.items():
        merged[key] = {**merged.get(key, {}), **value} if isinstance(value, dict) else value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(merged, allow_unicode=True), encoding="utf-8")
    return Config.load(path)


# ------------------------------------------------------------------ Dataset


def test_select_rows_picks_train_generator_and_its_real_subset(project):
    tmp_path, data = project
    config = Config.load(tmp_path / "test_base.yaml")
    with MetadataRepository(config.get("paths.db_path")) as repo:
        ai_rows = select_image_rows(config, repo, label="ai", split="train")
        real_rows = select_image_rows(config, repo, label="real", split="train")
        both = select_image_rows(config, repo, label=None, split="train")

    assert {r["generator"] for r in ai_rows} == {"SDv1.4"}
    assert {r["generator"] for r in real_rows} == {"ImageNet(real)@SDv1.4"}
    assert len(both) == len(ai_rows) + len(real_rows)


def test_excluded_generators_are_not_selected(project):
    """【OI-1】excluded_generators に入れた生成器は学習対象から外れること。"""
    tmp_path, data = project
    config = _write_config(tmp_path, data, "excl.yaml", dataset={**data["dataset"], "excluded_generators": ["SDv1.4"]})
    with MetadataRepository(config.get("paths.db_path")) as repo:
        assert select_image_rows(config, repo, label="ai", split="train") == []


def test_patch_dataset_returns_patch_and_label(project):
    tmp_path, data = project
    config = Config.load(tmp_path / "test_base.yaml")
    sampler = builders.build_patch_sampler(config)
    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = select_image_rows(config, repo, label=None, split="train")

    dataset = PatchDataset(rows, sampler)
    patch, target = dataset[0]
    assert patch.shape == (3, 64, 64)
    assert patch.max() > 1.0                      # 0〜255スケールのまま
    assert target.item() in (0.0, 1.0)

    labels = {dataset[i][1].item() for i in range(len(dataset))}
    assert labels == {0.0, 1.0}                   # 写真とAI画像の両方が含まれる


def test_patch_dataset_rejects_empty_rows():
    with pytest.raises(ValueError, match="0件"):
        PatchDataset([], builders.build_patch_sampler(Config.load("configs/base.yaml")))


def test_multipatch_dataset_returns_p_patches(project):
    tmp_path, data = project
    config = Config.load(tmp_path / "test_base.yaml")
    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = select_image_rows(config, repo, label=None, split="test")

    dataset = MultiPatchDataset(rows, builders.build_patch_sampler(config), num_patches=16)
    patches, target, image_id = dataset[0]
    assert patches.shape == (16, 3, 64, 64)
    assert target.item() in (0.0, 1.0)
    assert isinstance(image_id, int)
    # 同じ画像なら毎回同じパッチ位置（NFR-1）
    assert torch.equal(dataset[0][0], dataset[0][0])


def test_dataset_reads_images_from_inside_zip(project):
    tmp_path, data = project
    config = Config.load(tmp_path / "test_base.yaml")
    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = select_image_rows(config, repo, label="ai", split="train")
    assert all(".zip!" in row["filepath"] for row in rows)
    assert PatchDataset(rows, builders.build_patch_sampler(config))[0][0].shape == (3, 64, 64)


# ------------------------------------------------------------- Stage I 学習


def test_stage1_trains_and_records_to_db(project):
    tmp_path, data = project
    config = _write_config(
        tmp_path, data, "s1.yaml",
        experiment={"name": "t_stage1", "stage": "stage1_photo"},
        train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_s1"), "num_epochs": 2},
    )

    with TrainerStage1(config, "photo") as trainer:
        run_id = trainer.fit(max_steps_per_epoch=2)
        repo = trainer.repo
        run = repo.conn.execute("SELECT * FROM training_runs WHERE run_id = ?", (run_id,)).fetchone()
        logs = repo.conn.execute(
            "SELECT DISTINCT metric_name FROM training_logs WHERE run_id = ?", (run_id,)
        ).fetchall()
        checkpoints = repo.conn.execute(
            "SELECT * FROM model_checkpoints WHERE run_id = ?", (run_id,)
        ).fetchall()
        experiment = repo.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (trainer.experiment_id,)
        ).fetchone()

    assert run["status"] == "completed" and run["completed_at"]
    assert {"nll_loss", "val_nll", "lr"} <= {row["metric_name"] for row in logs}
    assert len(checkpoints) >= 2
    assert experiment["stage"] == "stage1_photo"
    assert experiment["seed"] == 42
    assert experiment["config_snapshot"]                      # NFR-7 トレーサビリティ
    assert (tmp_path / "ckpt_s1" / "best.pt").exists()


def test_stage1_photo_and_ai_use_different_data(project):
    tmp_path, data = project
    photo = _write_config(tmp_path, data, "s1p.yaml",
                          train={**data["train"], "checkpoint_dir": str(tmp_path / "c1")})
    ai = _write_config(tmp_path, data, "s1a.yaml",
                       train={**data["train"], "checkpoint_dir": str(tmp_path / "c2")})

    with TrainerStage1(photo, "photo") as p_trainer, TrainerStage1(ai, "ai") as a_trainer:
        assert p_trainer.stage == "stage1_photo" and a_trainer.stage == "stage1_ai"
        photo_paths = {row["filepath"] for row in p_trainer.train_loader.dataset.rows}
        ai_paths = {row["filepath"] for row in a_trainer.train_loader.dataset.rows}
    assert photo_paths and ai_paths and not (photo_paths & ai_paths)


def test_stage1_resume_continues_from_checkpoint(project):
    tmp_path, data = project
    config = _write_config(tmp_path, data, "s1r.yaml",
                           train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_r")})

    with TrainerStage1(config, "photo") as trainer:
        trainer.fit(num_epochs=1, max_steps_per_epoch=1)
        checkpoint = tmp_path / "ckpt_r" / "best.pt"

    with TrainerStage1(config, "photo") as resumed:
        resumed.load_checkpoint(checkpoint, resume=True)
        assert resumed.start_epoch == 1
        resumed.fit(num_epochs=2, max_steps_per_epoch=1)
        assert resumed.start_epoch == 1                        # 2エポック目のみ実行された


def test_failed_run_is_marked_failed(project, monkeypatch):
    tmp_path, data = project
    config = _write_config(tmp_path, data, "s1f.yaml",
                           train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_f")})

    with TrainerStage1(config, "photo") as trainer:
        monkeypatch.setattr(trainer, "train_step", lambda batch: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            trainer.fit(num_epochs=1)
        status = trainer.repo.conn.execute(
            "SELECT status FROM training_runs WHERE run_id = ?", (trainer.run_id,)
        ).fetchone()["status"]
    assert status == "failed"


# ------------------------------------------------------------ Stage II 学習


def _stage1_checkpoints(tmp_path, data):
    """Stage II のテスト用に pθ / qφ のチェックポイントを作る。"""
    paths = {}
    for target, directory in (("photo", "ck_p"), ("ai", "ck_a")):
        config = _write_config(tmp_path, data, f"s1_{target}.yaml",
                               train={**data["train"], "checkpoint_dir": str(tmp_path / directory)})
        with TrainerStage1(config, target) as trainer:
            trainer.fit(num_epochs=1, max_steps_per_epoch=1)
        paths[target] = str(tmp_path / directory / "best.pt")
    return paths


def test_stage2_trains_classifier_and_keeps_models_frozen(project):
    tmp_path, data = project
    checkpoints = _stage1_checkpoints(tmp_path, data)
    config = _write_config(
        tmp_path, data, "s2.yaml",
        experiment={"name": "t_stage2", "stage": "stage2_classifier"},
        train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_s2")},
        stage2={
            "photo_model_checkpoint": checkpoints["photo"],
            "ai_model_checkpoint": checkpoints["ai"],
            "use_photo_model": True,
            "use_ai_model": True,
            "freeze_conditional_models": True,
        },
    )

    with TrainerStage2(config) as trainer:
        assert trainer.classifier.in_channels == 120          # 【OI-2】案A: 60ch × 2モデル
        before_photo = trainer.photo_model.head.weight.clone()
        before_classifier = trainer.classifier.head.weight.clone()

        run_id = trainer.fit(num_epochs=1, max_steps_per_epoch=2)

        # Frozen: 条件付きモデルは更新されず、分類器だけが更新される
        assert torch.equal(trainer.photo_model.head.weight, before_photo)
        assert not torch.equal(trainer.classifier.head.weight, before_classifier)
        assert set(trainer.trainable_modules()) == {"classifier"}

        metrics = {
            row["metric_name"]
            for row in trainer.repo.conn.execute(
                "SELECT DISTINCT metric_name FROM training_logs WHERE run_id = ?", (run_id,)
            )
        }
    assert {"bce_loss", "accuracy", "val_bce", "val_accuracy"} <= metrics


def test_stage2_ablation_a_single_model(project):
    """Ablation-A: qφのみを使う設定では入力が60chになること。"""
    tmp_path, data = project
    checkpoints = _stage1_checkpoints(tmp_path, data)
    config = _write_config(
        tmp_path, data, "s2_a.yaml",
        train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_s2a")},
        stage2={
            "ai_model_checkpoint": checkpoints["ai"],
            "use_photo_model": False,
            "use_ai_model": True,
            "freeze_conditional_models": True,
        },
    )

    with TrainerStage2(config) as trainer:
        assert trainer.photo_model is None
        assert trainer.classifier.in_channels == 60
        trainer.fit(num_epochs=1, max_steps_per_epoch=1)


def test_stage2_ablation_d_full_finetuning(project):
    """Ablation-D: freeze=False なら条件付きモデルも更新され、保存対象に含まれること。"""
    tmp_path, data = project
    checkpoints = _stage1_checkpoints(tmp_path, data)
    config = _write_config(
        tmp_path, data, "s2_d.yaml",
        train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_s2d")},
        stage2={
            "photo_model_checkpoint": checkpoints["photo"],
            "ai_model_checkpoint": checkpoints["ai"],
            "use_photo_model": True,
            "use_ai_model": True,
            "freeze_conditional_models": False,
        },
    )

    with TrainerStage2(config) as trainer:
        before = trainer.photo_model.head.weight.clone()
        assert set(trainer.trainable_modules()) == {"classifier", "photo_model", "ai_model"}
        trainer.fit(num_epochs=1, max_steps_per_epoch=2)
        assert not torch.equal(trainer.photo_model.head.weight, before)

    saved = torch.load(tmp_path / "ckpt_s2d" / "best.pt", weights_only=False)
    assert {"classifier", "photo_model", "ai_model"} <= set(saved["modules"])
