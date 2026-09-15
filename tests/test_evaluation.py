"""評価（cross-generator / robustness / ablation）とレポート出力のテスト。"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.data import db as db_module
from src.data.ingest import ingest
from src.data.repository import MetadataRepository
from src.eval.evaluator import Evaluator, downsample_transform, jpeg_transform
from src.eval.metrics import compute_metrics, tune_threshold
from src.experiment.config import Config
from src.report.report_builder import add_mean_row, build_report, fetch_cross_generator
from src.train.train_stage2 import TrainerStage2


# ------------------------------------------------------------------ metrics


def test_compute_metrics_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    metrics = compute_metrics(scores, labels, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["ap"] == 1.0
    assert metrics["num_samples"] == 4


def test_compute_metrics_single_class_has_no_auc():
    metrics = compute_metrics(np.array([0.2, 0.3]), np.array([1.0, 1.0]), threshold=0.5)
    assert metrics["auc"] is None and metrics["ap"] is None
    assert metrics["accuracy"] == 0.0


def test_tune_threshold_finds_separating_point():
    scores = np.array([0.1, 0.15, 0.7, 0.75])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    threshold = tune_threshold(scores, labels)
    assert 0.15 < threshold <= 0.7
    assert ((scores > threshold).astype(float) == labels).all()


def test_tune_threshold_falls_back_when_single_class():
    assert tune_threshold(np.array([0.3, 0.4]), np.array([1.0, 1.0]), default=0.42) == 0.42


# ------------------------------------------------------- 劣化処理（Figure 6）


def test_jpeg_transform_keeps_shape_and_changes_pixels():
    image = torch.rand(3, 96, 96) * 255
    out = jpeg_transform(70)(image)
    assert out.shape == image.shape
    assert not torch.equal(out, image.round())


def test_downsample_transform_restores_resolution_and_blurs():
    image = torch.rand(3, 96, 96) * 255
    out = downsample_transform(0.6)(image)
    assert out.shape == image.shape
    assert out.min() >= 0 and out.max() <= 255
    # 高周波が落ちるので、隣接画素差の大きさが小さくなる
    assert out.diff(dim=-1).abs().mean() < image.diff(dim=-1).abs().mean()


# ------------------------------------------------------------ 評価の通し実行


def _png(color, size=96) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def trained(tmp_path):
    """2生成器分のデータをingestし、Stage IIを1エポックだけ学習した状態を作る。"""
    root = tmp_path / "Extreme Pro"
    root.mkdir()
    for archive, ai_color in (("genimage-stable-diffusion-v1-4.zip", (200, 40, 40)), ("BigGAN.zip", (40, 40, 200))):
        with zipfile.ZipFile(root / archive, "w") as zf:
            for i in range(6):
                zf.writestr(f"train/ai/ai_{i}.png", _png(ai_color))
                zf.writestr(f"train/nature/nat_{i}.png", _png((40, 200, 40)))
            for i in range(3):
                zf.writestr(f"val/ai/ai_{i}.png", _png(ai_color))
                zf.writestr(f"val/nature/nat_{i}.png", _png((40, 200, 40)))

    data = Config.load("configs/base.yaml").as_dict()
    data["paths"]["dataset_root"] = str(root)
    data["paths"]["db_path"] = str(tmp_path / "test.sqlite3")
    data["dataset"]["sources"] = [
        {"name": "SDv1.4", "path": "genimage-stable-diffusion-v1-4.zip", "category": "diffusion", "type": "auto"},
        {"name": "BigGAN", "path": "BigGAN.zip", "category": "gan", "type": "auto"},
    ]
    data["dataset"]["eval_generators"] = ["SDv1.4", "BigGAN", "Midjourney"]
    data["dataset"]["split"]["val_ratio_from_train"] = 0.5
    data["augmentation"]["jpeg"]["enabled"] = False
    data["model"]["conditional_unet"]["base_channels"] = 8
    data["inference"]["num_patches"] = 2
    data["train"].update({"batch_size": 2, "num_workers": 0, "num_epochs": 1, "max_eval_steps": 1,
                          "checkpoint_dir": str(tmp_path / "ckpt")})
    data["stage2"] = {"use_photo_model": True, "use_ai_model": True, "freeze_conditional_models": True,
                      "photo_model_checkpoint": None, "ai_model_checkpoint": None}
    data["evaluate"] = {"tune_threshold": True, "max_images_per_generator": 4,
                        "robustness": {"jpeg_qualities": [90], "downsample_ratios": [0.6]}}
    data["experiment"] = {"name": "eval_test", "stage": "stage2_classifier"}

    db_module.init_db(data["paths"]["db_path"])
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    config = Config.load(path)
    ingest(config, progress=False)

    with TrainerStage2(config) as trainer:
        run_id = trainer.fit(num_epochs=1, max_steps_per_epoch=2)
        checkpoint = str(trainer.checkpoint_dir / "best.pt")
    return tmp_path, config, run_id, checkpoint


def test_cross_generator_saves_results_per_generator(trained):
    tmp_path, config, run_id, checkpoint = trained

    with Evaluator(config, run_id, checkpoint) as evaluator:
        results = evaluator.run_cross_generator()
        rows = evaluator.repo.conn.execute(
            """
            SELECT g.name AS generator, er.eval_type, er.split, er.accuracy, er.num_samples,
                   er.perturbation_type
            FROM evaluation_results er JOIN generators g USING(generator_id)
            WHERE er.run_id = ?
            """,
            (run_id,),
        ).fetchall()

    # 【OI-1】Midjourney は excluded_generators により評価対象から外れる
    assert {r.generator for r in results} == {"SDv1.4", "BigGAN"}
    assert {row["generator"] for row in rows} == {"SDv1.4", "BigGAN"}
    assert all(row["eval_type"] == "cross_generator" and row["split"] == "test" for row in rows)
    assert all(row["perturbation_type"] == "none" for row in rows)
    # ラベルごとに上限を適用するので、AI画像と実写画像が同数ずつ入る
    assert all(row["num_samples"] % 2 == 0 and row["num_samples"] > 0 for row in rows)


def test_evaluation_pairs_generator_with_its_own_real_subset(trained):
    tmp_path, config, run_id, checkpoint = trained
    with Evaluator(config, run_id, checkpoint) as evaluator:
        rows = evaluator._rows_for("SDv1.4", "test")

    generators = {row["generator"] for row in rows}
    assert generators == {"SDv1.4", "ImageNet(real)@SDv1.4"}
    assert {row["label"] for row in rows} == {"ai", "real"}


def test_threshold_is_calibrated_on_val_split(trained):
    tmp_path, config, run_id, checkpoint = trained
    with Evaluator(config, run_id, checkpoint) as evaluator:
        before = evaluator.threshold
        after = evaluator.calibrate_threshold()
    assert 0.0 <= after <= 1.0
    assert isinstance(before, float)


def test_threshold_tuning_can_be_disabled(trained):
    tmp_path, config, run_id, checkpoint = trained
    data = config.as_dict()
    data["evaluate"]["tune_threshold"] = False
    data["inference"]["threshold"] = 0.42

    with Evaluator(Config(data), run_id, checkpoint) as evaluator:
        assert evaluator.calibrate_threshold() == 0.42


def test_robustness_records_each_perturbation_level(trained):
    tmp_path, config, run_id, checkpoint = trained

    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_robustness()
        rows = evaluator.repo.conn.execute(
            """
            SELECT DISTINCT er.perturbation_type, er.perturbation_level
            FROM evaluation_results er WHERE er.run_id = ? AND er.eval_type = 'robustness'
            ORDER BY er.perturbation_type
            """,
            (run_id,),
        ).fetchall()

    conditions = {(row["perturbation_type"], row["perturbation_level"]) for row in rows}
    assert ("none", None) in conditions
    assert ("jpeg", 90.0) in conditions
    assert ("downsample", 0.6) in conditions


# ---------------------------------------------------------------- ablation


def test_ablation_runs_variants_and_records_conditions(trained):
    from src.eval.ablation import run_ablation

    tmp_path, config, _, _ = trained
    data = config.as_dict()
    data["ablation"] = {
        "group": "A",
        "variants": [
            {"label": "photo_only", "use_photo_model": True, "use_ai_model": False},
            {"label": "both", "use_photo_model": True, "use_ai_model": True},
        ],
    }
    data["paths"]["checkpoint_root"] = str(tmp_path / "abl")

    results = run_ablation(Config(data), num_epochs=1, max_steps_per_epoch=1)
    assert [r.label for r in results] == ["photo_only", "both"]
    assert all(r.mean_accuracy is not None for r in results)

    with MetadataRepository(data["paths"]["db_path"]) as repo:
        conditions = repo.conn.execute(
            "SELECT ablation_group, parameter_name, parameter_value FROM ablation_configs"
        ).fetchall()
        eval_rows = repo.conn.execute(
            "SELECT COUNT(*) AS n FROM evaluation_results WHERE eval_type = 'ablation'"
        ).fetchone()

    assert {row["ablation_group"] for row in conditions} == {"A"}
    assert {row["parameter_name"] for row in conditions} == {"use_photo_model", "use_ai_model"}
    assert eval_rows["n"] > 0


def test_ablation_rejects_unknown_parameter(trained):
    from src.eval.ablation import build_variant_config

    tmp_path, config, _, _ = trained
    with pytest.raises(ValueError, match="未知のアブレーションパラメータ"):
        build_variant_config(config, {"label": "x", "no_such_param": 1}, str(tmp_path))


# ------------------------------------------------------------------ report


def test_report_writes_csv_markdown_and_figures(trained):
    tmp_path, config, run_id, checkpoint = trained
    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_cross_generator()
        evaluator.run_robustness()

    report_path = build_report(config, run_id, tmp_path / "reports")
    out_dir = report_path.parent

    assert report_path.exists()
    assert (out_dir / "cross_generator.csv").exists()
    assert (out_dir / "robustness.csv").exists()
    assert (out_dir / "robustness_jpeg.png").exists()
    assert (out_dir / "robustness_downsample.png").exists()

    text = report_path.read_text(encoding="utf-8")
    assert "Cross-generator評価（Table 1相当）" in text
    assert "ロバスト性評価（Figure 6相当）" in text
    assert "平均(2生成器)" in text
    assert f"実行ID: {run_id}" in text


def test_report_deduplicates_repeated_evaluations(trained):
    """同じ条件を2回評価しても、レポートには最新の1件だけが載ること。"""
    tmp_path, config, run_id, checkpoint = trained
    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_cross_generator()
        evaluator.run_cross_generator()

        rows = fetch_cross_generator(evaluator.repo, run_id)
        raw = evaluator.repo.conn.execute(
            "SELECT COUNT(*) AS n FROM evaluation_results WHERE run_id = ? AND eval_type = 'cross_generator'",
            (run_id,),
        ).fetchone()["n"]

    assert raw == 4                      # 2生成器 × 2回
    assert len(rows) == 2                # レポート側は最新のみ


def test_add_mean_row_computes_generator_average():
    rows = [
        {"generator": "A", "accuracy": 0.9, "auc": 0.95, "ap": 0.9, "num_samples": 10},
        {"generator": "B", "accuracy": 0.7, "auc": 0.75, "ap": 0.7, "num_samples": 20},
    ]
    with_mean = add_mean_row(rows)
    assert with_mean[-1]["generator"] == "平均(2生成器)"
    assert with_mean[-1]["accuracy"] == pytest.approx(0.8)
    assert with_mean[-1]["num_samples"] == 30
