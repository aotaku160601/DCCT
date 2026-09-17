"""学習ループの共通処理（03_詳細設計書 2節 `Trainer`）。

実験ID・実行IDの発行、エポックごとのDBログ、チェックポイント保存と再開、
中断時のステータス更新をここに集約する（FR-8 / 02_基本設計書 8節）。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from ..data.repository import MetadataRepository
from ..experiment.builders import build_device
from ..experiment.config import Config
from ..experiment.device import device_summary
from ..experiment.paths import resolve
from ..experiment.tracker import git_commit, set_seed

logger = logging.getLogger(__name__)


class Trainer(ABC):
    """学習ループの骨格。サブクラスは `setup` / `train_step` / `eval_step` を実装する。

    Attributes:
        stage: `experiments.stage` に入る値
        best_metric_name: チェックポイント選択に使う検証指標名
        better_is_lower: その指標が小さいほど良いか
    """

    stage: str = "stage1_photo"
    best_metric_name: str = "val_loss"
    better_is_lower: bool = True

    def __init__(self, config: Config, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or build_device(config)
        set_seed(int(config.get("project.seed", 42)))

        self.repo = MetadataRepository(config.get("paths.db_path"))
        self.experiment_id = self.repo.register_experiment(
            name=config.get("experiment.name", self.stage),
            stage=self.stage,
            config_snapshot=config.snapshot(),
            git_commit=git_commit(),
            seed=int(config.get("project.seed", 42)),
        )
        self.run_id = self.repo.create_training_run(self.experiment_id)

        self.checkpoint_dir = resolve(config.get("train.checkpoint_dir", f"checkpoints/{self.stage}"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.global_step = 0
        self.best_metric: float | None = None

        logger.info(
            "experiment_id=%d run_id=%d stage=%s device=%s",
            self.experiment_id,
            self.run_id,
            self.stage,
            device_summary(self.device),
        )
        self.setup()

    # ------------------------------------------------------------------
    # サブクラスが実装する部分
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        """モデル・オプティマイザ・DataLoaderを構築する。"""

    @abstractmethod
    def train_step(self, batch: Any) -> dict[str, float]:
        """1ミニバッチ分の学習を行い、指標を返す。"""

    @abstractmethod
    def eval_step(self, batch: Any) -> dict[str, float]:
        """1ミニバッチ分の検証を行い、指標を返す。"""

    @abstractmethod
    def trainable_modules(self) -> dict[str, torch.nn.Module]:
        """チェックポイントに保存するモジュール。"""

    # ------------------------------------------------------------------
    # 共通処理
    # ------------------------------------------------------------------

    def _run_epoch(
        self, loader: Iterable[Any], step_fn, max_steps: int | None, training: bool
    ) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        for index, batch in enumerate(loader):
            if max_steps is not None and index >= max_steps:
                break
            metrics = step_fn(batch)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
            if training:
                self.global_step += 1
        if count == 0:
            return {}
        return {key: value / count for key, value in totals.items()}

    def fit(self, num_epochs: int | None = None, max_steps_per_epoch: int | None = None) -> int:
        """学習を実行し、run_idを返す。"""
        num_epochs = num_epochs if num_epochs is not None else int(self.config.get("train.num_epochs", 20))
        max_steps_per_epoch = (
            max_steps_per_epoch
            if max_steps_per_epoch is not None
            else self.config.get("train.max_steps_per_epoch", None)
        )
        max_eval_steps = self.config.get("train.max_eval_steps", None)
        save_every = int(self.config.get("train.save_every_epoch", 1))

        try:
            for epoch in range(self.start_epoch, num_epochs):
                started = time.perf_counter()

                for module in self.trainable_modules().values():
                    module.train()
                train_metrics = self._run_epoch(
                    self.train_loader, self.train_step, max_steps_per_epoch, training=True
                )

                for module in self.trainable_modules().values():
                    module.eval()
                with torch.no_grad():
                    val_metrics = self._run_epoch(
                        self.val_loader, self.eval_step, max_eval_steps, training=False
                    )

                elapsed = time.perf_counter() - started
                metrics = {**train_metrics, **val_metrics, "epoch_seconds": elapsed}
                self._log_metrics(epoch, metrics)

                if save_every and (epoch + 1) % save_every == 0:
                    self.save_checkpoint(epoch, metrics)
                self._update_best(epoch, metrics)

            self._finish_run("completed")
        except BaseException:
            # 失敗の原因をログファイルにも残す（標準エラーだけだとnohup.outにしか出ない）
            logger.exception("学習が異常終了しました")
            self._finish_run("failed")
            raise
        return self.run_id

    def _finish_run(self, status: str) -> None:
        """実行ステータスを更新する。ここでの失敗は握りつぶす（元の例外を隠さないため）。"""
        try:
            self.repo.finish_training_run(self.run_id, status)
        except Exception:
            logger.exception("training_runs のステータス更新に失敗しました（status=%s）", status)

    def _log_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        summary = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("epoch %d | %s", epoch, summary)

        # DBへの記録は補助的なもの。ここで失敗しても学習そのものは続ける
        # （数時間の学習が記録の失敗で落ちるのを防ぐ）。ログには必ず残っている。
        try:
            for name, value in metrics.items():
                self.repo.log_training_metric(self.run_id, epoch, name, value, step=self.global_step)
            self.repo.log_training_metric(
                self.run_id, epoch, "lr", self.optimizer.param_groups[0]["lr"], step=self.global_step
            )
        except Exception:
            logger.exception("指標のDB記録に失敗しました（学習は継続します）")

    def _update_best(self, epoch: int, metrics: dict[str, float]) -> None:
        value = metrics.get(self.best_metric_name)
        if value is None:
            return
        improved = (
            self.best_metric is None
            or (value < self.best_metric if self.better_is_lower else value > self.best_metric)
        )
        if improved:
            self.best_metric = value
            self.save_checkpoint(epoch, metrics, filename="best.pt")

    def checkpoint_state(self, epoch: int, metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "metrics": metrics,
            "config": self.config.as_dict(),
            "modules": {name: module.state_dict() for name, module in self.trainable_modules().items()},
            "optimizer": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
        }

    def save_checkpoint(
        self, epoch: int, metrics: dict[str, float], filename: str | None = None
    ) -> Path:
        filename = filename or f"run{self.run_id}_epoch{epoch:03d}.pt"
        path = self.checkpoint_dir / filename
        # まずファイルを確実に残す。DBへの登録が失敗しても学習結果は失われない
        torch.save(self.checkpoint_state(epoch, metrics), path)
        logger.info("チェックポイントを保存しました: %s", path)

        try:
            self.repo.save_checkpoint_meta(
                self.run_id,
                epoch,
                str(path),
                metric_name=self.best_metric_name,
                metric_value=metrics.get(self.best_metric_name),
            )
        except Exception:
            logger.exception("チェックポイントのDB記録に失敗しました（ファイルは保存済み）")
        return path

    def load_checkpoint(self, path: str | Path, resume: bool = True) -> dict[str, Any]:
        """チェックポイントを読み込む。`resume=True` なら学習状態も復元する（03_詳細設計書 6節）。"""
        state = torch.load(resolve(path), map_location=self.device, weights_only=False)

        modules = self.trainable_modules()
        for name, module_state in state["modules"].items():
            if name in modules:
                modules[name].load_state_dict(module_state)

        if resume:
            self.optimizer.load_state_dict(state["optimizer"])
            self.start_epoch = int(state["epoch"]) + 1
            self.global_step = int(state.get("global_step", 0))
            self.best_metric = state.get("best_metric")
            logger.info("エポック %d から再開します: %s", self.start_epoch, path)
        return state

    def make_loader(self, dataset, shuffle: bool) -> DataLoader:
        num_workers = int(self.config.get("train.num_workers", 0))
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("train.batch_size", 16)),
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )

    def close(self) -> None:
        self.repo.close()

    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
