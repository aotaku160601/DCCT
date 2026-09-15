"""評価の実行（03_詳細設計書 2節 `Evaluator`）。

- cross-generator評価（Table 1相当）
- ロバスト性評価（Figure 6相当）

結果は `evaluation_results` テーブルに `eval_type` で区別して保存する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from ..data.dataset import real_generator_names
from ..data.repository import MetadataRepository
from ..experiment.config import Config
from ..preprocess.patch_sampler import jpeg_compress
from .metrics import compute_metrics, tune_threshold
from .predictor import DCCTPredictor, ImageTransform

logger = logging.getLogger(__name__)

_REAL_GENERATOR_PREFIX = "ImageNet(real)"


@dataclass
class GeneratorResult:
    """1生成器分の評価結果。"""

    generator: str
    metrics: dict[str, Any]
    perturbation_type: str = "none"
    perturbation_level: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def downsample_transform(ratio: float) -> ImageTransform:
    """倍率 `ratio` に縮小してから元の解像度に戻す（Figure 6のダウンサンプリング）。"""

    def transform(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        small = F.interpolate(
            image.unsqueeze(0), scale_factor=ratio, mode="bilinear", align_corners=False
        )
        restored = F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
        return restored.squeeze(0).clamp(0, 255)

    return transform


def jpeg_transform(quality: int) -> ImageTransform:
    """指定QFでJPEG再エンコードする（Figure 6のJPEG圧縮）。"""

    def transform(image: torch.Tensor) -> torch.Tensor:
        return jpeg_compress(image, quality)

    return transform


class Evaluator:
    """学習済みモデルを各生成器・各劣化条件で評価し、DBへ保存する。"""

    def __init__(
        self,
        config: Config,
        run_id: int,
        classifier_checkpoint: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.repo = MetadataRepository(config.get("paths.db_path"))
        self.predictor = DCCTPredictor(config, classifier_checkpoint, device)
        self.threshold = float(config.get("inference.threshold", 0.5))
        self.max_images = config.get("evaluate.max_images_per_generator", None)

    def close(self) -> None:
        self.repo.close()

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 評価対象の組み立て
    # ------------------------------------------------------------------

    def target_generators(self) -> list[str]:
        """評価対象の生成器（excluded_generators を除く）。"""
        excluded = set(self.config.get("dataset.excluded_generators", []))
        return [g for g in self.config.get("dataset.eval_generators", []) if g not in excluded]

    def _real_generator_for(self, generator: str) -> str:
        """その生成器に対応する実写generator名（ingestの命名規則に合わせる）。"""
        if self.config.get("dataset.real_generator_naming", "per_source") == "shared":
            return _REAL_GENERATOR_PREFIX
        return f"{_REAL_GENERATOR_PREFIX}@{generator}"

    def _rows_for(self, generator: str, split: str) -> list[dict[str, Any]]:
        """生成器のAI画像と、それに対応する実写画像を集める。

        `evaluate.max_images_per_generator` はラベルごとに適用する。まとめてLIMITすると
        片方のクラスだけが残り、AccuracyやAUCが意味を持たなくなるため。
        """
        rows: list[dict[str, Any]] = []
        for label, name in (("ai", generator), ("real", self._real_generator_for(generator))):
            found = self.repo.list_images(
                label=label, split=split, generator_names=[name], limit=self.max_images
            )
            rows.extend(dict(row) for row in found)
        return rows

    # ------------------------------------------------------------------
    # 閾値チューニング
    # ------------------------------------------------------------------

    def calibrate_threshold(self) -> float:
        """val split上でYouden指数を最大化する閾値τを決める（test splitには触れない）。"""
        if not self.config.get("evaluate.tune_threshold", True):
            return self.threshold

        rows: list[dict[str, Any]] = []
        for generator in self.config.get("dataset.train_generators", []):
            rows.extend(self._rows_for(generator, "val"))
        if not rows:
            logger.warning("val splitに画像がないため、閾値は既定値 %.3f のままにします", self.threshold)
            return self.threshold

        scores, labels = self.predictor.score_rows(rows)
        self.threshold = tune_threshold(scores, labels, default=self.threshold)
        logger.info("閾値τを %.4f にチューニングしました（val %d枚）", self.threshold, len(labels))
        return self.threshold

    # ------------------------------------------------------------------
    # 各評価モード
    # ------------------------------------------------------------------

    def evaluate_generator(
        self,
        generator: str,
        split: str = "test",
        transform: ImageTransform | None = None,
    ) -> GeneratorResult | None:
        rows = self._rows_for(generator, split)
        if not rows:
            logger.warning("評価対象の画像がありません: %s（split=%s）", generator, split)
            return None

        scores, labels = self.predictor.score_rows(rows, transform=transform)
        metrics = compute_metrics(scores, labels, self.threshold)
        logger.info(
            "%s: acc=%.4f auc=%s ap=%s (n=%d)",
            generator,
            metrics["accuracy"],
            f"{metrics['auc']:.4f}" if metrics["auc"] is not None else "N/A",
            f"{metrics['ap']:.4f}" if metrics["ap"] is not None else "N/A",
            metrics["num_samples"],
        )
        return GeneratorResult(generator=generator, metrics=metrics)

    def _save(self, result: GeneratorResult, eval_type: str, split: str) -> None:
        generator_id = self.repo.generator_id_by_name(
            self.repo.conn.execute("SELECT dataset_id FROM datasets LIMIT 1").fetchone()["dataset_id"],
            result.generator,
        )
        if generator_id is None:
            logger.warning("generators に %s が見つからないため保存をスキップします", result.generator)
            return

        self.repo.save_evaluation_result(
            self.run_id,
            generator_id,
            eval_type,
            split,
            accuracy=result.metrics["accuracy"],
            auc=result.metrics["auc"],
            ap=result.metrics["ap"],
            num_samples=result.metrics["num_samples"],
            perturbation_type=result.perturbation_type,
            perturbation_level=result.perturbation_level,
        )

    def run_cross_generator(self, split: str = "test", eval_type: str = "cross_generator") -> list[GeneratorResult]:
        """Table 1相当: 各生成器に対するAccuracy/AUC/APを算出して保存する。"""
        self.calibrate_threshold()

        results: list[GeneratorResult] = []
        for generator in self.target_generators():
            result = self.evaluate_generator(generator, split)
            if result is None:
                continue
            self._save(result, eval_type, split)
            results.append(result)

        if results:
            mean_accuracy = sum(r.metrics["accuracy"] for r in results) / len(results)
            logger.info("平均Accuracy（%d生成器）: %.4f", len(results), mean_accuracy)
        return results

    def run_robustness(self, split: str = "test") -> list[GeneratorResult]:
        """Figure 6相当: JPEG圧縮・ダウンサンプリングに対する精度を測る。"""
        self.calibrate_threshold()

        qualities = self.config.get("evaluate.robustness.jpeg_qualities", [100, 90, 80, 70])
        ratios = self.config.get("evaluate.robustness.downsample_ratios", [0.9, 0.8, 0.7, 0.6])

        conditions: list[tuple[str, float, ImageTransform | None]] = [("none", None, None)]
        conditions += [("jpeg", float(q), jpeg_transform(int(q))) for q in qualities]
        conditions += [("downsample", float(r), downsample_transform(float(r))) for r in ratios]

        results: list[GeneratorResult] = []
        for generator in self.target_generators():
            for perturbation_type, level, transform in conditions:
                result = self.evaluate_generator(generator, split, transform=transform)
                if result is None:
                    continue
                result.perturbation_type = perturbation_type
                result.perturbation_level = level
                logger.info("  %s / %s=%s → acc=%.4f", generator, perturbation_type, level, result.metrics["accuracy"])
                self._save(result, "robustness", split)
                results.append(result)
        return results
