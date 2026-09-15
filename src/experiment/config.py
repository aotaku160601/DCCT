"""実験設定（YAML）の読み込み・検証。

02_基本設計書 8節「設定管理」／01_要件定義書 NFR-5 に対応する。
全てのハイパーパラメータ・パス・アブレーション条件はYAMLに外出しし、
コード変更なしに実験を切り替えられるようにする。

各configは `extends: <相対パス>` で親configを継承できる（辞書は再帰的にマージ、
スカラー・リストは子の値で上書き）。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

_MAX_EXTENDS_DEPTH = 10


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`base` に `override` を再帰的に重ねた新しい辞書を返す。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_raw(path: Path, _depth: int = 0) -> dict[str, Any]:
    if _depth > _MAX_EXTENDS_DEPTH:
        raise ValueError(f"config の extends が深すぎます（循環参照の可能性）: {path}")

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config のトップレベルはマッピングである必要があります: {path}")

    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data

    parent_path = (path.parent / parent_ref).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"{path} が継承する config が見つかりません: {parent_path}")
    return _deep_merge(_load_raw(parent_path, _depth + 1), data)


class Config:
    """ドット区切りのキーで値を引ける設定オブジェクト。"""

    def __init__(self, data: dict[str, Any], source_path: Path | None = None) -> None:
        self._data = data
        self.source_path = source_path

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"config が見つかりません: {p}")
        cfg = cls(_load_raw(p), source_path=p)
        cfg.validate()
        return cfg

    _MISSING = object()

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """`get("paths.db_path")` のように取得する。既定値なしで欠けていれば KeyError。"""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is Config._MISSING:
                    raise KeyError(f"config に必要なキーがありません: {dotted_key}")
                return default
            node = node[part]
        return node

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def snapshot(self) -> str:
        """`experiments.config_snapshot` に保存する文字列表現（JSON）。"""
        return json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True, default=str)

    def validate(self) -> None:
        """必須キーと値域の最低限の検証を行う。"""
        for key in ("paths.db_path", "project.seed"):
            self.get(key)

        device_type = self.get("device.type", "auto")
        if device_type not in ("auto", "cuda", "mps", "cpu"):
            raise ValueError(f"device.type は auto/cuda/mps/cpu のいずれか: {device_type!r}")

        patch_size = self.get("preprocess.patch_size", 64)
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(f"preprocess.patch_size は正の整数: {patch_size!r}")

        t = self.get("preprocess.truncation_t", 7)
        if not isinstance(t, (int, float)) or t <= 0:
            raise ValueError(f"preprocess.truncation_t は正の数: {t!r}")

        feature_source = self.get("model.classifier.feature_source", "mixture_params")
        if feature_source not in ("mixture_params", "bottleneck"):
            raise ValueError(
                f"model.classifier.feature_source は mixture_params/bottleneck のいずれか: {feature_source!r}"
            )

        target_mode = self.get("model.conditional_unet.target_mode", "single_filter")
        if target_mode not in ("single_filter", "filter_bank"):
            raise ValueError(
                f"model.conditional_unet.target_mode は single_filter/filter_bank のいずれか: {target_mode!r}"
            )
        if target_mode == "filter_bank" and feature_source == "mixture_params":
            # y' が60chになると混合分布パラメータは 3*K*60 = 1800ch/モデルとなり、
            # 案A（分類器入力120ch）と両立しない
            raise ValueError(
                "target_mode=filter_bank のときは classifier.feature_source=bottleneck にしてください"
                "（混合分布パラメータが 3*K*60 チャンネルになり案Aの前提と矛盾するため）"
            )

        bayer = self.get("preprocess.bayer_pattern", "RGGB")
        if bayer not in ("RGGB", "BGGR", "GRBG", "GBRG"):
            raise ValueError(f"preprocess.bayer_pattern が不正です: {bayer!r}")

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return f"Config(source_path={self.source_path})"
