"""前処理パイプライン: パッチ → CFAマスク → ハイパスフィルタ → truncation。

Algorithm 1 / 2 の各行に対応する:

    I~ ← RandomCrop(I, s)
    x  ← CFAMask適用(I~)             → 観測1ch
    y  ← PackMissingChannels(I~)      → 隠された2ch
    x' ← Truncate(Stack([h_m * x]))   → 30ch（条件付きモデルの入力）
    y' ← Truncate(...)                → 2ch（条件付きモデルの予測対象）

y' の作り方は論文 Algorithm 1 line 8 に従い、x' と同じく30種すべてを適用する。

    filter_bank（既定・論文準拠）: 30種すべてを y に適用して 60ch にする
    single_filter:                target_filter の1種類のみ適用し 2ch に保つ（比較用）
"""

from __future__ import annotations

import torch

from .cfa_mask import build_mask
from .highpass import HighPassFilterBank


class DCCTPreprocessor:
    """RGBパッチから (x', y') を作る。"""

    def __init__(
        self,
        truncation_t: float = 7.0,
        use_cfa_mask: bool = True,
        use_high_pass: bool = True,
        bayer_pattern: str = "RGGB",
        target_mode: str = "filter_bank",
        target_filter: str = "square5x5",
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if target_mode not in ("single_filter", "filter_bank"):
            raise ValueError(f"未知の target_mode です: {target_mode!r}")

        self.mask = build_mask(use_cfa_mask, bayer_pattern, generator)
        self.bank = HighPassFilterBank(truncation_t, device=device, enabled=use_high_pass)
        self.target_mode = target_mode
        self.target_filter = target_filter

    @property
    def input_channels(self) -> int:
        """x' のチャンネル数（条件付きモデルの入力チャンネル数）。"""
        return self.bank.M if self.bank.enabled else 1

    @property
    def target_channels(self) -> int:
        """y' のチャンネル数（条件付きモデルの予測対象チャンネル数）。"""
        if self.target_mode == "single_filter" or not self.bank.enabled:
            return 2
        return 2 * self.bank.M

    def to(self, device: torch.device) -> "DCCTPreprocessor":
        self.bank.to(device)
        return self

    def __call__(self, patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prepare(patch)

    def prepare(self, patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`[...,3,s,s]`（0〜255）→ (x' `[...,30,s,s]`, y' `[...,2,s,s]`)。"""
        x, y = self.mask.apply(patch)
        x_prime = self.bank.apply(x)
        if self.target_mode == "filter_bank":
            y_prime = self.bank.apply(y)
        else:
            y_prime = self.bank.apply_single(y, self.target_filter)
        return x_prime, y_prime

    def prepare_input(self, patch: torch.Tensor) -> torch.Tensor:
        """Stage II / 推論で使う x' のみを作る。"""
        x, _ = self.mask.apply(patch)
        return self.bank.apply(x)
