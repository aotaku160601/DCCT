"""条件付き分布モデル pθ / qφ（03_詳細設計書 3.3 `ConditionalUNet`）。

観測chの高周波残差 x'（30ch）から、隠された2chの残差 y' の画素ごとの分布を
K成分のロジスティック混合として予測するU-Net。pθ（写真）とqφ（AI生成画像）は
同じクラスの別インスタンスで、重みは共有しない。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 分散が0に潰れて対数尤度が発散するのを防ぐ下限
_MIN_LOG_SCALE = -7.0


@dataclass
class MixtureParams:
    """画素ごとのロジスティック混合分布パラメータ。

    いずれも `[B, K, C, H, W]`（C = 予測対象のチャンネル数）。
    `log_w` は成分方向にlog_softmax済み。
    """

    log_w: torch.Tensor
    mu: torch.Tensor
    log_s: torch.Tensor

    @property
    def w(self) -> torch.Tensor:
        return self.log_w.exp()

    @property
    def num_mixtures(self) -> int:
        return self.log_w.shape[1]

    def as_feature_map(self) -> torch.Tensor:
        """分類器入力用に `[B, 3*K*C, H, W]` へ平坦化する（【OI-2】案A）。"""
        batch, k, c, h, w = self.log_w.shape
        stacked = torch.cat([self.log_w, self.mu, self.log_s], dim=1)  # [B,3K,C,H,W]
        return stacked.reshape(batch, 3 * k * c, h, w)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv-GN-ReLU を2回。バッチサイズ16でも安定するようGroupNormを使う。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.ReLU(inplace=True),
    )


class ConditionalUNet(nn.Module):
    """x'（30ch）→ 画素ごとの混合ロジスティックパラメータ。

    Args:
        in_channels: 入力チャンネル数（ハイパスフィルタ30種 × 観測1ch = 30）
        target_channels: 予測対象 y' のチャンネル数（既定2 = 隠された2色）
        num_mixtures: 混合成分数 K
        base_channels: エンコーダ最初の段のチャンネル数
        feature_source: Stage IIへ渡す特徴（`mixture_params` = 案A / `bottleneck` = 案B）
    """

    def __init__(
        self,
        in_channels: int = 30,
        target_channels: int = 2,
        num_mixtures: int = 10,
        base_channels: int = 64,
        feature_source: str = "mixture_params",
    ) -> None:
        super().__init__()
        if feature_source not in ("mixture_params", "bottleneck"):
            raise ValueError(f"未知の feature_source です: {feature_source!r}")

        self.in_channels = in_channels
        self.target_channels = target_channels
        self.num_mixtures = num_mixtures
        self.feature_source = feature_source

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.enc1 = _conv_block(in_channels, c1)
        self.enc2 = _conv_block(c1, c2)
        self.enc3 = _conv_block(c2, c3)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _conv_block(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _conv_block(c1 * 2, c1)

        # 画素ごとに (log_w, mu, log_s) × K × target_channels
        self.head = nn.Conv2d(c1, 3 * num_mixtures * target_channels, 1)

    @property
    def feature_channels(self) -> int:
        """`extract_feature` が返すチャンネル数。"""
        if self.feature_source == "mixture_params":
            return 3 * self.num_mixtures * self.target_channels
        return self.dec1[-2].num_channels  # 案B: デコーダ最終ブロックの出力チャンネル数

    def _trunk(self, x_prime: torch.Tensor) -> torch.Tensor:
        """U-Net本体。デコーダ最終ブロックの特徴マップ `[B, base, H, W]` を返す。"""
        e1 = self.enc1(x_prime)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return d1

    def forward(self, x_prime: torch.Tensor) -> MixtureParams:
        """`[B,30,H,W]` → `MixtureParams`（各 `[B,K,C,H,W]`）。"""
        raw = self.head(self._trunk(x_prime))
        batch, _, height, width = raw.shape
        raw = raw.reshape(batch, 3, self.num_mixtures, self.target_channels, height, width)

        log_w = F.log_softmax(raw[:, 0], dim=1)          # 成分方向に正規化
        mu = raw[:, 1]
        log_s = raw[:, 2].clamp(min=_MIN_LOG_SCALE)
        return MixtureParams(log_w=log_w, mu=mu, log_s=log_s)

    def extract_feature(self, x_prime: torch.Tensor) -> torch.Tensor:
        """Stage II の分類器へ渡す特徴マップを返す（1.3節の案A/案B切替）。"""
        trunk = self._trunk(x_prime)
        if self.feature_source == "bottleneck":
            return trunk

        raw = self.head(trunk)
        batch, _, height, width = raw.shape
        raw = raw.reshape(batch, 3, self.num_mixtures, self.target_channels, height, width)
        log_w = F.log_softmax(raw[:, 0], dim=1)
        params = MixtureParams(log_w=log_w, mu=raw[:, 1], log_s=raw[:, 2].clamp(min=_MIN_LOG_SCALE))
        return params.as_feature_map()

    def freeze(self) -> "ConditionalUNet":
        """Stage II 用にパラメータを固定する（Ablation-D で freeze しない設定も取れる）。"""
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        return self
