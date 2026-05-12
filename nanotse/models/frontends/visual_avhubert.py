"""Visual frontend: per-frame CNN encoder.

W3.1 of the sprint plan. Eats ``(B, F, H, W, 3) uint8`` video at 25 fps,
returns ``(B, F, d_visual)`` float features at 25 Hz. Lightweight on
purpose -- 4 stride-2 convs + AdaptiveAvgPool reduces a 112x112 frame to
a single ``d_visual``-dim vector. AV-HuBERT-frozen lands later (loading
pretrained weights from HF is W3.2-bis; not blocking the slot memory).
"""

from __future__ import annotations

import torch
from torch import nn


class VisualFrontend(nn.Module):
    """Per-frame CNN. (B, F, H, W, 3) uint8 -> (B, F, d_visual) float32."""

    def __init__(self, d_visual: int = 512, frame_size: int = 112) -> None:
        super().__init__()
        self.d_visual = d_visual
        self.frame_size = frame_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, d_visual, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.norm = nn.LayerNorm(d_visual)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """``video`` shape ``(B, F, H, W, 3) uint8`` -> ``(B, F, d_visual)``."""
        b, f, _, _, _ = video.shape
        x = video.permute(0, 1, 4, 2, 3).contiguous().float() / 255.0  # (B, F, 3, H, W)
        x = x.view(b * f, *x.shape[2:])
        x = self.conv(x).view(b * f, self.d_visual)
        out: torch.Tensor = self.norm(x).view(b, f, self.d_visual)
        return out
