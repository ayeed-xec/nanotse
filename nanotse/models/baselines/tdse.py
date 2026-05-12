"""Conv-TasNet-lite time-domain baseline (no speaker/face conditioning yet).

Placeholder for the W2 TDSE baseline -- small enough to overfit the smoke
set on M3 (MPS or CPU) while exercising the encoder/separator/decoder
plumbing that NanoTSE will sit on top of. Speaker conditioning lands in W3
with the named-slot memory.
"""

from __future__ import annotations

import torch
from torch import nn


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.act = nn.PReLU()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x + self.norm(self.act(self.conv(x)))
        return out


class TDSEBaseline(nn.Module):
    """Encoder -> bottleneck -> dilated TCN stack -> mask -> decoder."""

    def __init__(
        self,
        n_feat: int = 128,
        kernel: int = 16,
        stride: int = 8,
        bottleneck: int = 64,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        pad = stride // 2
        self.encoder = nn.Conv1d(1, n_feat, kernel, stride=stride, padding=pad)
        self.in_bn = nn.Conv1d(n_feat, bottleneck, 1)
        self.tcn = nn.Sequential(*[_TCNBlock(bottleneck, 2**i) for i in range(n_blocks)])
        self.mask_conv = nn.Conv1d(bottleneck, n_feat, 1)
        self.decoder = nn.ConvTranspose1d(n_feat, 1, kernel, stride=stride, padding=pad)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """``mix`` shape ``(B, T)``; returns clean-target estimate of the same shape."""
        x = mix.unsqueeze(1)
        t = x.shape[-1]
        enc = torch.relu(self.encoder(x))
        b = self.in_bn(enc)
        b = self.tcn(b)
        mask = torch.sigmoid(self.mask_conv(b))
        out: torch.Tensor = self.decoder(enc * mask)
        return out.squeeze(1)[..., :t]
