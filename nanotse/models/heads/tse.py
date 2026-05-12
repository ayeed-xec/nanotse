"""TSE head: predict a mask in encoder feature space, apply it, decode to audio.

Conv-TasNet-style: features from the backbone go through a small mask
projection -> sigmoid -> multiply with the AudioFrontend encoder output ->
ConvTranspose1d decoder back to time domain. With matched ``kernel`` /
``stride`` / ``padding`` against ``AudioFrontend``, the round-trip is
sample-exact: ``T -> Ta -> T``.
"""

from __future__ import annotations

import torch
from torch import nn


class TSEHead(nn.Module):
    """Mask projection + learned decoder."""

    def __init__(
        self,
        d_model: int = 256,
        kernel: int = 320,
        stride: int = 160,
    ) -> None:
        super().__init__()
        if (kernel - stride) % 2 != 0:
            raise ValueError(
                f"TSEHead needs (kernel - stride) even for clean padding; "
                f"got kernel={kernel}, stride={stride}"
            )
        padding = (kernel - stride) // 2
        self.mask_proj = nn.Linear(d_model, d_model)
        self.decoder = nn.ConvTranspose1d(d_model, 1, kernel, stride=stride, padding=padding)

    def forward(
        self,
        features: torch.Tensor,
        encoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            features:    ``(B, Ta, D)`` backbone output.
            encoder_out: ``(B, Ta, D)`` AudioFrontend output (pre-backbone).
        Returns:
            ``(B, T)`` clean-target estimate.
        """
        mask = torch.sigmoid(self.mask_proj(features))
        masked = encoder_out * mask
        x = masked.transpose(1, 2)  # (B, D, Ta)
        out: torch.Tensor = self.decoder(x).squeeze(1)
        return out
