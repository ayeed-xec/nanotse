"""Audio frontend: learned Conv1D encoder. 16 kHz audio -> 100 Hz feature stream.

W2.1 of the sprint plan. STFT branch is W2.5 (arrives alongside the real
VoxCeleb2-mix loader). With the default ``kernel=320, stride=160`` (20 ms
analysis window, 10 ms hop) and ``padding = (kernel - stride) / 2``, the
output rate is exactly ``T / stride`` -- clean round-trip with the matching
``TSEHead`` ConvTranspose1d decoder.
"""

from __future__ import annotations

import torch
from torch import nn


class AudioFrontend(nn.Module):
    """Conv1D encoder.

    Shape: ``(B, T) @ 16 kHz`` -> ``(B, Ta, d_model) @ 100 Hz``.
    """

    def __init__(
        self,
        d_model: int = 256,
        kernel: int = 320,
        stride: int = 160,
    ) -> None:
        super().__init__()
        if (kernel - stride) % 2 != 0:
            raise ValueError(
                f"AudioFrontend needs (kernel - stride) even for clean padding; "
                f"got kernel={kernel}, stride={stride}"
            )
        self.d_model = d_model
        self.stride = stride
        padding = (kernel - stride) // 2
        self.conv = nn.Conv1d(1, d_model, kernel, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """``audio`` shape ``(B, T)`` -> ``(B, Ta, d_model)``."""
        x = self.conv(audio.unsqueeze(1))  # (B, D, Ta)
        x = x.transpose(1, 2).contiguous()  # (B, Ta, D)
        out: torch.Tensor = self.norm(x)
        return out
