"""Audio frontend: learned Conv1D OR STFT log-magnitude encoder.

W2.1 + W2.5 of the sprint plan. Both branches output the same shape
``(B, Ta, d_model)`` at 100 Hz so the rest of the model doesn't care
which is plugged in. Pick via ``branch="conv1d" | "stft"``.

With the default ``kernel=320, stride=160`` (20 ms analysis window,
10 ms hop) and ``padding = (kernel - stride) / 2`` for Conv1D, the
output rate is exactly ``T / stride`` -- clean round-trip with the
matching ``TSEHead`` ConvTranspose1d decoder.

The STFT branch uses ``n_fft=kernel``, ``hop_length=stride``,
``win_length=kernel`` (Hann window), log-magnitude, and a linear
projection to ``d_model``. Output is truncated to ``T // stride``
frames so it aligns with the Conv1D path and the TSE decoder.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class AudioFrontend(nn.Module):
    """Conv1D or STFT encoder.

    Shape: ``(B, T) @ 16 kHz`` -> ``(B, Ta, d_model) @ 100 Hz``.
    """

    def __init__(
        self,
        d_model: int = 256,
        kernel: int = 320,
        stride: int = 160,
        branch: Literal["conv1d", "stft"] = "conv1d",
    ) -> None:
        super().__init__()
        if (kernel - stride) % 2 != 0:
            raise ValueError(
                f"AudioFrontend needs (kernel - stride) even for clean padding; "
                f"got kernel={kernel}, stride={stride}"
            )
        if branch not in ("conv1d", "stft"):
            raise ValueError(f"branch must be 'conv1d' or 'stft', got {branch!r}")

        self.d_model = d_model
        self.stride = stride
        self.branch = branch

        if branch == "conv1d":
            padding = (kernel - stride) // 2
            self.conv = nn.Conv1d(1, d_model, kernel, stride=stride, padding=padding)
        else:
            self.n_fft = kernel
            self.hop_length = stride
            self.win_length = kernel
            n_freq = kernel // 2 + 1
            self.stft_proj = nn.Linear(n_freq, d_model)
            self.register_buffer("stft_window", torch.hann_window(kernel), persistent=False)

        self.norm = nn.LayerNorm(d_model)

    def _forward_conv1d(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.conv(audio.unsqueeze(1))  # (B, D, Ta)
        out: torch.Tensor = x.transpose(1, 2).contiguous()  # (B, Ta, D)
        return out

    def _forward_stft(self, audio: torch.Tensor) -> torch.Tensor:
        window = self.stft_window
        assert isinstance(window, torch.Tensor)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
        )  # (B, n_freq, T_stft)
        mag = torch.log1p(spec.abs())  # (B, n_freq, T_stft)
        t_target = audio.shape[-1] // self.hop_length
        mag = mag[:, :, :t_target]  # truncate to align with Conv1D path
        x = mag.transpose(1, 2).contiguous()  # (B, Ta, n_freq)
        out: torch.Tensor = self.stft_proj(x)  # (B, Ta, d_model)
        return out

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """``audio`` shape ``(B, T)`` -> ``(B, Ta, d_model)``."""
        x = self._forward_conv1d(audio) if self.branch == "conv1d" else self._forward_stft(audio)
        out: torch.Tensor = self.norm(x)
        return out
