"""Audio enrollment encoder: a different clean clip of the target speaker -> identity vector.

The fundamental TSE design fix. Without a side-channel cue (visual or audio
enrollment), target-speaker extraction is unsolvable: given a 2-speaker mix,
the model has no way to know *which* speaker to extract, and falls back on
heuristics ("louder" / "higher pitch") that don't generalise to unseen
speakers. This module supplies the audio-enrollment cue used by SpEx,
Conformer-TSE, and most non-AV TSE work.

Architecture (small, deliberate -- runs on a 6 GB 3060):

    audio_enrollment (B, T_enroll)
        -> Conv1D(1 -> 64 -> 128 -> d_enroll)  stride 8 each
        -> TemporalMean -> (B, d_enroll)
        -> Linear -> norm -> (B, d_enroll)

A single embedding vector per sample, broadcast as conditioning into the
backbone (see NanoTSE.forward).
"""

from __future__ import annotations

import torch
from torch import nn


class EnrollmentEncoder(nn.Module):
    """``audio (B, T) -> (B, d_enroll)`` speaker-identity embedding."""

    def __init__(
        self,
        d_enroll: int = 192,
        kernel: int = 7,
        stride: int = 8,
    ) -> None:
        super().__init__()
        c1, c2 = 64, 128
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, c1, kernel, stride=stride, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel, stride=stride, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, d_enroll, kernel, stride=stride, padding=pad),
        )
        self.norm = nn.LayerNorm(d_enroll)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """``audio (B, T)`` -> ``(B, d_enroll)`` pooled embedding."""
        if audio.dim() != 2:
            raise ValueError(f"enrollment audio must be 2D (B, T), got {tuple(audio.shape)}")
        x = self.net(audio.unsqueeze(1))  # (B, d_enroll, T_down)
        x = x.mean(dim=-1)  # (B, d_enroll) -- temporal pool over the enrollment clip
        out: torch.Tensor = self.norm(x)
        return out
