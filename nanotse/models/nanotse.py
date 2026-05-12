"""Top-level NanoTSE assembly.

W2.4 version: **audio-only** -- AudioFrontend -> ChunkAttnBackbone -> TSEHead.
W3.5 will add VisualFrontend + DualCacheFusion + NamedSlotMemory + ASDHead,
and the public ``forward(audio, video=None)`` signature is already shaped for
that extension.
"""

from __future__ import annotations

import torch
from torch import nn

from nanotse.models.backbones.chunk_attn import ChunkAttnBackbone
from nanotse.models.frontends.audio_stft import AudioFrontend
from nanotse.models.heads.tse import TSEHead


class NanoTSE(nn.Module):
    """Audio-only NanoTSE (W2.4)."""

    def __init__(
        self,
        d_model: int = 256,
        kernel: int = 320,
        stride: int = 160,
        n_heads: int = 4,
        n_layers: int = 2,
        cache_len: int = 200,
    ) -> None:
        super().__init__()
        self.audio_frontend = AudioFrontend(d_model=d_model, kernel=kernel, stride=stride)
        self.backbone = ChunkAttnBackbone(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, cache_len=cache_len
        )
        self.tse_head = TSEHead(d_model=d_model, kernel=kernel, stride=stride)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """``audio`` shape ``(B, T)`` -> ``(B, T)`` clean-target estimate."""
        enc = self.audio_frontend(audio)
        feat = self.backbone(enc)
        out: torch.Tensor = self.tse_head(feat, enc)
        return out
