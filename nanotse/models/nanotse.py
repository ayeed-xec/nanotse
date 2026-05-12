"""Top-level NanoTSE assembly.

Two modes via the ``with_visual`` constructor flag:

* ``with_visual=False`` — audio-only path (W2.4): AudioFrontend → ChunkAttnBackbone
  → TSEHead. ``forward(audio)`` returns ``(tse_out, None)``.
* ``with_visual=True`` (default) — full AV path (W3.5): adds VisualFrontend +
  DualCacheFusion + NamedSlotMemory + ASDHead. ``forward(audio, video)`` returns
  ``(tse_out, asd_logits)``. If ``video`` is ``None`` the visual modules
  are skipped at forward time even though their params are allocated.

LRU eviction across slots is deferred to W3.5+ once we run real multi-speaker
sessions; the slot bank currently just keeps refining.
"""

from __future__ import annotations

import torch
from torch import nn

from nanotse.models.backbones.chunk_attn import ChunkAttnBackbone
from nanotse.models.frontends.audio_stft import AudioFrontend
from nanotse.models.frontends.visual_avhubert import VisualFrontend
from nanotse.models.fusion.dual_cache import DualCacheFusion
from nanotse.models.heads.asd import ASDHead
from nanotse.models.heads.tse import TSEHead
from nanotse.models.memory.slot_attention import NamedSlotMemory


class NanoTSE(nn.Module):
    """Audio-(visual) target speaker extraction with named-slot identity memory."""

    def __init__(
        self,
        d_model: int = 256,
        kernel: int = 320,
        stride: int = 160,
        n_heads: int = 4,
        n_layers: int = 2,
        cache_len: int = 200,
        with_visual: bool = True,
        d_visual: int = 512,
        n_slots: int = 16,
        d_slot: int = 256,
        n_slot_iters: int = 3,
        frame_size: int = 112,
        fusion_cache_len: int = 50,
    ) -> None:
        super().__init__()
        self.with_visual = with_visual
        self.audio_frontend = AudioFrontend(d_model=d_model, kernel=kernel, stride=stride)
        self.backbone = ChunkAttnBackbone(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, cache_len=cache_len
        )
        self.tse_head = TSEHead(d_model=d_model, kernel=kernel, stride=stride)

        if with_visual:
            self.visual_frontend = VisualFrontend(d_visual=d_visual, frame_size=frame_size)
            self.fusion = DualCacheFusion(
                d_model=d_model,
                d_visual=d_visual,
                n_heads=n_heads,
                cache_len=fusion_cache_len,
            )
            self.slot_memory = NamedSlotMemory(
                n_slots=n_slots, d_input=d_model, d_slot=d_slot, n_iters=n_slot_iters
            )
            self.slot_to_feat = nn.Linear(d_model + d_slot, d_model)
            self.asd_head = ASDHead(d_model=d_model, d_slot=d_slot)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``audio (B, T)`` + optional ``video (B, F, H, W, 3)``.

        Returns ``(tse_out (B, T), asd_logits (B, Ta, N) | None)``.
        """
        enc = self.audio_frontend(audio)

        if self.with_visual and video is not None:
            vis = self.visual_frontend(video)
            enc = self.fusion(enc, vis)

        feat = self.backbone(enc)

        if self.with_visual and video is not None:
            slot_aug, slots = self.slot_memory(feat)
            feat_with_id: torch.Tensor = self.slot_to_feat(slot_aug)
            tse_out = self.tse_head(feat_with_id, enc)
            asd_logits: torch.Tensor | None = self.asd_head(feat_with_id, slots)
        else:
            tse_out = self.tse_head(feat, enc)
            asd_logits = None

        return tse_out, asd_logits
