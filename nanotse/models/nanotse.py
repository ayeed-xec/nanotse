"""Top-level NanoTSE assembly.

Three identification modes via constructor flags:

* ``with_visual=True``     -- mouth-ROI video frames condition the extraction
                              (real lips when the data is fetched; otherwise
                              the visual stream contributes noise).
* ``with_enrollment=True`` -- a clean reference clip of the target speaker
                              is encoded into a (B, d_enroll) embedding and
                              broadcast into the audio encoder output. This
                              is the standard SpEx-style audio TSE cue.
* both                     -- both cues active; the model can use whichever
                              is most informative for a given frame.

At least one cue should be active; without one the task is under-specified
(2-speaker mix has no way to know which speaker is "target").

Forward signature:
    forward(audio, video=None, enrollment=None) -> (tse_out, asd_logits, slots)

with asd_logits / slots populated only on the AV path (with_visual=True).
"""

from __future__ import annotations

import torch
from torch import nn

from nanotse.models.backbones.chunk_attn import ChunkAttnBackbone
from nanotse.models.frontends.audio_stft import AudioFrontend
from nanotse.models.frontends.enrollment import EnrollmentEncoder
from nanotse.models.frontends.visual_avhubert import VisualFrontend
from nanotse.models.fusion.dual_cache import DualCacheFusion
from nanotse.models.heads.asd import ASDHead
from nanotse.models.heads.tse import TSEHead
from nanotse.models.memory.slot_attention import NamedSlotMemory

NanoTSEOutput = tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]


class NanoTSE(nn.Module):
    """Audio-(visual)-(enrollment) target speaker extraction with named-slot identity memory."""

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
        with_enrollment: bool = False,
        d_enroll: int = 192,
        with_slots: bool = True,
        with_asd: bool = True,
    ) -> None:
        super().__init__()
        self.with_visual = with_visual
        self.with_enrollment = with_enrollment
        # ``with_slots`` and ``with_asd`` only have effect on the AV path (visual=True);
        # without visual neither submodule is built or called.
        self.with_slots = with_slots and with_visual
        self.with_asd = with_asd and with_visual and with_slots
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
            if self.with_slots:
                self.slot_memory = NamedSlotMemory(
                    n_slots=n_slots, d_input=d_model, d_slot=d_slot, n_iters=n_slot_iters
                )
                self.slot_to_feat = nn.Linear(d_model + d_slot, d_model)
                if self.with_asd:
                    self.asd_head = ASDHead(d_model=d_model, d_slot=d_slot)

        if with_enrollment:
            self.enrollment_encoder = EnrollmentEncoder(d_enroll=d_enroll)
            # Project the (B, d_enroll) embedding into (B, d_model) for broadcast addition
            # to the encoder output. Zero-init weight + bias so enrollment contributes
            # nothing at step 0 -- the encoder sees the same activations it was random-
            # init'd for, and the enrollment branch grows in cleanly during warmup
            # instead of injecting random conditioning that disrupts early learning.
            self.enroll_to_feat = nn.Linear(d_enroll, d_model)
            nn.init.zeros_(self.enroll_to_feat.weight)
            nn.init.zeros_(self.enroll_to_feat.bias)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor | None = None,
        enrollment: torch.Tensor | None = None,
    ) -> NanoTSEOutput:
        """``audio (B, T)`` + optional ``video (B, F, H, W, 3)`` + optional ``enrollment (B, T_e)``.

        Returns ``(tse_out (B, T), asd_logits (B, Ta, N) | None, slots (B, N, S) | None)``.
        """
        enc = self.audio_frontend(audio)  # (B, Ta, D)

        # Enrollment cue: speaker-identity vector broadcast as conditioning.
        if self.with_enrollment and enrollment is not None:
            enroll_emb = self.enrollment_encoder(enrollment)  # (B, d_enroll)
            enroll_feat = self.enroll_to_feat(enroll_emb)  # (B, D)
            enc = enc + enroll_feat.unsqueeze(1)  # broadcast over time

        if self.with_visual and video is not None:
            vis = self.visual_frontend(video)
            enc = self.fusion(enc, vis)

        feat = self.backbone(enc)

        if self.with_visual and video is not None and self.with_slots:
            slot_aug, slots = self.slot_memory(feat)
            feat_with_id: torch.Tensor = self.slot_to_feat(slot_aug)
            tse_out = self.tse_head(feat_with_id, enc)
            asd_logits: torch.Tensor | None = (
                self.asd_head(feat_with_id, slots) if self.with_asd else None
            )
            return tse_out, asd_logits, slots

        tse_out = self.tse_head(feat, enc)
        return tse_out, None, None
