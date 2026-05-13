"""MeMo baseline (Li et al. arXiv:2507.15294, Jul 2025) -- architecture-only port.

Direct prior art for NanoTSE's slot memory (paper contribution 1). Uses
audio-only momentum banks: a SpeakerBank (mean-pooled embedding per step)
plus a ContextBank (per-frame context features). Self-enrolls from the
backbone features each chunk (FIFO replacement at ``N=1``).

**Scope warning** -- this is paper-architecture placeholder for the W5 ablation
harness. Paper-grade reproduction (``MeMo(TDSE) >= 9.85 dB SI-SNR`` on
Impaired-Visual) needs all of:
  - PAR training algorithm (2-stage with pseudo-memory rollout, paper Algo 1)
  - Visual impairment augmentation pipeline
  - Full VoxCeleb2-mix training data
  - ECAPA-TDNN speaker encoder for higher-quality momentum embeddings

All of those land in W4 on the 3060 + A100. Tests here verify the module
forwards + backwards on synthetic data with the same I/O contract as NanoTSE,
so swapping the two for an ablation row is one constructor call.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from nanotse.models.backbones.chunk_attn import ChunkAttnBackbone
from nanotse.models.frontends.audio_stft import AudioFrontend
from nanotse.models.frontends.visual_avhubert import VisualFrontend
from nanotse.models.heads.tse import TSEHead


class MeMoState(BaseModel):
    """Persistent MeMo bank state across streaming chunks (N=1 FIFO).

    Pydantic model so the project keeps a single container style; ``arbitrary
    _types_allowed`` lets the speaker/context slot fields hold ``torch.Tensor``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    speaker_slot: torch.Tensor | None = None  # (B, d_spk)
    context_slot: torch.Tensor | None = None  # (B, T, d_ctx)
    step: int = 0


class SpeakerBank(nn.Module):
    """N=1 FIFO audio-only speaker momentum (paper Eq. 3-4, simplified to N=1)."""

    def __init__(self, d_input: int = 256, d_spk: int = 192) -> None:
        super().__init__()
        self.d_spk = d_spk
        self.push_proj = nn.Linear(d_input, d_spk)
        self.retrieve_proj = nn.Linear(d_spk, d_input)

    def push(self, features: torch.Tensor) -> torch.Tensor:
        """``features (B, T, D)`` -> speaker embedding ``(B, d_spk)``."""
        out: torch.Tensor = self.push_proj(features.mean(dim=1))
        return out

    def retrieve(
        self,
        slot: torch.Tensor | None,
        batch_size: int,
        n_frames: int,
        d_input: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Returns per-frame momentum ``(B, T, D)``; zeros at cold start."""
        if slot is None:
            return torch.zeros(batch_size, n_frames, d_input, device=device)
        v: torch.Tensor = self.retrieve_proj(slot)  # (B, D)
        return v.unsqueeze(1).expand(-1, n_frames, -1).contiguous()


class ContextBank(nn.Module):
    """N=1 FIFO context bank (paper Eq. 10-13, simplified to N=1, 1-layer)."""

    def __init__(self, d_input: int = 256, d_ctx: int = 256) -> None:
        super().__init__()
        self.d_ctx = d_ctx
        self.push_proj = nn.Linear(d_input, d_ctx)
        self.retrieve_proj = nn.Linear(d_ctx, d_input)

    def push(self, features: torch.Tensor) -> torch.Tensor:
        """``features (B, T, D)`` -> stored slot ``(B, T, d_ctx)``."""
        out: torch.Tensor = self.push_proj(features)
        return out

    def retrieve(
        self,
        slot: torch.Tensor | None,
        batch_size: int,
        n_frames: int,
        d_input: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Returns per-frame context ``(B, T, D)``; zeros at cold start."""
        if slot is None:
            return torch.zeros(batch_size, n_frames, d_input, device=device)
        # Average-pool over the stored time axis, then broadcast to current chunk.
        v: torch.Tensor = self.retrieve_proj(slot.mean(dim=1))  # (B, D)
        return v.unsqueeze(1).expand(-1, n_frames, -1).contiguous()


class MeMoBaseline(nn.Module):
    """Audio-(visual) MeMo: SpeakerBank + ContextBank around a TSE backbone."""

    def __init__(
        self,
        d_model: int = 256,
        kernel: int = 320,
        stride: int = 160,
        n_heads: int = 4,
        n_layers: int = 2,
        cache_len: int = 200,
        d_spk: int = 192,
        d_ctx: int = 256,
        with_visual: bool = False,
        d_visual: int = 512,
        frame_size: int = 112,
    ) -> None:
        super().__init__()
        self.with_visual = with_visual
        self.d_model = d_model

        self.audio_frontend = AudioFrontend(d_model=d_model, kernel=kernel, stride=stride)
        if with_visual:
            self.visual_frontend = VisualFrontend(d_visual=d_visual, frame_size=frame_size)
            self.visual_proj = nn.Linear(d_visual, d_model)

        self.speaker_bank = SpeakerBank(d_input=d_model, d_spk=d_spk)
        self.context_bank = ContextBank(d_input=d_model, d_ctx=d_ctx)

        # Fuse: audio [+ visual] + speaker-momentum + context-momentum.
        n_concat = 4 if with_visual else 3
        self.fuse = nn.Linear(d_model * n_concat, d_model)
        self.backbone = ChunkAttnBackbone(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, cache_len=cache_len
        )
        self.tse_head = TSEHead(d_model=d_model, kernel=kernel, stride=stride)

    def init_state(self, batch_size: int, device: torch.device) -> MeMoState:
        return MeMoState(speaker_slot=None, context_slot=None, step=0)

    def forward_chunk(
        self,
        audio: torch.Tensor,
        state: MeMoState,
        video: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MeMoState]:
        enc = self.audio_frontend(audio)  # (B, T, D)
        b, t, _ = enc.shape

        m_s = self.speaker_bank.retrieve(state.speaker_slot, b, t, self.d_model, audio.device)
        m_c = self.context_bank.retrieve(state.context_slot, b, t, self.d_model, audio.device)

        if self.with_visual and video is not None:
            vis = self.visual_frontend(video)  # (B, Tv, Dv)
            vis_proj = self.visual_proj(vis)  # (B, Tv, D)
            vis_broadcast = vis_proj.mean(dim=1, keepdim=True).expand(-1, t, -1)
            fused = torch.cat([enc, vis_broadcast, m_s, m_c], dim=-1)
        else:
            fused = torch.cat([enc, m_s, m_c], dim=-1)

        r = self.fuse(fused)  # (B, T, D)
        feat = self.backbone(r)
        out: torch.Tensor = self.tse_head(feat, enc)  # (B, T)

        # Self-enrollment: push backbone features into the banks.
        new_state = MeMoState(
            speaker_slot=self.speaker_bank.push(feat),
            context_slot=self.context_bank.push(feat),
            step=state.step + 1,
        )
        return out, new_state

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.init_state(audio.shape[0], audio.device)
        out, _ = self.forward_chunk(audio, state, video=video)
        return out
