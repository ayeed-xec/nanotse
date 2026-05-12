"""Dual-cache cross-modal fusion: audio queries attend to a visual KV cache.

W3.2 of the sprint plan. Visual features arrive at 25 Hz, audio features
at 100 Hz. Each audio query attends to the rolling visual KV cache
(default ``cache_len=50`` frames = 2 s of visual context).

State is ``(k_cache, v_cache)`` -- one cache, shared across the cross-
attention layer. Visual frames may be empty for chunks where no new
visual is delivered (the cache still provides context).
"""

from __future__ import annotations

import math

import torch
from torch import nn

DualCacheState = tuple[torch.Tensor, torch.Tensor]


class DualCacheFusion(nn.Module):
    """Cross-attention: ``audio (B, Ta, D)`` attends to ``visual (B, Tv, Dv)``.

    Returns audio features augmented with visual context, same shape as audio.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_visual: int = 512,
        n_heads: int = 4,
        cache_len: int = 50,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.d_visual = d_visual
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.cache_len = cache_len

        self.norm_audio = nn.LayerNorm(d_model)
        self.norm_visual = nn.LayerNorm(d_visual)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_visual, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm_post = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def init_state(self, batch_size: int, device: torch.device) -> DualCacheState:
        return (
            torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=device),
            torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=device),
        )

    def _kv_from_visual(self, visual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, tv, _ = visual.shape
        kv = self.kv_proj(self.norm_visual(visual))
        kv = kv.reshape(b, tv, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # (B, H, Tv, D_h)
        return k, v

    def _cross_attend(self, audio: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``audio (B, Ta, D)``, ``k/v (B, H, Tkv, D_h)``."""
        b, ta, d = audio.shape
        q = (
            self.q_proj(self.norm_audio(audio))
            .reshape(b, ta, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, H, Ta, D_h)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, Ta, Tkv)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, H, Ta, D_h)
        out = out.transpose(1, 2).reshape(b, ta, d)
        out_proj: torch.Tensor = self.out_proj(out)
        return out_proj

    def forward_chunk(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        state: DualCacheState,
    ) -> tuple[torch.Tensor, DualCacheState]:
        residual = audio

        if visual.shape[1] > 0:
            k_new, v_new = self._kv_from_visual(visual)
        else:
            b = audio.shape[0]
            k_new = torch.zeros(b, self.n_heads, 0, self.head_dim, device=audio.device)
            v_new = torch.zeros(b, self.n_heads, 0, self.head_dim, device=audio.device)

        k_cache, v_cache = state
        k = torch.cat([k_cache, k_new], dim=2)
        v = torch.cat([v_cache, v_new], dim=2)

        if k.shape[2] > self.cache_len:
            k = k[:, :, -self.cache_len :]
            v = v[:, :, -self.cache_len :]

        # No visual context at all -> pass audio through (still ffn-augment).
        x = residual if k.shape[2] == 0 else residual + self._cross_attend(audio, k, v)
        x = x + self.ffn(self.norm_post(x))
        new_state: DualCacheState = (k, v)
        return x, new_state

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        """Offline cross-attention over the whole (audio, visual) pair."""
        residual = audio
        k, v = self._kv_from_visual(visual)
        x = residual + self._cross_attend(audio, k, v)
        out: torch.Tensor = x + self.ffn(self.norm_post(x))
        return out
