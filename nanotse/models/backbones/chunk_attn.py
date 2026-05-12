"""Causal chunked self-attention backbone with KV cache (streaming-safe).

Each layer is multi-head self-attention with a causal mask + a small FFN,
stacked ``n_layers`` deep. For streaming, the layer keeps a rolling KV
cache of the last ``cache_len`` feature frames; ``forward_chunk(x, state)``
appends to the cache, attends, and returns the updated state. ``forward(x)``
is the offline equivalent: full causal attention over the whole sequence,
no cache.

State is a list of ``(k_cache, v_cache)`` tuples -- one per layer.
Always passed in/out explicitly per the streaming contract in
``docs/ARCHITECTURE.md``: no globals, no module attributes that mutate
during ``forward()``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

LayerKV = tuple[torch.Tensor, torch.Tensor]  # (k_cache, v_cache) shape (B, H, cache_len, D_h)
ChunkAttnState = list[LayerKV]


class _CausalAttnLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cache_len: int,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.cache_len = cache_len

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, T, D_h)
        return q, k, v

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_offset: int,
    ) -> torch.Tensor:
        """``q``: (B, H, Tq, D_h), ``k/v``: (B, H, Tkv, D_h). ``Tkv = cache + Tq``.

        Position i in the new chunk can attend to: any cache position (j <
        ``cache_offset``) or any within-chunk position j with j - cache_offset <= i.
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, Tq, Tkv)
        tq, tkv = q.shape[2], k.shape[2]
        i = torch.arange(tq, device=q.device).unsqueeze(1)
        j = torch.arange(tkv, device=q.device).unsqueeze(0)
        mask = (j < cache_offset) | (j - cache_offset <= i)
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out: torch.Tensor = attn @ v
        return out

    def forward_chunk(self, x: torch.Tensor, state: LayerKV) -> tuple[torch.Tensor, LayerKV]:
        b, tc, d = x.shape
        residual = x
        q, k_new, v_new = self._project_qkv(self.norm1(x))

        k_cache, v_cache = state
        k = torch.cat([k_cache, k_new], dim=2)
        v = torch.cat([v_cache, v_new], dim=2)

        if k.shape[2] > self.cache_len:
            k = k[:, :, -self.cache_len :]
            v = v[:, :, -self.cache_len :]

        cache_offset = k.shape[2] - tc
        attn_out = self._attend(q, k, v, cache_offset)
        attn_out = attn_out.transpose(1, 2).reshape(b, tc, d)
        attn_out = self.out_proj(attn_out)

        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        new_state: LayerKV = (k, v)
        return x, new_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Offline: full causal self-attention over the whole sequence."""
        b, t, d = x.shape
        residual = x
        q, k, v = self._project_qkv(self.norm1(x))
        attn_out = self._attend(q, k, v, cache_offset=0)
        attn_out = attn_out.transpose(1, 2).reshape(b, t, d)
        attn_out = self.out_proj(attn_out)
        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class ChunkAttnBackbone(nn.Module):
    """Stack of causal self-attention layers. CPU / MPS / CUDA compatible."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        cache_len: int = 200,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.cache_len = cache_len
        self.layers = nn.ModuleList(
            [_CausalAttnLayer(d_model, n_heads, cache_len) for _ in range(n_layers)]
        )

    def init_state(self, batch_size: int, device: torch.device) -> ChunkAttnState:
        layer0 = self.layers[0]
        assert isinstance(layer0, _CausalAttnLayer)
        head_dim = layer0.head_dim
        n_heads = layer0.n_heads
        return [
            (
                torch.zeros(batch_size, n_heads, 0, head_dim, device=device),
                torch.zeros(batch_size, n_heads, 0, head_dim, device=device),
            )
            for _ in range(len(self.layers))
        ]

    def forward_chunk(
        self, x: torch.Tensor, state: ChunkAttnState
    ) -> tuple[torch.Tensor, ChunkAttnState]:
        new_state: ChunkAttnState = []
        for layer, layer_state in zip(self.layers, state, strict=True):
            assert isinstance(layer, _CausalAttnLayer)
            x, ns = layer.forward_chunk(x, layer_state)
            new_state.append(ns)
        return x, new_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            assert isinstance(layer, _CausalAttnLayer)
            x = layer(x)
        return x
