"""Exponential moving average of model parameters for stable eval.

Standard trick: keep a shadow copy of the model weights updated as
``ema = decay * ema + (1 - decay) * model`` after every optimizer step.
At val/inference time, use ``ema`` instead of the noisy live weights ---
typically picks up +0.3 to +1 dB on speech enhancement tasks.

We store the shadow as a flat dict of tensors so it round-trips through
``torch.save``/``torch.load`` without needing a second ``nn.Module``.
"""

from __future__ import annotations

import torch
from torch import nn


class EMA:
    """Exponential moving average of a model's parameters + buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow = self.shadow[name]
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Swap EMA weights into ``model`` and return the originals (for restore)."""
        backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for k in self.shadow:
            if k in state:
                self.shadow[k].copy_(state[k])
