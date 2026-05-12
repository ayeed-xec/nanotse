"""Shape + grad checks for ASDHead."""

from __future__ import annotations

import torch

from nanotse.models.heads.asd import ASDHead


def test_asd_head_shape() -> None:
    head = ASDHead(d_model=64, d_slot=32)
    features = torch.randn(2, 100, 64)
    slots = torch.randn(2, 16, 32)
    logits = head(features, slots)
    assert logits.shape == (2, 100, 16)


def test_asd_head_differentiable() -> None:
    head = ASDHead(d_model=32, d_slot=32)
    features = torch.randn(1, 10, 32, requires_grad=True)
    slots = torch.randn(1, 4, 32, requires_grad=True)
    head(features, slots).sum().backward()
    assert features.grad is not None
    assert slots.grad is not None
