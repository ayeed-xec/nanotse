"""Shape + grad checks for VisualFrontend."""

from __future__ import annotations

import torch

from nanotse.models.frontends.visual_avhubert import VisualFrontend


def test_visual_frontend_shape() -> None:
    vf = VisualFrontend(d_visual=512, frame_size=112)
    video = torch.randint(0, 256, (2, 100, 112, 112, 3), dtype=torch.uint8)
    feat = vf(video)
    assert feat.shape == (2, 100, 512)


def test_visual_frontend_custom_dim() -> None:
    vf = VisualFrontend(d_visual=128, frame_size=64)
    video = torch.randint(0, 256, (1, 25, 64, 64, 3), dtype=torch.uint8)
    feat = vf(video)
    assert feat.shape == (1, 25, 128)


def test_visual_frontend_differentiable() -> None:
    vf = VisualFrontend()
    video = torch.rand(1, 4, 112, 112, 3) * 255
    video_u8 = video.to(torch.uint8)
    feat = vf(video_u8)
    feat.sum().backward()
    # All conv params should have non-None gradients.
    for p in vf.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
