"""Plumbing test: a tiny conv stack trains on 4 synthetic AV clips without NaNs.

This is the canonical "does the gradient flow" integration test. It catches
issues with device placement, broadcasting, loss shape, optimizer setup, and
the dataset interface in one go — the same kind of plumbing bug a real M3
smoke train would surface, but in <2 s on CPU so it lives in unit-test CI.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from nanotse.data import SyntheticAVMixDataset
from nanotse.losses import negative_si_snr


class _TinyConvDenoiser(nn.Module):
    """1-conv plumbing model: Conv → ReLU → Conv on a single audio channel."""

    def __init__(self, channels: int = 32, kernel: int = 21) -> None:
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel, padding=pad),
        )

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        return self.net(mix.unsqueeze(1)).squeeze(1)


def test_smoke_overfit_4_clips_decreases_loss() -> None:
    torch.manual_seed(0)
    ds = SyntheticAVMixDataset(num_clips=4, clip_seconds=1.0, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = _TinyConvDenoiser()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    initial_params = {n: p.detach().clone() for n, p in model.named_parameters()}

    losses: list[float] = []
    for _ in range(150):
        for batch in loader:
            est = model(batch["mix"])
            loss = negative_si_snr(est, batch["target"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

    assert all(math.isfinite(loss_val) for loss_val in losses), (
        f"non-finite loss encountered: {[loss_val for loss_val in losses if not math.isfinite(loss_val)]}"
    )

    params_changed = any(not torch.equal(initial_params[n], p) for n, p in model.named_parameters())
    assert params_changed, "no parameters updated — optimizer or backward path is broken"

    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert final < initial - 0.5, (
        f"loss did not decrease meaningfully: initial {initial:.3f} → final {final:.3f}"
    )
