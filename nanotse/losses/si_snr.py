"""Scale-invariant SNR (Le Roux et al., 2019), the standard TSE training loss."""

from __future__ import annotations

import torch

EPS: float = 1e-8


def si_snr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Scale-invariant SNR in dB, computed per-item along the last dim.

    Args:
        estimate: ``(..., T)`` estimated signal.
        target:   ``(..., T)`` reference signal.
        eps:      numerical stability floor.

    Returns:
        SI-SNR in dB with shape ``(...,)`` — caller decides how to reduce.
    """
    if estimate.shape != target.shape:
        raise ValueError(f"estimate and target shape mismatch: {estimate.shape} vs {target.shape}")

    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)

    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target * target).sum(dim=-1, keepdim=True) + eps
    s_target = dot * target / target_energy
    e_noise = estimate - s_target

    num = (s_target * s_target).sum(dim=-1)
    den = (e_noise * e_noise).sum(dim=-1) + eps
    return 10.0 * torch.log10(num / den + eps)


def negative_si_snr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Mean negative SI-SNR — use directly as a loss (lower is better)."""
    return -si_snr(estimate, target, eps=eps).mean()
