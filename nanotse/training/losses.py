"""Composite training loss for NanoTSE.

Aggregates SI-SNR (reconstruction) with three optional auxiliary terms:

* ``infonce``     -- contrastive pull on per-sample pooled slot embeddings using
                     batch speaker IDs. Active by default at weight 0.1.
* ``asd``         -- per-slot active-speaker BCE. Uses an energy-VAD pseudo-
                     target derived from the clean target signal when no
                     external GT is available (gives the ASD head a real
                     gradient signal without inventing per-frame labels).
* ``consistency`` -- slot-bank MSE between two windows. When ``slots_prev``
                     isn't supplied, ``compute_loss`` itself doesn't split
                     clips -- the caller does. Provides a multi-window
                     identity-stability signal once it's wired.

Returns a plain ``dict[str, torch.Tensor]`` rather than a Pydantic model:
the loss is on the hot training path and the validation overhead of a
``BaseModel`` per step would be wasted (configs are Pydantic; tensors are
not). The keys are always present, with zero scalars where a term is
disabled, so logging is uniform across configs.
"""

from __future__ import annotations

import torch

from nanotse.losses import (
    asd_bce,
    multi_res_mag_stft,
    multi_res_mel_loss,
    negative_si_snr,
    slot_consistency,
    slot_infonce,
)
from nanotse.utils.config import LossWeights


def target_active_mask(
    target_audio: torch.Tensor,
    n_frames: int,
    rms_threshold_ratio: float = 0.1,
) -> torch.Tensor:
    """Per-frame VAD mask: True wherever the clean target speaker is active.

    Used to derive an HONEST ASD signal: we know *whether* a slot should
    be active per frame (from target energy), even though we don't yet know
    *which* slot is bound to the target. The proper ASD loss therefore
    supervises only the active/inactive bit, not the slot identity --
    which the caller computes via ``asd_logits.max(dim=-1)`` and BCE.

    Returning the mask (not a slot-index target) avoids the previous
    circular signal where the ASD head's own argmax was fed back as its
    "ground truth," which just reinforces self-confidence without
    learning real slot binding.

    Args:
        target_audio:       ``(B, T)`` clean target waveform.
        n_frames:           target frame count (typically ``asd_logits.shape[1]``).
        rms_threshold_ratio: fraction of clip-median RMS that counts as "active".

    Returns:
        ``(B, n_frames)`` bool mask.
    """
    b = target_audio.shape[0]
    t = target_audio.shape[-1]
    frame_len = max(1, t // n_frames)
    end = frame_len * n_frames
    trimmed = target_audio[..., :end]
    framed = trimmed.reshape(b, n_frames, frame_len)
    rms = framed.pow(2).mean(dim=-1).sqrt()  # (B, n_frames)
    median_rms = rms.median(dim=-1, keepdim=True).values
    return rms > (rms_threshold_ratio * median_rms.clamp_min(1e-8))


def compute_loss(
    *,
    estimate: torch.Tensor,
    target: torch.Tensor,
    slots: torch.Tensor | None,
    asd_logits: torch.Tensor | None,
    speaker_ids: torch.Tensor | None,
    weights: LossWeights,
    target_slot: torch.Tensor | None = None,
    slots_prev: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted sum of SI-SNR + auxiliary losses.

    Args:
        estimate:    ``(B, T)`` model output waveform.
        target:      ``(B, T)`` ground-truth target speaker.
        slots:       ``(B, N, S)`` slot bank from NanoTSE (None if audio-only).
        asd_logits:  ``(B, T, N)`` ASD logits (None if audio-only).
        speaker_ids: ``(B,)`` integer speaker labels for InfoNCE.
        weights:     Validated LossWeights instance.
        target_slot: ``(B, T)`` ASD GT slot indices (only consumed if
            ``weights.asd > 0`` and ASD logits + this arg are provided).
        slots_prev:  ``(B, N, S)`` slot bank from a prior window (only
            consumed if ``weights.consistency > 0``).

    Returns:
        Dict with keys ``total``, ``si_snr``, ``infonce``, ``asd``,
        ``consistency``. Always all five keys; disabled terms are zero
        scalars on the estimate's device.
    """
    device = estimate.device
    zero = torch.zeros((), device=device)

    si_snr_loss = negative_si_snr(estimate, target)

    infonce_loss = zero
    if weights.infonce > 0 and slots is not None and speaker_ids is not None:
        # Mean-pool across slots -> one embedding per sample.
        embeddings = slots.mean(dim=1)
        infonce_loss = slot_infonce(embeddings, speaker_ids)

    asd_loss = zero
    if weights.asd > 0 and asd_logits is not None and target_slot is not None:
        asd_loss = asd_bce(asd_logits, target_slot)

    consistency_loss = zero
    if weights.consistency > 0 and slots is not None and slots_prev is not None:
        consistency_loss = slot_consistency(slots_prev, slots)

    mag_stft_loss = zero
    if weights.mag_stft > 0:
        mag_stft_loss = multi_res_mag_stft(estimate, target)

    mel_loss = zero
    if weights.mel > 0:
        mel_loss = multi_res_mel_loss(estimate, target)

    total = (
        weights.si_snr * si_snr_loss
        + weights.infonce * infonce_loss
        + weights.asd * asd_loss
        + weights.consistency * consistency_loss
        + weights.mag_stft * mag_stft_loss
        + weights.mel * mel_loss
    )

    return {
        "total": total,
        "si_snr": si_snr_loss.detach(),
        "infonce": infonce_loss.detach(),
        "asd": asd_loss.detach(),
        "consistency": consistency_loss.detach(),
        "mag_stft": mag_stft_loss.detach(),
        "mel": mel_loss.detach(),
    }
