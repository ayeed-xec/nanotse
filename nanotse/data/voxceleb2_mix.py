"""VoxCeleb2-mix dataset loader plus a deterministic synthetic-data fallback.

The synthetic path lets the smoke-overfit plumbing test (and CI) run without
the 40+ GB HuggingFace download. Once `scripts/data_prep/fetch_voxceleb2_mix_smoke.py`
has populated ``data/smoke/`` with real wavs + manifest, swap in
``VoxCeleb2MixDataset`` (W2).
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch.utils.data import Dataset


class AVMixSample(TypedDict):
    """One audio-visual training item — the contract all loaders satisfy."""

    mix: torch.Tensor  # (T,)        float32 @ sample_rate
    target: torch.Tensor  # (T,)        float32 @ sample_rate (the speaker we want)
    interferer: torch.Tensor  # (T,)        float32 @ sample_rate
    face: torch.Tensor  # (F, H, W, 3) uint8  @ fps        (target speaker face)
    speaker_id: int
    mix_id: int


class SyntheticAVMixDataset(Dataset[AVMixSample]):
    """Deterministic synthetic AV mixes for plumbing tests.

    Each index produces a reproducible random ``(target, interferer, mix, face)``
    tuple. The audio model never needs real data to exercise its forward / backward
    pass — that's what this dataset is for.

    Re-seeded per item from ``seed + idx`` so ``ds[7]`` is always the same clip
    even across processes.
    """

    def __init__(
        self,
        num_clips: int = 200,
        clip_seconds: float = 4.0,
        sample_rate: int = 16000,
        fps: int = 25,
        face_size: int = 112,
        interferer_scale: float = 0.5,
        seed: int = 0,
    ) -> None:
        if num_clips <= 0:
            raise ValueError(f"num_clips must be > 0, got {num_clips}")
        if clip_seconds <= 0:
            raise ValueError(f"clip_seconds must be > 0, got {clip_seconds}")

        self.num_clips = num_clips
        self.clip_samples = int(clip_seconds * sample_rate)
        self.clip_frames = max(1, int(clip_seconds * fps))
        self.sample_rate = sample_rate
        self.fps = fps
        self.face_size = face_size
        self.interferer_scale = interferer_scale
        self.seed = seed

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> AVMixSample:
        if idx < 0 or idx >= self.num_clips:
            raise IndexError(idx)

        gen = torch.Generator().manual_seed(self.seed + idx)

        target = torch.randn(self.clip_samples, generator=gen)
        interferer = torch.randn(self.clip_samples, generator=gen) * self.interferer_scale
        mix = target + interferer

        face = torch.randint(
            0,
            256,
            (self.clip_frames, self.face_size, self.face_size, 3),
            generator=gen,
            dtype=torch.uint8,
        )

        speaker_id = int(torch.randint(0, 1_000_000, (1,), generator=gen).item())

        return AVMixSample(
            mix=mix,
            target=target,
            interferer=interferer,
            face=face,
            speaker_id=speaker_id,
            mix_id=idx,
        )
