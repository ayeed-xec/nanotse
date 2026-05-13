"""Pydantic config schemas for NanoTSE smoke + cloud training runs.

The single source of truth for hyperparameters. Both `configs/smoke.yaml`
and `configs/a100.yaml` (and `configs/3060.yaml`) round-trip through this
module. Every model uses ``extra="forbid"`` so YAML typos fail at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Device = Literal["cpu", "mps", "cuda", "auto"]
ModelName = Literal["tdse", "memo", "nanotse", "av_skima"]
Precision = Literal["fp32", "bf16"]


class DataConfig(BaseModel):
    """Dataset slice + DataLoader knobs."""

    model_config = ConfigDict(extra="forbid")

    root: Path
    sample_rate: int = Field(default=16000, gt=0)
    fps: int = Field(default=25, gt=0)
    clip_seconds: float = Field(default=4.0, gt=0)
    num_clips: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=4, gt=0)


class LossWeights(BaseModel):
    """Loss schedule weights consumed by ``nanotse.training.losses.compute_loss``.

    Defaults keep ``asd`` and ``consistency`` at ``0.0`` because their
    prerequisites are not yet in the data pipeline (real per-frame ASD GT
    arrives with W7-8 multi-session eval; slot-consistency needs a multi-
    window dataset wired in W5-6). Flip them on in YAML once the labels
    exist -- no code change required.
    """

    model_config = ConfigDict(extra="forbid")

    si_snr: float = Field(default=1.0, ge=0)
    infonce: float = Field(default=0.1, ge=0)
    asd: float = Field(default=0.0, ge=0)
    consistency: float = Field(default=0.0, ge=0)
    # Multi-resolution magnitude STFT loss. Standard "+0.5-1.0 dB free" pairing
    # for time-domain SI-SNR. Default 0 keeps v3 and earlier behaviour; set ~0.5
    # for run-5 onwards. Loss returns single scalar averaged across 3 STFT
    # resolutions (512/1024/2048 n_fft).
    mag_stft: float = Field(default=0.0, ge=0)
    # Multi-resolution log-mel perceptual loss. Complements mag_stft (linear)
    # with perceptual-frequency weighting (mel scale emphasises formant
    # frequencies). Standard companion in HiFi-GAN-style pipelines; typically
    # adds another +0.5 to +1.0 dB on top of mag_stft. Set ~0.2-0.5 for
    # run-6 onwards.
    mel: float = Field(default=0.0, ge=0)


class TrainConfig(BaseModel):
    """Optimizer + scheduler + loss-schedule + checkpoint + validation + DataLoader knobs."""

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(gt=0)
    lr: float = Field(default=3e-4, gt=0)
    log_every: int = Field(default=20, gt=0)
    ckpt_every: int = Field(default=500, gt=0)
    val_every: int = Field(default=200, gt=0)
    val_clips: int = Field(default=50, gt=0)
    loss_weights: LossWeights = Field(default_factory=LossWeights)
    snr_db_low: float = Field(default=0.0)
    snr_db_high: float = Field(default=5.0)
    grad_clip: float = Field(default=5.0, gt=0)
    warmup_steps: int = Field(default=0, ge=0)
    min_lr_ratio: float = Field(default=0.05, ge=0, le=1.0)
    ema_decay: float = Field(default=0.0, ge=0.0, lt=1.0)  # 0 disables EMA
    # DataLoader knobs. num_workers=0 keeps everything in the main process
    # (no parallelism); >0 spins up subprocesses to prefetch batches in
    # parallel with GPU compute -- hides wav/face .npz I/O latency.
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = Field(default=True)
    persistent_workers: bool = Field(default=True)
    prefetch_factor: int = Field(default=2, gt=0)
    # Gradient accumulation: simulate effective batch = data.batch_size * accum_steps
    # without growing VRAM. Each optimizer.step / EMA.update / scheduler tick
    # happens once per accum_steps mini-batches. Loss is scaled by 1/accum_steps
    # so summed grads match a true large-batch run. ``steps`` and all logging/
    # ckpt/val cadences are denominated in *effective* steps.
    accum_steps: int = Field(default=1, gt=0)
    # Mixed-precision autocast for forward+loss. bf16 has the same dynamic range
    # as fp32 (no GradScaler needed) and gives ~1.5-2x speedup on A100/H100 and
    # modest gains on Ampere consumer cards. Default fp32 preserves v3-v5 numerics.
    precision: Precision = Field(default="fp32")


class ModelConfig(BaseModel):
    """Selects which model to train + optional cue toggles + capacity knobs."""

    model_config = ConfigDict(extra="forbid")

    name: ModelName
    with_enrollment: bool = Field(default=False)
    with_visual: bool = Field(default=True)
    # ``with_slots`` and ``with_asd`` are AV-only sub-toggles. Default True
    # preserves v3 behaviour; set False to ablate. ``with_asd=True`` requires
    # ``with_slots=True`` (the head consumes the slot embeddings).
    with_slots: bool = Field(default=True)
    with_asd: bool = Field(default=True)
    # Capacity knobs (only consumed by ``nanotse`` model; ignored otherwise).
    # Defaults match the v2-v5 3060 model (4.75M params). Bump on A100 -- see
    # configs/a100_v1.yaml for the 18-20M target settings.
    d_model: int = Field(default=256, gt=0)
    n_layers: int = Field(default=2, gt=0)
    n_heads: int = Field(default=4, gt=0)


class Config(BaseModel):
    """Top-level run config -- load with :meth:`from_yaml`."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    device: Device = "cpu"
    data: DataConfig
    train: TrainConfig
    model: ModelConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load and validate a YAML config file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
