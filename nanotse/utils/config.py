"""Pydantic config schemas for NanoTSE smoke + cloud training runs.

The single source of truth for hyperparameters. Both `configs/smoke.yaml`
and `configs/a100.yaml` round-trip through this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Device = Literal["cpu", "mps", "cuda"]
ModelName = Literal["tdse", "memo", "nanotse", "av_skima"]


class DataConfig(BaseModel):
    """Dataset slice + DataLoader knobs."""

    root: Path
    sample_rate: int = 16000
    fps: int = 25
    clip_seconds: float = Field(default=4.0, gt=0)
    num_clips: int | None = None
    batch_size: int = Field(default=4, gt=0)


class TrainConfig(BaseModel):
    """Optimizer + scheduler knobs."""

    steps: int = Field(gt=0)
    lr: float = Field(default=3e-4, gt=0)
    log_every: int = Field(default=20, gt=0)
    ckpt_every: int = Field(default=500, gt=0)


class ModelConfig(BaseModel):
    """Selects which model to train."""

    name: ModelName


class Config(BaseModel):
    """Top-level run config — load with :meth:`from_yaml`."""

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
