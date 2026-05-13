"""Checkpoint save/load + best-by-val tracking.

A checkpoint dict on disk contains:

    model:        state_dict
    optimizer:    state_dict
    step:         int    (the training step at save time)
    best_val:     float  (best val SDRi seen so far, used for resume)
    config:       dict   (Config.model_dump() for traceability)
    model_kwargs: dict   (constructor kwargs to rebuild the model)
    ema:          dict | None (EMA shadow state, optional -- pre-EMA-resume
                  checkpoints just won't have this key)

Three flavours of save:

* ``ckpt_{step:06d}.pt`` -- periodic snapshot every cfg.train.ckpt_every steps.
* ``latest.pt``          -- always overwritten; what ``--resume`` reads.
* ``best.pt``            -- only overwritten when val SDRi beats the running best.

We use ``torch.save`` directly; no Pydantic on the disk format because
PyTorch state_dicts are nested ``OrderedDict[str, Tensor]`` and Pydantic
would force a custom serialiser per nesting depth.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    best_val: float,
    config: dict[str, Any],
    model_kwargs: dict[str, Any],
    ema_state: dict[str, torch.Tensor] | None = None,
) -> None:
    """Atomic save: write to ``.tmp`` then rename.

    ``ema_state`` is the EMA.state_dict() if EMA is in use; pass None when EMA
    is disabled. Resuming a checkpoint that includes EMA state restores the
    shadow weights so val/inference quality survives interruption.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val": best_val,
            "config": config,
            "model_kwargs": model_kwargs,
            "ema": ema_state,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    ema_state_target: dict[str, torch.Tensor] | None = None,
) -> tuple[int, float]:
    """Load model + optimizer state in place. Returns ``(step, best_val)``.

    If ``ema_state_target`` is provided (the EMA.shadow dict from the active EMA
    instance) and the checkpoint contains EMA state, its tensors are copied in
    place. Old checkpoints without EMA state silently skip this step -- caller
    must be ready for EMA-from-current-weights on legacy resumes.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ema_state_target is not None and ckpt.get("ema") is not None:
        for k, v in ckpt["ema"].items():
            if k in ema_state_target:
                ema_state_target[k].copy_(v.to(device))
    return int(ckpt["step"]), float(ckpt["best_val"])


def update_best(run_dir: Path, val_sdri: float, prev_best: float) -> tuple[bool, float]:
    """Copy ``latest.pt`` to ``best.pt`` if ``val_sdri`` beats ``prev_best``.

    Returns ``(was_updated, new_best)``.
    """
    latest = run_dir / "latest.pt"
    best = run_dir / "best.pt"
    if val_sdri > prev_best and latest.exists():
        shutil.copyfile(latest, best)
        return True, val_sdri
    return False, prev_best
