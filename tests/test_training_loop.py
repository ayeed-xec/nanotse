"""Tests for the training-loop helpers: checkpoint save/load, best.pt tracking, val pass."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from nanotse.data import SyntheticAVMixDataset
from nanotse.models.baselines.tdse import TDSEBaseline
from nanotse.models.nanotse import NanoTSE
from nanotse.training import (
    load_checkpoint,
    run_val_pass,
    save_checkpoint,
    update_best,
)


def test_save_and_load_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = TDSEBaseline()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Take one optimizer step so the state dict is non-trivial.
    x = torch.randn(2, 1600)
    out = model(x)
    out.sum().backward()
    opt.step()

    ckpt_path = tmp_path / "latest.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=opt,
        step=42,
        best_val=1.23,
        config={"dummy": True},
        model_kwargs={},
    )
    assert ckpt_path.exists()

    # New instances + load.
    model2 = TDSEBaseline()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    step, best_val = load_checkpoint(
        ckpt_path, model=model2, optimizer=opt2, device=torch.device("cpu")
    )
    assert step == 42
    assert best_val == pytest.approx(1.23)

    # Model weights match after load.
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1.data, p2.data)


def test_save_is_atomic(tmp_path: Path) -> None:
    """No ``.tmp`` left behind after a successful save."""
    model = TDSEBaseline()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "latest.pt"
    save_checkpoint(
        ckpt_path, model=model, optimizer=opt, step=1, best_val=0.0, config={}, model_kwargs={}
    )
    assert ckpt_path.exists()
    assert not (tmp_path / "latest.pt.tmp").exists()


def test_update_best_promotes_on_improvement(tmp_path: Path) -> None:
    """When val SDRi beats prev best, best.pt is overwritten from latest.pt."""
    model = TDSEBaseline()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        tmp_path / "latest.pt",
        model=model,
        optimizer=opt,
        step=10,
        best_val=0.0,
        config={},
        model_kwargs={},
    )

    updated, new_best = update_best(tmp_path, val_sdri=5.5, prev_best=2.0)
    assert updated is True
    assert new_best == pytest.approx(5.5)
    assert (tmp_path / "best.pt").exists()


def test_update_best_skips_on_no_improvement(tmp_path: Path) -> None:
    model = TDSEBaseline()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        tmp_path / "latest.pt",
        model=model,
        optimizer=opt,
        step=10,
        best_val=10.0,
        config={},
        model_kwargs={},
    )

    updated, new_best = update_best(tmp_path, val_sdri=2.0, prev_best=10.0)
    assert updated is False
    assert new_best == pytest.approx(10.0)
    assert not (tmp_path / "best.pt").exists()


def test_run_val_pass_audio_only_returns_finite_stats() -> None:
    torch.manual_seed(0)
    ds = SyntheticAVMixDataset(num_clips=4, clip_seconds=1.0, seed=0)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = NanoTSE(with_visual=False, d_model=32, n_heads=2, n_layers=1, cache_len=50)

    stats = run_val_pass(model, loader, torch.device("cpu"))
    assert stats["val_n"] == 4
    for k in ("val_baseline_si_snr_db", "val_estimate_si_snr_db", "val_sdri_db"):
        assert torch.isfinite(torch.tensor(stats[k])), f"{k} = {stats[k]}"


def test_run_val_pass_av_path_returns_finite_stats() -> None:
    torch.manual_seed(0)
    ds = SyntheticAVMixDataset(num_clips=4, clip_seconds=1.0, seed=0, face_size=32)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = NanoTSE(
        with_visual=True,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_visual=32,
        n_slots=4,
        d_slot=32,
        frame_size=32,
    )

    stats = run_val_pass(model, loader, torch.device("cpu"))
    assert stats["val_n"] == 4
    assert torch.isfinite(torch.tensor(stats["val_sdri_db"]))


def test_run_val_pass_max_clips_truncates() -> None:
    ds = SyntheticAVMixDataset(num_clips=10, clip_seconds=1.0, seed=0)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = NanoTSE(with_visual=False, d_model=32, n_heads=2, n_layers=1, cache_len=50)

    stats = run_val_pass(model, loader, torch.device("cpu"), max_clips=3)
    # max_clips=3 with batch_size=2 -> 2 batches consumed (4 items), then break.
    assert stats["val_n"] == 4
