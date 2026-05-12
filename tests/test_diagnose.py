"""End-to-end smoke for the diagnose tool: train tiny, save, diagnose, verify wavs."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import torch

from nanotse.eval.diagnose import diagnose
from nanotse.models.nanotse import NanoTSE


def test_diagnose_writes_expected_files(tmp_path: Path) -> None:
    """Train a tiny untrained model, save, diagnose; verify all files appear."""
    kwargs = {"d_model": 32, "n_heads": 2, "n_layers": 1, "cache_len": 20, "with_visual": False}
    model = NanoTSE(**kwargs)
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "model_kwargs": kwargs}, ckpt)

    out = tmp_path / "diagnose_out"
    summary = diagnose(
        ckpt_path=ckpt,
        out_dir=out,
        num_clips=2,
        clip_seconds=0.5,
        data_root=tmp_path / "_does_not_exist",  # forces synthetic fallback
    )

    # Wavs present
    for i in range(2):
        for kind in ("mix", "target", "interferer", "estimate"):
            f = out / f"{i:02d}_{kind}.wav"
            assert f.exists(), f"missing {f}"
            arr, sr = sf.read(f)
            assert sr == 16000
            assert arr.shape[0] == 8000  # 0.5 s @ 16 kHz

    # Summary JSON has the right keys
    js = json.loads((out / "diagnose.json").read_text())
    assert js["num_clips"] == 2
    assert js["has_visual"] is False
    assert len(js["per_clip"]) == 2
    for m in js["per_clip"]:
        assert "baseline_si_snr_db" in m
        assert "estimate_si_snr_db" in m
        assert "si_sdri_db" in m

    # Returned summary matches written JSON
    assert summary["num_clips"] == js["num_clips"]


def test_diagnose_detects_visual_from_ckpt(tmp_path: Path) -> None:
    """A ckpt with visual params should auto-load NanoTSE(with_visual=True)."""
    kwargs = {
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "cache_len": 20,
        "d_visual": 32,
        "n_slots": 2,
        "d_slot": 32,
        "n_slot_iters": 1,
        "frame_size": 16,
        "with_visual": True,
    }
    model = NanoTSE(**kwargs)  # type: ignore[arg-type]
    ckpt = tmp_path / "av_model.pt"
    torch.save({"model": model.state_dict(), "model_kwargs": kwargs}, ckpt)

    out = tmp_path / "av_out"
    summary = diagnose(
        ckpt_path=ckpt,
        out_dir=out,
        num_clips=1,
        clip_seconds=0.5,
        data_root=tmp_path / "_no_real",
    )
    assert summary["has_visual"] is True
