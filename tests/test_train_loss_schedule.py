"""Loss-schedule wiring: ``compute_loss`` + scripts/train.py end-to-end.

Covers:
* per-term gating by ``LossWeights`` (zero weight -> zero contribution),
* gradient flow through slots / asd_logits when their losses are active,
* `scripts/train.py` writes ``loss_total``/``loss_si_snr``/``loss_infonce``
  rows to ``metrics.jsonl`` and SI-SNR component decreases over 30 steps.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from nanotse.models.nanotse import NanoTSE
from nanotse.training import compute_loss
from nanotse.utils.config import LossWeights

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compute_loss_returns_all_keys() -> None:
    est = torch.randn(2, 1600)
    tgt = torch.randn(2, 1600)
    losses = compute_loss(
        estimate=est,
        target=tgt,
        slots=None,
        asd_logits=None,
        speaker_ids=None,
        weights=LossWeights(),
    )
    assert set(losses.keys()) == {"total", "si_snr", "infonce", "asd", "consistency"}
    for v in losses.values():
        assert torch.isfinite(v).all()


def test_compute_loss_zero_weights_yield_si_snr_only() -> None:
    """When infonce/asd/consistency weights are all 0, total == si_snr * si_snr_weight."""
    est = torch.randn(2, 1600)
    tgt = torch.randn(2, 1600)
    slots = torch.randn(2, 4, 32)
    speaker_ids = torch.tensor([0, 0])
    weights = LossWeights(si_snr=1.0, infonce=0.0, asd=0.0, consistency=0.0)
    losses = compute_loss(
        estimate=est,
        target=tgt,
        slots=slots,
        asd_logits=None,
        speaker_ids=speaker_ids,
        weights=weights,
    )
    assert torch.allclose(losses["total"], losses["si_snr"])
    assert losses["infonce"].item() == 0.0
    assert losses["asd"].item() == 0.0
    assert losses["consistency"].item() == 0.0


def test_compute_loss_infonce_active_with_collisions() -> None:
    """With a batch containing speaker collisions, infonce contributes a nonzero loss."""
    torch.manual_seed(0)
    est = torch.randn(4, 1600)
    tgt = torch.randn(4, 1600)
    slots = torch.randn(4, 4, 32)
    speaker_ids = torch.tensor([0, 0, 1, 1])  # guarantees two positive pairs
    weights = LossWeights(si_snr=1.0, infonce=0.5)
    losses = compute_loss(
        estimate=est,
        target=tgt,
        slots=slots,
        asd_logits=None,
        speaker_ids=speaker_ids,
        weights=weights,
    )
    assert losses["infonce"].item() > 0.0


def test_compute_loss_gradients_flow_through_slots() -> None:
    """InfoNCE must produce gradients on the slot bank (slot training signal)."""
    torch.manual_seed(0)
    est = torch.randn(4, 1600)
    tgt = torch.randn(4, 1600)
    slots = torch.randn(4, 4, 32, requires_grad=True)
    speaker_ids = torch.tensor([0, 0, 1, 1])
    weights = LossWeights(si_snr=0.0, infonce=1.0)  # isolate the infonce gradient
    losses = compute_loss(
        estimate=est,
        target=tgt,
        slots=slots,
        asd_logits=None,
        speaker_ids=speaker_ids,
        weights=weights,
    )
    losses["total"].backward()
    assert slots.grad is not None
    assert torch.isfinite(slots.grad).all()
    assert slots.grad.abs().sum() > 0


def test_compute_loss_full_av_path_backward_no_nan() -> None:
    """End-to-end model -> compute_loss -> backward on a tiny AV batch."""
    torch.manual_seed(0)
    model = NanoTSE(
        d_model=32, n_heads=2, n_layers=1, d_visual=32, n_slots=4, d_slot=32, frame_size=32
    )
    audio = torch.randn(2, 8000)
    video = torch.randint(0, 256, (2, 12, 32, 32, 3), dtype=torch.uint8)
    speaker_ids = torch.tensor([0, 0])

    est, asd_logits, slots = model(audio, video)
    losses = compute_loss(
        estimate=est,
        target=audio,
        slots=slots,
        asd_logits=asd_logits,
        speaker_ids=speaker_ids,
        weights=LossWeights(si_snr=1.0, infonce=0.1),
    )
    losses["total"].backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@pytest.mark.slow
def test_train_script_writes_loss_components_and_si_snr_drops(tmp_path: Path) -> None:
    """End-to-end: scripts/train.py runs 30 steps, logs all loss components,
    and si_snr loss decreases on the synthetic dataset."""
    cfg_path = tmp_path / "test_cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "device": "cpu",
                "data": {
                    "root": str(tmp_path / "nodata"),  # forces synthetic fallback
                    "clip_seconds": 1.0,
                    "num_clips": 16,
                    "batch_size": 4,
                },
                "train": {
                    "steps": 30,
                    "lr": 3.0e-3,
                    "log_every": 5,
                    "ckpt_every": 100,
                    "loss_weights": {
                        "si_snr": 1.0,
                        "infonce": 0.1,
                        "asd": 0.0,
                        "consistency": 0.0,
                    },
                },
                "model": {"name": "nanotse"},
            }
        )
    )

    run_name = "test_loss_schedule"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--config",
            str(cfg_path),
            "--run-name",
            run_name,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"train.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    metrics_path = REPO_ROOT / "runs" / run_name / "metrics.jsonl"
    assert metrics_path.exists(), "metrics.jsonl not written"

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    train_rows = [r for r in rows if "loss_total" in r]
    assert len(train_rows) >= 3, f"expected >=3 train rows, got {len(train_rows)}: {train_rows}"

    expected_keys = {"loss_total", "loss_si_snr", "loss_infonce", "loss_asd", "loss_consistency"}
    for r in train_rows:
        assert expected_keys.issubset(r.keys()), f"missing keys in {r}"
        assert r["loss_asd"] == 0.0  # weight 0 -> contribution 0
        assert r["loss_consistency"] == 0.0

    first_si_snr = train_rows[0]["loss_si_snr"]
    last_si_snr = train_rows[-1]["loss_si_snr"]
    assert last_si_snr < first_si_snr, (
        f"si_snr loss did not decrease: {first_si_snr:.3f} -> {last_si_snr:.3f}"
    )
