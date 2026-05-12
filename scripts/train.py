#!/usr/bin/env python3
"""Training entrypoint.

Loads a YAML config, builds the requested baseline on synthetic AV data,
runs N optimizer steps, logs metrics to ``runs/<timestamp>/metrics.jsonl``,
and saves a final checkpoint.

W1 status: only the synthetic data path is wired (no real loader yet).
W2 status: ``tdse`` works; other model names raise NotImplementedError.

Usage:
    python scripts/train.py --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nanotse.data import SyntheticAVMixDataset
from nanotse.losses import negative_si_snr, si_snr
from nanotse.models.baselines.tdse import TDSEBaseline
from nanotse.models.nanotse import NanoTSE
from nanotse.utils.config import Config
from nanotse.utils.tracker import Tracker


def _resolve_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            print("WARNING: MPS unavailable, falling back to CPU", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA unavailable, falling back to CPU", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(name: str) -> torch.nn.Module:
    if name == "tdse":
        return TDSEBaseline()
    if name == "nanotse":
        # Audio-only smoke training. AV training (with video=batch["face"]) lands
        # in a follow-up sprint with the real VoxCeleb2 loader.
        return NanoTSE(with_visual=False)
    raise NotImplementedError(f"model '{name}' not implemented yet -- see docs/PLAN.md")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--remote", default=None, help="vast.ai host for cloud runs")
    p.add_argument("--run-name", default=None)
    args = p.parse_args(argv)

    if args.remote:
        print(
            f"Remote training not implemented; would submit to {args.remote}",
            file=sys.stderr,
        )
        return 1

    cfg = Config.from_yaml(args.config)
    torch.manual_seed(cfg.seed)

    device = _resolve_device(cfg.device)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "runs" / (args.run_name or ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    tracker = Tracker(run_dir / "metrics.jsonl")

    print(f"run:    {run_dir.relative_to(repo_root)}")
    print(f"device: {device}   model: {cfg.model.name}   steps: {cfg.train.steps}")

    ds = SyntheticAVMixDataset(
        num_clips=cfg.data.num_clips or 200,
        clip_seconds=cfg.data.clip_seconds,
        sample_rate=cfg.data.sample_rate,
        fps=cfg.data.fps,
        seed=cfg.seed,
    )
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=True)

    model = _build_model(cfg.model.name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    with torch.no_grad():
        first = next(iter(loader))
        mix, tgt = first["mix"].to(device), first["target"].to(device)
        baseline = si_snr(mix, tgt).mean().item()
        tracker.log(0, baseline_si_snr_db=baseline)
        print(f"baseline SI-SNR(mix, target): {baseline:+.2f} dB")

    step = 0
    while step < cfg.train.steps:
        for batch in loader:
            if step >= cfg.train.steps:
                break
            mix = batch["mix"].to(device)
            tgt = batch["target"].to(device)
            out = model(mix)
            est = out[0] if isinstance(out, tuple) else out
            loss = negative_si_snr(est, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % cfg.train.log_every == 0:
                with torch.no_grad():
                    sdr_val = si_snr(est, tgt).mean().item()
                tracker.log(step, loss=loss.item(), si_snr_db=sdr_val)
                print(f"  step {step:4d}  loss {loss.item():+.3f}  SI-SNR {sdr_val:+.2f} dB")

    ckpt = run_dir / "model.pt"
    torch.save({"model": model.state_dict(), "config": cfg.model_dump()}, ckpt)
    print(f"\nsaved {ckpt.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
