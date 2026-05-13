#!/usr/bin/env python3
"""Visualization toolkit for NanoTSE.

Subcommands:

    training        plot one run's curves (train SI-SNR, val SDRi, losses)
    compare         plot multiple runs side-by-side (train + val + losses)
    dataset         distribution plots for data/v2/ (speakers, clips, face_ok)
    spectrograms    mel-spectrogram triptychs (mix vs target vs estimate) from
                    a checkpoint -- requires GPU/CPU inference

All outputs are PNGs in <run_dir>/viz/ or in --output-dir for compare/dataset.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------- helpers


def _load_run(run_dir: Path) -> tuple[list[dict], list[dict]]:
    """Returns ``(train_rows, val_rows)`` sorted by step."""
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        raise FileNotFoundError(f"missing {metrics}")
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    train = sorted(
        [r for r in rows if "si_snr_db" in r and "val_n" not in r], key=lambda r: r["step"]
    )
    val = sorted([r for r in rows if "val_sdri_db" in r], key=lambda r: r["step"])
    return train, val


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Returns simple rolling mean; first ``window-1`` entries pad with cumulative mean."""
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _format_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# ---------------------------------------------------------------------- training


def plot_training(run_dir: Path, out_dir: Path) -> None:
    train, val = _load_run(run_dir)
    if not train:
        print(f"no train rows in {run_dir}")
        return
    steps = [r["step"] for r in train]
    si_snr = [r["si_snr_db"] for r in train]
    si_snr_smooth = _rolling_mean(si_snr, 20)
    best_so_far = [max(si_snr[: i + 1]) for i in range(len(si_snr))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"NanoTSE training — {run_dir.name}", fontsize=14, y=1.00)

    # Train SI-SNR over steps
    ax = axes[0, 0]
    ax.plot(steps, si_snr, color="lightsteelblue", alpha=0.5, label="per-batch")
    ax.plot(steps, si_snr_smooth, color="C0", linewidth=2, label="rolling mean (20)")
    ax.plot(steps, best_so_far, color="C2", linestyle="--", linewidth=1.5, label="best so far")
    ax.axhline(0, color="gray", linewidth=0.5)
    _format_axes(ax, "Train SI-SNR per batch", "Step", "dB")
    ax.legend(loc="lower right", fontsize=9)

    # Val SDRi over steps
    ax = axes[0, 1]
    if val:
        v_steps = [r["step"] for r in val]
        v_sdri = [r["val_sdri_db"] for r in val]
        ax.plot(v_steps, v_sdri, marker="o", color="C3", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.5)
        for s, v in zip(v_steps, v_sdri):
            ax.annotate(f"{v:+.2f}", (s, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no val passes yet", ha="center", va="center", transform=ax.transAxes)
    _format_axes(ax, "Val SDRi over training", "Step", "dB")

    # Loss components over steps
    ax = axes[1, 0]
    loss_keys = [("loss_si_snr", "SI-SNR (neg)"), ("loss_mag_stft", "mag-STFT"), ("loss_mel", "mel"), ("loss_infonce", "InfoNCE")]
    for key, label in loss_keys:
        if key in train[0]:
            vals = [r.get(key, np.nan) for r in train]
            if any(not np.isnan(v) for v in vals):
                smooth = _rolling_mean(vals, 20)
                ax.plot(steps, smooth, label=label, linewidth=1.5)
    _format_axes(ax, "Loss components (rolling mean 20)", "Step", "value")
    ax.set_yscale("symlog")
    ax.legend(loc="upper right", fontsize=9)

    # Wall-clock throughput
    ax = axes[1, 1]
    if "t" in train[0]:
        t = [r["t"] for r in train]
        ax.plot(t, steps, color="C4", linewidth=2)
        _format_axes(ax, "Wall-clock progress", "Seconds elapsed", "Step")
        if len(train) > 1:
            sps = train[-1]["step"] / max(1, train[-1]["t"])
            ax.text(
                0.05, 0.95,
                f"avg sps: {sps:.2f}\nETA at current rate: {(12500 - train[-1]['step']) / sps / 3600:.1f}h",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    plt.tight_layout()
    out = out_dir / "training.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------- compare


def plot_compare(run_dirs: list[Path], out_dir: Path, labels: list[str] | None = None) -> None:
    if not run_dirs:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels or [d.name for d in run_dirs]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"][: len(run_dirs)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Run comparison — train SI-SNR (rolling) + val SDRi", fontsize=14, y=1.02)

    for run_dir, label, color in zip(run_dirs, labels, colors):
        try:
            train, val = _load_run(run_dir)
        except FileNotFoundError:
            print(f"skipping {run_dir}: no metrics")
            continue

        steps = [r["step"] for r in train]
        si_snr_smooth = _rolling_mean([r["si_snr_db"] for r in train], 20)
        axes[0].plot(steps, si_snr_smooth, color=color, label=label, linewidth=1.8)

        if val:
            v_steps = [r["step"] for r in val]
            v_sdri = [r["val_sdri_db"] for r in val]
            axes[1].plot(v_steps, v_sdri, marker="o", color=color, label=label, linewidth=1.8)

    _format_axes(axes[0], "Train SI-SNR rolling-20 mean", "Step", "dB")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].legend(loc="lower right", fontsize=9)
    _format_axes(axes[1], "Val SDRi over steps", "Step", "dB")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out = out_dir / "compare.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------- dataset


def plot_dataset(data_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        print(f"missing {manifest_path}")
        return
    manifest = json.loads(manifest_path.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Dataset stats — {data_root}", fontsize=14, y=1.00)

    # Clips per speaker (train + val separately)
    for ax, split_name, split_color in [
        (axes[0, 0], "train", "C0"),
        (axes[0, 1], "val", "C3"),
    ]:
        clips_per_spk: dict[str, int] = {}
        for r in manifest.get(split_name, []):
            clips_per_spk[r["speaker_id"]] = clips_per_spk.get(r["speaker_id"], 0) + 1
        counts = list(clips_per_spk.values())
        if counts:
            ax.hist(counts, bins=40, color=split_color, alpha=0.7, edgecolor="white")
            ax.axvline(np.median(counts), color="black", linestyle="--", linewidth=1, label=f"median={np.median(counts):.0f}")
            ax.legend(loc="upper right", fontsize=9)
        _format_axes(ax, f"{split_name}: clips per speaker", "Clips", "Speakers")
        ax.text(
            0.97, 0.85,
            f"speakers: {len(counts)}\nclips: {sum(counts)}",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    # face_ok ratios (per-speaker, weighted)
    ax = axes[1, 0]
    faces_root = data_root / "faces"
    if faces_root.exists():
        per_spk_face_ok: list[float] = []
        speakers = sorted({r["speaker_id"] for split in ("train", "val") for r in manifest.get(split, [])})
        for spk in speakers:
            spk_dir = faces_root / spk
            if not spk_dir.exists():
                continue
            weighted_ok = 0.0
            total = 0
            for npz_path in spk_dir.glob("*.npz"):
                try:
                    arr = np.load(npz_path)["face_ok"]
                    weighted_ok += float(arr.sum())
                    total += int(arr.size)
                except Exception:
                    continue
            if total:
                per_spk_face_ok.append(weighted_ok / total)
        if per_spk_face_ok:
            ax.hist(per_spk_face_ok, bins=30, color="C2", alpha=0.7, edgecolor="white")
            ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="face_ok=0.5 (filter threshold)")
            ax.axvline(np.median(per_spk_face_ok), color="black", linestyle="--", linewidth=1, label=f"median={np.median(per_spk_face_ok):.2f}")
            ax.legend(loc="upper left", fontsize=9)
        _format_axes(ax, "face_ok ratio per speaker (frame-weighted)", "face_ok ratio", "Speakers")
    else:
        ax.text(0.5, 0.5, "no faces/ dir", ha="center", va="center", transform=ax.transAxes)

    # Train vs val speaker overlap (should be 0)
    ax = axes[1, 1]
    train_spk = {r["speaker_id"] for r in manifest.get("train", [])}
    val_spk = {r["speaker_id"] for r in manifest.get("val", [])}
    overlap = train_spk & val_spk
    only_train = len(train_spk - val_spk)
    only_val = len(val_spk - train_spk)
    ax.bar(["train only", "val only", "overlap"], [only_train, only_val, len(overlap)], color=["C0", "C3", "red"])
    _format_axes(ax, "Speaker disjointness check", "Group", "Count")
    if not overlap:
        ax.text(0.5, 0.95, "OK: speaker-disjoint", ha="center", va="top", transform=ax.transAxes, color="green", fontsize=11)
    else:
        ax.text(0.5, 0.95, f"FAIL: {len(overlap)} overlap", ha="center", va="top", transform=ax.transAxes, color="red", fontsize=11)

    plt.tight_layout()
    out = out_dir / "dataset.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------- spectrograms


def plot_spectrograms(ckpt_path: Path, data_root: Path, out_dir: Path, n_samples: int = 4, device: str = "auto") -> None:
    """Inference-based: pick N val samples, run them, plot mix vs target vs estimate spectrograms."""
    import torch  # imported here so dataset/training plots don't pay torch import cost

    from nanotse.data import VoxCeleb2MixDataset
    from nanotse.models.nanotse import NanoTSE

    out_dir.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {ckpt_path} -> {device}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_kwargs = ckpt.get("model_kwargs", {})
    model = NanoTSE(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = VoxCeleb2MixDataset(data_root, split="val", num_items=n_samples, seed=42)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    n_fft, hop = 512, 128
    win = torch.hann_window(n_fft).to(device)

    def _spec(x: torch.Tensor) -> np.ndarray:
        s = torch.stft(x.to(device), n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True, center=True)
        return torch.log10(s.abs().clamp_min(1e-6)).cpu().numpy()

    with torch.no_grad():
        for i in range(n_samples):
            sample = ds[i]
            mix = sample["mix"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            video = sample["face"].unsqueeze(0).to(device)
            enroll = sample["enrollment"].unsqueeze(0).to(device)
            est, _, _ = model(mix, video, enroll)
            for j, (sig, title) in enumerate(zip([mix[0], target[0], est[0]], ["mix", "target", "estimate"])):
                spec = _spec(sig)
                ax = axes[i, j]
                ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
                if i == 0:
                    ax.set_title(title)
                if j == 0:
                    ax.set_ylabel(f"sample {i}\nfreq bins")
                ax.set_xticks([])
                ax.set_yticks([])

    fig.suptitle(f"Spectrogram triptych — {ckpt_path.name}", fontsize=14, y=1.00)
    plt.tight_layout()
    out = out_dir / "spectrograms.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("training", help="single-run training curves")
    pt.add_argument("--run", type=Path, required=True)

    pc = sub.add_parser("compare", help="multi-run comparison")
    pc.add_argument("--runs", type=Path, nargs="+", required=True)
    pc.add_argument("--labels", nargs="+", default=None)
    pc.add_argument("--output-dir", type=Path, default=Path("runs/_viz_compare"))

    pd = sub.add_parser("dataset", help="dataset distributions")
    pd.add_argument("--root", type=Path, default=Path("data/v2"))
    pd.add_argument("--output-dir", type=Path, default=Path("runs/_viz_dataset"))

    ps = sub.add_parser("spectrograms", help="mix/target/estimate spectrogram triptychs from a checkpoint")
    ps.add_argument("--ckpt", type=Path, required=True)
    ps.add_argument("--root", type=Path, default=Path("data/v2"))
    ps.add_argument("--output-dir", type=Path, default=None)
    ps.add_argument("--n-samples", type=int, default=4)
    ps.add_argument("--device", default="auto")

    args = p.parse_args(argv)

    if args.cmd == "training":
        plot_training(args.run, args.run / "viz")
    elif args.cmd == "compare":
        plot_compare(args.runs, args.output_dir, args.labels)
    elif args.cmd == "dataset":
        plot_dataset(args.root, args.output_dir)
    elif args.cmd == "spectrograms":
        out = args.output_dir or args.ckpt.parent / "viz"
        plot_spectrograms(args.ckpt, args.root, out, args.n_samples, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
