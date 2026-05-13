#!/usr/bin/env python3
"""Training entrypoint with periodic validation, checkpointing, and resume.

Loads a YAML config, builds the requested model, runs the training loop,
logs metrics to ``runs/<ts>/metrics.jsonl``, and saves checkpoints under
``runs/<ts>/`` as ``ckpt_<step>.pt`` + ``latest.pt`` + ``best.pt`` (the
last keyed on highest val SDRi).

Data source: auto-detected -- if ``cfg.data.root / manifest.json`` exists we
use the real VoxCeleb2MixDataset (train + val splits), otherwise we fall
back to SyntheticAVMixDataset (train only, val pass becomes a no-op).

Loss schedule: ``cfg.train.loss_weights`` (Pydantic LossWeights) drives the
weighted sum from :func:`nanotse.training.compute_loss`.

Batch shaping: when real data is in use we build a ``StratifiedSpeaker
BatchSampler`` so every batch contains positive pairs for InfoNCE. The
synthetic fallback uses plain ``DataLoader(shuffle=True)``.

Usage:
    python scripts/train.py --config configs/3060.yaml
    python scripts/train.py --config configs/3060.yaml --resume <run_name>
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from nanotse.data import (
    StratifiedSpeakerBatchSampler,
    SyntheticAVMixDataset,
    VoxCeleb2MixDataset,
)
from nanotse.losses import si_snr
from nanotse.models.baselines.tdse import TDSEBaseline
from nanotse.models.nanotse import NanoTSE
from nanotse.training import (
    EMA,
    compute_loss,
    load_checkpoint,
    run_val_pass,
    save_checkpoint,
    update_best,
    warmup_cosine_lr_multiplier,
)
from nanotse.utils.config import Config, LossWeights
from nanotse.utils.tracker import Tracker


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
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


def _build_model(cfg: Config) -> torch.nn.Module:
    name = cfg.model.name
    if name == "tdse":
        return TDSEBaseline()
    if name == "nanotse":
        return NanoTSE(
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            with_visual=cfg.model.with_visual,
            with_enrollment=cfg.model.with_enrollment,
            with_slots=cfg.model.with_slots,
            with_asd=cfg.model.with_asd,
        )
    raise NotImplementedError(f"model '{name}' not implemented yet -- see docs/PLAN.md")


def _build_datasets(
    cfg: Config,
) -> tuple[SyntheticAVMixDataset | VoxCeleb2MixDataset, VoxCeleb2MixDataset | None]:
    """Returns ``(train_ds, val_ds_or_None)``.

    Real data path uses both train and val splits; synthetic fallback
    has no val (val pass is then a no-op so the loop logs zeros).
    """
    real_manifest = cfg.data.root / "manifest.json"
    if real_manifest.exists():
        real_train = VoxCeleb2MixDataset(
            cfg.data.root,
            split="train",
            clip_seconds=cfg.data.clip_seconds,
            sample_rate=cfg.data.sample_rate,
            fps=cfg.data.fps,
            num_items=cfg.data.num_clips or 200,
            seed=cfg.seed,
            snr_db_range=(cfg.train.snr_db_low, cfg.train.snr_db_high),
        )
        real_val = VoxCeleb2MixDataset(
            cfg.data.root,
            split="val",
            clip_seconds=cfg.data.clip_seconds,
            sample_rate=cfg.data.sample_rate,
            fps=cfg.data.fps,
            num_items=cfg.train.val_clips,
            seed=cfg.seed + 1,  # different seed so val mixes differ from train
            snr_db_range=(cfg.train.snr_db_low, cfg.train.snr_db_high),
        )
        return real_train, real_val

    synth_train = SyntheticAVMixDataset(
        num_clips=cfg.data.num_clips or 200,
        clip_seconds=cfg.data.clip_seconds,
        sample_rate=cfg.data.sample_rate,
        fps=cfg.data.fps,
        seed=cfg.seed,
    )
    return synth_train, None


def _build_train_loader(
    train_ds: SyntheticAVMixDataset | VoxCeleb2MixDataset,
    cfg: Config,
) -> DataLoader[Any]:
    """Stratified sampler when real, plain shuffle when synthetic.

    When the dataset has a partial face cache (some speakers only), restrict
    the sampler to AV-eligible speakers so the visual stream sees real lips
    most batches. Falls back to all speakers when no face cache exists.

    DataLoader workers (cfg.train.num_workers > 0) prefetch batches in
    parallel subprocesses, hiding wav + face-npz I/O latency.
    """
    loader_kwargs: dict[str, Any] = {
        "num_workers": cfg.train.num_workers,
        "pin_memory": cfg.train.pin_memory,
    }
    if cfg.train.num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.train.persistent_workers
        loader_kwargs["prefetch_factor"] = cfg.train.prefetch_factor

    if isinstance(train_ds, VoxCeleb2MixDataset):
        eligible: set[int] | None = None
        if cfg.model.name == "nanotse" and cfg.model.with_visual:
            with_faces = train_ds.speakers_with_face_cache()
            if 0 < len(with_faces) < len(train_ds.speakers):
                eligible = with_faces
                print(
                    f"sampler: filtering to {len(eligible)}/{len(train_ds.speakers)} "
                    f"speakers with face cache (rest get zero-fallback frames)"
                )
        try:
            sampler = StratifiedSpeakerBatchSampler(
                train_ds,
                batch_size=cfg.data.batch_size,
                items_per_speaker=2,
                seed=cfg.seed,
                eligible_speakers=eligible,
            )
            print(
                f"sampler: stratified, {len(sampler)} batches/epoch, "
                f"{sampler.speakers_per_batch} spk x {sampler.items_per_speaker} clips"
                f"   workers={cfg.train.num_workers} pin_memory={cfg.train.pin_memory}"
            )
            return DataLoader(train_ds, batch_sampler=sampler, **loader_kwargs)
        except ValueError as e:
            print(
                f"WARNING: stratified sampler unusable ({e}); falling back to random shuffle",
                file=sys.stderr,
            )
    return DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True, **loader_kwargs)


def _forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    mix = batch["mix"].to(device)
    if isinstance(model, NanoTSE):
        video = batch["face"].to(device) if model.with_visual and "face" in batch else None
        enroll = (
            batch["enrollment"].to(device)
            if model.with_enrollment and "enrollment" in batch
            else None
        )
        tse_out, asd_logits, slots = model(mix, video, enroll)
        return tse_out, asd_logits, slots
    baseline_out: torch.Tensor = model(mix)
    return baseline_out, None, None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--resume",
        default=None,
        help="run name to resume from (loads runs/<name>/latest.pt + optimizer)",
    )
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    weights: LossWeights = cfg.train.loss_weights

    repo_root = Path(__file__).resolve().parents[1]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = repo_root / "runs" / (args.run_name or args.resume or ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    tracker = Tracker(run_dir / "metrics.jsonl", append=bool(args.resume))

    print(f"run:    {run_dir.relative_to(repo_root)}")
    print(f"device: {device}   model: {cfg.model.name}   steps: {cfg.train.steps}")
    print(
        f"loss:   si_snr={weights.si_snr} infonce={weights.infonce} "
        f"asd={weights.asd} consistency={weights.consistency}   "
        f"grad_clip={cfg.train.grad_clip}"
    )
    print(f"snr_db: ({cfg.train.snr_db_low}, {cfg.train.snr_db_high})")

    train_ds, val_ds = _build_datasets(cfg)
    print(f"data:   train n={len(train_ds)}  val n={len(val_ds) if val_ds else 0}")
    loader = _build_train_loader(train_ds, cfg)
    if val_ds is not None:
        val_loader_kwargs: dict[str, Any] = {
            "num_workers": cfg.train.num_workers,
            "pin_memory": cfg.train.pin_memory,
        }
        if cfg.train.num_workers > 0:
            val_loader_kwargs["persistent_workers"] = cfg.train.persistent_workers
            val_loader_kwargs["prefetch_factor"] = cfg.train.prefetch_factor
        val_loader = DataLoader(
            val_ds, batch_size=cfg.data.batch_size, shuffle=False, **val_loader_kwargs
        )
    else:
        val_loader = None

    model = _build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    ema = EMA(model, decay=cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
    if ema is not None:
        print(f"ema:    decay={cfg.train.ema_decay}")
    if cfg.train.warmup_steps > 0:
        print(
            f"lr:     warmup_cosine  warmup={cfg.train.warmup_steps}  min_ratio={cfg.train.min_lr_ratio}"
        )
    accum = cfg.train.accum_steps
    if accum > 1:
        eff_batch = cfg.data.batch_size * accum
        print(f"accum:  accum_steps={accum}  effective_batch={eff_batch}")
    # Mixed precision: bf16 on CUDA is the modern default for A100/H100 -- same
    # range as fp32 (no GradScaler) and ~1.5-2x speedup. Disabled (fp32) for CPU/MPS.
    use_amp = cfg.train.precision == "bf16" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.train.precision == "bf16" else torch.float32
    if use_amp:
        print(f"amp:    autocast={cfg.train.precision} (CUDA)")

    model_kwargs: dict[str, object] = (
        {
            "d_model": cfg.model.d_model,
            "n_layers": cfg.model.n_layers,
            "n_heads": cfg.model.n_heads,
            "with_visual": cfg.model.with_visual,
            "with_enrollment": cfg.model.with_enrollment,
            "with_slots": cfg.model.with_slots,
            "with_asd": cfg.model.with_asd,
        }
        if cfg.model.name == "nanotse"
        else {}
    )

    start_step = 0
    best_val = float("-inf")
    if args.resume:
        latest = run_dir / "latest.pt"
        if not latest.exists():
            print(f"ERROR: --resume {args.resume} but {latest} not found", file=sys.stderr)
            return 2
        start_step, best_val = load_checkpoint(
            latest,
            model=model,
            optimizer=opt,
            device=device,
            ema_state_target=ema.shadow if ema is not None else None,
        )
        print(f"resumed from step {start_step}  (best_val = {best_val:+.2f} dB)")

    # Baseline (only on a fresh run)
    if start_step == 0:
        with torch.no_grad():
            first = next(iter(loader))
            mix0, tgt0 = first["mix"].to(device), first["target"].to(device)
            baseline = si_snr(mix0, tgt0).mean().item()
            tracker.log(0, baseline_si_snr_db=baseline)
            print(f"baseline SI-SNR(mix, target): {baseline:+.2f} dB")

    step = start_step
    epoch = 0
    micro_in_step = 0
    while step < cfg.train.steps:
        if isinstance(loader.batch_sampler, StratifiedSpeakerBatchSampler):
            loader.batch_sampler.set_epoch(epoch)
        for batch in loader:
            if step >= cfg.train.steps:
                break
            tgt = batch["target"].to(device)
            speaker_ids = batch["speaker_id"].to(device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                est, asd_logits, slots = _forward(model, batch, device)

                # ASD loss currently dormant (weight 0). Real per-frame GT lands
                # in W7-8; until then nanotse.training.losses.target_active_mask
                # is the honest VAD signal but isn't fed in -- avoids a circular
                # self-target.
                losses = compute_loss(
                    estimate=est,
                    target=tgt,
                    slots=slots,
                    asd_logits=asd_logits,
                    speaker_ids=speaker_ids,
                    weights=weights,
                )
            # Gradient accumulation: zero grads at the start of each effective
            # step, scale loss by 1/accum so summed grads match a true large
            # batch, then only step/EMA/clip every ``accum`` micro-batches.
            if micro_in_step == 0:
                opt.zero_grad()
            total_loss: torch.Tensor = losses["total"] / accum
            total_loss.backward()  # type: ignore[no-untyped-call]
            micro_in_step += 1
            if micro_in_step < accum:
                continue
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            # Apply LR schedule (warmup + cosine) if configured.
            if cfg.train.warmup_steps > 0:
                mult = warmup_cosine_lr_multiplier(
                    step + 1,
                    warmup_steps=cfg.train.warmup_steps,
                    total_steps=cfg.train.steps,
                    min_lr_ratio=cfg.train.min_lr_ratio,
                )
                for g in opt.param_groups:
                    g["lr"] = cfg.train.lr * mult
            opt.step()
            if ema is not None:
                ema.update(model)
            step += 1
            micro_in_step = 0

            if step % cfg.train.log_every == 0:
                with torch.no_grad():
                    sdr_val = si_snr(est, tgt).mean().item()
                tracker.log(
                    step,
                    loss_total=losses["total"].item(),
                    loss_si_snr=losses["si_snr"].item(),
                    loss_infonce=losses["infonce"].item(),
                    loss_asd=losses["asd"].item(),
                    loss_consistency=losses["consistency"].item(),
                    loss_mag_stft=losses["mag_stft"].item() if "mag_stft" in losses else 0.0,
                    loss_mel=losses["mel"].item() if "mel" in losses else 0.0,
                    si_snr_db=sdr_val,
                )
                print(
                    f"  step {step:5d}  total {losses['total'].item():+.3f}  "
                    f"si_snr {losses['si_snr'].item():+.3f}  "
                    f"infonce {losses['infonce'].item():.3f}  "
                    f"SI-SNR {sdr_val:+.2f} dB"
                )

            if val_loader is not None and step % cfg.train.val_every == 0:
                # Use EMA weights for val if enabled -- typically more accurate
                # estimate of the model the user would deploy.
                backup: dict[str, torch.Tensor] = ema.apply_to(model) if ema is not None else {}
                val_stats = run_val_pass(model, val_loader, device, max_clips=cfg.train.val_clips)
                if ema is not None and backup:
                    ema.restore(model, backup)
                tracker.log(step, **val_stats)
                print(
                    f"  step {step:5d}  VAL  baseline {val_stats['val_baseline_si_snr_db']:+.2f} dB"
                    f"  est {val_stats['val_estimate_si_snr_db']:+.2f} dB"
                    f"  SDRi {val_stats['val_sdri_db']:+.2f} dB  n={val_stats['val_n']}"
                )
                save_checkpoint(
                    run_dir / "latest.pt",
                    model=model,
                    optimizer=opt,
                    step=step,
                    best_val=best_val,
                    config=cfg.model_dump(),
                    model_kwargs=model_kwargs,
                    ema_state=ema.state_dict() if ema is not None else None,
                )
                updated, best_val = update_best(run_dir, val_stats["val_sdri_db"], best_val)
                if updated:
                    print(f"  step {step:5d}  NEW BEST val SDRi = {best_val:+.2f} dB -> best.pt")

            if step % cfg.train.ckpt_every == 0:
                ckpt_path = run_dir / f"ckpt_{step:06d}.pt"
                save_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=opt,
                    step=step,
                    best_val=best_val,
                    config=cfg.model_dump(),
                    model_kwargs=model_kwargs,
                    ema_state=ema.state_dict() if ema is not None else None,
                )
                save_checkpoint(
                    run_dir / "latest.pt",
                    model=model,
                    optimizer=opt,
                    step=step,
                    best_val=best_val,
                    config=cfg.model_dump(),
                    model_kwargs=model_kwargs,
                    ema_state=ema.state_dict() if ema is not None else None,
                )
                print(f"  step {step:5d}  saved {ckpt_path.name} + latest.pt")
        epoch += 1

    # Final save mirrors the W3.5 contract: model.pt at the end.
    save_checkpoint(
        run_dir / "model.pt",
        model=model,
        optimizer=opt,
        step=step,
        best_val=best_val,
        config=cfg.model_dump(),
        model_kwargs=model_kwargs,
        ema_state=ema.state_dict() if ema is not None else None,
    )
    save_checkpoint(
        run_dir / "latest.pt",
        model=model,
        optimizer=opt,
        step=step,
        best_val=best_val,
        config=cfg.model_dump(),
        model_kwargs=model_kwargs,
        ema_state=ema.state_dict() if ema is not None else None,
    )
    print(f"\nsaved {(run_dir / 'model.pt').relative_to(repo_root)}")
    if best_val > float("-inf"):
        print(
            f"best val SDRi = {best_val:+.2f} dB -> {(run_dir / 'best.pt').relative_to(repo_root)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
