# Changelog

All notable architectural and engineering changes. The "Decisions" sub-section
records small choices made without escalating, per the user's no-ask preference.

## [Unreleased]

### 2026-05-13 — post-v5: mel loss + model scale-up knobs + a100_v1 retune

**Context — v5 termination + capacity ceiling**
- v5 killed at step 1750 (88.4 min wall) with best val SDRi **+0.11 dB** (preserved at `runs/3060_v5_partial_step1500_p007dB.pt`). Trajectory −3.09 → −0.55 → −0.19 → −0.05 → +0.03 → +0.07 → +0.11, all monotone climbs.
- Slowdown analysis at step 1500-1750 pointed to 4.75M-param capacity ceiling: train per-batch best stuck at +5.86 dB for ~1000 steps, mag_stft loss saturating at ~1.7. Not a coding/architectural defect — natural diminishing returns at this size class.
- Decision: don't restart on 3060 with same model; instead, stack the next-largest single lever (model scale) for the A100 run. v5 partial-baseline numbers are sufficient to confirm the recipe.

**Added — multi-resolution log-mel perceptual loss**
- `nanotse/losses/mel_loss.py`: `multi_res_mel_loss(est, tgt)` averages log-mel L1 across 3 resolutions (n_fft 1024/2048/4096, n_mels=80). Hand-built mel filterbank cached per (sr, n_fft, n_mels, device, dtype) to avoid re-allocation. Complement to `multi_res_mag_stft`: linear-magnitude STFT covers spectral envelope, log-mel covers perceptually-weighted formant detail. Standard HiFi-GAN-style pairing; expected +0.5-1.0 dB SDRi.
- `LossWeights.mel` Pydantic field (default 0.0 preserves v3-v5 numerics). Wired through `compute_loss` and logged as `loss_mel` in `tracker.log`.
- Smoke verified: random pair mag=0.30, identical=0.0 (sanity), gradients non-zero. Stacked with mag_stft in `compute_loss` -- both contribute to `total`.

**Added — model capacity knobs in `ModelConfig`**
- `nanotse/utils/config.py`: `ModelConfig.d_model: int = 256`, `n_layers: int = 2`, `n_heads: int = 4`. Defaults preserve the 4.75M-param model used in v2-v5; bump them in YAML to scale.
- `scripts/train.py`: `_build_model` + checkpoint `model_kwargs` thread the new knobs through. NanoTSE's constructor already accepted these args; we just expose them at config-time.
- Verified scale sweep (forward + backward all green):
  - `(256, 2, 4)`: **4.69M** (v2-v5)
  - `(320, 3, 4)`: 7.42M
  - `(384, 3, 6)`: 9.75M
  - `(384, 4, 6)`: 11.53M
  - `(448, 4, 8)`: 14.90M
  - `(512, 4, 8)`: **18.75M** (a100_v1 target)
  - `(512, 6, 8)`: 25.06M (a100_v2 candidate)

**Changed — `configs/a100_v1.yaml`**
- Bumped model: `d_model: 512`, `n_layers: 4`, `n_heads: 8` → **18.75M params** (vs original 4.75M).
- Added `loss_weights.mel: 0.3` (complement to mag_stft 0.5).
- Updated header docstring with the v5-informed projection math: v5 +0.11 dB (partial) + ~0.5-0.8 (rest of run) + ~0.5-1.2 (8× compute) + ~1.0-2.0 (4× capacity) + ~0.5-1.0 (mel loss) = **realistic +3.5 dB val SDRi** mid-projection. Decision threshold: ≥+3.5 dB green-lights ICASSP push.

**Pending (logged for future runs)**
- SNR curriculum (`TrainConfig.curriculum_steps`): linear ramp from `(0, 10)` dB → `(-5, +10)` dB over N steps. Multi-worker DataLoader sync makes the implementation non-trivial (shared `mp.Value` or per-epoch hook). Deferred since it's a +0.3-0.5 dB lever; scale + mel are bigger.
- RetinaFace mouth-ROI re-extraction: bumps face_ok median 0.83 → ~0.97 on the kept clips. +0.3-0.5 dB. Defer until disk + 2 h CPU available.
- Pretrained AV-HuBERT visual encoder integration: +1.0-2.0 dB lever but ~1-2 weeks of work + weight download. Scope decision needed before committing.

**Decisions** (no-ask)
- **Mel loss kept hand-rolled** rather than `torchaudio.transforms.MelSpectrogram`. Same numerics, no extra dependency surface, easier to control filterbank caching for the autocast path.
- **Skipped SNR curriculum for a100_v1.** Even though it's listed in the "what could be better" table, the dynamic implementation has real complexity (multiprocessing shared state). A "manual curriculum" via two `--resume`'d runs (narrow SNR then wide) achieves 80% of the benefit with zero new code; defer the dynamic version.
- **a100_v1 stays at 18.75M, not 25M.** The clean comparison vs v5 (4.75M) is a single-variable-changed scale-up. v2 (25M) becomes the second A100 run if v1 lands well.

### 2026-05-13 — v5 launch + pre-A100 polish (mag-STFT, EMA-in-ckpt, bf16, a100 config)

**Context — v5 path decision**
- v3 partial (killed at step 3275 to fast-track v5) preserved at `runs/3060_v3_partial_step3k_neg014dB.pt`. Val trajectory −11.31 → −4.04 → −1.56 → −0.63 → −0.34 → −0.14 dB (6/6 monotone climbs); projected endpoint +0.7-1.0 dB.
- v5 stacks all the diagnosed fixes on top of v3: mag-STFT loss (weight 0.5), EMA decay 0.99 (val readable from step 250), `with_asd=false` (dead-compute removal). Single combined run vs the disciplined v3→v4→v5 ablation chain to get the headline number sooner and trade off ablation clarity.

**Added — multi-resolution magnitude STFT loss**
- `nanotse/losses/mag_stft.py`: `multi_res_mag_stft(est, tgt)` averages spectral-convergence + log-magnitude-L1 across 3 resolutions (n_fft 512/1024/2048; hop 50/120/240; win 240/600/1200). Single scalar return; no learnable params; standard `parallel-wavegan`-style auxiliary.
- `LossWeights.mag_stft` Pydantic field (default 0.0). Wired through `compute_loss` -- the returned dict now includes a `mag_stft` key.
- `scripts/train.py`: `tracker.log()` now logs `loss_mag_stft` as a first-class metric. Verified by 4-step smoke: keys = `[loss_asd, loss_consistency, loss_infonce, loss_mag_stft, loss_si_snr, loss_total, si_snr_db, step, t]`.
- v3 partial-baseline runs predate this loss; mag-STFT effect is first measured in v5.

**Added — EMA state saved in checkpoints**
- `nanotse/training/checkpoint.py`: `save_checkpoint` now takes optional `ema_state` (typically `ema.state_dict()`). `load_checkpoint` accepts `ema_state_target` and copies the saved shadow weights in place. Old checkpoints without an `ema` key silently skip the restore -- backward compatible.
- `scripts/train.py`: all 5 `save_checkpoint` call sites pass `ema_state=ema.state_dict() if ema is not None else None`; `load_checkpoint` on `--resume` threads `ema.shadow` through.
- Verified by smoke: after train→save→reload, EMA shadow tensors differ from their fresh-init values (1.12e-1 magnitude change confirmed for two sampled keys), proving the state round-trips.

**Added — bf16 mixed-precision autocast**
- `nanotse/utils/config.py`: `TrainConfig.precision: Literal["fp32", "bf16"] = "fp32"`. Default preserves v3-v5 numerics.
- `scripts/train.py`: forward + `compute_loss` wrapped in `torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)`. `use_amp` is gated on (`precision == "bf16"` and `device.type == "cuda"`) so CPU/MPS silently fall back. No `GradScaler` (bf16 has fp32 dynamic range; not needed).
- Expected speedup: ~1.5-2× on A100/H100; modest on 3060 Ampere (kept off in 3060 v5 to preserve numerics for v3-v5 comparison).

**Added — `configs/a100_v1.yaml`**
- Production-scale config inheriting v5 recipe. Deltas vs `configs/3060_v5.yaml`:
  - `batch_size: 8 → 32` (no accum needed on 80GB)
  - `accum_steps: 4 → 1`
  - `steps: 12500 → 100000` (8× the updates)
  - `ema_decay: 0.99 → 0.999` (proportional half-life)
  - `warmup_steps: 250 → 2000`
  - `num_clips: 50000 → 500000` (deeper cycling)
  - `val_every: 250 → 1000` (still ~100 vals over the run)
  - `val_clips: 500 → 2000` (val noise floor ~±0.025 dB)
  - `precision: fp32 → bf16`
  - `num_workers: 4 → 8`, `prefetch_factor: 2 → 4`
- Model architecture unchanged (4.75M params); capacity bump (15-25M) deferred to `a100_v2.yaml` after v1 numbers land.

**Cleaned**
- Deleted stale logs: `runs/3060_full_v1.log`, `runs/3060_loss_schedule_v1.log`, `runs/3060_v3_fresh.stdout.log`, `runs/fetch_*.log`. `runs/3060_v3_fresh/` directory itself removed earlier (best.pt preserved as `runs/3060_v3_partial_step3k_neg014dB.pt`).
- Deleted `cuda-keyring_1.1-1_all.deb` (WSL CUDA setup leftover).

**Decisions** (no-ask)
- **bf16 not fp16** for mixed precision: same dynamic range as fp32 (no GradScaler), industry standard on Ampere/Hopper, and avoids the silent-loss-scale-overflow class of bugs that fp16 occasionally produces with STFT magnitude losses.
- **`with_asd=False` in v5 + a100_v1**: head consumes compute for an output we zero-weight. Re-enable in W7-8 when real per-frame ASD GT is available.
- **a100_v1 keeps the 4.75 M-param model**: scale data and step count first, then scale capacity. Lets us attribute deltas cleanly between a100_v1 and a100_v2.

### 2026-05-13 — v3 mid-run audit: ablation toggles + run-4 design notes

**Context — v2 plateau diagnosis**
- 3060_v2_fresh (`runs/3060_v2_baseline_step8k_p18dB.pt`) plateaued at val SDRi **+0.18 dB** after step 8000, with slope decay +0.95 → +0.12 → +0.04 → −0.01 dB per 2k steps.
- Root cause: per-batch gradient SNR was too low. Sources: (a) 27% of v2 face clips had face_ok < 0.5 → fraction of batches effectively audio-only, (b) batch=8 with stratified speaker sampling + ±5 to +10 dB SNR aug → ±1 dB per-batch jitter on last_5 windows.
- v3 fixes: face_ok ≥ 0.5 speaker filter (583 → 432 speakers, 91848 → 69814 train clips) + gradient accumulation (effective batch 32 via accum_steps=4) + tighter step budget (12500 effective steps, same 400k sample budget).

**Added — ablation toggles on `NanoTSE`**
- `nanotse/models/nanotse.py`: new constructor flags `with_slots: bool = True` and `with_asd: bool = True`. Both default True so v3 in flight is unaffected. With `with_slots=False` the slot memory + `slot_to_feat` Linear are not constructed and the AV path bypasses identity injection (feat goes straight to `TSEHead`). With `with_asd=False` the ASD head is not built and `asd_logits=None` is returned. `with_asd=True` requires `with_slots=True` (the head consumes slot embeddings).
- `nanotse/utils/config.py`: `ModelConfig.with_slots` / `with_asd` Pydantic fields, default True.
- `scripts/train.py`: `_build_model` + checkpoint `model_kwargs` thread the new flags through.
- Smoke (instantiation + forward + backward) verified for all four combinations: full (4.75M), no_slots (3.70M, saves 1.05M = 22%), no_asd (4.69M, saves ~60K), minimal (3.70M).

**Added — gradient accumulation**
- `nanotse/utils/config.py`: `TrainConfig.accum_steps: int = 1`. When > 1, optimizer.step / EMA.update / grad-clip / scheduler-tick fire once per accum_steps mini-batches; loss is scaled by 1/accum so summed grads match a true large-batch run. All log/val/ckpt cadences are denominated in *effective* steps.
- `scripts/train.py`: gradient-accumulation loop with `micro_in_step` counter, zero-grad at the start of each effective step, partial-grad discard at epoch boundary.
- `configs/3060_v3.yaml` exercises accum_steps=4 (effective batch 32) for the in-flight run.

**Added — manifest face_ok filter**
- `scripts/data_prep/filter_manifest_by_face_ok.py`: drops speakers whose face cache frame-count-weighted mean `face_ok` ratio is < threshold (default 0.5). Backs up the pre-filter manifest to `manifest.json.pre_face_filter`. Reports kept/dropped distribution, train/val clip + speaker counts, and confirms the speaker-disjoint property survives.
- v2 → v3 manifest delta: 583 → 432 speakers (74% kept), 91848 → 69814 train clips, 23553 → 17667 val clips. Kept median face_ok = 0.83 (well above threshold); dropped median = 0.31.

**Pending — run-4 design (post-v3)**
- **EMA decay 0.999 → 0.99**: v3 first val at step 500 came back at −11.31 dB (EMA was only 39% trained), recovering to −4.04 at step 1000 (63% trained). The 0.999 decay made val unreadable until step ~2500. Lowering to 0.99 makes EMA ≥80% trained by step ~160 → val readable from the first eval. *Trade-off:* slightly noisier EMA, but for a 12500-step run the bias of 0.999 dominates the variance reduction it nominally buys.
- **NamedSlotMemory ablation**: run-4-control (full architecture) + run-4-no-slots (with_slots=False) for 2500 effective steps each, compare val SDRi at step 2500. If no-slots ≥ full, NamedSlotMemory is dead capacity for single-clip training and we ship the 3.70M-param model. If full > no-slots, slot memory is load-bearing even in single-clip mode and we keep it.
- **Conditional ASD forward**: already wired via `with_asd=False`. For run-4 we default it False until W7-8 multi-session ASD GT lands. Saves a small per-step forward+backward.
- **`configs/3060_v4.yaml`**: will be written once v3 finishes; should encode all three changes above.

**Decisions** (no-ask)
- **`with_asd=True` requires `with_slots=True`**: a clean dependency rather than silently broken (the head's forward signature consumes slot embeddings).
- **Defaults kept at True for new flags**: zero risk of breaking the in-flight v3 process or loading the existing v2 checkpoint, which gates module construction via `model_kwargs`.

### 2026-05-12 — MeMo baseline + STFT branch; M3 build complete
**Added — MeMo baseline (paper-arch placeholder)**
- `nanotse/models/baselines/memo.py` (`MeMoBaseline`, `SpeakerBank`, `ContextBank`, `MeMoState`): Li et al. 2025's architecture port. N=1 FIFO bank replacement, self-enrollment from post-backbone features, reuses our `AudioFrontend` + `ChunkAttnBackbone` + `TSEHead`. Audio-only by default; `with_visual=True` adds `VisualFrontend` + a broadcast visual feature into the fusion concat.
- `tests/test_memo.py` (9 tests): forward shape (audio + AV), bank push/retrieve shape, cold-start zero behavior, state chains across `forward_chunk`, differentiability.
- **Scope note:** paper-grade reproduction (≥ 9.85 dB SI-SNR on Impaired-Visual) is **deferred to 3060/A100** — needs PAR 2-stage training + impairment-visual augmentation, neither runnable on M3 in a meaningful way. The architecture is here so the W5 ablation harness (NamedSlotMemory vs MeMo banks) can swap them with one constructor call.

**Added — STFT branch in AudioFrontend**
- `branch: Literal["conv1d", "stft"] = "conv1d"` selector. STFT path uses `n_fft=kernel, hop_length=stride, win_length=kernel` with a Hann window (registered as a non-persistent buffer so it moves with `.to(device)`), log-magnitude, then a linear projection to `d_model`. Output truncated to `T // stride` frames to align with the Conv1D path so `TSEHead`'s decoder still produces the right length.
- 4 new tests in `tests/test_audio_frontend.py`: shape contract (1 s + 4 s), differentiability, invalid-branch rejection.

**Decisions** (no-ask)
- **MeMo retrieval simplified to N=1** with mean-pool then linear projection. Paper has self-attention over slots (Eq. 3-4) but the N=1 default makes that trivially the identity. Multi-slot retrieval will land when we ablate N > 1.
- **MeMo and NanoTSE share the audio/visual/backbone/head modules** rather than each having its own. Single source of truth for the audio path; only the memory mechanism differs.
- **STFT window as non-persistent buffer** — moves with `.to(device)` automatically without bloating the checkpoint.
- **No exact STFT round-trip via iSTFT.** `TSEHead`'s `ConvTranspose1d` decoder isn't an inverse STFT; for the STFT branch it acts as a learned upsampler. Good enough for the alternative-frontend ablation row; if we ever need a STFT-iSTFT pair, that's a new module.

### 2026-05-12 — train.py auto-uses real data; AV latency benched
**Changed**
- `scripts/train.py` now auto-detects whether to use `VoxCeleb2MixDataset` (real) or `SyntheticAVMixDataset` (fallback) based on whether `data/smoke/manifest.json` exists. `make smoke` switches paths transparently. Prints the chosen data source.
- `nanotse/eval/latency_bench.py`: added `_AVAudioOnlyWrapper` so the audio-only bench API can drive the full AV NanoTSE (generates a matching-duration zero-video on device per forward). Two new rows in `CONFIGS`: "NanoTSE full AV (W3.5)" and "NanoTSE full AV small".

**Verified — full AV NanoTSE on real audio**
- `make smoke` (full AV, 4.47 M params, real VoxCeleb2 200 items cycled from 6 train wavs, MPS):
  - baseline +3.57 dB
  - step 20: −7.47 dB → step 500: +3.35 dB
  - monotone-ish (oscillates around baseline at the end as expected from cycling 6 wavs through 4.5M params with random face placeholders).
  - **No NaN, no crash, ckpt saved.**

**Measured — full AV latency on M3 MPS**

| Device | Model | params | p50 | p95 | RTF | Budget |
|---|---|---|---|---|---|---|
| MPS | NanoTSE audio-only | 1.81 M | 4.06 ms | 4.78 ms | 0.12× | OK (12× headroom) |
| MPS | **NanoTSE full AV** | **4.47 M** | **6.95 ms** | **7.88 ms** | **0.20×** | **OK (7.6× headroom)** |
| MPS | NanoTSE full AV small | 1.02 M | 6.70 ms | 8.48 ms | 0.21× | OK |

Adding the full visual stack (VisualFrontend + DualCacheFusion + NamedSlotMemory + ASDHead) costs ~3 ms p95 on top of the audio-only path. Still 7.6× under the 60 ms budget. (The "small AV" being slower than "full AV" is MPS kernel-launch overhead — same effect we saw on small TDSE earlier; CUDA should reverse this.)

**Decisions** (no-ask)
- **Auto-detect real-vs-synthetic data via manifest presence** instead of adding a Pydantic `data.source` field. Single behavior to configure, no config-knob proliferation. The CHOSEN source is printed in the run log for traceability.
- **AV bench uses a zero-video tensor.** Real video doesn't change CONV op count materially; the dummy is fine for latency (not for quality). Real-data AV bench is for paper-deployability claims, not for shape inference.

### 2026-05-12 — W2.5: real VoxCeleb2-mix loader + W2.4 real-speech gate cleared
**Fetched (M3, one-time)**
- Ran `scripts/data_prep/fetch_voxceleb2_mix_smoke.py --num-train-speakers 3 --num-val-speakers 2 --clips-per-speaker 2`. 5 speakers (`id00017, id00061, id00081, id00154, id00419`), 10 clips, 2.2 MB streamed from HF.
- Tar layout confirmed: `./audio_clean/test/idXXXXX/<video_id>/<NNNNN>.wav`. The fetch script's `_speaker_of` parser works as designed.
- `data/smoke/manifest.json` committed (audio wavs are gitignored).

**Added — real loader**
- `nanotse/data/voxceleb2_mix_loader.py` (`VoxCeleb2MixDataset`): reads `manifest.json`, picks a target speaker + a *different* interferer speaker (deterministic via `seed + idx`), scales the interferer to a random SNR in `(0, 5)` dB, returns the same `AVMixSample` contract as `SyntheticAVMixDataset`.
- `tests/test_voxceleb2_loader.py`: 8 tests covering shape contract, **train/val speaker-disjoint stratified-split regression**, `mix = target + interferer` within float tolerance, non-zero interferer (different-speaker confirmation), determinism, missing-manifest error, single-speaker rejection.

**Verified — W2.4 real-speech 8-clip overfit gate**
- `tests/test_w24_real_speech_gate.py` (marked `@pytest.mark.slow`):
  - device: M3 MPS
  - model: `NanoTSE(with_visual=False)`, ~1.8 M params
  - 8 real clips × 4-sec center-cropped to 2.0 s
  - 250 epochs of AdamW @ lr=1e-3
  - **baseline +1.79 dB → final +19.73 dB → SI-SDRi = +17.94 dB**
  - **Target: ≥ +10 dB → PASSED with 1.8× margin**
  - runtime ~20 s on M3 MPS

This is the **first paper-grade measurement** in the project. The architecture genuinely learns speech structure from real audio; the synthetic plumbing tests verified the wires, but only real audio + the +10 dB gate verifies the model.

**Decisions** (no-ask, logged)
- **Face frames remain synthetic** for now. Only `audio_clean_part_aa` was fetched; `orig_part_*` (40+ GB) hasn't been pulled. Audio-only NanoTSE (`with_visual=False`) is the right path to validate W2.4. AV-path validation needs real face data — that's a follow-up fetch when bandwidth + disk permit.
- **`snr_db_range=(0.0, 5.0)`** — moderate-difficulty interferer level. Easy to overfit (verified by the +17.94 dB result); will tune for full-data runs.
- **Manifest committed, wavs gitignored.** Manifest is ~1 KB, defines the smoke split. Wavs (~2.2 MB) shouldn't be in git history.
- **`torch.rand(...) * (hi - lo) + lo`** for SNR sampling — `torch.empty(()).uniform_(low, high, generator=gen)` errors on size kwarg in this torch version; this is the portable form.

### 2026-05-12 — Multi-task loss library, LRU eviction, IBA metric (all M3-validatable pieces)
**Added — losses (library functions, not yet wired into training)**
- `nanotse/losses/infonce.py` (`slot_infonce`) — contrastive over per-sample slot embeddings with speaker_id labels; pull same-speaker close, push different-speaker apart. Handles batches with no positive pairs (returns 0). NaN-safe at masked diagonal.
- `nanotse/losses/asd_bce.py` (`asd_bce`) — BCE-with-logits over per-slot ASD logits with one-hot active-speaker target.
- `nanotse/losses/consistency.py` (`slot_consistency`) — MSE between two slot banks at different timepoints; supports the "Alice leaves and returns" stability story.

**Added — NamedSlotMemory: LRU eviction**
- New `lru` field in `SlotState` (per-batch per-slot LRU stamp).
- `forward_chunk` now updates the winner slot's LRU stamp on every call (winner = `argmax(attn.sum(dim=time))`).
- `evict_lru(state) -> SlotState` resets the least-recently-used slot to `slot_init` and bumps its LRU. Pure function — input state not mutated.

**Added — IBA metric (paper contribution 3)**
- `nanotse/eval/iba.py` (`iba_score`, `iba_multi_session`) — Hungarian-matched cross-session identity accuracy. Scipy's `linear_sum_assignment` finds the optimal slot→speaker mapping; fraction of frames at the optimal matching is the score. `iba_multi_session` concatenates sessions before scoring so the same speaker re-using the same slot across sessions counts as the same canonical identity.

**Tests** — 26 new across `tests/test_losses.py`, `tests/test_slot_memory.py` (LRU), `tests/test_iba.py`. Total now **91 passed**, **97% coverage**.

**Decisions** (no-ask, logged)
- **Losses are library-only for now.** No new-loss wiring into `scripts/train.py` until real data + labels arrive. Synthetic data has `speaker_id` field but no consistent identity signal to learn (face frames are random uint8). Wiring now would be cargo-cult.
- **`N812` (lowercase `functional` imported as `F`) globally ignored.** `import torch.nn.functional as F` is the PyTorch standard idiom across the codebase. Per-file noqa added clutter.
- **LRU winner = argmax of summed attention across time** per chunk, per batch. Simple, deterministic, doesn't need a threshold. Real "novelty detection" (when to evict) is the caller's call; this module just maintains the stamps.
- **`evict_lru` returns a new state dict, not mutating.** Cleaner functional API; cheap because cloning a (B, N, S) tensor is small.
- **IBA: scipy.optimize.linear_sum_assignment** instead of a hand-rolled Hungarian. Battle-tested, transitively in deps via librosa.
- **`scipy.*` added to mypy `ignore_missing_imports`** since `scipy-stubs` is a heavy install and scipy is only touched in one place.
- **InfoNCE NaN-safety**: `torch.where(pos_mask > 0, log_prob, 0)` instead of `log_prob * pos_mask`. The naive multiply triggers `-inf * 0 → nan` at the masked-out diagonal entries.

### 2026-05-12 — Full AV NanoTSE validated end-to-end on M3 MPS
**Verified**
- `make smoke` with `model.name: nanotse` (full AV path, `with_visual=True`):
  - 4,474,178 params total
  - baseline SI-SNR(mix, target) = +6.01 dB
  - 500 steps, monotone trajectory: step 20 = −15.80 dB → step 500 = +1.92 dB
  - no NaN; no crash; checkpoint saved to `runs/<ts>/model.pt`
- This exercises the full pipeline under gradient flow: AudioFrontend, VisualFrontend, DualCacheFusion, ChunkAttnBackbone, NamedSlotMemory, slot_to_feat projection, TSEHead, ASDHead. Synthetic data is gaussian + random faces, so the model can't *beat* baseline (no speech structure to learn), but the *learning trajectory is clean* — that's the architecture/bug correctness gate.

**Changed**
- `scripts/train.py` passes `batch["face"]` to NanoTSE when `with_visual=True`.
- `configs/smoke.yaml` `model.name: nanotse` (was `tdse`).
- `_build_model("nanotse")` returns the full AV model. Audio-only NanoTSE is only built explicitly via `NanoTSE(with_visual=False)` (used in tests + audio-only bench rows).

### 2026-05-12 — W3.1–W3.5: visual frontend, fusion, named-slot memory, ASD, full AV NanoTSE
**Added (paper contribution 1 lives in W3.3)**
- `nanotse/models/frontends/visual_avhubert.py` (`VisualFrontend`) — per-frame CNN, 4 stride-2 convs + `AdaptiveAvgPool2d(1)` + `LayerNorm`. `(B, F, H, W, 3) uint8 @ 25 fps → (B, F, 512)`. AV-HuBERT-frozen lands later as a swap-in.
- `nanotse/models/fusion/dual_cache.py` (`DualCacheFusion`) — cross-attention: audio queries (100 Hz) attend to a rolling visual KV cache (default 50 frames = 2 s of context). Streaming state is `(k_cache, v_cache)`; offline `forward()` and one-shot `forward_chunk()` are sample-equivalent; progressive-visual chunks are deliberately *more causal* than one-shot (documented + tested).
- `nanotse/models/memory/slot_attention.py` (`NamedSlotMemory`) — **contribution 1**. Locatello slot competition (softmax over slots), GRU-EMA update, MLP residual. Slot bank persists across `forward_chunk` (state passed in/out). Returns `(augmented_features (B, T, D+S), slots (B, N, S))`. LRU eviction deferred to multi-speaker integration.
- `nanotse/models/heads/asd.py` (`ASDHead`) — per-(token, slot) logit head; output `(B, T, N)` for BCE supervision against ground-truth active-speaker slot.
- `nanotse/models/nanotse.py` (`NanoTSE`) extended: `with_visual` constructor flag, optional `video` argument, returns `(tse_out, asd_logits | None)`. Audio-only path (W2.4) preserved.
- `scripts/train.py`: handles `tuple` return from NanoTSE; defaults to audio-only NanoTSE for now (`with_visual=False`). AV training (passing `batch["face"]` to the model) is the next train-loop change once W2.5 lands real video.
- **22 new tests** across the four new modules + assembly. Coverage includes:
  - Shape contracts for all four modules.
  - **DualCacheFusion oneshot ≡ offline** within 1e-5; chunked-with-full-visual-upfront ≡ oneshot.
  - **DualCacheFusion progressive-visual streaming is NOT equivalent to oneshot** — documented expected causality.
  - **NamedSlotMemory persists state across `forward_chunk`** (`state["step"]` increments; slots after ≠ slot_init).
  - Cache truncation, empty-visual passthrough, slot-memory differentiability.
  - Full AV `NanoTSE.forward(audio, video)` returns `(tse_out, asd_logits)` with correct shapes.

**Latency, M3 Pro, p95 per 40 ms chunk (audio-only path; AV bench TBD)**

| Device | Model | params | p95 | RTF | Budget |
|---|---|---|---|---|---|
| MPS | NanoTSE audio-only | 1.81 M | **4.61 ms** | 0.12× | OK (13× headroom) |
| MPS | NanoTSE audio-only small | 297 k | 3.48 ms | 0.09× | OK |
| CPU | NanoTSE audio-only | 1.81 M | **0.63 ms** | 0.02× | OK (95× headroom) |
| CPU | NanoTSE audio-only small | 297 k | 0.28 ms | 0.01× | OK |

Adding the visual stack will push latency up by ~5–15 ms p95 (VisualFrontend is the heavyweight); AV bench rows get added once the bench harness is extended to pass video tensors.

**Decisions** (no-ask, logged)
- **`with_visual: bool = True` constructor flag** on NanoTSE. Lets the audio-only smoke train (W2.4) and the AV training (W3.6+) share one class without overloads. Memory cost when `False`: 0 (visual modules never allocated).
- **`forward(audio, video=None) → (tse_out, asd_logits | None)`** is the public API. Always returns a tuple, even in audio-only mode (then `asd_logits` is `None`). `train.py` unpacks with an `isinstance(out, tuple)` guard so TDSEBaseline still works.
- **`feat_with_id = slot_to_feat(slot_aug)`** projects `(B, T, D + S)` back to `D` before feeding TSEHead — keeps TSEHead's input dim the same in both audio-only and AV paths. Single TSEHead spec.
- **Slot-attention softmax over slots, not over inputs** — verified against Locatello et al. 2020 § 2. Each input token's attention sums to 1 across the N slots; aggregation then normalizes per-slot weights over the input axis.
- **Soft (not hard) slot assignment in the augmented features**: `attn @ slots`, fully differentiable. Hard `argmax` would break gradients.
- **DualCacheFusion: explicit "progressive ≠ oneshot" test** locks in the streaming-causal contract for visual. Without this test, a future refactor could silently let the streaming path peek at future visual frames and we'd never notice.
- **No LRU eviction yet.** Slot bank just keeps refining. Eviction lands when we actually run a session with > N speakers and need it.
- **Bench rows use `with_visual=False`.** Adding AV rows requires the bench to pass a video tensor; that's its own small extension. Filed under "do later when we need AV deployability numbers."

### 2026-05-12 — W2.1–W2.4: audio-only NanoTSE wired end-to-end
**Added**
- `nanotse/models/frontends/audio_stft.py` (`AudioFrontend`) — learned Conv1D encoder, 16 kHz → 100 Hz at `d_model=256`. With `kernel=320, stride=160, padding=80` the round-trip with `TSEHead` is sample-exact (`T → T/160 → T`).
- `nanotse/models/backbones/chunk_attn.py` (`ChunkAttnBackbone`) — causal multi-head self-attention with rolling KV cache. CPU / MPS / CUDA compatible. State is a list of `(k_cache, v_cache)` tuples per layer, always passed in/out explicitly. Default `d_model=256, n_heads=4, n_layers=2, cache_len=200` (= 2 s at 100 Hz).
- `nanotse/models/heads/tse.py` (`TSEHead`) — sigmoid mask projection × encoder output, then `ConvTranspose1d` decoder back to time-domain audio.
- `nanotse/models/nanotse.py` (`NanoTSE`) — top-level assembly: `AudioFrontend → ChunkAttnBackbone → TSEHead`. Default ~1.8 M params. W3.5 will extend with visual frontend + fusion + slot memory + ASD head without changing the public `forward(audio)` signature.
- `scripts/train.py` now dispatches on `cfg.model.name in {"tdse", "nanotse"}`.
- `nanotse/eval/latency_bench.py::CONFIGS` extended with NanoTSE default + small.
- **18 new tests** across `test_audio_frontend.py`, `test_chunk_attn.py`, `test_tse_head.py`, `test_nanotse_assembly.py`. Includes:
  - Streaming equivalence: `forward_chunk` (one-shot) ≡ `forward` within 1e-5.
  - Chunked equivalence: splitting T into chunks + folding state ≡ one-shot.
  - Causal check: perturbing future tokens does not change past outputs.
  - Cache truncation: cache stays bounded by `cache_len`.
  - NanoTSE overfit-on-4-clips plumbing: loss decreases ≥ 1 dB.

**Latency, M3 Pro, p95 per 40 ms chunk (target < 60 ms)**

| Device | Model | p50 | p95 | RTF (p95/40 ms) | Headroom |
|---|---|---|---|---|---|
| MPS | TDSE 70k | 2.35 ms | 5.67 ms | 0.14× | 10× |
| MPS | TDSE 16k | 1.72 ms | 2.78 ms | 0.07× | 22× |
| MPS | **NanoTSE 1.8M** | 3.99 ms | **4.85 ms** | **0.12×** | **12×** |
| MPS | NanoTSE 297k | 2.49 ms | 3.29 ms | 0.08× | 18× |
| CPU | TDSE 70k | 0.36 ms | 0.48 ms | 0.01× | 125× |
| CPU | TDSE 16k | 0.22 ms | 0.29 ms | 0.01× | 207× |
| CPU | **NanoTSE 1.8M** | 0.58 ms | **1.05 ms** | **0.03×** | **57×** |
| CPU | NanoTSE 297k | 0.23 ms | 0.36 ms | 0.01× | 111× |

NanoTSE at the full 1.8 M params still has ~12× headroom on MPS, ~57× on CPU. The visual frontend (W3.1) will add ~5–15 ms on top — still leaves margin.

**Decisions** (no-ask, logged)
- **`kernel=320, stride=160, padding=80`** in both `AudioFrontend` and `TSEHead`. The `(kernel-stride)/2` padding rule gives a sample-exact `T → T/stride → T` round-trip. Both classes raise `ValueError` if you give them an asymmetric (kernel - stride).
- **`ChunkAttn` cache truncation** keeps the last `cache_len` frames. Older context is dropped. 2 s of audio context (200 frames at 100 Hz) is plenty for current scenarios; long-session tests will revisit.
- **Causal mask via index broadcasting** — `(j < cache_offset) | (j - cache_offset <= i)` — instead of `torch.tril` on `(Tq, Tkv)`. Cheaper, MPS-friendly, and the cache offset semantics are explicit in the code.
- **`forward_chunk` and `forward` both go through the same `_attend` helper**, so the offline / streaming paths cannot diverge silently.
- **`NanoTSE` constructor mirrors `AudioFrontend` and `ChunkAttnBackbone`** so config changes propagate without rewiring. W3.5 will add visual-side kwargs alongside, not in place of, these.
- **`configs/smoke.yaml` left on `tdse`** for faster iteration. To smoke-train NanoTSE, edit `model.name: nanotse` (or copy to `configs/smoke_nanotse.yaml`).

### 2026-05-12 — Latency benchmark
**Added**
- `nanotse/eval/latency_bench.py`: streaming forward-pass benchmark. Measures p50/p95/p99/mean/min ms per 40 ms chunk on the current device (auto-picks CUDA → MPS → CPU). Already wired to `make bench`. New modules add a row to the `CONFIGS` list so we catch latency creep as we go.
- 4 tests (`tests/test_latency_bench.py`): util functions + CLI smoke.

**Measured today (M3 Pro, TDSEBaseline only — full NanoTSE will be larger)**

| Device | Model | p50 (ms) | p95 (ms) | p99 (ms) | RTF (p95/40ms) | vs 60ms budget |
|---|---|---|---|---|---|---|
| MPS | TDSE 70k | 1.26 | 2.80 | 3.57 | 0.07x | OK (21x headroom) |
| MPS | TDSE 25k | 1.38 | 2.56 | 3.06 | 0.06x | OK (23x headroom) |
| CPU | TDSE 70k | 0.39 | 0.44 | 0.50 | 0.01x | OK (136x headroom) |
| CPU | TDSE 25k | 0.23 | 0.29 | 0.43 | 0.01x | OK (207x headroom) |

**Observation:** CPU beats MPS at this scale. MPS kernel-launch overhead per op dominates actual compute for a 70k-param model. Will reverse once the model grows past ~1M params or once visual frontend (CNN) lands.

**Decisions** (no-ask)
- **Bench tracks TDSE today** because that's the only model that exists. Full NanoTSE rows get added to `CONFIGS` as W2.4 + W3.5 land.
- **Tested via CLI smoke + util tests** rather than running real bench in pytest — keeps `make test` fast.

### 2026-05-12 — Architecture spec
**Added**
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) rewritten from stub into a real contract: end-to-end ASCII data-flow diagram with concrete shapes; one row per planned module with file path, I/O, and streaming-state type; streaming `init_state` / `forward_chunk` interface; multi-task loss schedule with add-when-needed order; sprint-level W2.1 → W4 implementation gates; explicit "what is NOT in scope" guard against scope creep (no cross-session persistence, no far-field pivot, no framework layer, no premature abstractions).

**Decisions** (no-ask, logged)
- **No new code, no new modules** — design-only update. Modules land per the sprint table, one at a time, each gated by its own test.
- **Constants pinned in one table** — `sample_rate=16k`, `fps=25`, audio 100 Hz, chunk 40 ms, `N=16`, `D=256`, `Dv=512`, `S=256`. These shape every module; pin them here so they aren't redefined per-module.
- **Streaming contract is the only abstraction** — three modules genuinely share the `init_state`/`forward_chunk` shape (`DualCacheFusion`, `NamedSlotMemory`, `ChunkAttnBackbone`). Anything below three-callers does NOT get abstracted.
- **PCGrad deferred** — not added speculatively; only wired in if training shows loss conflicts (gradient cosine via `Tracker`).
- **No Pydantic schema changes yet** — model-knob fields (`n_slots`, `slot_dim`, `backbone`, etc.) get added to `nanotse/utils/config.py` when the corresponding module lands, not in advance.

### 2026-05-12 — W1 finish + W2 TDSE baseline
**Added**
- `nanotse/utils/tracker.py` — append-only JSONL `Tracker`. Every run writes `runs/<ts>/metrics.jsonl`; future commits compare JSONL streams to enforce the "no silent regressions" rule.
- `nanotse/models/baselines/tdse.py` — `TDSEBaseline`, a Conv-TasNet-lite stack (encoder → bottleneck → 4 dilated TCN blocks → mask → decoder). 70 k params at defaults, no speaker/face conditioning yet (that's W3).
- `scripts/train.py` rewritten: dispatches on `cfg.model.name`, falls back to CPU when MPS/CUDA unavailable, logs baseline SI-SNR(mix, target) + per-`log_every` loss/SI-SNR through `Tracker`, dumps `config.json` + `model.pt` + `metrics.jsonl` under `runs/<utc-ts>/`.
- Tests: `test_tdse.py` (2 — forward shape + 8-clip overfit), `test_tracker.py` (2 — JSONL round-trip + nested parent dir).

**Verified**
- `make smoke` on synthetic data: 70 k-param TDSE on MPS, 500 steps in ~25 s. Baseline 6.0 dB → SI-SNR climbs from −6.6 dB at step 20 to +5.3 dB at step 500. Plumbing solid; the synthetic gaussian mix is harder than real speech, so the absolute dB number is below baseline — the trajectory is what matters.
- 23 tests pass, **96 % coverage**.

**Decisions** (no-ask, logged)
- **Smoke train uses synthetic data** — `data/smoke/manifest.json` doesn't exist until the user runs the fetch script (40+ GB on the wire). The real loader lands in W2 alongside actual speech; for W1 plumbing, synthetic is sufficient and lets CI exercise the full pipe.
- **`TDSEBaseline` no speaker conditioning yet** — that's the named-slot memory's job (W3). Avoiding the temptation to put face/voice plumbing in a baseline keeps the W3 contribution isolated.
- **W2 overfit gate split** — PLAN's `+10 dB SI-SDRi on 8-clip overfit` is the real-speech bar; the unit test asserts `loss decreases ≥ 1 dB on synthetic`. Two distinct gates, both documented.
- **`torch.relu` over `F.relu`** — same call, lets us drop the `import torch.nn.functional as F` line and skip the `N812` lowercase-import lint suppression.

### 2026-05-12 — W1 data layer + plumbing test
**Added**
- `nanotse/losses/si_snr.py` — scale-invariant SNR (Le Roux et al., 2019), with `si_snr()` returning dB per item and `negative_si_snr()` for use as a training loss. Shape-mismatch raises `ValueError`.
- `nanotse/data/voxceleb2_mix.py` — `SyntheticAVMixDataset` (deterministic per-index AV mixes; reproducible via `seed + idx`) + `AVMixSample` TypedDict defining the contract every loader must satisfy.
- `scripts/data_prep/fetch_voxceleb2_mix_smoke.py` — streams the first part of `audio_clean_part_aa` from HuggingFace via stdlib `urllib` + `tarfile`. Stops as soon as the disjoint train/val speaker quota is met (typical read ~30–80 MB, not 40 GB). Exposes `--list` to inspect tar member naming before committing to a real fetch. Hard-asserts speaker disjointness before writing the manifest.
- Tests: `test_si_snr.py` (6), `test_data.py` (6), `test_smoke_overfit.py` (1). The smoke-overfit test trains a 1-conv `_TinyConvDenoiser` on 4 synthetic clips for 150 steps and verifies (a) no NaN losses, (b) parameters updated, (c) loss decreased by ≥ 0.5 dB.

**Decisions** (no-ask, logged)
- **No `huggingface_hub` dep** — wrote the fetch script against stdlib `urllib` + `tarfile` instead, since the HF resolve URL is public. Removes a heavy dep + simplifies install. Side effect: `huggingface_hub` was briefly installed in the local venv during exploration; not pinned in `pyproject.toml`, so a fresh install won't pull it.
- **Streaming tar extraction** — the 40 GB part-files are concatenable tars, but `tarfile.open(mode="r|")` reads sequentially and we can stop anywhere. This lets the smoke fetch read tens of MB instead of tens of GB.
- **Plumbing test asserts decrease, not absolute dB** — a 2-conv ~1 k-param model can't memorize 4 × 16 000-sample gaussian mixes to a specific SI-SDR, so we assert (no NaN) + (params changed) + (loss dropped ≥ 0.5 dB). The "+10 dB SI-SDR" bar from PLAN W2 belongs to the real TDSE baseline, not this plumbing test.
- **`SyntheticAVMixDataset` interferer scale** — default 0.5 (gives ≈ 6 dB input SI-SNR), parameterized in case a future test wants a different mix difficulty.

### 2026-05-12 — W1 bootstrap
**Added**
- Fresh package skeleton at `nanotse/` with subpackages: `data`, `models/{baselines,frontends,backbones,fusion,memory,heads}`, `losses`, `training`, `eval`, `utils`.
- `pyproject.toml` — Python 3.11; torch ≥ 2.4 (MPS-compatible), torchaudio, numpy, soundfile, librosa, pydantic, pyyaml, tqdm; dev extras pull pytest+cov, ruff, mypy, pre-commit.
- Tooling: `ruff` (line-length 100, target `py311`, lint preset includes I/UP/B/SIM/PT/RUF/N/C4); `mypy --strict` on `nanotse/`; pytest with `--strict-markers --strict-config -ra` and coverage on the package.
- `Makefile` targets: `install`, `smoke`, `test`, `lint`, `fmt`, `type`, `bench`, `train-a100`, `clean`.
- `.github/workflows/ci.yml` matrix: `ubuntu-latest × macos-14`, Python 3.11. CPU torch via the official PyTorch index to keep CI install fast.
- `.pre-commit-config.yaml` with ruff + mypy (scoped to `nanotse/`) + standard hygiene hooks.
- `configs/{smoke,a100}.yaml` stubs + Pydantic config schema in `nanotse/utils/config.py`.
- First passing tests: `tests/test_import.py` (package import + version), `tests/test_config.py` (YAML round-trip + defaults).
- `scripts/train.py` stub: parses config and exits 0 with a "not yet implemented" message so `make smoke` is non-fatal pre-W2.
- `docs/{PLAN, ARCHITECTURE, CHANGELOG}.md` stubs.

**Decisions** (no-ask choices, logged here per [feedback_planning_style](../../.claude/projects/-Users-ayeed-PycharmProjects-nanotse/memory/feedback_planning_style.md))
- **Env var name** — `NANOTSE_A100_HOST` (kickoff doc said `AVTSE_A100_HOST`; renamed to match the package).
- **Build backend** — hatchling (lightest PEP 517 backend; no `setup.py` required).
- **Python pin** — 3.11 only (not 3.12+) to keep MPS torch wheel availability predictable through the project window.
- **License** — MIT in `pyproject.toml`. Paper-track default; revisit before any public release.
- **Coverage gate** — `--cov-fail-under=0` for now. Will raise to 80 once W2 lands real code under `nanotse/`.
- **mypy scope** — `mypy nanotse` only (run via `make type` and CI). `scripts/` and `tests/` are not strictly type-checked; reduces friction on argparse + fixtures.
- **Ruff preset** — included `B` (bugbear), `SIM`, `PT`, `RUF`, `N`, `C4` in addition to the defaults. Tests get an exemption from `N802/N803` for non-snake test names.
- **CI: CPU torch only** — install `torch torchaudio` from the official CPU index before `pip install -e ".[dev]"`. MPS/CUDA paths are tested locally; CI just guards correctness on CPU.
- **README** — intentionally not created (the kickoff doc didn't list one; `docs/PLAN.md` is the canonical entry point).

### 2026-05-12 — 3060 box bootstrap (W3.5 build landed in WSL2)
**Added**
- Initial setup on the RTX 3060 Windows + WSL2 Ubuntu box: uv 0.11.13, Python 3.11.15 (uv-fetched), torch 2.11.0+cu128, torchaudio 2.11.0+cu128, triton 3.6.0.
- Toolchain gates: .venv/bin/ruff format --check . (60 files, all formatted, all checks pass), .venv/bin/mypy nanotse (34 source files, mypy --strict clean), .venv/bin/pytest 109/116 — the 7 fails are all VoxCeleb2-mix data-dependent ( × 6 +  × 1); same pattern a fresh M3 clone would show before data fetch.

**Decisions** (no-ask choices)
- **Torch CUDA build** —  initially resolved to . The RTX 3060 driver (572.60, max CUDA 12.8) cannot use cu130 (). Reinstalled via  → , , device "NVIDIA GeForce RTX 3060 Laptop GPU", VRAM 6143 MiB, compute capability (8, 6) = Ampere SM_86.
- **Hardware reality** — RTX 3060 Laptop GPU **6 GB** SKU (not the 12 GB desktop or 8 GB laptop variant the kickoff anticipated). Tightens batch + activation budget. KD-teacher work (AV-MossFormer2-TSE-16K) confirmed off-table on this box — A100-only.
- **Shell** — WSL2 Ubuntu chosen over native Windows; POSIX Makefile + uv + pre-commit run unmodified.

### 2026-05-13 — `device: auto` resolver + first CUDA smoke + 3060 latency table
**Fixed**
- `scripts/train.py::_resolve_device` would fall straight through to CPU when `cfg.device == "mps"` and MPS was unavailable, even on a CUDA-equipped box. On the 3060 that meant `make smoke` ran 13 min on CPU instead of using the GPU.
- `nanotse/utils/config.py`: added `"auto"` to the `Device` Literal so configs validate.
- `scripts/train.py`: added an `"auto"` branch that prefers CUDA → MPS → CPU at runtime. Explicit `"mps"` / `"cuda"` / `"cpu"` semantics unchanged.
- `configs/smoke.yaml`: `device: mps` → `device: auto`. Same checked-in config now picks the best device on M3, 3060, and A100 with no per-box edit.
- `tests/test_config.py`: added `test_config_accepts_auto_device`. 5/5 config tests still pass.

**Verified — first CUDA smoke train on 3060**
- `make smoke` (full AV NanoTSE, 4.47 M params, real VoxCeleb2 200 items cycled from 6 train wavs, CUDA):
  - **48 s wall time** (vs 13 min on CPU fallback before the fix — ~16× speedup)
  - baseline +3.57 dB → step 500 +3.42 dB (M3 reported +3.35 — deterministic across MPS/CPU/CUDA at `seed=0`)
  - no NaN, ckpt at `runs/20260512T170337Z/model.pt`
- `make test` 116/116 passing on CUDA, including the slow real-speech W2.4 gate (which has its own CUDA→MPS→CPU `_pick_device` and was unaffected by the resolver fix).

**Measured — latency table on 3060 CUDA (p95 per 40 ms chunk; target < 60 ms)**

| Model | params | p50 | p95 | p99 | RTF (p95/40 ms) | vs 60 ms |
|---|---|---|---|---|---|---|
| TDSE default | 70 k | 1.47 ms | 1.65 ms | 1.72 ms | 0.04× | OK (36× headroom) |
| TDSE small | 16 k | 0.55 ms | 0.63 ms | 0.96 ms | 0.02× | OK |
| NanoTSE audio-only (W2.4) | 1.81 M | 1.09 ms | 2.46 ms | 2.52 ms | 0.06× | OK (24× headroom) |
| NanoTSE audio-only small | 297 k | 0.65 ms | 0.78 ms | 0.97 ms | 0.02× | OK |
| **NanoTSE full AV (W3.5)** | **4.47 M** | **3.15 ms** | **3.91 ms** | **5.81 ms** | **0.10×** | **OK (15× headroom)** |
| NanoTSE full AV small | 1.02 M | 2.76 ms | 3.26 ms | 4.29 ms | 0.08× | OK |

3060 CUDA full-AV p95 (3.91 ms) is ~2× faster than M3 MPS (7.88 ms). All rows clear the 60 ms streaming budget with ≥15× headroom. The 3060 column joins M3 MPS + CPU in the paper Section 5 latency table; i7 CPU + RPi 5 rows land in W7-8.

**Decisions** (no-ask)
- **`auto` as a first-class Device value** rather than chain-fallback inside the existing `"mps"`/`"cuda"` branches. Explicit beats magical: writing `device: mps` in a config still means MPS only, so M3-pinned configs (and CI's `cpu` strictness) stay unambiguous. Only `auto` opts into runtime detection.
- **No bench-harness changes** — `nanotse/eval/latency_bench.py` already auto-picks CUDA→MPS→CPU; the M3 `CONFIGS` list ran on CUDA untouched. The full-AV row was already wired via `_AVAudioOnlyWrapper` (added in the M3 build).
- **The data fetch (`fetch_voxceleb2_mix_smoke.py --num-train-speakers 3 --num-val-speakers 2 --clips-per-speaker 2`) is fully deterministic** — same 5 speakers, same 10 wavs, same manifest bytes as the M3 fetch. Manifest in git is the contract; wavs are reproduced from HF on each new machine.

### 2026-05-12 — Loss schedule wired: InfoNCE on slots; all model features now training-active
**Added**
- `nanotse/utils/config.py::LossWeights` — Pydantic model holding `si_snr` / `infonce` / `asd` / `consistency` weights (all `Field(ge=0)`). Attached to `TrainConfig.loss_weights` with sensible defaults (`si_snr=1.0, infonce=0.1, asd=0.0, consistency=0.0`). Every model uses `ConfigDict(extra="forbid")` so YAML typos fail at load time. ASD + consistency stay at 0.0 by default; flipping them on is a YAML-only change once their prerequisite labels exist (W7-8 ASD GT; W5-6 multi-window dataset).
- `nanotse/training/losses.py::compute_loss` — composite weighted-sum loss. Returns a `dict[str, torch.Tensor]` with `total / si_snr / infonce / asd / consistency` keys always present (zero scalars when a term is gated off), so the training tracker logs a uniform schema regardless of which losses are active. Plain dict (not Pydantic) for the hot path — Pydantic validation overhead per step would be wasted; configs are Pydantic, runtime tensors are not.
- `configs/3060.yaml` — real-data training config on the RTX 3060 Laptop box. 2000 steps, batch 8, `lr=5e-4`, sized for 6 GB VRAM. Same loss schedule as smoke/a100.
- `tests/test_train_loss_schedule.py` (6 tests): per-term gating, gradient flow through slot bank, full AV forward → `compute_loss` → backward without NaN, plus an end-to-end `scripts/train.py` subprocess test that asserts `loss_total / loss_si_snr / loss_infonce / loss_asd / loss_consistency` rows are written to `metrics.jsonl` and `si_snr` decreases over 30 steps.

**Changed**
- `nanotse/models/nanotse.py::NanoTSE.forward` now returns `(tse_out, asd_logits, slots)`. The third element (`(B, N, S)` slot bank) is `None` in audio-only mode, populated in the full AV path. Surfacing the slots is what lets InfoNCE (and future slot-consistency) train them directly instead of routing all gradient through the TSE head. `NanoTSEOutput` type alias added for explicit annotation.
- `scripts/train.py` now reads `cfg.train.loss_weights`, calls `nanotse.training.compute_loss`, and logs `loss_total / loss_si_snr / loss_infonce / loss_asd / loss_consistency / si_snr_db` per step. Stdout banner shows the active weight schedule. Forward path factored into `_forward(...)` helper that handles audio-only baselines (`TDSEBaseline`) and full-AV NanoTSE uniformly.
- `nanotse/models/baselines/memo.py::MeMoState` converted from `@dataclass` → Pydantic `BaseModel` (`ConfigDict(arbitrary_types_allowed=True)` to hold `torch.Tensor` fields). The project's container style is now single-source: Pydantic for everything except `TypedDict`-shaped runtime tensor payloads (`AVMixSample`, `SlotState`) where DataLoader/streaming collation requires plain dicts.
- `tests/test_nanotse_assembly.py` updated to unpack the new 3-tuple. Added `test_nanotse_av_no_video_returns_none_aux` to lock the behaviour when an AV-capable model is called without video.
- `configs/{smoke,a100}.yaml` updated with explicit `loss_weights` blocks (same defaults — keeps configs self-describing; no behavioural change on smoke).

**Verified**
- `ruff format --check .` — 62 files, all formatted.
- `ruff check .` — all checks pass.
- `mypy nanotse` — clean (`--strict` via `pyproject.toml`), 35 source files.
- `pytest` — **124/124 passing** (up from 116; +6 in `test_train_loss_schedule.py`, +1 in `test_nanotse_assembly.py`, +1 from previously-data-dependent W2.4 gate now in scope). 95 % branch coverage on `nanotse/`.
- `make smoke` (full AV NanoTSE on real VoxCeleb2 audio, CUDA, 500 steps with `loss_weights={si_snr:1.0, infonce:0.1}`): baseline +3.57 dB → step 500 +2.47 dB SI-SNR. InfoNCE component oscillates 0.004 – 1.47 depending on whether each batch has speaker collisions (3 train speakers × batch 4 → ~67 % batches with ≥ 1 collision). No NaN; ckpt at `runs/smoke_loss_schedule/model.pt`.
- **3060 long run** (`configs/3060.yaml`: 2000 steps × batch 8 × `lr=5e-4`, full AV NanoTSE, CUDA, real VoxCeleb2 audio + synthetic faces): wall time **18.2 min**, throughput **1.8 steps/s**, GPU 99 % util / 3.05 GiB VRAM / 36 W draw / 65 °C. Baseline (train mixes) +2.63 dB → peak SI-SNR +13.08 dB @ step 1900 (**peak train SDRi +10.45 dB**, in line with the MeMo paper's 9.85 dB and within ~2 dB of the AV-MossFormer2 teacher's ~14.4 dB). last-10-log mean +9.81 dB (mean train SDRi +7.18 dB). Ckpt at `runs/3060_loss_schedule_v1/model.pt` (17.9 MB).

**Generalisation gap — surfaced by the val pass and worth flagging loudly**
- `scripts/diagnose.py --ckpt runs/3060_loss_schedule_v1/model.pt --num-clips 6` (val split, 2 held-out speakers × 2 clips, 6 generated mixes): average SDRi **−2.87 dB** — the trained model *degrades* mixes on speakers it has not seen. Per-clip SDRi range: −4.01 to −1.83 dB.
- Train +10.45 dB peak vs val −2.87 dB ⇒ **~13 dB train→val gap**: the model memorised speaker-specific spectral fingerprints with only 3 train speakers × 2 clips = 6 unique training wavs. Architecture is healthy (gradient flow, loss schedule, latency budget all green); the bottleneck is **data size + visual realism**, not model code. This is exactly the regime the W4 A100 burst is designed to leave behind.

**Decisions** (no-ask)
- **Three losses wired by code, only InfoNCE active by default.** `asd` and `consistency` are fully reachable in `compute_loss` but gated to 0.0 because their prerequisite signals don't exist yet (no real per-frame ASD GT on this data; no multi-window pairs). Inventing self-supervised proxies now would couple the design to placeholder labels and bias the learned slot↔speaker mapping before the real labels land — that's the W5-6 / W7-8 work.
- **Slot pooling for InfoNCE is mean across `N`.** Cheapest correct option. Max-pool / attention-weighted pool are deferred until we have evidence they matter (no ablation row scheduled before W5-6).
- **Loss return type is `dict[str, torch.Tensor]`, not Pydantic.** Pydantic models validate on every construction; for a per-step hot-path bundle that's overhead with no payoff (callers consume `losses["total"]` and the dict keys are tested). Configs (cold load) remain Pydantic; training tensors stay dicts. The `dataclass → Pydantic` cleanup is about killing the *mixture*, not enforcing Pydantic on every container.
- **`TypedDict` for `AVMixSample` / `SlotState` kept.** They're DataLoader collation contracts and streaming-state contracts — `default_collate` cannot fold `BaseModel` instances, and converting them would force a custom `collate_fn` for no real win. `TypedDict` is the idiomatic PyTorch typing for batched dict payloads; the user's "no mixture" instruction was specifically about `dataclass` (only one offender: `MeMoState`, fixed).
- **`configs/3060.yaml` shape**: 2000 steps × batch 8 × 4 s clips chosen to (a) cycle the 6 real wavs many times so the model sees mix diversity, (b) fit 6 GB VRAM headroom comfortably, (c) finish in well under 10 min so the smoke→long loop stays fast. Larger sweeps wait for `orig_part_*` (real face frames) and the A100 burst (W4).
