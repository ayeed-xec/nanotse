# NanoTSE — 10-week plan

> *NanoTSE: A Named-slot Audio-visual Network for Streaming Target
> Speaker Extraction with Persistent Identity Memory.*
> Target: Interspeech 2026 (June). Fallback: TASLP, ICASSP 2027.

## Three contributions

1. **Face-voice slot-attention identity memory** — Locatello-style slot
   competition, joint (face, voice) keys, GRU-EMA updates, LRU eviction,
   persistent across the streaming session.
2. **Streaming AV-TSE backbone, sub-60 ms** — dual-cache cross-modal
   fusion (visual KV @ 25 Hz, audio queries @ 100 Hz). Mamba-2 on CUDA,
   chunk-attention on MPS/edge; single architecture targets both.
3. **IBA benchmark + multi-session test** — Hungarian-matched
   cross-session identity accuracy; "Alice-leaves-returns" test set.

## Hard constraints

- M3 Pro Mac (MPS) for all local dev; Mamba-2 has no MPS kernel — local
  backbone is chunk-attention.
- Cloud: ~1×A100 80 GB on vast.ai (~$1.50/h spot); total budget ~$700–900.
- 16 kHz audio, 25 fps video throughout.
- Disjoint speakers in test vs train (regression test present day 1).
- Test-first with xfail; no silent train regressions.

## M3-first / A100-last loop

```
write test (xfail) → M3 smoke (≤10 min, ≤200 clips, batch 4, 500 steps)
                  → metric clear?
                  → yes: commit + push
                  → no: stop, debug; do NOT escalate to A100
```

A100 only when (a) M3 smoke shows the arch can learn the task; (b) diff
vs last A100 run is well-defined; (c) expected gain ≥ 0.5 dB SI-SDRi or
≥ 5 % IBA.

## Weeks

### W1 — repo bootstrap (M3 only)
- [x] `pyproject.toml`, Makefile, pre-commit, GitHub Actions CI.
- [x] Package skeleton (`nanotse/{data,models,...}`).
- [x] Pydantic config schema + `configs/{smoke,a100}.yaml`.
- [x] First passing tests: `test_import.py`, `test_config.py`.
- [x] `scripts/data_prep/fetch_voxceleb2_mix_smoke.py` — streams
      `audio_clean_part_aa` from HF, disjoint train/val speakers,
      writes `data/smoke/{audio,manifest.json}`. Stdlib only
      (urllib + tarfile); ~30–80 MB read for the default 40 speakers.
- [x] SI-SNR loss (`nanotse/losses/si_snr.py`) + 6 behavioural tests.
- [x] `SyntheticAVMixDataset` (deterministic AV-mix fallback) + 6 tests.
- [x] First plumbing test: 1-conv model trains on 4 synthetic clips
      without NaNs, parameters update, loss decreases.
- [x] JSONL `Tracker` (`nanotse/utils/tracker.py`) + 2 tests.
- [x] Wired `scripts/train.py`: dispatch on `cfg.model.name`,
      MPS/CUDA/CPU fallback, JSONL metrics, checkpoint to `runs/<ts>/`.
- [x] End-to-end M3 smoke train completes: TDSE 70k params on MPS,
      500 steps in ~25 s, baseline 6.0 dB → SI-SNR climbs −6.6 → +5.3 dB
      (synthetic gaussian — real-speech dB lands in W2 once data is fetched).
- **Gate:** `make test lint type` green; smoke loop runs end-to-end. ✓

### W2 — TSE baseline on M3
- [x] Conv-TasNet-lite `TDSEBaseline` (`nanotse/models/baselines/tdse.py`),
      ~70 k params, encoder → bottleneck → dilated TCN × 4 → mask → decoder.
- [x] Forward shape + 8-clip overfit plumbing test (synthetic gaussian:
      loss decreases ≥ 1 dB; the +10 dB bar applies to real speech).
- [x] **W2.1** `AudioFrontend` — Conv1D encoder, 16 kHz → 100 Hz, clean
      round-trip with TSEHead (`(kernel-stride)/2` padding).
- [x] **W2.2** `ChunkAttnBackbone` — causal self-attention with rolling
      KV cache. CPU / MPS / CUDA compatible. forward_chunk ≡ forward
      within 1e-5; chunked-equivalence tested.
- [x] **W2.3** `TSEHead` — mask projection + ConvTranspose1d decoder,
      shape-exact inverse of `AudioFrontend`.
- [x] **W2.4** `NanoTSE` audio-only assembly wired through `scripts/train.py`
      (`model.name: nanotse`). Bench on MPS / CPU shows ample headroom
      (see `docs/CHANGELOG.md`).
- [ ] **W2.5** Real `VoxCeleb2MixDataset` (depends on user running fetch
      script on the 3060 box). STFT branch lands alongside.
- [ ] Real-speech 8-clip overfit gate: NanoTSE ≥ +10 dB SI-SDRi.
- **Gates:** M3 overfit 8 clips ≥ +10 dB SI-SDRi (real speech); smoke ≥ +1 dB.

### W3 — MeMo baseline + NanoTSE slot memory (M3)
- [x] **W3.1** `VisualFrontend` — per-frame CNN encoder, 25 fps video
      `(B, F, H, W, 3) uint8` → `(B, F, 512)` features. AV-HuBERT-frozen
      lands later; mouth-ROI CNN unblocks W3.2–W3.5 now.
- [x] **W3.2** `DualCacheFusion` — cross-attention with rolling visual KV
      cache. CPU/MPS/CUDA. Streaming-equivalence tests (oneshot vs
      progressive-visual chunks) verify causality on visual stream.
- [x] **W3.3** `NamedSlotMemory` — Locatello slot competition (softmax
      over slots) + GRU update + MLP residual. Slot bank persists across
      `forward_chunk`. **Contribution 1 lives here.**
- [x] **W3.4** `ASDHead` — per-slot active-speaker logits over time.
- [x] **W3.5** Full `NanoTSE` assembly. `with_visual` constructor flag
      switches between audio-only (W2.4 path) and full AV (W3.5 path).
      `forward(audio, video=None) -> (tse_out, asd_logits | None)`.
- [ ] MeMo reimpl per `av-listen/docs/MEMO_REIMPL_PLAN.md`:
      SpeakerBank, ContextBank, MeMoWrapper.forward_{chunk,offline}.
      Deferred to 3060/real-data — needs validation against paper's 9.85 dB.
- [x] Loss library: `slot_infonce`, `asd_bce`, `slot_consistency` written
      in `nanotse/losses/` with tests. Not yet wired into training (synthetic
      data has no real speaker identity for InfoNCE / no GT for ASD).
- [x] LRU eviction in `NamedSlotMemory.evict_lru()` — slot LRU stamps
      updated on every `forward_chunk` (winner = argmax of summed attention).
- **Gate:** MeMo ≥ TDSE+1 dB; NanoTSE ≥ MeMo+0.5 dB on M3 smoke.

### W4 — first A100 burst
- [ ] Rent A100, train on full VoxCeleb2-mix.
- [ ] Reproduce MeMo(TDSE) ≥ 9.85 dB SI-SNR on Impaired-Visual
      (paper: 10.34).
- [ ] Train NanoTSE backbone.
- **Gate:** within 1.5 dB of AV-MossFormer2-TSE-16K teacher on SI-SDRi.

### W5–6 — NanoTSE ablations
- [ ] Full NanoTSE on full VoxCeleb2-mix.
- [ ] Ablations: slot-mem vs MeMo banks; face channel; cross-session
      loss; N ∈ {1, 4, 16, 32}.
- [x] `IBA` metric (`nanotse/eval/iba.py`): Hungarian-matched
      slot↔speaker accuracy. `iba_score` (single sequence) and
      `iba_multi_session` (concatenated sessions) with 10 tests covering
      perfect alignment, permutation invariance, Alice-leaves-returns,
      random-baseline, shape guards. Real-data multi-session test set
      curation is W7-8.
- **Gate:** NanoTSE beats MeMo reimpl on IBA by ≥ 10 %.

### W7–8 — benchmarks + paper draft
- [ ] Multi-session test set ≥ 50 speakers × ≥ 3 non-contiguous clips.
- [ ] Full eval matrix: clean / impaired (missing, conceal, low-res,
      noise) × N ∈ {1, 4, 16}.
- [ ] Latency bench: M3 Pro, i7 CPU, RPi 5.
- [ ] Cross-dataset: FaceStar, AVA-AVD, MISP 2025.
- [ ] LaTeX draft v1.

### W9–10 — polish + submit
- [ ] Ablation table, qualitative wav samples, demo video.
- [ ] Reviewer-question rehearsal (`docs/novelty_audit.md`).
- [ ] Submit.

## Acceptance bars

| Milestone | Bar |
|---|---|
| W1 skeleton green | `make test lint type` passes; smoke dataloader yields 200 items |
| W2 TSE baseline | M3 overfit 8 clips ≥ +10 dB SI-SDRi; smoke ≥ +1 dB |
| W3 MeMo + NanoTSE | MeMo ≥ TDSE+1 dB; NanoTSE ≥ MeMo+0.5 dB (M3) |
| W4 A100 baseline | MeMo reproduction ≥ 9.85 dB SI-SNR (paper: 10.34) |
| W5–6 NanoTSE numbers | NanoTSE beats MeMo on IBA by ≥ 10 % |
| W7–8 paper draft | full eval matrix complete; LaTeX draft v1 |
| W9–10 submit | reviewer-question rehearsal done; paper submitted |
