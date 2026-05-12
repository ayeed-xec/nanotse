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
- [ ] `scripts/data_prep/fetch_voxceleb2_mix_smoke.py` — 200 clips,
      disjoint speakers, `data/smoke/`.
- [ ] First smoke run: dataloader → SI-SNR(mix, mix) baseline.
- **Gate:** `make test lint type` green; smoke dataloader yields 200 items.

### W2 — TSE baseline on M3
- [ ] Audio frontend (Conv1D + STFT).
- [ ] Visual frontend (mouth-ROI or pre-extracted AV-HuBERT).
- [ ] Dual-cache cross-modal fusion (port from `RIA/dualcache_avtse`).
- [ ] Mamba-2 backbone (CUDA) + chunk-attention backbone (MPS).
- [ ] TSE head + iSTFT.
- **Gates:** M3 overfit 8 clips ≥ +10 dB SI-SDRi; smoke ≥ +1 dB.

### W3 — MeMo baseline + NanoTSE slot memory (M3)
- [ ] MeMo reimpl per `av-listen/docs/MEMO_REIMPL_PLAN.md`:
      SpeakerBank, ContextBank, MeMoWrapper.forward_{chunk,offline}.
- [ ] Port `av-listen/av_listen/memory/slot_attention.py` →
      `nanotse/models/memory/`, adapt face+voice joint key.
- [ ] Wire InfoNCE + slot-consistency losses.
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
- [ ] Implement IBA (Hungarian over multi-session test).
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
