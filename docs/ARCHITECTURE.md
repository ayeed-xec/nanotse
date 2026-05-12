# NanoTSE — architecture (stub)

> Filled out as W2 implementation lands. Cross-ref the system diagram
> ported from `RIA/dualcache_avtse/docs/paper_b_architecture.svg`
> (relabelled for NanoTSE), and the literature comparison in
> `docs/novelty_audit.md` (to be ported).

## High-level dataflow

```
                    ┌──────────────────────────────┐
   video 25 fps ──▶ │ visual frontend (AV-HuBERT)  │ ──▶ KV cache (25 Hz)
                    └──────────────────────────────┘
                                                          │
                                                          ▼
                    ┌──────────────────────────────┐  ┌────────────────────────┐
   audio 16 kHz ─▶  │ audio frontend (Conv1D+STFT) │─▶│ dual-cache cross-modal │
                    └──────────────────────────────┘  │       fusion           │
                                                      └─────────┬──────────────┘
                                                                ▼
                                                    ┌────────────────────────┐
                                                    │ named-slot identity    │ ◀── EMA via GRU,
                                                    │ memory (Locatello)     │     LRU eviction
                                                    └─────────┬──────────────┘
                                                              ▼
                                       ┌──────────────────────────────────────┐
                                       │ backbone:                            │
                                       │   CUDA → Mamba-2                     │
                                       │   MPS/edge → chunk-attention         │
                                       └─────────┬────────────────────────────┘
                                                 ▼
                                       ┌─────────────────────┐
                                       │ TSE head + iSTFT    │ ──▶ clean wav
                                       └─────────────────────┘
```

## Module boundaries (`nanotse/`)

- `models/frontends/`     — audio_stft, visual_avhubert
- `models/fusion/`        — dual-cache cross-modal attention
- `models/memory/`        — slot_attention (named slots, GRU update)
- `models/backbones/`     — mamba2 (CUDA), chunk_attn (MPS fallback)
- `models/heads/`         — tse, asd
- `models/baselines/`     — tdse, memo, av_skima_repro
- `models/nanotse.py`     — top-level assembly
- `losses/`               — si_snr, infonce, pcgrad, kd_feat
- `training/`             — par, paris, train_loop
- `eval/`                 — iba, latency_bench, ab_compare, diagnose
- `utils/`                — config, tracker, streaming_state
- `data/`                 — voxceleb2_mix loader, augmentation (FFT-fast)

## Key invariants

- **Sample rate:** 16 kHz audio, 25 fps video. Any path that violates
  this raises at construction time.
- **Streaming state:** all streaming modules expose
  `init_state(batch_size, device) -> State` and
  `forward_chunk(x, state) -> (y, state)`. No hidden globals.
- **Speaker disjointness:** train / val / test splits are speaker-disjoint
  by construction; regression test guards this.
- **Provenance:** every cherry-picked file carries a header comment with
  source repo + commit hash.
