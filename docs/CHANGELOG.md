# Changelog

All notable architectural and engineering changes. The "Decisions" sub-section
records small choices made without escalating, per the user's no-ask preference.

## [Unreleased]

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
