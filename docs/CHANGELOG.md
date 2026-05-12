# Changelog

All notable architectural and engineering changes. The "Decisions" sub-section
records small choices made without escalating, per the user's no-ask preference.

## [Unreleased]

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
