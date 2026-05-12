# Changelog

All notable architectural and engineering changes. The "Decisions" sub-section
records small choices made without escalating, per the user's no-ask preference.

## [Unreleased]

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
