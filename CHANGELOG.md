# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## Load-once session API (2026-07-19, branch `master`, local)

`LVChordiaSession` (added 2026-07-15) was a lifecycle facade in name only: its
`load()` deliberately deferred to the legacy pipeline, which reloaded all five
ChordNet ensemble members from `cache_data/*.sdict` on every
`chord_recognition()` call. This change makes the session real.

### Added
- `chord_recognition.load_ensemble(use_gpu)`: loads the five ensemble members
  once (the expensive part -- five `torch.load` calls; chord-dictionary
  independent).
- `chord_recognition.recognize_with_ensemble(ensemble, audio_path,
  chord_dict_name)`: the per-call part -- HMM decoder for the chosen chord
  dictionary (the only thing the dictionary drives), CQT features, ensemble
  inference, decode. No weights are loaded here.
- `LVChordiaSession(device=...)`: the session now accepts the same device
  contract as `chord_recognition()` (`'cpu'`/`'cuda'`/`'cuda:N'`/`'auto'`/
  None), resolved once at `load()` through `device_utils.resolve_use_gpu`'s
  fail-loudly check.
- `tests/test_session.py`: 6 real tests -- ensemble loaded exactly once across
  two `infer()` calls, session-vs-one-shot result identity on
  `test_data/yellow.wav`, release/close state transitions, and the device
  contract at `load()`.

### Changed
- `LVChordiaSession.load()` genuinely loads the ensemble once; `infer()`
  reuses it (with an optional per-call `chord_dict_name` override);
  `release()` drops the model references (reloadable), `close()` is final.
- `chord_recognition()` now composes `load_ensemble()` +
  `recognize_with_ensemble()`. Behavior and results are unchanged (the
  byte-for-byte regression gate `test_chord_recognition_regression.py` stays
  green; full suite 47 passed); the only observable difference is failure
  order -- model loading now happens before a URL download instead of after.

## CI matrix + dependency floor refresh (2026-07-12, branch `fix/ci-matrix`)

An org audit found this repo tested only Python 3.10, and only inside
publish.yml's release-gate job -- there was no dedicated `test.yml` running
on every push/PR, despite classifiers claiming 3.8-3.12 support. This change
builds that CI and refreshes stale dependency floors, verified empirically at
every step (org constitution article 3: "floors, not ceilings"; article 7:
"wheel-from-sdist install smoke test"). Dependabot: 0 open alerts, reconfirmed
via `gh api repos/openmirlab/lv-chordia/dependabot/alerts` -- no security
fixes needed here.

### Added
- `.github/workflows/test.yml`: a `test` job running the full 24-test
  `pytest` suite across a Python 3.10/3.11/3.12/3.13 matrix via `uv`
  (`astral-sh/setup-uv@v4` + `uv python install` + `uv sync --extra dev`),
  triggered on push to `master`, on every PR, and via `workflow_dispatch`.
  All four versions verified green in fresh `uv`-managed venvs before being
  added -- none were excluded.
- A `build` job (`needs: [test]`) doing the wheel-from-sdist install smoke
  test: `python -m build`, install the wheel into a clean `venv`, import
  `lv_chordia` and touch `__version__` and
  `lv_chordia.chord_recognition.chord_recognition`. Also asserts the wheel
  still contains its expected `lv_chordia/**/*.py` files (>=10; actual count
  36) -- guarding against the exact `[tool.hatch.build.targets.wheel]
  packages` misconfiguration class of bug the constitution flags, given this
  repo's wheel target also carries a `shared-data` entry for `cache_data`.
  Note: unlike some openmirlab siblings, lv-chordia's wheel intentionally
  bundles its pretrained-model `.sdict` weight files via that `shared-data`
  mechanism (pre-existing design, not runtime-downloaded) -- the build job
  does not flag bundled weights as an error here, since that would be a false
  positive against this repo's actual packaging contract.
- README `Test` CI badge.
- Python 3.13 classifier (confirmed green; previously absent even though
  3.8/3.9 were claimed and untested).

### Changed
- Stale dependency floors bumped, each verified across the full 3.10-3.13
  matrix before being kept:
  - `torch>=2.0.0` -> `>=2.13.0` (current latest stable; ships cp310-cp313
    wheels).
  - `h5py>=2.9.0` -> `>=3.16.0` (current latest; very stale before).
  - `joblib>=0.13.2` -> `>=1.5.3` (current latest; very stale before).
  - `librosa>=0.7.2` -> `>=0.11.0` (current latest; doesn't force an
    incompatible numpy/scipy/numba floor for Python 3.10).
  - `numpy>=1.19.2` -> `>=2.2.6` -- **the numpy/Python-version trap**:
    `numpy>=2.3` requires Python>=3.11 and `numpy>=2.5` requires
    Python>=3.12, which would break the Python 3.10 leg of this repo's new
    CI matrix. `2.2.6` is the newest 2.x release whose own `requires-python`
    (`>=3.10`) is still compatible with all four matrix versions (confirmed
    via PyPI JSON metadata, not assumed).
- **`requires-python` bumped `>=3.8` -> `>=3.10`** (a change beyond the
  original CI-only ask, forced by the floor bumps above, not an independent
  scope decision): `torch>=2.13.0` itself only ships cp310+ wheels, and
  `numpy>=2.2.6` itself requires Python>=3.10, so the second the mandated
  torch/numpy floor bumps landed, the `>=3.8` claim became factually false --
  `uv sync`'s universal resolver confirmed this empirically (unsatisfiable
  resolution errors citing exactly these two packages) before the bump.
  Classifiers updated to match: 3.8/3.9 dropped, 3.13 added.
- **Consolidated the two dev-dependency declarations into one.** This repo
  previously declared dev deps in both `[project.optional-dependencies].dev`
  (pytest, black, flake8, build, twine) and a separate PEP-735
  `[dependency-groups].dev` (build, hatchling, pytest). Empirically, `uv
  sync --extra dev` was already installing both (uv installs the default
  dependency group alongside any requested extra unless told not to), so the
  duplication only added confusion, not failures. Removed
  `[dependency-groups]` entirely; `[project.optional-dependencies].dev` is
  now the single source of dev deps (pytest, black, flake8, `build>=1.0.0`,
  twine). `hatchling` was dropped as an explicit dev dependency since it is
  already declared in `[build-system].requires` and uv/pip provide it via
  build isolation regardless.
- `uv.lock` regenerated (110 packages resolved, `requires-python = ">=3.10"`).
- README: `Test` badge added; Python badge `3.8+` -> `3.10+`; PyTorch badge
  `2.0+` -> `2.13+`; dependencies list and Requirements section floors
  updated to match pyproject.toml.
- `publish.yml`: added a comment clarifying its Python-3.10-only `test` job
  is a deliberate release-gate simplification now that the full matrix lives
  in `test.yml`; the job itself is unchanged.

### Test counts
- Before: 24 passed (Python 3.10 only, via `pip install -e ".[dev]"` --
  reconfirmed, matches prior audit baseline).
- After: 24 passed on every matrix leg (3.10, 3.11, 3.12, 3.13), both via
  `uv sync --extra dev` + `uv run pytest` and via the original
  `pip install -e ".[dev]"` path (re-verified on 3.10 for publish.yml
  parity).

## [1.1.0] - 2026-07-11

### Removed

- Removed all remaining training/eval code so the package is genuinely
  inference-only (the 2025-11-29 "inference-only" commit had left ~1000 LOC
  of it in place): `results.py`, `results_ismir2017.py`,
  `chordnet_ismir_naive_eval.py`, `test_for_all.py`, `storage_creation.py`,
  `train_eval_test_split.py`, and the `datasets` module.
- Removed 13 files under `lv_chordia/extractors/` and 14 files under
  `lv_chordia/io_new/` that existed solely to support the training/eval
  scripts above (beat/downbeat/lyric/salami annotation readers, JAMS I/O,
  dataset-only feature extractors) and were left orphaned once those scripts
  were deleted.
- Removed the last remaining file in `lv_chordia/io_new/`,
  `chordlab_io.py`, and the `lv_chordia/io_new/` package with it:
  `chord_recognition.py` imported `ChordLabIO` from it but never actually
  called it (confirmed by removing the import, re-running the CLI, and
  diffing output against the regression fixture before deleting the file).
  Also dropped two other unused imports discovered the same way in
  `chord_recognition.py`: `chord_limit` and `ChordNetCNN` from
  `chordnet_ismir_naive.py` (the latter is a model class that is defined but
  never instantiated anywhere in the live path).
- Removed unused, training-era leftover constants from `lv_chordia/settings.py`
  (11 hardcoded dataset paths pointing at a developer's local machine, e.g.
  `JAM_DATASET_PATH`, `BILLBOARD_DATASET_PATH`, plus 4 further unused
  constants); only `DEFAULT_SR` and `DEFAULT_HOP_LENGTH` were ever read by
  the live path.
- Removed dependencies that were only required by the deleted code:
  `figures`, `jams`, `pumpp`, `mir_eval` (training/eval-only), and
  `matplotlib`, `scikit_learn` (unused anywhere in the codebase).

### Changed

- `lv_chordia/__init__.py` no longer eagerly imports the training-only
  `datasets` module, and no longer wraps its imports in a blanket
  `try/except ImportError: pass` -- a real import error in the inference
  path now surfaces instead of being silently swallowed.
- Raised the `torch` floor from `>=1.4.0` to `>=2.0.0`, reflecting what the
  inference path actually requires and runs on (verified against torch
  2.7.1).
- Package version is now single-sourced from `lv_chordia.__version__`
  (`[tool.hatch.version] path = "lv_chordia/__init__.py"`) instead of being
  duplicated in `pyproject.toml`.
- The PyPI publish workflow (`.github/workflows/publish.yml`) now runs the
  test suite as a gate before build/publish.

### Added

- A real pytest suite (`tests/`): import smoke tests (including regression
  guards that the deleted training modules stay deleted), unit tests for
  `audio_utils`, and a baseline regression test that runs the CLI against
  the tracked `test_data/yellow.wav` fixture and asserts the chord
  recognition JSON output is byte-identical to a golden fixture.

### Notes

- GPU/CPU device selection (`mir/nn/train.py`'s automatic CUDA detection)
  is unchanged; GPU support is fully preserved. The regression test forces
  CPU only via `CUDA_VISIBLE_DEVICES` on a subprocess, not by touching
  device-selection source.
