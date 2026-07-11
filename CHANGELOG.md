# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
