# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository.

## What this package is

`lv-chordia` is an **inference-only** chord recognition package: given an
audio file (or URL), it returns a time-aligned chord sequence as JSON. It
packages the pre-trained ensemble models from the ISMIR 2019 paper
"Large-Vocabulary Chord Transcription via Chord Structure Decomposition".

There is no training or evaluation code in this repository, and no
dependency on any training dataset. If you find yourself adding a training
loop, a dataset loader, or an eval/benchmark script here, stop -- that
doesn't belong in this package.

## Weights hosting: documented size-based exception to org constitution article 4

Unlike most other openmirlab inference packages, this repo does **not**
download its weights at runtime. The pre-trained ensemble
(`cache_data/*.sdict`, 5 files, ~28MB total, 5.5MB each) is committed
directly to git and shipped inside the built wheel/sdist via
`pyproject.toml`'s `shared-data`/`sdist` config -- this is the package's
pre-existing, original design, not a recent regression.

This is a **deliberate, documented exception** to the org's default weights
contract (constitution article 4: weights are normally downloaded at
runtime, never committed to git), confirmed acceptable on 2026-07-12. The
justification is size: 28MB total is small enough that bundling costs
little and buys a fully-offline install with zero download/caching/sha256
machinery -- the same size-vs-simplicity tradeoff behind
drum-classifier-infer's bundled checkpoint (there the driver was license
instead of size, but the org-level precedent -- bundling is fine when the
weight is genuinely small -- is the same one applied here). This is not a
defect to migrate away from; do not treat it as a TODO.

**Still do not delete or otherwise touch `cache_data/*.sdict` or any
git-tracked weight file casually** -- if a future change genuinely needs to
move to runtime download (e.g. the ensemble grows well past this size, or
the org tightens the exception threshold), build the downloader, host the
weights (the org's usual pattern is a versioned external host + sha256
verification, as in bs-roformer-infer/melband-roformer-infer), then update
this note, `pyproject.toml`'s packaging config, and README's Scope section
together -- but that is a deliberate future call, not a standing violation
to clear.

## Entry points and the live import graph

- CLI: `lv_chordia/cli.py` (`lv-chordia` console script) -> `chord_recognition()`
- Python API (one-shot): `lv_chordia.chord_recognition.chord_recognition()` /
  `chord_recognition_json()` (alias) -- loads a throwaway five-model ensemble
  per call.
- Python API (resident): `lv_chordia.LVChordiaSession` (`session.py`) -- loads
  the ensemble once at `load()` for a resolved device and reuses it across
  `infer()` calls (made a real load-once session on 2026-07-19; it previously
  deferred to the per-call reload). The chord dictionary is a per-call choice:
  it only drives the HMM decoder built in `recognize_with_ensemble()`, never
  the ensemble load. `chord_recognition()` itself now composes
  `load_ensemble()` + `recognize_with_ensemble()` -- the split lives in
  `chord_recognition.py` and is the one owner of the pipeline math.

`chord_recognition()` transitively imports:
`chordnet_ismir_naive.py` (model definitions), `mir.nn.network.NetworkInterface`
(checkpoint loading + inference), `extractors.cqt` / `extractors.xhmm_ismir`
(feature extraction + HMM decoding), `mir.io` / `mir.DataEntry`,
`settings.py`, `audio_utils.py` (local file + URL handling).

Before adding, removing, or "cleaning up" any file, trace whether it's
reachable from this chain. A file not imported by anything on this chain is
dead code -- but verify with `grep -rn` for every import spelling (relative
and absolute) before deleting; a prior survey once misclassified a live file
as dead. (The `lv_chordia/io_new/` package used to have one file,
`chordlab_io.py`, imported here, but the import was itself dead -- the
class was never actually called. Confirmed empirically by removing the
import and re-running the accuracy-regression test before deleting the
file; `io_new/` no longer exists.)

## The `mir/` subpackage

`lv_chordia/mir/` is a vendored MIR toolkit subset. The live inference path
uses `mir.io`, `mir.data_file.DataEntry`, `mir.nn.network`, and
`mir.extractors.ExtractorBase`. Single-entry proxies, TextureBuilder and
inference/visualization helpers remain; do not prune these in unrelated work.

The dedicated inference-boundary pass removed the three `mir.nn.data_*`
modules, DataPool/export, evaluation adapters, training losses/augmentation,
and dataset-storage configuration. No runtime imports of h5py/joblib remain;
they are no longer direct dependencies (librosa may still bring joblib).
Ten unused cross-fold count `.pkl` assets were removed, with names and hashes
retained in `tests/fixtures/inference_boundary_original/removed_training_assets.json`.
The five `.sdict` checkpoints were untouched. `ChordNet(None)` and
`ChordNetCNN(None)` retain layers, RNG construction order, state-dict names,
input slices and inference math; non-None training counters fail before model
construction. NetworkInterface still accepts historical checkpoint metadata.

`mir/nn/train.py` was renamed to `mir/nn/network.py` on 2026-07-19: despite
its old name it held genuinely load-bearing inference code
(`NetworkBehavior`, `NetworkInterface.inference()`/`inference_function()`),
mixed with training-only dead weight (`loss()`/`evaluation()` abstract
stubs never called anywhere, the unreachable `is_training=True` branch of
`init_settings` -- every call site in this package passes `is_training`
implicitly false now, and `get_optimizer()`/the checkpoint's `opt` restore,
which nothing downstream ever read). The dead pieces were deleted and the
file renamed once its remaining content was honestly inference-only; the
CSV cross-validation fold manifests in `data/train0{0-4}.csv` (unreferenced
by any code, README, or CI) were deleted the same day. The later dedicated
boundary pass removed the remaining model-specific losses as described above.

## Device handling -- do not touch casually

GPU/CPU selection defaults to automatic: `NetworkBehavior.__init__`
(`mir/nn/network.py`) checks `torch.cuda.device_count() > 0` and moves the
model to `.cuda()` if so, and this default **must stay untouched** -- GPU
support is a hard requirement of this package.

As of 2026-07, that default has an explicit, opt-in override:
`chord_recognition(..., device=...)` / `lv-chordia --device ...` accept
`'cpu'`, `'cuda'`, `'cuda:N'`, or `'auto'` (`'mps'` is rejected outright --
Apple MLX/MPS backends are permanently out of scope, org canon art. 4b),
resolved by `device_utils.resolve_device()` and threaded through
`NetworkBehavior`/`ChordNet`/`ChordNetCNN`. Passing nothing (`device=None`,
no `--device` flag) is byte-for-byte the same auto-detect as before this
change -- the override is additive, not a replacement of the default. An
explicit `'cuda:N'` is validated against `torch.cuda.device_count()` and is
passed into model and tensor `.to()` calls without changing process-global
CUDA state.

Do not simplify, remove, or hardcode the *default* auto-detect to CPU or
GPU. Do not add `torch.cuda.set_device`: explicit indexes are carried on
individual model/tensor operations and must not mutate global CUDA state.

## Accuracy rule

Any change touching the inference path (`cli.py`, `chord_recognition.py`,
`chordnet_ismir_naive.py`, `mir/nn/*`, `extractors/cqt.py`,
`extractors/xhmm_ismir.py`, `settings.py`, `audio_utils.py`,
`complex_chord.py`) must produce byte-identical chord recognition JSON
output on the regression fixture before and after. Verify
with:

```bash
pytest tests/test_chord_recognition_regression.py -v
```

This compares the CLI's output on the tracked `test_data/yellow.wav` against
`tests/fixtures/expected_chords_yellow.json`. If a change is expected to
alter model output (e.g. retraining, a genuine bug fix in decoding), update
the fixture deliberately and say so in the commit message -- don't let it
change silently.

## Running tests

```bash
uv sync --extra dev   # or: pip install -e ".[dev]"
pytest tests/ -v
```

No network access or GPU is required; model weights (`cache_data/*.sdict`)
and the test audio (`test_data/yellow.wav`) are tracked in the repo.

`tests/test_inference_boundary.py` establishes the import/capability boundary
(13 failures before removal). `tests/test_inference_baseline.py` runs the guarded
original-current-port CPU replay: 84 arrays and eight result lists, real music
and exact silence, every ensemble member/head and all four dictionaries. The
original capture was committed at `42a92c9` before production edits. Use:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python tests/fixtures/inference_boundary_original/capture.py --output /tmp/lv-chordia-replay
```

The fixture README records the measured environment and capture command. Exact
float checks reject different software/build/native-library/CPU-dispatch
fingerprints; do not regenerate or widen bounds to hide a mismatch. This is
current-port CPU preservation, not upstream full-model or GPU validation.
The public skills chord-recognition entry was checked: its bundled-weight
installation guidance remains accurate and needs no edit.


## Versioning

Packaging verification: `python -m build` must include
`lv_chordia/config/checkpoints.toml` in the sdist as well as the wheel. Install
the wheel into a fresh environment, then run that environment's Python on
`tools/check_installed_wheel.py` from this checkout. It rejects source-tree
imports, verifies all five installed checkpoint hashes, and runs the public
CPU session against the existing Yellow JSON fixture. The original
`aa6841b` wheel-from-sdist omitted the manifest and failed installed import;
source-tree tests alone did not expose this.

The package version is single-sourced from `lv_chordia.__version__` in
`lv_chordia/__init__.py` (`[tool.hatch.version] path = ...` in
`pyproject.toml` reads it at build time). Don't add a second, hand-edited
version field to `pyproject.toml`.
