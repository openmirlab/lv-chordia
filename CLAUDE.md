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
- Python API: `lv_chordia.chord_recognition.chord_recognition()` /
  `chord_recognition_json()` (alias)

`chord_recognition()` transitively imports:
`chordnet_ismir_naive.py` (model definitions), `mir.nn.train.NetworkInterface`
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

`lv_chordia/mir/` is a vendored, general-purpose MIR toolkit (has its own
`README.MD` and `requirements.txt`) that predates this package. Only a
fraction of it is used by the live inference path
(`mir.io`, `mir.data_file.DataEntry`, `mir.nn.train`, `mir.nn.data_storage`,
`mir.nn.data_decorator`, `mir.nn.data_provider`, `mir.extractors.ExtractorBase`).
It is treated as a lower-churn vendored dependency rather than lv-chordia's
own code: don't prune it opportunistically in an unrelated change. (A few
files under `mir/extractors/` -- `misc.py`, `librosa_extractor.py`,
`vamp_extractor.py` -- are themselves unreachable dead code left over from
training tooling, but were left in place during the 2026-07 inference-only
cleanup rather than touching the vendored subpackage; flagged here for a
future, dedicated pass.)

## Device handling -- do not touch casually

GPU/CPU selection defaults to automatic: `NetworkBehavior.__init__`
(`mir/nn/train.py`) checks `torch.cuda.device_count() > 0` and moves the
model to `.cuda()` if so, and this default **must stay untouched** -- GPU
support is a hard requirement of this package.

As of 2026-07, that default has an explicit, opt-in override:
`chord_recognition(..., device=...)` / `lv-chordia --device ...` accept
`'cpu'`, `'cuda'`, `'cuda:N'`, or `'auto'`, resolved by
`device_utils.resolve_use_gpu()` into the `use_gpu` bool threaded through
`NetworkBehavior`/`ChordNet`/`ChordNetCNN`. Passing nothing (`device=None`,
no `--device` flag) is byte-for-byte the same auto-detect as before this
change -- the override is additive, not a replacement of the default. Only
GPU-yes/no is wired through the pipeline; `'cuda:N'` is validated against
`torch.cuda.device_count()` but does not itself select which physical GPU
runs the model (every `.cuda()` call site targets torch's "current" device,
not a caller-chosen index) -- pinning a specific GPU still requires
`CUDA_VISIBLE_DEVICES` set before the process starts, same as today.

Do not simplify, remove, or hardcode the *default* auto-detect to CPU or
GPU. Do not add index-level device selection (e.g. `torch.cuda.set_device`)
without re-reading this note -- it was deliberately left out to avoid
mutating process-global CUDA state from a library call.

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

## Versioning

The package version is single-sourced from `lv_chordia.__version__` in
`lv_chordia/__init__.py` (`[tool.hatch.version] path = ...` in
`pyproject.toml` reads it at build time). Don't add a second, hand-edited
version field to `pyproject.toml`.
