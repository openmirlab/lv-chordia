# Original inference-boundary baseline

Captured from unmodified current-port source
`aa6841bacbfd740fae2b2aee860ddc85e5339ff4` before removing training surface.
This is a behavior-preservation reference, not upstream full-model validation.
No model, preprocessing, decoder, dependency, or checkpoint was changed.

The input was captured as `test_data/yellow.wav`, now located at
`tests/fixtures/yellow.wav` with identical bytes (22.197 seconds,
stereo 44100 Hz) and an equal-length exact-zero float32 signal. The original
loader resamples both to mono 22050 Hz. The original five bundled checkpoints
are verified against `config/checkpoints.toml`, in its original order.

`outputs.npz` contains 84 float arrays: loaded audio and CQT, all six probability
heads of every ensemble member, all six ensemble means, and unrounded HMM
timestamps for all four shipped dictionaries, for music and silence.
`decoded.json` contains all eight rounded public chord-result lists. Capture
uses the real `recognize_with_ensemble` path for submission and its original
decoder for the other dictionaries. The music submission result also equals
the existing `expected_chords_yellow.json`. Silence remains exactly zero at
audio/CQT, produces nonzero model probabilities, and decodes to one N segment.

Two fresh ensembles produce exactly equal arrays and JSON. Each array records
shape, dtype, RMS, peak, and relative-to-reference RMS/peak repeat errors (zero).
Metadata records all 39 production Python hashes before/after, source revision,
input and five checkpoint hashes, state-dict shapes/dtypes, script hash, full
command, software/build/native-library fingerprints, CPU, dispatch settings,
thread counts and seed. Downloads are blocked. This CPU proof makes no GPU
claim. Floating-point exact replay rejects a different recorded environment;
the existing portable discrete regression remains independent.

From the repository root, replay without changing these immutable files:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/worzpro/Desktop/dev/openmirlab/lv-chordia/.venv/bin/python \
  tests/fixtures/inference_boundary_original/capture.py \
  --output /tmp/lv-chordia-boundary-replay
```

The interpreter used was Python 3.10.18, Torch 2.13.0+cu130, NumPy 2.2.6,
SciPy 1.15.3, librosa 0.11.0 on an Intel i5-13600K with four Torch threads.
Use a matching environment; the path above identifies the measured environment,
not a required installation location. `--record --output NEW_DIRECTORY` is
restricted to the original source commit and refuses to overwrite an existing
NPZ. To recreate the original capture, copy this script into that original
checkout before running the recorded command. Preserve the committed fixture.

Initial validation: original suite **56 passed**; standalone original replay
**84 arrays and eight result lists exact**, including fresh repeat capture.
