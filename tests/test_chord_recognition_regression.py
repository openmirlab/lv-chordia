"""
Baseline regression test for the live inference path (cli.py -> chord_recognition.py).

This is the P0 accuracy gate for the inference-only enforcement campaign: chord
recognition output on a fixed test clip must stay byte-identical across
refactors that are supposed to be behavior-preserving (dead-code removal,
dependency pruning, etc.).

Device handling is deliberately NOT hardcoded here. lv_chordia auto-selects
CUDA when available (see mir/nn/train.py); that selection logic is untouched
and GPU support is fully preserved. This test forces CPU only through the
package's normal, external mechanism -- the CUDA_VISIBLE_DEVICES environment
variable, applied to a subprocess -- so it never depends on import order and
never edits/monkeypatches the source's device-selection code. On a machine
where the GPU is unavailable or busy, this still exercises the exact same
code path the CLI uses in production; it just runs on CPU as the fallback
that already exists in mir/nn/train.py. (An explicit `--device cpu` /
`device='cpu'` override also exists now -- see test_device_selection.py --
but this test intentionally keeps exercising the external
CUDA_VISIBLE_DEVICES path, since that's what's known to be
byte-identical to the golden fixture.)

The golden fixture (tests/fixtures/expected_chords_yellow.json) was captured
against test_data/yellow.wav (already tracked in this repo) using the
"submission" chord dictionary. It was verified deterministic across two
consecutive CPU runs and, separately, identical to a GPU run on the same
clip (see the campaign's final report for the GPU comparison -- GPU-path
verification in CI is deferred until a GPU runner is available).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO = REPO_ROOT / "test_data" / "yellow.wav"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "expected_chords_yellow.json"


@pytest.fixture(scope="module")
def expected_chords():
    with open(FIXTURE) as f:
        return json.load(f)


def _run_cli_on_cpu(audio_path: str) -> list:
    """Invoke the CLI in a subprocess with CUDA hidden, to force CPU deterministically."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-m", "lv_chordia.cli", audio_path],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    return json.loads(result.stdout)


@pytest.mark.skipif(not TEST_AUDIO.exists(), reason="test_data/yellow.wav not present")
def test_chord_recognition_matches_baseline_fixture(expected_chords):
    actual = _run_cli_on_cpu(str(TEST_AUDIO))
    assert actual == expected_chords


@pytest.mark.skipif(not TEST_AUDIO.exists(), reason="test_data/yellow.wav not present")
def test_chord_recognition_is_deterministic_across_runs():
    run1 = _run_cli_on_cpu(str(TEST_AUDIO))
    run2 = _run_cli_on_cpu(str(TEST_AUDIO))
    assert run1 == run2
