"""Guarded current-port floating-point preservation alongside portable CLI tests."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_original_model_heads_and_decoders_exact(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    result = subprocess.run(
        [sys.executable, str(root / "tests/fixtures/inference_boundary_original/capture.py"),
         "--output", str(tmp_path)], cwd=root, env=env, text=True,
        capture_output=True, timeout=120,
    )
    if result.returncode and "Reference environment mismatch:" in result.stderr:
        fields = result.stderr.split("Reference environment mismatch:")[-1].strip()
        pytest.skip("Original CPU fixture environment mismatch: " + fields)
    assert result.returncode == 0, result.stdout + result.stderr
