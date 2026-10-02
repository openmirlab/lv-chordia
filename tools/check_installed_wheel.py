"""Verify an installed wheel's manifest, bundled weights and public inference.

Run this script with a fresh environment's Python after installing the wheel.
The repository supplies only the tracked audio and expected discrete result.
"""

import hashlib
import json
from pathlib import Path
import sys

import lv_chordia
from lv_chordia import LVChordiaSession
from lv_chordia.config import resolve_checkpoint_paths


def main():
    repository = Path(__file__).resolve().parents[1]
    package = Path(lv_chordia.__file__).resolve()
    assert repository not in package.parents, f"Source import: {package}"
    assert Path(sys.prefix).resolve() in package.parents, package
    root, entries = resolve_checkpoint_paths()
    assert len(entries) == 5
    assert Path(sys.prefix).resolve() in root.resolve().parents, root
    for entry in entries:
        assert entry["cached"], entry["path"]
        assert hashlib.sha256(entry["path"].read_bytes()).hexdigest() == entry["sha256"]
    session = LVChordiaSession(device="cpu")
    session.load()
    result = session.infer(str(repository / "test_data/yellow.wav"))
    expected = json.loads((repository / "tests/fixtures/expected_chords_yellow.json").read_text())
    assert result == expected
    session.close()
    print(f"PASS: installed {package}; five checkpoint hashes; CPU session matches {len(result)} segments")


if __name__ == "__main__":
    main()
