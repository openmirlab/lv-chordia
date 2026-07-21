"""
Real tests for LVChordiaSession's load-once lifecycle: the ensemble is loaded
exactly once per session, reused across infer() calls, produces results
identical to the one-shot chord_recognition() path, and honors the same
fail-loudly device contract as everything else (test_device_selection.py).

The inference-bearing tests run the real five-model ensemble on CPU against
test_data/yellow.wav -- slow but honest, same tradeoff as
test_chord_recognition_regression.py.

Reads: lv_chordia/session.py, lv_chordia/chord_recognition.py
"""

from pathlib import Path

import pytest
import torch

import lv_chordia.session
from lv_chordia import LVChordiaSession
from lv_chordia.chord_recognition import chord_recognition

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO = REPO_ROOT / "test_data" / "yellow.wav"

needs_audio = pytest.mark.skipif(not TEST_AUDIO.exists(), reason="test_data/yellow.wav not present")


@needs_audio
def test_session_loads_ensemble_once_and_reuses_it_across_calls(monkeypatch):
    """Two infer() calls on one loaded session must trigger exactly one ensemble load --
    the whole point of the session API versus the one-shot path."""
    load_calls = {"count": 0}
    real_load_ensemble = lv_chordia.session.load_ensemble

    def counting_load_ensemble(*args, **kwargs):
        load_calls["count"] += 1
        return real_load_ensemble(*args, **kwargs)

    monkeypatch.setattr(lv_chordia.session, "load_ensemble", counting_load_ensemble)

    session = LVChordiaSession(device="cpu").load()
    assert load_calls["count"] == 1

    first = session.infer(TEST_AUDIO)
    second = session.infer(TEST_AUDIO)

    assert load_calls["count"] == 1, "infer() must never reload the ensemble"
    assert first == second
    assert len(first) > 0
    assert set(first[0]) == {"start_time", "end_time", "chord"}

    # load() on an already-loaded session is a no-op, not a reload.
    session.load()
    assert load_calls["count"] == 1
    session.close()


@needs_audio
def test_session_result_identical_to_one_shot_chord_recognition():
    """The session path and the throwaway-ensemble one-shot path must agree exactly,
    for the same file, chord dictionary, and device."""
    with LVChordiaSession(device="cpu") as session:
        session_result = session.infer(TEST_AUDIO)
    one_shot_result = chord_recognition(str(TEST_AUDIO), device="cpu")

    assert session_result == one_shot_result


def test_infer_before_load_raises():
    session = LVChordiaSession()
    with pytest.raises(RuntimeError, match="load"):
        session.infer("irrelevant.wav")


def test_release_drops_ensemble_and_infer_raises(monkeypatch):
    # Loading is stubbed out here -- this test is about state transitions, not weights.
    monkeypatch.setattr(lv_chordia.session, "load_ensemble", lambda use_gpu, **_kwargs: ["fake-ensemble"])

    session = LVChordiaSession(device="cpu").load()
    assert session.loaded
    session.release()
    assert not session.loaded
    with pytest.raises(RuntimeError, match="load"):
        session.infer("irrelevant.wav")

    # release() is not final: load() brings the session back.
    session.load()
    assert session.loaded


def test_closed_session_refuses_load(monkeypatch):
    monkeypatch.setattr(lv_chordia.session, "load_ensemble", lambda use_gpu, **_kwargs: ["fake-ensemble"])

    session = LVChordiaSession(device="cpu").load()
    session.close()
    assert not session.loaded
    with pytest.raises(RuntimeError, match="closed"):
        session.load()


def test_session_load_enforces_the_device_contract(monkeypatch):
    """An explicit CUDA request with no CUDA visible must raise at load(), exactly like
    chord_recognition(device='cuda') does (device_utils.resolve_use_gpu's contract)."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError):
        LVChordiaSession(device="cuda").load()

    with pytest.raises(ValueError):
        LVChordiaSession(device="tpu").load()


def test_session_contract_uses_one_resident_ensemble_and_reloads_after_release(monkeypatch):
    """Lifecycle behavior is verified with doubles, without audio or weights."""
    calls = []

    def fake_load(use_gpu, *, device=None):
        calls.append((use_gpu, device))
        return [object()]

    monkeypatch.setattr(lv_chordia.session, "load_ensemble", fake_load)
    monkeypatch.setattr(lv_chordia.session, "recognize_with_ensemble", lambda ensemble, path, chord_dict: [path, chord_dict])

    session = LVChordiaSession(device="cpu")
    assert session.status == "created"
    with pytest.raises(RuntimeError, match="load"):
        session.infer("unused.wav")
    session.load().load()
    assert calls == [(False, torch.device("cpu"))]
    assert session.infer("first.wav") == ["first.wav", "submission"]
    assert session.status == "ready"
    session.release()
    assert session.status == "released"
    session.load()
    assert len(calls) == 2
    session.close().close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()
    with pytest.raises(RuntimeError, match="load"):
        session.infer("unused.wav")


def test_session_failed_load_is_visible_and_cache_info_is_toml_backed(monkeypatch, tmp_path):
    monkeypatch.setattr(lv_chordia.session, "load_ensemble", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("broken")))
    session = LVChordiaSession(device="cpu")
    with pytest.raises(OSError, match="broken"):
        session.load()
    assert session.status == "failed"

    import importlib

    config = importlib.import_module("lv_chordia.config")

    artifact = config.checkpoint_entries()[0]
    target_root = tmp_path / "bundled"
    target_root.mkdir()
    monkeypatch.setattr(lv_chordia.session, "resolve_checkpoint_paths", lambda: (target_root, ({**artifact, "path": target_root / artifact["name"], "cached": False},)))
    info = session.cache_info()
    assert info["path"] == str(target_root)
    assert info["artifacts"] == [artifact["name"]]
    assert info["cached"] is False


def test_checkpoint_path_resolver_and_cache_info_do_not_materialize_missing_root(monkeypatch, tmp_path):
    import importlib

    config = importlib.import_module("lv_chordia.config")
    common = importlib.import_module("lv_chordia.mir.common")
    missing_root = tmp_path / "not-created"
    monkeypatch.setattr(common, "CACHE_DATA_PATH", str(missing_root))

    root, entries = config.resolve_checkpoint_paths()
    assert root == missing_root
    assert not missing_root.exists()
    assert not any(entry["cached"] for entry in entries)

    monkeypatch.setattr(lv_chordia.session, "resolve_checkpoint_paths", config.resolve_checkpoint_paths)
    info = LVChordiaSession().cache_info()
    assert info["path"] == str(missing_root)
    assert not missing_root.exists()
