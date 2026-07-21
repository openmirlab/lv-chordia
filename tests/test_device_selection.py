"""Cheap unit tests for lv_chordia.device_utils and its threading into
NetworkBehavior/ChordNet (no model weights or real GPU needed)."""

import pytest
import torch

from lv_chordia.chordnet_ismir_naive import ChordNet
from lv_chordia.device_utils import resolve_device, resolve_use_gpu
from lv_chordia.mir.nn.network import NetworkBehavior


def test_device_none_defers_to_auto_detect():
    # None must return None -- NetworkBehavior computes the exact same
    # torch.cuda.device_count() > 0 expression it always has, untouched.
    assert resolve_use_gpu(None) is None


def test_device_auto_also_defers_to_auto_detect():
    assert resolve_use_gpu("auto") is None


def test_device_cpu_forces_cpu_even_when_cuda_reported_available(monkeypatch):
    # The regression case: an explicit device="cpu" request must be honored
    # even when CUDA is reported available, not silently overridden by
    # auto-detection.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_use_gpu("cpu") is False


def test_device_cuda_forces_gpu_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_use_gpu("cuda") is True


def test_device_cuda_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError):
        resolve_use_gpu("cuda")


def test_device_cuda_index_is_preserved(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_use_gpu("cuda:1") is True
    assert resolve_device("cuda:1") == torch.device("cuda:1")


def test_device_cuda_index_out_of_range_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError):
        resolve_use_gpu("cuda:5")


def test_device_cuda_index_when_no_cuda_visible_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError):
        resolve_use_gpu("cuda:0")


def test_device_mps_requires_an_available_backend(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")


@pytest.mark.parametrize("bad_device", ["tpu", "gpu", "cuda:", "cuda:one", "mps:0", ""])
def test_invalid_device_string_raises_value_error(bad_device):
    with pytest.raises(ValueError):
        resolve_use_gpu(bad_device)


def test_networkbehavior_default_still_auto_detects(monkeypatch):
    # No override passed at all -- must match today's exact behavior.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert NetworkBehavior().use_gpu is False

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert NetworkBehavior().use_gpu is True


def test_networkbehavior_explicit_false_overrides_cuda_available(monkeypatch):
    # Even when CUDA is reported available, an explicit override to False
    # must be honored -- this is the "force CPU from Python" capability
    # that previously required the external CUDA_VISIBLE_DEVICES workaround.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert NetworkBehavior(use_gpu=False).use_gpu is False


def test_chordnet_threads_use_gpu_override_through_to_networkbehavior(monkeypatch):
    # ChordNet is the class chord_recognition() actually constructs; prove
    # the override reaches NetworkBehavior.use_gpu through ChordNet's
    # constructor, not just directly on NetworkBehavior.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    net = ChordNet(None, use_gpu=False)
    assert net.use_gpu is False


def test_chordnet_default_still_auto_detects(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert ChordNet(None).use_gpu is False


def test_cuda_index_reaches_chordnet_construction(monkeypatch):
    """The public one-shot path must not collapse ``cuda:1`` into a bool."""
    import importlib

    api = importlib.import_module("lv_chordia.chord_recognition")

    captured = {}

    class FakeChordNet:
        def __init__(self, _counter, *, use_gpu, device):
            captured["use_gpu"] = use_gpu
            captured["device"] = device

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(api, "ChordNet", FakeChordNet)
    monkeypatch.setattr(api, "NetworkInterface", lambda net, name, load_checkpoint: (net, name))

    api.load_ensemble(True, device=resolve_device("cuda:1"))
    assert captured == {"use_gpu": True, "device": torch.device("cuda:1")}


def test_production_ensemble_names_come_from_package_toml():
    import importlib

    api = importlib.import_module("lv_chordia.chord_recognition")
    config = importlib.import_module("lv_chordia.config")
    assert api.MODEL_NAMES == config.model_names()
