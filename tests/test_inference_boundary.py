"""The shipped package supports recognition, not dataset/training/evaluation work."""

import importlib
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("name", ["data_provider", "data_storage", "data_decorator"])
def test_training_data_modules_are_not_importable(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lv_chordia.mir.nn." + name)


def test_package_import_does_not_require_hdf5_dataset_storage():
    code = """
import importlib.abc
import sys
class NoHDF5(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'h5py' or fullname.startswith('h5py.'):
            raise ImportError('Dataset-only HDF5 is unavailable')
sys.meta_path.insert(0, NoHDF5())
import lv_chordia
assert callable(lv_chordia.chord_recognition)
assert not any(name.startswith('lv_chordia.mir.nn.data_') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("name", ["ReweightedLoss", "FocalLoss", "ComplexChordShifter"])
def test_model_module_has_no_training_components(name):
    module = importlib.import_module("lv_chordia.chordnet_ismir_naive")
    assert not hasattr(module, name)


@pytest.mark.parametrize("name", ["ChordNet", "ChordNetCNN"])
def test_models_keep_inference_without_training_loss(name):
    module = importlib.import_module("lv_chordia.chordnet_ismir_naive")
    model = getattr(module, name)(None, use_gpu=False)
    assert callable(model.inference)
    assert not hasattr(model, "loss")


@pytest.mark.parametrize("name", ["ChordNet", "ChordNetCNN"])
def test_training_counter_rejected_before_model_construction(name, monkeypatch):
    module = importlib.import_module("lv_chordia.chordnet_ismir_naive")
    def unexpected_construction():
        pytest.fail("Constructed the CNN for an unsupported training request")
    monkeypatch.setattr(module, "CNNFeatureExtractor", unexpected_construction)
    with pytest.raises(ValueError, match="training.*counter|counter.*training"):
        getattr(module, name)(object(), use_gpu=False)


def test_single_entry_processing_survives_without_dataset_or_evaluation_api():
    from lv_chordia import mir
    from lv_chordia.mir import data_file, io
    from lv_chordia.mir.extractors import misc
    assert callable(mir.DataEntry)
    assert callable(mir.TextureBuilder)
    assert not hasattr(mir, "DataPool")
    assert not hasattr(data_file, "DataPool")
    assert not hasattr(misc, "Evaluate")
    assert not hasattr(io.FeatureIO, "file_to_evaluation_format")
    assert not hasattr(io.FeatureIO, "data_to_evaluation_format")


def test_training_count_assets_are_not_shipped():
    import lv_chordia
    data = Path(lv_chordia.__file__).parent / "data"
    assert not list(data.glob("cross*_weight*.pkl"))
