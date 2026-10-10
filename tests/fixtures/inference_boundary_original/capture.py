"""Capture/replay the unmodified current-port inference boundary, not upstream goldens.

Run with CUDA hidden and OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=4. Recording is
restricted to the original source revision; replay never overwrites fixtures.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
ORIGINAL = "aa6841bacbfd740fae2b2aee860ddc85e5339ff4"
HEADS = ("triad", "bass", "seventh", "ninth", "eleventh", "thirteenth")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "lv_chordia").rglob("*.py"))}


def environment(np, torch):
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    cpu = Path("/proc/cpuinfo").read_text().split("\n\n")[0]
    cpu = [line for line in cpu.splitlines() if line.split(":")[0].strip() in ("vendor_id", "cpu family", "model", "model name", "stepping", "microcode", "flags")]
    native = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if path.startswith("/") and any(s in path for s in ("libc.so", "libm.so", "libtorch_cpu", "libopenblas", "libscipy_openblas")):
            native[Path(path).name] = sha(path)
    return {
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "librosa", "numba", "llvmlite", "soundfile", "soxr")},
        "device": "cpu", "machine": platform.machine(), "libc": platform.libc_ver(),
        "cpu": cpu, "native_libraries": native,
        "torch_config": torch.__config__.show(), "numpy_config": config.getvalue(),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "mkldnn": torch.backends.mkldnn.enabled,
        "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NPY_DISABLE_CPU_FEATURES", "GLIBC_TUNABLES")},
    }


def metrics(np, value):
    v = np.asarray(value, dtype=np.float64)
    assert np.isfinite(v).all()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "rms": float(np.sqrt(np.mean(v * v))), "peak": float(np.max(np.abs(v))), "minimum": float(v.min()), "maximum": float(v.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("OMP_NUM_THREADS") == os.environ.get("OPENBLAS_NUM_THREADS") == "4"
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(4)
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    def no_download(*a, **kw):
        raise AssertionError("Baseline must not download")
    urllib.request.urlopen = no_download
    urllib.request.urlretrieve = no_download
    import importlib
    pipeline = importlib.import_module("lv_chordia.chord_recognition")
    from lv_chordia.config import resolve_checkpoint_paths
    from lv_chordia.extractors.cqt import CQTV2
    from lv_chordia.extractors.xhmm_ismir import XHMMDecoder

    source_before = source_hashes()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    reference_dir = Path(__file__).parent
    if args.record:
        assert revision == ORIGINAL, "Capture only the original production source"
        assert not (args.output / "outputs.npz").exists(), "Never overwrite a captured baseline"
    checkpoints = {}
    _, checkpoint_paths = resolve_checkpoint_paths()
    for item in checkpoint_paths:
        path = item["path"]
        assert sha(path) == item["sha256"], path
        checkpoints[item["name"]] = {"sha256": sha(path), "bytes": path.stat().st_size}
    # Recording runs on the original revision; replay uses the reorganized fixture.
    music = ROOT / ("test_data/yellow.wav" if args.record else "tests/fixtures/yellow.wav")
    samples, sr = sf.read(music, dtype="float32", always_2d=True)
    assert np.sqrt(np.mean(samples.astype(np.float64) ** 2)) > 0.01
    original_extract = CQTV2.extract
    collected, decoded, layouts = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        silence = Path(tmp) / "silence.wav"
        sf.write(silence, np.zeros_like(samples), sr, subtype="FLOAT")
        assert np.count_nonzero(sf.read(silence)[0]) == 0
        for repeat in range(2):
            arrays, segments, states = {}, {}, {}
            ensemble = pipeline.load_ensemble(False, device=torch.device("cpu"))
            assert len(ensemble) == 5
            for index, model in enumerate(ensemble):
                assert all(p.device.type == "cpu" for p in model.net.parameters())
                states[str(index)] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in model.net.state_dict().items()}
            for case, path in (("music", music), ("silence", silence)):
                entries = []
                def capture_extract(self, entry, **kwargs):
                    result = original_extract(self, entry, **kwargs)
                    entries.append(entry)
                    arrays[case + "/audio"] = entry.music.copy()
                    arrays[case + "/cqt"] = result.copy()
                    return result
                CQTV2.extract = capture_extract
                originals = [model.inference for model in ensemble]
                predictions = []
                for index, model in enumerate(ensemble):
                    def capture_inference(x, index=index):
                        output = originals[index](x)
                        predictions.append(output)
                        for head, value in zip(HEADS, output):
                            assert value.ndim == 2 and np.isfinite(value).all()
                            assert np.max(np.abs(value.sum(axis=1) - 1)) < 1e-6
                            arrays[f"{case}/fold{index}/{head}"] = value.copy()
                        return output
                    model.inference = capture_inference
                segments[case + "/submission"] = pipeline.recognize_with_ensemble(ensemble, str(path), "submission")
                for model, original in zip(ensemble, originals):
                    model.inference = original
                assert len(entries) == 1 and len(predictions) == 5
                means = [np.mean([p[i] for p in predictions], axis=0) for i in range(6)]
                for head, value in zip(HEADS, means):
                    arrays[f"{case}/ensemble/{head}"] = value
                for dictionary in ("submission", "ismir2017", "full", "extended"):
                    decoder = XHMMDecoder(template_file=str(ROOT / "lv_chordia/data" / (dictionary + "_chord_list.txt")))
                    raw = decoder.decode_to_chordlab(entries[0], means, False)
                    arrays[f"{case}/{dictionary}/times"] = np.asarray([[r[0], r[1]] for r in raw], dtype=np.float64)
                    rendered = [{"start_time": float(f"{r[0]:.2f}"), "end_time": float(f"{r[1]:.2f}"), "chord": str(r[2])} for r in raw]
                    if dictionary == "submission":
                        assert rendered == segments[case + "/submission"]
                    segments[f"{case}/{dictionary}"] = rendered
                assert np.count_nonzero(arrays[case + "/audio"]) == 0 if case == "silence" else np.count_nonzero(arrays[case + "/audio"]) > 0
                print(f"repeat={repeat} case={case} frames={means[0].shape[0]}", flush=True)
            collected.append(arrays)
            decoded.append(segments)
            layouts.append(states)
    CQTV2.extract = original_extract
    assert decoded[0] == decoded[1] and layouts[0] == layouts[1]
    assert collected[0].keys() == collected[1].keys()
    for key, value in collected[0].items():
        assert np.array_equal(value, collected[1][key]), key
    assert any(row["chord"] != "N" for row in decoded[0]["music/submission"])
    assert decoded[0]["music/submission"] == json.loads((ROOT / "tests/fixtures/expected_chords_yellow.json").read_text())
    assert not np.array_equal(collected[0]["music/ensemble/triad"], collected[0]["silence/ensemble/triad"])
    assert source_before == source_hashes()
    runtime = environment(np, torch)
    # JSON-normalize tuple-valued runtime fields for portable comparison.
    runtime = json.loads(json.dumps(runtime))
    # Keep the original input identifier in provenance; its bytes are unchanged.
    metadata = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "Original current-port baseline, not upstream full-model golden parity",
        "source_sha": revision, "production_sha256_before": source_before, "production_sha256_after": source_hashes(),
        "capture_sha256": sha(__file__), "command": "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 " + sys.executable + " " + " ".join(sys.argv),
        "input": {"path": "test_data/yellow.wav", "sha256": sha(music), "sample_rate": sr, "shape": list(samples.shape), "decoded_dtype": str(samples.dtype), "silence": "same shape/rate, exact zero float32 samples written as FLOAT WAV"},
        "checkpoints": checkpoints, "model_order": pipeline.MODEL_NAMES, "model_state_layouts": layouts[0],
        "runtime": runtime, "seed": 1234, "repeat_exact": True, "downloads_blocked": True,
        "arrays": {k: metrics(np, v) for k, v in collected[0].items()},
        "repeat_comparison": {k: {"max_abs": 0.0, "rms_error_over_reference_rms": 0.0, "max_abs_over_reference_peak": 0.0} for k in collected[0]},
        "imported_package_modules": sorted(k for k in sys.modules if k.startswith("lv_chordia")),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if args.record:
        np.savez_compressed(args.output / "outputs.npz", **collected[0])
        (args.output / "decoded.json").write_text(json.dumps(decoded[0], indent=2) + "\n")
        metadata["artifact_sha256"] = {name: sha(args.output / name) for name in ("outputs.npz", "decoded.json")}
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        reference = json.loads((reference_dir / "metadata.json").read_text())
        for name, digest in reference["artifact_sha256"].items():
            assert sha(reference_dir / name) == digest, "Corrupt fixture: " + name
        if runtime != reference["runtime"]:
            raise SystemExit("Reference environment mismatch: " + ", ".join(k for k in runtime if runtime[k] != reference["runtime"][k]))
        assert metadata["input"] == reference["input"]
        assert checkpoints == reference["checkpoints"]
        assert layouts[0] == reference["model_state_layouts"]
        assert decoded[0] == json.loads((reference_dir / "decoded.json").read_text())
        with np.load(reference_dir / "outputs.npz") as expected:
            assert set(expected.files) == set(collected[0])
            for key, value in collected[0].items():
                assert np.array_equal(value, expected[key]), key
        (args.output / "replay.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"PASS: {len(collected[0])} arrays, 8 decoded results, two exact captures")


if __name__ == "__main__":
    main()
