"""
The inference pipeline: audio in, time-aligned chord JSON out.

This is the one real entry point of the package (cli.py just wraps it;
session.py holds a preloaded ensemble across calls). The pipeline runs a
5-model ensemble (ChordNet, defined in chordnet_ismir_naive.py) over CQT
features (extractors/cqt.py) loaded via mir.nn.train.NetworkInterface,
averages the ensemble's per-frame probabilities, and decodes them into chord
segments with an HMM (extractors/xhmm_ismir.py) driven by a chosen chord
dictionary (lv_chordia/data/*_chord_list.txt). Split into load_ensemble()
(the expensive part: five torch.load calls, chord-dict independent) and
recognize_with_ensemble() (per-call: CQT + inference + HMM decode);
chord_recognition() composes the two for one-shot callers. Accuracy-sensitive:
any change here must keep tests/test_chord_recognition_regression.py passing
byte-for-byte.

Reads: chordnet_ismir_naive.py, mir/nn/train.py, extractors/cqt.py,
extractors/xhmm_ismir.py, settings.py, audio_utils.py, device_utils.py
"""

from .chordnet_ismir_naive import ChordNet
from .mir.nn.train import NetworkInterface
from .extractors.cqt import CQTV2,SimpleChordToID
from .mir import io,DataEntry
from .extractors.xhmm_ismir import XHMMDecoder
import numpy as np
from .settings import DEFAULT_SR,DEFAULT_HOP_LENGTH
from .audio_utils import get_audio_path, cleanup_temp_audio
from .device_utils import resolve_use_gpu
import sys
import json
import os
from typing import List, Dict, Union, Optional
import importlib.resources

MODEL_NAMES = ['joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best' % i for i in range(5)]


def load_ensemble(use_gpu: Optional[bool] = None) -> List[NetworkInterface]:
    """
    Load all five ChordNet ensemble members from their bundled checkpoints, once.

    This is the expensive part of the pipeline (five torch.load calls over
    cache_data/*.sdict) and it is chord-dictionary independent: the chord dict
    only drives the HMM decoder built per call in recognize_with_ensemble().
    Callers that recognize chords repeatedly from a resident process should
    load the ensemble once (or hold an LVChordiaSession, which does exactly
    that) and pass it to recognize_with_ensemble() per call.

    Args:
        use_gpu: The resolved GPU flag NetworkBehavior consumes -- True forces
            GPU, False forces CPU, None preserves auto-detection. Produce it
            from a caller-facing device string with
            device_utils.resolve_use_gpu().

    Returns:
        The five loaded NetworkInterface ensemble members, in MODEL_NAMES order.
    """
    return [
        NetworkInterface(ChordNet(None, use_gpu=use_gpu), model_name, load_checkpoint=False)
        for model_name in MODEL_NAMES
    ]


def recognize_with_ensemble(
    ensemble: List[NetworkInterface],
    audio_path: str,
    chord_dict_name: str = 'submission',
) -> List[Dict[str, Union[float, str]]]:
    """
    Run chord recognition on one audio file using an already-loaded ensemble.

    This is the per-call part of the pipeline: build the HMM decoder for the
    chosen chord dictionary (cheap -- reads one bundled text template), compute
    CQT features, run the five preloaded ensemble members, average their
    per-frame probabilities, and decode to chord segments. No model weights
    are loaded here.

    Args:
        ensemble: The loaded ensemble members from load_ensemble().
        audio_path: Path to the input audio file or URL (http://, https://)
        chord_dict_name: Chord dictionary to use ('submission', 'ismir2017', or 'full')

    Returns:
        List of chord annotations -- same shape as chord_recognition().
    """
    # Handle URL downloads
    actual_audio_path, is_temp = get_audio_path(audio_path)

    try:
        # Use importlib.resources to access the data file inside the package
        with importlib.resources.path("lv_chordia.data", f"{chord_dict_name}_chord_list.txt") as data_file:
            hmm = XHMMDecoder(template_file=str(data_file))
        entry = DataEntry()
        entry.prop.set('sr', DEFAULT_SR)
        entry.prop.set('hop_length', DEFAULT_HOP_LENGTH)
        entry.append_file(actual_audio_path, io.MusicIO, 'music')
        entry.append_extractor(CQTV2, 'cqt')
        probs = []
        for net in ensemble:
            print('Inference: %s on %s' % (net.save_name, audio_path), file=sys.stderr)
            probs.append(net.inference(entry.cqt))
        probs = [np.mean([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]
        chordlab = hmm.decode_to_chordlab(entry, probs, False)

        # Convert chordlab data to JSON format
        json_data = []
        for chord_segment in chordlab:
            json_data.append({
                'start_time': float(f"{chord_segment[0]:.2f}"),
                'end_time': float(f"{chord_segment[1]:.2f}"),
                'chord': str(chord_segment[2])
            })
        return json_data
    finally:
        # Clean up temporary file if downloaded from URL
        if is_temp:
            cleanup_temp_audio(actual_audio_path)


def chord_recognition(audio_path: str, chord_dict_name: str = 'submission', device: Optional[str] = None) -> List[Dict[str, Union[float, str]]]:
    """
    Perform chord recognition on an audio file and return results as JSON.

    Supports both local files and URLs. If a URL is provided, the audio will be
    downloaded automatically before processing.

    Args:
        audio_path: Path to the input audio file or URL (http://, https://)
        chord_dict_name: Chord dictionary to use ('submission', 'ismir2017', or 'full')
        device: Optional device override -- one of 'cpu', 'cuda', 'cuda:N', or
            'auto'. None (the default) preserves today's behavior exactly:
            auto-detect GPU via torch.cuda.device_count() > 0. 'cpu' forces
            CPU even when CUDA is available; 'cuda'/'cuda:N' force GPU,
            raising RuntimeError if no CUDA device is visible. Only
            GPU-yes/no is wired through the inference pipeline -- 'cuda:N' is
            validated but does not itself pick which physical GPU runs the
            model (see device_utils.resolve_use_gpu).

    Returns:
        List of chord annotations as dictionaries with keys:
        - start_time: Start time in seconds (float)
        - end_time: End time in seconds (float)
        - chord: Chord label (string)

    Raises:
        RuntimeError: device requests CUDA but none is visible, or requests
            an out-of-range CUDA index.
        ValueError: device is not a recognized string.

    Examples:
        >>> # Local file
        >>> results = chord_recognition("song.mp3")
        >>> print(results)
        [
            {"start_time": 0.0, "end_time": 2.5, "chord": "C:maj"},
            {"start_time": 2.5, "end_time": 5.0, "chord": "F:maj"},
            ...
        ]

        >>> # URL
        >>> results = chord_recognition("https://example.com/audio.mp3")

        >>> # Force CPU even on a CUDA-capable machine
        >>> results = chord_recognition("song.mp3", device="cpu")
    """
    # Resolve device before doing any work, so an invalid/unavailable device
    # request fails fast rather than after loading models or downloading a URL.
    use_gpu = resolve_use_gpu(device)

    # One-shot convenience path: a throwaway ensemble is loaded per call, so
    # every call pays the five torch.load calls. Callers that recognize chords
    # repeatedly should hold an LVChordiaSession (session.py) instead.
    ensemble = load_ensemble(use_gpu)
    return recognize_with_ensemble(ensemble, audio_path, chord_dict_name)


def chord_recognition_json(audio_path: str, chord_dict_name: str = 'submission', device: Optional[str] = None) -> List[Dict[str, Union[float, str]]]:
    """
    Alias for chord_recognition function for backward compatibility.

    Args:
        audio_path: Path to the input audio file
        chord_dict_name: Chord dictionary to use ('submission', 'ismir2017', or 'full')
        device: Optional device override -- see chord_recognition() for the
            accepted values and defaults.

    Returns:
        List of chord annotations as dictionaries with keys:
        - start_time: Start time in seconds (float)
        - end_time: End time in seconds (float)
        - chord: Chord label (string)
    """
    return chord_recognition(audio_path, chord_dict_name, device)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        results = chord_recognition(sys.argv[1])
        print(json.dumps(results, indent=2))
    elif len(sys.argv) == 3:
        results = chord_recognition(sys.argv[1], sys.argv[2])
        print(json.dumps(results, indent=2))
    else:
        print('Usage: chord_recognition.py path_to_audio_file [chord_dict=submission]')
        print('\tChord dict can be one of the following: full, ismir2017, submission, extended')
        exit(0)