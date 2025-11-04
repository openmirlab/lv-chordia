from .chordnet_ismir_naive import ChordNet,chord_limit,ChordNetCNN
from .mir.nn.train import NetworkInterface
from .extractors.cqt import CQTV2,SimpleChordToID
from .mir import io,DataEntry
from .extractors.xhmm_ismir import XHMMDecoder
import numpy as np
from .io_new.chordlab_io import ChordLabIO
from .settings import DEFAULT_SR,DEFAULT_HOP_LENGTH
from .audio_utils import get_audio_path, cleanup_temp_audio
import sys
import json
import os
from typing import List, Dict, Union, Optional
import importlib.resources

MODEL_NAMES = ['joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best' % i for i in range(5)]


def chord_recognition(audio_path: str, chord_dict_name: str = 'submission') -> List[Dict[str, Union[float, str]]]:
    """
    Perform chord recognition on an audio file and return results as JSON.

    Supports both local files and URLs. If a URL is provided, the audio will be
    downloaded automatically before processing.

    Args:
        audio_path: Path to the input audio file or URL (http://, https://)
        chord_dict_name: Chord dictionary to use ('submission', 'ismir2017', or 'full')

    Returns:
        List of chord annotations as dictionaries with keys:
        - start_time: Start time in seconds (float)
        - end_time: End time in seconds (float)
        - chord: Chord label (string)

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
        for model_name in MODEL_NAMES:
            net = NetworkInterface(ChordNet(None), model_name, load_checkpoint=False)
            print('Inference: %s on %s' % (model_name, audio_path), file=sys.stderr)
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


def chord_recognition_json(audio_path: str, chord_dict_name: str = 'submission') -> List[Dict[str, Union[float, str]]]:
    """
    Alias for chord_recognition function for backward compatibility.
    
    Args:
        audio_path: Path to the input audio file
        chord_dict_name: Chord dictionary to use ('submission', 'ismir2017', or 'full')
    
    Returns:
        List of chord annotations as dictionaries with keys:
        - start_time: Start time in seconds (float)
        - end_time: End time in seconds (float)
        - chord: Chord label (string)
    """
    return chord_recognition(audio_path, chord_dict_name)


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