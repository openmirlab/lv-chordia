"""
lv-chordia: Large-Vocabulary Chord Transcription via Chord Structure Decomposition

A Python package for chord recognition and transcription from audio files.

Based on the ISMIR 2019 paper by Junyan Jiang, Ke Chen, Wei Li, and Gus Xia.
https://archives.ismir.net/ismir2019/paper/000078.pdf

This package provides state-of-the-art chord recognition capabilities with support
for extensive chord vocabularies including complex jazz chords.
"""

__version__ = "1.0.0"
__author__ = "Junyan Jiang, Ke Chen, Wei Li, Gus Xia"
__maintainer__ = "Package Maintainers"
__license__ = "MIT"
__url__ = "https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition"

# Import main modules for easy access. This package is inference-only: no
# training/eval modules (e.g. the former `datasets` module) are imported here,
# and import errors are not swallowed -- a broken dependency should fail loudly.
from . import chord_recognition
from . import extractors
from . import mir

# Import the main function for easy access
from .chord_recognition import chord_recognition

__all__ = [
    "chord_recognition",
    "extractors",
    "mir",
    "__version__",
    "__author__",
    "__maintainer__",
    "__license__",
    "__url__",
]
