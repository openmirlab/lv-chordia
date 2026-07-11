"""
Runtime settings for the inference path.

Only constants actually consumed by chord_recognition.py live here. The
training-era dataset path constants (JAM_DATASET_PATH, BILLBOARD_DATASET_PATH,
etc., pointing at a developer's local D:/dataset/... paths) and several other
unused constants (DEFAULT_BEAT_HOP_LENGTH, DEFAULT_WIN_SIZE,
DEFAULT_CHROMA_TUPLE_SIZE, DEFAULT_CHORD_DICT) were removed during the
2026-07 inference-only cleanup: none were referenced anywhere in the codebase.

Reads: nothing (leaf module).
"""

DEFAULT_SR = 22050
DEFAULT_HOP_LENGTH = 512
