"""
Load-once lifecycle facade for chord recognition.

LVChordiaSession loads the five-model ChordNet ensemble exactly once (at
load(), for a resolved device) and reuses it across infer() calls -- the API
for callers that keep a process resident and recognize chords repeatedly
(e.g. a long-running provider service). The chord dictionary stays a per-call
choice: it only drives the HMM decoder built inside recognize_with_ensemble(),
never the ensemble load, so one loaded session serves any chord vocabulary.
release() drops the ensemble references; close() ends the session for good.
The one-shot chord_recognition() remains the throwaway-ensemble convenience
path.

Reads: chord_recognition.py (load_ensemble, recognize_with_ensemble),
device_utils.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .chord_recognition import load_ensemble, recognize_with_ensemble
from .config import resolve_checkpoint_paths
from .device_utils import resolve_device


class LVChordiaSession:
    """Hold the loaded five-model ensemble across chord-recognition calls."""

    def __init__(self, *, chord_dict_name: str = "submission", device: Optional[str] = None):
        """
        Args:
            chord_dict_name: Default chord dictionary for infer() calls
                ('submission', 'ismir2017', or 'full'); overridable per call.
            device: Optional device override -- one of 'cpu', 'cuda', 'cuda:N', 'mps',
                or 'auto'. None (the default) preserves auto-detection, exactly
                as chord_recognition()'s own device parameter documents.
        """
        self.chord_dict_name = chord_dict_name
        self.device = device
        self._ensemble = None
        self._state = "created"

    @property
    def status(self):
        return self._state

    @property
    def loaded(self) -> bool:
        """Whether the ensemble is currently resident in memory."""
        return self._ensemble is not None

    def load(self):
        """
        Resolve the device and load the five ensemble members onto it, once.
        Idempotent while the ensemble is resident; after release(), load()
        reloads. Raises RuntimeError/ValueError for an unavailable or invalid
        explicit device (device_utils.resolve_use_gpu's fail-loudly contract),
        and RuntimeError if the session is closed.
        """
        if self._state == "closed":
            raise RuntimeError("session is closed")
        if self._ensemble is None:
            try:
                resolved_device = resolve_device(self.device)
                use_gpu = None if resolved_device is None else resolved_device.type == "cuda"
                self._ensemble = load_ensemble(use_gpu, device=resolved_device)
            except Exception:
                self._state = "failed"
                raise
        self._state = "ready"
        return self

    def infer(self, audio_path: str | Path, chord_dict_name: Optional[str] = None):
        """
        Run chord recognition on one audio file using the preloaded ensemble.

        Args:
            audio_path: Path to the input audio file or URL.
            chord_dict_name: Per-call chord-dictionary override; None uses the
                session's default. Only the HMM decoder depends on this -- no
                model weights are (re)loaded whichever dictionary is chosen.

        Returns:
            List of chord annotations -- identical to chord_recognition() for
            the same file, dictionary, and device.
        """
        if self._state != "ready":
            raise RuntimeError("call load() before infer()")
        return recognize_with_ensemble(
            self._ensemble,
            str(audio_path),
            self.chord_dict_name if chord_dict_name is None else chord_dict_name,
        )

    def release(self):
        """Drop the ensemble references so they can be garbage-collected."""
        self._ensemble = None
        self._state = "released" if self._state != "closed" else self._state
        return self

    def close(self):
        """Release the ensemble and end the session; load() is refused afterwards."""
        self.release()
        self._state = "closed"
        return self

    def cache_info(self):
        root, entries = resolve_checkpoint_paths()
        return {
            "path": str(root),
            "exists": root.exists(),
            "cached": all(entry["cached"] for entry in entries),
            "artifacts": [entry["name"] for entry in entries],
            "entries": entries,
        }

    def __enter__(self):
        return self.load()

    def __exit__(self, *_):
        self.close()
