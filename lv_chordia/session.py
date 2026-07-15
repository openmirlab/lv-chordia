"""Independent lifecycle facade for chord recognition."""
from __future__ import annotations
from pathlib import Path
from .chord_recognition import chord_recognition

class LVChordiaSession:
    """Coordinates one chord-recognition runtime without global model ownership."""
    def __init__(self, *, chord_dict_name="submission"):
        self.chord_dict_name = chord_dict_name
        self._state = "created"

    @property
    def status(self):
        return self._state

    def load(self):
        if self._state == "closed":
            raise RuntimeError("session is closed")
        # Models are intentionally loaded lazily by the legacy pipeline on infer.
        self._state = "ready"
        return self

    def infer(self, audio_path: str | Path):
        if self._state != "ready":
            raise RuntimeError("call load() before infer()")
        return chord_recognition(str(audio_path), self.chord_dict_name)

    def release(self):
        self._state = "released" if self._state != "closed" else self._state
        return self

    def close(self):
        self.release()
        self._state = "closed"

    def cache_info(self):
        root = Path(__file__).parent.parent / "cache_data"
        return {"path": str(root), "exists": root.exists(), "artifacts": sorted(p.name for p in root.glob("*.sdict"))}

    def __enter__(self):
        return self.load()
    def __exit__(self, *_):
        self.close()
