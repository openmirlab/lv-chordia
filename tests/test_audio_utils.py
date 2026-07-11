"""Cheap unit tests for lv_chordia.audio_utils (no model/audio decoding needed)."""

import os

import pytest

from lv_chordia.audio_utils import cleanup_temp_audio, get_audio_path, is_url


@pytest.mark.parametrize(
    "path,expected",
    [
        ("http://example.com/song.mp3", True),
        ("https://example.com/song.mp3", True),
        ("ftp://example.com/song.mp3", True),
        ("song.mp3", False),
        ("/abs/path/song.wav", False),
        ("C:\\Users\\song.wav", False),
        ("", False),
    ],
)
def test_is_url(path, expected):
    assert is_url(path) is expected


def test_get_audio_path_local_file_exists(tmp_path):
    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"not-really-audio-but-exists")

    path, is_temp = get_audio_path(str(audio_file))

    assert path == str(audio_file)
    assert is_temp is False


def test_get_audio_path_local_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.wav"

    with pytest.raises(FileNotFoundError):
        get_audio_path(str(missing))


def test_cleanup_temp_audio_removes_file(tmp_path):
    audio_file = tmp_path / "lv_chordia_tmp" / "clip.wav"
    audio_file.parent.mkdir()
    audio_file.write_bytes(b"data")

    cleanup_temp_audio(str(audio_file))

    assert not audio_file.exists()


def test_cleanup_temp_audio_missing_file_does_not_raise(tmp_path):
    missing = tmp_path / "already_gone.wav"
    # Should not raise even though the file was never created.
    cleanup_temp_audio(str(missing))
