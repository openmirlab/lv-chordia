"""
Audio utility functions for lv-chordia.

Includes support for loading audio from local files and URLs.
"""

import os
import tempfile
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple


def is_url(path: str) -> bool:
    """
    Check if a path is a URL.

    Args:
        path: Path or URL string

    Returns:
        True if path is a URL, False otherwise
    """
    parsed = urllib.parse.urlparse(path)
    return parsed.scheme in ('http', 'https', 'ftp')


def download_audio(url: str, output_dir: Optional[str] = None) -> str:
    """
    Download audio file from URL to a temporary or specified location.

    Args:
        url: URL of the audio file
        output_dir: Optional directory to save the file. If None, uses temp directory.

    Returns:
        Path to the downloaded audio file

    Raises:
        urllib.error.URLError: If download fails
        ValueError: If URL is invalid
    """
    if not is_url(url):
        raise ValueError(f"Invalid URL: {url}")

    # Get filename from URL
    parsed_url = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed_url.path)

    # If no filename in URL, generate one
    if not filename or '.' not in filename:
        # Try to get extension from Content-Type header
        try:
            with urllib.request.urlopen(url) as response:
                content_type = response.headers.get('Content-Type', '')
                if 'audio/mpeg' in content_type or 'audio/mp3' in content_type:
                    ext = '.mp3'
                elif 'audio/wav' in content_type or 'audio/x-wav' in content_type:
                    ext = '.wav'
                elif 'audio/flac' in content_type:
                    ext = '.flac'
                else:
                    ext = '.mp3'  # Default to mp3
        except Exception:
            ext = '.mp3'  # Default fallback

        filename = f"downloaded_audio{ext}"

    # Determine output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
    else:
        # Use temporary directory
        temp_dir = tempfile.mkdtemp(prefix='lv_chordia_')
        output_path = os.path.join(temp_dir, filename)

    # Download the file
    print(f"Downloading audio from: {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to: {output_path}")

    return output_path


def get_audio_path(audio_input: str, download_dir: Optional[str] = None) -> Tuple[str, bool]:
    """
    Get the audio file path, downloading if necessary.

    Args:
        audio_input: Local file path or URL
        download_dir: Optional directory for downloaded files

    Returns:
        Tuple of (audio_path, is_temporary)
        - audio_path: Path to the audio file
        - is_temporary: True if file was downloaded and should be cleaned up

    Raises:
        FileNotFoundError: If local file doesn't exist
        urllib.error.URLError: If download fails
    """
    if is_url(audio_input):
        # Download from URL
        audio_path = download_audio(audio_input, download_dir)
        return audio_path, True
    else:
        # Local file
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        return audio_input, False


def cleanup_temp_audio(audio_path: str) -> None:
    """
    Clean up temporary audio file and its directory.

    Args:
        audio_path: Path to the temporary audio file
    """
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)

        # Remove temporary directory if empty
        parent_dir = os.path.dirname(audio_path)
        if parent_dir and 'lv_chordia_' in parent_dir:
            try:
                os.rmdir(parent_dir)
            except OSError:
                pass  # Directory not empty or other error
    except Exception as e:
        # Don't raise errors during cleanup
        print(f"Warning: Could not clean up temporary file: {e}")
