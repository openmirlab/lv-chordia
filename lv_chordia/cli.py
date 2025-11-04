"""
Command-line interface for lv-chordia package.
"""

import argparse
import sys
import json
from pathlib import Path

from .chord_recognition import chord_recognition


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Large-Vocabulary Chord Transcription via Chord Structure Decomposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local files
  lv-chordia example.mp3
  lv-chordia audio.wav --chord-dict submission
  lv-chordia music.mp3 --chord-dict ismir2017
  lv-chordia example.mp3 > output.json

  # URLs (auto-download)
  lv-chordia https://example.com/song.mp3
  lv-chordia https://example.com/audio.wav --chord-dict ismir2017
        """
    )

    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the input audio file or URL (http://, https://)"
    )
    
    parser.add_argument(
        "--chord-dict",
        dest="chord_dict",
        default="submission",
        choices=["submission", "ismir2017", "full"],
        help="Chord dictionary to use for decoding (default: submission)"
    )
    
    args = parser.parse_args()

    # Import here to check if it's a URL
    from .audio_utils import is_url

    # Validate input file exists (skip for URLs)
    if not is_url(args.audio_file) and not Path(args.audio_file).exists():
        print(f"Error: Audio file '{args.audio_file}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        # Perform chord recognition and output JSON
        results = chord_recognition(args.audio_file, args.chord_dict)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error during chord recognition: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 