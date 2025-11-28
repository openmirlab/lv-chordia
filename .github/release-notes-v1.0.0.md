### Initial PyPI Release

**v1.0.0** - First production-ready release of lv-chordia on PyPI

### Features

- **Large-Vocabulary Chord Recognition**: Supports hundreds of chord types including complex jazz chords
- **Ensemble Model**: 5 pre-trained networks for robust predictions
- **Multiple Chord Dictionaries**: Submission (default), ISMIR2017, and full vocabularies
- **Easy Installation**: Available via `pip install lv-chordia` or `uv add lv-chordia`
- **Command-Line Interface**: Simple CLI for quick chord recognition
- **Python API**: Clean API for integration into your projects
- **URL Support**: Automatically download and process audio from URLs
- **JSON Output**: Structured data format for easy integration

### Technical Improvements

- Fixed package build configuration to properly include model files
- Updated code to find cache_data in shared location when installed via pip
- Modern Python packaging with pyproject.toml
- Compatible with PyTorch 2.x
- GitHub Actions workflow for automated PyPI publishing

### Installation

```bash
# Using pip
pip install lv-chordia

# Using uv (recommended)
uv add lv-chordia
```

### Usage

```bash
# Command line
lv-chordia input_audio.mp3

# Python API
from lv_chordia.chord_recognition import chord_recognition
results = chord_recognition("audio.mp3")
```

### Documentation

Full documentation available in the [README](https://github.com/openmirlab/lv-chordia#readme).

### Acknowledgments

Based on the ISMIR 2019 paper "[Large-Vocabulary Chord Transcription via Chord Structure Decomposition](https://archives.ismir.net/ismir2019/paper/000078.pdf)" by Junyan Jiang, Ke Chen, Wei Li, and Gus Xia.
