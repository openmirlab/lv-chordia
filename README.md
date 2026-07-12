# lv-chordia

**Large-Vocabulary Chord Transcription via Chord Structure Decomposition**

[![Test](https://github.com/openmirlab/lv-chordia/actions/workflows/test.yml/badge.svg)](https://github.com/openmirlab/lv-chordia/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-ee4c2c.svg)](https://pytorch.org/)
[![PyPI version](https://badge.fury.io/py/lv-chordia.svg)](https://pypi.org/project/lv-chordia/)

A high-quality chord recognition system capable of transcribing complex chord progressions from audio recordings using deep learning.

---

## Why This Exists

**lv-chordia** is an implementation of the research presented in the ISMIR
2019 paper "[Large-Vocabulary Chord Transcription via Chord Structure
Decomposition](https://archives.ismir.net/ismir2019/paper/000078.pdf)" by
Junyan Jiang, Ke Chen, Wei Li, and Gus Xia. The
[original research code](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition)
is a research-lab checkout: no `pyproject.toml`, no PyPI package, no pinned
modern dependency set, and no PyTorch 2.x compatibility -- installing and
running it means manually cloning the repo, chasing down its original
(now years-stale) dependency versions, and wiring up the pre-trained
checkpoints yourself.

lv-chordia reprovides that research as a pip/uv-installable package: a clean
Python API and CLI, PyTorch 2.x compatibility, JSON output, and an
inference-only scope with no dependency on any training dataset or
training/eval tooling. The model architectures, weights, and recognition
algorithm are unchanged from the original research.

---

## Acknowledgments

lv-chordia is based on the research published at ISMIR 2019 by the
[Music X Lab](https://www.musicxlab.com/) (at the time based at NYU Shanghai,
now at MBZUAI), in collaboration with Fudan University:

- **Junyan Jiang** -- lead author, model development
- **Ke Chen** -- algorithm design, implementation
- **Wei Li** -- data preparation, evaluation (Fudan University)
- **Gus Xia** -- research supervision, methodology (Music X Lab, NYU Shanghai)
- **Source repository**: [music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) -- the original research code this package repackages
- **Weights host**: [Google Drive folder](https://drive.google.com/drive/folders/1y5-zTFaBliymPe7uY2MZfUAsvPzwmGBL) -- the original authors' additional pre-trained ensemble variants with different label-reweighting parameters, beyond the ones bundled directly in this package (see [Scope](#scope) for how this package ships its own weights)

## Citation

If you use lv-chordia in your research, please cite the original ISMIR 2019 paper:

```bibtex
@inproceedings{jiang2019large,
  title={Large-Vocabulary Chord Transcription via Chord Structure Decomposition},
  author={Jiang, Junyan and Chen, Ke and Li, Wei and Xia, Gus},
  booktitle={Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR 2019)},
  year={2019},
  pages={644--651},
  address={Delft, The Netherlands}
}
```

---

## Features

- **Large Vocabulary**: Supports hundreds of chord types including complex jazz chords
- **High Accuracy**: Ensemble model with 5 pre-trained networks, decoded with an HMM for temporal smoothing
- **Multiple Chord Dictionaries**: Submission (default), ISMIR2017, and full vocabularies
- **URL Support**: Automatically download and process audio from URLs (HTTP, HTTPS, FTP)
- **Easy-to-Use API**: Both Python API and command-line interface
- **JSON Output**: Structured, time-aligned data format for easy integration
- **Modern PyTorch**: Compatible with PyTorch 2.x
- **GPU acceleration**: Automatic CUDA support when available

**Model performance (as reported in the ISMIR 2019 paper):**
- McGill Billboard: **~81% accuracy** (submission vocabulary)
- RWC Pop: **~78% accuracy** (submission vocabulary)
- Isophonics Beatles: **~83% accuracy** (submission vocabulary)

---

## Scope

**In scope**: inference (forward pass) only. This package ships the
pre-trained ensemble models and the code path needed to run chord
recognition on audio (`lv_chordia.chord_recognition` / the `lv-chordia`
CLI).

**Out of scope, forever**: model training or evaluation/benchmarking
scripts, and any dependency on a training dataset or dataset-preparation
tooling.

### Model weights: bundled by design (documented size-based exception)

Unlike most other openmirlab inference packages, lv-chordia does **not**
download its weights at runtime. The pre-trained ensemble (`cache_data/*.sdict`,
5 files, ~28MB total -- 5.5MB each) is committed directly to this git
repository and shipped inside the built wheel/sdist via `pyproject.toml`'s
`shared-data`/`sdist` configuration, so inference runs fully offline
immediately after `pip install lv-chordia`, with no first-run download step.

This is a deliberate, documented exception to the org's default weights
contract (constitution article 4: weights are normally downloaded at
runtime, not committed to git). The exception is size-based: at 28MB total,
these weights are small enough that bundling them costs little (comparable
to, e.g., drum-classifier-infer's bundled ~7.3MB checkpoint, a similar
documented exception for different reasons) and buys a materially simpler,
fully-offline install with no download/caching/sha256-verification
machinery to build or maintain. This was confirmed as an acceptable
exception, not a defect to migrate away from, on 2026-07-12. See
`CLAUDE.md` for the same note aimed at future contributors.

If you want ensemble variants beyond the ones bundled in this package (e.g.
different label-reweighting parameters), the original authors' additional
checkpoints are available from the [Google Drive folder](https://drive.google.com/drive/folders/1y5-zTFaBliymPe7uY2MZfUAsvPzwmGBL) linked in
Acknowledgments.

---

## Install

**Available on PyPI:** [https://pypi.org/project/lv-chordia/](https://pypi.org/project/lv-chordia/)

lv-chordia supports both **UV** (recommended, faster) and **pip** (traditional) installation methods.

### Option 1: UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a blazing-fast Python package installer and resolver.

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to existing project
uv add lv-chordia

# Or create new project with lv-chordia
uv init my-music-project
cd my-music-project
uv add lv-chordia

# Run Python with lv-chordia available
uv run python your_script.py
```

### Option 2: pip (Traditional)

```bash
# Install in current environment
pip install lv-chordia

# Or create virtual environment first (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install lv-chordia
```

---

## Quick Start

```bash
# Basic usage - outputs JSON to stdout
lv-chordia input_audio.mp3
```

```python
from lv_chordia.chord_recognition import chord_recognition

results = chord_recognition(audio_path="input_audio.mp3", chord_dict_name="submission")
print(results)
# [
#   {"start_time": 0.0, "end_time": 2.5, "chord": "C:maj"},
#   {"start_time": 2.5, "end_time": 5.0, "chord": "F:maj"},
#   ...
# ]
```

---

## Usage

### Command Line Interface

```bash
# With specific chord dictionary
lv-chordia input_audio.mp3 --chord-dict submission
lv-chordia input_audio.mp3 --chord-dict ismir2017

# Save JSON output to file
lv-chordia input_audio.mp3 > output_chords.json

# Process audio from URL (auto-download)
lv-chordia https://example.com/song.mp3
lv-chordia https://example.com/audio.wav --chord-dict ismir2017 > output.json
```

**With UV:**
```bash
uv run lv-chordia input_audio.mp3
uv run lv-chordia input_audio.mp3 --chord-dict ismir2017 > output.json

# URLs work with UV too
uv run lv-chordia https://example.com/song.mp3
```

### Python API

```python
from lv_chordia.chord_recognition import chord_recognition

# Local file
results = chord_recognition(
    audio_path="input_audio.mp3",
    chord_dict_name="submission"
)

# URL (auto-download)
results = chord_recognition(
    audio_path="https://example.com/song.mp3",
    chord_dict_name="submission"
)

# Save to file if needed
import json
with open("output_chords.json", "w") as f:
    json.dump(results, f, indent=2)
```

### URL Audio Support

lv-chordia automatically downloads and processes audio from URLs:

```python
from lv_chordia.chord_recognition import chord_recognition

# Process audio directly from URL
results = chord_recognition("https://example.com/song.mp3")

# Works with any supported audio format
results = chord_recognition("https://example.com/audio.wav")
results = chord_recognition("https://example.com/track.flac")

# The temporary file is automatically cleaned up after processing
```

**Supported URL schemes**: HTTP, HTTPS, FTP

**Supported audio formats** (via librosa): MP3, WAV, FLAC, OGG, M4A, and more

### Batch Processing

```python
from pathlib import Path
from lv_chordia.chord_recognition import chord_recognition
import json

audio_files = list(Path("audio_dir/").glob("*.mp3"))

for audio_file in audio_files:
    print(f"Processing: {audio_file.name}")
    results = chord_recognition(str(audio_file))

    output_file = audio_file.with_suffix('.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
```

---

## Output Format

The package returns chord recognition results as structured JSON data. Each chord segment is represented as a dictionary:

```json
{
  "start_time": 0.0,    // Start time in seconds
  "end_time": 2.5,      // End time in seconds
  "chord": "C:maj"      // Chord label in JAMS format
}
```

### Chord Label Format

Chord labels follow the JAMS (JSON Annotated Music Specification) format:

- **Root Note**: A-G with optional # or b (e.g., "C", "F#", "Bb")
- **Separator**: Colon ":"
- **Chord Type**: maj, min, dim, aug, 7, maj7, min7, etc.
- **Special**: "N" indicates no chord/silence

**Examples:** `C:maj` (C major), `A:min7` (A minor 7th), `F#:dim` (F# diminished), `Bb:maj7` (B-flat major 7th), `N` (no chord)

---

## Chord Dictionaries

lv-chordia supports three different chord vocabularies to balance accuracy and vocabulary size:

| Dictionary | Vocabulary Size | Description | Use Case |
|-----------|----------------|-------------|----------|
| **submission** | ~170 chords | Default vocabulary (recommended) | General purpose, best balance |
| **ismir2017** | ~25 chords | MIREX/ISMIR2017 standard | Research comparison, simpler analysis |
| **full** | ~600+ chords | Complete MARL dataset vocabulary | Jazz, complex harmony analysis |

```python
results = chord_recognition("audio.mp3", chord_dict_name="ismir2017")  # or "submission" (default) / "full"
```

```bash
lv-chordia audio.mp3 --chord-dict ismir2017
```

---

## How It Works

### Chord Structure Decomposition

The key innovation of this approach is decomposing chord recognition into three sub-tasks:

1. **Root Note Recognition**: Identifying the root note of the chord (C, D, E, etc.)
2. **Bass Note Recognition**: Identifying the bass note (for slash chords)
3. **Chord Type Recognition**: Classifying the chord quality (maj, min, 7, etc.)

This decomposition allows the model to handle rare chords not seen in training data, learn compositional structure of chords, and generalize better to complex chord vocabularies.

### Processing Pipeline

```
Audio File
    -> CQT Feature Extraction (Constant-Q Transform)
    -> Deep CNN Ensemble (5 models)
    -> Probability Fusion
    -> HMM Decoding with Chord Dictionary
    -> Chord Sequence (JSON)
```

---

## Dependencies

All core dependencies are needed by the inference path (`lv_chordia.chord_recognition` / the CLI); none are training/eval-only.

```
torch>=2.13.0         # Deep learning framework
librosa>=0.11.0       # Audio loading and CQT feature extraction
numpy>=2.2.6          # Numerical computing
h5py>=3.16.0          # HDF5 file format (model checkpoint storage backend)
pydub>=0.23.1         # Audio file manipulation
pretty_midi>=0.2.9    # MIDI file handling
joblib>=1.5.3         # Parallel computing
```

```bash
# For development
pip install lv-chordia[dev]  # Adds: pytest, black, flake8, build, twine
```

---

## Advanced Usage

### Custom Model Loading

```python
from lv_chordia.chordnet_ismir_naive import ChordNet
from lv_chordia.mir.nn.train import NetworkInterface

# Load specific model from ensemble
model_name = 'joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s0.best'
net = NetworkInterface(ChordNet(None), model_name, load_checkpoint=False)

# Use for inference
# ... (see chord_recognition.py for full implementation)
```

### Processing with GPU

```python
import torch

if torch.cuda.is_available():
    print(f"Using: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# The package automatically uses GPU when available
results = chord_recognition("audio.mp3")
```

---

## Troubleshooting

### ImportError: No module named 'lv_chordia'

```bash
# With UV
uv add lv-chordia
uv run python your_script.py

# With pip
pip install lv-chordia
python -c "import lv_chordia; print('Success!')"
```

### Model Files Not Found

The package includes pre-trained model files. If you encounter model loading errors, reinstall:

```bash
pip uninstall lv-chordia
pip install lv-chordia --no-cache-dir

# Or with UV
uv pip uninstall lv-chordia
uv add lv-chordia --refresh
```

### CUDA Out of Memory

For very long audio files, GPU memory might be insufficient:

```python
# Option 1: Force CPU mode
import torch
torch.cuda.is_available = lambda: False

# Option 2: Process shorter segments
from pydub import AudioSegment

audio = AudioSegment.from_file("long_audio.mp3")
chunk_length_ms = 30000  # 30 seconds

for i, chunk_start in enumerate(range(0, len(audio), chunk_length_ms)):
    chunk = audio[chunk_start:chunk_start + chunk_length_ms]
    chunk.export(f"chunk_{i}.mp3", format="mp3")
    results = chord_recognition(f"chunk_{i}.mp3")
```

### Audio File Format Issues

```bash
# Install ffmpeg for broader format support
sudo apt-get install ffmpeg   # Ubuntu/Debian
brew install ffmpeg           # macOS
# Windows: download from https://ffmpeg.org/
```

```python
from pydub import AudioSegment

audio = AudioSegment.from_file("input.mp3")
audio.export("input.wav", format="wav")
results = chord_recognition("input.wav")
```

---

## Requirements

- **Python**: 3.10 or later
- **PyTorch**: 2.13 or later
- **OS**: Linux, macOS, Windows
- **GPU**: Optional (CUDA-capable GPU recommended for faster processing)
- **Memory**: 4GB RAM minimum, 8GB+ recommended for long audio files

---

## Research Applications

```python
# Extract chord progressions for MIR research
results = chord_recognition("dataset/song001.mp3")
unique_chords = len(set(r['chord'] for r in results))
print(f"Harmonic complexity: {unique_chords} unique chords")
```

```python
# Batch-annotate a dataset
from pathlib import Path
import json

dataset_path = Path("music_dataset/")
output_path = Path("annotations/")
output_path.mkdir(exist_ok=True)

for audio_file in dataset_path.glob("*.mp3"):
    results = chord_recognition(str(audio_file))
    output_file = output_path / f"{audio_file.stem}_chords.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
```

### Related Work

The research builds upon and extends several prior works in chord recognition:

- **MIREX Chord Recognition**: Annual evaluation campaign for chord recognition systems
- **JAMS Format**: JSON Annotated Music Specification for music annotations
- **CQT Features**: Constant-Q Transform for music analysis

---

## Development

### Setting Up Development Environment

```bash
# Clone the repository (if working from source)
git clone https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition.git
cd ISMIR2019-Large-Vocabulary-Chord-Recognition

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install in development mode
uv pip install -e ".[dev]"
```

**With pip:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Building the Package

```bash
uv build
# Or: python -m build
ls -lh dist/
```

### Publishing to PyPI

```bash
uv add twine
uv build
twine upload dist/*
# Or to TestPyPI first: twine upload --repository testpypi dist/*
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=lv_chordia
```

The test suite includes an import smoke test (the package must not depend on
any training/eval module), unit tests for `audio_utils`, and a regression
test that runs the CLI against the tracked `test_data/yellow.wav` fixture and
asserts the chord-recognition JSON output is byte-identical to a golden
fixture -- the accuracy gate for any refactor of the inference path.

---

## License

**MIT License**

Copyright (c) 2019 Junyan Jiang, Ke Chen, Wei Li, Gus Xia (Original Research)
Copyright (c) 2025 Package Maintainers (Package Maintenance)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

See [LICENSE](LICENSE) for full details.

---

## Support

### Getting Help

- **Documentation**: Read this README and code examples
- **Issues**: Report bugs or ask questions on [GitHub Issues](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/issues)
- **Discussions**: Join discussions about chord recognition and MIR

### Contributing

Contributions are welcome! This package aims to maintain the original research quality while improving usability.

1. **Bug Reports**: Open an issue with details about the problem
2. **Feature Requests**: Suggest improvements or new features
3. **Pull Requests**: Submit PRs for bug fixes or enhancements
4. **Documentation**: Help improve documentation and examples

Please maintain compatibility with original research results, add tests for new features, and follow existing code style.

### Common Questions

**Q: How accurate is the chord recognition?**
A: The system achieves ~80% accuracy on benchmark datasets (Billboard, RWC Pop, Beatles), which is state-of-the-art for large-vocabulary chord recognition.

**Q: Can it recognize jazz chords?**
A: Yes! Use the "full" dictionary for extensive jazz chord support including 9th, 11th, 13th chords, and alterations.

**Q: How fast is the processing?**
A: On GPU: ~10-20x real-time. On CPU: ~2-5x real-time. A 3-minute song takes about 10-30 seconds on modern hardware.

**Q: Can I use this commercially?**
A: Yes, the MIT license allows commercial use. Please cite the original research paper.

---

**Made for the music and research community, built on the research of Junyan Jiang, Ke Chen, Wei Li, and Gus Xia (ISMIR 2019)**
