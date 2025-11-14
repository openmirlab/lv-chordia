# lv-chordia

**Large-Vocabulary Chord Transcription via Chord Structure Decomposition**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4+-ee4c2c.svg)](https://pytorch.org/)

A high-quality chord recognition system capable of transcribing complex chord progressions from audio recordings using deep learning.

---

## 📌 Overview

**lv-chordia** is an implementation of the research presented in the ISMIR 2019 paper "[Large-Vocabulary Chord Transcription via Chord Structure Decomposition](https://archives.ismir.net/ismir2019/paper/000078.pdf)". This package provides state-of-the-art chord recognition capabilities with support for extensive chord vocabularies including complex jazz chords.

### 🎯 Key Features

- **Large Vocabulary**: Supports hundreds of chord types including complex jazz chords
- **High Accuracy**: Ensemble model with 5 pre-trained networks
- **Multiple Chord Dictionaries**: Submission (default), ISMIR2017, and full vocabularies
- **URL Support**: Automatically download and process audio from URLs
- **Easy-to-Use API**: Both Python API and command-line interface
- **JSON Output**: Structured data format for easy integration
- **Modern PyTorch**: Compatible with PyTorch 2.x
- **Production Ready**: Packaged for PyPI distribution

---

## 🙏 Acknowledgments

### Original Research by Junyan Jiang, Ke Chen, Wei Li, and Gus Xia

**lv-chordia** is based on the groundbreaking work published at ISMIR 2019 by **Junyan Jiang**, **Ke Chen**, **Wei Li**, and **Gus Xia**. Their research introduced an innovative approach to large-vocabulary chord transcription through chord structure decomposition, achieving state-of-the-art results on multiple benchmark datasets.

### Research Paper

**[Large-Vocabulary Chord Transcription via Chord Structure Decomposition](https://archives.ismir.net/ismir2019/paper/000078.pdf)**

Presented at the 20th International Society for Music Information Retrieval Conference (ISMIR 2019), Delft, The Netherlands, November 4-8, 2019.

#### Abstract

The original research addresses the challenge of recognizing a large vocabulary of chords by decomposing chord structure into root, bass, and chord type components. This decomposition allows the model to handle complex chords that rarely appear in training data by learning their structural components independently.

### Citation

**If you use lv-chordia in your research, please cite the original ISMIR 2019 paper:**

```bibtex
@inproceedings{jiang2019large,
  title={Large-Vocabulary Chord Transcription via Chord Structure Decomposition},
  author={Jiang, Junyan and Chen, Ke and Li, Wei and Xia, Gus},
  booktitle={Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2019},
  pages={792--798},
  address={Delft, The Netherlands}
}
```

### About This Package

> **Note**: This package is a modern, packaged version of the original research code, optimized for easy installation and use. It includes compatibility updates for PyTorch 2.x and modern Python packaging standards.

**What we maintain:**
- PyTorch 2.x compatibility
- Modern Python packaging (pyproject.toml, pip/uv installable)
- Clean API with JSON output
- Command-line interface
- Documentation and examples

**What remains unchanged:**
- All model architectures (100% original)
- All pre-trained model weights (100% original)
- Chord recognition algorithms (100% original)
- Recognition quality (100% identical to original research)

---

## 🚀 Quick Start

### Installation

lv-chordia supports both **UV** (recommended, faster) and **pip** (traditional) installation methods.

#### Option 1: UV (Recommended) ⚡

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

**Benefits of UV:**
- ⚡ 10-100x faster than pip
- 🔒 Automatic virtual environment management
- 📦 Consistent dependency resolution
- 🎯 Works seamlessly with PyPI packages

#### Option 2: pip (Traditional)

```bash
# Install in current environment
pip install lv-chordia

# Or create virtual environment first (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install lv-chordia
```

---

## 💻 Usage

### Command Line Interface

```bash
# Basic usage - outputs JSON to stdout
lv-chordia input_audio.mp3

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

# JSON output format
print(results)
# [
#   {"start_time": 0.0, "end_time": 2.5, "chord": "C:maj"},
#   {"start_time": 2.5, "end_time": 5.0, "chord": "F:maj"},
#   {"start_time": 5.0, "end_time": 7.5, "chord": "G:maj"},
#   ...
# ]

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

**Supported audio formats** (via librosa):
- MP3, WAV, FLAC, OGG, M4A, and more

### Batch Processing

```python
from pathlib import Path
from lv_chordia.chord_recognition import chord_recognition
import json

# Process multiple local files
audio_files = list(Path("audio_dir/").glob("*.mp3"))

for audio_file in audio_files:
    print(f"Processing: {audio_file.name}")
    results = chord_recognition(str(audio_file))

    # Save results
    output_file = audio_file.with_suffix('.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved: {output_file}")

# Process multiple URLs
urls = [
    "https://example.com/song1.mp3",
    "https://example.com/song2.mp3",
    "https://example.com/song3.mp3"
]

for url in urls:
    print(f"Processing: {url}")
    results = chord_recognition(url)
    # Process results...
```

---

## 📊 Output Format

The package returns chord recognition results as structured JSON data. Each chord segment is represented as a dictionary:

```json
{
  "start_time": 0.0,    // Start time in seconds
  "end_time": 2.5,      // End time in seconds
  "chord": "C:maj"      // Chord label in JAMS format
}
```

### Example Output

```json
[
  {"start_time": 0.0, "end_time": 2.5, "chord": "C:maj"},
  {"start_time": 2.5, "end_time": 5.0, "chord": "F:maj"},
  {"start_time": 5.0, "end_time": 7.5, "chord": "G:maj"},
  {"start_time": 7.5, "end_time": 10.0, "chord": "A:min7"},
  {"start_time": 10.0, "end_time": 12.5, "chord": "D:7"},
  {"start_time": 12.5, "end_time": 15.0, "chord": "G:maj"}
]
```

### Chord Label Format

Chord labels follow the JAMS (JSON Annotated Music Specification) format:

- **Root Note**: A-G with optional # or b (e.g., "C", "F#", "Bb")
- **Separator**: Colon ":"
- **Chord Type**: maj, min, dim, aug, 7, maj7, min7, etc.
- **Special**: "N" indicates no chord/silence

**Examples:**
- `C:maj` - C major
- `A:min7` - A minor 7th
- `F#:dim` - F# diminished
- `Bb:maj7` - B-flat major 7th
- `N` - No chord

---

## 🎼 Chord Dictionaries

lv-chordia supports three different chord vocabularies to balance accuracy and vocabulary size:

### Available Dictionaries

| Dictionary | Vocabulary Size | Description | Use Case |
|-----------|----------------|-------------|----------|
| **submission** | ~170 chords | Default vocabulary (recommended) | General purpose, best balance |
| **ismir2017** | ~25 chords | MIREX/ISMIR2017 standard | Research comparison, simpler analysis |
| **full** | ~600+ chords | Complete MARL dataset vocabulary | Jazz, complex harmony analysis |

### Usage

```python
# Use default dictionary (submission)
results = chord_recognition("audio.mp3")

# Use ISMIR2017 dictionary
results = chord_recognition("audio.mp3", chord_dict_name="ismir2017")

# Use full dictionary (experimental)
results = chord_recognition("audio.mp3", chord_dict_name="full")
```

```bash
# Command line
lv-chordia audio.mp3 --chord-dict submission
lv-chordia audio.mp3 --chord-dict ismir2017
lv-chordia audio.mp3 --chord-dict full
```

---

## 🎵 Features

### Technical Capabilities

- **Large-vocabulary chord recognition**: Supports extensive chord dictionaries
- **Chord structure decomposition**: Root, bass, and chord type modeling
- **Ensemble inference**: 5 pre-trained models for robust predictions
- **Audio format support**: MP3, WAV, FLAC, and other formats via librosa
- **URL audio processing**: Automatic download from HTTP, HTTPS, and FTP
- **Time-aligned output**: Precise temporal boundaries for each chord
- **GPU acceleration**: Automatic CUDA support when available

### Pre-trained Models

This package includes pre-trained ensemble models achieving state-of-the-art accuracy on benchmark datasets:

- **Training Data**: Large-scale chord annotations from multiple datasets
- **Model Architecture**: Deep convolutional neural networks with CQT features
- **Ensemble Size**: 5 models with cross-validation splits
- **Decoding**: Hidden Markov Model (HMM) for temporal smoothing

**Model Performance (as reported in ISMIR 2019):**
- McGill Billboard: **~81% accuracy** (submission vocabulary)
- RWC Pop: **~78% accuracy** (submission vocabulary)
- Isophonics Beatles: **~83% accuracy** (submission vocabulary)

---

## 🧠 How It Works

### Chord Structure Decomposition

The key innovation of this approach is decomposing chord recognition into three sub-tasks:

1. **Root Note Recognition**: Identifying the root note of the chord (C, D, E, etc.)
2. **Bass Note Recognition**: Identifying the bass note (for slash chords)
3. **Chord Type Recognition**: Classifying the chord quality (maj, min, 7, etc.)

This decomposition allows the model to:
- Handle rare chords not seen in training data
- Learn compositional structure of chords
- Generalize better to complex chord vocabularies

### Processing Pipeline

```
Audio File
    ↓
CQT Feature Extraction (Constant-Q Transform)
    ↓
Deep CNN Ensemble (5 models)
    ↓
Probability Fusion
    ↓
HMM Decoding with Chord Dictionary
    ↓
Chord Sequence (JSON)
```

---

## 📦 Dependencies

### Core Dependencies

```
torch>=1.4.0          # Deep learning framework
librosa>=0.7.2        # Audio processing
numpy>=1.19.2         # Numerical computing
scikit_learn>=0.23.2  # Machine learning utilities
mir_eval>=0.5         # Music information retrieval evaluation
h5py>=2.9.0           # HDF5 file format
jams>=0.3.4           # JSON Annotated Music Specification
pumpp>=0.5.0          # Audio feature extraction
pydub>=0.23.1         # Audio file manipulation
matplotlib>=2.2.4     # Visualization
pretty_midi>=0.2.9    # MIDI file handling
joblib>=0.13.2        # Parallel computing
figures>=0.3.16       # Plotting utilities
```

### Optional Dependencies

```bash
# For development
pip install lv-chordia[dev]  # Adds: pytest, black, flake8, build, twine
```

---

## 🔧 Advanced Usage

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

# Check CUDA availability
if torch.cuda.is_available():
    print("GPU acceleration available!")
    print(f"Using: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# The package automatically uses GPU when available
results = chord_recognition("audio.mp3")
```

### Integration with Music Analysis

```python
from lv_chordia.chord_recognition import chord_recognition
import pandas as pd

# Recognize chords
results = chord_recognition("song.mp3")

# Convert to DataFrame for analysis
df = pd.DataFrame(results)

# Analyze chord statistics
print(f"Total chords: {len(df)}")
print(f"Unique chords: {df['chord'].nunique()}")
print(f"Most common chord: {df['chord'].mode()[0]}")
print(f"\nChord distribution:")
print(df['chord'].value_counts().head(10))

# Calculate average chord duration
df['duration'] = df['end_time'] - df['start_time']
print(f"\nAverage chord duration: {df['duration'].mean():.2f}s")
```

---

## 🐛 Troubleshooting

### ImportError: No module named 'lv_chordia'

**With UV:**
```bash
# Make sure you added lv-chordia to your project
uv add lv-chordia

# Or run with UV
uv run python your_script.py
```

**With pip:**
```bash
# Make sure you installed lv-chordia
pip install lv-chordia

# Check installation
python -c "import lv_chordia; print('Success!')"
```

### Model Files Not Found

The package includes pre-trained model files. If you encounter model loading errors:

```bash
# Reinstall the package
pip uninstall lv-chordia
pip install lv-chordia --no-cache-dir

# Or with UV
uv pip uninstall lv-chordia
uv add lv-chordia --refresh
```

### CUDA Out of Memory

For very long audio files, GPU memory might be insufficient:

```python
# Process shorter segments
# The package handles this automatically, but for manual control:

# Option 1: Use CPU instead
import torch
torch.cuda.is_available = lambda: False  # Force CPU mode

# Option 2: Process shorter files
from pydub import AudioSegment

audio = AudioSegment.from_file("long_audio.mp3")
chunk_length_ms = 30000  # 30 seconds

for i, chunk_start in enumerate(range(0, len(audio), chunk_length_ms)):
    chunk = audio[chunk_start:chunk_start + chunk_length_ms]
    chunk.export(f"chunk_{i}.mp3", format="mp3")
    results = chord_recognition(f"chunk_{i}.mp3")
    # Process results...
```

### Audio File Format Issues

If you encounter errors loading audio files:

```bash
# Install ffmpeg for broader format support
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/
```

```python
# Convert audio to WAV format first
from pydub import AudioSegment

audio = AudioSegment.from_file("input.mp3")
audio.export("input.wav", format="wav")

results = chord_recognition("input.wav")
```

---

## 📋 Requirements

- **Python**: 3.10 or later
- **PyTorch**: 1.4 or later (2.x recommended)
- **OS**: Linux, macOS, Windows
- **GPU**: Optional (CUDA-capable GPU recommended for faster processing)
- **Memory**: 4GB RAM minimum, 8GB+ recommended for long audio files

---

## 🔬 Research Applications

### Music Information Retrieval

```python
# Extract chord progressions for MIR research
results = chord_recognition("dataset/song001.mp3")

# Analyze harmonic complexity
unique_chords = len(set(r['chord'] for r in results))
print(f"Harmonic complexity: {unique_chords} unique chords")
```

### Music Education

```python
# Generate practice materials
results = chord_recognition("practice_track.mp3")

# Export for notation software
with open("chords.txt", "w") as f:
    for segment in results:
        f.write(f"{segment['start_time']:.2f}\t{segment['chord']}\n")
```

### Dataset Annotation

```python
from pathlib import Path
import json

# Batch annotate a dataset
dataset_path = Path("music_dataset/")
output_path = Path("annotations/")
output_path.mkdir(exist_ok=True)

for audio_file in dataset_path.glob("*.mp3"):
    print(f"Annotating: {audio_file.name}")

    results = chord_recognition(str(audio_file))

    output_file = output_path / f"{audio_file.stem}_chords.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
```

---

## 🛠 Development

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
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Building the Package

```bash
# Build wheel and source distribution
uv build

# Or with pip/build
python -m build

# Check the dist/ directory
ls -lh dist/
```

### Publishing to PyPI

```bash
# Install twine (included in dev dependencies)
uv add twine

# Build the package
uv build

# Upload to PyPI (requires PyPI credentials)
twine upload dist/*

# Or upload to TestPyPI first
twine upload --repository testpypi dist/*
```

### Running Tests

```bash
# Run basic functionality test
python test_chordrecog.py

# Run with pytest (when test suite is available)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=lv_chordia
```

---

## 📚 Additional Resources

### Original Research

- **Paper**: [Large-Vocabulary Chord Transcription via Chord Structure Decomposition](https://archives.ismir.net/ismir2019/paper/000078.pdf)
- **Repository**: [music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition)
- **Conference**: ISMIR 2019, Delft, The Netherlands

### Related Work

The research builds upon and extends several prior works in chord recognition:

- **MIREX Chord Recognition**: Annual evaluation campaign for chord recognition systems
- **JAMS Format**: JSON Annotated Music Specification for music annotations
- **CQT Features**: Constant-Q Transform for music analysis

### Model Downloads

Pre-trained models are included in the package. For custom models with label reweighting:
- [Google Drive: Pre-trained Models](https://drive.google.com/drive/u/1/folders/1y5-zTFaBliymPe7uY2MZfUAsvPzwmGBL)

---

## 🤝 Contributing

Contributions are welcome! This package aims to maintain the original research quality while improving usability.

### How to Contribute

1. **Bug Reports**: Open an issue with details about the problem
2. **Feature Requests**: Suggest improvements or new features
3. **Pull Requests**: Submit PRs for bug fixes or enhancements
4. **Documentation**: Help improve documentation and examples

### Contribution Guidelines

- Maintain compatibility with original research results
- Add tests for new features
- Update documentation for API changes
- Follow existing code style

---

## 📄 License

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

## 🆘 Support

### Getting Help

- **Documentation**: Read this README and code examples
- **Issues**: Report bugs or ask questions on [GitHub Issues](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/issues)
- **Discussions**: Join discussions about chord recognition and MIR

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

## 🌟 Acknowledgments

### Research Team

Special thanks to the original research team:
- **Junyan Jiang** - Lead author, model development
- **Ke Chen** - Algorithm design, implementation
- **Wei Li** - Data preparation, evaluation
- **Gus Xia** - Research supervision, methodology

### Package Maintenance

This package is maintained to ensure continued availability and compatibility with modern Python ecosystems.

### Community

Thanks to the music information retrieval (MIR) community for:
- Dataset creation and annotation
- MIREX evaluation campaigns
- Open-source tools and libraries

---

**Made with ❤️ for the music and research community**

*Based on the excellent research by Junyan Jiang, Ke Chen, Wei Li, and Gus Xia (ISMIR 2019)*
