# MIR Helper inference subset

This vendored toolkit provides single-entry audio/feature IO, lazy feature
extraction and caching, visualization helpers, and the network interface used
by LVChordia. It is part of `lv-chordia`, not a separately installed package.
Install from the repository root; `pyproject.toml` owns runtime dependencies.

The recognition path uses `DataEntry`, `io.MusicIO`, `ExtractorBase`, and
`nn.network.NetworkInterface`. `TextureBuilder` and inference/visualization IO
helpers remain available. The five bundled checkpoints and their loading
format are unchanged.

Dataset collections, dataset storage/providers, training augmentation and
losses, and evaluation adapters are excluded. Historical training examples
are available in git history, not as supported package workflows.

See the repository README for recognition and session usage, and CLAUDE.md
for the import graph and exact verification commands.
