# Changelog

All notable changes to this independently maintained fork are recorded here.
The upstream project history before this fork remains available in the
[upstream repository](https://github.com/haoheliu/versatile_audio_super_resolution).

## [1.0.0] - 2026-08-24

This is the first independently maintained release of
`1llum1n4t1s/KG.versatile_audio_super_resolution`.

### Added

- A prominent fork notice and repository-specific installation guidance.
- Native loading of local Safetensors checkpoints.
- In-memory batch inference for Gradio chunk processing.
- Deterministic seed control in the Gradio interface.
- A single-model Gradio cache that avoids rebuilding the same approximately
  6.18 GB model for every request.
- Python package build metadata, a console entry point, Dependabot
  configuration, and regression tests for CLI, pipeline, Gradio, Cog, audio
  loading, and output handling.

### Changed

- Raised the supported Python range to 3.10 through 3.14.
- Updated the normal installation path to current PyTorch, Transformers,
  Gradio, timm, NumPy, SoundFile, and related dependencies.
- Replaced the unmaintained `progressbar` package with `progressbar2`.
- Reworked CLI and package imports so `audiosr --help` does not initialize the
  model stack.
- Switched ordinary audio decoding, duration inspection, and output writing to
  SoundFile.
- Made the fork repository the package URL, support location, and installation
  source.

### Fixed

- Preserved mono, stereo, and other channel layouts through standard,
  long-audio, Gradio, and output paths.
- Corrected integer sample padding, short final chunks, overlap boundaries, and
  exact output-length trimming.
- Prevented NaN and infinity propagation for silent or malformed audio.
- Bypassed unsafe low-pass cutoff values at zero or Nyquist.
- Validated DDIM steps at the CLI and sampler boundaries.
- Removed unused CLAP construction and import-time RoBERTa initialization from
  the super-resolution path.
- Corrected the DDPM sample channel reference and modernized timm imports.

### Security

- PyTorch checkpoint loading now uses `weights_only=True`.
- Safetensors files are loaded without pickle deserialization.
- Transformers was updated to the 5.x line to resolve known vulnerabilities in
  the previous 4.x dependency range.

### Known limitations

- The included Cog configuration still targets CUDA 11.7 and PyTorch 2.0.1;
  upgrading that GPU runtime requires a separate deployment-runtime migration.
- The `basic` and `speech` checkpoints are each approximately 6.18 GB.
- DirectML is not supported, and Apple MPS availability depends on the installed
  PyTorch build and hardware.
