# Changelog

All notable changes to this independently maintained fork are recorded here.
The upstream project history before this fork remains available in the
[upstream repository](https://github.com/haoheliu/versatile_audio_super_resolution).

## [Unreleased]

## [1.0.1] - 2026-08-25

### Changed

- Pinned the `basic` and `speech` checkpoint downloads to verified Hugging
  Face revisions for reproducible model selection.

### Fixed

- Kept one seeded low-pass filter family across every chunk and batch of a
  recording, avoiding conditioning drift at chunk boundaries.
- Returned the requested number of valid DDIM timesteps across the full
  supported range while preserving the established default schedule.
- Preserved the generated sample count after resampling instead of trimming
  output again from the source-file duration.
- Used SoundFile consistently for Gradio input and duration inspection,
  including supported non-WAV formats such as FLAC.
- Unified deterministic seed handling across the Python, CLI, and Gradio
  inference paths.

### Removed

- Removed the legacy Cog/Replicate deployment configuration, Predictor entry
  points, and their dedicated regression tests. Supported execution paths are
  now the `audiosr` CLI, Python API, and local Gradio application.

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
- PyPI distribution under the distinct `kagayoi-audiosr` name while retaining
  the existing `audiosr` import package and command.

### Changed

- Raised the supported Python range to 3.10 through 3.14.
- Use librosa 0.11 on Python 3.10/3.11 and librosa 1.x on Python 3.12 through
  3.14, allowing the newest compatible major without dropping older Python
  support.
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
- Repaired dormant CLAP audio-window inference and replaced missing optional
  training helpers with explicit unsupported-feature errors.
- Restored missing AudioMAE and latent-distribution imports so their configured
  model paths fail neither linting nor initialization with `NameError`.

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
