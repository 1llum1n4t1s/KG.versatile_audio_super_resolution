# Changelog

All notable changes to this independently maintained fork are recorded here.
The upstream project history before this fork remains available in the
[upstream repository](https://github.com/haoheliu/versatile_audio_super_resolution).

## [Unreleased]

## [1.0.4] - 2026-09-01

### Breaking changes

- Dropped Python 3.10 through 3.13. Install and use Python 3.14 before
  upgrading this independently maintained fork.

### Changed

- Standardized on Librosa 1.x and removed the legacy Python-version markers.

### Fixed

- Enforced tensor-only checkpoint loading across model, vocoder, CLAP, and
  training-resume paths, including native Safetensors loading.
- Aligned the Python API and Gradio defaults with the CLI's 50 DDIM steps.
- Preserved in-range output levels in Gradio while still scaling down peaks
  above full scale.
- Released MPS caches during model replacement and OOM retry as well as CUDA
  caches.
- Made quadratic DDIM schedules strictly increasing without changing the
  requested step count.

## [1.0.3] - 2026-09-01

### Added

- Added a `dpmpp2m` sampler that integrates the probability-flow ODE with
  multistep DPM-Solver++, reaching a comparable result in far fewer network
  evaluations than the ancestral schedule. At first order it is algebraically
  identical to DDIM with `ddim_eta=0.0`, which the regression tests assert.
- Exposed `sampler` and `ddim_eta` on `super_resolution`,
  `super_resolution_long_audio`, `super_resolution_batch`, and the CLI, so the
  sampling algorithm and how stochastic each update is can be selected per run.
  Both are validated at the public entry points and at the sampler boundary.
- Added a `trailing` timestep spacing, selectable through `discretize` on the
  public entry points and `--discretize` on the CLI. The established `uniform`
  spacing reaches the noisiest timestep only when the step count does not divide
  the training schedule; for exact divisors it stops short, which leaves a
  signal component that sampling from pure noise never accounts for. The default
  remains `uniform`.
- Added `tools/benchmark_samplers.py`, which degrades a high-quality reference,
  restores it with each requested configuration, and reports wall-clock cost
  next to log-spectral distance and band energy. Scoring covers only the band
  the reference can judge, energy generated beyond the reference's own bandwidth
  is reported separately rather than counted as error, and the run warns when no
  configuration beats the degraded input. Every restoration is written out so
  the same run can feed a blind listening comparison.
- Added `restore_high_rate`, and `--preserve_input_rate` on the CLI, which
  return the restoration at the source file's own rate instead of at 48 kHz and
  splice the source's content above 24 kHz back in. The model cannot reach past
  24 kHz, so this preserves what the source carried rather than adding
  bandwidth; on real recordings that band usually holds only the noise floor.
  The established entry points and their 48 kHz contract are unchanged. The
  stock output suffix names the rate the file was actually written at, so a
  preserved 96 kHz result is no longer labelled `48K`; an explicit `--suffix`
  is left alone.

- Added a learned output calibration, off by default. Measured against
  full-band references the model puts 10 to 28 dB too much energy into the
  generated band, worst where the source is quiet, so pauses come back as hiss.
  A small MLP trained on full-band recordings and their band-limited versions
  (`tools/train_calibration.py`) predicts the envelope each fractional-octave
  band should have from the source's own envelope, and `--calibration` rescales
  the restoration toward that prediction inside clamped, smoothed gains. It
  redistributes energy; it cannot invent content, and it never touches the band
  the source itself carries. Inside sustained pauses — stretches the source
  itself shows as silent, well below its loud frames, for around 200 ms or
  more — the lower gain clamp opens from -24 dB toward -60 dB so the hiss the
  model lays over silence can be pushed further down; material without pauses
  passes through unchanged. The package ships a ready calibration at
  `audiosr.bundled_calibration_path()`, trained across piano, synthesizer,
  orchestra, and speech (つくよみちゃんコーパス CV.夢前黎, VCTK CC BY 4.0) so
  that no gate material regresses.

### Changed

- Added bounded long-audio chunk batching without reducing the requested DDIM
  steps, with automatic single-chunk retry when an accelerator runs out of
  memory.
- Use PyTorch inference mode at the public inference boundaries to avoid
  autograd bookkeeping while retaining the established output contract.
- Reduced DDIM inference overhead by fusing the conditional and unconditional
  guidance branches for bounded batches, reusing schedules and timestep
  metadata, and avoiding intermediate tensors that the public pipeline discards.
- Stopped encoding the target spectrogram during generation. Sampling starts
  from noise and only the conditioning latent reaches the network, so that
  first-stage forward pass was computed and discarded on every call. Its shape
  now comes from the conditioning latent, which an identically configured
  autoencoder produces from a spectrogram of the same dimensions.
- Replaced the per-item librosa band-replacement loop with one batched
  transform that runs on the device holding the waveform, keeping the vocoder
  output a tensor through normalization. The crossover bin, the level-matching
  gain, and its limits are unchanged.
- Replaced the `ddim` / `use_plms` flag pair with one `sampler` name across
  `generate_batch` and `sample_log`. The full `ddpm` schedule remains selectable
  by name.
- Build one autoencoder instead of two when the checkpoint stores the same
  weights for the diffusion model's first stage and for the conditioning stage
  that encodes the low band. The released checkpoints do, so the second copy
  was built, moved to the accelerator, and held for the life of the process
  without ever differing from the first. The comparison is made against the
  checkpoint rather than against the configuration, so a checkpoint that
  trained the two apart still loads both.
- Read the checkpoint before constructing the model, so the weights that turn
  out to be redundant are released before the model allocates.
- Take the benchmark tool's usable reference band from the recording's noise
  floor rather than from its loudest bin. Music with a steep spectral slope sits
  far below its own fundamental while still carrying content: a 96 kHz piano
  recording measures 88 dB down at 16 kHz, and the previous peak-relative cut
  discarded most of the band such a reference can judge.

### Removed

- Removed the PLMS sampler. No public entry point could reach it, and its
  `register_buffer` moved every schedule tensor to `cuda` unconditionally, so
  constructing it failed on CPU, ROCm, and MPS. `dpmpp2m` supersedes it.

### Fixed

- Dropped the `duration` argument the public pipeline passed to
  `generate_batch`, which reached `**kwargs` and was never read.
- `get_input` now rejects `return_decoding_output` and `return_encoder_output`
  when the first-stage encode is skipped, instead of failing on an unbound
  local.
- Band replacement now moves the conditioning batch onto the device holding the
  generated sample. The conditioning batch reaches that stage on the host, which
  the previous element assignment absorbed implicitly.
- Band replacement now aligns the source to the generated length explicitly. The
  vocoder overshoots the conditioning length by part of one hop, which the
  previous implementation absorbed only because both lengths happened to produce
  the same number of analysis frames.

### Note

- Skipping the discarded first-stage encode removes the random draw it made, so
  a given seed reaches the sampler with a different noise tensor than in 1.0.2.
  Output for a fixed seed therefore changes, while the sampling distribution and
  the established schedule do not.
- Measured on one Radeon 760M against a 10 s excerpt of a 96 kHz 24-bit piano
  recording band-limited to 4 kHz, `trailing` at 20 steps scored best (1.498)
  and the shipped `uniform` 50-step default scored worst of the sound
  configurations (1.633) while taking 14% longer. Dropping to 12 steps cost
  nothing (1.499), so reducing the step count works, but only with `trailing`.
  `uniform` at 20 steps lost to the degraded input itself (3.270 against 3.189)
  and scattered energy above the reference's band at 1262 times the band's own
  level. Below a step count that divides 1000, `trailing` is not a preference
  but the absence of a defect, and the same holds for every kind of material.
  At the shipped 50 steps it is a preference, and it did not survive a change of
  material: `trailing` measured better on piano and worse on speech. The
  defaults are therefore unchanged, and this fork does not claim `trailing` is
  generally better.
- The same configurations measured against `example/speech.wav` separated by
  0.03%, with nothing beating the degraded input. A log-spectral distance cannot
  judge material whose missing band is noise: the model restores the right
  amount of energy without the right detail, which scores no better than
  silence. The benchmark therefore also reports a fractional-octave envelope
  distance, which accepts a different draw of the same noise and asks only
  whether the right amount of energy is in the right band at the right time.
  Re-scored on that metric, the speech restorations still measured worse than
  the degraded input, so on that clip the model misplaces energy rather than
  merely randomizing it; the piano ranking is unchanged on both metrics.
- The benchmark also reports `quiet_xs`, the level in dB that a restoration
  adds to the frames where the reference itself is quiet — the model's worst
  measured habit is returning pauses as hiss, and both distances dilute that
  across the whole clip. And unless `--no_spectrograms` is passed, every
  configuration gets a spectrogram sheet ending in a signed difference map
  against the reference, because a filled-in spectrogram alone cannot separate
  right content from wrong: red is energy the restoration added beyond the
  reference, blue is structure it missed. Sheet paths land in the JSON report
  next to the numbers so an optimization loop can consume both.
- The vocoder accounted for 82% of one call's wall clock on that machine, so
  step count moves total runtime much less there than the sampling share
  suggests.

## [1.0.2] - 2026-08-25

### Added

- Added opt-in ROCm hardware smoke and full-inference tests for validating real
  AMD GPU kernels and AudioSR's automatic device selection.
- Included `requirements.txt` in source distributions so accelerator-specific
  environments can reproduce the supported dependency ranges.

### Changed

- Lowered the PyTorch and TorchVision minimums to the current Windows ROCm 7.14
  wheel family while retaining compatibility with newer CUDA and CPU wheels.
- Documented Windows ROCm installation, HIP device semantics, and the Radeon
  hardware verification procedure.

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

## [0.1.0] — Git 記録日: 2023-09-07

- 上流 AudioSR の初期構成。モデル・音声処理・デモ画面を追加。後続の上流履歴では 0.0.x の版番号を使用。

出典: [版の記録](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/commit/1309a3b0cee07cebbb8771dc53e56fedf6ffb4ea)。

## [0.0.7] — Git 記録日: 2024-02-05

- 上流 AudioSR の履歴。モデルのダウンロード先と音声前処理、ライセンス情報を更新。

出典: [版の記録](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/commit/8d9542e80d7e83ff5dc4e4e88eb7b00974b153bd) / [変更差分](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/compare/420f712b83d5661d7e1ac179a533d18c1b9b3aae...8d9542e80d7e83ff5dc4e4e88eb7b00974b153bd)。

## [0.0.6] — Git 記録日: 2023-10-26

- 上流 AudioSR の履歴。入力ファイル一覧の指定を修正し、アップサンプリング後の無音部分を除去する処理を追加。

出典: [版の記録](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/commit/420f712b83d5661d7e1ac179a533d18c1b9b3aae) / [変更差分](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/compare/593a03cbb010f23349badc5cba241624d3b133be...420f712b83d5661d7e1ac179a533d18c1b9b3aae)。

## [0.0.5] — Git 記録日: 2023-09-24

- 上流 AudioSR の履歴。ドットを含む入力ファイル名から正しく出力名を作るよう修正し、ほぼ無音の音声の処理を調整。

出典: [版の記録](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/commit/593a03cbb010f23349badc5cba241624d3b133be) / [変更差分](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/compare/3d0779d4f75e743b3286905037ba93d5ea917c27...593a03cbb010f23349badc5cba241624d3b133be)。

## [0.0.4] — Git 記録日: 2023-09-24

- 上流 AudioSR の履歴。モジュールとしてのコマンド実行と Replicate 向けデモを追加し、音声処理を修正。

出典: [版の記録](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/commit/3d0779d4f75e743b3286905037ba93d5ea917c27) / [変更差分](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/compare/98e1038f3c17fd2bed79fae428fec9b6276437c4...3d0779d4f75e743b3286905037ba93d5ea917c27)。

## [0.0.3] — Git 記録日: 2023-09-16

- 上流 AudioSR の履歴。DC オフセットと音声の長さに関する不具合を修正し、ライセンス表記を更新。

出典: [版の記録](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/commit/98e1038f3c17fd2bed79fae428fec9b6276437c4) / [変更差分](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/compare/c2a9bb2457780e8140c5c1a055151c8bfff0d03f...98e1038f3c17fd2bed79fae428fec9b6276437c4)。

## [0.0.1] — Git 記録日: 2023-09-07

- 上流 AudioSR の履歴。初期のデモ画面と未使用の生成処理を整理し、配布構成を変更。

出典: [版の記録](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/commit/c2a9bb2457780e8140c5c1a055151c8bfff0d03f) / [変更差分](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/compare/1309a3b0cee07cebbb8771dc53e56fedf6ffb4ea...c2a9bb2457780e8140c5c1a055151c8bfff0d03f)。
