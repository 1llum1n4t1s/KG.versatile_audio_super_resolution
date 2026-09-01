
# AudioSR: Versatile Audio Super-resolution at Scale

> [!IMPORTANT]
> **このリポジトリは
> [haoheliu/versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution)
> からフォークした、独立メンテナンス版です。**
> 上流の公式配布物ではありません。開発、Issue、リリースは
> [このフォーク](https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution)
> で管理し、上流の変更は自動同期しません。
>
> **This is an independently maintained fork.** Development, issues, and
> releases are managed here rather than by the upstream project.

[![Version](https://img.shields.io/badge/version-1.0.4-blue.svg?style=flat-square)](CHANGELOG.md)
[![PyPI](https://img.shields.io/pypi/v/kagayoi-audiosr?style=flat-square)](https://pypi.org/project/kagayoi-audiosr/)
[![Python](https://img.shields.io/badge/python-3.14-blue.svg?style=flat-square)](setup.py)
[![arXiv](https://img.shields.io/badge/upstream_arXiv-2309.07314-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2309.07314)
[![Upstream samples](https://img.shields.io/badge/upstream-audio_samples-blue?logo=Github&style=flat-square)](https://audioldm.github.io/audiosr)

Current fork release: **1.0.4**. See the [changelog](CHANGELOG.md) for details.

AudioSR restores high-frequency detail from a low-pass audio condition and writes a 48 kHz result. It works with many kinds of audio, including music, speech, and environmental recordings.

![AudioSR visualization](visualization.png)

## What AudioSR can and cannot do

AudioSR is a generative audio super-resolution model. It is designed to preserve the supplied low-frequency content and synthesize plausible high-frequency detail that is missing after low-pass filtering.

The inference pipeline uses a fixed 48 kHz working and output rate. Other input sample rates are resampled to 48 kHz; this does not mean that the original recording contained recoverable information above its input bandwidth.

A source recorded above 48 kHz therefore loses everything over 24 kHz, and the result comes back at 48 kHz. `--preserve_input_rate` writes at the input's own rate instead and puts the input's own content above 24 kHz back, so a 96 kHz file stays a 96 kHz file and is named `_AudioSR_Processed_96K`. It adds no bandwidth: the model generates nothing up there either way, and on real recordings that band usually carries only the noise floor. A 96 kHz 24-bit solo piano recording measures a flat -91 dB above 24 kHz, with no musical content at all.

AudioSR is not a general restoration filter. It is not trained or intended to repair time-domain losses, packet loss, or low-frequency noise. Inputs with an unfamiliar cutoff pattern, such as some MP3 encodings, may produce weaker high-frequency inpainting. See [Important things to know to make AudioSR work](example/how_to_make_audiosr_work.md) for examples and low-pass preprocessing guidance.

Bit depth is a separate property from sample rate and generated bandwidth. Increasing the output sample rate or synthesizing high frequencies does not recover information that was lost through low bit depth.

## Highlights in this fork

- Mono and multi-channel audio are preserved through CLI, long-audio, Gradio,
  and output-writing paths.
- Long recordings use overlap-and-add chunking with bounded GPU batches,
  including safe handling of short final chunks and single-chunk OOM retry.
- Long-audio and local Gradio batching keep deterministic seeds and reuse the
  selected model between requests.
- Local Safetensors checkpoints are supported. PyTorch checkpoints are loaded
  with `weights_only=True`.
- Audio decoding and output use SoundFile directly, without requiring
  `ffprobe` for ordinary inference.
- Python 3.14 and current PyTorch installations are supported.

## Project history

- 2025-06-28: Added an [LSD calculation pitfall demonstration](example/lsd_calculation_pitfall/README.md) showing the importance of energy scaling for fair Log Spectral Distance evaluation.
- 2024-12-31: The training code of AudioSR can be found [here](https://drive.google.com/file/d/1BaZuHbk1AfURX7SvkaD5_ZWLwun-wdpW/view?usp=drive_link) (for reference only; the code is not carefully organized).
- 2024-12-16: Added [Important things to know to make AudioSR work](example/how_to_make_audiosr_work.md).
- 2023-09-24: Fixed a Windows error and a librosa warning (@ORI-Muchim).
- 2023-09-16: Fixed the DC shift issue and duration padding bug, and updated the default DDIM steps to 50.

## Installation

```shell
# Windows PowerShell (venv, using uv)
uv venv --python 3.14 .venv
.venv\Scripts\Activate.ps1

# macOS/Linux (venv, using uv)
uv venv --python 3.14 .venv
source .venv/bin/activate

# Or use conda instead of venv:
# conda create -n audiosr python=3.14
# conda activate audiosr
```

AudioSR requires Python 3.14. Use an isolated environment and activate it before installing anything.

Install PyTorch before AudioSR. Select the wheel for the operating system, Python version, and accelerator from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). For an RTX 50-series/Blackwell machine, a current official stable CUDA 13.0 wheel can be installed with this example:

```shell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

The selector is the source of truth for CPU, CUDA, and macOS. For AMD GPUs and Ryzen APUs, use the hardware-specific command from the [AMD ROCm PyTorch installer](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). For example, the current Windows command for a `gfx1103` Radeon 760M/780M with ROCm 7.14 is:

```shell
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1103]==2.12.0+rocm7.14.0" "torchvision[device-gfx1103]==0.27.0+rocm7.14.0" "torchaudio==2.11.0+rocm7.14.0"
```

These commands are examples for the matching platform and may change as PyTorch or ROCm publishes new compatible wheels. Always use the current command from the corresponding official installer.

Install this fork from PyPI:

```shell
python -m pip install kagayoi-audiosr
```

To install the latest source instead, use the repository or a local checkout:

```shell
python -m pip install git+https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution.git
# Local checkout:
# python -m pip install .
```

The PyPI distribution is named `kagayoi-audiosr`, while the Python import and
command remain `audiosr`. The separate `audiosr==0.0.7` distribution is the
older upstream release and does not contain the changes documented here.

### Model downloads and disk space

The first run downloads the selected checkpoint from Hugging Face automatically:

- `basic`: approximately 6.18 GB.
- `speech`: approximately 6.18 GB.
- Both checkpoints: approximately 12.36 GB, plus the Hugging Face cache and temporary working space.

Hugging Face reuses a cached checkpoint on later runs, so each model needs to be downloaded only once per cache. By default this is below `~/.cache/huggingface/hub` (including the corresponding user-profile path on Windows); `HF_HOME` can select another cache location. Make sure the cache volume has room for the model files and temporary output before starting inference.

## Command-line usage

Run `python -m audiosr --help` for the installed command's help. Supply one input file with `-i`, or a newline-separated list with `-il`. Results are written below `./output/<timestamp>` by default.

Process one file:

```shell
python -m audiosr -i path/to/input.wav
```

Process a list of files and select the speech checkpoint:

```shell
python -m audiosr -il batch.lst --model_name speech -s ./output
```

Process long audio with overlap-and-add chunking. `--chunk_duration` must be greater than `--overlap_duration`:

```shell
python -m audiosr -i long_recording.wav \
  --chunking --chunk_duration 15 --overlap_duration 2
```

After installation, the `audiosr` executable is also available on supported shells, so the equivalent form is `audiosr -i path/to/input.wav`.

### Device selection

`-d/--device` defaults to `auto`, which selects the PyTorch CUDA/HIP device when available, then Apple MPS, and otherwise CPU. The supported values are:

- `auto`
- `cpu`
- `cuda`
- `cuda:N` (for example, `cuda:0` or `cuda:1`)
- `mps`

PyTorch intentionally exposes ROCm through the `torch.cuda` API, so ROCm users must select `cuda` or `cuda:N`, not `rocm`. `torch.version.hip` can be used to distinguish a ROCm build from a CUDA build. DirectML is not supported. MPS support depends on the installed PyTorch build and the macOS/Apple hardware; if MPS fails, use a compatible CUDA or CPU installation.

Examples:

```shell
python -m audiosr -i path/to/input.wav -d cuda:0
python -m audiosr -i path/to/input.wav -d mps
python -m audiosr -i path/to/input.wav -d cpu
```

### ROCm hardware verification

The ROCm test is skipped during the normal CPU test suite. On a machine with an AMD ROCm PyTorch wheel installed, run the opt-in smoke test to verify real GPU kernels and AudioSR's automatic device selection:

```powershell
$env:AUDIOSR_RUN_ROCM_HARDWARE = "1"
python -m pytest -q tests/test_rocm_hardware.py
```

To additionally download the basic checkpoint and perform an end-to-end AudioSR conversion, enable the slower full-inference test. It uses one DDIM step by default; set `AUDIOSR_ROCM_DDIM_STEPS` to exercise an application-specific value:

```powershell
$env:AUDIOSR_RUN_ROCM_FULL_INFERENCE = "1"
$env:AUDIOSR_RUN_ROCM_HARDWARE = "1"
$env:AUDIOSR_ROCM_DDIM_STEPS = "100"
python -m pytest -q tests/test_rocm_hardware.py
```

The full-inference test is intentionally separate because it downloads a model of about 6.2 GB and requires substantially more memory than the kernel smoke test. If first-run MIOpen convolution tuning is too slow, setting `$env:MIOPEN_FIND_MODE = "FAST"` uses AMD's immediate fallback for faster startup at the cost of some GPU performance.

### Tuning and performance

The most relevant options are:

| Option | Default | Description |
| --- | ---: | --- |
| `-i`, `--input_audio_file` | — | One input audio file. Use either this or `-il`. |
| `-il`, `--input_file_list` | — | Text file containing one input path per line. |
| `-s`, `--save_path` | `./output` | Parent directory for timestamped output. |
| `--model_name` | `basic` | Checkpoint: `basic` or `speech`. |
| `-d`, `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`, or `mps`. |
| `--ddim_steps` | `50` | Sampling steps from 1 to 1000. More steps usually take longer. |
| `--sampler` | `ddim` | `ddim`, `dpmpp2m`, or `ddpm`. `dpmpp2m` needs far fewer steps. |
| `--ddim_eta` | `1.0` | How stochastic each `ddim` update is: `0.0` is deterministic, `1.0` is fully ancestral. Only `ddim` uses it. |
| `--discretize` | `uniform` | Timestep spacing: `uniform`, `trailing`, or `quad`. Use `trailing` when lowering `--ddim_steps` to a divisor of 1000. |
| `--preserve_input_rate` | off | Write at the input's own rate instead of 48 kHz, keeping whatever the input held above 24 kHz. |
| `--calibration` | off | Path to a learned output calibration (`tools/train_calibration.py`). Tempers the model's measured 10-28 dB overshoot in the generated band. The package ships one at `audiosr.bundled_calibration_path()`. |
| `-gs`, `--guidance_scale` | `3.5` | Strength of the low-pass audio conditioning. |
| `--seed` | `42` | Integer seed for reproducible sampling. |
| `--suffix` | `_AudioSR_Processed_48K` | Suffix appended to the output filename. |
| `--chunking` | off | Enable long-audio chunking. |
| `--chunk_duration` | `15` | Chunk length in seconds when chunking is enabled. |
| `--overlap_duration` | `2` | Cross-fade overlap in seconds when chunking is enabled. |

The bundled calibration (`audiosr/weights/calibration_v3.pt`) was trained on
piano, synthesizer, and orchestra excerpts plus speech from the
[つくよみちゃんコーパス](https://tyc.rei-yumesaki.net/material/corpus/)
(CV.夢前黎) and [VCTK](https://datashare.ed.ac.uk/handle/10283/3443)
(CC BY 4.0); no corpus audio is redistributed, only the 134 KB envelope
predictor learned from it.

Start `--guidance_scale` around 2.5–5.0 and adjust by ear. This guidance scale controls adherence to the low-pass audio condition; it is not a text prompt relevance or text-to-audio control. Memory use and speed vary with input length, sampling steps, selected device, and whether chunking is enabled.

The default `--sampler ddim` with `--ddim_eta 1.0` reproduces the established
schedule, where every update is fully ancestral and quality keeps improving with
step count. `--sampler dpmpp2m` instead solves the deterministic
probability-flow ODE, which converges in far fewer steps for the same number of
network evaluations per step:

```shell
python -m audiosr -i path/to/input.wav --sampler dpmpp2m --ddim_steps 30
```

**Lowering the step count needs `--discretize trailing`.** The default
`uniform` spacing reaches the noisiest timestep only when the step count does
not divide 1000; at 20, 25, 50, 100 and other exact divisors it stops short, so
sampling starts from pure noise against a schedule that still holds part of the
signal: 7.5% of it at 20 steps, 2.8% at 50. On a music reference, `uniform` at
20 steps scored worse than not restoring at all, while `trailing` at 20 scored
best of everything tried and ran 14% faster, and 12 steps matched it:

```shell
python -m audiosr -i path/to/input.wav --ddim_steps 20 --discretize trailing
```

At the shipped 50 steps the shortfall is far smaller, and there the advantage
did not survive a change of material: `trailing` measured better on piano and
worse on speech. A difference that reverses with the material is not a basis for
a default, so `uniform` stays, and this fork makes no general claim that
`trailing` sounds better. Measure your own material before changing it.

Which configuration wins depends on the material and the accelerator, so measure
before settling on one. `tools/benchmark_samplers.py` degrades a high-quality
reference, restores it with each configuration, and reports time next to
log-spectral distance:

```shell
python tools/benchmark_samplers.py --reference reference.flac \
  --config ddim:50:1.0:uniform --config ddim:20:0.0:trailing \
  --config dpmpp2m:20::trailing --config dpmpp2m:30::trailing
```

Restorations are written to `--output_dir` so the same run can also feed a blind
listening comparison. The reference must be a genuinely high-quality
recording. Three numbers are reported per configuration: a per-bin
log-spectral distance, which judges harmonic material such as solo piano
cleanly; a fractional-octave envelope distance, which stays meaningful when
the missing band is noise — fricatives, applause, cymbals — where no
restoration can match the reference bin by bin; and `quiet_xs`, the dB the
restoration adds to the reference's own quiet frames, which isolates hiss laid
over pauses. Each configuration also gets a spectrogram sheet whose last row
is a signed difference against the reference (red = added beyond the
reference, blue = missed structure), since a filled-in spectrogram alone reads
the same whether the content is right or wrong; `--no_spectrograms` skips the
sheets. The run warns when no configuration beats the degraded input by a
useful margin on either distance.

## Gradio demo

The local Gradio demo is implemented in `app.py`:

```shell
python app.py
```

Open the local URL printed by Gradio. The demo supports the `basic` and
`speech` checkpoints, guidance scale, DDIM steps, deterministic seeds, and
optional GPU chunk batching. Batch size defaults to 1 because larger batches
use more accelerator memory. The active model is cached and reused until a
different checkpoint is selected.

## Troubleshooting and known limitations

- **Out of memory:** keep Gradio batch size at 1, enable CLI chunking for long
  files, or reduce the chunk duration.
- **First run appears stalled:** each checkpoint is approximately 6.18 GB and
  must be downloaded before inference starts. Later runs reuse the Hugging Face
  cache.
- **Unexpected high-frequency results:** AudioSR generates plausible detail;
  it cannot recover the exact information removed from the source recording.
- **MP3 or unusual cutoff patterns:** apply a clean low-pass filter first. See
  [the upstream guidance](example/how_to_make_audiosr_work.md).
- **DirectML:** it is not supported. Use CUDA, Apple MPS where compatible, or
  CPU.

## Developer notes

The latent diffusion path operates on latent tensors shaped `[B, 16, 128, 32]` for the default model configuration. Code that integrates with the model should obtain inputs through `get_input(...)` so conditioning and device placement stay consistent with the inference path. Generation starts from noise and never samples the target latent, so the inference path calls it with `return_first_stage_encode=False`; pass `True` only when the encoded target is actually used.

## Cite our work

If you find this repo useful, please consider citing:
```bibtex
@inproceedings{liu2024audiosr,
  title={{AudioSR}: Versatile audio super-resolution at scale},
  author={Liu, Haohe and Chen, Ke and Tian, Qiao and Wang, Wenwu and Plumbley, Mark D},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={1076--1080},
  year={2024},
  organization={IEEE}
}
```

# Understanding the Impact of Cutoff Patterns on AudioSR Performance

**AudioSR** is a powerful tool for audio super-resolution. However, its performance can be significantly influenced by the characteristics of the input data, especially the cutoff pattern. 

## 🚩 When AudioSR May Fail
1. **Input Audio with Unfamiliar Cutoff Patterns**  
   If the input audio file contains a cutoff pattern that is **significantly different** from those used in training, AudioSR may fail to perform effectively.
   
2. **Input Audio with Severe Distortions**  
   Strong distortions such as excessive noise or reverb can degrade the performance of AudioSR.

## ❓ Why Do Cutoff Patterns Have Such a Huge Impact on AudioSR?
During training, our data was simulated using **low-pass filtering**. The model was not trained to handle other causes of high-frequency loss, such as MP3 compression. As a result, AudioSR struggles when encountering unfamiliar cutoff patterns.

For example, MP3 compression can introduce a cutoff pattern that looks like this:

![MP3 Cutoff Example](example/figs/mp3.png)

### Why This Matters
As you can see, there are **spectrogram holes** near the cutoff range, which differ significantly from the patterns seen during training. When you apply AudioSR to such data, the output may look like this:

![AudioSR Output on MP3](example/figs/mp3_after.png)

The higher frequencies are not adequately inpainted due to the unfamiliar cutoff pattern.

### A Simple Solution: Low-Pass Filtering
To mitigate this issue, you can perform a **low-pass filtering** on the audio before feeding it into AudioSR. After low-pass filtering, the audio would resemble a standard low-pass cutoff pattern, like this:

![Low-Pass Filtered Audio](example/figs/lowpass.jpg)

When processed by AudioSR, the output will then be as expected, with improved high-frequency inpainting:

![AudioSR Output on Low-Pass](example/figs/lowpass_after.png)

---

By understanding the limitations and addressing them with preprocessing, you can maximize the performance of AudioSR!
