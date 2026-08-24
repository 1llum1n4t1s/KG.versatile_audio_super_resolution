
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

[![Version](https://img.shields.io/badge/version-1.0.1-blue.svg?style=flat-square)](CHANGELOG.md)
[![PyPI](https://img.shields.io/pypi/v/kagayoi-audiosr?style=flat-square)](https://pypi.org/project/kagayoi-audiosr/)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue.svg?style=flat-square)](setup.py)
[![arXiv](https://img.shields.io/badge/upstream_arXiv-2309.07314-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2309.07314)
[![Upstream samples](https://img.shields.io/badge/upstream-audio_samples-blue?logo=Github&style=flat-square)](https://audioldm.github.io/audiosr)

Current fork release: **1.0.1**. See the [changelog](CHANGELOG.md) for details.

AudioSR restores high-frequency detail from a low-pass audio condition and writes a 48 kHz result. It works with many kinds of audio, including music, speech, and environmental recordings.

![AudioSR visualization](visualization.png)

## What AudioSR can and cannot do

AudioSR is a generative audio super-resolution model. It is designed to preserve the supplied low-frequency content and synthesize plausible high-frequency detail that is missing after low-pass filtering.

The inference pipeline uses a fixed 48 kHz working and output rate. Other input sample rates are resampled to 48 kHz; this does not mean that the original recording contained recoverable information above its input bandwidth.

AudioSR is not a general restoration filter. It is not trained or intended to repair time-domain losses, packet loss, or low-frequency noise. Inputs with an unfamiliar cutoff pattern, such as some MP3 encodings, may produce weaker high-frequency inpainting. See [Important things to know to make AudioSR work](example/how_to_make_audiosr_work.md) for examples and low-pass preprocessing guidance.

Bit depth is a separate property from sample rate and generated bandwidth. Increasing the output sample rate or synthesizing high frequencies does not recover information that was lost through low bit depth.

## Highlights in this fork

- Mono and multi-channel audio are preserved through CLI, long-audio, Gradio,
  and output-writing paths.
- Long recordings use overlap-and-add chunking, including safe handling of
  short final chunks.
- The local Gradio app supports GPU chunk batching, deterministic seeds, and
  reuses the selected model between requests.
- Local Safetensors checkpoints are supported. PyTorch checkpoints are loaded
  with `weights_only=True`.
- Audio decoding and output use SoundFile directly, without requiring
  `ffprobe` for ordinary inference.
- Python 3.10 through 3.14 and current PyTorch installations are supported.

## Project history

- 2025-06-28: Added an [LSD calculation pitfall demonstration](example/lsd_calculation_pitfall/README.md) showing the importance of energy scaling for fair Log Spectral Distance evaluation.
- 2024-12-31: The training code of AudioSR can be found [here](https://drive.google.com/file/d/1BaZuHbk1AfURX7SvkaD5_ZWLwun-wdpW/view?usp=drive_link) (for reference only; the code is not carefully organized).
- 2024-12-16: Added [Important things to know to make AudioSR work](example/how_to_make_audiosr_work.md).
- 2023-09-24: Fixed a Windows error and a librosa warning (@ORI-Muchim).
- 2023-09-16: Fixed the DC shift issue and duration padding bug, and updated the default DDIM steps to 50.

## Installation

```shell
# Windows PowerShell (venv, using uv)
uv venv --python 3.12 .venv
.venv\Scripts\Activate.ps1

# macOS/Linux (venv, using uv)
uv venv --python 3.12 .venv
source .venv/bin/activate

# Or use conda instead of venv:
# conda create -n audiosr python=3.12
# conda activate audiosr
```

AudioSR is intended for Python 3.10 through 3.14. Use an isolated environment and activate it before installing anything.

Install PyTorch before AudioSR. Select the wheel for the operating system, Python version, and accelerator from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). For an RTX 50-series/Blackwell machine, a current official stable CUDA 13.0 wheel can be installed with this example:

```shell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

The selector is the source of truth. The CUDA 13.0 command above is an example for the matching platform and may need to change as PyTorch publishes new compatible wheels; CPU, CUDA, ROCm, and macOS installations should use the selector's command instead.

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

`-d/--device` defaults to `auto`, which selects CUDA when available, then Apple MPS, and otherwise CPU. The supported values are:

- `auto`
- `cpu`
- `cuda`
- `cuda:N` (for example, `cuda:0` or `cuda:1`)
- `mps`

DirectML is not supported. MPS support depends on the installed PyTorch build and the macOS/Apple hardware; if MPS fails, use a compatible CUDA or CPU installation.

Examples:

```shell
python -m audiosr -i path/to/input.wav -d cuda:0
python -m audiosr -i path/to/input.wav -d mps
python -m audiosr -i path/to/input.wav -d cpu
```

### Tuning and performance

The most relevant options are:

| Option | Default | Description |
| --- | ---: | --- |
| `-i`, `--input_audio_file` | — | One input audio file. Use either this or `-il`. |
| `-il`, `--input_file_list` | — | Text file containing one input path per line. |
| `-s`, `--save_path` | `./output` | Parent directory for timestamped output. |
| `--model_name` | `basic` | Checkpoint: `basic` or `speech`. |
| `-d`, `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`, or `mps`. |
| `--ddim_steps` | `50` | DDIM sampling steps from 1 to 1000. More steps usually take longer. |
| `-gs`, `--guidance_scale` | `3.5` | Strength of the low-pass audio conditioning. |
| `--seed` | `42` | Integer seed for reproducible sampling. |
| `--suffix` | `_AudioSR_Processed_48K` | Suffix appended to the output filename. |
| `--chunking` | off | Enable long-audio chunking. |
| `--chunk_duration` | `15` | Chunk length in seconds when chunking is enabled. |
| `--overlap_duration` | `2` | Cross-fade overlap in seconds when chunking is enabled. |

Start `--guidance_scale` around 2.5–5.0 and adjust by ear. This guidance scale controls adherence to the low-pass audio condition; it is not a text prompt relevance or text-to-audio control. Memory use and speed vary with input length, DDIM steps, selected device, and whether chunking is enabled.

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

The latent diffusion path operates on latent tensors shaped `[B, 16, 128, 32]` for the default model configuration. Code that integrates with the model should obtain inputs through `get_input(...)` so first-stage encoding, conditioning, and device placement stay consistent with the inference path.

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

