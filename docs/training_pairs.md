# Preparing training pairs

`tools/prepare_training_pairs.py` creates paired recordings for future codec
restoration training. It generates MP3 at **64, 96 and 128 kbps**, then decodes
each variant to PCM. These bitrates are totals across channels, including for
stereo. The tool prepares data only; it does not fine-tune an AudioSR model or
feed the existing envelope-calibration trainer. For the U-Net trainer, see
[LoRA training](lora_training.md).

## Requirements and usage

Use Python 3.14 with the project's NumPy, SciPy, SoundFile, PyTorch and
Torchaudio dependencies installed. Install an FFmpeg build with the
`libmp3lame` encoder separately, and either put it on PATH or pass its executable
with `--ffmpeg`. FFmpeg is used only by this offline preparation tool; ordinary
AudioSR inference does not require it.

Run from the repository root, for example in PowerShell:

```powershell
uv run --isolated --no-project --python 3.14 `
  --with-requirements requirements.txt python tools/prepare_training_pairs.py `
  --input-dir "D:\Audio\originals" `
  --output-dir "D:\Audio\training-pairs" `
  --ffmpeg "D:\Tools\ffmpeg\bin\ffmpeg.exe"
```

The input directory is scanned recursively for `.wav`, `.flac`, `.aif`,
`.aiff`, `.w64` and `.caf`. Only lossless PCM/float mono and stereo are
accepted. Channels remain separate and retain their order. The output directory
must be new and outside the input tree; existing data is never overwritten.

The source is resampled to 48 kHz with Torchaudio when needed. This 48 kHz PCM
becomes the reference for every bitrate. A shared gain is applied only if its
peak exceeds 0.99, and is recorded in the manifest. There is no independent
normalization of reference and degraded audio. WAV files use float32 PCM, so
codec overshoots are not clipped by integer output conversion.

## Dataset layout

```text
training-pairs/
  dataset.json
  manifest.jsonl
  reference/<recording_id>.wav
  encoded/<recording_id>_64k.mp3
  encoded/<recording_id>_96k.mp3
  encoded/<recording_id>_128k.mp3
  degraded/<recording_id>_64k.wav
  degraded/<recording_id>_96k.wav
  degraded/<recording_id>_128k.wav
```

Each JSONL row describes one pair. `reference`, `degraded` and `encoded` are
paths relative to the dataset root. Rows also include `recording_id`, `source`
(relative to the original input directory), `split`, `bitrate_kbps`,
`sample_rate`, `channels`, `num_samples`, `source_sample_rate`,
`reference_gain` and alignment results. `dataset.json` records the FFmpeg
version, preparation settings, duplicate sources and split counts.

Recordings are compressed in full before any future training-window extraction.
Seekable MP3 files retain Xing/LAME delay and padding information. After decoding,
the tool checks sample rate, channel count, exact length, finite samples and
zero lag in energetic windows near the beginning, middle and end. It rejects
mismatches instead of trimming, padding or guessing a correction. Digital silence
is accepted and explicitly marked `silent` because its lag cannot be measured.

Use a dataset only when `dataset.json` has `status: "complete"`. Failed runs
return a nonzero exit code and retain `status: "failed"` with an error. Partial
outputs remain available for diagnosis; rerun into a new directory after fixing
the input or FFmpeg installation. There is no resume or overwrite mode.

## Recording-level splits

By default, hashes assign approximately 80% of recordings to `train`, 10% to
`validation` and 10% to `test`. All three bitrates inherit the same split.
`--seed`, `--validation-fraction` and `--test-fraction` control this assignment.
For example, `--validation-fraction 0 --test-fraction 0` puts everything in train.
Fractions are approximate; small datasets can have empty splits, which is reported.

The recording ID hashes the canonical reference PCM and channel count. Identical
references under different filenames are deduplicated and cannot leak across
splits. Keep this manifest for downstream training and split **before** extracting
windows. Different edits, masters or recordings of the same work are not detected
as duplicates; those need grouping when curating the source material. Broader
collections of voices and instruments are more useful than many encodings of the
same few recordings.
