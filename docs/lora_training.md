# U-Net LoRA training for codec restoration

This repository includes a standalone trainer for datasets created by
[prepare_training_pairs.py](training_pairs.md). It updates only low-rank
adapters in U-Net attention projections (`to_q`, `to_k`, `to_v`, `to_out`).
The VAE, conditioning encoder, vocoder and original U-Net parameters stay frozen.
No PEFT, Lightning or additional training framework is required.

## Run training

Use a repository checkout, Python 3.14 and the project's dependencies. Install
the appropriate official PyTorch/Torchaudio wheels for your accelerator first.
A CUDA GPU is recommended for the full model; CPU execution is supported but
the small-model smoke test does not establish full-model speed or VRAM needs.
Dataset preparation requires FFmpeg; this trainer reads the prepared WAV files
with SoundFile and does not run FFmpeg.

From the repository root, in PowerShell:

```powershell
uv run --isolated --no-project --python 3.14 `
  --with-requirements requirements.txt python tools/train_lora.py `
  --dataset "D:\Audio\training-pairs" `
  --output-dir "D:\Audio\lora-run" `
  --model-name basic --device cuda `
  --rank 8 --alpha 8 --learning-rate 0.0001 `
  --steps 1000 --batch-size 1 --gradient-accumulation 4
```

`--output-dir` must be new and outside the dataset. The base checkpoint is
downloaded through the existing pinned model-download path when `--ckpt-path`
is omitted. Use `--ckpt-path` and, when needed, `--config` for a local base.
The training run reads `dataset.json` with `status: "complete"`, then
`manifest.jsonl`. It rejects paths outside the dataset, inconsistent audio
headers, and recording IDs or reference paths assigned to multiple splits.

All channels are used as separate mono examples, without averaging them.
Recordings are divided into matching 245,760-sample (5.12-second) windows;
the last short window is zero-padded. The recording-level split is inherited
before window extraction. Training uses `train`, validation uses `validation`,
and `test` audio is never used for optimization or model selection.

The degraded window determines normalization; the same offset and gain are
applied to its clean target. Both use the actual mel extractor. The clean mel
is encoded as the target latent, and the degraded mel is the condition.
There is no extra low-pass filter. The existing `v`-prediction diffusion loss
is retained; the trainer does not add waveform, GAN or perceptual losses.

## Outputs and controls

| Output | Meaning |
| --- | --- |
| `last.safetensors` | Adapter at the final optimizer step |
| `best.safetensors` | Adapter with the lowest measured validation loss, if a validation split exists |
| `step-0000100.safetensors` etc. | Periodic adapter checkpoints |
| `history.jsonl` | Per-step training loss and periodic validation loss |
| `training.json` | Run settings, last completed step and `training`/`complete`/`failed` state |

Adapters contain only LoRA tensors and metadata, including the base checkpoint
SHA-256, model configuration fingerprint, manifest fingerprint, target layer
shapes and LoRA settings. They do not contain a copy of the base model.
`--save-every` and `--validation-every` default to 100 optimizer steps.
`--validation-batches` defaults to 8; validation uses fixed windows and a fixed
Torch RNG seed. Validation loss helps choose an adapter, but is not a listening
test or proof of improved audio quality.

`--cfg-dropout` defaults to 0.1 to retain classifier-free conditioning dropout.
`--dropout` controls LoRA dropout and defaults to 0. Gradient clipping defaults
to `--max-grad-norm 1`. Training uses float32 and one data-loading process.
`--seed` controls shuffling and model randomness. Reproducibility across different
hardware and library versions is not guaranteed.

There is no optimizer/RNG resume mode or mixed precision in this initial trainer.
On interruption, completed periodic adapters remain available; inspect
`training.json` rather than treating a failed run as complete. Non-finite audio,
losses or gradients fail the run.

## Apply an adapter

```powershell
uv run --isolated --no-project --python 3.14 `
  --with-requirements requirements.txt python -m audiosr `
  -i "D:\Audio\compressed.wav" -s "D:\Audio\restored" `
  --model_name basic --lora "D:\Audio\lora-run\best.safetensors"
```

If there was no validation split, use `last.safetensors`. Use the exact base
checkpoint and configuration used in training. For a local base, also pass
`--ckpt-path` and `--config` as needed. A different checkpoint file, including a
re-encoded copy in another checkpoint format, fails the fingerprint check.
Applying a second adapter to the same model is rejected.

The Python API supports `build_model(..., lora_path="adapter.safetensors")`.
For this codec-restoration adapter, the pipeline uses degraded mel directly as
the condition and disables the original mel/low-band replacement steps. Those
steps would otherwise restore the input's compression artifacts. The ordinary
pipeline is retained when no adapter is provided. Existing Gradio controls do
not expose adapter selection; use the CLI or Python API.

The mechanism has been smoke-tested with a small real VAE/U-Net checkpoint.
Full-size training and improved sound quality require a suitable GPU, curated
training data, held-out recordings and listening evaluation.
