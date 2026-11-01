"""Fine-tune AudioSR U-Net attention LoRA from a prepared codec-pair dataset."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


SEGMENT_SAMPLES = 245760


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=("basic", "speech"), default="basic")
    parser.add_argument("--ckpt-path", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def read_pairs(root):
    """Validate the dataset contract and prevent recording/split leakage."""
    root = root.resolve(strict=True)
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("schema_version") != 1:
        raise ValueError("Dataset must have schema_version=1 and status=complete")
    manifest = (root / "manifest.jsonl").read_bytes()
    rows = []
    groups = {}
    references = {}
    import soundfile as sf

    for line in manifest.decode("utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Unknown split in manifest")
        for key in ("num_samples", "channels", "sample_rate"):
            if type(row.get(key)) is not int or row[key] <= 0:
                raise ValueError(f"Invalid manifest field: {key}")
        if row["channels"] not in (1, 2) or row["sample_rate"] != 48000:
            raise ValueError("Training expects 48 kHz mono/stereo pairs")
        recording = row["recording_id"]
        if not isinstance(recording, str) or not recording:
            raise ValueError("Invalid recording_id")
        if groups.setdefault(recording, row["split"]) != row["split"]:
            raise ValueError("A recording occurs in more than one split")
        resolved = dict(row)
        for key in ("reference", "degraded"):
            relative = Path(row[key])
            path = (root / relative).resolve(strict=True)
            if relative.is_absolute() or not path.is_relative_to(root):
                raise ValueError("Manifest paths must stay inside the dataset")
            info = sf.info(path)
            if (info.samplerate, info.channels, info.frames) != (
                    48000, row["channels"], row["num_samples"]):
                raise ValueError(f"Audio header differs from manifest: {path}")
            resolved[key] = path
        reference = resolved["reference"]
        if references.setdefault(reference, row["split"]) != row["split"]:
            raise ValueError("A reference occurs in more than one split")
        if row["split"] != "test":
            rows.append(resolved)
    if not any(row["split"] == "train" for row in rows):
        raise ValueError("Dataset has no training recordings")
    return rows, hashlib.sha256(manifest).hexdigest()


class PairDataset:
    """Read matching 5.12-second windows; each channel is a separate example."""

    def __init__(self, rows, split):
        self.examples = [(row, channel, start)
                         for row in rows if row["split"] == split
                         for channel in range(row["channels"])
                         for start in range(0, row["num_samples"], SEGMENT_SAMPLES)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        import numpy as np
        import soundfile as sf
        import torch
        from audiosr.utils import wav_feature_extraction

        row, channel, start = self.examples[index]
        count = min(SEGMENT_SAMPLES, row["num_samples"] - start)
        audio = []
        for key in ("reference", "degraded"):
            with sf.SoundFile(row[key]) as handle:
                handle.seek(start)
                values = handle.read(count, dtype="float32", always_2d=True)[:, channel]
            if len(values) != count or not np.isfinite(values).all():
                raise ValueError(f"Short read or non-finite training audio: {row[key]}")
            audio.append(values)
        reference, degraded = audio
        # 条件側の正規化を推論と揃え、同じoffset/gainを正解にも適用する。
        offset = degraded.mean()
        peak = np.max(np.abs(degraded - offset)) + 1e-8
        reference = (reference - offset) / peak * 0.5
        degraded = (degraded - offset) / peak * 0.5
        waveforms = [torch.from_numpy(np.pad(value, (0, SEGMENT_SAMPLES - count))).unsqueeze(0)
                     for value in (reference, degraded)]
        clean_mel, clean_stft = wav_feature_extraction(waveforms[0], 512)
        degraded_mel, _ = wav_feature_extraction(waveforms[1], 512)
        return {"waveform": waveforms[0], "stft": clean_stft,
                "log_mel_spec": clean_mel, "lowpass_mel": degraded_mel,
                "waveform_lowpass": waveforms[1], "sampling_rate": 48000}


def batch_loss(model, batch, cfg_dropout):
    import torch

    with torch.no_grad():
        latent, conditioning = model.get_input(
            batch, model.first_stage_key, return_first_stage_encode=True,
            unconditional_prob_cfg=cfg_dropout,
        )
    loss, _ = model(latent, model.filter_useful_cond_dict(conditioning))
    if loss.ndim or not torch.isfinite(loss):
        raise ValueError("Training produced a non-finite or nonscalar loss")
    return loss


def evaluate(model, loader, device, batches, seed):
    import torch

    model.eval()
    scores = []
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(seed)
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            scores.append((float(batch_loss(model, batch, 0.0)), len(batch["waveform"])))
    model.model.diffusion_model.train()
    return sum(loss * size for loss, size in scores) / sum(size for _, size in scores)


def train(args):
    import torch
    from audiosr.lora import inject_lora, lora_parameters, save_lora
    from audiosr.pipeline import build_model
    from audiosr.utils import seed_everything

    for key in ("steps", "batch_size", "gradient_accumulation", "validation_every",
                "validation_batches", "save_every", "rank"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    for key in ("learning_rate", "max_grad_norm", "alpha"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be finite and positive")
    for key in ("dropout", "cfg_dropout"):
        if not math.isfinite(getattr(args, key)) or not 0 <= getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be in [0,1)")
    output = args.output_dir.resolve()
    dataset_root = args.dataset.resolve(strict=True)
    if output.exists() or output.is_relative_to(dataset_root) or dataset_root.is_relative_to(output):
        raise ValueError("Choose a new output directory outside the dataset")
    rows, manifest_hash = read_pairs(dataset_root)
    training = PairDataset(rows, "train")
    validation = PairDataset(rows, "validation")
    seed_everything(args.seed)
    model = build_model(
        ckpt_path=args.ckpt_path, config=str(args.config) if args.config else None,
        device=args.device, model_name=args.model_name, record_base_identity=True,
    )
    if model.parameterization != "v":
        raise ValueError("This trainer expects the AudioSR v-prediction objective")
    inject_lora(model, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    model._audiosr_codec_lora = True
    model.eval()
    model.model.diffusion_model.train()
    parameters = list(lora_parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(training, batch_size=args.batch_size,
                                         shuffle=True, generator=generator, num_workers=0)
    val_loader = torch.utils.data.DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    device = next(model.parameters()).device
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"conditioning_mode": "codec", "manifest_sha256": manifest_hash,
                "seed": args.seed, "segment_samples": SEGMENT_SAMPLES,
                "training_examples": len(training), "validation_examples": len(validation),
                "learning_rate": args.learning_rate, "cfg_dropout": args.cfg_dropout,
                "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
                "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout}
    state = {"status": "training", "step": 0, **metadata}
    state_path = output / "training.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Training {sum(p.numel() for p in parameters):,} LoRA parameters on {device}", flush=True)
    if not validation:
        print("No validation split: best adapter selection is disabled.", flush=True)
    iterator = iter(loader)
    best = math.inf
    try:
        with (output / "history.jsonl").open("x", encoding="utf-8") as history:
            for step in range(1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                train_loss = 0.0
                for _ in range(args.gradient_accumulation):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        batch = next(iterator)
                    loss = batch_loss(model, batch, args.cfg_dropout)
                    (loss / args.gradient_accumulation).backward()
                    train_loss += float(loss.detach()) / args.gradient_accumulation
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                record = {"step": step, "train_loss": train_loss}
                if validation and (step % args.validation_every == 0 or step == args.steps):
                    score = evaluate(model, val_loader, device, args.validation_batches, args.seed)
                    record["validation_loss"] = score
                    if score < best:
                        best = score
                        save_lora(model, output / "best.safetensors", {**metadata, **record})
                if step % args.save_every == 0:
                    save_lora(model, output / f"step-{step:07d}.safetensors", {**metadata, **record})
                history.write(json.dumps(record, allow_nan=False) + "\n")
                history.flush()
                state.update(record)
                state_path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                print(json.dumps(record), flush=True)
            save_lora(model, output / "last.safetensors", {**metadata, **record})
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=str(exc) or type(exc).__name__)
        raise
    finally:
        state_path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return state


def main(argv=None):
    args = build_parser().parse_args(argv)
    # checkoutから直接実行しても、インストール済みの別版へ接続しない。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        train(args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
