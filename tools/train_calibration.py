"""Train the output calibration from full-band recordings.

Each recording is band-limited at several cutoffs the way real sources are,
and the predictor learns to map the band-limited envelope to the full-band
one. The audio itself never enters the model — only fractional-octave
envelopes do — so an hour of material and a CPU are enough.

Usage:
    python tools/train_calibration.py --audio a.wav b.wav --out calibration.pt
"""

import argparse
import os
import pathlib

CUTOFFS_HZ = (2000, 4000, 8000, 12000)
SAMPLE_RATE = 48000


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="Full-band 48 kHz recordings to learn from.",
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=list(CUTOFFS_HZ),
        help="Band limits applied to build training pairs.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def degrade(waveform, cutoff_hz):
    import torchaudio

    narrow_rate = int(cutoff_hz * 2)
    narrow = torchaudio.functional.resample(
        waveform, orig_freq=SAMPLE_RATE, new_freq=narrow_rate
    )
    restored = torchaudio.functional.resample(
        narrow, orig_freq=narrow_rate, new_freq=SAMPLE_RATE
    )
    return restored[..., : waveform.shape[-1]]


def training_pairs(paths, cutoffs):
    """Yield (features, targets, mask) built from every file and cutoff."""
    import soundfile as sf
    import torch

    from audiosr import calibration

    edges = calibration.band_edges()
    for path in paths:
        audio, rate = sf.read(os.fspath(path), dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{path} is {rate} Hz; training expects 48 kHz")
        waveform = torch.from_numpy(audio.T).mean(dim=0, keepdim=True)

        reference_envelope = calibration.band_envelope(waveform[0], edges)
        for cutoff_hz in cutoffs:
            degraded = degrade(waveform, cutoff_hz)
            source_envelope = calibration.band_envelope(degraded[0], edges)
            frames = min(
                reference_envelope.shape[-1], source_envelope.shape[-1]
            )
            offset = calibration.level_offset(source_envelope[:, :frames])
            features = calibration.stack_context(
                source_envelope[:, :frames] - offset
            )
            targets = (reference_envelope[:, :frames] - offset).T

            cutoff_index = calibration.cutoff_band(
                source_envelope[:, :frames], edges
            )
            mask = torch.zeros(len(edges) - 1)
            mask[cutoff_index + 1 :] = 1.0
            yield features, targets, mask.expand(frames, -1)


def train(features, targets, mask, epochs, learning_rate, seed):
    import torch

    from audiosr import calibration

    torch.manual_seed(seed)
    model = calibration.EnvelopePredictor(bands=targets.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weight_total = mask.sum().clamp_min(1.0)

    losses = []
    for _epoch in range(epochs):
        optimizer.zero_grad()
        predicted = model(features)
        loss = (((predicted - targets) ** 2) * mask).sum() / weight_total
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, losses


def main(argv=None):
    args = build_parser().parse_args(argv)

    import torch

    from audiosr import calibration

    parts = list(training_pairs(args.audio, args.cutoffs))
    features = torch.cat([part[0] for part in parts])
    targets = torch.cat([part[1] for part in parts])
    mask = torch.cat([part[2] for part in parts])
    print(
        f"training on {features.shape[0]} frames from {len(args.audio)} files "
        f"x {len(args.cutoffs)} cutoffs"
    )

    model, losses = train(
        features, targets, mask, args.epochs, args.learning_rate, args.seed
    )
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} epochs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    calibration.save_calibration(os.fspath(args.out), model)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
