#!/usr/bin/env python3
"""Compare AudioSR sampler configurations on speed and reconstruction quality.

The harness degrades a high-quality reference recording, restores it with each
requested configuration, and reports wall-clock cost alongside log-spectral
distance against the untouched reference. Restored files are written out so the
same run can feed a blind listening comparison.

Example::

    uv run --with-requirements requirements.txt python tools/benchmark_samplers.py \\
        --reference reference.flac \\
        --config ddim:100:1.0 --config ddim:50:0.0 \\
        --config dpmpp2m:20 --config dpmpp2m:30 --config dpmpp2m:50

Scoring only covers the band the reference can actually judge: from the
degradation cutoff up to where the reference itself stops carrying content. A
reference that is band-limited above the cutoff cannot say whether invented
content is right or wrong, so energy generated above that point is reported
separately rather than counted as error.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

_SAMPLE_RATE = 48000
_LSD_N_FFT = 2048
_LSD_HOP = 512
_LSD_FLOOR = 1e-8
# How far below the reference's loudest bin still counts as content rather than
# the recording's noise floor.
_REFERENCE_FLOOR_MARGIN_DB = 10.0
_REFERENCE_FLOOR_QUANTILE = 0.02
# A scored band narrower than this cannot separate configurations.
_MIN_SCORED_BAND_HZ = 4000.0
_MIN_USEFUL_IMPROVEMENT = 0.02


def parse_config(value):
    """Parse a ``sampler[:steps[:eta[:discretize]]]`` specification."""
    from audiosr.sampling import (
        DEFAULT_DISCRETIZATION,
        normalize_ddim_eta,
        normalize_discretize,
        normalize_sampler,
    )

    parts = value.split(":")
    if not 1 <= len(parts) <= 4:
        raise argparse.ArgumentTypeError(
            "a configuration looks like 'sampler', 'sampler:steps', "
            "'sampler:steps:eta', or 'sampler:steps:eta:discretize'"
        )
    try:
        sampler = normalize_sampler(parts[0])
        steps = int(parts[1]) if len(parts) > 1 else 50
        eta = normalize_ddim_eta(parts[2]) if len(parts) > 2 and parts[2] else 1.0
        discretize = (
            normalize_discretize(parts[3]) if len(parts) > 3 else DEFAULT_DISCRETIZATION
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not 1 <= steps <= 1000:
        raise argparse.ArgumentTypeError("steps must be between 1 and 1000")
    return {
        "sampler": sampler,
        "ddim_steps": steps,
        "ddim_eta": eta,
        "discretize": discretize,
    }


def config_label(config):
    label = f"{config['sampler']}/{config['ddim_steps']}"
    if config["sampler"] == "ddim":
        label += f"/eta{config['ddim_eta']:g}"
    return f"{label}/{config['discretize']}"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark AudioSR sampler configurations on speed and quality."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help=(
            "A high-quality 48 kHz recording used as ground truth. Lossy sources "
            "make the quality columns unreliable."
        ),
    )
    parser.add_argument(
        "--config",
        dest="configs",
        type=parse_config,
        action="append",
        required=True,
        help="A sampler configuration; repeat the flag to compare several.",
    )
    parser.add_argument(
        "--cutoff_hz",
        type=int,
        default=12000,
        help="Bandwidth of the degraded input the model has to restore.",
    )
    parser.add_argument(
        "--duration_s",
        type=float,
        default=15.0,
        help="Seconds taken from the reference; 0 uses the whole file.",
    )
    parser.add_argument(
        "--offset_s",
        type=float,
        default=0.0,
        help="Where to start reading the reference.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Timed runs per configuration; the median is reported.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--model_name", type=str, default="basic", choices=["basic", "speech"]
    )
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./benchmark_output"),
        help="Where the degraded input and each restoration are written.",
    )
    parser.add_argument(
        "--json_report",
        type=Path,
        default=None,
        help="Optional path for a machine-readable copy of the results.",
    )
    parser.add_argument(
        "--skip_warmup",
        action="store_true",
        help="Skip the untimed warm-up run that absorbs lazy accelerator setup.",
    )
    return parser


def load_reference(path, offset_s, duration_s):
    """Return a mono float32 reference at 48 kHz."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    waveform, sample_rate = sf.read(os.fspath(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(waveform.T))
    # A single channel keeps the comparison about the sampler rather than about
    # how channels are blended.
    waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != _SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=_SAMPLE_RATE
        )

    start = int(round(offset_s * _SAMPLE_RATE))
    if start >= waveform.shape[-1]:
        raise SystemExit("--offset_s starts past the end of the reference")
    waveform = waveform[:, start:]
    if duration_s > 0:
        waveform = waveform[:, : int(round(duration_s * _SAMPLE_RATE))]
    if waveform.shape[-1] < _SAMPLE_RATE:
        raise SystemExit("the selected reference span is shorter than one second")
    return waveform.contiguous()


def degrade(waveform, cutoff_hz):
    """Band-limit the reference the way a low-bandwidth source would be.

    Resampling down and back models a real narrow-band recording, and keeps the
    degradation independent of the filter family the model conditions on.
    """
    import torchaudio

    narrow_rate = int(cutoff_hz * 2)
    if not 0 < narrow_rate < _SAMPLE_RATE:
        raise SystemExit("--cutoff_hz must be above 0 and below 24000")
    narrow = torchaudio.functional.resample(
        waveform, orig_freq=_SAMPLE_RATE, new_freq=narrow_rate
    )
    restored = torchaudio.functional.resample(
        narrow, orig_freq=narrow_rate, new_freq=_SAMPLE_RATE
    )
    return restored[:, : waveform.shape[-1]].contiguous()


def _unit_rms(waveform, floor=1e-8):
    import torch

    return waveform / torch.sqrt(torch.mean(waveform**2) + floor**2)


def _magnitude(signal):
    import torch

    return torch.stft(
        _unit_rms(signal.flatten()),
        n_fft=_LSD_N_FFT,
        hop_length=_LSD_HOP,
        window=torch.hann_window(_LSD_N_FFT, periodic=True),
        center=True,
        return_complex=True,
    ).abs()


def reference_top_bin(reference, margin_db=_REFERENCE_FLOOR_MARGIN_DB):
    """Return the highest bin where the reference still carries content.

    Anything above this is the recording's own noise floor, so the reference
    cannot judge whether content generated there is right or wrong.

    The cut is made against that noise floor rather than against the loudest
    bin. Music with a steep spectral slope sits far below its own fundamental
    while still carrying real content: a solo piano at 96 kHz measures 88 dB
    down at 16 kHz, so a peak-relative threshold would discard most of the band
    the recording can actually judge. A quiet quantile rather than the strict
    minimum keeps one dead bin, typically DC, from setting the floor.

    A reference whose spectrum never rises clear of its own floor is flat, which
    means it carries content everywhere, so the whole band is judgeable.
    """
    import torch

    bin_energy = _magnitude(reference).mean(dim=-1)
    if float(bin_energy.max()) <= 0:
        raise SystemExit("the reference contains no energy")
    floor = float(torch.quantile(bin_energy, _REFERENCE_FLOOR_QUANTILE))
    usable = (bin_energy > floor * 10 ** (margin_db / 20)).nonzero().flatten()
    if usable.numel() == 0:
        return int(bin_energy.numel() - 1)
    return int(usable[-1])


def ranking_is_unresolved(baseline_lsd, scores, margin=_MIN_USEFUL_IMPROVEMENT):
    """Report whether the run separated the configurations from doing nothing.

    A restoration that ties the degraded input has not been shown to help, and
    neither have the differences between restorations. Material whose missing
    band is noise rather than structure lands here: the model can put back the
    right amount of energy without putting back the right detail, and a
    log-spectral distance cannot tell that apart from silence. Requiring a
    margin rather than any improvement at all keeps a hair's-width win from
    reading as a result.
    """
    if not scores:
        return True
    return min(scores) >= baseline_lsd * (1.0 - margin)


def scored_band_distance(reference, estimate, low_bin, high_bin):
    """Return log-spectral distance across ``low_bin``..``high_bin`` inclusive.

    Both signals are level-matched first, so the metric reports spectral shape
    rather than the output stage's normalization.
    """
    import torch

    length = min(reference.shape[-1], estimate.shape[-1])
    reference_magnitude = _magnitude(reference[..., :length])
    estimate_magnitude = _magnitude(estimate[..., :length])
    frames = min(reference_magnitude.shape[-1], estimate_magnitude.shape[-1])
    reference_magnitude = reference_magnitude[:, :frames]
    estimate_magnitude = estimate_magnitude[:, :frames]

    error = (
        torch.log10(reference_magnitude**2 + _LSD_FLOOR)
        - torch.log10(estimate_magnitude**2 + _LSD_FLOOR)
    ) ** 2
    band = error[low_bin : high_bin + 1]
    if band.shape[0] == 0:
        return float("nan")
    # LSD averages the per-frame root-mean-square error across frequency.
    return float(torch.mean(torch.sqrt(torch.mean(band, dim=0))))


def band_energy_ratios(reference, estimate, low_bin, high_bin):
    """Return in-band and above-reference energy, relative to the scored band.

    ``in_band`` is 1.0 when the estimate matches the reference's energy across
    the scored band. ``above_reference`` measures what the estimate generated
    beyond the reference's own bandwidth; that is not an error, because the
    reference has nothing there to compare against.
    """
    import torch

    length = min(reference.shape[-1], estimate.shape[-1])
    reference_magnitude = _magnitude(reference[..., :length])
    estimate_magnitude = _magnitude(estimate[..., :length])
    frames = min(reference_magnitude.shape[-1], estimate_magnitude.shape[-1])
    reference_magnitude = reference_magnitude[:, :frames]
    estimate_magnitude = estimate_magnitude[:, :frames]

    anchor = float(torch.sum(reference_magnitude[low_bin : high_bin + 1] ** 2))
    if anchor <= 0:
        return float("nan"), float("nan")
    in_band = float(torch.sum(estimate_magnitude[low_bin : high_bin + 1] ** 2)) / anchor
    above = float(torch.sum(estimate_magnitude[high_bin + 1 :] ** 2)) / anchor
    return in_band, above


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    import numpy as np
    import soundfile as sf
    import torch

    from audiosr import build_model, super_resolution

    reference = load_reference(args.reference, args.offset_s, args.duration_s)
    degraded = degrade(reference, args.cutoff_hz)
    duration_s = reference.shape[-1] / _SAMPLE_RATE

    args.output_dir.mkdir(parents=True, exist_ok=True)
    degraded_path = args.output_dir / "input_degraded.wav"
    sf.write(
        os.fspath(degraded_path),
        degraded[0].numpy(),
        samplerate=_SAMPLE_RATE,
        subtype="PCM_24",
    )
    sf.write(
        os.fspath(args.output_dir / "reference.wav"),
        reference[0].numpy(),
        samplerate=_SAMPLE_RATE,
        subtype="PCM_24",
    )

    bin_hz = _SAMPLE_RATE / _LSD_N_FFT
    low_bin = int(args.cutoff_hz / bin_hz) + 1
    high_bin = reference_top_bin(reference)
    scored_span_hz = (high_bin - low_bin + 1) * bin_hz

    print(
        f"reference: {args.reference} | {duration_s:.2f}s mono @ {_SAMPLE_RATE} Hz | "
        f"degraded to {args.cutoff_hz} Hz"
    )
    print(
        f"reference carries content up to {high_bin * bin_hz:.0f} Hz; scoring "
        f"{args.cutoff_hz} Hz .. {high_bin * bin_hz:.0f} Hz"
    )
    if high_bin <= low_bin or scored_span_hz < _MIN_SCORED_BAND_HZ:
        raise SystemExit(
            f"the reference only leaves {max(scored_span_hz, 0):.0f} Hz above the "
            f"cutoff, which cannot separate configurations. Use a reference with "
            f"more bandwidth, or lower --cutoff_hz."
        )

    baseline_lsd = scored_band_distance(reference, degraded, low_bin, high_bin)
    baseline_band, _ = band_energy_ratios(reference, degraded, low_bin, high_bin)
    print(
        f"degraded input baseline: lsd={baseline_lsd:.3f} band_energy="
        f"{baseline_band:.3f} (a restoration must beat this lsd to be worth it)"
    )

    model = build_model(
        ckpt_path=args.ckpt_path, device=args.device, model_name=args.model_name
    )
    device = next(model.parameters()).device
    print(f"device: {device}")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def run(config):
        started = time.perf_counter()
        generated = super_resolution(
            model,
            os.fspath(degraded_path),
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            ddim_steps=config["ddim_steps"],
            sampler=config["sampler"],
            ddim_eta=config["ddim_eta"],
            discretize=config["discretize"],
        )
        synchronize()
        return generated, time.perf_counter() - started

    if not args.skip_warmup:
        warmup = dict(args.configs[0])
        warmup["ddim_steps"] = min(4, warmup["ddim_steps"])
        print(f"warm-up: {config_label(warmup)} (untimed)")
        run(warmup)

    results = []
    for config in args.configs:
        label = config_label(config)
        elapsed = []
        generated = None
        for _repeat in range(max(1, args.repeats)):
            generated, seconds = run(config)
            elapsed.append(seconds)

        restored = torch.from_numpy(np.asarray(generated, dtype=np.float32)).reshape(
            1, -1
        )
        lsd = scored_band_distance(reference, restored, low_bin, high_bin)
        band_energy, above_reference = band_energy_ratios(
            reference, restored, low_bin, high_bin
        )
        seconds = statistics.median(elapsed)

        output_path = args.output_dir / f"restored_{label.replace('/', '_')}.wav"
        sf.write(
            os.fspath(output_path),
            restored[0].numpy(),
            samplerate=_SAMPLE_RATE,
            subtype="PCM_24",
        )

        results.append(
            {
                "config": label,
                "sampler": config["sampler"],
                "ddim_steps": config["ddim_steps"],
                "ddim_eta": config["ddim_eta"],
                "discretize": config["discretize"],
                "seconds": seconds,
                "realtime_factor": duration_s / seconds if seconds > 0 else float("inf"),
                "lsd": lsd,
                "band_energy": band_energy,
                "above_reference": above_reference,
                "output": os.fspath(output_path),
            }
        )
        print(
            f"  {label:<30} {seconds:8.2f}s  x{duration_s / seconds:6.2f} realtime  "
            f"lsd={lsd:6.3f}  band_energy={band_energy:7.3f}  "
            f"above_ref={above_reference:6.3f}"
        )

    print()
    header = (
        f"{'config':<30}{'seconds':>10}{'xRT':>8}{'lsd':>9}"
        f"{'band_energy':>13}{'above_ref':>11}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'[degraded input]':<30}{'':>10}{'':>8}{baseline_lsd:>9.3f}"
        f"{baseline_band:>13.3f}{0.0:>11.3f}"
    )
    for row in sorted(results, key=lambda item: item["lsd"]):
        print(
            f"{row['config']:<30}{row['seconds']:>10.2f}{row['realtime_factor']:>8.2f}"
            f"{row['lsd']:>9.3f}{row['band_energy']:>13.3f}{row['above_reference']:>11.3f}"
        )

    if ranking_is_unresolved(baseline_lsd, [row["lsd"] for row in results]):
        print()
        print(
            "WARNING: no configuration beat the degraded input by a useful "
            f"margin ({_MIN_USEFUL_IMPROVEMENT:.0%}) inside the scored band. "
            "The reference cannot separate these configurations; treat the "
            "ranking as unresolved and measure against material whose missing "
            "band carries structure rather than noise."
        )

    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(
                {
                    "reference": os.fspath(args.reference),
                    "cutoff_hz": args.cutoff_hz,
                    "duration_s": duration_s,
                    "device": str(device),
                    "guidance_scale": args.guidance_scale,
                    "seed": args.seed,
                    "scored_band_hz": [args.cutoff_hz, high_bin * bin_hz],
                    "baseline": {"lsd": baseline_lsd, "band_energy": baseline_band},
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
