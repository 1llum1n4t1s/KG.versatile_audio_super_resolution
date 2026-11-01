"""Build aligned 48 kHz PCM training pairs from lossless recordings and MP3.

Run with --help for usage. This development tool never loads an AudioSR model.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


SAMPLE_RATE = 48_000
BITRATES = (64, 96, 128)
LOSSLESS_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".w64", ".caf"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New dataset directory; existing paths are never overwritten.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable with libmp3lame.")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run_ffmpeg(executable, arguments):
    result = subprocess.run(
        [executable, "-hide_banner", "-nostdin", "-v", "error", "-n", *arguments],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise ValueError(f"FFmpeg failed: {result.stderr.strip()[-3000:]}")
    return result.stdout


def check_ffmpeg(value):
    executable = shutil.which(value)
    if executable is None:
        raise ValueError("FFmpeg was not found; install it or pass --ffmpeg PATH")
    encoders = run_ffmpeg(executable, ["-encoders"])
    if "libmp3lame" not in encoders:
        raise ValueError("FFmpeg must include the libmp3lame encoder")
    version = run_ffmpeg(executable, ["-version"]).splitlines()[0]
    return executable, version


def read_reference(path):
    import numpy as np
    import soundfile as sf

    info = sf.info(path)
    if not (info.subtype.startswith("PCM_") or info.subtype in {"FLOAT", "DOUBLE"}):
        raise ValueError(f"{path}: expected lossless PCM/float audio, got {info.subtype}")
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or audio.shape[1] not in (1, 2):
        raise ValueError(f"{path}: MP3 pairs require nonempty mono or stereo audio")
    if not np.isfinite(audio).all():
        raise ValueError(f"{path}: NaN or Infinity in source audio")
    if rate != SAMPLE_RATE:
        import torch
        import torchaudio

        waveform = torch.from_numpy(np.ascontiguousarray(audio.T))
        # 高レート原音の折り返しを抑え、全bitrateで同じ正解PCMを共有する。
        audio = torchaudio.functional.resample(
            waveform, rate, SAMPLE_RATE,
            lowpass_filter_width=64, rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser", beta=14.769656459379492,
        ).T.contiguous().numpy()
    if not np.isfinite(audio).all():
        raise ValueError(f"{path}: resampling produced non-finite samples")
    peak = float(np.max(np.abs(audio)))
    gain = min(1.0, 0.99 / peak) if peak else 1.0
    # 正解と劣化音を別々に正規化しない。codec投入前の共通gainだけを記録する。
    audio = np.ascontiguousarray(audio * gain, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(f"{SAMPLE_RATE}:{audio.shape[1]}:".encode("ascii"))
    digest.update(memoryview(audio).cast("B"))
    return audio, info, gain, digest.hexdigest()


def verify_channel_alignment(reference, decoded):
    """Check zero lag in up to three energetic windows of one channel."""
    import numpy as np
    from scipy import signal

    source = reference.astype(np.float64)
    restored = decoded.astype(np.float64)
    if np.max(np.abs(source)) <= 1e-8:
        if np.max(np.abs(decoded)) > 1e-6:
            raise ValueError("Silent reference decoded to non-silent audio")
        return {"status": "silent", "lag_samples": None, "correlation": None}
    # 圧縮で消える高域を比較から外し、元から残る帯域で時刻だけを照合する。
    sos = signal.butter(4, 6000, fs=SAMPLE_RATE, output="sos")
    source = signal.sosfilt(sos, source)
    restored = signal.sosfilt(sos, restored)
    window = min(len(source), SAMPLE_RATE)
    starts = list(range(0, len(source), window))
    starts = [start for start in starts if len(source) - start >= window]
    # 冒頭・中央・末尾の各領域で最も鳴っている1秒を検査する。
    regions = np.array_split(starts, min(3, len(starts)))
    correlations = []
    for region in regions:
        start = max(region, key=lambda s: np.dot(source[s:s + window], source[s:s + window]))
        a = source[start:start + window].copy()
        b = restored[start:start + window].copy()
        a -= a.mean()
        b -= b.mean()
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm < 1e-16:
            continue
        correlation = signal.correlate(b, a, mode="full", method="fft") / norm
        center = len(a) - 1
        radius = min(2304, center)
        nearby = correlation[center - radius:center + radius + 1]
        lag = int(np.argmax(nearby)) - radius
        score = float(nearby[lag + radius])
        if score < 0.2:
            raise ValueError("Cannot verify time alignment: low reference/decoded correlation")
        if lag:
            raise ValueError(f"Decoded audio is shifted by {lag} samples; pair was rejected")
        correlations.append(score)
    if not correlations:
        raise ValueError("Cannot verify time alignment in non-silent audio")
    return {"status": "verified", "lag_samples": 0, "correlation": min(correlations)}


def verify_alignment(reference, decoded):
    # 左右を混ぜずに全chを検査し、逆相や片側だけのずれを見逃さない。
    channels = [verify_channel_alignment(reference[:, ch], decoded[:, ch])
                for ch in range(reference.shape[1])]
    scores = [channel["correlation"] for channel in channels
              if channel["status"] == "verified"]
    return {"status": "verified" if scores else "silent",
            "lag_samples": 0 if scores else None,
            "correlation": min(scores) if scores else None, "per_channel": channels}


def make_pair(ffmpeg, reference_path, encoded_path, decoded_path, reference, bitrate):
    import numpy as np
    import soundfile as sf

    run_ffmpeg(ffmpeg, [
        "-i", os.fspath(reference_path), "-map", "0:a:0", "-map_metadata", "-1",
        "-c:a", "libmp3lame", "-b:a", f"{bitrate}k", "-ar", str(SAMPLE_RATE),
        "-ac", str(reference.shape[1]), "-write_xing", "1", "-id3v2_version", "0",
        os.fspath(encoded_path),
    ])
    # seek可能なMP3を経由し、Xing/LAMEのdelay/padding情報をdecoderに渡す。
    run_ffmpeg(ffmpeg, [
        "-i", os.fspath(encoded_path), "-map", "0:a:0", "-map_metadata", "-1",
        "-c:a", "pcm_f32le", os.fspath(decoded_path),
    ])
    decoded, rate = sf.read(decoded_path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or decoded.shape != reference.shape:
        raise ValueError(
            f"{encoded_path}: decoded rate/shape {rate}/{decoded.shape} differs from "
            f"reference {SAMPLE_RATE}/{reference.shape}; no trim or padding was applied"
        )
    if not np.isfinite(decoded).all():
        raise ValueError(f"{encoded_path}: non-finite decoded samples")
    return verify_alignment(reference, decoded)


def split_for(recording_id, seed, validation_fraction, test_fraction):
    digest = hashlib.sha256(f"{seed}:{recording_id}".encode("ascii")).digest()
    position = int.from_bytes(digest, "big") / (1 << 256)
    if position < test_fraction:
        return "test"
    if position < test_fraction + validation_fraction:
        return "validation"
    return "train"


def write_metadata(path, metadata):
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(args):
    import soundfile as sf

    for value in (args.validation_fraction, args.test_fraction):
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("Split fractions must be finite and in [0, 1)")
    if args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")
    source_root = args.input_dir.resolve(strict=True)
    output_root = args.output_dir.resolve()
    if not source_root.is_dir():
        raise ValueError("--input-dir must be a directory")
    if output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError("Input and output directories must not contain each other")
    if output_root.exists():
        raise ValueError("--output-dir already exists; choose a new directory")
    paths = sorted(path for path in source_root.rglob("*")
                   if path.is_file() and path.suffix.lower() in LOSSLESS_EXTENSIONS)
    if not paths:
        raise ValueError("No supported lossless source files were found")
    ffmpeg, version = check_ffmpeg(args.ffmpeg)
    # 全入力のheaderを先に検査し、明らかな非対応入力では出力を作らない。
    for path in paths:
        info = sf.info(path)
        if not info.frames or info.channels not in (1, 2):
            raise ValueError(f"{path}: MP3 pairs require nonempty mono or stereo audio")
        if not (info.subtype.startswith("PCM_") or info.subtype in {"FLOAT", "DOUBLE"}):
            raise ValueError(f"{path}: source must be lossless PCM/float audio")
    output_root.mkdir(parents=True, exist_ok=False)
    for name in ("reference", "encoded", "degraded"):
        (output_root / name).mkdir()
    metadata = {
        "schema_version": 1, "status": "preparing", "sample_rate": SAMPLE_RATE,
        "codec": "mp3", "encoder": "libmp3lame", "bitrates_kbps": list(BITRATES),
        "bitrate_mode": "cbr", "bitrate_scope": "total_across_channels",
        "pcm_subtype": "FLOAT", "source_root": os.fspath(source_root),
        "ffmpeg_version": version, "seed": args.seed,
        "validation_fraction": args.validation_fraction, "test_fraction": args.test_fraction,
    }
    metadata_path = output_root / "dataset.json"
    write_metadata(metadata_path, metadata)
    entries = []
    seen = {}
    duplicates = []
    try:
        for index, path in enumerate(paths, 1):
            print(f"[{index}/{len(paths)}] {path.relative_to(source_root)}", flush=True)
            reference, info, gain, recording_id = read_reference(path)
            source_name = path.relative_to(source_root).as_posix()
            if recording_id in seen:
                duplicates.append({"source": source_name, "duplicate_of": seen[recording_id]})
                continue
            seen[recording_id] = source_name
            reference_path = output_root / "reference" / f"{recording_id}.wav"
            with sf.SoundFile(reference_path, mode="x", samplerate=SAMPLE_RATE,
                              channels=reference.shape[1], subtype="FLOAT") as handle:
                handle.write(reference)
            split = split_for(recording_id, args.seed, args.validation_fraction, args.test_fraction)
            for bitrate in BITRATES:
                filename = f"{recording_id}_{bitrate}k"
                encoded = output_root / "encoded" / f"{filename}.mp3"
                decoded = output_root / "degraded" / f"{filename}.wav"
                alignment = make_pair(ffmpeg, reference_path, encoded, decoded, reference, bitrate)
                entries.append({
                    "schema_version": 1, "recording_id": recording_id, "split": split,
                    "source": source_name, "source_sample_rate": info.samplerate,
                    "reference": reference_path.relative_to(output_root).as_posix(),
                    "degraded": decoded.relative_to(output_root).as_posix(),
                    "encoded": encoded.relative_to(output_root).as_posix(),
                    "codec": "mp3", "bitrate_kbps": bitrate, "sample_rate": SAMPLE_RATE,
                    "channels": reference.shape[1], "num_samples": len(reference),
                    "reference_gain": gain, "alignment": alignment,
                })
        with (output_root / "manifest.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        counts = {split: sum(entry["split"] == split for entry in entries)
                  for split in ("train", "validation", "test")}
        metadata.update(status="complete", recordings=len(seen), pairs=len(entries),
                        split_pair_counts=counts, duplicates=duplicates)
        write_metadata(metadata_path, metadata)
    except BaseException as exc:
        metadata.update(status="failed", error=str(exc) or type(exc).__name__)
        write_metadata(metadata_path, metadata)
        raise
    print(f"Created {len(entries)} pairs from {len(seen)} recordings: {output_root}", flush=True)
    if not counts["validation"] or not counts["test"]:
        print("Note: a split is empty; hash-based fractions are approximate for small datasets.")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        prepare(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
