"""Small Gradio front-end for AudioSR.

The long-audio path deliberately keeps chunk orchestration here while the
model-specific work is provided by ``audiosr.pipeline.super_resolution_batch``.
Keeping those concerns separate makes the chunking behaviour testable without
loading a model or downloading a checkpoint.
"""

from __future__ import annotations

import gc
import threading
from typing import Any, Iterable

import gradio as gr
import librosa
import numpy as np


# Resolve model code only when inference starts. Importing this UI module must
# not initialize tokenizers or require an accelerator runtime.
super_resolution_batch = None
load_audio = None
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _release_cached_model() -> None:
    """Drop the cached model before loading a different checkpoint."""

    previous_models = tuple(_MODEL_CACHE.values())
    _MODEL_CACHE.clear()
    del previous_models
    gc.collect()

    try:
        import torch
    except ImportError:  # pragma: no cover - UI-only import environments
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_module = getattr(torch, "mps", None)
    if (
        mps_backend is not None
        and mps_module is not None
        and mps_backend.is_available()
        and callable(getattr(mps_module, "empty_cache", None))
    ):
        mps_module.empty_cache()


def _get_model(model_name: str):
    """Reuse one heavyweight model and serialize cache misses."""

    with _MODEL_CACHE_LOCK:
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]

        _release_cached_model()
        from audiosr import build_model

        model = build_model(model_name=model_name)
        _MODEL_CACHE[model_name] = model
        return model


def calculate_amplitude_stats(audio: np.ndarray) -> tuple[float, float]:
    """Calculate RMS and peak amplitude for a one-dimensional waveform."""

    audio = _as_1d(audio)
    if audio.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    return rms, peak


def normalize_chunk_amplitude(
    processed_chunk: np.ndarray, original_chunk: np.ndarray
) -> np.ndarray:
    """Match a processed chunk's amplitude to its source chunk."""

    processed_chunk = _as_1d(processed_chunk)
    original_chunk = _as_1d(original_chunk)
    if processed_chunk.size == 0 or original_chunk.size == 0:
        return processed_chunk

    orig_rms, orig_peak = calculate_amplitude_stats(original_chunk)
    proc_rms, proc_peak = calculate_amplitude_stats(processed_chunk)
    if proc_rms < 1e-8:
        return processed_chunk

    scale_factor = orig_rms / proc_rms
    peak_ratio = orig_peak / proc_peak if proc_peak > 0 else 1.0
    return processed_chunk * min(scale_factor, peak_ratio)


def _as_1d(audio: Any) -> np.ndarray:
    """Convert numpy/torch-like model output to the batch API's 1-D contract."""

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    array = np.asarray(audio)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    squeezed = np.squeeze(array)
    if squeezed.ndim == 1:
        return squeezed
    # The pipeline promises one waveform per list entry.  This fallback keeps
    # compatibility with old [batch, channel, samples] model outputs without
    # accidentally using the batch dimension as the sample count.
    return np.asarray(squeezed).reshape(-1)


def _batch_results(
    audiosr: Any,
    chunks: Iterable[np.ndarray],
    guidance_scale: float,
    ddim_steps: int,
    seed: int = 42,
    lowpass_seed: int | None = None,
) -> list[np.ndarray]:
    """Run the pipeline batch function and normalize its return container."""

    chunk_list = [_as_1d(chunk) for chunk in chunks]
    if not chunk_list:
        return []
    global super_resolution_batch
    if super_resolution_batch is None:
        try:
            from audiosr.pipeline import super_resolution_batch as batch_function
        except ImportError as exc:  # pragma: no cover - old installed package
            raise RuntimeError(
                "audiosr.pipeline.super_resolution_batch is required for the app"
            ) from exc
        super_resolution_batch = batch_function

    # Keep the positional order matching the public pipeline contract:
    # (model, waveforms, seed, ddim_steps, guidance_scale).
    batch_kwargs = {}
    if lowpass_seed is not None:
        batch_kwargs["lowpass_seed"] = int(lowpass_seed)
    raw_results: Any = super_resolution_batch(
        audiosr,
        chunk_list,
        seed,
        ddim_steps,
        guidance_scale,
        **batch_kwargs,
    )
    if isinstance(raw_results, np.ndarray):
        if raw_results.ndim == 1:
            results = [raw_results]
        else:
            results = [raw_results[index] for index in range(raw_results.shape[0])]
    else:
        results = list(raw_results)

    if len(results) != len(chunk_list):
        raise ValueError(
            "super_resolution_batch returned "
            f"{len(results)} results for {len(chunk_list)} chunks"
        )
    return [_as_1d(result) for result in results]


def _bounded_ddim_steps(
    ddim_steps: int, chunk_length: int, sr: int, is_last_chunk: bool
) -> int:
    """Keep the user-visible DDIM value within the sampler's valid range."""

    del chunk_length, sr, is_last_chunk
    return min(max(int(ddim_steps), 1), 1000)


def _trim_and_normalize(
    result: np.ndarray, chunk: np.ndarray, target_length: int | None = None
) -> np.ndarray:
    """Trim model padding to the source length and restore source amplitude."""

    result = _as_1d(result)
    chunk = _as_1d(chunk)
    if result.size == 0:
        return result

    # AudioSR preserves the 48 kHz sample rate.  The batch pipeline already
    # returns outputs trimmed to each input length, but retaining this guard is
    # important for older pipelines and makes padding harmless.
    expected_length = len(chunk)
    if target_length is not None:
        expected_length = min(expected_length, max(int(target_length), 0))
    if expected_length == 0:
        return result[:0]
    result = result[:expected_length]
    return normalize_chunk_amplitude(result, chunk[: len(result)])


def process_chunk(
    audiosr: Any,
    chunk: np.ndarray,
    sr: int,
    guidance_scale: float,
    ddim_steps: int,
    is_last_chunk: bool = False,
    target_length: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Process one chunk through the same in-memory batch path as long audio."""

    chunk = _as_1d(chunk)
    adjusted_ddim_steps = _bounded_ddim_steps(
        ddim_steps, len(chunk), sr, is_last_chunk
    )
    result = _batch_results(
        audiosr,
        [chunk],
        guidance_scale,
        adjusted_ddim_steps,
        seed=int(seed),
    )[0]
    result = _trim_and_normalize(result, chunk, target_length)

    return result


def _crossfade_chunks(
    processed_chunks: list[np.ndarray], requested_overlap: int
) -> np.ndarray:
    """Crossfade processed chunks, skipping empty/effectively zero overlaps."""

    if not processed_chunks:
        return np.empty(0, dtype=np.float32)

    merged: list[np.ndarray] = []
    for index, chunk in enumerate(processed_chunks):
        current = _as_1d(chunk).copy()
        if index > 0 and merged:
            previous = merged[-1]
            effective_overlap = min(
                max(int(requested_overlap), 0),
                int(current.shape[-1]),
                int(previous.shape[-1]),
            )
            if effective_overlap > 0:
                fade_in = np.linspace(0.0, 1.0, effective_overlap)
                fade_out = np.linspace(1.0, 0.0, effective_overlap)
                current_overlap = current[:effective_overlap]
                previous_overlap = previous[-effective_overlap:]

                current_rms = float(np.sqrt(np.mean(np.square(current_overlap))))
                previous_rms = float(np.sqrt(np.mean(np.square(previous_overlap))))
                if current_rms > 0 and previous_rms > 0:
                    fade_in = fade_in * np.sqrt(previous_rms / current_rms)

                previous[-effective_overlap:] = (
                    previous_overlap * fade_out + current_overlap * fade_in
                )
                current = current[effective_overlap:]
        merged.append(current)

    return np.concatenate(merged, axis=0)


def process_audio_channel(
    audiosr: Any,
    audio_channel: np.ndarray,
    sr: int,
    guidance_scale: float,
    ddim_steps: int,
    batch_size: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Process one channel in bounded batches while preserving chunk order."""

    if not isinstance(batch_size, (int, np.integer)) or not 1 <= int(batch_size) <= 8:
        raise ValueError("batch_size must be an integer between 1 and 8")
    batch_size = int(batch_size)

    audio_channel = _as_1d(audio_channel)
    if audio_channel.size == 0:
        return audio_channel.copy()

    chunk_size = int(round(5.1 * sr))
    overlap_size = int(round(0.5 * sr))
    if chunk_size <= 0:
        raise ValueError("sample rate must be positive")
    step_size = max(chunk_size - overlap_size, 1)
    total_samples = len(audio_channel)
    num_chunks = int(np.ceil(total_samples / step_size))

    chunks: list[np.ndarray] = []
    for index in range(num_chunks):
        start = index * step_size
        end = min(start + chunk_size, total_samples)
        chunks.append(audio_channel[start:end])
        if end == total_samples:
            break

    processed_results: list[np.ndarray] = []
    for batch_start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_start : batch_start + batch_size]
        includes_last = batch_start + len(batch_chunks) == len(chunks)
        # A batch API has one DDIM-step value for the entire batch. Preserve
        # the value selected in the UI for every chunk in that batch.
        batch_steps = _bounded_ddim_steps(
            ddim_steps,
            len(batch_chunks[-1]),
            sr,
            includes_last,
        )
        print(
            f"Processing chunks {batch_start + 1}-"
            f"{batch_start + len(batch_chunks)}/{len(chunks)} "
            f"(batch_size={len(batch_chunks)})"
        )
        batch_results = _batch_results(
            audiosr,
            batch_chunks,
            guidance_scale,
            batch_steps,
            seed=int(seed) + batch_start,
            lowpass_seed=int(seed),
        )
        for result, chunk in zip(batch_results, batch_chunks):
            processed_results.append(_trim_and_normalize(result, chunk))

    return _crossfade_chunks(processed_results, overlap_size)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Scale down clipped audio while preserving an in-range signal's level."""

    audio = np.asarray(audio)
    if audio.size == 0:
        return audio
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        return audio / max_val
    return audio


def convert_audio_for_gradio(audio: np.ndarray) -> np.ndarray:
    """Convert output to Gradio's float32 waveform representation."""

    audio = np.asarray(audio, dtype=np.float32)
    return normalize_audio(audio).astype(np.float32, copy=False)


def inference(
    audio_file: str,
    model_name: str,
    guidance_scale: float,
    ddim_steps: int,
    batch_size: int = 1,
    seed: int = 42,
):
    """Load, process, and return mono or multi-channel audio without reshaping it."""

    audiosr = _get_model(model_name)
    global load_audio
    if load_audio is None:
        from audiosr.utils import load_audio as canonical_load_audio

        load_audio = canonical_load_audio

    audio, sr = load_audio(audio_file, target_sample_rate=48000)
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 2:
        raise ValueError(f"expected channels-first audio, got {audio.shape}")
    channels = [audio[index] for index in range(audio.shape[0])]

    processed_channels = []
    for channel_index, channel in enumerate(channels):
        processed_channels.append(
            process_audio_channel(
                audiosr,
                channel,
                sr,
                guidance_scale,
                ddim_steps,
                batch_size=batch_size,
                seed=int(seed) + channel_index * 1_000_003,
            )
        )

    if len(processed_channels) == 1:
        final_audio = processed_channels[0]
    else:
        lengths = {len(channel) for channel in processed_channels}
        if len(lengths) != 1:
            raise ValueError("processed channels have different sample lengths")
        # Gradio expects samples on the first axis for multi-channel numpy
        # audio.  Channel count and ordering are retained exactly.
        final_audio = np.stack(processed_channels, axis=1)

    final_audio = convert_audio_for_gradio(final_audio)
    return (sr, final_audio)


def create_interface() -> gr.Interface:
    """Build the UI without launching it (safe for imports and tests)."""

    return gr.Interface(
        fn=inference,
        inputs=[
            gr.Audio(type="filepath", label="Input Audio"),
            gr.Dropdown(["basic", "speech"], value="basic", label="Model"),
            gr.Slider(1, 10, value=2.6, step=0.1, label="Guidance Scale"),
            gr.Slider(1, 100, value=50, step=1, label="DDIM Steps"),
            gr.Slider(1, 8, value=1, step=1, label="Batch Size"),
            gr.Number(value=42, precision=0, label="Seed"),
        ],
        outputs=gr.Audio(type="numpy", label="Output Audio"),
        title="AudioSR",
        description="Audio Super Resolution with AudioSR",
    )


if __name__ == "__main__":
    iface = create_interface()
    iface.launch()
