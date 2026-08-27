import os
import re
from collections.abc import Mapping

import yaml
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F

import audiosr.latent_diffusion.modules.phoneme_encoder.text as text
from audiosr.latent_diffusion.models.ddpm import LatentDiffusion
from audiosr.sampling import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_SAMPLER,
    normalize_ddim_eta,
    normalize_discretize,
    normalize_sampler,
)
from audiosr.utils import (
    _select_lowpass_filter_type,
    default_audioldm_config,
    download_checkpoint,
    load_audio,
    seed_everything,
    read_audio_file,
    lowpass_filtering_prepare_inference,
    wav_feature_extraction,
    normalize_wav,
    pad_wav,
)


_SAMPLE_RATE = 48000
_SEGMENT_SAMPLES = 245760  # 5.12 seconds at the model's 48 kHz rate


def text2phoneme(data):
    return text._clean_text(re.sub(r"<.*?>", "", data), ["english_cleaners2"])


def text_to_filename(text):
    return text.replace(" ", "_").replace("'", "_").replace('"', "_")


def extract_kaldi_fbank_feature(waveform, sampling_rate, log_mel_spec):
    norm_mean = -4.2677393
    norm_std = 4.5689974

    if sampling_rate != 16000:
        waveform_16k = torchaudio.functional.resample(
            waveform, orig_freq=sampling_rate, new_freq=16000
        )
    else:
        waveform_16k = waveform

    waveform_16k = waveform_16k - waveform_16k.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform_16k,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )

    TARGET_LEN = log_mel_spec.size(0)

    # cut and pad
    n_frames = fbank.shape[0]
    p = TARGET_LEN - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[:TARGET_LEN, :]

    fbank = (fbank - norm_mean) / (norm_std * 2)

    return {"ta_kaldi_fbank": fbank}  # [1024, 128]


def _as_mono_numpy(waveform):
    """Return one mono waveform as a float32 NumPy array."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim != 1:
        raise ValueError("waveform must be a mono one-dimensional array")
    if waveform.size == 0:
        raise ValueError("waveform must contain at least one sample")
    return waveform


def _padded_sample_count(sample_count):
    if sample_count <= 0:
        raise ValueError("waveform must contain at least one sample")
    return ((int(sample_count) + _SEGMENT_SAMPLES - 1) // _SEGMENT_SAMPLES) * _SEGMENT_SAMPLES


def _prepare_mono_batch(
    waveform,
    padded_samples,
    add_batch_dimension=True,
    lowpass_filter_type=None,
):
    """Build model features for one normalized mono waveform.

    ``padded_samples`` is explicit so callers processing several waveforms can
    use one common feature shape before stacking them into a model batch.
    """
    waveform = _as_mono_numpy(waveform)
    original_samples = waveform.size
    if padded_samples < original_samples:
        raise ValueError("padded_samples must not be shorter than waveform")

    duration = original_samples / _SAMPLE_RATE
    target_frame = int(padded_samples) // 480
    normalized = normalize_wav(waveform)
    normalized = pad_wav(normalized, target_length=int(padded_samples))
    normalized = np.asarray(normalized, dtype=np.float32)
    if normalized.ndim == 1:
        normalized = normalized[None, :]

    log_mel_spec, stft = wav_feature_extraction(
        torch.from_numpy(normalized), target_frame
    )
    batch = {
        "waveform": torch.as_tensor(normalized, dtype=torch.float32),
        "stft": torch.as_tensor(stft, dtype=torch.float32),
        "log_mel_spec": torch.as_tensor(log_mel_spec, dtype=torch.float32),
        "sampling_rate": _SAMPLE_RATE,
    }
    batch.update(
        lowpass_filtering_prepare_inference(
            batch, filter_type=lowpass_filter_type
        )
    )
    if "waveform_lowpass" not in batch:
        raise RuntimeError("lowpass feature preparation did not return waveform_lowpass")

    lowpass_mel, _ = wav_feature_extraction(batch["waveform_lowpass"], target_frame)
    batch["lowpass_mel"] = lowpass_mel

    if add_batch_dimension:
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.float().unsqueeze(0)

    return batch, duration


def make_batch_for_super_resolution(
    input_file,
    waveform=None,
    fbank=None,
    lowpass_filter_type=None,
):
    if waveform is None:
        log_mel_spec, stft, waveform, duration, target_frame = read_audio_file(input_file)
        batch = {
            "waveform": torch.as_tensor(waveform, dtype=torch.float32),
            "stft": torch.as_tensor(stft, dtype=torch.float32),
            "log_mel_spec": torch.as_tensor(log_mel_spec, dtype=torch.float32),
            "sampling_rate": _SAMPLE_RATE,
        }
        batch.update(
            lowpass_filtering_prepare_inference(
                batch, filter_type=lowpass_filter_type
            )
        )
        if "waveform_lowpass" not in batch:
            raise RuntimeError("lowpass feature preparation did not return waveform_lowpass")
        lowpass_mel, _ = wav_feature_extraction(batch["waveform_lowpass"], target_frame)
        batch["lowpass_mel"] = lowpass_mel
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.float().unsqueeze(0)
        return batch, duration

    waveform = _as_mono_numpy(waveform)
    padded_samples = _padded_sample_count(waveform.size)
    return _prepare_mono_batch(
        waveform,
        padded_samples,
        add_batch_dimension=True,
        lowpass_filter_type=lowpass_filter_type,
    )


def round_up_duration(duration):
    return int(round(duration / 2.5) + 1) * 2.5


_VAE_FEATURE_EXTRACT = (
    "audiosr.latent_diffusion.modules.encoders.modules.VAEFeatureExtract"
)


def _read_state_dict(ckpt_path):
    if os.fsdecode(ckpt_path).lower().endswith(".safetensors"):
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device="cpu")
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _tensors_match(left, right):
    if right is None or left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(torch.equal(left, right))


def duplicated_first_stage_cond(config, state_dict):
    """Report conditioning stages whose weights repeat the first stage's.

    AudioSR conditions on a VAE encode of the low band, and configures that VAE
    a second time inside the conditioning stage. In the released checkpoints
    both copies are the same tensors, so one module can answer for both names
    and the duplicate is never built, moved to the accelerator, or held.

    The decision comes from the checkpoint rather than from the configuration,
    because two stages can be configured alike and still have been trained
    apart. A checkpoint whose copies differ keeps the separate modules it needs.

    Returns the conditioning stage keys that can share, together with the
    checkpoint keys that become redundant once they do.
    """
    prefix = "first_stage_model."
    first_stage = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not first_stage:
        return (), ()

    shared, redundant = [], []
    cond_stage_config = config["model"]["params"].get("cond_stage_config") or {}
    for index, cond_key in enumerate(cond_stage_config):
        if cond_stage_config[cond_key].get("target") != _VAE_FEATURE_EXTRACT:
            continue
        cond_prefix = f"cond_stage_models.{index}.vae."
        cond_keys = [key for key in state_dict if key.startswith(cond_prefix)]
        if len(cond_keys) != len(first_stage):
            continue
        if all(
            _tensors_match(state_dict[key], first_stage.get(key[len(cond_prefix) :]))
            for key in cond_keys
        ):
            shared.append(cond_key)
            redundant.extend(cond_keys)
    return tuple(shared), tuple(redundant)


def build_model(ckpt_path=None, config=None, device=None, model_name="basic"):
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print("Loading AudioSR: %s" % model_name)
    print("Loading model on %s" % device)

    if ckpt_path is None:
        ckpt_path = download_checkpoint(model_name)
    else:
        # os.fspath keeps pathlib.Path and other PathLike values supported while
        # preserving an explicitly supplied checkpoint instead of downloading one.
        ckpt_path = os.fspath(ckpt_path)

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config(model_name)

    # # Use text as condition instead of using waveform during training
    config["model"]["params"]["device"] = device
    # config["model"]["params"]["cond_stage_key"] = "text"

    # The checkpoint is read before the model is built so the duplicated
    # conditioning VAE can be dropped from both the weights and the model.
    state_dict = _read_state_dict(ckpt_path)
    shared, redundant = duplicated_first_stage_cond(config, state_dict)
    for key in redundant:
        del state_dict[key]
    config["model"]["params"]["share_first_stage_cond"] = shared

    # No normalization here
    latent_diffusion = LatentDiffusion(**config["model"]["params"])
    latent_diffusion.load_state_dict(state_dict, strict=False)

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)

    return latent_diffusion


def _as_generated_tensor(generated):
    if isinstance(generated, np.ndarray):
        generated = torch.from_numpy(generated)
    if not isinstance(generated, torch.Tensor):
        generated = torch.as_tensor(generated)
    return generated.detach().cpu().float()


def _extract_single_generated(generated):
    generated = _as_generated_tensor(generated)
    if generated.ndim == 3:
        if generated.shape[0] != 1:
            raise ValueError("single-waveform generation must have batch size 1")
        generated = generated[0]
    if generated.ndim == 2:
        if generated.shape[0] != 1:
            raise ValueError("single-waveform generation must have one audio channel")
        generated = generated[0]
    if generated.ndim != 1:
        raise ValueError("generated waveform must have shape [samples], [1, samples], or [1, 1, samples]")
    return generated


def _extract_batch_generated(generated, index, batch_size):
    generated = _as_generated_tensor(generated)
    if generated.ndim == 3:
        if generated.shape[0] != batch_size or generated.shape[1] != 1:
            raise ValueError("batched generation must have shape [batch, 1, samples]")
        return generated[index, 0]
    if generated.ndim == 2:
        if generated.shape[0] != batch_size:
            raise ValueError("batched generation must have one item per input waveform")
        return generated[index]
    if generated.ndim == 1 and batch_size == 1:
        return generated
    raise ValueError("generated batch must have shape [batch, samples] or [batch, 1, samples]")


def _trim_or_pad(waveform, sample_count):
    waveform = torch.as_tensor(waveform, dtype=torch.float32).flatten()
    if waveform.numel() < sample_count:
        waveform = F.pad(waveform, (0, sample_count - waveform.numel()))
    return waveform[:sample_count]


def _fade_pair(overlap_samples, dtype=torch.float32):
    if overlap_samples <= 0:
        return None, None
    window = torch.hann_window(2 * overlap_samples, periodic=False, dtype=dtype)
    return window[:overlap_samples], window[overlap_samples:]


def _is_accelerator_out_of_memory(error):
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _clear_accelerator_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_long_audio_batch(
    latent_diffusion,
    chunks,
    seed,
    lowpass_seed,
    ddim_steps,
    guidance_scale,
    sampler=DEFAULT_SAMPLER,
    ddim_eta=1.0,
    discretize=DEFAULT_DISCRETIZATION,
):
    """Generate one chunk group and retry as single items after accelerator OOM."""
    try:
        return super_resolution_batch(
            latent_diffusion,
            chunks,
            seed=seed,
            lowpass_seed=lowpass_seed,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            sampler=sampler,
            ddim_eta=ddim_eta,
            discretize=discretize,
        )
    except RuntimeError as error:
        if len(chunks) <= 1 or not _is_accelerator_out_of_memory(error):
            raise

        _clear_accelerator_cache()
        generated = []
        for offset, chunk in enumerate(chunks):
            generated.extend(
                super_resolution_batch(
                    latent_diffusion,
                    [chunk],
                    seed=int(seed) + offset,
                    lowpass_seed=lowpass_seed,
                    ddim_steps=ddim_steps,
                    guidance_scale=guidance_scale,
                    sampler=sampler,
                    ddim_eta=ddim_eta,
                    discretize=discretize,
                )
            )
        return generated


def _blend_long_audio_chunk(
    final_waveform,
    contribution_map,
    channel_index,
    start_sample,
    end_sample,
    overlap_samples,
    processed,
    original_peak,
):
    current_chunk_len = end_sample - start_sample
    processed = _trim_or_pad(processed, current_chunk_len)
    processed_peak = torch.max(torch.abs(processed)) + 1e-8
    processed = (processed / processed_peak) * original_peak

    left_overlap = min(overlap_samples, current_chunk_len) if start_sample > 0 else 0
    right_overlap = (
        min(overlap_samples, current_chunk_len)
        if end_sample < final_waveform.shape[-1]
        else 0
    )
    if left_overlap:
        fade_in, _ = _fade_pair(left_overlap, processed.dtype)
        processed[:left_overlap] *= fade_in
    if right_overlap:
        _, fade_out = _fade_pair(right_overlap, processed.dtype)
        processed[-right_overlap:] *= fade_out

    final_waveform[0, channel_index, start_sample:end_sample] += processed
    contribution = torch.ones(current_chunk_len, dtype=processed.dtype)
    if left_overlap:
        contribution[:left_overlap] *= _fade_pair(left_overlap, processed.dtype)[0]
    if right_overlap:
        contribution[-right_overlap:] *= _fade_pair(right_overlap, processed.dtype)[1]
    contribution_map[0, channel_index, start_sample:end_sample] += contribution


_HIGH_RATE_N_FFT = 4096
_HIGH_RATE_HOP = _HIGH_RATE_N_FFT // 4
_MODEL_NYQUIST_HZ = _SAMPLE_RATE // 2


def _level_match_gain(source, reference, floor=1e-12):
    """Return the gain that puts ``source`` at ``reference``'s overall level."""
    source_power = float(torch.mean(source.double() ** 2))
    reference_power = float(torch.mean(reference.double() ** 2))
    if source_power <= floor:
        return 1.0
    return (reference_power / source_power) ** 0.5


def restore_high_rate(generated, source_file, target_sample_rate=None):
    """Return the restoration at the source's rate, keeping the source's top band.

    The model works at 48 kHz, so a source recorded above that loses everything
    over 24 kHz on the way in and the restoration comes back at 48 kHz.
    Resampling it up recreates the rate but leaves that band empty, so the
    source's own content above 24 kHz is put back: the same splice the model
    already performs at the low end, applied at the top.

    Nothing is invented. The band above 24 kHz either came from the source or
    stays empty, and on real recordings it usually holds only the noise floor —
    a 96 kHz 24-bit solo piano recording measures a flat -91 dB up there. What
    this preserves is the source's rate and whatever it genuinely carried, not
    added bandwidth.

    Returns the waveform as ``[1, channels, samples]`` together with its rate.
    """
    import torchaudio

    restored = _as_generated_tensor(generated)
    if restored.dim() == 3:
        if restored.size(0) != 1:
            raise ValueError(
                "generated must hold a single item, got batch of "
                f"{restored.size(0)}"
            )
        restored = restored[0]
    if restored.dim() != 2:
        raise ValueError(
            "generated must have shape [channels, samples] or "
            f"[1, channels, samples], got {tuple(restored.shape)}"
        )

    source, source_rate = load_audio(source_file)
    target = int(source_rate if target_sample_rate is None else target_sample_rate)
    if target <= 0:
        raise ValueError("target_sample_rate must be positive")
    if source.size(0) != restored.size(0):
        raise ValueError(
            f"the source has {source.size(0)} channels but the restoration has "
            f"{restored.size(0)}"
        )

    if target != _SAMPLE_RATE:
        restored = torchaudio.functional.resample(
            restored, orig_freq=_SAMPLE_RATE, new_freq=target
        )
    if target <= _SAMPLE_RATE:
        # The source held nothing above the model's own Nyquist, so the rate
        # change is the whole of the work.
        return restored.unsqueeze(0).cpu().numpy(), target

    if source_rate != target:
        source = torchaudio.functional.resample(
            source, orig_freq=source_rate, new_freq=target
        )

    # The source file decides the length, since that is what the caller handed
    # in; resampling rounds the restoration to within a sample or two of it.
    length = source.size(-1)
    if restored.size(-1) < length:
        restored = torch.nn.functional.pad(restored, (0, length - restored.size(-1)))
    else:
        restored = restored[..., :length]

    # The model level-matches its output to the source, but not exactly, so the
    # band being spliced in is put on the restoration's scale first. Overall
    # level is dominated by the low end, where the two already agree.
    source = source * _level_match_gain(source, restored)

    window = torch.hann_window(_HIGH_RATE_N_FFT, periodic=True, dtype=restored.dtype)
    spectrum = torch.stft(
        restored,
        n_fft=_HIGH_RATE_N_FFT,
        hop_length=_HIGH_RATE_HOP,
        window=window,
        center=True,
        return_complex=True,
    )
    source_spectrum = torch.stft(
        source,
        n_fft=_HIGH_RATE_N_FFT,
        hop_length=_HIGH_RATE_HOP,
        window=window,
        center=True,
        return_complex=True,
    )
    crossover = int(_MODEL_NYQUIST_HZ * _HIGH_RATE_N_FFT / target)
    spectrum[:, crossover + 1 :] = source_spectrum[:, crossover + 1 :]
    output = torch.istft(
        spectrum,
        n_fft=_HIGH_RATE_N_FFT,
        hop_length=_HIGH_RATE_HOP,
        window=window,
        center=True,
        length=length,
    )
    return output.unsqueeze(0).cpu().numpy(), target


def super_resolution(
    latent_diffusion,
    input_file,
    seed=42,
    ddim_steps=200,
    guidance_scale=3.5,
    latent_t_per_second=12.8,
    config=None,
    sampler=DEFAULT_SAMPLER,
    ddim_eta=1.0,
    discretize=DEFAULT_DISCRETIZATION,
):
    sampler = normalize_sampler(sampler)
    ddim_eta = normalize_ddim_eta(ddim_eta)
    discretize = normalize_discretize(discretize)
    waveform, sampling_rate = load_audio(input_file, target_sample_rate=_SAMPLE_RATE)
    if sampling_rate != _SAMPLE_RATE:
        raise ValueError(f"load_audio returned {sampling_rate} Hz; expected {_SAMPLE_RATE} Hz")
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("load_audio must return waveform with shape [channels, samples]")

    original_samples = waveform.shape[-1]
    outputs = []
    for channel in waveform:
        # Reset the seed for every channel so channel processing follows the
        # same stochastic path while retaining the existing mono model path.
        seed_everything(int(seed))
        batch, duration = make_batch_for_super_resolution(
            None, waveform=channel.detach().cpu().numpy()
        )
        with torch.inference_mode():
            generated = latent_diffusion.generate_batch(
                batch,
                unconditional_guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                sampler=sampler,
                ddim_eta=ddim_eta,
                discretize=discretize,
            )
        outputs.append(_trim_or_pad(_extract_single_generated(generated), original_samples))

    return torch.stack(outputs, dim=0).unsqueeze(0).cpu().numpy()


def super_resolution_long_audio(
    latent_diffusion,
    input_file,
    seed=42,
    ddim_steps=200,
    guidance_scale=3.5,
    chunk_duration_s=15,
    overlap_duration_s=2,
    batch_size=1,
    sampler=DEFAULT_SAMPLER,
    ddim_eta=1.0,
    discretize=DEFAULT_DISCRETIZATION,
):
    """
    Process a multi-channel file in overlapping chunks and return [1, C, T].
    """
    sampler = normalize_sampler(sampler)
    ddim_eta = normalize_ddim_eta(ddim_eta)
    discretize = normalize_discretize(discretize)
    if chunk_duration_s <= 0 or overlap_duration_s < 0:
        raise ValueError(
            "chunk_duration_s must be positive and overlap_duration_s non-negative"
        )
    if overlap_duration_s >= chunk_duration_s:
        raise ValueError("overlap_duration_s must be less than chunk_duration_s")
    if not isinstance(batch_size, (int, np.integer)) or not 1 <= int(batch_size) <= 8:
        raise ValueError("batch_size must be an integer between 1 and 8")
    batch_size = int(batch_size)

    waveform, sampling_rate = load_audio(input_file, target_sample_rate=_SAMPLE_RATE)
    if sampling_rate != _SAMPLE_RATE:
        raise ValueError(f"load_audio returned {sampling_rate} Hz; expected {_SAMPLE_RATE} Hz")
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("load_audio must return waveform with shape [channels, samples]")
    channels, total_samples = waveform.shape
    if total_samples <= 0:
        raise ValueError("input audio must contain at least one sample")

    chunk_samples = int(round(chunk_duration_s * _SAMPLE_RATE))
    overlap_samples = int(round(overlap_duration_s * _SAMPLE_RATE))
    if chunk_samples <= 0 or overlap_samples < 0 or overlap_samples >= chunk_samples:
        raise ValueError("chunk_duration_s and overlap_duration_s are too small")
    step_samples = chunk_samples - overlap_samples

    final_waveform = torch.zeros(
        (1, channels, total_samples), dtype=torch.float32
    )
    contribution_map = torch.zeros_like(final_waveform)

    for channel_index in range(channels):
        # Reset per channel to make stochastic processing independent but
        # reproducible across channels.
        seed_everything(int(seed))
        lowpass_filter_type = _select_lowpass_filter_type()
        chunk_records = []
        for start_sample in range(0, total_samples, step_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = waveform[channel_index, start_sample:end_sample]
            chunk_records.append(
                (
                    start_sample,
                    end_sample,
                    chunk,
                    torch.max(torch.abs(chunk)) + 1e-8,
                )
            )

        if batch_size == 1:
            for start_sample, end_sample, chunk, original_peak in chunk_records:
                batch, duration = make_batch_for_super_resolution(
                    None,
                    waveform=chunk.detach().cpu().numpy(),
                    lowpass_filter_type=lowpass_filter_type,
                )
                with torch.inference_mode():
                    generated = latent_diffusion.generate_batch(
                        batch,
                        unconditional_guidance_scale=guidance_scale,
                        ddim_steps=ddim_steps,
                        sampler=sampler,
                        ddim_eta=ddim_eta,
                        discretize=discretize,
                    )
                _blend_long_audio_chunk(
                    final_waveform,
                    contribution_map,
                    channel_index,
                    start_sample,
                    end_sample,
                    overlap_samples,
                    _extract_single_generated(generated),
                    original_peak,
                )
            continue

        for batch_start in range(0, len(chunk_records), batch_size):
            record_batch = chunk_records[batch_start : batch_start + batch_size]
            generated_batch = _generate_long_audio_batch(
                latent_diffusion,
                [record[2].detach().cpu().numpy() for record in record_batch],
                seed=int(seed) + batch_start,
                lowpass_seed=int(seed),
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
                sampler=sampler,
                ddim_eta=ddim_eta,
                discretize=discretize,
            )
            for record, processed in zip(record_batch, generated_batch):
                start_sample, end_sample, _chunk, original_peak = record
                _blend_long_audio_chunk(
                    final_waveform,
                    contribution_map,
                    channel_index,
                    start_sample,
                    end_sample,
                    overlap_samples,
                    torch.as_tensor(processed, dtype=torch.float32),
                    original_peak,
                )

    final_waveform = final_waveform / contribution_map.clamp_min(1e-8)
    return torch.clamp(final_waveform, -1.0, 1.0)



def super_resolution_batch(
    latent_diffusion,
    waveforms_list,
    seed=42,
    ddim_steps=200,
    guidance_scale=3.5,
    lowpass_seed=None,
    sampler=DEFAULT_SAMPLER,
    ddim_eta=1.0,
    discretize=DEFAULT_DISCRETIZATION,
):
    """
    Process caller-grouped mono 48 kHz waveforms in one model invocation.

    Inputs may have different lengths; all feature batches share one padded
    sample count, and each returned NumPy array is trimmed to its input length.
    """
    sampler = normalize_sampler(sampler)
    ddim_eta = normalize_ddim_eta(ddim_eta)
    discretize = normalize_discretize(discretize)
    waveforms = list(waveforms_list)
    if not waveforms:
        return []

    prepared = [_as_mono_numpy(waveform) for waveform in waveforms]
    original_lengths = [waveform.size for waveform in prepared]
    padded_samples = _padded_sample_count(max(original_lengths))

    seed_everything(int(seed))
    lowpass_filter_type = _select_lowpass_filter_type(
        seed if lowpass_seed is None else lowpass_seed
    )
    batch_list = [
        _prepare_mono_batch(
            waveform,
            padded_samples,
            add_batch_dimension=False,
            lowpass_filter_type=lowpass_filter_type,
        )[0]
        for waveform in prepared
    ]
    combined_batch = {}
    for key, value in batch_list[0].items():
        if isinstance(value, torch.Tensor):
            combined_batch[key] = torch.stack([batch[key] for batch in batch_list], dim=0)
        else:
            combined_batch[key] = value

    with torch.inference_mode():
        generated = latent_diffusion.generate_batch(
            combined_batch,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            sampler=sampler,
            ddim_eta=ddim_eta,
            discretize=discretize,
        )

    return [
        _trim_or_pad(
            _extract_batch_generated(generated, index, len(prepared)),
            original_length,
        ).numpy()
        for index, original_length in enumerate(original_lengths)
    ]
