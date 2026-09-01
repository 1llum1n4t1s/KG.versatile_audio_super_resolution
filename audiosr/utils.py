import contextlib
import hashlib
import importlib
from huggingface_hub import hf_hub_download
import numpy as np
import torch

from inspect import isfunction
import os
import soundfile as sf
import time
import wave
import torchaudio
import progressbar
from librosa.filters import mel as librosa_mel_fn
from audiosr.lowpass import lowpass

hann_window = {}
mel_basis = {}
_LOWPASS_FILTER_TYPES = ("butter", "cheby1", "ellip", "bessel")
_NUMPY_SEED_MODULUS = 2**32
_CHECKPOINT_REVISIONS = {
    "basic": (
        "haoheliu/audiosr_basic",
        "74a47f49061a1e788e968cc43ad45c0b6243f37d",
    ),
    "speech": (
        "haoheliu/audiosr_speech",
        "413f1d734411663e95310c17d381279a0c049960",
    ),
}


def _normalize_seed(seed):
    """Map any integer seed to the range shared by NumPy's legacy RNGs."""
    return int(seed) % _NUMPY_SEED_MODULUS


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def load_audio(path, target_sample_rate=None):
    """Load an audio file as a ``[channels, samples]`` float32 tensor.

    ``soundfile`` returns samples in ``[samples, channels]`` order.  Keeping
    the conversion here makes the rest of the audio pipeline independent of
    the decoder used by callers.  Resampling is deliberately performed only
    when a target rate was requested and differs from the file's rate.
    """

    audio, sample_rate = sf.read(
        path,
        always_2d=True,
        dtype="float32",
    )
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    waveform = torch.from_numpy(np.ascontiguousarray(audio.T))
    sample_rate = int(sample_rate)

    if target_sample_rate is None:
        return waveform, sample_rate

    target_sample_rate = int(target_sample_rate)
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            target_sample_rate,
        )
        sample_rate = target_sample_rate

    return waveform.to(dtype=torch.float32), sample_rate


def _locate_cutoff_freq(stft, percentile=0.97):
    def _find_cutoff(x, percentile=0.95):
        percentile = x[-1] * percentile
        for i in range(1, x.shape[0]):
            if x[-i] < percentile:
                return x.shape[0] - i
        return 0

    magnitude = torch.abs(stft)
    energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
    return _find_cutoff(energy, percentile)


def pad_wav(waveform, target_length):
    """Crop or zero-pad the final (time) dimension to ``target_length``."""

    target_length = int(target_length)
    if target_length < 0:
        raise ValueError("target_length must be non-negative")

    waveform_length = int(waveform.shape[-1])
    if waveform_length == target_length:
        return waveform

    if waveform_length > target_length:
        return waveform[..., :target_length]

    pad_amount = target_length - waveform_length
    if torch.is_tensor(waveform):
        return torch.nn.functional.pad(waveform, (0, pad_amount))

    waveform = np.asarray(waveform)
    padding = [(0, 0)] * (waveform.ndim - 1) + [(0, pad_amount)]
    return np.pad(waveform, padding, mode="constant")


def _select_lowpass_filter_type(seed=None):
    if seed is None:
        return str(np.random.choice(_LOWPASS_FILTER_TYPES))
    return str(np.random.RandomState(_normalize_seed(seed)).choice(_LOWPASS_FILTER_TYPES))


def lowpass_filtering_prepare_inference(dl_output, filter_type=None):
    waveform = dl_output["waveform"]  # [1, samples]
    sampling_rate = dl_output["sampling_rate"]

    cutoff_freq = (
        _locate_cutoff_freq(dl_output["stft"], percentile=0.985) / 1024
    ) * 24000
    
    # If the audio is almost empty, or the cutoff is at/above Nyquist, there
    # is no safe filter to apply.  Return a copy so callers can still mutate
    # the result without changing the input batch.
    if cutoff_freq < 1000 or cutoff_freq >= sampling_rate / 2:
        return {"waveform_lowpass": waveform.clone()}

    if filter_type is None:
        filter_type = _select_lowpass_filter_type()
    elif filter_type not in _LOWPASS_FILTER_TYPES:
        raise ValueError(
            f"filter_type must be one of {', '.join(_LOWPASS_FILTER_TYPES)}"
        )

    order = 8
    filtered_audio = lowpass(
        waveform.numpy().squeeze(),
        highcut=cutoff_freq,
        fs=sampling_rate,
        order=order,
        _type=filter_type,
    )

    filtered_audio = torch.FloatTensor(filtered_audio.copy()).unsqueeze(0)

    if waveform.size(-1) <= filtered_audio.size(-1):
        filtered_audio = filtered_audio[..., : waveform.size(-1)]
    else:
        filtered_audio = torch.functional.pad(
            filtered_audio, (0, waveform.size(-1) - filtered_audio.size(-1))
        )

    return {"waveform_lowpass": filtered_audio}


def mel_spectrogram_train(y):
    global mel_basis, hann_window

    sampling_rate = 48000
    filter_length = 2048
    hop_length = 480
    win_length = 2048
    n_mel = 256
    mel_fmin = 20
    mel_fmax = 24000

    mel_cache_key = str(mel_fmax) + "_" + str(y.device)
    window_cache_key = str(y.device)
    if mel_cache_key not in mel_basis or window_cache_key not in hann_window:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=n_mel, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis[mel_cache_key] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[window_cache_key] = torch.hann_window(win_length).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((filter_length - hop_length) / 2), int((filter_length - hop_length) / 2)),
        mode="reflect",
    )

    y = y.squeeze(1)

    stft_spec = torch.stft(
        y,
        filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[window_cache_key],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    stft_spec = torch.abs(stft_spec)

    mel = spectral_normalize_torch(
        torch.matmul(mel_basis[mel_cache_key], stft_spec)
    )

    return mel[0], stft_spec[0]


def pad_spec(log_mel_spec, target_frame):
    n_frames = log_mel_spec.shape[0]
    p = target_frame - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        log_mel_spec = m(log_mel_spec)
    elif p < 0:
        log_mel_spec = log_mel_spec[0:target_frame, :]

    if log_mel_spec.size(-1) % 2 != 0:
        log_mel_spec = log_mel_spec[..., :-1]

    return log_mel_spec


def wav_feature_extraction(waveform, target_frame):
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    log_mel_spec, stft = mel_spectrogram_train(waveform.unsqueeze(0))

    log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    stft = torch.FloatTensor(stft.T)

    log_mel_spec, stft = pad_spec(log_mel_spec, target_frame), pad_spec(
        stft, target_frame
    )
    return log_mel_spec, stft


def normalize_wav(waveform):
    waveform = np.nan_to_num(
        np.asarray(waveform), nan=0.0, posinf=1.0, neginf=-1.0
    )
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5

def read_wav_file(filename, channel=0):
    waveform, sr = load_audio(filename, target_sample_rate=48000)

    if channel is not None:
        if not isinstance(channel, (int, np.integer)):
            raise TypeError("channel must be an integer or None")
        channel = int(channel)
        if channel < 0:
            channel += waveform.shape[0]
        if channel < 0 or channel >= waveform.shape[0]:
            raise IndexError("channel index out of range")
        waveform = waveform[channel : channel + 1]

    source_samples = int(waveform.shape[-1])
    if source_samples <= 100:
        raise ValueError(
            f"audio is too short for AudioSR ({source_samples} samples; need more than 100)"
        )
    duration = source_samples / float(sr)

    if(duration > 10.24):
        print("\033[93m {}\033[00m" .format("Warning: audio is longer than 10.24 seconds, may degrade the model performance. It's recommand to truncate your audio to 5.12 seconds before input to AudioSR to get the best performance."))

    padding_samples = 245_760  # 5.12 seconds at 48 kHz
    target_samples = (
        (source_samples + padding_samples - 1) // padding_samples
    ) * padding_samples
    target_frame = target_samples // 480
    pad_duration = target_samples / 48000.0

    waveform = waveform.numpy()
    if waveform.shape[-1] > 0:
        waveform = normalize_wav(
            waveform
        )  # TODO rescaling the waveform will cause low LSD score
    waveform = pad_wav(waveform, target_length=target_samples)
    return waveform, target_frame, pad_duration

def read_audio_file(filename):
    waveform, target_frame, duration = read_wav_file(filename)
    log_mel_spec, stft = wav_feature_extraction(waveform, target_frame)
    return log_mel_spec, stft, waveform, duration, target_frame


def read_list(fname):
    result = []
    with open(fname, "r", encoding="utf-8") as f:
        for each in f.readlines():
            each = each.strip("\n")
            result.append(each)
    return result


def get_duration(fname):
    return sf.info(fname).duration


def get_bit_depth(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        bit_depth = f.getsampwidth() * 8
        return bit_depth


def get_time():
    t = time.localtime()
    return time.strftime("%d_%m_%Y_%H_%M_%S", t)


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    seed = _normalize_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def save_wave(waveform, inputpath, savepath, name="outwav", samplerate=48000):
    """Save ``[batch, channels, samples]`` without changing its sample count.

    ``inputpath`` remains part of the public call signature for compatibility,
    but the generated waveform is the authority for output length.
    """

    if samplerate <= 0:
        raise ValueError("samplerate must be positive")

    if torch.is_tensor(waveform):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.asarray(waveform)
    if waveform.ndim == 1:
        batches = waveform[np.newaxis, np.newaxis, :]
    elif waveform.ndim == 2:
        # Preserve the historical contract: two dimensions mean a batch of
        # mono outputs. Multi-channel output is represented explicitly as BCT.
        batches = waveform[:, np.newaxis, :]
    elif waveform.ndim == 3:
        batches = waveform
    else:
        raise ValueError(
            "waveform must have shape [samples], [batch, samples], or "
            "[batch, channels, samples]"
        )

    if isinstance(name, (list, tuple)):
        if len(name) != batches.shape[0]:
            raise ValueError("name list length must match waveform batch size")
        names = list(name)
    else:
        names = [name] * batches.shape[0]

    os.makedirs(savepath, exist_ok=True)

    for index, (channels_first, output_name) in enumerate(zip(batches, names)):
        output_name = os.path.basename(os.fspath(output_name))
        while output_name.lower().endswith(".wav"):
            output_name = output_name[:-4]
        if not output_name:
            output_name = "outwav"
        if batches.shape[0] > 1:
            output_name = f"{output_name}_{index}"
        fname = output_name + ".wav"

        # Keep each path component within the common 255-character limit.
        # Python's hash() is randomized, so use a stable digest instead.
        if len(fname) > 255:
            digest = hashlib.sha256(fname.encode("utf-8")).hexdigest()
            fname = digest[:32] + ".wav"

        data_to_save = np.asarray(channels_first.T)

        save_path = os.path.join(savepath, fname)
        print("\033[98m {}\033[00m" .format("Don't forget to try different seeds by setting --seed <int> so that AudioSR can have optimal performance on your hardware."))
        print("Save audio to %s." % save_path)
        sf.write(save_path, data_to_save, samplerate=samplerate)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    try:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))
    except:
        import ipdb

        ipdb.set_trace()


def default_audioldm_config(model_name="basic"):
    basic_config = get_basic_config()
    return basic_config


class MyProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def download_checkpoint(checkpoint_name="basic"):
    try:
        model_id, revision = _CHECKPOINT_REVISIONS[checkpoint_name]
    except KeyError:
        raise ValueError("Invalid Model Name %s" % checkpoint_name)
    return hf_hub_download(
        repo_id=model_id,
        filename="pytorch_model.bin",
        revision=revision,
    )


def get_basic_config():
    return {
        "preprocessing": {
            "audio": {
                "sampling_rate": 48000,
                "max_wav_value": 32768,
                "duration": 10.24,
            },
            "stft": {"filter_length": 2048, "hop_length": 480, "win_length": 2048},
            "mel": {"n_mel_channels": 256, "mel_fmin": 20, "mel_fmax": 24000},
        },
        "augmentation": {"mixup": 0.5},
        "model": {
            "target": "audiosr.latent_diffusion.models.ddpm.LatentDiffusion",
            "params": {
                "first_stage_config": {
                    "base_learning_rate": 0.000008,
                    "target": "audiosr.latent_encoder.autoencoder.AutoencoderKL",
                    "params": {
                        "reload_from_ckpt": "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/vae/vae_48k_256/ds_8_kl_1/checkpoints/ckpt-checkpoint-484999.ckpt",
                        "sampling_rate": 48000,
                        "batchsize": 4,
                        "monitor": "val/rec_loss",
                        "image_key": "fbank",
                        "subband": 1,
                        "embed_dim": 16,
                        "time_shuffle": 1,
                        "ddconfig": {
                            "double_z": True,
                            "mel_bins": 256,
                            "z_channels": 16,
                            "resolution": 256,
                            "downsample_time": False,
                            "in_channels": 1,
                            "out_ch": 1,
                            "ch": 128,
                            "ch_mult": [1, 2, 4, 8],
                            "num_res_blocks": 2,
                            "attn_resolutions": [],
                            "dropout": 0.1,
                        },
                    },
                },
                "base_learning_rate": 0.0001,
                "warmup_steps": 5000,
                "optimize_ddpm_parameter": True,
                "sampling_rate": 48000,
                "batchsize": 16,
                "beta_schedule": "cosine",
                "linear_start": 0.0015,
                "linear_end": 0.0195,
                "num_timesteps_cond": 1,
                "log_every_t": 200,
                "timesteps": 1000,
                "unconditional_prob_cfg": 0.1,
                "parameterization": "v",
                "first_stage_key": "fbank",
                "latent_t_size": 128,
                "latent_f_size": 32,
                "channels": 16,
                "monitor": "val/loss_simple_ema",
                "scale_by_std": True,
                "unet_config": {
                    "target": "audiosr.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel",
                    "params": {
                        "image_size": 64,
                        "in_channels": 32,
                        "out_channels": 16,
                        "model_channels": 128,
                        "attention_resolutions": [8, 4, 2],
                        "num_res_blocks": 2,
                        "channel_mult": [1, 2, 3, 5],
                        "num_head_channels": 32,
                        "extra_sa_layer": True,
                        "use_spatial_transformer": True,
                        "transformer_depth": 1,
                    },
                },
                "evaluation_params": {
                    "unconditional_guidance_scale": 3.5,
                    "ddim_sampling_steps": 200,
                    "n_candidates_per_samples": 1,
                },
                "cond_stage_config": {
                    "concat_lowpass_cond": {
                        "cond_stage_key": "lowpass_mel",
                        "conditioning_key": "concat",
                        "target": "audiosr.latent_diffusion.modules.encoders.modules.VAEFeatureExtract",
                        "params": {
                            "first_stage_config": {
                                "base_learning_rate": 0.000008,
                                "target": "audiosr.latent_encoder.autoencoder.AutoencoderKL",
                                "params": {
                                    "sampling_rate": 48000,
                                    "batchsize": 4,
                                    "monitor": "val/rec_loss",
                                    "image_key": "fbank",
                                    "subband": 1,
                                    "embed_dim": 16,
                                    "time_shuffle": 1,
                                    "ddconfig": {
                                        "double_z": True,
                                        "mel_bins": 256,
                                        "z_channels": 16,
                                        "resolution": 256,
                                        "downsample_time": False,
                                        "in_channels": 1,
                                        "out_ch": 1,
                                        "ch": 128,
                                        "ch_mult": [1, 2, 4, 8],
                                        "num_res_blocks": 2,
                                        "attn_resolutions": [],
                                        "dropout": 0.1,
                                    },
                                },
                            }
                        },
                    }
                },
            },
        },
    }
