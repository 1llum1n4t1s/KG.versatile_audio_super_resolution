"""Public AudioSR API with lazy imports.

Keeping package import lightweight lets ``python -m audiosr --help`` and the
installed console entry point show their parser without importing PyTorch or
initializing model/tokenizer code.
"""

from importlib import import_module


_UTILS_EXPORTS = {
    "save_wave",
    "get_time",
    "get_duration",
    "read_list",
    "load_audio",
    "lowpass_filtering_prepare_inference",
    "normalize_wav",
    "pad_wav",
    "read_audio_file",
    "wav_feature_extraction",
    "download_checkpoint",
    "default_audioldm_config",
}

_PIPELINE_EXPORTS = {
    "seed_everything",
    "text2phoneme",
    "text_to_filename",
    "extract_kaldi_fbank_feature",
    "make_batch_for_super_resolution",
    "round_up_duration",
    "build_model",
    "super_resolution",
    "super_resolution_long_audio",
    "super_resolution_batch",
}

__all__ = [
    "build_model",
    "default_audioldm_config",
    "download_checkpoint",
    "extract_kaldi_fbank_feature",
    "get_duration",
    "get_time",
    "load_audio",
    "lowpass_filtering_prepare_inference",
    "make_batch_for_super_resolution",
    "normalize_wav",
    "pad_wav",
    "read_audio_file",
    "read_list",
    "round_up_duration",
    "save_wave",
    "seed_everything",
    "super_resolution",
    "super_resolution_batch",
    "super_resolution_long_audio",
    "text2phoneme",
    "text_to_filename",
    "wav_feature_extraction",
]


def __getattr__(name):
    if name in _UTILS_EXPORTS:
        module = import_module(".utils", __name__)
    elif name in _PIPELINE_EXPORTS:
        module = import_module(".pipeline", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
