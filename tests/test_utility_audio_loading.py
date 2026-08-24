import numpy as np
import soundfile as sf
import torch

from audiosr.utils import load_audio
from audiosr.utilities.audio import tools as audio_tools
from audiosr.utilities.data import dataset as dataset_module


def _write_wav(path, channels, sample_rate=22050, frames=256):
    samples = np.arange(frames * channels, dtype=np.float32).reshape(
        frames, channels
    )
    samples = samples / max(np.max(np.abs(samples)), 1.0)
    if channels == 1:
        samples = samples[:, 0]
    sf.write(path, samples, sample_rate)
    return samples


def test_load_audio_preserves_mono_shape_and_sample_rate(tmp_path):
    path = tmp_path / "mono.wav"
    samples = _write_wav(path, channels=1, sample_rate=22050)

    waveform, sample_rate = load_audio(path)

    assert waveform.shape == (1, samples.shape[-1])
    assert sample_rate == 22050


def test_load_audio_preserves_stereo_shape_and_sample_rate(tmp_path):
    path = tmp_path / "stereo.wav"
    samples = _write_wav(path, channels=2, sample_rate=32000)

    waveform, sample_rate = load_audio(path)

    assert waveform.shape == (2, samples.shape[0])
    assert sample_rate == 32000


def test_audio_tools_read_wav_file_keeps_first_channel_and_batch_shape(monkeypatch):
    source = torch.stack(
        [torch.linspace(-1.0, 1.0, 256), torch.linspace(1.0, -1.0, 256)]
    )
    monkeypatch.setattr(audio_tools, "load_audio", lambda _path: (source, 22050))
    monkeypatch.setattr(
        audio_tools.torchaudio.functional,
        "resample",
        lambda waveform, orig_freq, new_freq: waveform,
    )

    waveform = audio_tools.read_wav_file("ignored.wav", segment_length=192)

    assert waveform.shape == (1, 192)


def test_audio_tools_silent_waveform_remains_finite(monkeypatch):
    source = torch.zeros((1, 256), dtype=torch.float32)
    monkeypatch.setattr(audio_tools, "load_audio", lambda _path: (source, 16000))

    waveform = audio_tools.read_wav_file("ignored.wav", segment_length=256)

    assert np.isfinite(waveform).all()
    assert not np.any(waveform)


def test_dataset_read_wav_file_keeps_first_channel_and_batch_shape(monkeypatch):
    source = torch.ones((2, 256), dtype=torch.float32)
    monkeypatch.setattr(dataset_module, "load_audio", lambda _path: (source, 22050))

    dataset = dataset_module.AudioDataset.__new__(dataset_module.AudioDataset)
    dataset.duration = 256 / 16000
    dataset.sampling_rate = 16000
    dataset.trim_wav = False
    monkeypatch.setattr(
        dataset,
        "random_segment_wav",
        lambda waveform, target_length: (waveform, 0),
    )
    monkeypatch.setattr(dataset, "resample", lambda waveform, sr: waveform)
    monkeypatch.setattr(dataset, "normalize_wav", lambda waveform: waveform)
    monkeypatch.setattr(
        dataset,
        "pad_wav",
        lambda waveform, target_length: waveform[..., :target_length],
    )

    waveform, random_start = dataset.read_wav_file("ignored.wav")

    assert waveform.shape == (1, 256)
    assert random_start == 0


def test_dataset_read_audio_file_waveform_only_returns_expected_shape(
    monkeypatch, tmp_path
):
    path = tmp_path / "audio.wav"
    path.touch()

    dataset = dataset_module.AudioDataset.__new__(dataset_module.AudioDataset)
    dataset.waveform_only = True
    expected_waveform = np.zeros((1, 256), dtype=np.float32)
    monkeypatch.setattr(
        dataset,
        "read_wav_file",
        lambda _path: (expected_waveform, 0),
    )

    log_mel_spec, stft, mix_lambda, waveform, random_start = dataset.read_audio_file(
        str(path)
    )

    assert log_mel_spec is None
    assert stft is None
    assert mix_lambda == 0.0
    assert waveform.shape == (1, 256)
    assert random_start == 0
