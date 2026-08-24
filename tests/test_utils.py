import numpy as np
import pytest
import soundfile as sf
import torch

from audiosr import utils


def test_normalize_wav_replaces_non_finite_values():
    result = utils.normalize_wav(
        np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    )
    assert np.isfinite(result).all()


def test_pad_wav_crops_or_pads_all_leading_dimensions():
    waveform = np.arange(2 * 205, dtype=np.float32).reshape(2, 205)

    cropped = utils.pad_wav(waveform, 204)
    padded = utils.pad_wav(waveform, 206)

    np.testing.assert_array_equal(cropped, waveform[..., :204])
    np.testing.assert_array_equal(padded[..., :205], waveform)
    np.testing.assert_array_equal(padded[..., 205:], np.zeros((2, 1), dtype=np.float32))


def test_load_audio_returns_channels_first_float32_and_resamples(tmp_path):
    source = np.arange(16, dtype=np.float32).reshape(8, 2) / 16
    path = tmp_path / "stereo.flac"
    sf.write(path, source, 8000)

    waveform, sample_rate = utils.load_audio(path)
    assert waveform.shape == (2, 8)
    assert waveform.dtype == torch.float32
    assert sample_rate == 8000

    resampled, resampled_rate = utils.load_audio(path, target_sample_rate=16000)
    assert resampled.shape[0] == 2
    assert resampled.dtype == torch.float32
    assert resampled_rate == 16000


def test_read_wav_file_uses_mono_default_and_integer_padding(monkeypatch):
    source = torch.arange(2 * 1000, dtype=torch.float32).reshape(2, 1000)
    calls = []

    def fake_load_audio(path, target_sample_rate=None):
        calls.append(target_sample_rate)
        return source, 48000

    monkeypatch.setattr(utils, "load_audio", fake_load_audio)
    waveform, target_frame, duration = utils.read_wav_file("ignored.wav")

    assert calls == [48000]
    assert waveform.shape == (1, 245760)
    assert target_frame == 512
    assert duration == pytest.approx(5.12)

    all_channels, _, _ = utils.read_wav_file("ignored.wav", channel=None)
    assert all_channels.shape == (2, 245760)


@pytest.mark.parametrize("cutoff", [0, 1024])
def test_lowpass_bypasses_unsafe_cutoff(monkeypatch, cutoff):
    waveform = torch.randn(1, 64)
    calls = []

    monkeypatch.setattr(utils, "_locate_cutoff_freq", lambda *args, **kwargs: cutoff)

    def fail_if_called(*args, **kwargs):
        calls.append(True)
        raise AssertionError("lowpass must not run for an unsafe cutoff")

    monkeypatch.setattr(utils, "lowpass", fail_if_called)
    result = utils.lowpass_filtering_prepare_inference(
        {
            "waveform": waveform,
            "sampling_rate": 48000,
            "stft": torch.ones(1024, 2),
        }
    )["waveform_lowpass"]

    assert calls == []
    assert result is not waveform
    torch.testing.assert_close(result, waveform)


def test_lowpass_uses_explicit_filter_type(monkeypatch):
    waveform = torch.ones(1, 64)
    filter_types = []
    monkeypatch.setattr(utils, "_locate_cutoff_freq", lambda *_args, **_kwargs: 512)

    def capture_filter(audio, *, highcut, fs, order, _type):
        filter_types.append(_type)
        return np.asarray(audio)

    monkeypatch.setattr(utils, "lowpass", capture_filter)
    result = utils.lowpass_filtering_prepare_inference(
        {
            "waveform": waveform,
            "sampling_rate": 48000,
            "stft": torch.ones(1024, 2),
        },
        filter_type="ellip",
    )["waveform_lowpass"]

    assert filter_types == ["ellip"]
    torch.testing.assert_close(result, waveform)


@pytest.mark.parametrize("suffix", [".wav", ".flac"])
def test_get_duration_uses_soundfile_for_supported_formats(tmp_path, suffix):
    path = tmp_path / f"duration{suffix}"
    sf.write(path, np.zeros(480, dtype=np.float32), 48000)

    assert utils.get_duration(path) == pytest.approx(0.01)


@pytest.mark.parametrize(
    "checkpoint_name, repo_id, revision",
    [
        (
            "basic",
            "haoheliu/audiosr_basic",
            "74a47f49061a1e788e968cc43ad45c0b6243f37d",
        ),
        (
            "speech",
            "haoheliu/audiosr_speech",
            "413f1d734411663e95310c17d381279a0c049960",
        ),
    ],
)
def test_download_checkpoint_pins_revision(
    monkeypatch, checkpoint_name, repo_id, revision
):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "checkpoint.bin"

    monkeypatch.setattr(utils, "hf_hub_download", fake_download)

    assert utils.download_checkpoint(checkpoint_name) == "checkpoint.bin"
    assert calls == [
        {
            "repo_id": repo_id,
            "filename": "pytorch_model.bin",
            "revision": revision,
        }
    ]


@pytest.mark.parametrize("waveform", [
    np.zeros((2, 2, 600), dtype=np.float32),
    torch.zeros((2, 2, 600), dtype=torch.float32),
])
def test_save_wave_writes_each_stereo_batch_without_retrimming(tmp_path, waveform):
    input_path = tmp_path / "input.wav"
    sf.write(input_path, np.zeros(80, dtype=np.float32), 8000)

    utils.save_wave(
        waveform,
        inputpath=input_path,
        savepath=tmp_path,
        name="result.wav.wav",
    )

    for index in range(2):
        info = sf.info(tmp_path / f"result_{index}.wav")
        assert info.samplerate == 48000
        assert info.channels == 2
        assert info.frames == 600


def test_save_wave_preserves_resampled_sample_count(tmp_path):
    input_path = tmp_path / "input.wav"
    sf.write(input_path, np.zeros(44_101, dtype=np.float32), 44_100)
    waveform, sample_rate = utils.load_audio(input_path, target_sample_rate=48_000)

    assert waveform.shape == (1, 48_002)

    utils.save_wave(
        waveform[None, ...],
        inputpath=input_path,
        savepath=tmp_path,
        name="resampled",
        samplerate=sample_rate,
    )

    assert sf.info(tmp_path / "resampled.wav").frames == waveform.shape[-1]


def test_save_wave_shortens_long_names_deterministically(tmp_path):
    input_path = tmp_path / "input.wav"
    sf.write(input_path, np.zeros(1, dtype=np.float32), 48000)
    name = "x" * 300

    utils.save_wave(
        np.zeros(1, dtype=np.float32),
        inputpath=input_path,
        savepath=tmp_path,
        name=name,
    )
    output_names = [path.name for path in tmp_path.glob("*.wav") if path.name != "input.wav"]

    assert len(output_names) == 1
    assert len(output_names[0]) <= 255
    assert output_names[0].endswith(".wav")
    assert not output_names[0].endswith(".wav.wav")
