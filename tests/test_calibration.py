import importlib.util
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from audiosr import calibration
import audiosr.pipeline as pipeline


@pytest.fixture(scope="module")
def trainer():
    path = pathlib.Path(__file__).parents[1] / "tools" / "train_calibration.py"
    spec = importlib.util.spec_from_file_location("audiosr_train_calibration", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tone(frequency, seconds=1.0, sample_rate=48000, amplitude=0.3):
    t = torch.arange(int(seconds * sample_rate), dtype=torch.float64) / sample_rate
    return (amplitude * torch.sin(2 * torch.pi * frequency * t)).float()


def test_band_edges_cover_the_spectrum_without_gaps():
    edges = calibration.band_edges()

    assert edges[0] == calibration._MIN_BAND_BIN
    assert edges[-1] == calibration._N_FFT // 2 + 1
    widths = [stop - start for start, stop in zip(edges, edges[1:])]
    assert all(width >= 1 for width in widths)
    assert widths[-1] > widths[0]


def test_band_envelope_shape_and_level_offset():
    edges = calibration.band_edges()
    envelope = calibration.band_envelope(torch.randn(48000) * 0.1, edges)

    assert envelope.shape[0] == len(edges) - 1
    assert envelope.shape[1] > 0
    offset = calibration.level_offset(envelope)
    assert abs(calibration.level_offset(envelope - offset)) < 1e-6


def test_stack_context_gives_one_row_per_frame():
    envelope = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    features = calibration.stack_context(envelope, context=2)

    assert features.shape == (8, 3 * 5)


def test_cutoff_band_follows_the_source_bandwidth():
    edges = calibration.band_edges()
    generator = torch.Generator().manual_seed(3)
    noise = torch.randn(48000, generator=generator) * 0.1
    spectrum = torch.fft.rfft(noise)
    freqs = torch.fft.rfftfreq(noise.shape[-1], 1 / 48000)
    narrow = torch.fft.irfft(torch.where(freqs < 4000, spectrum, 0), noise.shape[-1])
    wide = torch.fft.irfft(torch.where(freqs < 16000, spectrum, 0), noise.shape[-1])

    narrow_cut = calibration.cutoff_band(
        calibration.band_envelope(narrow, edges), edges
    )
    wide_cut = calibration.cutoff_band(calibration.band_envelope(wide, edges), edges)

    assert narrow_cut < wide_cut < len(edges) - 1


def test_predictor_shapes_and_save_load_roundtrip(tmp_path):
    bands = len(calibration.band_edges()) - 1
    model = calibration.EnvelopePredictor(bands)
    features = torch.randn(7, bands * (2 * calibration._CONTEXT_FRAMES + 1))

    assert model(features).shape == (7, bands)

    path = tmp_path / "calibration.pt"
    calibration.save_calibration(path, model)
    reloaded = calibration.load_calibration(path)
    assert torch.allclose(model(features), reloaded(features))


def test_bundled_calibration_exists_and_loads():
    path = calibration.bundled_calibration_path()
    assert pathlib.Path(path).is_file()

    model = calibration.load_calibration(path)
    bands = len(calibration.band_edges()) - 1
    features = torch.zeros(3, bands * (2 * calibration._CONTEXT_FRAMES + 1))
    assert model(features).shape == (3, bands)


def test_load_rejects_a_mismatched_transform(tmp_path):
    bands = len(calibration.band_edges()) - 1
    model = calibration.EnvelopePredictor(bands)
    path = tmp_path / "calibration.pt"
    calibration.save_calibration(path, model)

    payload = torch.load(path, weights_only=True)
    payload["n_fft"] = 1024
    torch.save(payload, path)

    with pytest.raises(ValueError, match="n_fft"):
        calibration.load_calibration(path)


class _FixedPrediction(torch.nn.Module):
    """Predict a constant envelope, for exercising the gain math alone."""

    def __init__(self, bands, level):
        super().__init__()
        self.bands = bands
        self.context = calibration._CONTEXT_FRAMES
        self.level = level

    def forward(self, features):
        return torch.full((features.shape[0], self.bands), self.level)


def test_gains_are_clamped_and_leave_the_source_band_alone():
    edges = calibration.band_edges()
    bands = len(edges) - 1
    low = _tone(1000) + _tone(15000, amplitude=0.1)
    source = _tone(1000)

    # A prediction far below everything asks for the maximum cut.
    model = _FixedPrediction(bands, level=-30.0)
    gains_db, cutoff = calibration._band_gains(model, low, source, edges)

    assert float(gains_db.min()) >= calibration._GAIN_LIMITS_DB[0] - 1e-5
    assert float(gains_db.max()) <= calibration._GAIN_LIMITS_DB[1] + 1e-5
    assert torch.all(gains_db[: cutoff + 1] == 0.0)
    assert float(gains_db[cutoff + 1 :].mean()) < 0.0


def test_apply_calibration_attenuates_hallucinated_content():
    """A silent source must pull generated top-end energy down."""
    edges = calibration.band_edges()
    bands = len(edges) - 1
    restored = (_tone(1000) + _tone(12000, amplitude=0.2)).reshape(1, -1)
    source = _tone(1000).reshape(1, -1)

    model = _FixedPrediction(bands, level=-30.0)
    calibrated = calibration.apply_calibration(restored, source, model)

    assert calibrated.shape == restored.shape

    def band_energy(waveform, low_hz, high_hz):
        spectrum = torch.fft.rfft(waveform[0].double())
        freqs = torch.fft.rfftfreq(waveform.shape[-1], 1 / 48000)
        band = (freqs >= low_hz) & (freqs < high_hz)
        return float((spectrum[band].abs() ** 2).sum())

    before = band_energy(restored, 11000, 13000)
    after = band_energy(calibrated, 11000, 13000)
    kept = band_energy(calibrated, 900, 1100) / band_energy(restored, 900, 1100)
    assert after < 0.05 * before
    assert 0.9 < kept < 1.1


def _with_pause(gap_seconds, sample_rate=48000):
    """A 1 kHz tone with a silent gap in the middle, and frame indexes for it."""
    tone = _tone(1000, seconds=1.0)
    gap = torch.zeros(int(gap_seconds * sample_rate))
    source = torch.cat([tone, gap, tone])
    first = tone.shape[-1] // calibration._HOP
    last = (tone.shape[-1] + gap.shape[-1]) // calibration._HOP
    return source, first, last


def test_sustained_pauses_open_the_gain_floor():
    edges = calibration.band_edges()
    bands = len(edges) - 1
    source, first, last = _with_pause(gap_seconds=1.0)
    restored = source + _tone(12000, seconds=3.0, amplitude=0.05)

    model = _FixedPrediction(bands, level=-30.0)
    gains_db, cutoff = calibration._band_gains(model, restored, source, edges)

    margin = calibration._QUIET_SUSTAIN_FRAMES + calibration._GAIN_SMOOTHING_FRAMES
    pause = gains_db[cutoff + 1 :, first + margin : last - margin]
    voiced = gains_db[cutoff + 1 :, margin : first - margin]
    assert float(pause.min()) < calibration._GAIN_LIMITS_DB[0] - 10.0
    assert float(pause.min()) >= calibration._QUIET_GAIN_FLOOR_DB - 1e-5
    assert float(voiced.min()) >= calibration._GAIN_LIMITS_DB[0] - 1e-5


def test_short_gaps_keep_the_base_floor():
    """A gap briefer than the sustain window could hide above-cutoff content."""
    edges = calibration.band_edges()
    bands = len(edges) - 1
    source, _first, _last = _with_pause(gap_seconds=0.1)
    restored = source + _tone(12000, seconds=2.1, amplitude=0.05)

    model = _FixedPrediction(bands, level=-30.0)
    gains_db, _cutoff = calibration._band_gains(model, restored, source, edges)

    assert float(gains_db.min()) >= calibration._GAIN_LIMITS_DB[0] - 1e-5


def test_apply_calibration_rejects_a_channel_mismatch():
    bands = len(calibration.band_edges()) - 1
    model = _FixedPrediction(bands, level=0.0)

    with pytest.raises(ValueError, match="channels"):
        calibration.apply_calibration(
            torch.zeros(2, 48000), torch.zeros(1, 48000), model
        )


def test_training_reduces_the_masked_loss(trainer):
    torch.manual_seed(5)
    bands = len(calibration.band_edges()) - 1
    features = torch.randn(64, bands * (2 * calibration._CONTEXT_FRAMES + 1))
    weights = torch.randn(features.shape[-1], bands) * 0.1
    targets = features @ weights
    mask = torch.ones(64, bands)

    _model, losses = trainer.train(
        features, targets, mask, epochs=60, learning_rate=1e-2, seed=5
    )

    assert losses[-1] < 0.5 * losses[0]


def test_training_pairs_mask_only_bands_above_the_cutoff(trainer, tmp_path):
    audio = (_tone(1000, seconds=2.0) + _tone(18000, seconds=2.0, amplitude=0.1))
    path = tmp_path / "full_band.wav"
    sf.write(str(path), audio.numpy(), 48000, subtype="FLOAT")

    parts = list(trainer.training_pairs([path], [4000]))

    assert len(parts) == 1
    features, targets, mask = parts[0]
    assert features.shape[0] == targets.shape[0] == mask.shape[0]
    assert torch.all((mask == 0) | (mask == 1))
    assert 0.0 < float(mask.mean()) < 1.0
    per_band = mask[0]
    change_points = (per_band[1:] != per_band[:-1]).sum()
    assert change_points == 1  # zeros below the cutoff, ones above, once


def test_calibrate_output_runs_through_the_pipeline(tmp_path):
    source = _tone(1000, seconds=1.0)
    source_path = tmp_path / "source.wav"
    sf.write(str(source_path), source.numpy(), 48000, subtype="FLOAT")

    bands = len(calibration.band_edges()) - 1
    model = calibration.EnvelopePredictor(bands)
    model_path = tmp_path / "calibration.pt"
    calibration.save_calibration(model_path, model)

    generated = (_tone(1000) + _tone(12000, amplitude=0.2)).reshape(1, 1, -1).numpy()
    output = pipeline.calibrate_output(generated, source_path, model_path)

    assert output.shape == generated.shape
    assert np.isfinite(output).all()
