"""Regression tests for the sampler benchmark harness.

The harness is a development tool, so these cover its argument handling and its
metrics rather than exercising the model, which needs the full checkpoint.
"""

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "benchmark_samplers.py"


def load_tool_module():
    spec = importlib.util.spec_from_file_location("audiosr_benchmark_tool", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return load_tool_module()


def test_config_specifications_are_parsed_and_defaulted(tool):
    assert tool.parse_config("dpmpp2m") == {
        "sampler": "dpmpp2m",
        "ddim_steps": 50,
        "ddim_eta": 1.0,
        "discretize": "uniform",
    }
    assert tool.parse_config("DDIM:100:0.0:trailing") == {
        "sampler": "ddim",
        "ddim_steps": 100,
        "ddim_eta": 0.0,
        "discretize": "trailing",
    }
    assert (
        tool.config_label(tool.parse_config("ddim:100:0.0"))
        == "ddim/100/eta0/uniform"
    )
    assert (
        tool.config_label(tool.parse_config("dpmpp2m:20::trailing"))
        == "dpmpp2m/20/trailing"
    )


@pytest.mark.parametrize(
    "specification",
    ["euler:20", "ddim:0", "ddim:1001", "ddim:20:-1", "ddim:20:0:leading", "a:b:c:d:e"],
)
def test_invalid_config_specifications_are_rejected(tool, specification):
    with pytest.raises(argparse.ArgumentTypeError):
        tool.parse_config(specification)


def test_degradation_removes_energy_above_the_cutoff(tool):
    duration = 48000
    time_axis = torch.arange(duration, dtype=torch.float32) / 48000
    # One tone well inside the retained band and one well above it.
    waveform = (
        torch.sin(2 * math.pi * 1000 * time_axis)
        + torch.sin(2 * math.pi * 18000 * time_axis)
    ).reshape(1, -1) * 0.4

    degraded = tool.degrade(waveform, cutoff_hz=12000)

    assert degraded.shape == waveform.shape
    spectrum = torch.stft(
        degraded[0],
        n_fft=2048,
        hop_length=512,
        window=torch.hann_window(2048),
        return_complex=True,
    ).abs()
    bin_hz = 48000 / 2048
    low = float(spectrum[int(1000 / bin_hz)].mean())
    high = float(spectrum[int(18000 / bin_hz)].mean())
    assert low > 1.0
    assert high < low * 1e-2


def test_scored_band_distance_is_zero_for_a_rescaled_copy(tool):
    generator = torch.Generator().manual_seed(11)
    waveform = torch.randn(1, 48000, generator=generator) * 0.1

    distance = tool.scored_band_distance(waveform, waveform * 0.25, 100, 800)

    assert distance == pytest.approx(0.0, abs=1e-4)


def test_scored_band_distance_grows_when_the_band_is_removed(tool):
    generator = torch.Generator().manual_seed(23)
    waveform = torch.randn(1, 48000, generator=generator) * 0.1
    degraded = tool.degrade(waveform, cutoff_hz=12000)

    bin_hz = 48000 / 2048
    low_bin = int(12000 / bin_hz) + 1
    high_bin = 1000

    intact = tool.scored_band_distance(waveform, waveform, low_bin, high_bin)
    removed = tool.scored_band_distance(waveform, degraded, low_bin, high_bin)
    in_band, above = tool.band_energy_ratios(waveform, degraded, low_bin, high_bin)

    assert removed > intact
    assert intact == pytest.approx(0.0, abs=1e-4)
    assert in_band < 0.05
    assert above == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("floor_db", [-40, -60, -80])
def test_reference_top_bin_finds_where_the_reference_stops(tool, floor_db):
    """Full-band noise reaches the top; a band-limited copy stops at its edge.

    The band limit is applied in the transform domain so the result does not
    depend on any resampling filter's transition shape, then laid on a noise
    floor. Every real band-limited recording has one — dither, preamp hiss, the
    codec's own noise — and it is what the cut is measured against. A stopband
    that holds nothing but float rounding is not a case the metric has to serve,
    and the floor's level is allowed to vary widely without moving the answer.
    """
    generator = torch.Generator().manual_seed(29)
    waveform = torch.randn(1, 48000, generator=generator) * 0.1

    n_fft, hop = 2048, 512
    window = torch.hann_window(n_fft)
    spectrum = torch.stft(
        waveform[0], n_fft=n_fft, hop_length=hop, window=window, return_complex=True
    )
    edge_bin = 400
    spectrum[edge_bin:] = 0
    band_limited = torch.istft(
        spectrum,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        length=waveform.shape[-1],
    ).reshape(1, -1)
    hiss = torch.randn(1, 48000, generator=torch.Generator().manual_seed(31))
    band_limited = band_limited + hiss * 0.1 * 10 ** (floor_db / 20)

    assert tool.reference_top_bin(waveform) > 900
    # The analysis window smears the edge across a few bins, so the detected
    # top only has to land on it rather than exactly at it.
    assert abs(tool.reference_top_bin(band_limited) - edge_bin) < 16


def test_reference_top_bin_keeps_a_steep_spectrum_that_never_cliffs(tool):
    """Music sits far below its own fundamental while still carrying content.

    A peak-relative cut would throw that band away; the noise floor is what
    separates content from nothing.
    """
    generator = torch.Generator().manual_seed(41)
    noise = torch.randn(1, 48000, generator=generator) * 0.1
    spectrum = torch.stft(
        noise[0], n_fft=2048, hop_length=512, window=torch.hann_window(2048),
        return_complex=True,
    )
    bins = torch.arange(spectrum.shape[0]).reshape(-1, 1)
    # 120 dB of tilt across the band, with no cliff anywhere.
    spectrum = spectrum * 10 ** (-6.0 * bins / spectrum.shape[0])
    sloped = torch.istft(
        spectrum, n_fft=2048, hop_length=512, window=torch.hann_window(2048),
        length=48000,
    ).reshape(1, -1)

    top = tool.reference_top_bin(sloped)
    bin_count = 1 + 2048 // 2
    assert top > 0.8 * bin_count

    # A cut taken 60 dB below the loudest bin would throw away half the band
    # that this reference can still judge.
    bin_energy = tool._magnitude(sloped).mean(dim=-1)
    peak = float(bin_energy.max())
    peak_relative = int(
        (bin_energy > peak * 10 ** (-60 / 20)).nonzero().flatten()[-1]
    )
    assert peak_relative < 0.6 * bin_count

def test_band_energy_ratio_is_one_for_a_rescaled_copy(tool):
    generator = torch.Generator().manual_seed(37)
    waveform = torch.randn(1, 48000, generator=generator) * 0.1

    in_band, above = tool.band_energy_ratios(waveform, waveform * 4.0, 100, 800)

    assert in_band == pytest.approx(1.0, rel=1e-4)
    assert above > 0.0


def test_reference_loading_downmixes_resamples_and_trims(tool, tmp_path):
    path = tmp_path / "reference.wav"
    stereo = np.stack(
        [
            np.linspace(-0.5, 0.5, 44100, dtype=np.float32),
            np.linspace(0.5, -0.5, 44100, dtype=np.float32),
        ],
        axis=-1,
    )
    sf.write(path, stereo, samplerate=44100, subtype="PCM_24")

    loaded = tool.load_reference(path, offset_s=0.0, duration_s=1.0)

    assert loaded.shape == (1, 48000)
    assert torch.isfinite(loaded).all()


def test_reference_loading_rejects_a_span_shorter_than_one_second(tool, tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(24000, dtype=np.float32), samplerate=48000)

    with pytest.raises(SystemExit):
        tool.load_reference(path, offset_s=0.0, duration_s=0.25)
    with pytest.raises(SystemExit):
        tool.load_reference(path, offset_s=10.0, duration_s=0.0)


def test_a_hairs_width_win_does_not_count_as_a_result(tool):
    """Beating the degraded input by 0.03% is a tie, not a ranking.

    Measured: speech restored at the shipped default scored 1.967896 against a
    1.968455 baseline, while the same run on music separated the configurations
    by 8%.
    """
    assert tool.ranking_is_unresolved(1.968455, [1.967896, 2.093403, 2.129551])
    assert not tool.ranking_is_unresolved(3.189229, [1.498266, 3.269670])


def test_an_empty_run_is_unresolved(tool):
    assert tool.ranking_is_unresolved(1.0, [])


def _band_noise(seed, low_bin, high_bin, frames=100, n_fft=2048, hop=512):
    """Noise confined to low_bin..high_bin, built in the transform domain."""
    generator = torch.Generator().manual_seed(seed)
    length = frames * hop
    noise = torch.randn(length, generator=generator) * 0.1
    window = torch.hann_window(n_fft)
    spectrum = torch.stft(
        noise, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
    )
    spectrum[:low_bin] = 0
    spectrum[high_bin + 1 :] = 0
    return torch.istft(
        spectrum, n_fft=n_fft, hop_length=hop, window=window, length=length
    ).reshape(1, -1)


def test_envelope_distance_is_zero_for_a_rescaled_copy(tool):
    generator = torch.Generator().manual_seed(11)
    waveform = torch.randn(1, 48000, generator=generator) * 0.1

    assert tool.envelope_band_distance(waveform, waveform * 0.25, 100, 800) == (
        pytest.approx(0.0, abs=1e-4)
    )


def test_envelope_distance_accepts_a_different_draw_of_the_same_noise(tool):
    """The reason this metric exists.

    Two draws of the same noise band share no bins, so the per-bin distance
    scores the correct restoration as badly as silence. The envelope tells
    them apart: the right energy in the right band scores near zero, silence
    scores far worse.
    """
    reference = _band_noise(23, low_bin=200, high_bin=600)
    other_draw = _band_noise(29, low_bin=200, high_bin=600)
    silence = torch.zeros_like(reference)

    envelope_other = tool.envelope_band_distance(reference, other_draw, 200, 600)
    envelope_silence = tool.envelope_band_distance(reference, silence, 200, 600)
    per_bin_other = tool.scored_band_distance(reference, other_draw, 200, 600)

    assert envelope_other < 0.25 * envelope_silence
    # The per-bin metric charges the correct restoration a large distance
    # anyway, because no two draws agree bin by bin. The envelope does not.
    assert per_bin_other > 5 * envelope_other


def test_envelope_distance_penalizes_energy_in_the_wrong_band(tool):
    reference = _band_noise(31, low_bin=200, high_bin=400)
    wrong_band = _band_noise(37, low_bin=450, high_bin=650)
    right_band = _band_noise(41, low_bin=200, high_bin=400)

    right = tool.envelope_band_distance(reference, right_band, 200, 650)
    wrong = tool.envelope_band_distance(reference, wrong_band, 200, 650)

    assert right < 0.5 * wrong


def test_envelope_band_edges_cover_the_range_without_gaps(tool):
    edges = tool._envelope_band_edges(170, 630)

    assert edges[0] == 170
    assert edges[-1] == 631
    widths = [stop - start for start, stop in zip(edges, edges[1:])]
    assert all(width >= 1 for width in widths)
    # Log spacing: the top band is wider than the bottom one.
    assert widths[-1] > widths[0]


def test_quiet_frame_excess_flags_noise_added_to_silence(tool):
    """Hiss laid over the reference's silent half must read as positive dB."""
    generator = torch.Generator().manual_seed(43)
    loud = torch.randn(1, 24000, generator=generator) * 0.1
    silence = torch.zeros(1, 24000)
    reference = torch.cat([loud, silence], dim=-1)

    hiss = torch.randn(1, 48000, generator=torch.Generator().manual_seed(47)) * 0.01
    noisy = reference + hiss

    added = tool.quiet_frame_excess_db(reference, noisy, 100, 800)
    clean = tool.quiet_frame_excess_db(reference, reference, 100, 800)

    assert added > 10.0
    assert abs(clean) < 1.0


def test_quiet_frame_excess_is_negative_for_a_muted_estimate(tool):
    generator = torch.Generator().manual_seed(53)
    reference = torch.randn(1, 48000, generator=generator) * 0.1

    excess = tool.quiet_frame_excess_db(reference, reference * 0.1, 100, 800)

    # A uniform level drop cannot register: both signals are normalized to
    # unit RMS first, so only a change in distribution moves the number.
    assert abs(excess) < 1.0


def test_spectrogram_sheet_writes_a_png(tool, tmp_path):
    pytest.importorskip("matplotlib")
    generator = torch.Generator().manual_seed(59)
    reference = torch.randn(1, 48000, generator=generator) * 0.1
    restored = tool.degrade(reference, cutoff_hz=8000)

    path = tool.write_spectrogram_sheet(
        tmp_path / "sheet.png",
        [("reference", reference), ("restored", restored)],
        sheet_title="test",
    )

    assert path is not None
    assert (tmp_path / "sheet.png").stat().st_size > 10_000


def test_spectrogram_flag_is_accepted(tool):
    parser = tool.build_parser()
    args = parser.parse_args(
        ["--reference", "r.wav", "--config", "ddim:4", "--no_spectrograms"]
    )
    assert args.no_spectrograms
