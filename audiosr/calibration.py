"""Learned output calibration.

The model overshoots: measured against full-band references, its restorations
put 10 to 28 dB too much energy into the generated band, worst where the
source is quiet — pauses come back as hiss. This module learns, from pairs of
full-band recordings and their band-limited versions, how much energy each
fractional-octave band above the cutoff should carry given the source's own
envelope, and rescales the restored signal toward that prediction.

It moves energy between bands and frames; it never invents content. Gains are
clamped so the calibration can temper the restoration but not replace it, and
bands at or below the source's cutoff are never touched — that band belongs to
the source and is already spliced back by the pipeline. Inside sustained
pauses — frames the source itself shows as silent for long enough that no
above-cutoff content could hide there — the lower clamp opens further, so the
hiss the model lays over silence can be pushed down to the noise floor.

Calibration is opt-in. It ships disabled until it has been shown to improve
every kind of material it was measured on, not just some.
"""

from importlib import resources

import torch
import torch.nn as nn

_N_FFT = 2048
_HOP = _N_FFT // 4
_BANDS_PER_OCTAVE = 3
_MIN_BAND_BIN = 8
_POWER_FLOOR = 1e-10
_CONTEXT_FRAMES = 2
_HIDDEN_UNITS = 128
_GAIN_LIMITS_DB = (-24.0, 6.0)
_QUIET_GAIN_FLOOR_DB = -60.0
_QUIET_ONSET_DB = 25.0
_QUIET_FULL_DB = 40.0
_QUIET_REFERENCE_QUANTILE = 0.9
_QUIET_SUSTAIN_FRAMES = 9
_GAIN_SMOOTHING_FRAMES = 3
_CUTOFF_ENERGY_FRACTION = 0.985


def band_edges(
    n_bins=_N_FFT // 2 + 1,
    bands_per_octave=_BANDS_PER_OCTAVE,
    min_bin=_MIN_BAND_BIN,
):
    """Return log-spaced bin edges from ``min_bin`` to the top bin."""
    ratio = 2.0 ** (1.0 / bands_per_octave)
    edges = [min_bin]
    position = float(min_bin)
    while True:
        position = max(position * ratio, position + 1.0)
        if position >= n_bins - 1:
            break
        edges.append(int(round(position)))
    edges.append(n_bins)
    return edges


def _spectrum(waveform):
    window = torch.hann_window(
        _N_FFT, periodic=True, device=waveform.device, dtype=waveform.dtype
    )
    return torch.stft(
        waveform,
        n_fft=_N_FFT,
        hop_length=_HOP,
        window=window,
        center=True,
        return_complex=True,
    )


def band_envelope(waveform, edges):
    """Return per-band log10 power over time, shape ``[bands, frames]``."""
    power = _spectrum(waveform).abs() ** 2
    bands = torch.stack(
        [power[start:stop].mean(dim=0) for start, stop in zip(edges, edges[1:])]
    )
    return torch.log10(bands + _POWER_FLOOR)


def level_offset(envelope):
    """Return the clip's overall level, used to make features level-invariant.

    The median is taken over every band and frame of the source's envelope, so
    a quiet recording and a loud one present the same features and the same
    offset restores absolute level to the prediction.
    """
    return float(envelope.median())


def stack_context(envelope, context=_CONTEXT_FRAMES):
    """Return per-frame features with ``context`` frames on each side."""
    frames = envelope.shape[-1]
    padded = torch.nn.functional.pad(
        envelope, (context, context), mode="replicate"
    )
    columns = [padded[:, offset : offset + frames] for offset in range(2 * context + 1)]
    return torch.cat(columns, dim=0).T


def cutoff_band(envelope, edges, fraction=_CUTOFF_ENERGY_FRACTION):
    """Return the index of the highest band the source still fills.

    Cumulative band energy reaches ``fraction`` of the total at the cutoff,
    the same rule the pipeline itself uses to locate a source's bandwidth.
    """
    power = (10.0**envelope).mean(dim=-1)
    weights = torch.tensor(
        [stop - start for start, stop in zip(edges, edges[1:])],
        dtype=power.dtype,
        device=power.device,
    )
    cumulative = torch.cumsum(power * weights, dim=0)
    threshold = cumulative[-1] * fraction
    below = int((cumulative < threshold).sum())
    return min(below, envelope.shape[0] - 1)


class EnvelopePredictor(nn.Module):
    """Predict the full-band envelope a frame should have from the source's."""

    def __init__(
        self,
        bands,
        context=_CONTEXT_FRAMES,
        hidden=_HIDDEN_UNITS,
    ):
        super().__init__()
        self.bands = bands
        self.context = context
        self.network = nn.Sequential(
            nn.Linear(bands * (2 * context + 1), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, bands),
        )

    def forward(self, features):
        return self.network(features)


def save_calibration(path, model):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "bands": model.bands,
            "context": model.context,
            "hidden": model.network[0].out_features,
            "n_fft": _N_FFT,
            "hop": _HOP,
            "bands_per_octave": _BANDS_PER_OCTAVE,
            "min_band_bin": _MIN_BAND_BIN,
        },
        path,
    )


def bundled_calibration_path():
    """Return the calibration the package ships.

    Trained across material types (piano, synthesizer, orchestra, Japanese
    and English speech) so no gate material regresses; every retrain must
    pass the sign gate over all of them before replacing this file.
    """
    return str(resources.files(__package__) / "weights" / "calibration_v3.pt")


def load_calibration(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key, expected in (
        ("n_fft", _N_FFT),
        ("hop", _HOP),
        ("bands_per_octave", _BANDS_PER_OCTAVE),
        ("min_band_bin", _MIN_BAND_BIN),
    ):
        if payload.get(key) != expected:
            raise ValueError(
                f"calibration was trained with {key}={payload.get(key)}, "
                f"this build uses {expected}"
            )
    model = EnvelopePredictor(
        payload["bands"], context=payload["context"], hidden=payload["hidden"]
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _frame_gain_floor(source_envelope):
    """Return the per-frame lower gain limit in dB.

    The floor stays at the base limit wherever the source carries signal and
    opens toward ``_QUIET_GAIN_FLOOR_DB`` only inside sustained pauses: frames
    well below the level of the clip's loud frames whose whole neighbourhood
    is equally quiet. The reference is an upper quantile, not the median — a
    clip that is mostly pauses would otherwise make the pauses themselves the
    reference and never detect one. The band-limited source cannot tell true
    silence from content that lived entirely above its cutoff (a lone
    hi-hat), but such content always has loud neighbours, so the sustain
    requirement keeps the base floor there.
    """
    power = (10.0**source_envelope).mean(dim=0)
    level_db = 10.0 * torch.log10(power + _POWER_FLOOR)
    depth_db = level_db.quantile(_QUIET_REFERENCE_QUANTILE) - level_db
    quietness = (
        (depth_db - _QUIET_ONSET_DB) / (_QUIET_FULL_DB - _QUIET_ONSET_DB)
    ).clamp(0.0, 1.0)
    window = 2 * _QUIET_SUSTAIN_FRAMES + 1
    sustained = -torch.nn.functional.max_pool1d(
        -quietness.view(1, 1, -1), kernel_size=window, stride=1, padding=window // 2
    ).view(-1)
    return _GAIN_LIMITS_DB[0] + sustained * (_QUIET_GAIN_FLOOR_DB - _GAIN_LIMITS_DB[0])


def _band_gains(model, restored, source, edges):
    """Return per-band gains in dB, zeroed at and below the source's cutoff."""
    source_envelope = band_envelope(source, edges)
    restored_envelope = band_envelope(restored, edges)
    frames = min(source_envelope.shape[-1], restored_envelope.shape[-1])
    source_envelope = source_envelope[:, :frames]
    restored_envelope = restored_envelope[:, :frames]

    offset = level_offset(source_envelope)
    with torch.no_grad():
        predicted = model(stack_context(source_envelope - offset)).T + offset

    cutoff = cutoff_band(source_envelope, edges)
    gains_db = 10.0 * (predicted - restored_envelope)
    gains_db = gains_db.clamp(max=_GAIN_LIMITS_DB[1])
    floor_db = _frame_gain_floor(source_envelope[: cutoff + 1])
    gains_db = torch.maximum(gains_db, floor_db.unsqueeze(0).to(gains_db.dtype))
    gains_db[: cutoff + 1] = 0.0

    if _GAIN_SMOOTHING_FRAMES > 1:
        gains_db = torch.nn.functional.avg_pool1d(
            gains_db.unsqueeze(0),
            kernel_size=_GAIN_SMOOTHING_FRAMES,
            stride=1,
            padding=_GAIN_SMOOTHING_FRAMES // 2,
            count_include_pad=False,
        ).squeeze(0)
    return gains_db, cutoff


def _gains_to_bins(gains_db, edges, n_bins):
    """Spread band gains across bins, linearly interpolated between centers."""
    centers = torch.tensor(
        [(start + stop - 1) / 2.0 for start, stop in zip(edges, edges[1:])],
        dtype=gains_db.dtype,
    )
    bins = torch.arange(n_bins, dtype=gains_db.dtype)
    position = torch.bucketize(bins, centers).clamp(1, centers.numel() - 1)
    left = centers[position - 1]
    right = centers[position]
    weight = ((bins - left) / (right - left)).clamp(0.0, 1.0)
    lower = gains_db[position - 1]
    upper = gains_db[position]
    interpolated = lower + (upper - lower) * weight.unsqueeze(-1)
    interpolated[: edges[0]] = 0.0
    return interpolated


def apply_calibration(restored, source, model):
    """Return ``restored`` with its envelope pulled toward the prediction.

    ``restored`` and ``source`` are ``[channels, samples]`` tensors at the
    model rate. Channels are calibrated independently against the matching
    source channel.
    """
    if restored.dim() != 2 or source.dim() != 2:
        raise ValueError("restored and source must have shape [channels, samples]")
    if restored.size(0) != source.size(0):
        raise ValueError(
            f"the source has {source.size(0)} channels but the restoration has "
            f"{restored.size(0)}"
        )

    length = restored.size(-1)
    if source.size(-1) < length:
        source = torch.nn.functional.pad(source, (0, length - source.size(-1)))
    else:
        source = source[..., :length]

    edges = band_edges()
    outputs = []
    for channel in range(restored.size(0)):
        spectrum = _spectrum(restored[channel])
        gains_db, _cutoff = _band_gains(
            model, restored[channel], source[channel], edges
        )
        frames = min(spectrum.shape[-1], gains_db.shape[-1])
        gain = 10.0 ** (
            _gains_to_bins(gains_db[:, :frames], edges, spectrum.shape[0]) / 20.0
        )
        calibrated = spectrum[:, :frames] * gain.to(spectrum.real.dtype)
        window = torch.hann_window(
            _N_FFT, periodic=True, dtype=restored.dtype, device=restored.device
        )
        outputs.append(
            torch.istft(
                calibrated,
                n_fft=_N_FFT,
                hop_length=_HOP,
                window=window,
                center=True,
                length=length,
            )
        )
    return torch.stack(outputs)
