import sys
import types
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import audiosr.pipeline as pipeline
import audiosr.utils as utils
from audiosr.latent_diffusion.modules.diffusionmodules.util import make_ddim_timesteps


def _patch_model_constructor(monkeypatch):
    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loaded = None

        def load_state_dict(self, state_dict, strict=False):
            self.loaded = (state_dict, strict)

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr(pipeline, "LatentDiffusion", FakeModel)
    monkeypatch.setattr(
        pipeline,
        "default_audioldm_config",
        lambda _model_name: {"model": {"params": {}}},
    )
    return FakeModel


@pytest.mark.parametrize("suffix", [".bin", ".BIN"])
def test_build_model_uses_explicit_bin_path_and_safe_load(monkeypatch, tmp_path, suffix):
    _patch_model_constructor(monkeypatch)
    requested = []

    def fake_load(path, **kwargs):
        requested.append((path, kwargs))
        return {"state_dict": {"weight": torch.tensor(1)}}

    monkeypatch.setattr(pipeline.torch, "load", fake_load)
    monkeypatch.setattr(pipeline, "download_checkpoint", lambda _name: pytest.fail("downloaded"))

    model = pipeline.build_model(
        ckpt_path=Path(tmp_path / f"model{suffix}"), device="cpu"
    )

    assert requested == [
        (str(tmp_path / f"model{suffix}"), {"map_location": "cpu", "weights_only": True})
    ]
    assert model.loaded[0] == {"weight": torch.tensor(1)}
    assert model.loaded[1] is False


def test_build_model_uses_safetensors_path_and_direct_state_dict(monkeypatch, tmp_path):
    _patch_model_constructor(monkeypatch)
    loaded = []

    safetensors_torch = types.ModuleType("safetensors.torch")

    def fake_load_file(path, **kwargs):
        loaded.append((path, kwargs))
        return {"weight": torch.tensor(2)}

    safetensors_torch.load_file = fake_load_file
    safetensors = types.ModuleType("safetensors")
    safetensors.torch = safetensors_torch
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)

    model = pipeline.build_model(
        ckpt_path=Path(tmp_path / "model.safetensors"), device="cpu"
    )

    assert loaded == [(str(tmp_path / "model.safetensors"), {"device": "cpu"})]
    assert model.loaded[0] == {"weight": torch.tensor(2)}


def test_pipeline_import_does_not_initialize_roberta_tokenizer():
    code = textwrap.dedent(
        """
        from transformers import RobertaTokenizer

        def fail(*args, **kwargs):
            raise RuntimeError("from_pretrained called during import")

        RobertaTokenizer.from_pretrained = fail
        import audiosr.pipeline
        import audiosr.latent_diffusion.modules.encoders.modules
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("steps", [0, 1001])
def test_make_ddim_timesteps_rejects_invalid_step_count(steps):
    with pytest.raises(ValueError, match="between 1"):
        make_ddim_timesteps("uniform", steps, 1000, verbose=False)


@pytest.mark.parametrize("steps", [1, 17, 500, 501, 1000])
def test_make_ddim_timesteps_matches_count_and_stays_in_bounds(steps):
    timesteps = make_ddim_timesteps("uniform", steps, 1000, verbose=False)

    assert len(timesteps) == steps
    assert np.all(np.diff(timesteps) > 0)
    assert timesteps.min() >= 0
    assert timesteps.max() <= 999


def test_make_ddim_timesteps_preserves_default_schedule():
    timesteps = make_ddim_timesteps("uniform", 50, 1000, verbose=False)

    np.testing.assert_array_equal(timesteps, np.arange(0, 1000, 20) + 1)


def test_pipeline_uses_canonical_seed_helper():
    assert pipeline.seed_everything is utils.seed_everything

    pipeline.seed_everything(123)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_make_batch_waveform_uses_integer_padding(monkeypatch):
    calls = {}

    def fake_pad(waveform, target_length):
        calls["target_length"] = target_length
        return np.zeros((1, target_length), dtype=np.float32)

    def fake_features(waveform, target_frame):
        calls.setdefault("target_frames", []).append(target_frame)
        return torch.zeros(target_frame, 2), torch.zeros(target_frame, 2)

    monkeypatch.setattr(pipeline, "normalize_wav", lambda waveform: waveform)
    monkeypatch.setattr(pipeline, "pad_wav", fake_pad)
    monkeypatch.setattr(pipeline, "wav_feature_extraction", fake_features)
    monkeypatch.setattr(
        pipeline,
        "lowpass_filtering_prepare_inference",
        lambda batch, filter_type=None: {
            "waveform_lowpass": batch["waveform"]
        },
    )

    original_samples = pipeline._SEGMENT_SAMPLES + 1
    batch, duration = pipeline.make_batch_for_super_resolution(
        None, waveform=np.ones(original_samples, dtype=np.float32)
    )

    assert calls["target_length"] == 2 * pipeline._SEGMENT_SAMPLES
    assert calls["target_frames"] == [2 * pipeline._SEGMENT_SAMPLES // 480] * 2
    assert duration == pytest.approx(original_samples / 48000)
    assert batch["waveform"].shape == (1, 1, 2 * pipeline._SEGMENT_SAMPLES)


def test_super_resolution_processes_stereo_and_returns_bct(monkeypatch):
    source = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    monkeypatch.setattr(pipeline, "load_audio", lambda *_args, **_kwargs: (source, 48000))
    monkeypatch.setattr(
        pipeline,
        "make_batch_for_super_resolution",
        lambda _input, waveform=None: ({"waveform": torch.as_tensor(waveform)}, 3 / 48000),
    )
    seeds = []
    monkeypatch.setattr(pipeline, "seed_everything", lambda seed: seeds.append(seed))

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def generate_batch(self, batch, **_kwargs):
            self.calls += 1
            return torch.full((1, 1, 5), float(self.calls))

    model = FakeModel()
    result = pipeline.super_resolution(model, "input.wav", seed=17)

    assert model.calls == 2
    assert seeds == [17, 17]
    assert result.shape == (1, 2, 3)
    np.testing.assert_array_equal(result[0, 0], np.ones(3))
    np.testing.assert_array_equal(result[0, 1], np.full(3, 2))


def test_super_resolution_batch_groups_once_and_trims(monkeypatch):
    inputs = [np.ones(3, dtype=np.float32), np.ones(7, dtype=np.float32)]
    padded = []

    def fake_prepare(
        waveform,
        padded_samples,
        add_batch_dimension=False,
        lowpass_filter_type=None,
    ):
        padded.append(
            (
                len(waveform),
                padded_samples,
                add_batch_dimension,
                lowpass_filter_type,
            )
        )
        return {"waveform": torch.zeros(1, padded_samples)}, len(waveform) / 48000

    monkeypatch.setattr(pipeline, "_prepare_mono_batch", fake_prepare)
    monkeypatch.setattr(
        pipeline,
        "_select_lowpass_filter_type",
        lambda seed=None: "ellip" if seed == 17 else pytest.fail(str(seed)),
    )
    calls = []

    class FakeModel:
        def generate_batch(self, batch, **kwargs):
            calls.append((batch, kwargs))
            size = batch["waveform"].shape[0]
            return torch.ones(size, 1, pipeline._SEGMENT_SAMPLES)

    result = pipeline.super_resolution_batch(
        FakeModel(), inputs, seed=42, lowpass_seed=17
    )

    assert len(calls) == 1
    assert calls[0][0]["waveform"].shape == (2, 1, pipeline._SEGMENT_SAMPLES)
    assert calls[0][1]["sampler"] == "ddim"
    assert calls[0][1]["ddim_eta"] == 1.0
    assert padded == [
        (3, pipeline._SEGMENT_SAMPLES, False, "ellip"),
        (7, pipeline._SEGMENT_SAMPLES, False, "ellip"),
    ]
    assert [item.shape for item in result] == [(3,), (7,)]


def test_super_resolution_batch_empty_list_does_not_warm_up():
    class FailModel:
        def generate_batch(self, *_args, **_kwargs):
            raise AssertionError("must not generate for an empty list")

    assert pipeline.super_resolution_batch(FailModel(), []) == []


def test_long_audio_stereo_short_tail_and_exact_length(monkeypatch):
    source = torch.stack([torch.arange(8, dtype=torch.float32), torch.arange(8, dtype=torch.float32) + 1])
    source[0, 0] = torch.nan
    monkeypatch.setattr(pipeline, "load_audio", lambda *_args, **_kwargs: (source, 48000))
    filter_types = []

    def fake_make_batch(
        _input, waveform=None, fbank=None, lowpass_filter_type=None
    ):
        filter_types.append(lowpass_filter_type)
        return {"waveform": torch.as_tensor(waveform)}, len(waveform) / 48000

    monkeypatch.setattr(pipeline, "make_batch_for_super_resolution", fake_make_batch)
    monkeypatch.setattr(
        pipeline, "_select_lowpass_filter_type", lambda seed=None: "bessel"
    )

    class FakeModel:
        def generate_batch(self, *_args, **_kwargs):
            return torch.ones(1, 1, 16)

    result = pipeline.super_resolution_long_audio(
        FakeModel(),
        "input.wav",
        chunk_duration_s=6 / 48000,
        overlap_duration_s=4 / 48000,
    )

    assert result.shape == (1, 2, 8)
    assert torch.isfinite(result).all()
    assert filter_types == ["bessel"] * 8


@pytest.mark.parametrize(
    "chunk, overlap",
    [(0, 1), (-1, 1), (1, -1), (1, 1), (1, 2)],
)
def test_long_audio_rejects_invalid_chunk_arguments(monkeypatch, chunk, overlap):
    monkeypatch.setattr(
        pipeline,
        "load_audio",
        lambda *_args, **_kwargs: (torch.ones(1, 8), 48000),
    )
    with pytest.raises(ValueError):
        pipeline.super_resolution_long_audio(
            object(), "input.wav", chunk_duration_s=chunk, overlap_duration_s=overlap
        )


def test_long_audio_allows_zero_overlap(monkeypatch):
    source = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    monkeypatch.setattr(
        pipeline, "load_audio", lambda *_args, **_kwargs: (source, 48000)
    )
    monkeypatch.setattr(
        pipeline,
        "make_batch_for_super_resolution",
        lambda _input, waveform=None, lowpass_filter_type=None: (
            {"waveform": torch.as_tensor(waveform)},
            len(waveform) / 48000,
        ),
    )

    class FakeModel:
        def generate_batch(self, batch, **_kwargs):
            return torch.as_tensor(batch["waveform"]).reshape(1, 1, -1)

    result = pipeline.super_resolution_long_audio(
        FakeModel(),
        "input.wav",
        chunk_duration_s=4 / 48000,
        overlap_duration_s=0,
    )

    assert result.shape == (1, 1, 8)
    assert torch.isfinite(result).all()


def test_long_audio_batches_chunks_without_reducing_ddim_steps(monkeypatch):
    source = torch.arange(12, dtype=torch.float32).unsqueeze(0)
    monkeypatch.setattr(
        pipeline, "load_audio", lambda *_args, **_kwargs: (source, 48000)
    )
    monkeypatch.setattr(pipeline, "_padded_sample_count", lambda samples: samples)
    monkeypatch.setattr(
        pipeline, "_select_lowpass_filter_type", lambda seed=None: "bessel"
    )

    def fake_prepare(
        waveform,
        padded_samples,
        add_batch_dimension=False,
        lowpass_filter_type=None,
    ):
        del add_batch_dimension, lowpass_filter_type
        return {
            "waveform": torch.as_tensor(waveform).reshape(1, padded_samples)
        }, len(waveform) / 48000

    monkeypatch.setattr(pipeline, "_prepare_mono_batch", fake_prepare)
    calls = []

    class FakeModel:
        def generate_batch(self, batch, **kwargs):
            calls.append(
                (
                    batch["waveform"].shape[0],
                    kwargs,
                    torch.is_inference_mode_enabled(),
                )
            )
            return batch["waveform"]

    result = pipeline.super_resolution_long_audio(
        FakeModel(),
        "input.wav",
        ddim_steps=200,
        chunk_duration_s=4 / 48000,
        overlap_duration_s=0,
        batch_size=2,
    )

    assert result.shape == (1, 1, 12)
    assert torch.isfinite(result).all()
    assert [batch_size for batch_size, _kwargs, _inference_mode in calls] == [2, 1]
    assert [
        kwargs["ddim_steps"] for _batch_size, kwargs, _inference_mode in calls
    ] == [200, 200]
    assert all(inference_mode for _batch_size, _kwargs, inference_mode in calls)


def test_long_audio_batch_retries_single_chunks_after_accelerator_oom(monkeypatch):
    source = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    monkeypatch.setattr(
        pipeline, "load_audio", lambda *_args, **_kwargs: (source, 48000)
    )
    monkeypatch.setattr(pipeline, "_padded_sample_count", lambda samples: samples)
    monkeypatch.setattr(
        pipeline, "_select_lowpass_filter_type", lambda seed=None: "ellip"
    )

    def fake_prepare(
        waveform,
        padded_samples,
        add_batch_dimension=False,
        lowpass_filter_type=None,
    ):
        del add_batch_dimension, lowpass_filter_type
        return {
            "waveform": torch.as_tensor(waveform).reshape(1, padded_samples)
        }, len(waveform) / 48000

    monkeypatch.setattr(pipeline, "_prepare_mono_batch", fake_prepare)
    batch_sizes = []

    class OomModel:
        def generate_batch(self, batch, **_kwargs):
            current_batch_size = batch["waveform"].shape[0]
            batch_sizes.append(current_batch_size)
            if current_batch_size > 1:
                raise RuntimeError("CUDA out of memory")
            return batch["waveform"]

    result = pipeline.super_resolution_long_audio(
        OomModel(),
        "input.wav",
        ddim_steps=200,
        chunk_duration_s=4 / 48000,
        overlap_duration_s=0,
        batch_size=2,
    )

    assert result.shape == (1, 1, 8)
    assert torch.isfinite(result).all()
    assert batch_sizes == [2, 1, 1]


@pytest.mark.parametrize("batch_size", [0, 9, 1.5])
def test_long_audio_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        pipeline.super_resolution_long_audio(
            object(),
            "input.wav",
            batch_size=batch_size,
        )


def test_super_resolution_forwards_the_requested_sampler(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "load_audio",
        lambda *_args, **_kwargs: (np.zeros((1, pipeline._SEGMENT_SAMPLES), dtype=np.float32), 48000),
    )
    monkeypatch.setattr(
        pipeline,
        "make_batch_for_super_resolution",
        lambda *_args, **_kwargs: ({"waveform": torch.zeros(1, 1, 4)}, 5.12),
    )
    calls = []

    class FakeModel:
        def generate_batch(self, batch, **kwargs):
            calls.append(kwargs)
            return torch.ones(1, 1, pipeline._SEGMENT_SAMPLES)

    pipeline.super_resolution(
        FakeModel(), "input.wav", sampler="dpmpp2m", ddim_eta=0.0
    )

    assert len(calls) == 1
    assert calls[0]["sampler"] == "dpmpp2m"
    assert calls[0]["ddim_eta"] == 0.0


@pytest.mark.parametrize(
    "entry_point",
    ["super_resolution", "super_resolution_long_audio", "super_resolution_batch"],
)
def test_public_entry_points_reject_an_unknown_sampler(entry_point):
    with pytest.raises(ValueError, match="sampler must be one of"):
        getattr(pipeline, entry_point)(object(), "input.wav", sampler="euler")


@pytest.mark.parametrize("bad_eta", [-0.1, float("nan"), float("inf")])
def test_public_entry_points_reject_an_invalid_eta(bad_eta):
    with pytest.raises(ValueError, match="ddim_eta must"):
        pipeline.super_resolution(object(), "input.wav", ddim_eta=bad_eta)


# --------------------------------------------------------- shared cond VAE ---


def _cond_config(target=None):
    if target is None:
        target = pipeline._VAE_FEATURE_EXTRACT
    return {
        "model": {
            "params": {
                "cond_stage_config": {
                    "concat_lowpass_cond": {
                        "target": target,
                        "cond_stage_key": "lowpass_mel",
                        "conditioning_key": "concat",
                        "params": {"first_stage_config": {"target": "irrelevant"}},
                    }
                }
            }
        }
    }


def _cond_state_dict(cond_weight=None, cond_keys=("encoder.weight", "encoder.bias")):
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.zeros(2)
    state_dict = {
        "first_stage_model.encoder.weight": weight,
        "first_stage_model.encoder.bias": bias,
        "model.diffusion_model.weight": torch.ones(1),
    }
    source = {"encoder.weight": weight if cond_weight is None else cond_weight,
              "encoder.bias": bias}
    for key in cond_keys:
        state_dict[f"cond_stage_models.0.vae.{key}"] = source.get(key, torch.zeros(1)).clone()
    return state_dict


def test_matching_conditioning_weights_are_reported_as_shareable():
    shared, redundant = pipeline.duplicated_first_stage_cond(
        _cond_config(), _cond_state_dict()
    )

    assert shared == ("concat_lowpass_cond",)
    assert set(redundant) == {
        "cond_stage_models.0.vae.encoder.weight",
        "cond_stage_models.0.vae.encoder.bias",
    }


def test_differing_conditioning_weights_are_kept_separate():
    """A checkpoint that trained the two stages apart must keep both modules."""
    state_dict = _cond_state_dict(cond_weight=torch.ones(2, 3))

    assert pipeline.duplicated_first_stage_cond(_cond_config(), state_dict) == ((), ())


def test_a_partially_matching_conditioning_stage_is_kept_separate():
    state_dict = _cond_state_dict(cond_keys=("encoder.weight",))

    assert pipeline.duplicated_first_stage_cond(_cond_config(), state_dict) == ((), ())


def test_only_a_vae_conditioning_stage_can_share():
    config = _cond_config(target="audiosr.something.Else")

    assert pipeline.duplicated_first_stage_cond(config, _cond_state_dict()) == ((), ())


def test_a_configuration_without_a_first_stage_shares_nothing():
    assert pipeline.duplicated_first_stage_cond(
        _cond_config(), {"model.diffusion_model.weight": torch.ones(1)}
    ) == ((), ())


def test_build_model_drops_the_duplicated_conditioning_weights(monkeypatch, tmp_path):
    FakeModel = _patch_model_constructor(monkeypatch)
    monkeypatch.setattr(pipeline, "default_audioldm_config", lambda _name: _cond_config())
    monkeypatch.setattr(
        pipeline.torch, "load", lambda *a, **k: {"state_dict": _cond_state_dict()}
    )

    model = pipeline.build_model(ckpt_path=tmp_path / "model.bin", device="cpu")

    assert model.kwargs["share_first_stage_cond"] == ("concat_lowpass_cond",)
    assert not [key for key in model.loaded[0] if key.startswith("cond_stage_models.")]
    assert "first_stage_model.encoder.weight" in model.loaded[0]


def test_build_model_keeps_conditioning_weights_that_differ(monkeypatch, tmp_path):
    _patch_model_constructor(monkeypatch)
    monkeypatch.setattr(pipeline, "default_audioldm_config", lambda _name: _cond_config())
    monkeypatch.setattr(
        pipeline.torch,
        "load",
        lambda *a, **k: {"state_dict": _cond_state_dict(cond_weight=torch.ones(2, 3))},
    )

    model = pipeline.build_model(ckpt_path=tmp_path / "model.bin", device="cpu")

    assert model.kwargs["share_first_stage_cond"] == ()
    assert "cond_stage_models.0.vae.encoder.weight" in model.loaded[0]


def test_the_feature_extractor_aliases_a_supplied_vae():
    from audiosr.latent_diffusion.modules.encoders.modules import VAEFeatureExtract

    vae = torch.nn.Linear(2, 2)

    assert VAEFeatureExtract(shared_vae=vae).vae is vae


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"first_stage_config": {"target": "x"}, "shared_vae": torch.nn.Linear(2, 2)}],
)
def test_the_feature_extractor_needs_exactly_one_source(kwargs):
    from audiosr.latent_diffusion.modules.encoders.modules import VAEFeatureExtract

    with pytest.raises(ValueError, match="exactly one"):
        VAEFeatureExtract(**kwargs)


# ------------------------------------------------------ high rate passthrough ---


def _tone(frequency, sample_rate, seconds=1.0, amplitude=1.0):
    t = torch.arange(int(sample_rate * seconds), dtype=torch.float64) / sample_rate
    return (amplitude * torch.sin(2 * torch.pi * frequency * t)).float()


def _write_source(tmp_path, channels, sample_rate, name="source.wav"):
    import soundfile as sf

    path = tmp_path / name
    sf.write(str(path), torch.stack(channels, dim=-1).numpy(), sample_rate)
    return path


def _band_energy(waveform, sample_rate, low_hz, high_hz):
    spectrum = torch.fft.rfft(torch.as_tensor(waveform, dtype=torch.float64))
    freqs = torch.fft.rfftfreq(waveform.shape[-1], 1 / sample_rate)
    band = (freqs >= low_hz) & (freqs < high_hz)
    return float((spectrum[band].abs() ** 2).sum())


def _downsample_to_model_rate(waveform, sample_rate):
    import torchaudio

    return torchaudio.functional.resample(waveform, sample_rate, pipeline._SAMPLE_RATE)


def test_a_96k_source_keeps_its_own_ultrasonic_band(tmp_path):
    """The model cannot reach past 24 kHz, so what the source held goes back."""
    source = _tone(1000, 96000) + _tone(30000, 96000, amplitude=0.5)
    path = _write_source(tmp_path, [source], 96000)
    restored = _downsample_to_model_rate(source.unsqueeze(0), 96000)

    output, rate = pipeline.restore_high_rate(restored, path)

    assert rate == 96000
    assert output.shape == (1, 1, source.shape[-1])
    # The 48 kHz restoration cannot hold the 30 kHz tone; the output does.
    ultrasonic = _band_energy(output[0, 0], 96000, 29000, 31000)
    audible = _band_energy(output[0, 0], 96000, 900, 1100)
    assert ultrasonic > 0.01 * audible


def test_the_passthrough_invents_nothing_above_the_model_ceiling(tmp_path):
    """A source with an empty top band must come back with an empty top band."""
    source = _tone(1000, 96000)
    path = _write_source(tmp_path, [source], 96000)
    restored = _downsample_to_model_rate(source.unsqueeze(0), 96000)

    output, rate = pipeline.restore_high_rate(restored, path)

    assert rate == 96000
    above = _band_energy(output[0, 0], 96000, 25000, 48000)
    audible = _band_energy(output[0, 0], 96000, 900, 1100)
    assert above < 1e-6 * audible


def test_a_48k_source_is_returned_unchanged(tmp_path):
    source = _tone(1000, 48000)
    path = _write_source(tmp_path, [source], 48000)

    output, rate = pipeline.restore_high_rate(source.unsqueeze(0), path)

    assert rate == 48000
    assert torch.allclose(torch.from_numpy(output[0, 0]), source, atol=1e-6)


def test_a_44k_source_is_resampled_without_a_splice(tmp_path):
    source = _tone(1000, 44100)
    path = _write_source(tmp_path, [source], 44100)
    restored = _downsample_to_model_rate(source.unsqueeze(0), 44100)

    output, rate = pipeline.restore_high_rate(restored, path)

    assert rate == 44100
    assert abs(output.shape[-1] - source.shape[-1]) <= 2


def test_an_explicit_target_rate_overrides_the_source(tmp_path):
    source = _tone(1000, 96000) + _tone(30000, 96000, amplitude=0.5)
    path = _write_source(tmp_path, [source], 96000)
    restored = _downsample_to_model_rate(source.unsqueeze(0), 96000)

    output, rate = pipeline.restore_high_rate(restored, path, target_sample_rate=64000)

    assert rate == 64000
    assert _band_energy(output[0, 0], 64000, 29000, 31000) > 0.0


def test_stereo_is_spliced_per_channel(tmp_path):
    left = _tone(1000, 96000) + _tone(30000, 96000, amplitude=0.5)
    right = _tone(1000, 96000)
    path = _write_source(tmp_path, [left, right], 96000)
    restored = _downsample_to_model_rate(torch.stack([left, right]), 96000)

    output, _ = pipeline.restore_high_rate(restored, path)

    assert output.shape[1] == 2
    assert _band_energy(output[0, 0], 96000, 29000, 31000) > 100 * _band_energy(
        output[0, 1], 96000, 29000, 31000
    )


def test_a_channel_count_mismatch_is_rejected(tmp_path):
    source = _tone(1000, 96000)
    path = _write_source(tmp_path, [source, source], 96000)
    restored = _downsample_to_model_rate(source.unsqueeze(0), 96000)

    with pytest.raises(ValueError, match="channels"):
        pipeline.restore_high_rate(restored, path)


def test_a_batched_restoration_is_rejected(tmp_path):
    source = _tone(1000, 96000)
    path = _write_source(tmp_path, [source], 96000)
    restored = _downsample_to_model_rate(source.unsqueeze(0), 96000)

    with pytest.raises(ValueError, match="single item"):
        pipeline.restore_high_rate(restored.unsqueeze(0).repeat(2, 1, 1), path)
