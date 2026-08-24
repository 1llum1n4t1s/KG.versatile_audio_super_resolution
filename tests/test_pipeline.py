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
    assert calls[0][1]["duration"] == pipeline._SEGMENT_SAMPLES / 48000
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
