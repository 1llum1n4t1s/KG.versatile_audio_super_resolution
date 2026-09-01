import importlib
import sys
import types

import numpy as np


class _FakeComponent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _load_app(monkeypatch):
    calls = []

    fake_gradio = types.ModuleType("gradio")

    class FakeInterface(_FakeComponent):
        launch_count = 0

        def launch(self):
            type(self).launch_count += 1

    fake_gradio.Interface = FakeInterface
    fake_gradio.Audio = _FakeComponent
    fake_gradio.Dropdown = _FakeComponent
    fake_gradio.Slider = _FakeComponent
    fake_gradio.Number = _FakeComponent

    fake_librosa = types.ModuleType("librosa")
    fake_librosa.feature = types.SimpleNamespace(
        rms=lambda y, frame_length, hop_length: np.ones(
            (1, max(1, int(np.ceil(len(y) / hop_length))),), dtype=np.float32
        )
    )
    fake_librosa.amplitude_to_db = lambda values, ref: np.zeros_like(values)

    def fake_batch(
        model,
        chunks,
        seed,
        ddim_steps,
        guidance_scale,
        lowpass_seed=None,
    ):
        calls.append(
            {
                "model": model,
                "chunks": [np.asarray(chunk).copy() for chunk in chunks],
                "seed": seed,
                "ddim_steps": ddim_steps,
                "guidance_scale": guidance_scale,
                "lowpass_seed": lowpass_seed,
            }
        )
        return [np.asarray(chunk).copy() for chunk in chunks]

    fake_audiosr = types.ModuleType("audiosr")
    fake_audiosr.build_calls = []

    def fake_build_model(model_name):
        fake_audiosr.build_calls.append(model_name)
        return object()

    fake_audiosr.build_model = fake_build_model
    fake_pipeline = types.ModuleType("audiosr.pipeline")
    fake_pipeline.super_resolution_batch = fake_batch
    fake_audiosr.pipeline = fake_pipeline

    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "audiosr", fake_audiosr)
    monkeypatch.setitem(sys.modules, "audiosr.pipeline", fake_pipeline)
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    app.load_audio = lambda *_args, **_kwargs: (
        np.ones((1, 8), dtype=np.float32),
        48000,
    )
    return app, fake_librosa, calls, fake_gradio


def test_import_does_not_launch_and_interface_keeps_batch_controls(monkeypatch):
    app, _librosa, _calls, fake_gradio = _load_app(monkeypatch)

    assert fake_gradio.Interface.launch_count == 0
    interface = app.create_interface()
    sliders = [
        component
        for component in interface.kwargs["inputs"]
        if isinstance(component, _FakeComponent)
        and "Batch Size" in component.kwargs.get("label", "")
    ]
    assert len(sliders) == 1
    assert sliders[0].args[:2] == (1, 8)
    assert sliders[0].kwargs["value"] == 1
    seed_inputs = [
        component
        for component in interface.kwargs["inputs"]
        if component.kwargs.get("label") == "Seed"
    ]
    assert len(seed_inputs) == 1
    assert seed_inputs[0].kwargs["value"] == 42
    ddim_inputs = [
        component
        for component in interface.kwargs["inputs"]
        if component.kwargs.get("label") == "DDIM Steps"
    ]
    assert len(ddim_inputs) == 1
    assert ddim_inputs[0].kwargs["value"] == 50


def test_normalize_audio_preserves_level_unless_peak_exceeds_one(monkeypatch):
    app, _librosa, _calls, _fake_gradio = _load_app(monkeypatch)
    quiet = np.array([-0.25, 0.5], dtype=np.float32)
    loud = np.array([-2.0, 1.0], dtype=np.float32)

    np.testing.assert_array_equal(app.normalize_audio(quiet), quiet)
    np.testing.assert_array_equal(
        app.normalize_audio(loud), np.array([-1.0, 0.5], dtype=np.float32)
    )


def test_release_cached_model_clears_mps_cache(monkeypatch):
    app, _librosa, _calls, _fake_gradio = _load_app(monkeypatch)
    cache_calls = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        ),
        mps=types.SimpleNamespace(empty_cache=lambda: cache_calls.append("mps")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    app._MODEL_CACHE["basic"] = object()

    app._release_cached_model()

    assert app._MODEL_CACHE == {}
    assert cache_calls == ["mps"]


def test_process_audio_channel_groups_batches_and_preserves_order(monkeypatch):
    app, _librosa, calls, _fake_gradio = _load_app(monkeypatch)
    source = np.arange(120, dtype=np.float32)

    output = app.process_audio_channel(
        object(), source, sr=10, guidance_scale=2.6, ddim_steps=100, batch_size=2
    )

    assert [len(call["chunks"]) for call in calls] == [2, 1]
    assert [int(call["chunks"][0][0]) for call in calls] == [0, 92]
    assert [call["seed"] for call in calls] == [42, 44]
    assert [call["lowpass_seed"] for call in calls] == [42, 42]
    assert output.shape == source.shape


def test_chunk_reaching_eof_is_not_followed_by_contained_tail(monkeypatch):
    app, _librosa, calls, _fake_gradio = _load_app(monkeypatch)
    source = np.ones(470, dtype=np.float32)

    output = app.process_audio_channel(
        object(), source, sr=100, guidance_scale=2.6, ddim_steps=100, batch_size=1
    )

    # chunk=510 samples, so the first chunk already contains the entire input.
    assert [len(call["chunks"][0]) for call in calls] == [470]
    assert output.shape == source.shape
    np.testing.assert_array_equal(output, source)


def test_process_chunk_flattens_legacy_batch_axis_and_trims(monkeypatch):
    app, _librosa, _calls, _fake_gradio = _load_app(monkeypatch)
    app.super_resolution_batch = lambda *_args: [
        np.ones((1, 32), dtype=np.float32)
    ]

    result = app.process_chunk(
        object(),
        np.ones(8, dtype=np.float32),
        sr=48000,
        guidance_scale=2.6,
        ddim_steps=100,
        is_last_chunk=True,
        target_length=8,
    )

    assert result.ndim == 1
    assert result.shape == (8,)


def test_inference_uses_canonical_loader_and_preserves_channel_shapes(monkeypatch):
    app, _fake_librosa, _calls, _fake_gradio = _load_app(monkeypatch)
    load_calls = []

    def load_mono(path, target_sample_rate=None):
        load_calls.append((path, target_sample_rate))
        return np.arange(8, dtype=np.float32)[None, :], 48000

    app.load_audio = load_mono
    _sr, mono = app.inference("mono.wav", "basic", 2.6, 100, 1)
    assert mono.ndim == 1
    assert load_calls == [("mono.wav", 48000)]

    app.load_audio = lambda *_args, **_kwargs: (
        np.stack(
            [np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) + 1]
        ),
        48000,
    )
    _sr, stereo = app.inference("stereo.wav", "basic", 2.6, 100, 1)
    assert stereo.shape == (8, 2)
    np.testing.assert_allclose(stereo[:, 1] - stereo[:, 0], 1 / 8)


def test_inference_reuses_model_until_model_name_changes(monkeypatch):
    app, _fake_librosa, _calls, _fake_gradio = _load_app(monkeypatch)
    fake_audiosr = sys.modules["audiosr"]

    app.inference("first.wav", "basic", 2.6, 50, 1, 7)
    app.inference("second.wav", "basic", 2.6, 50, 1, 8)
    app.inference("third.wav", "speech", 2.6, 50, 1, 9)

    assert fake_audiosr.build_calls == ["basic", "speech"]


def test_process_audio_channel_uses_distinct_reproducible_batch_seeds(monkeypatch):
    app, _fake_librosa, calls, _fake_gradio = _load_app(monkeypatch)

    app.process_audio_channel(
        object(),
        np.ones(120, dtype=np.float32),
        sr=10,
        guidance_scale=2.6,
        ddim_steps=50,
        batch_size=1,
        seed=7,
    )

    assert [call["seed"] for call in calls] == [7, 8, 9]
    assert [call["lowpass_seed"] for call in calls] == [7, 7, 7]
