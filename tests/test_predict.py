import importlib
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf


def test_cog_predictor_writes_bct_output_as_multichannel_audio(
    monkeypatch, tmp_path
):
    fake_cog = types.ModuleType("cog")
    fake_cog.BasePredictor = object
    fake_cog.Input = lambda **_kwargs: None
    fake_cog.Path = Path

    source = np.stack(
        [np.linspace(-0.5, 0.5, 32), np.linspace(0.5, -0.5, 32)]
    ).astype(np.float32)
    fake_audiosr = types.ModuleType("audiosr")
    fake_audiosr.build_model = lambda **_kwargs: object()
    fake_audiosr.super_resolution = lambda *_args, **_kwargs: source[None, ...]

    monkeypatch.setitem(sys.modules, "cog", fake_cog)
    monkeypatch.setitem(sys.modules, "audiosr", fake_audiosr)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("predict", None)
    predict = importlib.import_module("predict")

    predictor = predict.Predictor()
    predictor.audiosr = object()
    output = predictor.predict(Path("input.wav"), seed=42)

    info = sf.info(output)
    assert info.frames == 32
    assert info.channels == 2
    assert info.samplerate == 48000


def test_cog_uses_supported_python_and_local_fork_code():
    cog = (Path(__file__).resolve().parents[1] / "cog.yaml").read_text(
        encoding="utf-8"
    )
    assert 'python_version: "3.10"' in cog
    assert '"audiosr==0.0.7"' not in cog
    for dependency in ("Pillow", "requests", "scikit-learn", "matplotlib"):
        assert f'"{dependency}==' in cog
