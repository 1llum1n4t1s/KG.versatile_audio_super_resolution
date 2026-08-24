import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "audiosr" / "__main__.py"


def load_cli_module():
    """Load the CLI source without importing audiosr's model package."""
    spec = importlib.util.spec_from_file_location("audiosr_cli_test_module", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parser_requires_one_input_and_rejects_both():
    parser = load_cli_module().build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["-i", "input.wav", "-il", "inputs.txt"])

    args = parser.parse_args(
        [
            "--input_file_list",
            "inputs.txt",
            "--chunking",
            "--chunk_duration",
            "11",
            "--overlap_duration",
            "3",
        ]
    )
    assert args.input_file_list == "inputs.txt"
    assert args.input_audio_file is None
    assert args.chunking is True
    assert args.chunk_duration == 11
    assert args.overlap_duration == 3


@pytest.mark.parametrize("steps", ["0", "1001", "not-an-integer"])
def test_parser_rejects_ddim_steps_outside_sampler_range(steps):
    parser = load_cli_module().build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["-i", "input.wav", "--ddim_steps", steps])


def test_main_forwards_chunking_arguments_and_skips_blank_list_entries(
    monkeypatch, tmp_path
):
    cli = load_cli_module()
    build_model = Mock(return_value="model")
    read_list = Mock(return_value=["", "  ", "first.wav", "\t", "second.wav"])
    super_resolution_long_audio = Mock(return_value="waveform")
    save_wave = Mock()

    fake_audiosr = ModuleType("audiosr")
    fake_audiosr.build_model = build_model
    fake_audiosr.get_time = Mock(return_value="timestamp")
    fake_audiosr.read_list = read_list
    fake_audiosr.save_wave = save_wave
    fake_audiosr.super_resolution = Mock()
    fake_audiosr.super_resolution_long_audio = super_resolution_long_audio
    fake_torch = SimpleNamespace(set_float32_matmul_precision=Mock())
    monkeypatch.setitem(sys.modules, "audiosr", fake_audiosr)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    status = cli.main(
        [
            "-il",
            "inputs.txt",
            "-s",
            str(tmp_path),
            "--model_name",
            "speech",
            "-d",
            "cpu",
            "--ddim_steps",
            "17",
            "-gs",
            "4.25",
            "--seed",
            "99",
            "--suffix",
            "_out",
            "--chunking",
            "--chunk_duration",
            "11",
            "--overlap_duration",
            "3",
        ]
    )

    assert status == 0
    build_model.assert_called_once_with(model_name="speech", device="cpu")
    super_resolution_long_audio.assert_has_calls(
        [
            call(
                "model",
                "first.wav",
                seed=99,
                guidance_scale=4.25,
                ddim_steps=17,
                chunk_duration_s=11,
                overlap_duration_s=3,
            ),
            call(
                "model",
                "second.wav",
                seed=99,
                guidance_scale=4.25,
                ddim_steps=17,
                chunk_duration_s=11,
                overlap_duration_s=3,
            ),
        ]
    )
    assert save_wave.call_count == 2
    assert all(call.kwargs["samplerate"] == 48000 for call in save_wave.call_args_list)


def test_main_forwards_standard_processing_arguments(monkeypatch, tmp_path):
    cli = load_cli_module()
    build_model = Mock(return_value="model")
    super_resolution = Mock(return_value="waveform")
    fake_audiosr = ModuleType("audiosr")
    fake_audiosr.build_model = build_model
    fake_audiosr.get_time = Mock(return_value="timestamp")
    fake_audiosr.read_list = Mock()
    fake_audiosr.save_wave = Mock()
    fake_audiosr.super_resolution = super_resolution
    fake_audiosr.super_resolution_long_audio = Mock()
    fake_torch = SimpleNamespace(set_float32_matmul_precision=Mock())
    monkeypatch.setitem(sys.modules, "audiosr", fake_audiosr)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert cli.main(["-i", "input.wav", "-s", str(tmp_path)]) == 0

    super_resolution.assert_called_once_with(
        "model",
        "input.wav",
        seed=42,
        guidance_scale=3.5,
        ddim_steps=50,
        latent_t_per_second=12.8,
    )


def test_setup_metadata_and_requirements_are_synchronized(monkeypatch):
    captured = {}

    def fake_setup(**kwargs):
        captured.update(kwargs)

    fake_setuptools = ModuleType("setuptools")
    fake_setuptools.setup = fake_setup
    fake_setuptools.find_packages = lambda: ["audiosr"]
    monkeypatch.setitem(sys.modules, "setuptools", fake_setuptools)
    namespace = runpy.run_path(str(ROOT / "setup.py"))

    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert set(namespace["REQUIRED"]) == requirements
    assert captured["install_requires"] == namespace["REQUIRED"]
    assert captured["python_requires"] == ">=3.10,<3.15"
    assert captured["version"] == "1.0.0"
    assert captured["url"] == "https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution"
    assert captured["author_email"] == ""
    assert captured["extras_require"]["test"] == ["pytest>=9.1.1,<10"]
    assert captured["entry_points"] == {
        "console_scripts": ["audiosr=audiosr.__main__:main"]
    }
    assert "scripts" not in captured
    assert "cmdclass" not in captured


def test_build_system_declares_modern_setuptools():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires = ["setuptools>=84", "wheel>=0.48"]' in pyproject
    assert 'build-backend = "setuptools.build_meta"' in pyproject


def test_module_help_does_not_import_model_pipeline():
    code = (
        "import runpy, sys; "
        "sys.argv = ['audiosr', '--help']; "
        "runpy.run_module('audiosr', run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--input_audio_file" in completed.stdout
    assert "Loading AudioSR" not in completed.stdout + completed.stderr


def test_legacy_preprocessing_exports_remain_available():
    import audiosr

    for name in (
        "read_audio_file",
        "lowpass_filtering_prepare_inference",
        "wav_feature_extraction",
        "normalize_wav",
        "pad_wav",
    ):
        assert callable(getattr(audiosr, name))
