import os

import numpy as np
import pytest
import soundfile as sf
import torch

from audiosr import pipeline

RUN_ROCM_HARDWARE = os.environ.get("AUDIOSR_RUN_ROCM_HARDWARE") == "1"
RUN_FULL_INFERENCE = os.environ.get("AUDIOSR_RUN_ROCM_FULL_INFERENCE") == "1"
ROCM_DDIM_STEPS = int(os.environ.get("AUDIOSR_ROCM_DDIM_STEPS", "1"))

pytestmark = pytest.mark.skipif(
    not RUN_ROCM_HARDWARE,
    reason="set AUDIOSR_RUN_ROCM_HARDWARE=1 on a ROCm machine",
)


def _rocm_device():
    assert torch.version.hip, "the installed PyTorch build does not include ROCm/HIP"
    assert torch.cuda.is_available(), "ROCm is installed but no GPU is available"
    return torch.device("cuda:0")


def test_rocm_executes_representative_audio_kernels():
    device = _rocm_device()

    left = torch.randn((512, 512), device=device, requires_grad=True)
    right = torch.randn((512, 512), device=device)
    loss = (left @ right).square().mean()
    loss.backward()

    convolution = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1).to(device)
    feature_map = convolution(torch.randn((1, 4, 64, 64), device=device))
    spectrum = torch.stft(
        torch.randn(4096, device=device),
        n_fft=256,
        hop_length=64,
        window=torch.hann_window(256, device=device),
        return_complex=True,
    )
    torch.cuda.synchronize()

    assert left.grad is not None
    assert torch.isfinite(left.grad).all()
    assert torch.isfinite(feature_map).all()
    assert torch.isfinite(spectrum).all()
    assert torch.cuda.get_device_name(0)


def test_rocm_uses_audiosr_auto_device_path(monkeypatch, tmp_path):
    _rocm_device()

    class TinyModel(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1))

        def load_state_dict(self, _state_dict, strict=False):
            return None

    monkeypatch.setattr(pipeline, "LatentDiffusion", TinyModel)
    monkeypatch.setattr(
        pipeline,
        "default_audioldm_config",
        lambda _model_name: {"model": {"params": {}}},
    )
    monkeypatch.setattr(
        pipeline.torch,
        "load",
        lambda *_args, **_kwargs: {"state_dict": {}},
    )

    model = pipeline.build_model(
        ckpt_path=tmp_path / "tiny.bin",
        device="auto",
    )

    assert model.anchor.device.type == "cuda"
    assert torch.version.hip


@pytest.mark.skipif(
    not RUN_FULL_INFERENCE,
    reason="set AUDIOSR_RUN_ROCM_FULL_INFERENCE=1 for model inference",
)
def test_rocm_runs_basic_model_inference(tmp_path):
    _rocm_device()
    assert ROCM_DDIM_STEPS > 0
    sample_rate = 16000
    sample_count = sample_rate // 4
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    source = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    input_path = tmp_path / "input.wav"
    sf.write(input_path, source, sample_rate)

    model = pipeline.build_model(device="auto", model_name="basic")
    result = pipeline.super_resolution(
        model,
        input_path,
        seed=42,
        ddim_steps=ROCM_DDIM_STEPS,
        guidance_scale=3.5,
    )

    assert result.shape == (1, 1, sample_count * 3)
    assert np.isfinite(result).all()


def test_rocm_band_replacement_accepts_a_host_conditioning_batch():
    """The conditioning batch stays on the host while generation runs on the GPU.

    Both replacement stages have to bridge that gap themselves, so this pins the
    combination a CPU-only test run cannot reach.
    """
    from audiosr.latent_diffusion.models import ddpm

    device = _rocm_device()
    model = object.__new__(ddpm.LatentDiffusion)

    lowpass_mel = (
        torch.linspace(-6.0, 1.0, 256).reshape(1, 1, 256).expand(2, 64, 256).contiguous()
    )
    samples = torch.zeros(2, 1, 64, 256, device=device)
    replaced = model.mel_replace_ops(samples.clone(), lowpass_mel)

    assert replaced.device.type == "cuda"
    assert torch.isfinite(replaced).all()
    assert not torch.equal(replaced, samples)

    generated = torch.randn(2, 1, 8192, device=device) * 0.2
    source = torch.randn(2, 1, 8192) * 0.2
    renewed = model.postprocessing(generated.clone(), source)

    torch.cuda.synchronize()
    assert renewed.device.type == "cuda"
    assert renewed.shape == generated.shape
    assert torch.isfinite(renewed).all()


@pytest.mark.skipif(
    not RUN_FULL_INFERENCE,
    reason="set AUDIOSR_RUN_ROCM_FULL_INFERENCE=1 to load the real checkpoint",
)
def test_the_released_checkpoint_shares_one_vae_between_both_stages():
    """The released weights repeat the VAE, so only one copy should be built."""
    model = pipeline.build_model(device="cpu")

    extractor = model.cond_stage_models[0]
    assert extractor.vae is model.first_stage_model
    assert model.share_first_stage_cond == ("concat_lowpass_cond",)
