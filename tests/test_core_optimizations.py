from types import SimpleNamespace

import librosa
import numpy as np
import pytest
import torch

from audiosr import utils
from audiosr.latent_diffusion.models import ddim as ddim_module
from audiosr.latent_diffusion.models import ddpm
from audiosr.latent_diffusion.models import dpm_solver as dpm_solver_module
from audiosr.latent_diffusion.modules import attention
from audiosr.latent_diffusion.modules.diffusionmodules import util as diffusion_util


class _SamplerModel:
    def __init__(self):
        self.num_timesteps = 1000
        self.betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.ones(1), self.alphas_cumprod[:-1])
        )
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.ddim_sigmas_for_original_num_steps = torch.zeros_like(self.betas)
        self.device = torch.device("cpu")
        self.parameterization = "eps"
        self.apply_model_calls = 0

    def apply_model(self, x, timesteps, conditioning):
        self.apply_model_calls += 1
        condition = conditioning["concat"].expand_as(x)
        time_value = timesteps.float().reshape(-1, 1, 1, 1)
        return x * 0.125 + condition * 0.01 + time_value * 0.0001


class _VelocitySamplerModel(_SamplerModel):
    """A stub that predicts velocity, matching the shipped checkpoints."""

    def __init__(self):
        super().__init__()
        self.parameterization = "v"
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            diffusion_util.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)
            * x_t
            - diffusion_util.extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )
            * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            diffusion_util.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)
            * v
            + diffusion_util.extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )
            * x_t
        )


@pytest.mark.parametrize("mask_kind", ["none", "partial", "all_false"])
def test_cuda_sdpa_attention_matches_legacy_attention(monkeypatch, mask_kind):
    torch.manual_seed(42)
    module = attention.CrossAttention(
        query_dim=32,
        context_dim=32,
        heads=4,
        dim_head=8,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 12, 32)
    context = torch.randn(2, 9, 32)

    mask = None
    if mask_kind == "partial":
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=torch.bool,
        )
    elif mask_kind == "all_false":
        mask = torch.zeros(2, 9, dtype=torch.bool)

    monkeypatch.setattr(attention, "_should_use_sdpa", lambda _tensor: False)
    expected = module(x, context=context, mask=mask)

    monkeypatch.setattr(attention, "_should_use_sdpa", lambda _tensor: True)
    actual = module(x, context=context, mask=mask)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_sdpa_selection_is_limited_to_nvidia_cuda(monkeypatch):
    cpu_tensor = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    cuda_tensor = SimpleNamespace(device=SimpleNamespace(type="cuda"))

    monkeypatch.setattr(attention.torch.version, "hip", None, raising=False)
    assert not attention._should_use_sdpa(cpu_tensor)
    assert attention._should_use_sdpa(cuda_tensor)

    monkeypatch.setattr(attention.torch.version, "hip", "7.14", raising=False)
    assert not attention._should_use_sdpa(cuda_tensor)


def test_mel_spectrogram_reuses_device_cache(monkeypatch):
    mel_calls = 0
    original_mel = utils.librosa_mel_fn

    def counted_mel(*args, **kwargs):
        nonlocal mel_calls
        mel_calls += 1
        return original_mel(*args, **kwargs)

    monkeypatch.setattr(utils, "mel_basis", {})
    monkeypatch.setattr(utils, "hann_window", {})
    monkeypatch.setattr(utils, "librosa_mel_fn", counted_mel)
    waveform = torch.linspace(-0.5, 0.5, 4096).unsqueeze(0)

    first_mel, first_stft = utils.mel_spectrogram_train(waveform)
    second_mel, second_stft = utils.mel_spectrogram_train(waveform)

    assert mel_calls == 1
    torch.testing.assert_close(second_mel, first_mel, rtol=0, atol=0)
    torch.testing.assert_close(second_stft, first_stft, rtol=0, atol=0)


def _librosa_band_replacement(source, generated):
    """The per-item librosa implementation the batched transform replaces."""
    stft_gt = librosa.stft(source)
    stft_out = librosa.stft(generated)
    energy = np.cumsum(np.sum(np.abs(stft_gt), axis=-1))
    threshold = energy[-1] * 0.985
    cutoff = 0
    for i in range(1, energy.shape[0]):
        if energy[-i] < threshold:
            cutoff = energy.shape[0] - i
            break
    energy_ratio = np.mean(
        np.sum(np.abs(stft_gt[cutoff])) / np.sum(np.abs(stft_out[cutoff, ...]))
    )
    energy_ratio = min(max(energy_ratio, 0.8), 1.2)
    stft_out[:cutoff, ...] = stft_gt[:cutoff, ...] / energy_ratio
    return librosa.istft(stft_out, length=generated.shape[0]), cutoff


def test_batched_band_replacement_matches_the_librosa_reference():
    """The device-resident transform must reproduce the per-item CPU result."""
    rng = np.random.default_rng(42)
    sources, generateds, expected = [], [], []
    for item in range(3):
        source = rng.normal(0, 0.2, 8192).astype(np.float32)
        # Give the source a genuine band limit so the crossover is meaningful.
        source_stft = librosa.stft(source)
        source_stft[300 + 100 * item :, :] = 0
        source = librosa.istft(source_stft, length=8192).astype(np.float32)
        generated = rng.normal(0, 0.2, 8192).astype(np.float32)
        reference, cutoff = _librosa_band_replacement(source, generated)
        assert cutoff > 0
        sources.append(source)
        generateds.append(generated)
        expected.append(reference)

    model = object.__new__(ddpm.LatentDiffusion)
    actual = model.postprocessing(
        torch.from_numpy(np.stack(generateds)).unsqueeze(1),
        torch.from_numpy(np.stack(sources)).unsqueeze(1),
    )

    assert actual.shape == (3, 1, 8192)
    torch.testing.assert_close(
        actual[:, 0],
        torch.from_numpy(np.stack(expected)),
        rtol=1e-4,
        atol=1e-5,
    )


def test_band_replacement_rejects_mismatched_waveforms():
    model = object.__new__(ddpm.LatentDiffusion)

    with pytest.raises(ValueError, match=r"generated waveforms must have shape"):
        model.postprocessing(torch.zeros(2, 4096), torch.zeros(2, 1, 4096))
    with pytest.raises(ValueError, match=r"source waveforms must have shape"):
        model.postprocessing(torch.zeros(2, 1, 4096), torch.zeros(2, 4096))
    with pytest.raises(ValueError, match="same batch size"):
        model.postprocessing(torch.zeros(2, 1, 4096), torch.zeros(3, 1, 4096))


@pytest.mark.parametrize("overshoot", [16, -16])
def test_band_replacement_aligns_a_source_of_a_different_length(overshoot):
    """The vocoder overshoots the conditioning length by part of one hop."""
    generator = torch.Generator().manual_seed(47)
    length = 8192
    generated = torch.randn(2, 1, length, generator=generator) * 0.2
    source = torch.randn(2, 1, length - overshoot, generator=generator) * 0.2

    model = object.__new__(ddpm.LatentDiffusion)
    renewed = model.postprocessing(generated.clone(), source)

    assert renewed.shape == generated.shape
    assert torch.isfinite(renewed).all()


def test_band_replacement_zero_padding_matches_an_already_aligned_source():
    """Padding to the generated length must not change the replaced band.

    The centred transform already treats everything past the end of the signal
    as zero, so appending zeros has to be a no-op.
    """
    generator = torch.Generator().manual_seed(53)
    length = 8192
    generated = torch.randn(1, 1, length, generator=generator) * 0.2
    source = torch.randn(1, 1, length, generator=generator) * 0.2
    short_source = source[..., : length - 16]

    model = object.__new__(ddpm.LatentDiffusion)
    padded = torch.zeros_like(source)
    padded[..., : length - 16] = short_source

    from_short = model.postprocessing(generated.clone(), short_source)
    from_padded = model.postprocessing(generated.clone(), padded)

    torch.testing.assert_close(from_short, from_padded, rtol=0, atol=0)


def test_mel_replace_matches_the_per_item_loop():
    """The vectorised crossover must reproduce the scan it replaces."""
    torch.manual_seed(7)
    lowpass_mel = torch.randn(3, 16, 64) * 0.5
    # Decay the upper bins so each item lands on a different crossover.
    for item in range(3):
        lowpass_mel[item, :, 20 + 10 * item :] -= 12.0
    samples = torch.randn(3, 1, 16, 64)

    expected = samples.clone()
    for item in range(3):
        magnitude = torch.exp(lowpass_mel[item])
        energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
        threshold = energy[-1] * 0.985
        cutoff = 0
        for i in range(1, energy.shape[0]):
            if energy[-i] < threshold:
                cutoff = energy.shape[0] - i
                break
        assert cutoff > 0
        expected[item][..., :cutoff] = lowpass_mel[item][..., :cutoff]

    model = object.__new__(ddpm.LatentDiffusion)
    actual = model.mel_replace_ops(samples.clone(), lowpass_mel)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ddim_schedule_is_reused_until_its_inputs_change(monkeypatch):
    model = _SamplerModel()
    timestep_calls = 0
    parameter_calls = 0
    original_timesteps = ddim_module.make_ddim_timesteps
    original_parameters = ddim_module.make_ddim_sampling_parameters

    def counted_timesteps(*args, **kwargs):
        nonlocal timestep_calls
        timestep_calls += 1
        return original_timesteps(*args, **kwargs)

    def counted_parameters(*args, **kwargs):
        nonlocal parameter_calls
        parameter_calls += 1
        return original_parameters(*args, **kwargs)

    monkeypatch.setattr(ddim_module, "make_ddim_timesteps", counted_timesteps)
    monkeypatch.setattr(
        ddim_module, "make_ddim_sampling_parameters", counted_parameters
    )

    first = ddim_module.DDIMSampler(model, device=model.device)
    first.make_schedule(50, ddim_eta=1.0, verbose=False)
    second = ddim_module.DDIMSampler(model, device=model.device)
    second.make_schedule(50, ddim_eta=1.0, verbose=False)

    assert timestep_calls == 1
    assert parameter_calls == 1
    assert second.ddim_alphas.data_ptr() == first.ddim_alphas.data_ptr()

    third = ddim_module.DDIMSampler(model, device=model.device)
    third.make_schedule(17, ddim_eta=1.0, verbose=False)
    assert timestep_calls == 2
    assert parameter_calls == 2

    model.alphas_cumprod = model.alphas_cumprod.clone()
    fourth = ddim_module.DDIMSampler(model, device=model.device)
    fourth.make_schedule(17, ddim_eta=1.0, verbose=False)
    assert timestep_calls == 3
    assert parameter_calls == 3


def test_ddim_fuses_classifier_free_guidance_without_changing_formula(monkeypatch):
    conditional = {"concat": torch.full((1, 1, 1, 1), 2.0)}
    unconditional = {"concat": torch.full((1, 1, 1, 1), -1.0)}
    initial = torch.linspace(-1.0, 1.0, 4).reshape(1, 1, 2, 2)

    def run(fuse_cfg):
        model = _SamplerModel()
        sampler = ddim_module.DDIMSampler(model, device=model.device)
        sampler.make_schedule(3, ddim_eta=1.0, verbose=False)
        torch.manual_seed(42)
        output, intermediates = sampler.ddim_sampling(
            conditional,
            initial.shape,
            x_T=initial.clone(),
            unconditional_guidance_scale=3.5,
            unconditional_conditioning=unconditional,
            return_intermediates=False,
            fuse_cfg=fuse_cfg,
        )
        return output, intermediates, model.apply_model_calls

    expected, expected_intermediates, legacy_calls = run(False)
    actual, actual_intermediates, fused_calls = run(True)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert expected_intermediates is None
    assert actual_intermediates is None
    assert legacy_calls == 6
    assert fused_calls == 3


def test_ddim_default_path_does_not_allocate_full_coefficient_tensors(monkeypatch):
    model = _SamplerModel()
    sampler = ddim_module.DDIMSampler(model, device=model.device)
    sampler.make_schedule(3, ddim_eta=1.0, verbose=False)
    conditional = {"concat": torch.ones(1, 1, 1, 1)}

    def unexpected_full(*_args, **_kwargs):
        raise AssertionError("DDIM coefficients must use cached broadcast views")

    monkeypatch.setattr(ddim_module.torch, "full", unexpected_full)
    output, _ = sampler.ddim_sampling(
        conditional,
        (1, 1, 2, 2),
        x_T=torch.zeros(1, 1, 2, 2),
        return_intermediates=False,
    )

    assert torch.isfinite(output).all()


def test_conditioning_concatenation_preserves_nested_branch_order():
    unconditional = {
        "concat": torch.tensor([[[[-1.0]]]]),
        "crossattn": [torch.tensor([[1.0]]), {"mask": torch.tensor([[0]])}],
    }
    conditional = {
        "concat": torch.tensor([[[[2.0]]]]),
        "crossattn": [torch.tensor([[3.0]]), {"mask": torch.tensor([[1]])}],
    }

    joined = ddim_module._concatenate_conditioning(unconditional, conditional)

    assert joined["concat"].flatten().tolist() == [-1.0, 2.0]
    assert joined["crossattn"][0].flatten().tolist() == [1.0, 3.0]
    assert joined["crossattn"][1]["mask"].flatten().tolist() == [0, 1]


def test_timestep_embedding_accepts_precomputed_frequencies(monkeypatch):
    timesteps = torch.tensor([0, 1, 42], dtype=torch.long)
    expected = diffusion_util.timestep_embedding(timesteps, 8)
    frequencies = torch.exp(
        -np.log(10000)
        * torch.arange(0, 4, dtype=torch.float32)
        / 4
    )

    def unexpected_arange(*_args, **_kwargs):
        raise AssertionError("precomputed frequencies must avoid torch.arange")

    monkeypatch.setattr(diffusion_util.torch, "arange", unexpected_arange)
    actual = diffusion_util.timestep_embedding(
        timesteps,
        8,
        frequencies=frequencies,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


_SAMPLER_STUBS = {"eps": _SamplerModel, "v": _VelocitySamplerModel}


@pytest.mark.parametrize("parameterization", sorted(_SAMPLER_STUBS))
def test_first_order_dpm_solver_matches_deterministic_ddim(parameterization):
    """DPM-Solver++ at order 1 is algebraically identical to DDIM at eta=0.

    Verifying that identity pins the schedule handling, the x0 conversion, and
    the update rule without needing the real checkpoint.
    """
    conditional = {"concat": torch.full((1, 1, 1, 1), 2.0)}
    unconditional = {"concat": torch.full((1, 1, 1, 1), -1.0)}
    initial = torch.linspace(-1.0, 1.0, 8).reshape(1, 1, 2, 4)
    stub = _SAMPLER_STUBS[parameterization]

    ddim_model = stub()
    ddim_sampler = ddim_module.DDIMSampler(ddim_model, device=ddim_model.device)
    expected, _ = ddim_sampler.sample(
        20,
        1,
        (1, 2, 4),
        conditional,
        eta=0.0,
        verbose=False,
        x_T=initial.clone(),
        return_intermediates=False,
        unconditional_guidance_scale=3.5,
        unconditional_conditioning=unconditional,
    )

    dpm_model = stub()
    dpm_sampler = dpm_solver_module.DPMSolverSampler(dpm_model, device=dpm_model.device)
    actual, _ = dpm_sampler.sample(
        20,
        1,
        (1, 2, 4),
        conditional,
        verbose=False,
        x_T=initial.clone(),
        return_intermediates=False,
        order=1,
        unconditional_guidance_scale=3.5,
        unconditional_conditioning=unconditional,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert dpm_model.apply_model_calls == ddim_model.apply_model_calls


def test_second_order_dpm_solver_keeps_one_network_call_per_step():
    """The multistep correction reuses the previous estimate instead of resampling."""
    conditional = {"concat": torch.full((1, 1, 1, 1), 2.0)}
    unconditional = {"concat": torch.full((1, 1, 1, 1), -1.0)}
    initial = torch.zeros(1, 1, 2, 4)

    model = _VelocitySamplerModel()
    sampler = dpm_solver_module.DPMSolverSampler(model, device=model.device)
    second_order, _ = sampler.sample(
        12,
        1,
        (1, 2, 4),
        conditional,
        verbose=False,
        x_T=initial.clone(),
        return_intermediates=False,
        order=2,
        fuse_cfg=True,
        unconditional_guidance_scale=3.5,
        unconditional_conditioning=unconditional,
    )

    assert model.apply_model_calls == 12
    assert torch.isfinite(second_order).all()

    first_order_model = _VelocitySamplerModel()
    first_order_sampler = dpm_solver_module.DPMSolverSampler(
        first_order_model, device=first_order_model.device
    )
    first_order, _ = first_order_sampler.sample(
        12,
        1,
        (1, 2, 4),
        conditional,
        verbose=False,
        x_T=initial.clone(),
        return_intermediates=False,
        order=1,
        fuse_cfg=True,
        unconditional_guidance_scale=3.5,
        unconditional_conditioning=unconditional,
    )

    # The correction term has to actually change the trajectory, otherwise the
    # second-order branch would be silently inert.
    assert not torch.allclose(second_order, first_order, rtol=1e-4, atol=1e-5)


def test_dpm_solver_rejects_unsupported_order():
    model = _VelocitySamplerModel()
    sampler = dpm_solver_module.DPMSolverSampler(model, device=model.device)

    with pytest.raises(ValueError, match="order must be one of"):
        sampler.sample(4, 1, (1, 2, 4), {"concat": torch.zeros(1, 1, 1, 1)}, order=3)


def test_dpm_solver_reports_an_unsupported_parameterization():
    model = _VelocitySamplerModel()
    model.parameterization = "elbo"
    sampler = dpm_solver_module.DPMSolverSampler(model, device=model.device)

    with pytest.raises(NotImplementedError, match="unsupported parameterization"):
        sampler.sample(
            4,
            1,
            (1, 2, 4),
            {"concat": torch.zeros(1, 1, 1, 1)},
            verbose=False,
            x_T=torch.zeros(1, 1, 2, 4),
            return_intermediates=False,
        )


def _minimal_latent_diffusion():
    model = object.__new__(ddpm.LatentDiffusion)
    model.device = torch.device("cpu")
    model.cond_stage_model_metadata = {}
    model.conditional_dry_run_finished = True
    return model


def _feature_batch():
    return {
        "waveform": torch.zeros(1, 1, 480),
        "stft": torch.zeros(1, 1, 8),
        "log_mel_spec": torch.zeros(1, 4, 256),
    }


def test_get_input_can_skip_the_discarded_first_stage_encode():
    """Generation never samples the target latent, so its encoder pass is optional."""
    model = _minimal_latent_diffusion()

    def unexpected_encode(_x):
        raise AssertionError("the first-stage encoder must not run")

    model.encode_first_stage = unexpected_encode

    latent, conditioning = model.get_input(
        _feature_batch(),
        "fbank",
        return_first_stage_encode=False,
        unconditional_prob_cfg=0.0,
    )

    assert latent is None
    assert conditioning == {}


@pytest.mark.parametrize(
    "derived_output", ["return_decoding_output", "return_encoder_output"]
)
def test_get_input_rejects_derived_outputs_without_the_encode(derived_output):
    model = _minimal_latent_diffusion()

    with pytest.raises(ValueError, match="require return_first_stage_encode"):
        model.get_input(
            _feature_batch(),
            "fbank",
            return_first_stage_encode=False,
            unconditional_prob_cfg=0.0,
            **{derived_output: True},
        )


def test_conditioning_latent_reference_sizes_the_diffusion_sample():
    model = object.__new__(ddpm.LatentDiffusion)
    latent = torch.zeros(3, 16, 64, 32)

    reference = model._conditioning_latent_reference(
        {"film_extra": torch.zeros(3, 8), "concat_lowpass_cond": latent}
    )

    assert reference is latent

    with pytest.raises(RuntimeError, match="concatenated conditioning latent"):
        model._conditioning_latent_reference({"film_extra": torch.zeros(3, 8)})


def test_mel_replace_aligns_a_host_source_to_the_sample():
    """The conditioning batch arrives on the host and in its own dtype."""
    lowpass_mel = torch.linspace(-6.0, 1.0, 32, dtype=torch.float64).reshape(1, 1, 32)
    lowpass_mel = lowpass_mel.expand(2, 8, 32).contiguous()
    samples = torch.zeros(2, 1, 8, 32, dtype=torch.float32)

    model = object.__new__(ddpm.LatentDiffusion)
    replaced = model.mel_replace_ops(samples.clone(), lowpass_mel)

    assert replaced.dtype == torch.float32
    assert replaced.device == samples.device
    # Something below the crossover must actually have been substituted.
    assert not torch.equal(replaced, samples)


def test_band_replacement_aligns_a_host_source_to_the_generated_waveform():
    generator = torch.Generator().manual_seed(31)
    generated = torch.randn(2, 1, 8192, generator=generator) * 0.2
    source = (torch.randn(2, 1, 8192, generator=generator) * 0.2).double()

    model = object.__new__(ddpm.LatentDiffusion)
    renewed = model.postprocessing(generated.clone(), source)

    assert renewed.dtype == generated.dtype
    assert renewed.shape == generated.shape
    assert torch.isfinite(renewed).all()


_TRAINING_TIMESTEPS = 1000


@pytest.mark.parametrize("steps", [6, 9, 12, 16, 20, 21, 25, 26, 30, 50, 100, 250])
def test_trailing_discretisation_always_reaches_the_noisiest_timestep(steps):
    """Sampling starts from pure noise, so the schedule has to start there too.

    A schedule that stops short leaves a signal component at its first timestep
    that the sampler never accounts for, and the error survives when there are
    few steps to absorb it.
    """
    timesteps = diffusion_util.make_ddim_timesteps(
        "trailing", steps, _TRAINING_TIMESTEPS, verbose=False
    )

    assert timesteps.size == steps
    assert timesteps[-1] == _TRAINING_TIMESTEPS - 1
    assert timesteps[0] >= 0
    assert np.all(np.diff(timesteps) > 0)


@pytest.mark.parametrize("steps", [20, 25, 50, 100, 125, 200])
def test_uniform_discretisation_stops_short_for_exact_divisors(steps):
    """Pin the established schedule's shortfall so it cannot change silently.

    ``uniform`` is kept as the default for compatibility, and for step counts
    that divide the training schedule it never reaches the final timestep. That
    is the behaviour ``trailing`` exists to avoid.
    """
    assert _TRAINING_TIMESTEPS % steps == 0

    timesteps = diffusion_util.make_ddim_timesteps(
        "uniform", steps, _TRAINING_TIMESTEPS, verbose=False
    )

    assert timesteps.size == steps
    assert timesteps[-1] < _TRAINING_TIMESTEPS - 1
    assert timesteps[-1] == _TRAINING_TIMESTEPS - (_TRAINING_TIMESTEPS // steps) + 1


@pytest.mark.parametrize("steps", [9, 12, 16, 21, 26, 30])
def test_uniform_discretisation_reaches_the_end_for_non_divisors(steps):
    assert _TRAINING_TIMESTEPS % steps != 0

    timesteps = diffusion_util.make_ddim_timesteps(
        "uniform", steps, _TRAINING_TIMESTEPS, verbose=False
    )

    assert timesteps.size == steps
    assert timesteps[-1] == _TRAINING_TIMESTEPS - 1


def test_unknown_discretisation_is_rejected_at_the_sampler_boundary():
    model = _VelocitySamplerModel()
    sampler = ddim_module.DDIMSampler(model, device=model.device)

    with pytest.raises(ValueError, match="discretize must be one of"):
        sampler.make_schedule(10, ddim_discretize="leading", verbose=False)


def test_negative_eta_is_rejected_at_the_sampler_boundary():
    model = _VelocitySamplerModel()
    sampler = ddim_module.DDIMSampler(model, device=model.device)

    with pytest.raises(ValueError, match="ddim_eta must not be negative"):
        sampler.make_schedule(10, ddim_eta=-0.5, verbose=False)


def test_the_schedule_cache_separates_discretisations():
    """Two spacings at the same step count must not share cached coefficients."""
    model = _VelocitySamplerModel()

    uniform = ddim_module.DDIMSampler(model, device=model.device)
    uniform.make_schedule(20, ddim_discretize="uniform", verbose=False)
    uniform_timesteps = uniform.ddim_timesteps.copy()

    trailing = ddim_module.DDIMSampler(model, device=model.device)
    trailing.make_schedule(20, ddim_discretize="trailing", verbose=False)

    assert trailing.ddim_timesteps[-1] == _TRAINING_TIMESTEPS - 1
    assert uniform_timesteps[-1] < _TRAINING_TIMESTEPS - 1
    assert not np.array_equal(uniform_timesteps, trailing.ddim_timesteps)
