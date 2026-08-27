"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from audiosr.latent_diffusion.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)
from audiosr.sampling import DEFAULT_DISCRETIZATION, normalize_discretize


_SCHEDULE_CACHE_ATTRIBUTES = (
    "ddim_timesteps",
    "betas",
    "alphas_cumprod",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "ddim_sigmas",
    "ddim_alphas",
    "ddim_alphas_prev",
    "ddim_sqrt_one_minus_alphas",
    "ddim_sigmas_for_original_num_steps",
    "_ddim_step_coefficients",
)


def _tensor_cache_key(tensor):
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        str(tensor.device),
        str(tensor.dtype),
    )


def _concatenate_conditioning(unconditional, conditional):
    """Join classifier-free guidance branches along their batch dimension."""
    if isinstance(conditional, torch.Tensor):
        if not isinstance(unconditional, torch.Tensor):
            raise TypeError("conditioning branches must have matching tensor types")
        return torch.cat((unconditional, conditional), dim=0)
    if isinstance(conditional, dict):
        if not isinstance(unconditional, dict) or unconditional.keys() != conditional.keys():
            raise TypeError("conditioning branches must have matching dictionary keys")
        return {
            key: _concatenate_conditioning(unconditional[key], conditional[key])
            for key in conditional
        }
    if isinstance(conditional, (list, tuple)):
        if not isinstance(unconditional, type(conditional)) or len(unconditional) != len(
            conditional
        ):
            raise TypeError("conditioning branches must have matching sequences")
        joined = [
            _concatenate_conditioning(unconditional_item, conditional_item)
            for unconditional_item, conditional_item in zip(
                unconditional, conditional
            )
        ]
        return type(conditional)(joined)
    raise TypeError(f"unsupported conditioning type: {type(conditional)!r}")


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = device

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                is_mps = self.device == "mps" or self.device == torch.device("mps")
                if is_mps and attr.dtype == torch.float64:
                    attr = attr.to(self.device, dtype=torch.float32)
                else:
                    attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(
        self,
        ddim_num_steps,
        ddim_discretize=DEFAULT_DISCRETIZATION,
        ddim_eta=0.0,
        verbose=True,
    ):
        ddim_discretize = normalize_discretize(ddim_discretize)
        ddim_eta = float(ddim_eta)
        if not np.isfinite(ddim_eta):
            raise ValueError("ddim_eta must be finite")
        if ddim_eta < 0.0:
            raise ValueError("ddim_eta must not be negative")

        schedule_key = (
            int(ddim_num_steps),
            ddim_discretize,
            ddim_eta,
            self.ddpm_num_timesteps,
            str(self.device),
            _tensor_cache_key(self.model.betas),
            _tensor_cache_key(self.model.alphas_cumprod),
            _tensor_cache_key(self.model.alphas_cumprod_prev),
        )
        cached_schedule = getattr(self.model, "_audiosr_ddim_schedule_cache", None)
        if (
            not verbose
            and cached_schedule is not None
            and cached_schedule[0] == schedule_key
        ):
            for name, value in cached_schedule[1].items():
                setattr(self, name, value)
            return

        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"
        to_torch = lambda x: torch.as_tensor(x).clone().detach().to(
            self.device, dtype=torch.float32
        )
        alphas_cumprod_cpu_tensor = alphas_cumprod.detach().cpu()
        alphas_cumprod_cpu = alphas_cumprod_cpu_tensor.numpy()

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer(
            "alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod", to_torch(torch.sqrt(alphas_cumprod_cpu_tensor))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(torch.sqrt(1.0 - alphas_cumprod_cpu_tensor)),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod",
            to_torch(torch.log(1.0 - alphas_cumprod_cpu_tensor)),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            to_torch(torch.sqrt(1.0 / alphas_cumprod_cpu_tensor)),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(torch.sqrt(1.0 / alphas_cumprod_cpu_tensor - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod_cpu,
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", to_torch(ddim_sigmas))
        self.register_buffer("ddim_alphas", to_torch(ddim_alphas))
        self.register_buffer("ddim_alphas_prev", to_torch(ddim_alphas_prev))
        self.register_buffer(
            "ddim_sqrt_one_minus_alphas", to_torch(np.sqrt(1.0 - ddim_alphas))
        )
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )
        self._ddim_step_coefficients = tuple(
            torch.as_tensor(values, dtype=torch.float32, device=self.device)
            for values in (
                self.ddim_alphas,
                self.ddim_alphas_prev,
                self.ddim_sigmas,
                self.ddim_sqrt_one_minus_alphas,
            )
        )
        setattr(
            self.model,
            "_audiosr_ddim_schedule_cache",
            (
                schedule_key,
                {
                    name: getattr(self, name)
                    for name in _SCHEDULE_CACHE_ATTRIBUTES
                },
            ),
        )

    @torch.no_grad()
    def guided_model_output(
        self,
        x,
        c,
        t,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        fuse_cfg=False,
    ):
        """Return the classifier-free guided network output for one timestep.

        Every sampler shares this so the guidance branches, the fused batch
        layout, and the network call count stay identical across samplers.
        """
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            return self.model.apply_model(x, t, c)

        assert isinstance(c, dict)
        assert isinstance(unconditional_conditioning, dict)
        if fuse_cfg:
            try:
                combined_conditioning = _concatenate_conditioning(
                    unconditional_conditioning, c
                )
            except TypeError:
                fuse_cfg = False
        if fuse_cfg:
            model_uncond, model_t = self.model.apply_model(
                torch.cat((x, x), dim=0),
                torch.cat((t, t), dim=0),
                combined_conditioning,
            ).chunk(2)
        else:
            model_t = self.model.apply_model(x, t, c)
            model_uncond = self.model.apply_model(x, t, unconditional_conditioning)

        return model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        ddim_discretize=DEFAULT_DISCRETIZATION,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        return_intermediates=True,
        fuse_cfg=False,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,  # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        dynamic_threshold=None,
        ucg_schedule=None,
        **kwargs,
    ):
        # if conditioning is not None:
        #     if isinstance(conditioning, dict):
        #         ctmp = conditioning[list(conditioning.keys())[0]]
        #         while isinstance(ctmp, list): ctmp = ctmp[0]
        #         cbs = ctmp.shape[0]
        #         if cbs != batch_size:
        #             print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

        #     elif isinstance(conditioning, list):
        #         for ctmp in conditioning:
        #             if ctmp.shape[0] != batch_size:
        #                 print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

        #     else:
        #         if conditioning.shape[0] != batch_size:
        #             print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(
            ddim_num_steps=S,
            ddim_discretize=ddim_discretize,
            ddim_eta=eta,
            verbose=verbose,
        )
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            return_intermediates=return_intermediates,
            fuse_cfg=fuse_cfg,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            dynamic_threshold=dynamic_threshold,
            ucg_schedule=ucg_schedule,
        )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        return_intermediates=True,
        fuse_cfg=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        ucg_schedule=None,
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps
                if ddim_use_original_steps
                else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (
                int(
                    min(timesteps / self.ddim_timesteps.shape[0], 1)
                    * self.ddim_timesteps.shape[0]
                )
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = (
            {"x_inter": [img], "pred_x0": [img]}
            if return_intermediates
            else None
        )
        time_range = (
            list(reversed(range(0, timesteps)))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        timestep_rows = (
            torch.as_tensor(
                np.ascontiguousarray(time_range),
                device=device,
                dtype=torch.long,
            )
            .unsqueeze(1)
            .expand(-1, b)
            .contiguous()
        )

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = timestep_rows[i]

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(
                    x0, ts
                )  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                dynamic_threshold=dynamic_threshold,
                fuse_cfg=fuse_cfg,
            )
            img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if return_intermediates and (
                index % log_every_t == 0 or index == total_steps - 1
            ):
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        fuse_cfg=False,
    ):
        b, *_, device = *x.shape, x.device

        model_output = self.guided_model_output(
            x,
            c,
            t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            fuse_cfg=fuse_cfg,
        )

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", "not implemented"
            e_t = score_corrector.modify_score(
                self.model, e_t, x, t, c, **corrector_kwargs
            )

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )
        # select parameters corresponding to the currently considered timestep
        if use_original_steps:
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(
                (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
            )
        else:
            a_t, a_prev, sigma_t, sqrt_one_minus_at = (
                values[index].reshape(1, 1, 1, 1)
                for values in self._ddim_step_coefficients
            )

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def encode(
        self,
        x0,
        c,
        t_enc,
        use_original_steps=False,
        return_intermediates=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        callback=None,
    ):
        num_reference_steps = (
            self.ddpm_num_timesteps
            if use_original_steps
            else self.ddim_timesteps.shape[0]
        )

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc="Encoding Image"):
            t = torch.full(
                (x0.shape[0],), i, device=self.model.device, dtype=torch.long
            )
            if unconditional_guidance_scale == 1.0:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(
                        torch.cat((x_next, x_next)),
                        torch.cat((t, t)),
                        torch.cat((unconditional_conditioning, c)),
                    ),
                    2,
                )
                noise_pred = e_t_uncond + unconditional_guidance_scale * (
                    noise_pred - e_t_uncond
                )

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = (
                alphas_next[i].sqrt()
                * ((1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt())
                * noise_pred
            )
            x_next = xt_weighted + weighted_noise_pred
            if (
                return_intermediates
                and i % (num_steps // return_intermediates) == 0
                and i < num_steps - 1
            ):
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback:
                callback(i)

        out = {"x_encoded": x_next, "intermediate_steps": inter_steps}
        if return_intermediates:
            out.update({"intermediates": intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    @torch.no_grad()
    def decode(
        self,
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_original_steps=False,
        callback=None,
    ):
        timesteps = (
            np.arange(self.ddpm_num_timesteps)
            if use_original_steps
            else self.ddim_timesteps
        )
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="Decoding image", total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full(
                (x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long
            )
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
            if callback:
                callback(i)
        return x_dec
