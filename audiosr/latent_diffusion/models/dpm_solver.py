"""DPM-Solver++ sampling for the AudioSR latent diffusion model.

The sampler walks exactly the noise levels that :class:`DDIMSampler` builds, so a
given step count drives the same number of network evaluations and reaches the
same terminal signal-to-noise ratio. Only the update rule differs: DPM-Solver++
integrates the probability-flow ODE in ``lambda`` space using the data (``x0``)
prediction, which converges in far fewer steps than the ancestral update that
``ddim_eta=1.0`` selects.

At ``order=1`` the update is algebraically identical to DDIM with ``eta=0``,
which the regression tests rely on to verify the implementation without a real
checkpoint.
"""

import numpy as np
import torch
from tqdm import tqdm

from audiosr.latent_diffusion.models.ddim import DDIMSampler
from audiosr.sampling import DEFAULT_DISCRETIZATION


_SUPPORTED_ORDERS = (1, 2)


class DPMSolverSampler(DDIMSampler):
    """Multistep DPM-Solver++ over the DDIM noise-level chain."""

    def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):
        super().__init__(model, schedule=schedule, device=device, **kwargs)

    def _predict_x0(self, x, t, model_output, alpha_t, sigma_t):
        """Convert a network output to the denoised sample it implies."""
        if self.model.parameterization == "v":
            return self.model.predict_start_from_z_and_v(x, t, model_output)
        if self.model.parameterization == "eps":
            return (x - sigma_t * model_output) / alpha_t
        if self.model.parameterization == "x0":
            return model_output
        raise NotImplementedError(
            f"unsupported parameterization: {self.model.parameterization!r}"
        )

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        img_callback=None,
        eta=0.0,
        ddim_discretize=DEFAULT_DISCRETIZATION,
        mask=None,
        x0=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        return_intermediates=True,
        fuse_cfg=False,
        order=2,
        lower_order_final=True,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        ucg_schedule=None,
        **kwargs,
    ):
        order = int(order)
        if order not in _SUPPORTED_ORDERS:
            raise ValueError(f"order must be one of {_SUPPORTED_ORDERS}")

        # DPM-Solver++ integrates the deterministic probability-flow ODE, so the
        # stochastic term the eta schedule controls has no role here. Building
        # the schedule at eta=0 keeps the alpha chain identical to DDIM's.
        self.make_schedule(
            ddim_num_steps=S,
            ddim_discretize=ddim_discretize,
            ddim_eta=0.0,
            verbose=verbose,
        )

        channels, height, width = shape
        size = (batch_size, channels, height, width)

        return self.dpm_solver_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            mask=mask,
            x0=x0,
            x_T=x_T,
            log_every_t=log_every_t,
            return_intermediates=return_intermediates,
            fuse_cfg=fuse_cfg,
            order=order,
            lower_order_final=lower_order_final,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            ucg_schedule=ucg_schedule,
        )

    @torch.no_grad()
    def dpm_solver_sampling(
        self,
        cond,
        shape,
        x_T=None,
        callback=None,
        img_callback=None,
        mask=None,
        x0=None,
        log_every_t=100,
        return_intermediates=True,
        fuse_cfg=False,
        order=2,
        lower_order_final=True,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        ucg_schedule=None,
    ):
        device = self.model.betas.device
        batch_size = shape[0]
        img = torch.randn(shape, device=device) if x_T is None else x_T

        timesteps = self.ddim_timesteps
        total_steps = timesteps.shape[0]
        time_range = np.flip(timesteps)

        intermediates = (
            {"x_inter": [img], "pred_x0": [img]} if return_intermediates else None
        )
        print(f"Running DPM-Solver++ Sampling with {total_steps} timesteps")

        alphas, alphas_prev = self._ddim_step_coefficients[0], self._ddim_step_coefficients[1]
        timestep_rows = (
            torch.as_tensor(
                np.ascontiguousarray(time_range), device=device, dtype=torch.long
            )
            .unsqueeze(1)
            .expand(-1, batch_size)
            .contiguous()
        )

        previous_x0 = None
        previous_h = None
        iterator = tqdm(time_range, desc="DPM-Solver++ Sampler", total=total_steps)

        for i, _step in enumerate(iterator):
            index = total_steps - i - 1
            ts = timestep_rows[i]

            if mask is not None:
                assert x0 is not None
                img = self.model.q_sample(x0, ts) * mask + (1.0 - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == total_steps
                unconditional_guidance_scale = ucg_schedule[i]

            # Noise level of the current sample and of the one this step targets.
            alpha_t = alphas[index].sqrt().reshape(1, 1, 1, 1)
            sigma_t = (1.0 - alphas[index]).sqrt().reshape(1, 1, 1, 1)
            alpha_s = alphas_prev[index].sqrt().reshape(1, 1, 1, 1)
            sigma_s = (1.0 - alphas_prev[index]).sqrt().reshape(1, 1, 1, 1)

            lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
            lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
            h = lambda_s - lambda_t

            model_output = self.guided_model_output(
                img,
                cond,
                ts,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                fuse_cfg=fuse_cfg,
            )
            pred_x0 = self._predict_x0(img, ts, model_output, alpha_t, sigma_t)

            # The multistep correction needs a usable previous estimate, and the
            # final step drops to first order because the terminal step size is
            # where the second-order extrapolation is least stable.
            use_second_order = (
                order == 2
                and previous_x0 is not None
                and previous_h is not None
                and not (lower_order_final and index == 0)
            )
            if use_second_order:
                ratio = previous_h / h
                correction = 1.0 / (2.0 * ratio)
                derivative = (1.0 + correction) * pred_x0 - correction * previous_x0
            else:
                derivative = pred_x0

            img = (sigma_s / sigma_t) * img - alpha_s * (
                torch.expm1(-h)
            ) * derivative

            previous_x0 = pred_x0
            previous_h = h

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
