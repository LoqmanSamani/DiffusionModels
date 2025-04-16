"""Reverse diffusion process for Denoising Diffusion Implicit Models (DDIM).

This module implements the reverse diffusion process for DDIM, as described in Song et al.
(2021, "Denoising Diffusion Implicit Models"). The reverse process iteratively denoises a
noisy input to reconstruct the original data distribution using a subset of time steps.
"""

import torch
import torch.nn as nn



class ReverseDDIM(nn.Module):
    """Reverse diffusion process of DDIM.

    Implements the reverse diffusion process for Denoising Diffusion Implicit Models
    (DDIM), which denoises a noisy input `xt` using a predicted noise component and a
    subsampled time step schedule, as defined in Song et al. (2021).

    Parameters
    ----------
    hyper_params : object
        Hyperparameter object containing the noise schedule parameters. Expected to have
        attributes:
        - `tau_num_steps`: Number of subsampled time steps (int).
        - `eta`: Noise scaling factor for the reverse process (float).
        - `get_tau_schedule`: Method to compute the subsampled noise schedule (callable),
          returning a tuple of (betas, alphas, alpha_bars, sqrt_alpha_cumprod,
          sqrt_one_minus_alpha_cumprod).

    Attributes
    ----------
    hyper_params : object
        Stores the provided hyperparameter object.
    """
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params

    def forward(self, xt, predicted_noise, time_steps, prev_time_steps):
        """Applies the reverse diffusion process to the noisy input.

        Denoises the input `xt` at time step `t` to produce the previous step `xt_prev`
        at `prev_time_steps` using the predicted noise and the DDIM reverse process.
        Optionally includes stochastic noise scaled by `eta`.

        Parameters
        ----------
        xt : torch.Tensor
            Noisy input tensor at time step `t`, shape (batch_size, channels, height, width).
        predicted_noise : torch.Tensor
            Predicted noise tensor, same shape as `xt`, typically output by a neural network.
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, hyper_params.tau_num_steps - 1].
        prev_time_steps : torch.Tensor
            Tensor of previous time step indices (long), shape (batch_size,), where each
            value is in the range [0, hyper_params.tau_num_steps - 1].

        Returns
        -------
        tuple
            A tuple containing:
            - xt_prev: Denoised tensor at `prev_time_steps`, same shape as `xt`.
            - x0: Estimated original data (t=0), same shape as `xt`.

        Raises
        ------
        ValueError
            If any value in `time_steps` or `prev_time_steps` is outside the valid range
            [0, hyper_params.tau_num_steps - 1].
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.hyper_params.tau_num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.hyper_params.tau_num_steps - 1}")
        if not torch.all((prev_time_steps >= 0) & (prev_time_steps < self.hyper_params.tau_num_steps)):
            raise ValueError(f"prev_time_steps must be between 0 and {self.hyper_params.tau_num_steps - 1}")

        _, _, _, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod = self.hyper_params.get_tau_schedule()
        tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
        tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)

        eta = self.hyper_params.eta
        x0 = (xt - tau_sqrt_one_minus_alpha_cumprod_t * predicted_noise) / tau_sqrt_alpha_cumprod_t
        noise_coeff = eta * ((tau_sqrt_one_minus_alpha_cumprod_t / prev_tau_sqrt_alpha_cumprod_t) *
                             prev_tau_sqrt_one_minus_alpha_cumprod_t / torch.clamp(tau_sqrt_one_minus_alpha_cumprod_t, min=1e-8))
        direction_coeff = torch.clamp(prev_tau_sqrt_one_minus_alpha_cumprod_t ** 2 - noise_coeff ** 2, min=1e-8).sqrt()
        xt_prev = prev_tau_sqrt_alpha_cumprod_t * x0 + noise_coeff * torch.randn_like(xt) + direction_coeff * predicted_noise

        return xt_prev, x0