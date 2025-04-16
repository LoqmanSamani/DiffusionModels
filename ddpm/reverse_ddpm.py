"""Reverse diffusion process for Denoising Diffusion Probabilistic Models (DDPM).

This module implements the reverse diffusion process as described in the DDPM paper
(Ho et al., 2020, "Denoising Diffusion Probabilistic Models"). The reverse process
gradually denoises a noisy input to reconstruct the original data distribution.
"""


import torch
import torch.nn as nn



class ReverseDDPM(nn.Module):
    """Reverse diffusion process of DDPM.

    Implements the reverse diffusion process for DDPM, which iteratively denoises a
    noisy input `xt` using a predicted noise component, as defined in Ho et al. (2020).
    The process relies on a noise schedule that can be either fixed or trainable,
    specified through the provided hyperparameters.

    Parameters
    ----------
    hyper_params : object
        Hyperparameter object containing the noise schedule parameters. Expected to have
        attributes:
        - `num_steps`: Number of diffusion steps (int).
        - `trainable_beta`: Whether the noise schedule is trainable (bool).
        - `betas`: Noise schedule parameters (torch.Tensor, optional if trainable_beta is True).
        - `alphas`: Precomputed alpha values (torch.Tensor, optional if trainable_beta is False).
        - `alpha_bars`: Precomputed cumulative product of alphas (torch.Tensor, optional if trainable_beta is False).
        - `compute_schedule`: Method to compute the noise schedule (callable, optional if trainable_beta is True).

    Attributes
    ----------
    hyper_params : object
        Stores the provided hyperparameter object for use in the reverse process.
    """
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params # hyperparameters class

    def forward(self, xt, predicted_noise, time_steps):
        """Applies the reverse diffusion process to the noisy input.

        Denoises the input `xt` by computing the mean of the reverse process
        distribution using the predicted noise and optionally adding stochastic noise
        for time steps greater than 0, as per the DDPM reverse process.

        Parameters
        ----------
        xt : torch.Tensor
            Noisy input tensor at time step `t`, of shape (batch_size, channels, height, width).
        predicted_noise : torch.Tensor
            Predicted noise tensor, of the same shape as `xt`, typically output by a neural network.
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, hyper_params.num_steps - 1].

        Returns
        -------
        torch.Tensor
            Denoised tensor `xt_minus_1` at time step `t-1`, with the same shape as `xt`.
            For time_steps == 0, returns the mean of the reverse process without added noise.

        Raises
        ------
        ValueError
            If any value in `time_steps` is outside the valid range
            [0, hyper_params.num_steps - 1].
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.hyper_params.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.hyper_params.num_steps - 1}")

        if self.hyper_params.trainable_beta:
            betas_t, alphas_t, alpha_bars_t, _, _ = self.hyper_params.compute_schedule(self.hyper_params.betas)
            betas_t = betas_t[time_steps].to(xt.device)
            alphas_t = alphas_t[time_steps].to(xt.device)
            alpha_bars_t = alpha_bars_t[time_steps].to(xt.device)
            alpha_bars_t_minus_1 = alpha_bars_t[time_steps - 1].to(xt.device) if time_steps.any() else None
        else:
            betas_t = self.hyper_params.betas[time_steps].to(xt.device)
            alphas_t = self.hyper_params.alphas[time_steps].to(xt.device)
            alpha_bars_t = self.hyper_params.alpha_bars[time_steps].to(xt.device)
            alpha_bars_t_minus_1 = self.hyper_params.alpha_bars[time_steps - 1].to(xt.device) if time_steps.any() else None

        sqrt_alphas_t = torch.sqrt(alphas_t).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bars_t = torch.sqrt(1 - alpha_bars_t).view(-1, 1, 1, 1)
        betas_t = betas_t.view(-1, 1, 1, 1)

        mu = (xt - (betas_t / sqrt_one_minus_alpha_bars_t) * predicted_noise) / sqrt_alphas_t

        mask = (time_steps == 0)
        if mask.all():
            return mu

        variance = (1 - alpha_bars_t_minus_1) / (1 - alpha_bars_t) * betas_t.squeeze()
        std = torch.sqrt(variance).view(-1, 1, 1, 1)

        z = torch.randn_like(xt).to(xt.device)
        xt_minus_1 = mu + (~mask).float().view(-1, 1, 1, 1) * std * z
        return xt_minus_1