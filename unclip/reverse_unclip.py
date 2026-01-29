import torch
import torch.nn as nn
from typing import Tuple



class ReverseUnCLIP(nn.Module):
    """Reverse diffusion process for UnCLIP

    Denoises input using DDIM-style sampling with the tau (subsampled) schedule.
    Supports both noise prediction and x0 prediction modes.
    Works with both 2D (latent embeddings) and 4D (images) inputs.
    """
    def __init__(self, scheduler: nn.Module, pred_type: str = "noise", eta: float = 0.0, clip_: bool = True):
        super().__init__()
        valid_pred_types = ["noise", "x0"]
        if pred_type not in valid_pred_types:
            raise ValueError(f"pred_type must be one of {valid_pred_types}")

        self.vs = scheduler
        self.pred_type = pred_type
        self.eta = eta  # noise scaling factor (0 = deterministic)
        self.clip_ = clip_

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Convert model output to x0 prediction based on prediction type"""
        actual_t = self.vs.inference_timesteps[t]
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[actual_t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[actual_t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, xt.shape)
        if self.pred_type == "noise":
            # x_0 = (x_t - √(1 - ᾱ_t) * ε_θ) / √ᾱ_t
            x0_pred = (xt - sqrt_one_minus_alpha_cumprod_t * pred) / sqrt_alpha_cumprod_t
        elif self.pred_type == "x0":
            # directly predict x_0
            x0_pred = pred
        if self.clip_:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        return x0_pred

    def predict_noise(self, xt: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor) -> torch.Tensor:
        """Predict noise from x0

        ε̂ = (x_t - √ᾱ_t * x̂_0) / √(1 - ᾱ_t)
        """
        actual_t = self.vs.inference_timesteps[t]
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[actual_t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[actual_t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, xt.shape)
        pred_noise = (xt - sqrt_alpha_cumprod_t * x0_pred) / sqrt_one_minus_alpha_cumprod_t
        return pred_noise

    def forward(self, xt: torch.Tensor, t: torch.Tensor, t_pre: torch.Tensor, pred: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """UnCLIP reverse step from x_t to x_{t_prev}

        Uses tau schedule (subsampled timesteps) for faster sampling.

        Args:
            xt: (batch, ...) current state (2D or 4D)
            t: (batch,) current tau timestep indices [0, sample_steps-1]
            t_pre: (batch,) previous tau timestep indices
            pred: (batch, ...) model prediction

        Returns:
            x_prev: (batch, ...) previous state x_{t_prev}
            pred_x0: (batch, ...) predicted x0 (if return_pred_x0=True)
        """
        if not torch.all((t >= 0) & (t < self.vs.sample_steps)):
            raise ValueError(f"t must be in [0, {self.vs.sample_steps - 1}]")
        if not torch.all((t_pre >= 0) & (t_pre < self.vs.sample_steps)):
            raise ValueError(f"t_prev must be in [0, {self.vs.sample_steps - 1}]")

        pred_x0 = self.predict_x0(xt, t, pred)
        pred_noise = self.predict_noise(xt, t, pred_x0)
        actual_t = self.vs.inference_timesteps[t]
        actual_t_prev = self.vs.inference_timesteps[t_pre]
        alpha_cumprod_t = self.vs.alphas_cumprod[actual_t]
        alpha_cumprod_t_prev = self.vs.alphas_cumprod[actual_t_prev]
        alpha_cumprod_t = self.vs.get_index(alpha_cumprod_t, xt.shape)
        alpha_cumprod_t_prev = self.vs.get_index(alpha_cumprod_t_prev, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
        sqrt_one_minus_alpha_cumprod_t_prev = torch.sqrt(1.0 - alpha_cumprod_t_prev)

        noise_coeff = self.eta * (
                (sqrt_one_minus_alpha_cumprod_t / sqrt_alpha_cumprod_t_prev) *
                sqrt_one_minus_alpha_cumprod_t_prev /
                torch.clamp(sqrt_one_minus_alpha_cumprod_t, min=1e-8)
        )
        direction_coeff = torch.sqrt(
            torch.clamp(
                sqrt_one_minus_alpha_cumprod_t_prev ** 2 - noise_coeff ** 2,
                min=1e-8
            )
        )
        noise = torch.randn_like(xt)
        mask = (actual_t_prev != 0).float()
        mask = self.vs.get_index(mask, xt.shape)
        # x_{t_prev} = √ᾱ_{t_prev} * x̂_0 + noise_coeff * z + direction_coeff * ε̂
        x_prev = (
                sqrt_alpha_cumprod_t_prev * pred_x0 +
                noise_coeff * mask * noise +
                direction_coeff * pred_noise
        )
        return x_prev, pred_x0

    def set_pred_type(self, pred_type: str):
        """Change the prediction type after initialization"""
        if pred_type not in ["noise", "x0"]:
            raise ValueError(f"pred_type must be 'noise' or 'x0'")
        self.pred_type = pred_type