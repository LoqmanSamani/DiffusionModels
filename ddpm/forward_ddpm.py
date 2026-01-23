import torch
import torch.nn as nn
from typing import Tuple



class ForwardDDPM(nn.Module):
    """Forward diffusion process for DDPM
    q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1 - ᾱ_t)I)
    """
    def __init__(self, scheduler: nn.Module, pred_type: str = "v") -> None:
        super().__init__()

        valid_types = ["noise","x0", "v"]
        if pred_type not in valid_types:
            raise ValueError(f"prediction_type must be one of {valid_types}, got {pred_type}")

        self.vs = scheduler
        self.pred_type = pred_type

    def forward(
            self,
            x0: torch.Tensor,
            t: torch.Tensor,
            noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from q(x_t | x_0) and compute prediction target

        Args:
            x0: (batch, ...) clean data
            t: (batch, ) discrete timesteps in [0, time_steps-1]
            noise: (batch, ...) standard Gaussian noise

        Returns:
            xt: (batch, ...) noised data
            target: (batch, ...) prediction target (x0, v-prediction)
        """
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, x0.shape)
        # x_t ~ q(x_t | x_0)
        # x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

        if self.pred_type == 'noise':
            target = noise
        elif self.pred_type == "x0":
            target = x0
        elif self.pred_type == "v":
            # v-prediction: v = √ᾱ_t * ε - √(1 - ᾱ_t) * x_0
            target = sqrt_alpha_cumprod_t * noise - sqrt_one_minus_alpha_cumprod_t * x0
        return xt, target