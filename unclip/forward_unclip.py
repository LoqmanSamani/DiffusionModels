import torch
import torch.nn as nn
from typing import Tuple



class ForwardUnCLIP(nn.Module):
    """Forward diffusion process for UnCLIP

    Applies Gaussian noise to input data according to the forward diffusion process.
    Supports both 2D (latent embeddings) and 4D (images) inputs.

    q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1 - ᾱ_t)I)
    """
    def __init__(self, scheduler: nn.Module, pred_type: str = "noise"):
        super().__init__()
        valid_types = ["noise", "x0"]
        if pred_type not in valid_types:
            raise ValueError(f"pred_type must be one of {valid_types}, got {pred_type}")
        self.vs = scheduler
        self.pred_type = pred_type

    def forward(
            self,
            x0: torch.Tensor,
            noise: torch.Tensor,
            t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from q(x_t | x_0) and compute prediction target

        Args:
            x0: (batch, ...) clean data (2D or 4D)
            t: (batch,) discrete timesteps in [0, train_steps-1]
            noise: (batch, ...) gaussian noise

        Returns:
            xt: (batch, ...) noised data
            target: (batch, ...) prediction target (noise or x0)
        """
        if not torch.all((t >= 0) & (t < self.vs.train_steps)):
            raise ValueError(f"t must be in [0, {self.vs.train_steps - 1}]")
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, x0.shape)
        # x_t ~ q(x_t | x_0)
        # x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        if self.pred_type == "noise":
            # predict noise ε (ddim-style)
            target = noise
        elif self.pred_type == "x0":
            # predict original data x_0 (unclip prior style)
            target = x0
        return xt, target