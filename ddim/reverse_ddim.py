import torch
import torch.nn as nn
from typing import Tuple, Optional



class ReverseDDIM(nn.Module):
    """
    Implements the reverse (denoising) process of DDIM.

    Computes x_{t_prev} from x_t using the DDIM update rule:

        x_{t_prev} = sqrt(alphā_{t_prev}) * x̂_0
                     + sqrt(1 - alphā_{t_prev} - σ_t²) * ε̂_t
                     + σ_t * z

    where σ_t controls stochasticity via eta.
    Setting eta=0 results in deterministic DDIM sampling.

    Args:
        scheduler: Noise scheduler containing diffusion coefficients.
        pred_type: Model prediction type ["noise", "x0", "v"].
        eta: Controls stochasticity of sampling (0 = deterministic).
        clip_: Whether to clip predicted x0 to [-1, 1].
    """

    def __init__(self, scheduler: nn.Module, pred_type: str = "noise", eta: float = 0.0, clip_: bool = True):
        super().__init__()
        valid_pred_types = ["noise", "x0", "v"]
        if pred_type not in valid_pred_types:
            raise ValueError(f"prediction_type must be one of {valid_pred_types}")
        self.vs = scheduler
        self.pred_type = pred_type
        self.eta = eta
        self.clip_ = clip_

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Convert model output into a prediction of the clean sample x0.

        The conversion depends on the chosen prediction parameterization
        (noise, x0, or v-prediction).

        Args:
            xt: Noisy input at timestep t of shape (batch, ...).
            t: Current timesteps of shape (batch,).
            pred: Model output of shape (batch, ...).

        Returns:
            x0_pred: Predicted clean sample x0.
        """
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, xt.shape)
        if self.pred_type == "noise":
            x0_pred = (xt - sqrt_one_minus_alpha_cumprod_t * pred) / sqrt_alpha_cumprod_t
        elif self.pred_type == "x0":
            x0_pred = pred
        elif self.pred_type == "v":
            x0_pred = sqrt_alpha_cumprod_t * xt - sqrt_one_minus_alpha_cumprod_t * pred
        if self.clip_:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        return x0_pred

    def predict_noise(self, xt: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the predicted noise ε̂_t from x_t and predicted x0.

        Uses the identity:
            ε̂_t = (x_t - sqrt(alphā_t) * x̂_0) / sqrt(1 - alphā_t)

        Args:
            xt: Noisy input at timestep t.
            t: Current timesteps.
            x0_pred: Predicted clean sample x0.

        Returns:
            pred_noise: Predicted noise ε̂_t.
        """
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, xt.shape)
        pred_noise = (xt - sqrt_alpha_cumprod_t * x0_pred) / sqrt_one_minus_alpha_cumprod_t
        return pred_noise

    def forward(
            self,
            xt: torch.Tensor,
            t: torch.Tensor,
            t_prev: torch.Tensor,
            pred: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform one DDIM reverse diffusion step.

        Computes x_{t_prev} from x_t using the DDIM update equation.
        Allows non-adjacent timesteps, enabling accelerated sampling.

        Args:
            xt: Current noisy sample x_t of shape (batch, ...).
            t: Current timestep indices of shape (batch,).
            t_prev: Previous timestep indices of shape (batch,).
            pred: Model prediction at timestep t.

        Returns:
            x_prev: Sample at timestep t_prev.
            pred_x0: Predicted clean sample x0.
        """
        pred_x0 = self.predict_x0(xt, t, pred)
        # predict noise from x_0
        pred_noise = self.predict_noise(xt, t, pred_x0)
        alpha_cumprod_t = self.vs.alphas_cumprod[t]
        alpha_cumprod_t_prev = self.vs.alphas_cumprod[t_prev]
        alpha_cumprod_t = self.vs.get_index(alpha_cumprod_t, xt.shape)
        alpha_cumprod_t_prev = self.vs.get_index(alpha_cumprod_t_prev, xt.shape)

        # compute variance σ_t
        # eta=0: fully deterministic (σ_t=0)
        # eta=1: maximum stochasticity (similar to ddpm)
        sigma_t = self.eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )
        # dir_xt = √(1 - ᾱ_{t_prev} - σ_t²) * ε̂_t
        sqrt_one_minus_alpha_cumprod_t_prev_minus_sigma = torch.sqrt(
            1.0 - alpha_cumprod_t_prev - sigma_t ** 2
        )
        dir_xt = sqrt_one_minus_alpha_cumprod_t_prev_minus_sigma * pred_noise
        noise = torch.randn_like(xt)
        mask = (t_prev != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        # x_{t_prev} = √ᾱ_{t_prev} * x̂_0 + dir_xt + σ_t * z
        x_prev = (torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + sigma_t * mask * noise)
        return x_prev, pred_x0