import torch
import torch.nn as nn
from typing import Tuple, Optional



class ReverseDDPM(nn.Module):
    """Reverse diffusion process for DDPM

    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_t)
    """
    def __init__(
            self,
            scheduler: nn.Module,
            pred_type: str = "v",
            var_type: str = "fixed_small",
            clip_out: bool = True
    ) -> None:
        super().__init__()

        valid_pred_types = ["noise", "x0", "v"]
        valid_var_types = ["fixed_small", "fixed_large", "learned"]

        if pred_type not in valid_pred_types:
            raise ValueError(f"prediction_type must be one of {valid_pred_types}")
        if var_type not in valid_var_types:
            raise ValueError(f"var_type must be one of {valid_var_types}")

        self.vs = scheduler
        self.pred_type = pred_type
        self.var_type = var_type
        self.clip_out = clip_out

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Convert model output to x0 prediction based on prediction type"""

        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, xt.shape)

        if self.pred_type == "noise":
            # x_0 = (x_t - √(1 - ᾱ_t) * ε_θ) / √ᾱ_t
            x0 = (xt - sqrt_one_minus_alpha_cumprod_t * pred) / sqrt_alpha_cumprod_t

        elif self.pred_type == "x0":
            # directly predict x_0
            x0 = pred

        elif self.pred_type == "v":
            # x_0 = √ᾱ_t * x_t - √(1 - ᾱ_t) * v_θ
            x0 = sqrt_alpha_cumprod_t * xt - sqrt_one_minus_alpha_cumprod_t * pred

        if self.clip_out:
            x0 = torch.clamp(x0, -1.0, 1.0)

        return x0

    def get_variance(self, t: torch.Tensor, pred_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get variance for reverse process based on variance type"""
        if self.var_type == "fixed_small":
            # posterior variance: β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
            var = self.vs.posterior_variance[t]

        elif self.var_type == "fixed_large":
            # β_t
            var = self.vs.betas[t]

        elif self.var_type == "learned":
            # model-predicted variance
            if pred_var is None:
                raise ValueError("predicted_variance must be provided when variance_type='learned'")
            # interpolate between fixed_small and fixed_large
            min_log = self.vs.posterior_log_variance[t]
            max_log = torch.log(self.vs.betas[t])
            frac = (pred_var + 1) / 2  # map from [-1, 1] to [0, 1]
            var = torch.exp(frac * max_log + (1 - frac) * min_log)
        return var

    def forward(
            self,
            xt: torch.Tensor,
            pred: torch.Tensor,
            t: torch.Tensor,
            pred_var: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Single reverse step from x_t to x_{t-1}

        Args:
            xt: (batch, ...) current state
            t: (batch, ) current timesteps
            pred: (batch, ...) model prediction
            pred_var: (batch, ...) optional learned variance

        Returns:
            x_prev: (batch, ...) previous state x_{t-1}
            pred_x0: (batch, ...) predicted x0 (if return_pred_x0=True)
        """
        # predict x_0 from model output
        pred_x0 = self.predict_x0(xt, t, pred)
        # get posterior mean coefficients
        coef1 = self.vs.posterior_mean_coef1[t]
        coef2 = self.vs.posterior_mean_coef2[t]
        coef1 = self.vs.get_index(coef1, xt.shape)
        coef2 = self.vs.get_index(coef2, xt.shape)
        # posterior mean: μ_θ(x_t, t) = coef1 * x_0 + coef2 * x_t
        posterior_mean = coef1 * pred_x0 + coef2 * xt
        # variance
        variance = self.get_variance(t, pred_var)
        variance = self.vs.get_index(variance, xt.shape)
        # sample noise (no noise for t=0)
        noise = torch.randn_like(xt)
        mask = (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        # sample x_{t-1} ~ p_θ(x_{t-1} | x_t)
        x_prev = posterior_mean + mask * torch.sqrt(variance) * noise
        return x_prev, pred_x0