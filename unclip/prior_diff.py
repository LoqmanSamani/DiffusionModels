import torch
import torch.nn as nn
from typing import Optional, Tuple


class ForwardDIF(nn.Module):
    def __init__(self, hyper_params: torch.nn.Module) -> None:
        super().__init__()
        self.hyper_params = hyper_params

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, time_steps: torch.tensor) -> torch.Tensor:

        if not torch.all((time_steps >= 0) & (time_steps < self.hyper_params.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.hyper_params.num_steps - 1}")

        if self.hyper_params.trainable_beta:
            _, _, _, sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = self.hyper_params.compute_schedule(time_steps)
            sqrt_alpha_bar_t = sqrt_alpha_bar_t.to(x0.device)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.to(x0.device)
        else:
            sqrt_alpha_bar_t = self.hyper_params.sqrt_alpha_bars[time_steps].to(x0.device)
            sqrt_one_minus_alpha_bar_t = self.hyper_params.sqrt_one_minus_alpha_bars[time_steps].to(x0.device)

        sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(-1, 1, 1, 1)
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return xt


class HyperParamsDIF(nn.Module):
    def __init__(
            self,
            num_steps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            trainable_beta: bool = False,
            beta_method: str = "linear"
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trainable_beta = trainable_beta
        self.beta_method = beta_method

        if not (0 < beta_start < beta_end < 1):
            raise ValueError(f"beta_start ({beta_start}) and beta_end ({beta_end}) must satisfy 0 < start < end < 1")
        if num_steps <= 0:
            raise ValueError(f"num_steps ({num_steps}) must be positive")

        beta_range = (beta_start, beta_end)
        betas_init = self.compute_beta_schedule(beta_range, num_steps, beta_method)

        if trainable_beta:
            self.betas = nn.Parameter(torch.log(betas_init))
        else:
            self.register_buffer('betas', betas_init)
            self.register_buffer('alphas', 1 - self.betas)
            self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
            self.register_buffer('sqrt_alpha_bars', torch.sqrt(self.alpha_bars))
            self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1 - self.alpha_bars))

    def compute_beta_schedule(self, beta_range: Tuple[float, float], num_steps: int, method: str) -> torch.Tensor:

        beta_min, beta_max = beta_range
        if method == "sigmoid":
            x = torch.linspace(-6, 6, num_steps)
            beta = torch.sigmoid(x) * (beta_max - beta_min) + beta_min
        elif method == "quadratic":
            x = torch.linspace(beta_min**0.5, beta_max**0.5, num_steps)
            beta = x**2
        elif method == "constant":
            beta = torch.full((num_steps,), beta_max)
        elif method == "inverse_time":
            beta = 1.0 / torch.linspace(num_steps, 1, num_steps)
            beta = beta_min + (beta_max - beta_min) * (beta - beta.min()) / (beta.max() - beta.min())
        elif method == "linear":
            beta = torch.linspace(beta_min, beta_max, num_steps)
        else:
            raise ValueError(f"Unknown beta_method: {method}. Supported: linear, sigmoid, quadratic, constant, inverse_time")

        beta = torch.clamp(beta, min=beta_min, max=beta_max)
        return beta

    def compute_schedule(
            self,
            time_steps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.trainable_beta:
            # Compute betas from trainable log_betas using sigmoid
            betas = torch.sigmoid(self.betas) * (self.beta_end - self.beta_start) + self.beta_start
        else:
            betas = self.betas

        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        if time_steps is not None:
            betas = betas[time_steps]
            alphas = alphas[time_steps]
            alpha_bars = alpha_bars[time_steps]

        return betas, alphas, alpha_bars, torch.sqrt(alpha_bars), torch.sqrt(1 - alpha_bars)