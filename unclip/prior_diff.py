import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class ForwardUnCLIP(nn.Module):
    def __init__(self, variance_scheduler: torch.nn.Module) -> None:
        super().__init__()
        self.variance_scheduler = variance_scheduler

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:

        if not torch.all((time_steps >= 0) & (time_steps < self.variance_scheduler.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.variance_scheduler.num_steps - 1}")

        if self.variance_scheduler.trainable_beta:
            _, _, _, sqrt_alpha_cumprod_t, sqrt_one_minus_alpha_cumprod_t = self.variance_scheduler.compute_schedule(
                time_steps
            )
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.to(x0.device)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.to(x0.device)
        else:
            sqrt_alpha_cumprod_t = self.variance_scheduler.sqrt_alpha_cumprod[time_steps].to(x0.device)
            sqrt_one_minus_alpha_cumprod_t = self.variance_scheduler.sqrt_one_minus_alpha_cumprod[time_steps].to(x0.device)

        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1, 1)

        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

        return xt


class ReverseUnCLIP(nn.Module):
    def __init__(self, variance_scheduler: torch.nn.Module):
        super().__init__()
        self.variance_scheduler = variance_scheduler

    def forward(self, xt: torch.Tensor, predicted_noise: torch.Tensor, time_steps: torch.Tensor, prev_time_steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if not torch.all((time_steps >= 0) & (time_steps < self.variance_scheduler.tau_num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.variance_scheduler.tau_num_steps - 1}")
        if not torch.all((prev_time_steps >= 0) & (prev_time_steps < self.variance_scheduler.tau_num_steps)):
            raise ValueError(f"prev_time_steps must be between 0 and {self.variance_scheduler.tau_num_steps - 1}")

        _, _, _, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod =self.variance_scheduler.get_tau_schedule()
        tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
        tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)

        eta = self.variance_scheduler.eta
        x0 = (xt - tau_sqrt_one_minus_alpha_cumprod_t * predicted_noise) / tau_sqrt_alpha_cumprod_t
        noise_coeff = eta * ((tau_sqrt_one_minus_alpha_cumprod_t / prev_tau_sqrt_alpha_cumprod_t) *
                             prev_tau_sqrt_one_minus_alpha_cumprod_t / torch.clamp(tau_sqrt_one_minus_alpha_cumprod_t, min=1e-8))
        direction_coeff = torch.clamp(prev_tau_sqrt_one_minus_alpha_cumprod_t ** 2 - noise_coeff ** 2, min=1e-8).sqrt()
        xt_prev = prev_tau_sqrt_alpha_cumprod_t * x0 + noise_coeff * torch.randn_like(xt) + direction_coeff * predicted_noise

        return xt_prev, x0


class VarianceSchedulerUnCLIP(nn.Module):
    def __init__(
            self,
            eta: Optional[float] = None,
            num_steps: int = 1000,
            tau_num_steps: int = 100,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            trainable_beta: bool = False,
            beta_method: str = "linear"
    ):
        super().__init__()
        self.eta = eta or 0
        self.num_steps = num_steps
        self.tau_num_steps = tau_num_steps
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
            self.beta_raw = nn.Parameter(torch.logit((betas_init - beta_start) / (beta_end - beta_start)))
        else:
            self.register_buffer('betas_buffer', betas_init)
            self.register_buffer('alphas', 1 - self.betas)
            self.register_buffer('alpha_cumprod', torch.cumprod(self.alphas, dim=0))
            self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(self.alpha_cumprod))
            self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1 - self.alpha_cumprod))

        self.register_buffer('tau_indices', torch.linspace(0, num_steps - 1, tau_num_steps, dtype=torch.long))


    @property
    def betas(self) -> torch.Tensor:
        """returns the beta values, applying reparameterization if trainable."""
        if self.trainable_beta:
            return self.beta_start + (self.beta_end - self.beta_start) * torch.sigmoid(self.beta_raw)

        return self._buffers['betas_buffer']


    def compute_beta_schedule(self, beta_range: Tuple[float, float], num_steps: int, method: str) -> torch.Tensor:

        beta_min, beta_max = beta_range
        if method == "sigmoid":
            x = torch.linspace(-6, 6, num_steps)
            beta = torch.sigmoid(x) * (beta_max - beta_min) + beta_min
        elif method == "quadratic":
            x = torch.linspace(beta_min ** 0.5, beta_max ** 0.5, num_steps)
            beta = x ** 2
        elif method == "constant":
            beta = torch.full((num_steps,), beta_max)
        elif method == "inverse_time":
            beta = 1.0 / torch.linspace(num_steps, 1, num_steps)
            beta = beta_min + (beta_max - beta_min) * (beta - beta.min()) / (beta.max() - beta.min())
        elif method == "linear":
            beta = torch.linspace(beta_min, beta_max, num_steps)
        elif method == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            s = 0.008  # small offset
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        else:
            raise ValueError(
                f"Unknown beta_method: {method}. Supported: linear, sigmoid, quadratic, constant, inverse_time")

        beta = torch.clamp(beta, min=beta_min, max=beta_max)
        return beta

    def get_tau_schedule(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.trainable_beta:
            # use the property to get constrained betas
            betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod = self.compute_schedule()
        else:
            betas = self.betas
            alphas = self.alphas
            alpha_cumprod = self.alpha_cumprod
            sqrt_alpha_cumprod = self.sqrt_alpha_cumprod
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod

        tau_betas = betas[self.tau_indices]
        tau_alphas = alphas[self.tau_indices]
        tau_alpha_cumprod = alpha_cumprod[self.tau_indices]
        tau_sqrt_alpha_cumprod = sqrt_alpha_cumprod[self.tau_indices]
        tau_sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod[self.tau_indices]

        return tau_betas, tau_alphas, tau_alpha_cumprod, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod

    def compute_schedule(self, time_steps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # use the property to get constrained betas
        betas = self.betas
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

        if time_steps is not None:
            return (betas[time_steps], alphas[time_steps], alpha_cumprod[time_steps],
                    sqrt_alpha_cumprod[time_steps], sqrt_one_minus_alpha_cumprod[time_steps])
        else:
            return betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod


"""
hyp = VarianceSchedulerUnCLIP(
    num_steps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=False,
    beta_method="sigmoid"
)

forward = ForwardUnCLIP(hyp)
x = torch.randn((10, 3, 100, 100))
t = torch.randint(0, 1000, (10,))
noise = torch.randn_like(x)

xt = forward(x, noise, t)
print(xt.size())
"""

