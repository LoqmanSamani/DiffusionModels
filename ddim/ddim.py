import torch
import torch.nn as nn


class HyperParams(nn.Module):
    """hyperparameters for DDPM/DDIM noise schedule with flexible beta computation."""

    def __init__(self, eta=None, num_steps=1000, tau_num_steps=100, beta_start=1e-4, beta_end=0.02,
                 trainable_beta=False, beta_method="linear"):
        super().__init__()

        self.eta = eta or 0  # 1 for ddpm (stochastic), 0 for fully ddim
        self.num_steps = num_steps  # used for forward diffusion during training
        self.tau_num_steps = tau_num_steps  # used for generation (fewer steps for efficiency)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trainable_beta = trainable_beta
        self.beta_method = beta_method

        # validate inputs
        if not (0 < beta_start < beta_end < 1):
            raise ValueError(f"beta_start ({beta_start}) and beta_end ({beta_end}) must satisfy 0 < start < end < 1")
        if num_steps <= 0:
            raise ValueError(f"num_steps ({num_steps}) must be positive")

        # compute initial beta schedule
        beta_range = (beta_start, beta_end)
        betas_init = self.compute_beta_schedule(beta_range, num_steps, beta_method)
        tau_betas_init = self.compute_beta_schedule(beta_range, tau_num_steps, beta_method)

        # initialize betas
        if trainable_beta:
            self.betas = nn.Parameter(betas_init)
            self.tau_betas = nn.Parameter(betas_init)
        else:
            self.register_buffer('betas', betas_init)
            self.register_buffer('tau_betas', tau_betas_init)
            self.register_buffer('alphas', 1 - self.betas)
            self.register_buffer('tau_alphas', 1 - self.tau_betas)
            self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
            self.register_buffer('tau_alpha_bars', torch.cumprod(self.tau_alphas, dim=0))
            self.register_buffer('sqrt_alpha_bars', torch.sqrt(self.alpha_bars))
            self.register_buffer('tau_sqrt_alpha_bars', torch.sqrt(self.tau_alpha_bars))
            self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1 - self.alpha_bars))
            self.register_buffer('tau_sqrt_one_minus_alpha_bars', torch.sqrt(1 - self.tau_alpha_bars))

    def compute_beta_schedule(self, beta_range, num_steps, method="linear"):

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
            # scale to beta_range
            beta = beta_min + (beta_max - beta_min) * (beta - beta.min()) / (beta.max() - beta.min())
        elif method == "linear":
            beta = torch.linspace(beta_min, beta_max, num_steps)
        else:
            raise ValueError(
                f"Unknown beta_method: {method}. Supported: linear, sigmoid, quadratic, constant, inverse_time")

        return beta

    @staticmethod
    def compute_schedule(betas):
        """computes noise schedule parameters dynamically from betas."""
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars, torch.sqrt(alpha_bars), torch.sqrt(1 - alpha_bars)

    def constrain_betas(self):
        """constrains betas to a valid range during training."""
        if self.trainable_beta:
            with torch.no_grad():
                self.betas.clamp_(min=self.beta_start, max=self.beta_end)



class ForwardDDIM(nn.Module):
    def __init__(self, hyper_params):
        """forward diffusion process of DDIM"""
        super().__init__()
        self.hyper_params = hyper_params

    def forward(self, x0, noise, time_steps):
        if not torch.all((time_steps >= 0) & (time_steps < self.hyper_params.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.hyper_params.num_steps - 1}")

        if self.hyper_params.trainable_beta:
            _, _, _, sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = self.hyper_params.compute_schedule(
                self.hyper_params.betas
            )
            sqrt_alpha_bar_t = sqrt_alpha_bar_t[time_steps].to(x0.device)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[time_steps].to(x0.device)
        else:
            sqrt_alpha_bar_t = self.hyper_params.sqrt_alpha_bars[time_steps].to(x0.device)
            sqrt_one_minus_alpha_bar_t = self.hyper_params.sqrt_one_minus_alpha_bars[time_steps].to(x0.device)

        sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(-1, 1, 1, 1)

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt


class ReverseDDIM(nn.Module):
    """reverse diffusion process of DDIM."""

    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params  # hyperparameters class

    def forward(self, xt, predicted_noise, time_steps, prev_time_steps):
        if not torch.all((time_steps >= 0) & (time_steps < self.hyper_params.tau_num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.hyper_params.tau_num_steps - 1}")

        if self.hyper_params.trainable_beta:
            _, _, _, tau_sqrt_alpha_bar_t, tau_sqrt_one_minus_alpha_bar_t = self.hyper_params.compute_schedule(self.hyper_params.tau_betas)
            tau_sqrt_alpha_bar_t = tau_sqrt_alpha_bar_t[time_steps].to(xt.device)
            tau_sqrt_one_minus_alpha_bar_t = tau_sqrt_one_minus_alpha_bar_t[time_steps].to(xt.device)
            prev_tau_sqrt_alpha_bar_t = tau_sqrt_alpha_bar_t[prev_time_steps].to(xt.device)
            prev_tau_sqrt_one_minus_alpha_bar_t = tau_sqrt_one_minus_alpha_bar_t[prev_time_steps].to(xt.device)

        else:
            tau_sqrt_alpha_bar_t = self.hyper_params.tau_sqrt_alpha_bar_t[time_steps].to(xt.device)
            tau_sqrt_one_minus_alpha_bar_t = self.hyper_params.tau_sqrt_one_minus_alpha_bar_t[time_steps].to(xt.device)
            prev_tau_sqrt_alpha_bar_t = self.hyper_params.tau_sqrt_alpha_bar_t[prev_time_steps].to(xt.device)
            prev_tau_sqrt_one_minus_alpha_bar_t = self.hyper_params.tau_sqrt_one_minus_alpha_bar_t[prev_time_steps].to(xt.device) if time_steps.any() else None

        eta = self.hyper_params.eta

        x0 = xt - predicted_noise * tau_sqrt_one_minus_alpha_bar_t / tau_sqrt_alpha_bar_t
        c1 = eta * ((tau_sqrt_one_minus_alpha_bar_t /  prev_tau_sqrt_alpha_bar_t) * prev_tau_sqrt_one_minus_alpha_bar_t / torch.clamp(tau_sqrt_one_minus_alpha_bar_t, min=1e-8))
        c2 = torch.clamp(prev_tau_sqrt_one_minus_alpha_bar_t - c1 ** 2, min=1e-8)
        xt_prev =  prev_tau_sqrt_alpha_bar_t * x0 + c1 * torch.randn_like(xt) + c2 * predicted_noise

        return xt_prev, x0


