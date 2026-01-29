import torch
import torch.nn as nn
from typing import Optional
import math



class SchedulerUnCLIP(nn.Module):
    """Variance scheduler for UnCLIP supporting multiple schedule types

    Manages noise schedule parameters with support for both full training schedule
    and subsampled inference schedule  for faster sampling.
    """
    def __init__(
            self,
            schedule_type: str = "linear",
            train_steps: int = 1000,
            sample_steps: Optional[int] = None,
            beta_min: float = 1e-4,
            beta_max: float = 0.02,
            cosine_s: float = 0.008,
            clip_min: float = 1e-4,
            clip_max: float = 0.9999,
            learn_var: bool = False
    ):
        super().__init__()
        valid_schedules = ["linear", "cosine", "quadratic", "sigmoid", "constant", "inverse_time"]
        if schedule_type not in valid_schedules:
            raise ValueError(f"schedule_type must be one of {valid_schedules}, got {schedule_type}")
        if not (0 < beta_min < beta_max < 1):
            raise ValueError(f"beta_start and beta_end must satisfy 0 < beta_start < beta_end < 1")

        self.schedule_type = schedule_type
        self.train_steps = train_steps
        self.sample_steps = sample_steps or train_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cosine_s = cosine_s
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.learn_var = learn_var
        self._setup_schedule()
        self._setup_inf_timesteps()

    def _setup_schedule(self):
        """Setup the noise schedule and precompute all coefficients"""
        if self.schedule_type == "linear":
            betas = torch.linspace(self.beta_min, self.beta_max, self.train_steps)
        elif self.schedule_type == "cosine":
            steps = self.train_steps + 1
            t = torch.linspace(0, self.train_steps, steps)
            alphas_cumprod = torch.cos(((t / self.train_steps) + self.cosine_s) / (1 + self.cosine_s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, self.clip_min, self.clip_max)

        elif self.schedule_type == "quadratic":
            betas = torch.linspace(
                self.beta_min ** 0.5,
                self.beta_max ** 0.5,
                self.train_steps
            ) ** 2

        elif self.schedule_type == "sigmoid":
            x = torch.linspace(-6, 6, self.train_steps)
            betas = torch.sigmoid(x) * (self.beta_max - self.beta_min) + self.beta_min

        elif self.schedule_type == "constant":
            betas = torch.full((self.train_steps,), self.beta_max)

        elif self.schedule_type == "inverse_time":
            beta = 1.0 / torch.linspace(self.train_steps, 1, self.train_steps)
            betas = self.beta_min + (self.beta_max - self.beta_min) * (
                    (beta - beta.min()) / (beta.max() - beta.min())
            )
        betas = torch.clamp(betas, min=self.clip_min, max=self.clip_max)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))

        if self.learn_var:
            self.register_parameter('log_variance', nn.Parameter(posterior_log_variance.clone()))
        else:
            self.register_buffer('log_variance', posterior_log_variance)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', posterior_log_variance)

    def _setup_inf_timesteps(self):
        """Setup inference timesteps (tau schedule for UnCLIP)

        Creates a uniform subset of timesteps for faster sampling.
        Similar to DDIM but called 'tau schedule' in UnCLIP literature.
        """
        step_ratio = self.train_steps // self.sample_steps
        # Create uniform spacing: [0, step_ratio, 2*step_ratio, ...]
        inf_timesteps = torch.arange(0, self.train_steps, step_ratio, dtype=torch.long)
        self.register_buffer('inference_timesteps', inf_timesteps)

    def set_inf_timesteps(self, num_inf_timesteps: int):
        """Dynamically change the number of inference steps

        Allows using different numbers of steps at inference time.
        """
        self.sample_steps = num_inf_timesteps
        self._setup_inf_timesteps()

    def get_index(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract coefficients at timestep t and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = t.to(t.device)
        if len(x_shape) == 2:
            return out.reshape(batch_size, 1)
        else:
            return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))