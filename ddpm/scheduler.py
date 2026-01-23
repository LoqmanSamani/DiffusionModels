import torch
import torch.nn as nn



class SchedulerDDPM(nn.Module):
    """ Scheduler for DDPM supporting linear, cosine, and other schedules"""
    def __init__(
            self,
            schedule_type: str = "linear",
            time_steps: int = 1000,
            beta_min: float = 0.0001,
            beta_max: float = 0.02,
            cosine_s: float = 0.008,
            clip_min: float = 0.0001,
            clip_max: float = 0.9999
    ):
        super().__init__()
        valid_schedules = ["linear", "cosine", "quadratic", "sigmoid"]
        if schedule_type not in valid_schedules:
            raise ValueError(f"schedule_type must be one of {valid_schedules}, got {schedule_type}")

        self.schedule_type = schedule_type
        self.time_steps = time_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cosine_s = cosine_s
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._setup_schedule()

    def _setup_schedule(self):
        """Setup the noise schedule and precompute all coefficients"""
        if self.schedule_type == "linear":
            betas = torch.linspace(self.beta_min, self.beta_max, self.time_steps)

        elif self.schedule_type == "cosine":
            steps = self.time_steps + 1
            t = torch.linspace(0, self.time_steps, steps)
            alphas_cumprod = torch.cos(((t / self.time_steps) + self.cosine_s) / (1 + self.cosine_s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, self.clip_min, self.clip_max)

        elif self.schedule_type == "quadratic":
            betas = torch.linspace(self.beta_min ** 0.5, self.beta_max ** 0.5, self.time_steps) ** 2

        elif self.schedule_type == "sigmoid":
            betas = torch.linspace(-6, 6, self.time_steps)
            betas = torch.sigmoid(betas) * (self.beta_max - self.beta_min) + self.beta_min

        # compute alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # compute coefficients for q(x_t | x_0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # compute coefficients for q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        # register as buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', posterior_log_variance)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

    def get_index(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract coefficients at timestep t and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = t.to(t.device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))