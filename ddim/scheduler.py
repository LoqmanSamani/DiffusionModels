import torch
import torch.nn as nn
from typing import Optional



class SchedulerDDIM(nn.Module):
    """
    Noise scheduler for DDIM.

    Responsible for constructing the diffusion noise schedule and
    precomputing all coefficients required for both training and sampling.

    Supports multiple beta schedules and allows using fewer inference
    steps than training steps.

    Args:
        schedule_type: Type of beta schedule ("linear", "cosine", etc.).
        train_steps: Number of diffusion steps used during training.
        sample_steps: Number of steps used during inference.
        beta_min: Minimum beta value.
        beta_max: Maximum beta value.
        cosine_s: Offset parameter for cosine schedule.
        clip_min: Minimum clipping value for betas.
        clip_max: Maximum clipping value for betas.
        learn_var: Whether posterior variance is learnable.
    """
    def __init__(
            self,
            schedule_type: str = "linear",
            train_steps: int = 1000,  # total timesteps for training
            sample_steps: Optional[int] = None,  # can use fewer steps for sampling
            beta_min: float = 0.0001,
            beta_max: float = 0.02,
            cosine_s: float = 0.008,
            clip_min: float = 0.0001,
            clip_max: float = 0.9999,
            learn_var: bool = False
    ):
        super().__init__()
        valid_schedules = ["linear", "cosine", "quadratic", "sigmoid"]
        if schedule_type not in valid_schedules:
            raise ValueError(f"schedule_type must be one of {valid_schedules}, got {schedule_type}")

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
        self._setup_inference_timesteps()

    def _setup_schedule(self):
        """
        Construct the diffusion noise schedule and precompute coefficients.

        Computes betas, alphas, cumulative products, square roots, and
        posterior variances required for forward and reverse diffusion.
        """
        if self.schedule_type == "linear":
            betas = torch.linspace(self.beta_min, self.beta_max, self.train_steps)
        elif self.schedule_type == "cosine":
            steps = self.train_steps + 1
            t = torch.linspace(0, self.train_steps, steps)
            alphas_cumprod = torch.cos(((t / self.train_steps) + self.cosine_s) / (1 + self.cosine_s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, self.clip_min, self.clip_max)
        elif self.schedule_type == "quadratic":
            betas = torch.linspace(self.beta_min ** 0.5, self.beta_max ** 0.5, self.train_steps) ** 2
        elif self.schedule_type == "sigmoid":
            betas = torch.linspace(-6, 6, self.train_steps)
            betas = torch.sigmoid(betas) * (self.beta_max - self.beta_min) + self.beta_min
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

    def _setup_inference_timesteps(self):
        """
        Create the set of timesteps used during inference.

        DDIM allows skipping timesteps for faster sampling by selecting
        a subset of training timesteps.
        """
        step_ratio = self.train_steps // self.sample_steps
        inference_timesteps = torch.arange(0, self.train_steps, step_ratio)

        self.register_buffer('inference_timesteps', inference_timesteps)

    def set_inference_timesteps(self, num_inference_timesteps: int):
        """
        Update the number of inference timesteps dynamically.

        Allows changing sampling speed and quality trade-offs at inference
        time without retraining the model.

        Args:
            num_inference_timesteps: Number of timesteps to use for sampling.
        """
        self.sample_steps = num_inference_timesteps
        self._setup_inference_timesteps()

    def get_index(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Reshape timestep-dependent coefficients for broadcasting.

        Extracts values indexed by t and reshapes them to match the
        dimensionality of a given tensor shape.

        Args:
            t: Timesteps tensor of shape (batch,).
            x_shape: Shape of the target tensor.

        Returns:
            Reshaped tensor suitable for broadcasting.
        """
        batch_size = t.shape[0]
        out = t.to(t.device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))