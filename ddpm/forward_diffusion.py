import torch
import torch.nn as nn




class ForwardDDPM(nn.Module):
    """
    Forward diffusion process of Denoising Diffusion Probabilistic Models (DDPM).
    Precomputes the noise schedule and adds Gaussian noise to images at specified timesteps.
    """
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # precompute noise schedule parameters
        betas, alphas, alpha_bars = self._compute_schedule(num_steps, beta_start, beta_end)
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars))  # sqrt(ᾱ_t)
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1 - alpha_bars))  # sqrt(1 - ᾱ_t)

    @staticmethod
    def _compute_schedule(num_steps, beta_start, beta_end):
        """computes the noise schedule parameters."""
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars

    def forward(self, x0, noise, time_steps):
        """
        Adds Gaussian noise to the input images at specified timesteps.
        Args:
            x0: Input images, shape [batch_size, channels, height, width]
            noise: Gaussian noise, same shape as x0
            time_steps: Tensor of timesteps, shape [batch_size]
        Returns:
            xt: Noisy images at time t, same shape as x0
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.num_steps - 1}")

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[time_steps].to(x0.device)  # [batch_size]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[time_steps].to(x0.device)  # [batch_size]

        # reshape for broadcasting
        sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(-1, 1, 1, 1)

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt