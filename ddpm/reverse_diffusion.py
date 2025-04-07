import torch
import torch.nn as nn




class ReverseDDPM(nn.Module):
    """
    Reverse diffusion process of Denoising Diffusion Probabilistic Models (DDPM).
    Removes noise from images step-by-step to estimate the previous timestep.
    """
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # precompute noise schedule parameters
        betas, alphas, alpha_bars = self._compute_schedule(num_steps, beta_start, beta_end)
        self.register_buffer('betas', betas)  # β_t
        self.register_buffer('alphas', alphas)  # α_t = 1 - β_t
        self.register_buffer('alpha_bars', alpha_bars)  # ᾱ_t = ∏ α_s

    @staticmethod
    def _compute_schedule(num_steps, beta_start, beta_end):
        """computes the noise schedule parameters (shared with ForwardDiffusion)."""
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars

    def forward(self, xt, predicted_noise, time_steps):
        """
        Performs one step of the reverse diffusion process.
        Args:
            xt: Noisy images at timestep t, shape [batch_size, channels, height, width]
            predicted_noise: Noise predicted by a model, same shape as xt
            time_steps: Tensor of timesteps, shape [batch_size]
        Returns:
            xt_minus_1: Estimated images at timestep t-1, same shape as xt
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.num_steps - 1}")

        # extract noise schedule parameters
        alphas_t = self.alphas[time_steps].to(xt.device)  # [batch_size]
        alpha_bars_t = self.alpha_bars[time_steps].to(xt.device)  # [batch_size]
        betas_t = self.betas[time_steps].to(xt.device)  # [batch_size]

        # reshape for broadcasting
        sqrt_alphas_t = torch.sqrt(alphas_t).view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
        sqrt_one_minus_alpha_bars_t = torch.sqrt(1 - alpha_bars_t).view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
        betas_t = betas_t.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]

        # compute mean: μ_θ = (1 / sqrt(α_t)) * (x_t - (β_t / sqrt(1 - ᾱ_t)) * ε_θ)
        mu = (xt - (betas_t / sqrt_one_minus_alpha_bars_t) * predicted_noise) / sqrt_alphas_t

        # if all time_steps are 0, return mean without noise
        mask = (time_steps == 0)  # [batch_size]
        if mask.all():
            return mu

        # compute variance: σ_t^2 = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        alpha_bars_t_minus_1 = self.alpha_bars[time_steps - 1].to(xt.device)  # [batch_size]
        variance = (1 - alpha_bars_t_minus_1) / (1 - alpha_bars_t) * betas_t.squeeze()  # Squeeze to [batch_size]
        std = torch.sqrt(variance).view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]

        # add noise for t > 0
        z = torch.randn_like(xt).to(xt.device)
        xt_minus_1 = mu + (~mask).float().view(-1, 1, 1, 1) * std * z
        return xt_minus_1