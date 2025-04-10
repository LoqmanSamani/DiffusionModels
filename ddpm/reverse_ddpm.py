import torch
import torch.nn as nn




class ReverseDDPM(nn.Module):
    """reverse diffusion process of DDPM."""
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params # hyperparameters class

    def forward(self, xt, predicted_noise, time_steps):
        if not torch.all((time_steps >= 0) & (time_steps < self.hyper_params.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.hyper_params.num_steps - 1}")

        if self.hyper_params.trainable_beta:
            betas_t, alphas_t, alpha_bars_t, _, _ = self.hyper_params.compute_schedule(self.hyper_params.betas)
            betas_t = betas_t[time_steps].to(xt.device)
            alphas_t = alphas_t[time_steps].to(xt.device)
            alpha_bars_t = alpha_bars_t[time_steps].to(xt.device)
            alpha_bars_t_minus_1 = alpha_bars_t[time_steps - 1].to(xt.device) if time_steps.any() else None
        else:
            betas_t = self.hyper_params.betas[time_steps].to(xt.device)
            alphas_t = self.hyper_params.alphas[time_steps].to(xt.device)
            alpha_bars_t = self.hyper_params.alpha_bars[time_steps].to(xt.device)
            alpha_bars_t_minus_1 = self.hyper_params.alpha_bars[time_steps - 1].to(xt.device) if time_steps.any() else None

        sqrt_alphas_t = torch.sqrt(alphas_t).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bars_t = torch.sqrt(1 - alpha_bars_t).view(-1, 1, 1, 1)
        betas_t = betas_t.view(-1, 1, 1, 1)

        mu = (xt - (betas_t / sqrt_one_minus_alpha_bars_t) * predicted_noise) / sqrt_alphas_t

        mask = (time_steps == 0)
        if mask.all():
            return mu

        variance = (1 - alpha_bars_t_minus_1) / (1 - alpha_bars_t) * betas_t.squeeze()
        std = torch.sqrt(variance).view(-1, 1, 1, 1)

        z = torch.randn_like(xt).to(xt.device)
        xt_minus_1 = mu + (~mask).float().view(-1, 1, 1, 1) * std * z
        return xt_minus_1