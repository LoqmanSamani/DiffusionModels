import torch
import torch.nn as nn




class ReverseSDE(nn.Module):
    def __init__(self, hyper_params, method):
        super().__init__()
        self.hyper_params = hyper_params
        self.method = method

    def forward(self, xt, noise, predicted_noise, time_steps):

        dt = self.hyper_params.dt
        betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
        cum_betas = self.hyper_params.cum_betas[time_steps].view(-1, 1, 1, 1)
        if self.method == "ve":
            sigma_t = self.hyper_params.sigmas[time_steps]
            sigma_t_prev = self.hyper_params.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
            sigma_diff = torch.sqrt(torch.clamp(sigma_t ** 2 - sigma_t_prev ** 2, min=0))
            drift = -(sigma_t ** 2 - sigma_t_prev ** 2).view(-1, 1, 1, 1) * predicted_noise * dt
            diffusion = sigma_diff.view(-1, 1, 1, 1) * noise if noise is not None else 0
            xt = xt + drift + diffusion
            xt = torch.clamp(xt, -1e5, 1e5)

        elif self.method == "vp":
            drift = -0.5 * betas * xt * dt - betas * predicted_noise * dt
            diffusion = torch.sqrt(betas * dt) * noise if noise is not None else 0
            xt = xt + drift + diffusion

        elif self.method == "sub-vp":
            drift = -0.5 * betas * xt * dt - betas * (1 - torch.exp(-2 * cum_betas)) * predicted_noise * dt
            diffusion = torch.sqrt(betas * (1 - torch.exp(-2 * cum_betas)) * dt) * noise if noise is not None else 0
            xt = xt + drift + diffusion

        elif self.method == "ode":
            if self.method == "ve":
                sigma_t = self.hyper_params.sigmas[time_steps]
                sigma_t_prev = self.hyper_params.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
                drift = -0.5 * (sigma_t ** 2 - sigma_t_prev ** 2).view(-1, 1, 1, 1) * predicted_noise * dt
            else:
                drift = -0.5 * betas * xt * dt - 0.5 * betas * predicted_noise * dt
            xt = xt + drift
            xt = torch.clamp(xt, -1e5, 1e5)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return xt