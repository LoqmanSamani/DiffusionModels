import torch
import torch.nn as nn



class ForwardSDE(nn.Module):
    def __init__(self, hyper_params, method):
        super().__init__()
        self.hyper_params = hyper_params
        self.method = method

    def forward(self, x0, noise, time_steps):

        dt = self.hyper_params.dt
        if self.method == "ve":
            sigma_t = self.hyper_params.sigmas[time_steps]
            sigma_t_prev = self.hyper_params.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
            sigma_diff = torch.sqrt(torch.clamp(sigma_t ** 2 - sigma_t_prev ** 2, min=0))
            x0 = x0 + noise * sigma_diff.view(-1, 1, 1, 1)

        elif self.method == "vp":
            betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            diffusion = torch.sqrt(betas * dt) * noise
            x0 = x0 + drift + diffusion

        elif self.method == "sub-vp":
            betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
            cum_betas = self.hyper_params.cum_betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            diffusion = torch.sqrt(betas * (1 - torch.exp(-2 * cum_betas)) * dt) * noise
            x0 = x0 + drift + diffusion

        elif self.method == "ode":
            if self.method == "ve":
                x0 = x0
            else:
                betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
                drift = -0.5 * betas * x0 * dt
                x0 = x0 + drift
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return x0