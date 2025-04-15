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




class ReverseDDIM(nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params

    def forward(self, xt, predicted_noise, time_steps, prev_time_steps):

        if not torch.all((time_steps >= 0) & (time_steps < self.hyper_params.tau_num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.hyper_params.tau_num_steps - 1}")
        if not torch.all((prev_time_steps >= 0) & (prev_time_steps < self.hyper_params.tau_num_steps)):
            raise ValueError(f"prev_time_steps must be between 0 and {self.hyper_params.tau_num_steps - 1}")

        _, _, _, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod = self.hyper_params.get_tau_schedule()
        tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
        tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)

        eta = self.hyper_params.eta
        x0 = (xt - tau_sqrt_one_minus_alpha_cumprod_t * predicted_noise) / tau_sqrt_alpha_cumprod_t
        noise_coeff = eta * ((tau_sqrt_one_minus_alpha_cumprod_t / prev_tau_sqrt_alpha_cumprod_t) *
                             prev_tau_sqrt_one_minus_alpha_cumprod_t / torch.clamp(tau_sqrt_one_minus_alpha_cumprod_t, min=1e-8))
        direction_coeff = torch.clamp(prev_tau_sqrt_one_minus_alpha_cumprod_t ** 2 - noise_coeff ** 2, min=1e-8).sqrt()
        xt_prev = prev_tau_sqrt_alpha_cumprod_t * x0 + noise_coeff * torch.randn_like(xt) + direction_coeff * predicted_noise

        return xt_prev, x0



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