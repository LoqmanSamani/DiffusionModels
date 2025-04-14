import torch
import torch.nn as nn



class ForwardDDIM(nn.Module):
    def __init__(self, hyper_params):
        """forward diffusion process of DDIM"""
        super().__init__()
        self.hyper_params = hyper_params

    def forward(self, x0, noise, time_steps):
        if not torch.all((time_steps >= 0) & (time_steps < self.hyper_params.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.hyper_params.num_steps - 1}")

        if self.hyper_params.trainable_beta:
            _, _, _, sqrt_alpha_cumprod_t, sqrt_one_minus_alpha_cumprod_t = self.hyper_params.compute_schedule(
                self.hyper_params.betas
            )
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t[time_steps].to(x0.device)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t[time_steps].to(x0.device)
        else:
            sqrt_alpha_cumprod_t = self.hyper_params.sqrt_alpha_cumprod[time_steps].to(x0.device)
            sqrt_one_minus_alpha_cumprod_t = self.hyper_params.sqrt_one_minus_alpha_cumprod[time_steps].to(x0.device)

        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1, 1)

        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt


