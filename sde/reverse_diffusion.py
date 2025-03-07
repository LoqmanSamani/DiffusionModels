import torch
from config import Config




class ReverseSDE:
    def __init__(self, config, model=None):
        self.config = config
        if model:
            self.model = model
        else:
            # assume that a model (noise predictor) is attached to the config if not provided directly.
            self.model = self.config.model

    def forward(self, x):

        dt = self.config.step_size
        if self.config.method == "smld":  # SMLD (VE SDE) reverse process
            for i in reversed(range(1, self.config.max_steps)):
                noise = torch.randn_like(x)
                p_noise = self.model(x=x, t=i)  # predicted noise
                sigma_diff = torch.sqrt(self.config.sigmas[i] ** 2 - self.config.sigmas[i - 1] ** 2)
                x = x - sigma_diff * p_noise + sigma_diff * noise

        elif self.config.method == "ddim":  # DDIM (VP SDE) reverse process
            for i in reversed(range(1, self.config.max_steps)):
                p_noise = self.model(x=x, t=i)
                noise = torch.randn_like(x)
                drift = -0.5 * self.config.betas[i] * x * dt - self.config.betas[i] * p_noise * dt
                diffusion = torch.sqrt(self.config.betas[i] * dt) * noise
                x = x + drift + diffusion

        elif self.config.method == "subvp":  # Sub-VP SDE reverse process
            cum_beta = torch.cumsum(self.config.betas, dim=0) * dt
            for i in reversed(range(1, self.config.max_steps)):
                p_noise = self.model(x=x, t=i)
                noise = torch.randn_like(x)
                drift = -0.5 * self.config.betas[i] * x * dt - self.config.betas[i] * (1 - torch.exp(-2 * cum_beta[i])) * p_noise * dt
                diffusion = torch.sqrt(self.config.betas[i] * (1 - torch.exp(-2 * cum_beta[i])) * dt) * noise
                x = x + drift + diffusion

        else:
            raise ValueError(f"unknown method: {self.config.method}")

        return x