import torch




class ForwardSDE:
    def __init__(self, config):
        self.config = config

    def forward(self, x):

        dt = self.config.step_size
        if self.config.method == "smld":  # SMLD (VE SDE)
            for i in range(1, self.config.max_steps):
                noise = torch.randn_like(x)
                # sigma_diff from discretization
                sigma_diff = torch.sqrt(self.config.sigmas[i] ** 2 - self.config.sigmas[i - 1] ** 2)
                x = x + noise * sigma_diff

        elif self.config.method == "ddim":  # DDIM (VP SDE)
            for i in range(1, self.config.max_steps):
                noise = torch.randn_like(x)
                # Drift term: Euler-Maruyama discretization
                drift = -0.5 * self.config.betas[i] * x * dt
                # diffusion term
                diffusion = torch.sqrt(self.config.betas[i] * dt) * noise
                x = x + drift + diffusion

        elif self.config.method == "subvp":  # Sub-VP SDE
            # cumulative beta for the time integral
            cum_betas = self.config.cum_betas
            for i in range(1, self.config.max_steps):
                noise = torch.randn_like(x)
                drift = -0.5 * self.config.betas[i] * x * dt
                # diffusion term
                diffusion = torch.sqrt(self.config.betas[i] * (1 - torch.exp(-2 * cum_betas[i])) * dt) * noise
                x = x + drift + diffusion

        else:
            raise ValueError(f"unknown method: {self.config.method}")

        return x