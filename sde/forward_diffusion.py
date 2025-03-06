import torch



class ForwardSDE:
    def __init__(self, config):
        self.config = config

    def forward(self, x):

        dt = self.config.step_size
        if self.config.method == "smld":  # SMLD (VE SDE)
            for i in range(1, self.config.max_steps):
                noise = torch.randn_like(x)
                sigma_diff = torch.sqrt(self.config.sigmas[i]**2 - self.config.sigmas[i-1]**2)
                x = x + noise * sigma_diff

        elif self.config.method == "ddim":  # DDIM (VP SDE)
            for i in range(1, self.config.max_steps):
                noise = torch.randn_like(x)
                # Euler–Maruyama discretization of: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW
                drift = -0.5 * self.config.betas[i] * x * dt
                diffusion = torch.sqrt(self.config.betas[i]) * noise
                x = x + drift + diffusion

        elif self.config.method == "subvp":  # Sub-VP SDE
            # precompute an approximation of the integral of beta(s) ds using cumulative sum
            # this gives a tensor cum_beta where cum_beta[i] approximates ∫₀^(tᵢ) beta(s) ds.
            cum_beta = torch.cumsum(self.config.betas, dim=0) * dt
            for i in range(1, self.config.max_steps):
                noise = torch.randn_like(x)
                # diffusion term: sqrt(beta(t) * (1 - exp(-2∫₀^(t) beta(s) ds)))
                diffusion = torch.sqrt(self.config.betas[i] * (1 - torch.exp(-2 * cum_beta[i])))
                drift = -0.5 * self.config.betas[i] * x * dt
                x = x + drift + diffusion * noise

        else:
            raise ValueError(f"unknown method: {self.config.method}")
            
        return x

    


