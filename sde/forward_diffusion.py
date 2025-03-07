import torch



class ForwardSDE:
    def __init__(self, config):
        """
        forward sde simulates the diffusion process for different noise models:
        smld (variance exploding sde), ddim (variance preserving sde), and subvp (sub-variance preserving sde).

        attributes:
            config (Config): configuration object containing hyperparameters for the sde.
        """
        self.config = config

    def forward(self, x, t):
        """
        applies one step of the forward diffusion process at a given time step t.

        args:
            x (torch.Tensor): input tensor of shape (batch_size, channels, height, width)
            t (torch.Tensor): tensor of shape (batch_size,) representing the current time step for each image

        returns:
            torch.Tensor: tensor of the same shape as x after applying the forward sde step
        """

        dt = self.config.step_size
        if self.config.method == "smld":  # SMLD (VE SDE)
            noise = torch.randn_like(x)
            # sigma_diff from discretization
            sigma_diff = torch.sqrt(self.config.sigmas[t] ** 2 - self.config.sigmas[t-1] ** 2)
            x = x + noise * sigma_diff[:, None, None, None]

        elif self.config.method == "ddim":  # DDIM (VP SDE)

            noise = torch.randn_like(x)
            # drift term: Euler-Maruyama discretization
            drift = -0.5 * self.config.betas[t][:, None, None, None] * x * dt
            # diffusion term
            diffusion = torch.sqrt(self.config.betas[t][:, None, None, None] * dt) * noise
            x = x + drift + diffusion

        elif self.config.method == "subvp":  # Sub-VP SDE
            # cumulative beta for the time integral
            cum_betas = self.config.cum_betas[t][:, None, None, None]
            noise = torch.randn_like(x)
            drift = -0.5 * self.config.betas[t][:, None, None, None] * x * dt
            # diffusion term
            diffusion = torch.sqrt(self.config.betas[t][:, None, None, None] * (1 - torch.exp(-2 * cum_betas)) * dt) * noise
            x = x + drift + diffusion

        else:
            raise ValueError(f"unknown method: {self.config.method}")

        return x