import torch




class ReverseSDE:
    def __init__(self, config, model=None):
        """
        reverse sde simulates the denoising process for different noise models:
        smld (variance exploding sde), ddim (variance preserving sde), and subvp (sub-variance preserving sde).

        attributes:
            config (Config): configuration object containing hyperparameters for the sde.
            model (callable): a function or neural network that predicts noise given x and t.
        """
        self.config = config
        if model:
            self.model = model
        else:
            # assume that a model (noise predictor) is attached to the config if not provided directly.
            self.model = self.config.model

    def forward(self, x, t):
        """
        applies one step of the reverse denoising process at a given time step t.

        args:
            x (torch.Tensor): input tensor of shape (batch_size, channels, height, width), representing a noisy sample.
            t (torch.Tensor): tensor of shape (batch_size,) representing the current time step for each image.

        returns:
            torch.Tensor: tensor of the same shape as x after applying one step of the reverse sde.
        """
        dt = self.config.step_size
        betas = self.config.betas[t][:, None, None, None]
        cum_beta = torch.cumsum(self.config.betas, dim=0) * dt
        cum_beta = cum_beta[t][:, None, None, None]

        if self.config.method == "smld":  # SMLD (VE SDE) reverse process

            noise = torch.randn_like(x)
            p_noise = self.model(x=x, t=t)  # predicted noise
            sigma_diff = torch.sqrt(self.config.sigmas[t] ** 2 - self.config.sigmas[t-1] ** 2)
            x = x - sigma_diff[:, None, None, None] * p_noise + sigma_diff[:, None, None, None] * noise

        elif self.config.method == "ddim":  # DDIM (VP SDE) reverse process
            p_noise = self.model(x=x, t=t)
            noise = torch.randn_like(x)
            drift = -0.5 * betas * x * dt - betas * p_noise * dt
            diffusion = torch.sqrt(betas * dt) * noise
            x = x + drift + diffusion

        elif self.config.method == "subvp":  # Sub-VP SDE reverse process


            p_noise = self.model(x=x, t=t)
            noise = torch.randn_like(x)
            drift = -0.5 * betas * x * dt - betas * (1 - torch.exp(-2 * cum_beta)) * p_noise * dt
            diffusion = torch.sqrt(betas * (1 - torch.exp(-2 * cum_beta)) * dt) * noise
            x = x + drift + diffusion

        else:
            raise ValueError(f"unknown method: {self.config.method}")

        return x