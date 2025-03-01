import torch



class ForwardDDPM:
    """
    forward diffusion process of the  "Denoising Diffusion Probabilistic Models (DDPM)".
    the class precomputes the noise schedule and provides a method to add Gaussian noise to images
    at different time steps in the diffusion process.
    """

    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):

        params = self.compute_params(num_steps, beta_start, beta_end)
        self.alpha_bars_sqrt = params[0]  # square root of alpha_bars
        self.com_alpha_bars_sqrt = params[1]  # square root of (1 - alpha_bars)
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

    @staticmethod
    def compute_params(num_steps, beta_start, beta_end):
        """
        computes the noise schedule parameters used in the forward diffusion process.
        """
        betas = torch.linspace(start=beta_start, end=beta_end, steps=num_steps)  # Linear noise schedule
        alphas = 1 - betas  # noise reduction factors
        alpha_bars = torch.cumprod(alphas, dim=0)  # cumulative product of alphas
        alpha_bars_sqrt = torch.sqrt(alpha_bars)  # square root of cumulative product
        com_alpha_bars_sqrt = torch.sqrt(1 - alpha_bars)  # complementary noise levels

        return alpha_bars_sqrt, com_alpha_bars_sqrt

    def add_noise(self, batch, noise, time_steps):
        """
        adds Gaussian noise to the input images according to the diffusion process.
        """
        # select precomputed noise factors based on time_steps
        alpha_bar_sqrt_t = self.alpha_bars_sqrt.to(batch.device)[time_steps]
        com_alpha_bar_sqrt_t = self.com_alpha_bars_sqrt.to(batch.device)[time_steps]

        # reshape tensors for broadcasting across the image batch
        alpha_bar_sqrt_t = alpha_bar_sqrt_t[:, None, None, None]
        com_alpha_bar_sqrt_t = com_alpha_bar_sqrt_t[:, None, None, None]

        # apply the diffusion formula: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        out = (alpha_bar_sqrt_t * batch) + (com_alpha_bar_sqrt_t * noise)

        return out
