import torch



class ForwardDDPM:
    """
    Implements the forward diffusion process for Denoising Diffusion Probabilistic Models (DDPM).

    This class precomputes the noise schedule and provides a method to add Gaussian noise to images
    at different time steps in the diffusion process.

    Attributes:
        num_steps (int): The number of diffusion steps.
        beta_start (float): The starting value of the noise variance (beta).
        beta_end (float): The ending value of the noise variance (beta).
        alpha_bars_sqrt (torch.Tensor): Square root of alpha_bars.
        com_alpha_bars_sqrt (torch.Tensor): Square root of (1 - alpha_bars), representing complementary noise levels.
    """

    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        """
        Initializes the ForwardDDPM class by precomputing the noise schedule parameters.

        Args:
            num_steps (int, optional): Total number of diffusion steps. Default is 1000.
            beta_start (float, optional): Initial noise variance. Default is 1e-4.
            beta_end (float, optional): Final noise variance. Default is 0.02.
        """
        params = self.compute_params(num_steps, beta_start, beta_end)
        self.alpha_bars_sqrt = params[0]  # Square root of alpha_bars
        self.com_alpha_bars_sqrt = params[1]  # Square root of (1 - alpha_bars)

        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

    @staticmethod
    def compute_params(num_steps, beta_start, beta_end):
        """
        Computes the noise schedule parameters used in the forward diffusion process.

        Args:
            num_steps (int): Total number of diffusion steps.
            beta_start (float): Initial noise variance.
            beta_end (float): Final noise variance.

        Returns:
            tuple:
                - alpha_bars_sqrt (torch.Tensor): Square root of alpha_bars.
                - com_alpha_bars_sqrt (torch.Tensor): Square root of (1 - alpha_bars).
        """
        betas = torch.linspace(start=beta_start, end=beta_end, steps=num_steps)  # Linear noise schedule
        alphas = 1 - betas  # Noise reduction factors
        alpha_bars = torch.cumprod(alphas, dim=0)  # Cumulative product of alphas
        alpha_bars_sqrt = torch.sqrt(alpha_bars)  # Square root of cumulative product
        com_alpha_bars_sqrt = torch.sqrt(1 - alpha_bars)  # Complementary noise levels

        return alpha_bars_sqrt, com_alpha_bars_sqrt

    def add_noise(self, batch, noise, time_steps):
        """
        Adds Gaussian noise to the input images according to the diffusion process.

        Args:
            batch (torch.Tensor): Batch of images with shape (batch_size, channels, height, width).
            noise (torch.Tensor): Random Gaussian noise of the same shape as batch.
            time_steps (torch.Tensor): Time steps for each image in the batch, shape (batch_size,).

        Returns:
            torch.Tensor: Noisy images after applying the diffusion step.
        """
        # Select precomputed noise factors based on time_steps
        alpha_bar_sqrt_t = self.alpha_bars_sqrt.to(batch.device)[time_steps]
        com_alpha_bar_sqrt_t = self.com_alpha_bars_sqrt.to(batch.device)[time_steps]

        # Reshape tensors for broadcasting across the image batch
        alpha_bar_sqrt_t = alpha_bar_sqrt_t[:, None, None, None]
        com_alpha_bar_sqrt_t = com_alpha_bar_sqrt_t[:, None, None, None]

        # Apply the diffusion formula: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        out = (alpha_bar_sqrt_t * batch) + (com_alpha_bar_sqrt_t * noise)

        return out






