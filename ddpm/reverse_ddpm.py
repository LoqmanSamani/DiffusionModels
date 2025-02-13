import torch



class ReverseDDPM:
    """
    Implements the reverse diffusion process of a Denoising Diffusion Probabilistic Model (DDPM).

    This class is responsible for removing noise from an image that has undergone a diffusion process.
    It reconstructs the original image (`x0`) and predicts the denoised image at the previous time step (`xt-1`).

    Attributes:
        num_steps (int): Total number of diffusion steps.
        beta_start (float): Initial noise variance in the schedule.
        beta_end (float): Final noise variance in the schedule.
        betas (torch.Tensor): Linearly spaced noise variances.
        alphas (torch.Tensor): Noise reduction factors (1 - betas).
        alpha_bars (torch.Tensor): Cumulative product of alphas, representing retained information.
    """

    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        """
        Initializes the ReverseDDPM class and precomputes noise schedule parameters.

        Args:
            num_steps (int, optional): Number of diffusion steps (default: 1000).
            beta_start (float, optional): Initial noise variance (default: 1e-4).
            beta_end (float, optional): Final noise variance (default: 0.02).
        """
        params = self.compute_params(num_steps, beta_start, beta_end)
        self.betas = params[0]  # Noise variances
        self.alphas = params[1]  # (1 - betas)
        self.alpha_bars = params[2]  # Cumulative product of alphas

        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

    @staticmethod
    def compute_params(num_steps, beta_start, beta_end):
        """
        Computes the noise schedule parameters used in the diffusion process.

        Args:
            num_steps (int): Total number of diffusion steps.
            beta_start (float): Initial noise variance.
            beta_end (float): Final noise variance.

        Returns:
            tuple:
                - betas (torch.Tensor): Linearly spaced noise variances.
                - alphas (torch.Tensor): 1 - betas.
                - alpha_bars (torch.Tensor): Cumulative product of alphas.
        """
        betas = torch.linspace(start=beta_start, end=beta_end, steps=num_steps)  # Linear noise schedule
        alphas = 1 - betas  # Noise reduction factors
        alpha_bars = torch.cumprod(alphas, dim=0)  # Cumulative product of alphas

        return betas, alphas, alpha_bars

    def remove_noise(self, batch_t, predicted_noise, time_step):
        """
        Performs one step of the reverse diffusion process to estimate the original image
        and predict the previous time step image.

        Args:
            batch_t (torch.Tensor): The noisy image at time step `t`.
            predicted_noise (torch.Tensor): The estimated noise component.
            time_step (int): The current time step.

        Returns:
            tuple:
                - predicted (torch.Tensor): The denoised image at time step `t-1`.
                - batch0 (torch.Tensor): The estimated original image.
        """
        # Estimate the original clean image x0 using the DDPM formula
        batch0 = (batch_t - (torch.sqrt(1 - self.alpha_bars.to(batch_t.device)[time_step])) * predicted_noise) / \
                 (torch.sqrt(self.alpha_bars.to(batch_t.device)[time_step]))
        batch0 = torch.clamp(batch0, min=-1.0, max=1.0)  # Clamp values to [-1,1]

        # Alternative formulation for x0 (slightly different)
        predicted0 = (batch_t - ((1 - self.alphas.to(batch_t.device)[time_step]) * predicted_noise) /
                     (torch.sqrt(1 - self.alpha_bars.to(batch_t.device)[time_step]))) / \
                     (torch.sqrt(self.alphas.to(batch_t.device)[time_step]))

        # If t=0, return x0 since we don’t predict earlier steps
        if time_step == 0:
            return predicted0, batch0

        # Compute the variance term for adding noise
        var = (1 - self.alpha_bars.to(batch_t.device)[time_step - 1]) / (1 - self.alpha_bars.to(batch_t.device)[time_step])
        var = var * self.betas.to(batch_t.device)[time_step]
        std = var ** 0.5  # Standard deviation

        # Sample Gaussian noise
        z = torch.randn(batch_t.shape).to(batch_t.device)

        # Predict the next image x_{t-1}
        predicted = predicted0 + std * z

        return predicted, batch0











