import torch
import torch.nn as nn





class HyperParamsDDPM(nn.Module):
    """hyperparameters for DDPM noise schedule with flexible beta computation."""
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, trainable_beta=False, beta_method="linear"):
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trainable_beta = trainable_beta
        self.beta_method = beta_method

        # validate inputs
        if not (0 < beta_start < beta_end < 1):
            raise ValueError(f"beta_start ({beta_start}) and beta_end ({beta_end}) must satisfy 0 < start < end < 1")
        if num_steps <= 0:
            raise ValueError(f"num_steps ({num_steps}) must be positive")

        # compute initial beta schedule
        beta_range = (beta_start, beta_end)
        betas_init = self.compute_beta_schedule(beta_range, num_steps, beta_method)

        # initialize betas
        if trainable_beta:
            self.betas = nn.Parameter(betas_init)  # Trainable parameter
        else:
            self.register_buffer('betas', betas_init)  # Fixed buffer
            self.register_buffer('alphas', 1 - self.betas)
            self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
            self.register_buffer('sqrt_alpha_bars', torch.sqrt(self.alpha_bars))
            self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1 - self.alpha_bars))

    def compute_beta_schedule(self, beta_range, num_steps, method):
        """
        Computes the beta schedule based on the selected method.
        Args:
            beta_range: Tuple of (min_beta, max_beta) values
            num_steps: Number of diffusion steps
            method: Method for beta schedule ("linear", "sigmoid", "quadratic", "constant", "inverse_time")
        Returns:
            Tensor of beta values, shape [num_steps]
        """
        beta_min, beta_max = beta_range
        if method == "sigmoid":
            x = torch.linspace(-6, 6, num_steps)
            beta = torch.sigmoid(x) * (beta_max - beta_min) + beta_min
        elif method == "quadratic":
            x = torch.linspace(beta_min**0.5, beta_max**0.5, num_steps)
            beta = x**2
        elif method == "constant":
            beta = torch.full((num_steps,), beta_max)
        elif method == "inverse_time":
            beta = 1.0 / torch.linspace(num_steps, 1, num_steps)
            # scale to beta_range
            beta = beta_min + (beta_max - beta_min) * (beta - beta.min()) / (beta.max() - beta.min())
        elif method == "linear":
            beta = torch.linspace(beta_min, beta_max, num_steps)
        else:
            raise ValueError(f"Unknown beta_method: {method}. Supported: linear, sigmoid, quadratic, constant, inverse_time")

        beta = torch.clamp(beta, min=beta_min, max=beta_max)
        return beta

    @staticmethod
    def compute_schedule(betas):
        """computes noise schedule parameters dynamically from betas."""
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars, torch.sqrt(alpha_bars), torch.sqrt(1 - alpha_bars)

    def constrain_betas(self):
        """constrains betas to a valid range during training."""
        if self.trainable_beta:
            with torch.no_grad():
                self.betas.clamp_(min=self.beta_start, max=self.beta_end)