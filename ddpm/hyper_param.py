"""Hyperparameters for Denoising Diffusion Probabilistic Models (DDPM) noise schedule.

This module implements a flexible noise schedule for DDPM, as described in Ho et al.
(2020, "Denoising Diffusion Probabilistic Models"). It supports multiple beta schedule
methods and allows for trainable or fixed noise schedules.
"""


import torch
import torch.nn as nn




class HyperParamsDDPM(nn.Module):
    """Hyperparameters for DDPM noise schedule with flexible beta computation.

    Manages the noise schedule parameters for DDPM, including the computation of beta
    values and derived quantities (alphas, alpha_bars, etc.), with support for
    trainable or fixed schedules and various beta scheduling methods, as inspired by
    Ho et al. (2020).

    Parameters
    ----------
    num_steps : int, optional
        Number of diffusion steps (default: 1000).
    beta_start : float, optional
        Starting value for beta (default: 1e-4).
    beta_end : float, optional
        Ending value for beta (default: 0.02).
    trainable_beta : bool, optional
        Whether the beta schedule is trainable (default: False).
    beta_method : str, optional
        Method for computing the beta schedule (default: "linear").
        Supported methods: "linear", "sigmoid", "quadratic", "constant", "inverse_time".

    Attributes
    ----------
    num_steps : int
        Number of diffusion steps.
    beta_start : float
        Minimum beta value.
    beta_end : float
        Maximum beta value.
    trainable_beta : bool
        Whether the beta schedule is trainable.
    beta_method : str
        Method used for beta schedule computation.
    betas : torch.Tensor
        Beta schedule values, shape (num_steps,). Trainable if `trainable_beta` is True,
        otherwise a fixed buffer.
    alphas : torch.Tensor, optional
        Alpha values (1 - betas), shape (num_steps,). Available if `trainable_beta` is False.
    alpha_bars : torch.Tensor, optional
        Cumulative product of alphas, shape (num_steps,). Available if `trainable_beta` is False.
    sqrt_alpha_bars : torch.Tensor, optional
        Square root of alpha_bars, shape (num_steps,). Available if `trainable_beta` is False.
    sqrt_one_minus_alpha_bars : torch.Tensor, optional
        Square root of (1 - alpha_bars), shape (num_steps,). Available if `trainable_beta` is False.

    Raises
    ------
    ValueError
        If `beta_start` or `beta_end` do not satisfy 0 < beta_start < beta_end < 1,
        or if `num_steps` is not positive.
    """
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
        """Computes the beta schedule based on the specified method.

        Generates a sequence of beta values for the DDPM noise schedule using the
        chosen method, ensuring values are clamped within the specified range.

        Parameters
        ----------
        beta_range : tuple
            Tuple of (min_beta, max_beta) specifying the valid range for beta values.
        num_steps : int
            Number of diffusion steps.
        method : str
            Method for computing the beta schedule. Supported methods:
            "linear", "sigmoid", "quadratic", "constant", "inverse_time".

        Returns
        -------
        torch.Tensor
            Tensor of beta values, shape (num_steps,).

        Raises
        ------
        ValueError
            If `method` is not one of the supported beta schedule methods.
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
        """Computes noise schedule parameters dynamically from betas.

        Calculates the derived noise schedule parameters (alphas, alpha_bars, etc.)
        from the provided beta values, as used in the DDPM forward and reverse processes.

        Parameters
        ----------
        betas : torch.Tensor
            Tensor of beta values, shape (num_steps,).

        Returns
        -------
        tuple
            A tuple containing:
            - betas: Input beta values, shape (num_steps,).
            - alphas: 1 - betas, shape (num_steps,).
            - alpha_bars: Cumulative product of alphas, shape (num_steps,).
            - sqrt_alpha_bars: Square root of alpha_bars, shape (num_steps,).
            - sqrt_one_minus_alpha_bars: Square root of (1 - alpha_bars), shape (num_steps,).
        """
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars, torch.sqrt(alpha_bars), torch.sqrt(1 - alpha_bars)

    def constrain_betas(self):
        """Constrains trainable betas to a valid range during training.

        Ensures that trainable beta values remain within the specified range
        [beta_start, beta_end] by clamping them in-place.

        Notes
        -----
        This method only applies when `trainable_beta` is True.
        """
        if self.trainable_beta:
            with torch.no_grad():
                self.betas.clamp_(min=self.beta_start, max=self.beta_end)