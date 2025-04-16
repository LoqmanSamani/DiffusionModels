import torch
import torch.nn as nn



class ForwardDDIM(nn.Module):
    """Forward diffusion process of DDIM.

    Implements the forward diffusion process for Denoising Diffusion Implicit Models
    (DDIM), which perturbs input data by adding Gaussian noise over a series of time
    steps, as defined in Song et al. (2021, "Denoising Diffusion Implicit Models").

    Parameters
    ----------
    hyper_params : object
        Hyperparameter object containing the noise schedule parameters. Expected to have
        attributes:
        - `num_steps`: Number of diffusion steps (int).
        - `trainable_beta`: Whether the noise schedule is trainable (bool).
        - `betas`: Noise schedule parameters (torch.Tensor, optional if trainable_beta is True).
        - `sqrt_alpha_cumprod`: Precomputed cumulative product of alphas (torch.Tensor, optional if trainable_beta is False).
        - `sqrt_one_minus_alpha_cumprod`: Precomputed square root of one minus cumulative alpha product (torch.Tensor, optional if trainable_beta is False).
        - `compute_schedule`: Method to compute the noise schedule (callable, optional if trainable_beta is True).

    Attributes
    ----------
    hyper_params : object
        Stores the provided hyperparameter object.
    """
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params

    def forward(self, x0, noise, time_steps):
        """Applies the forward diffusion process to the input data.

        Perturbs the input data `x0` by adding Gaussian noise according to the DDIM
        forward process at specified time steps, using cumulative noise schedule
        parameters.

        Parameters
        ----------
        x0 : torch.Tensor
            Input data tensor of shape (batch_size, channels, height, width).
        noise : torch.Tensor
            Gaussian noise tensor of the same shape as `x0`.
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, hyper_params.num_steps - 1].

        Returns
        -------
        torch.Tensor
            Noisy data tensor `xt` at the specified time steps, with the same shape as `x0`.

        Raises
        ------
        ValueError
            If any value in `time_steps` is outside the valid range
            [0, hyper_params.num_steps - 1].
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.hyper_params.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.hyper_params.num_steps - 1}")

        if self.hyper_params.trainable_beta:
            _, _, _, sqrt_alpha_cumprod_t, sqrt_one_minus_alpha_cumprod_t = self.hyper_params.compute_schedule(
                self.hyper_params.betas
            )
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t[time_steps].to(x0.device)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t[time_steps].to(x0.device)
        else:
            sqrt_alpha_cumprod_t = self.hyper_params.sqrt_alpha_cumprod[time_steps].to(x0.device)
            sqrt_one_minus_alpha_cumprod_t = self.hyper_params.sqrt_one_minus_alpha_cumprod[time_steps].to(x0.device)

        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1, 1)

        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt