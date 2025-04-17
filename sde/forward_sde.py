import torch
import torch.nn as nn



class ForwardSDE(nn.Module):
    """Forward diffusion process for SDE-based generative models.

    Implements the forward diffusion process for score-based generative models using
    Stochastic Differential Equations (SDEs), supporting Variance Exploding (VE),
    Variance Preserving (VP), sub-Variance Preserving (sub-VP), and ODE methods, as
    described in Song et al. (2021).

    Parameters
    ----------
    hyper_params : object
        Hyperparameter object containing SDE-specific parameters. Expected to have
        attributes:
        - `dt`: Time step size for SDE integration (float).
        - `sigmas`: Sigma values for VE method (torch.Tensor, optional).
        - `betas`: Beta values for VP, sub-VP, or ODE methods (torch.Tensor).
        - `cum_betas`: Cumulative beta values for sub-VP method (torch.Tensor, optional).
    method : str
        SDE method to use. Supported methods: "ve", "vp", "sub-vp", "ode".

    Attributes
    ----------
    hyper_params : object
        Stores the provided hyperparameter object.
    method : str
        Selected SDE method.

    Raises
    ------
    ValueError
        If `method` is not one of the supported methods ("ve", "vp", "sub-vp", "ode").
    """
    def __init__(self, hyper_params, method):
        super().__init__()
        self.hyper_params = hyper_params
        self.method = method

    def forward(self, x0, noise, time_steps):
        """Applies the forward SDE diffusion process to the input data.

        Perturbs the input data `x0` by adding noise according to the specified SDE
        method at given time steps, incorporating drift and diffusion terms as applicable.

        Parameters
        ----------
        x0 : torch.Tensor
            Input data tensor, shape (batch_size, channels, height, width).
        noise : torch.Tensor
            Gaussian noise tensor, same shape as `x0`.
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, hyper_params.num_steps - 1].

        Returns
        -------
        torch.Tensor
            Noisy data tensor at the specified time steps, same shape as `x0`.

        Raises
        ------
        ValueError
            If `method` is not one of the supported methods ("ve", "vp", "sub-vp", "ode").
        """
        dt = self.hyper_params.dt
        if self.method == "ve":
            sigma_t = self.hyper_params.sigmas[time_steps]
            sigma_t_prev = self.hyper_params.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
            sigma_diff = torch.sqrt(torch.clamp(sigma_t ** 2 - sigma_t_prev ** 2, min=0))
            x0 = x0 + noise * sigma_diff.view(-1, 1, 1, 1)

        elif self.method == "vp":
            betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            diffusion = torch.sqrt(betas * dt) * noise
            x0 = x0 + drift + diffusion

        elif self.method == "sub-vp":
            betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
            cum_betas = self.hyper_params.cum_betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            diffusion = torch.sqrt(betas * (1 - torch.exp(-2 * cum_betas)) * dt) * noise
            x0 = x0 + drift + diffusion

        elif self.method == "ode":
            if self.method == "ve":
                x0 = x0
            else:
                betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
                drift = -0.5 * betas * x0 * dt
                x0 = x0 + drift
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return x0