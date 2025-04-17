import torch
import torch.nn as nn




class ReverseSDE(nn.Module):
    """Reverse diffusion process for SDE-based generative models.

    Implements the reverse diffusion process for score-based generative models using
    Stochastic Differential Equations (SDEs), supporting Variance Exploding (VE),
    Variance Preserving (VP), sub-Variance Preserving (sub-VP), and ODE methods, as
    described in Song et al. (2021). The reverse process denoises a noisy input using
    predicted noise estimates.

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

    def forward(self, xt, noise, predicted_noise, time_steps):
        """Applies the reverse SDE diffusion process to the noisy input.

        Denoises the input `xt` by applying the reverse SDE process, using predicted
        noise estimates and optional stochastic noise, according to the specified SDE
        method at given time steps. Incorporates drift and diffusion terms as applicable.

        Parameters
        ----------
        xt : torch.Tensor
            Noisy input tensor at time step `t`, shape (batch_size, channels, height, width).
        noise : torch.Tensor or None
            Gaussian noise tensor, same shape as `xt`, used for stochasticity. If None,
            no stochastic noise is added (e.g., for deterministic ODE).
        predicted_noise : torch.Tensor
            Predicted noise tensor, same shape as `xt`, typically output by a neural network.
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, hyper_params.num_steps - 1].

        Returns
        -------
        torch.Tensor
            Denoised tensor at the previous time step, same shape as `xt`.

        Raises
        ------
        ValueError
            If `method` is not one of the supported methods ("ve", "vp", "sub-vp", "ode").

        Notes
        -----
        - For the "ve" and "ode" methods, the output is clamped to [-1e5, 1e5] to prevent
          numerical instability.
        - Stochastic noise (`noise`) is only added if provided and the method supports it
          (not applicable for "ode" in non-VE cases).
        """
        dt = self.hyper_params.dt
        betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
        cum_betas = self.hyper_params.cum_betas[time_steps].view(-1, 1, 1, 1)
        if self.method == "ve":
            sigma_t = self.hyper_params.sigmas[time_steps]
            sigma_t_prev = self.hyper_params.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
            sigma_diff = torch.sqrt(torch.clamp(sigma_t ** 2 - sigma_t_prev ** 2, min=0))
            drift = -(sigma_t ** 2 - sigma_t_prev ** 2).view(-1, 1, 1, 1) * predicted_noise * dt
            diffusion = sigma_diff.view(-1, 1, 1, 1) * noise if noise is not None else 0
            xt = xt + drift + diffusion
            xt = torch.clamp(xt, -1e5, 1e5)

        elif self.method == "vp":
            drift = -0.5 * betas * xt * dt - betas * predicted_noise * dt
            diffusion = torch.sqrt(betas * dt) * noise if noise is not None else 0
            xt = xt + drift + diffusion

        elif self.method == "sub-vp":
            drift = -0.5 * betas * xt * dt - betas * (1 - torch.exp(-2 * cum_betas)) * predicted_noise * dt
            diffusion = torch.sqrt(betas * (1 - torch.exp(-2 * cum_betas)) * dt) * noise if noise is not None else 0
            xt = xt + drift + diffusion

        elif self.method == "ode":
            if self.method == "ve":
                sigma_t = self.hyper_params.sigmas[time_steps]
                sigma_t_prev = self.hyper_params.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
                drift = -0.5 * (sigma_t ** 2 - sigma_t_prev ** 2).view(-1, 1, 1, 1) * predicted_noise * dt
            else:
                drift = -0.5 * betas * xt * dt - 0.5 * betas * predicted_noise * dt
            xt = xt + drift
            xt = torch.clamp(xt, -1e5, 1e5)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return xt