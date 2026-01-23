import torch
import torch.nn as nn



class ReverseSDE(nn.Module):
    """
    Unified reverse-time diffusion process for continuous-time sde diffusion models

    This module implements a single-step numerical solver for the *reverse-time*
    stochastic differential equation (SDE) or probability flow ordinary
    differential equation (ODE) corresponding to a trained score-based model.

    Given a noisy sample x_t at time t and an estimate of the score
    ∇ₓ log p_t(x), the reverse process evolves the system backward in time
    (t → 0) using an Euler–Maruyama discretization.

    Supported reverse dynamics:

        • Variance Preserving (VP-SDE)
        • Variance Exploding (VE-SDE)
        • Sub-Variance Preserving (Sub-VP-SDE)
        • Probability Flow ODE (ODE)

    General reverse SDE form:
        dx = [f(x, t) - g²(t) ∇ₓ log p_t(x)] dt + g(t) dW̄_t

    where:
        • f(x, t) is the forward drift
        • g(t) is the diffusion coefficient
        • dW̄_t denotes reverse-time Brownian motion

    For the probability flow ODE, the diffusion term vanishes and the dynamics
    become deterministic while preserving the same marginals as the VP-SDE.

    Parameters
    ----------
    scheduler : nn.Module
        Scheduler providing β(t) and related quantities for VP and Sub-VP
        dynamics. Typically an instance of `SchedulerSDE`.

    method : str, default="vp"
        Type of reverse-time dynamics. Must be one of:
        {"vp", "ve", "sub-vp", "ode"}.

    sigma_min : float, default=0.01
        Minimum noise scale for the VE-SDE.

    sigma_max : float, default=50.0
        Maximum noise scale for the VE-SDE.

    Notes
    -----
    • Time t is assumed to be normalized to [0, 1].
    • Reverse integration proceeds with a *negative* time step dt < 0.
    • The score ∇ₓ log p_t(x) is typically predicted by a neural network.
    • For the final step or ODE-based sampling, stochastic noise can be disabled.
    • All tensor operations support broadcasting over arbitrary data shapes.

    Numerical Integration
    ---------------------
    The update rule implemented is the Euler–Maruyama scheme:

        x_{t+dt} = x_t
                   + [f(x_t, t) - g²(t)·score(x_t, t)] dt
                   + g(t) √|dt| ε

    where ε ~ N(0, I). For ODE sampling, the stochastic term is omitted.

    References
    ----------
    - Anderson, "Reverse-Time Diffusion Equation Models", 1982
    - Song et al., "Score-Based Generative Modeling through SDEs", ICLR 2021
    - Kingma et al., "Variational Diffusion Models", NeurIPS 2021
    """
    def __init__(
            self,
            scheduler: nn.Module,
            method: str = "vp",
            sigma_min: float = 0.01,
            sigma_max: float = 50.0,
            eps: float = 1e-8
    ):
        super().__init__()

        valid_methods = ["vp", "ve", "sub-vp", "ode"]
        if method not in valid_methods:
            raise ValueError(f"sde_method must be one of {valid_methods}, got {method}")

        self.vs = scheduler
        self.method = method
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = eps

    def _broadcast_to_shape(self, tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Broadcast tensor to target shape by adding trailing dimensions"""
        while tensor.dim() < len(target_shape):
            tensor = tensor.unsqueeze(-1)
        return tensor

    def get_reverse_coeffs(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get drift and diffusion coefficients for reverse SDE

        Returns:
            drift_coeff: coefficient for drift term
            g_squared: squared diffusion coefficient (for score term)
            diffusion_coeff: coefficient for diffusion term
        """
        if self.method == "vp":
            # vp-sde: dx = [-½β(t)x - β(t)∇log p_t(x)]dt + √β(t)dw̄
            drift_coeff = -0.5 * self.vs.beta(t)
            g_squared = self.vs.beta(t)
            diffusion_coeff = torch.sqrt(self.vs.beta(t))

        elif self.method == "ve":
            # ve-sde: dx = [-σ(t)dσ/dt ∇log p_t(x)]dt + √(2σ(t)dσ/dt)dw̄
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            dsigma_dt = sigma_t * torch.log(torch.tensor(self.sigma_max / self.sigma_min))
            drift_coeff = torch.zeros_like(t)
            g_squared = 2 * sigma_t * dsigma_dt
            diffusion_coeff = torch.sqrt(g_squared)

        elif self.method == "sub-vp":
            # sub-vp-sde: dx = [-β(t)∇log p_t(x)]dt + √β(t)dw̄
            drift_coeff = torch.zeros_like(t)
            g_squared = self.vs.beta(t)
            diffusion_coeff = torch.sqrt(self.vs.beta(t))

        elif self.method == "ode":
            # probability flow ode: deterministic
            drift_coeff = -0.5 * self.vs.beta(t)
            g_squared = self.vs.beta(t)
            diffusion_coeff = torch.zeros_like(t) # no diffusion in ode

        return drift_coeff, g_squared, diffusion_coeff

    def forward(self, xt: torch.Tensor, score: torch.Tensor, t: torch.Tensor, dt: float, last_step: bool = False) -> torch.Tensor:
        """Single reverse Euler-Maruyama step
        Args:
            xt: (batch, ..., dims) current state
            score: (batch, ..., dims) score estimate ∇_x log p_t(x)
            t: (batch,) current time
            dt: scalar time step (negative for reverse)
            last_step: if True, skip noise for deterministic final step

        Returns:
            x_prev: (batch, ..., dims) previous state
        """
        if not torch.is_tensor(dt):
            assert dt < 0.0, "dt must be negative for reverse diffusion!"
            dt = torch.tensor(dt, device=xt.device, dtype=xt.dtype)

        drift_coeff, g_squared, diffusion_coeff = self.get_reverse_coeffs(t)
        # broadcast to match xt shape
        drift_coeff = self._broadcast_to_shape(drift_coeff, xt.shape)
        g_squared = self._broadcast_to_shape(g_squared, xt.shape)
        diffusion_coeff = self._broadcast_to_shape(diffusion_coeff, xt.shape)
        # [-½β(t)x - β(t)∇log p_t(x)]dt + √β(t)dw̄
        # reverse drift: f(x,t) - g²(t)·score
        drift = drift_coeff * xt - g_squared * score
        # diffusion term
        if last_step or self.method == "ode":
            noise = torch.zeros_like(xt)
        else:
            noise = torch.randn_like(xt)
        diffusion = diffusion_coeff * noise
        # Euler-Maruyama step
        x_prev = xt + drift * dt + diffusion * torch.sqrt(torch.abs(dt))
        return x_prev