import torch
import torch.nn as nn



class ForwardSDE(nn.Module):
    """
    Unified forward diffusion process for continuous-time diffusion models.

    This module implements the marginal forward noising process
    p(x_t | x_0) for several commonly used stochastic differential equation
    (SDE) formulations, including:

        • Variance Preserving (VP-SDE)
        • Variance Exploding (VE-SDE)
        • Sub-Variance Preserving (Sub-VP-SDE)
        • Probability Flow ODE (ODE)

    Given clean data x₀, Gaussian noise ε ~ N(0, I), and continuous time
    t ∈ [0, 1], the forward process samples x_t and provides the *true score*
    ∇ₓ log p(x_t | x₀), which is commonly used for score matching objectives.

    Supported forward marginals:

    1. VP-SDE:
        p(x_t | x_0) = N(α(t) x_0, σ²(t) I)

    2. VE-SDE:
        p(x_t | x_0) = N(x_0, σ²(t) I),
        where σ(t) = σ_min (σ_max / σ_min)^t

    3. Sub-VP-SDE:
        p(x_t | x_0) = N(x_0, σ²(t) I),
        where σ²(t) = 1 - exp(-∫₀ᵗ β(s) ds)

    4. Probability Flow ODE:
        Shares the same marginals as VP-SDE but corresponds to a
        deterministic dynamics during sampling.

    The returned score is analytically computed as:

        ∇ₓ log p(x_t | x₀) = -(x_t - μ(t)) / σ²(t) = -ε / σ(t)

    where μ(t) is the mean of the forward transition.

    Parameters
    ----------
    scheduler : SchedulerSDE
        Scheduler providing β(t), α(t), and σ(t) for VP and Sub-VP processes.

    method : str, default="vp"
        Forward process type. Must be one of:
        {"vp", "ve", "sub-vp", "ode"}.

    sigma_min : float, default=0.01
        Minimum noise scale for the VE-SDE.

    sigma_max : float, default=50.0
        Maximum noise scale for the VE-SDE.

    eps : float, default=1e-8
        Small constant for numerical stability when computing the score.

    Notes
    -----
    • Time t is assumed to be normalized to [0, 1].
    • All operations are vectorized and support arbitrary data dimensions.
    • Broadcasting is handled automatically to match the shape of x₀.
    • For the ODE method, noise is still used to compute the analytical
      score during training, even though sampling is deterministic.

    References
    ----------
    - Song et al., "Score-Based Generative Modeling through SDEs", ICLR 2021
    - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
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

    def get_forward_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean coefficient and std for the forward process based on SDE method
        Returns:
            mean_coeff: coefficient for clean data x_0
            std: standard deviation of noise
        """
        mean_coeff = None
        std = None
        if self.method == "vp":
            # vp-sde: p(x_t | x_0) = N(α(t)x_0, σ²(t)I)
            mean_coeff = self.vs.alpha(t)
            std = self.vs.std(t)

        elif self.method == "ve":
            # ve-sde: p(x_t | x_0) = N(x_0, σ²(t)I)
            # σ(t) grows from sigma_min to sigma_max
            mean_coeff = torch.ones_like(t)
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            std = sigma_t

        elif self.method == "sub-vp":
            # sub-vp-sde: p(x_t | x_0) = N(x_0, σ²(t)I) where σ²(t) = 1 - e^(-∫β(s)ds)
            mean_coeff = torch.ones_like(t)
            std = self.vs.std(t)

        elif self.method == "ode":
            # probability flow ode: same marginals as vp-sde but deterministic
            mean_coeff = self.vs.alpha(t)
            std = self.vs.std(t)

        return mean_coeff, std

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from transition kernel and compute true score

        Arguments:
            x0: (batch, ..., dims) clean data
            noise: (batch, ..., dims) standard Gaussian noise
            t: (batch, ) continuous time in [0, 1]

        Returns:
            xt: (batch, ..., dims) noised data
            score: (batch, ..., dims) true score ∇_x log p(x_t | x_0)
        """
        mean_coeff, std = self.get_forward_params(t)
        # broadcast to match x0 shape
        mean_coeff = self._broadcast_to_shape(mean_coeff, x0.shape)
        std = self._broadcast_to_shape(std, x0.shape)
        # x_t = mean_coeff * x_0 + std * ε
        xt = mean_coeff * x0 + std * noise
        # ∇_x log p(x_t | x_0) = -(x_t - mean_coeff*x_0) / σ²(t) = -ε / σ(t)
        score = -noise / (std + self.eps)
        return xt, score