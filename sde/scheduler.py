import torch
import torch.nn as nn



class SchedulerSDE(nn.Module):
    """
    Continuous-time variance (noise) scheduler for diffusion models formulated
    as stochastic differential equations (SDEs).

    This class defines the time-dependent noise rate β(t) and its derived
    quantities used in forward diffusion processes of the form:

        dx = -½ β(t) x dt + √β(t) dW_t

    where t ∈ [0, 1] is continuous time and W_t is standard Brownian motion.

    Supported schedules:
        • Linear schedule:
            β(t) = β_min + t (β_max - β_min)

        • Cosine schedule:
            Defined implicitly via the cumulative signal power
            ᾱ(t) = cos²((t + s) / (1 + s) · π / 2),
            following Nichol & Dhariwal (2021).

    The scheduler provides convenient access to commonly used quantities:
        • β(t)             — instantaneous noise rate
        • ∫₀ᵗ β(s) ds      — cumulative noise
        • α(t)             — signal scaling factor
        • σ²(t)            — noise variance
        • SNR(t)           — signal-to-noise ratio

    All methods operate on PyTorch tensors and support broadcasting.

    Parameters
    ----------
    schedule_type : str, default="linear"
        Type of noise schedule. Must be one of {"linear", "cosine"}.

    beta_min : float, default=0.1
        Minimum noise rate for the linear schedule. Must satisfy
        0 < beta_min < beta_max. Ignored for cosine schedule.

    beta_max : float, default=20.0
        Maximum noise rate for the linear schedule. Ignored for cosine schedule.

    cosine_s : float, default=0.008
        Small offset used in the cosine schedule to prevent singularities
        near t = 0. Matches the formulation from improved DDPMs.

    Notes
    -----
    • Time t is assumed to be normalized to [0, 1].
    • α(t) and σ(t) satisfy:
          α²(t) + σ²(t) = 1
      for both schedules.
    • The cosine schedule defines β(t) implicitly through α²(t); the β(t)
      returned in this case is an approximation derived from finite differences.

    References
    ----------
    - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
    - Song et al., "Score-Based Generative Modeling through SDEs", ICLR 2021
    - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021
    """
    def __init__(
            self,
            schedule_type: str = "linear",
            beta_min: float = 0.1,
            beta_max: float = 20.0,
            cosine_s: float = 0.008
    ):
        super().__init__()
        valid_schedules = ["linear", "cosine"]
        if schedule_type not in valid_schedules:
            raise ValueError(f"schedule_type must be one of {valid_schedules}, got {schedule_type}")

        self.schedule_type = schedule_type
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cosine_s = cosine_s
        if schedule_type == "linear" and not (0.0 < beta_min < beta_max):
            raise ValueError("For linear schedule, require 0 < beta_min < beta_max")

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """β(t) - noise schedule"""
        if self.schedule_type == "linear":
            return self.beta_min + t * (self.beta_max - self.beta_min)

        elif self.schedule_type == "cosine":
            # approximated β(t) from ᾱ(t)
            alpha_sq = self.alpha_squared(t)
            alpha_sq_prev = self.alpha_squared(torch.clamp(t - 0.001, min=0))
            return torch.clamp(1 - alpha_sq / (alpha_sq_prev + 1e-8), min=0, max=0.999)

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """∫₀ᵗ β(s) ds"""
        if self.schedule_type == "linear":
            return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

        elif self.schedule_type == "cosine":
            return -torch.log(self.alpha_squared(t))

    def _cosine_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """ᾱ(t) = cos²((t+s)/(1+s) · π/2) for cosine schedule"""
        return torch.cos((t + self.cosine_s) / (1 + self.cosine_s) * torch.pi / 2) ** 2

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """α(t) = exp(-½∫₀ᵗ β(s) ds)"""
        if self.schedule_type == "cosine":
            return torch.sqrt(self.alpha_squared(t))
        return torch.exp(-0.5 * self.integral_beta(t))

    def alpha_squared(self, t: torch.Tensor) -> torch.Tensor:
        """α²(t) = exp(-∫₀ᵗ β(s) ds)"""
        if self.schedule_type == "cosine":
            return self._cosine_alpha_bar(t) / self._cosine_alpha_bar(torch.zeros_like(t))
        return torch.exp(-self.integral_beta(t))

    def variance(self, t: torch.Tensor) -> torch.Tensor:
        """σ²(t) = 1 - α²(t)"""
        return 1.0 - self.alpha_squared(t)

    def std(self, t: torch.Tensor) -> torch.Tensor:
        """σ(t) = √(1 - α²(t))"""
        return torch.sqrt(self.variance(t))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """signal-to-noise ratio: SNR(t) = α²(t) / σ²(t)"""
        alpha_sq = self.alpha_squared(t)
        var = self.variance(t)
        return alpha_sq / (var + 1e-8)