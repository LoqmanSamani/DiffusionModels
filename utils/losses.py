import torch



def mse_loss(pred: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
    """
    Standard mean squared error (MSE) loss.

    Computes the element-wise squared difference between `pred` and `target`
    and returns the mean across all elements.

    Args:
        pred: Predicted tensor, shape [B, ...].
        target: Target tensor, same shape as `pred`.
        *args: Placeholder for optional unused arguments for API compatibility.

    Returns:
        Scalar tensor representing mean squared error.
    """
    return ((pred - target) ** 2).mean()


def snr_capped_loss(pred_noise: torch.Tensor, target_noise: torch.Tensor, variance: torch.Tensor,
                    gamma: float = 5.0, *args) -> torch.Tensor:
    """
    Signal-to-noise-ratio (SNR) capped noise prediction loss for diffusion models.

    This implements a weighted MSE where the weight is the SNR of the timestep,
    capped at a maximum value `gamma`. Typically used in VP/VE noise prediction.

    Args:
        pred_noise: Predicted noise tensor, same shape as target_noise.
        target_noise: True noise tensor.
        variance: Variance (sigma^2) corresponding to the timestep t, shape broadcastable to pred_noise.
        gamma: Maximum SNR weight (default 5.0).
        *args: Placeholder for optional unused arguments for API compatibility.

    Returns:
        Scalar tensor representing the SNR-weighted mean squared error.
    """
    snr = (1 - variance) / variance.clamp(min=1e-8)
    weight = torch.minimum(snr, torch.tensor(gamma, device=snr.device))
    while weight.dim() < target_noise.dim():
        weight = weight.unsqueeze(-1)
    return ((pred_noise - target_noise) ** 2 * weight).mean()


def ve_sigma_weighted_score_loss(pred_score: torch.Tensor, target_score: torch.Tensor, sigma: torch.Tensor, *args) -> torch.Tensor:
    """
    VE-SDE sigma-weighted score matching loss.

    Implements the recommended loss for Variance Exploding SDEs:
        E[ || sigma(t) * s_theta(x_t, t) + epsilon ||^2 ]
    where epsilon is the true noise used to perturb x_0.

    Args:
        pred_score: Model-predicted score tensor (∇_x log p(x_t)), shape [B, ...].
        target_score: Target score, typically -epsilon / sigma(t).
        sigma: Standard deviation (σ(t)) at the corresponding timesteps, shape broadcastable to pred_score.
        *args: Placeholder for optional unused arguments for API compatibility.

    Returns:
        Scalar tensor representing the sigma-weighted score matching loss.
    """
    while sigma.dim() < pred_score.dim():
        sigma = sigma.unsqueeze(-1)
    eps = -target_score * sigma
    return ((sigma * pred_score + eps) ** 2).mean()