import torch
import torch.nn as nn



class ForwardDDIM(nn.Module):
    """
    Implements the forward (noising) process of DDIM.

    This module samples x_t from the forward diffusion distribution:

        q(x_t | x_0) = N(x_t; sqrt(alphā_t) * x_0, (1 - alphā_t) * I)

    It also computes the appropriate training target depending on the
    prediction parameterization (noise, x0, or v-prediction).

    Args:
        scheduler: Noise scheduler containing precomputed diffusion coefficients.
        pred_type: Type of model prediction. One of ["noise", "x0", "v"].
    """
    def __init__(
            self,
            scheduler: nn.Module,
            pred_type: str = "noise"
    ):
        super().__init__()
        valid_types = ["noise", "x0", "v"]
        if pred_type not in valid_types:
            raise ValueError(f"prediction_type must be one of {valid_types}, got {pred_type}")
        self.vs = scheduler
        self.pred_type = pred_type

    def forward(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the forward diffusion step and compute the training target.

        Samples x_t by adding noise to the clean input x_0 at timestep t,
        and returns the corresponding supervision target for training.

        Args:
            x0: Clean input data of shape (batch, ...).
            t: Discrete diffusion timesteps of shape (batch,).
            noise: Gaussian noise of same shape as x0.

        Returns:
            xt: Noised data x_t of shape (batch, ...).
            target: Training target corresponding to pred_type:
                - "noise": the added noise ε
                - "x0": the original clean input x0
                - "v": the velocity parameterization
        """
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, x0.shape)
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        if self.pred_type == "noise":
            target = noise
        elif self.pred_type == "x0":
            target = x0
        elif self.pred_type == "v":
            target = sqrt_alpha_cumprod_t * noise - sqrt_one_minus_alpha_cumprod_t * x0
        return xt, target