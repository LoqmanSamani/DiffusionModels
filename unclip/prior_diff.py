import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class VarianceSchedulerUnCLIP(nn.Module):
    """Manages noise schedule parameters for UnCLIP diffusion models.

    Handles beta values, derived noise schedule quantities, and a subsampled time step schedule
    (tau schedule) for UnCLIP diffusion processes. Supports trainable or fixed beta schedules
    and multiple scheduling methods, including linear, sigmoid, quadratic, constant, inverse_time,
    and cosine schedules.

    Parameters
    ----------
    `eta` : float, optional
        Noise scaling factor for the reverse process (default: 0, deterministic).
    `num_steps` : int, optional
        Total number of diffusion steps (default: 1000).
    `tau_num_steps` : int, optional
        Number of subsampled time steps for sampling (default: 100).
    `beta_start` : float, optional
        Starting value for beta (default: 1e-4).
    `beta_end` : float, optional
        Ending value for beta (default: 0.02).
    `trainable_beta` : bool, optional
        Whether the beta schedule is trainable (default: False).
    `beta_method` : str, optional
        Method for computing the beta schedule (default: "linear").
        Supported methods: "linear", "sigmoid", "quadratic", "constant", "inverse_time", "cosine".
    """
    def __init__(
            self,
            eta: Optional[float] = None,
            num_steps: int = 1000,
            tau_num_steps: int = 100,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            trainable_beta: bool = False,
            beta_method: str = "linear"
    ) -> None:
        super().__init__()
        self.eta = eta or 0
        self.num_steps = num_steps
        self.tau_num_steps = tau_num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trainable_beta = trainable_beta
        self.beta_method = beta_method

        if not (0 < beta_start < beta_end < 1):
            raise ValueError(f"beta_start ({beta_start}) and beta_end ({beta_end}) must satisfy 0 < start < end < 1")
        if num_steps <= 0:
            raise ValueError(f"num_steps ({num_steps}) must be positive")

        beta_range = (beta_start, beta_end)
        betas_init = self.compute_beta_schedule(beta_range, num_steps, beta_method)

        if trainable_beta:
            self.beta_raw = nn.Parameter(torch.logit((betas_init - beta_start) / (beta_end - beta_start)))
        else:
            self.register_buffer('betas_buffer', betas_init)
            self.register_buffer('alphas', 1 - self.betas)
            self.register_buffer('alpha_cumprod', torch.cumprod(self.alphas, dim=0))
            self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(self.alpha_cumprod))
            self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1 - self.alpha_cumprod))

        self.register_buffer('tau_indices', torch.linspace(0, num_steps - 1, tau_num_steps, dtype=torch.long))

    @property
    def betas(self) -> torch.Tensor:
        """Returns the beta values, applying reparameterization if trainable.

        Returns the beta values, using sigmoid reparameterization for trainable betas
        or directly accessing the stored buffer for fixed betas.

        Returns
        -------
        betas : torch.Tensor
            Beta values, shape (num_steps,).
        """
        if self.trainable_beta:
            return self.beta_start + (self.beta_end - self.beta_start) * torch.sigmoid(self.beta_raw)
        return self._buffers['betas_buffer']

    def compute_beta_schedule(self, beta_range: Tuple[float, float], num_steps: int, method: str) -> torch.Tensor:
        """Computes the beta schedule based on the specified method.

        Generates a sequence of beta values for the noise schedule using the chosen method,
        ensuring values are clamped within the specified range. Supports linear, sigmoid,
        quadratic, constant, inverse_time, and cosine schedules.

        Parameters
        ----------
        `beta_range` : tuple
            Tuple of (min_beta, max_beta) specifying the valid range for beta values.
        `num_steps` : int
            Number of diffusion steps.
        `method` : str
            Method for computing the beta schedule. Supported methods:
            "linear", "sigmoid", "quadratic", "constant", "inverse_time", "cosine".

        Returns
        -------
        beta : torch.Tensor
            Tensor of beta values, shape (num_steps,).
        """
        beta_min, beta_max = beta_range
        if method == "sigmoid":
            x = torch.linspace(-6, 6, num_steps)
            beta = torch.sigmoid(x) * (beta_max - beta_min) + beta_min
        elif method == "quadratic":
            x = torch.linspace(beta_min ** 0.5, beta_max ** 0.5, num_steps)
            beta = x ** 2
        elif method == "constant":
            beta = torch.full((num_steps,), beta_max)
        elif method == "inverse_time":
            beta = 1.0 / torch.linspace(num_steps, 1, num_steps)
            beta = beta_min + (beta_max - beta_min) * (beta - beta.min()) / (beta.max() - beta.min())
        elif method == "linear":
            beta = torch.linspace(beta_min, beta_max, num_steps)
        elif method == "cosine":
            s = 0.008
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        else:
            raise ValueError(f"Unknown beta_method: {method}")
        beta = torch.clamp(beta, min=beta_min, max=beta_max)
        return beta

    def get_tau_schedule(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the subsampled (tau) noise schedule for UnCLIP.

        Returns the noise schedule parameters for the subsampled time steps used in
        UnCLIP sampling, based on the `tau_indices`.

        Returns
        -------
        tau_betas : torch.Tensor
            Beta values for subsampled steps, shape (tau_num_steps,).
        tau_alphas : torch.Tensor
            Alpha values for subsampled steps, shape (tau_num_steps,).
        tau_alpha_cumprod : torch.Tensor
            Cumulative product of alphas for subsampled steps, shape (tau_num_steps,).
        tau_sqrt_alpha_cumprod : torch.Tensor
            Square root of alpha_cumprod for subsampled steps, shape (tau_num_steps,).
        tau_sqrt_one_minus_alpha_cumprod : torch.Tensor
            Square root of (1 - alpha_cumprod) for subsampled steps, shape (tau_num_steps,).
        """
        if self.trainable_beta:
            betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod = self.compute_schedule()
        else:
            betas = self.betas
            alphas = self.alphas
            alpha_cumprod = self.alpha_cumprod
            sqrt_alpha_cumprod = self.sqrt_alpha_cumprod
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod

        tau_betas = betas[self.tau_indices]
        tau_alphas = alphas[self.tau_indices]
        tau_alpha_cumprod = alpha_cumprod[self.tau_indices]
        tau_sqrt_alpha_cumprod = sqrt_alpha_cumprod[self.tau_indices]
        tau_sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod[self.tau_indices]

        return tau_betas, tau_alphas, tau_alpha_cumprod, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod

    def compute_schedule(self, time_steps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes noise schedule parameters dynamically from betas.

        Calculates the derived noise schedule parameters (alphas, alpha_cumprod, etc.)
        from the provided beta values for the UnCLIP diffusion process.

        Parameters
        ----------
        `time_steps` : torch.Tensor, optional
            If provided, returns parameters only for specified time steps.
            If None, returns parameters for all time steps.

        Returns
        -------
        betas : torch.Tensor
            Beta values, shape (num_steps,) or (len(time_steps),).
        alphas : torch.Tensor
            1 - betas, shape (num_steps,) or (len(time_steps),).
        alpha_cumprod : torch.Tensor
            Cumulative product of alphas, shape (num_steps,) or (len(time_steps),).
        sqrt_alpha_cumprod : torch.Tensor
            Square root of alpha_cumprod, shape (num_steps,) or (len(time_steps),).
        sqrt_one_minus_alpha_cumprod : torch.Tensor
            Square root of (1 - alpha_cumprod), shape (num_steps,) or (len(time_steps),).
        """
        betas = self.betas
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
        if time_steps is not None:
            return (betas[time_steps], alphas[time_steps], alpha_cumprod[time_steps],
                    sqrt_alpha_cumprod[time_steps], sqrt_one_minus_alpha_cumprod[time_steps])
        return betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod

class ForwardUnCLIP(nn.Module):
    """Forward diffusion process for UnCLIP diffusion models.

    Applies Gaussian noise to input data (2D or 4D tensors) according to the UnCLIP
    forward diffusion process at specified time steps, using cumulative noise schedule
    parameters from the variance scheduler.

    Parameters
    ----------
    `variance_scheduler` : torch.nn.Module
        Variance scheduler module (e.g., VarianceSchedulerUnCLIP) containing the noise
        schedule parameters.
    """
    def __init__(self, variance_scheduler: torch.nn.Module) -> None:
        super().__init__()
        self.variance_scheduler = variance_scheduler

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Applies the forward diffusion process to the input data.

        Perturbs the input data `x0` by adding Gaussian noise at specified time steps,
        supporting both 2D (e.g., latent embeddings) and 4D (e.g., image) inputs.

        Parameters
        ----------
        `x0` : torch.Tensor
            Input data tensor, shape (batch_size, embedding_dim) for 2D or
            (batch_size, channels, height, width) for 4D.
        `noise` : torch.Tensor
            Gaussian noise tensor, same shape as `x0`.
        `time_steps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,),
            where each value is in the range [0, variance_scheduler.num_steps - 1].

        Returns
        -------
        xt : torch.Tensor
            Noisy data tensor at the specified time steps, same shape as `x0`.
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.variance_scheduler.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.variance_scheduler.num_steps - 1}")

        if self.variance_scheduler.trainable_beta:
            _, _, _, sqrt_alpha_cumprod_t, sqrt_one_minus_alpha_cumprod_t = self.variance_scheduler.compute_schedule(
                time_steps
            )
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.to(x0.device)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.to(x0.device)
        else:
            sqrt_alpha_cumprod_t = self.variance_scheduler.sqrt_alpha_cumprod[time_steps].to(x0.device)
            sqrt_one_minus_alpha_cumprod_t = self.variance_scheduler.sqrt_one_minus_alpha_cumprod[time_steps].to(x0.device)

        # check input dimensions and adjust reshaping for 2D or 4D tensors
        is_2d = x0.dim() == 2  # check if input is 2D (batch_size, embedding_dim)
        if is_2d:
            # for 2D inputs, reshape to [batch_size, 1]
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1)
        else:
            # for 4D inputs, reshape to [batch_size, 1, 1, 1]
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1, 1)

        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt


class ReverseUnCLIP(nn.Module):
    """Reverse diffusion process for UnCLIP diffusion models.

    Denoises a noisy input `xt` using either a predicted noise component or predicted clean image
    and a subsampled time step schedule, supporting both 2D (e.g., latent embeddings) and 4D (e.g., image) inputs.

    Parameters
    ----------
    `variance_scheduler` : torch.nn.Module
        Variance scheduler module (e.g., VarianceSchedulerUnCLIP) containing the noise
        schedule parameters.
    `prediction_type` : str, default "noise"
        Type of prediction the model makes. Either "noise" (predicts noise like DDIM) or
        "x0" (predicts clean image like UnCLIP prior).
    """

    def __init__(self, variance_scheduler: torch.nn.Module, prediction_type: str = "noise"):
        super().__init__()
        self.variance_scheduler = variance_scheduler
        if prediction_type not in ["noise", "x0"]:
            raise ValueError(f"prediction_type must be either 'noise' or 'x0', got {prediction_type}")
        self.prediction_type = prediction_type

    def forward(self, xt: torch.Tensor, model_prediction: torch.Tensor, time_steps: torch.Tensor,
                prev_time_steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the reverse diffusion process to the noisy input.

        Denoises the input `xt` at time step `t` to produce the previous step `xt_prev`
        at `prev_time_steps` using either the predicted noise or predicted clean image
        and the UnCLIP reverse process. Supports both 2D and 4D inputs.

        Parameters
        ----------
        `xt` : torch.Tensor
            Noisy input tensor at time step `t`, shape (batch_size, embedding_dim) for 2D
            or (batch_size, channels, height, width) for 4D.
        `model_prediction` : torch.Tensor
            Model prediction tensor, same shape as `xt`. Can be either predicted noise
            or predicted clean image depending on `prediction_type`.
        `time_steps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, variance_scheduler.tau_num_steps - 1].
        `prev_time_steps` : torch.Tensor
            Tensor of previous time step indices (long), shape (batch_size,), where each
            value is in the range [0, variance_scheduler.tau_num_steps - 1].

        Returns
        -------
        xt_prev : torch.Tensor
            Denoised tensor at `prev_time_steps`, same shape as `xt`.
        x0 : torch.Tensor
            Estimated original data (t=0), same shape as `xt`.
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.variance_scheduler.tau_num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.variance_scheduler.tau_num_steps - 1}")
        if not torch.all((prev_time_steps >= 0) & (prev_time_steps < self.variance_scheduler.tau_num_steps)):
            raise ValueError(f"prev_time_steps must be between 0 and {self.variance_scheduler.tau_num_steps - 1}")

        _, _, _, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod = self.variance_scheduler.get_tau_schedule()

        # Check input dimensions and adjust reshaping for 2D or 4D tensors
        is_2d = xt.dim() == 2  # check if input is 2D (batch_size, embedding_dim)
        if is_2d:
            # for 2D inputs, reshape to [batch_size, 1]
            tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[time_steps].to(xt.device).view(-1, 1)
            tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[time_steps].to(xt.device).view(-1, 1)
            prev_tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1)
            prev_tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[prev_time_steps].to(
                xt.device).view(-1, 1)
        else:
            # for 4D inputs, reshape to [batch_size, 1, 1, 1]
            tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
            tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[time_steps].to(xt.device).view(-1, 1,
                                                                                                                 1, 1)
            prev_tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)
            prev_tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[prev_time_steps].to(
                xt.device).view(-1, 1, 1, 1)

        eta = self.variance_scheduler.eta

        predicted_noise = None
        x0 = None
        # Handle different prediction types
        if self.prediction_type == "noise":
            # model predicts noise
            predicted_noise = model_prediction
            x0 = (xt - tau_sqrt_one_minus_alpha_cumprod_t * predicted_noise) / tau_sqrt_alpha_cumprod_t
        elif self.prediction_type == "x0":
            # model predicts clean image
            x0 = model_prediction
            # Calculate implied noise from the predicted clean image
            predicted_noise = (xt - tau_sqrt_alpha_cumprod_t * x0) / tau_sqrt_one_minus_alpha_cumprod_t

        # DDIM sampling step (same for both prediction types)
        noise_coeff = eta * ((tau_sqrt_one_minus_alpha_cumprod_t / prev_tau_sqrt_alpha_cumprod_t) *
                             prev_tau_sqrt_one_minus_alpha_cumprod_t / torch.clamp(tau_sqrt_one_minus_alpha_cumprod_t,
                                                                                   min=1e-8))
        direction_coeff = torch.clamp(prev_tau_sqrt_one_minus_alpha_cumprod_t ** 2 - noise_coeff ** 2, min=1e-8).sqrt()
        xt_prev = prev_tau_sqrt_alpha_cumprod_t * x0 + noise_coeff * torch.randn_like(xt) + direction_coeff * predicted_noise

        return xt_prev, x0

    def set_prediction_type(self, prediction_type: str):
        """Change the prediction type after initialization.

        Parameters
        ----------
        prediction_type : str
            Type of prediction the model makes. Either "noise" or "x0".
        """
        if prediction_type not in ["noise", "x0"]:
            raise ValueError(f"prediction_type must be either 'noise' or 'x0', got {prediction_type}")
        self.prediction_type = prediction_type

"""
hyp = VarianceSchedulerUnCLIP(
    num_steps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=False,
    beta_method="sigmoid"
)

forward = ForwardUnCLIP(hyp)
x = torch.randn((10, 3, 100, 100))
t = torch.randint(0, 1000, (10,))
noise = torch.randn_like(x)

xt = forward(x, noise, t)
print(xt.size())
"""