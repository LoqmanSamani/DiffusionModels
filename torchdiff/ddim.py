__version__ = "1.0.0"

"""Denoising Diffusion Implicit Models (DDIM) implementation.

This module provides a complete implementation of DDIM, as described in Song et al.
(2021, "Denoising Diffusion Implicit Models"). It includes components for forward and
reverse diffusion processes, hyperparameter management, training, and image sampling.
Supports both unconditional and conditional generation with text prompts, using a
subsampled time step schedule for faster sampling compared to DDPM.

Components:
- ForwardDDIM: Forward diffusion process to add noise.
- ReverseDDIM: Reverse diffusion process to denoise with subsampled steps.
- HyperParamsDDIM: Noise schedule management with subsampled (tau) schedule.
- TrainDDIM: Training loop with mixed precision and scheduling.
- SampleDDIM: Image generation from trained models with subsampled steps.

Notes
-----
- The subsampled time step schedule (tau) enables faster sampling, controlled by the
  `tau_num_steps` parameter in HyperParamsDDIM.

References:
- Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models.

Examples
--------
>>> from torchdiff.ddim import HyperParamsDDIM, ForwardDDIM, ReverseDDIM, TrainDDIM, SampleDDIM
>>> from torchdiff.utils import TextEncoder, NoisePredictor, Metrics
>>> from torch.optim import Adam
>>> import torch.nn as nn
...
>>> hyper_params = HyperParamsDDIM(num_steps=1000, tau_num_steps=100)
>>> forward_ddim = ForwardDDIM(hyper_params)
>>> reverse_ddim = ReverseDDIM(hyper_params)
>>> noise_predictor = NoisePredictor(in_channels=3, down_channels=[32, 64, 128], mid_channels=[128, 128, 128],
...                                  up_channels=[128, 64, 32], down_sampling=[True, True, True], time_embed_dim=128,
...                                  y_embed_dim=128, num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2, dropout_rate=0.1,
...                                  down_sampling_factor=2, where_y=True, y_to_all=False)
>>> text_encoder = TextEncoder(use_pretrained_model=True, model_name="bert-base-uncased", vocabulary_size=30522,
...                            num_layers=2, input_dimension=128, output_dimension=128, num_heads=4, context_length=77,
...                            dropout_rate=0.1, qkv_bias=False, scaling_value=4, epsilon=1e-5)
>>> optimizer = Adam(compressor.parameters(), lr=1e-4)
>>> metrics = Metrics(device='cuda', fid=True, metrics=True, lpips=True)
>>> train_ddim = TrainDDIM(noise_predictor=noise_predictor, hyper_params=hyper_params, data_loader=data_loader,
...                        optimizer=optimizer, objective=nn.MSELoss(), conditional_model=text_encoder, metrics_=metrics)
>>> train_losses, best_val_loss = train_ddim()
>>> sampler = SampleDDIM(reverse_ddim, noise_predictor, image_shape=(64, 64))
>>> images = sampler(conditions="A cat", normalize_output=True, save_images=True, save_path="ddim_generated")

License
-------
MIT License.

Version
-------
1.0.0
"""


import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
import warnings
from torchvision.utils import save_image
import os

###==================================================================================================================###

class ForwardDDIM(nn.Module):
    """Forward diffusion process of DDIM.

    Implements the forward diffusion process for Denoising Diffusion Implicit Models
    (DDIM), which perturbs input data by adding Gaussian noise over a series of time
    steps, as defined in Song et al. (2021, "Denoising Diffusion Implicit Models").

    Parameters
    ----------
    hyper_params : object
        Hyperparameter object (HyperParamsDDIM) containing the noise schedule parameters. Expected to have
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

###==================================================================================================================###


class ReverseDDIM(nn.Module):
    """Reverse diffusion process of DDIM.

    Implements the reverse diffusion process for Denoising Diffusion Implicit Models
    (DDIM), which denoises a noisy input `xt` using a predicted noise component and a
    subsampled time step schedule, as defined in Song et al. (2021).

    Parameters
    ----------
    hyper_params : object
        Hyperparameter object (HyperParamsDDIM) containing the noise schedule parameters. Expected to have
        attributes:
        - `tau_num_steps`: Number of subsampled time steps (int).
        - `eta`: Noise scaling factor for the reverse process (float).
        - `get_tau_schedule`: Method to compute the subsampled noise schedule (callable),
          returning a tuple of (betas, alphas, alpha_bars, sqrt_alpha_cumprod,
          sqrt_one_minus_alpha_cumprod).

    Attributes
    ----------
    hyper_params : object
        Stores the provided hyperparameter object.
    """
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params

    def forward(self, xt, predicted_noise, time_steps, prev_time_steps):
        """Applies the reverse diffusion process to the noisy input.

        Denoises the input `xt` at time step `t` to produce the previous step `xt_prev`
        at `prev_time_steps` using the predicted noise and the DDIM reverse process.
        Optionally includes stochastic noise scaled by `eta`.

        Parameters
        ----------
        xt : torch.Tensor
            Noisy input tensor at time step `t`, shape (batch_size, channels, height, width).
        predicted_noise : torch.Tensor
            Predicted noise tensor, same shape as `xt`, typically output by a neural network.
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, hyper_params.tau_num_steps - 1].
        prev_time_steps : torch.Tensor
            Tensor of previous time step indices (long), shape (batch_size,), where each
            value is in the range [0, hyper_params.tau_num_steps - 1].

        Returns
        -------
        tuple
            A tuple containing:
            - xt_prev: Denoised tensor at `prev_time_steps`, same shape as `xt`.
            - x0: Estimated original data (t=0), same shape as `xt`.

        Raises
        ------
        ValueError
            If any value in `time_steps` or `prev_time_steps` is outside the valid range
            [0, hyper_params.tau_num_steps - 1].
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.hyper_params.tau_num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.hyper_params.tau_num_steps - 1}")
        if not torch.all((prev_time_steps >= 0) & (prev_time_steps < self.hyper_params.tau_num_steps)):
            raise ValueError(f"prev_time_steps must be between 0 and {self.hyper_params.tau_num_steps - 1}")

        _, _, _, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod = self.hyper_params.get_tau_schedule()
        tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
        tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)

        eta = self.hyper_params.eta
        x0 = (xt - tau_sqrt_one_minus_alpha_cumprod_t * predicted_noise) / tau_sqrt_alpha_cumprod_t
        noise_coeff = eta * ((tau_sqrt_one_minus_alpha_cumprod_t / prev_tau_sqrt_alpha_cumprod_t) *
                             prev_tau_sqrt_one_minus_alpha_cumprod_t / torch.clamp(tau_sqrt_one_minus_alpha_cumprod_t, min=1e-8))
        direction_coeff = torch.clamp(prev_tau_sqrt_one_minus_alpha_cumprod_t ** 2 - noise_coeff ** 2, min=1e-8).sqrt()
        xt_prev = prev_tau_sqrt_alpha_cumprod_t * x0 + noise_coeff * torch.randn_like(xt) + direction_coeff * predicted_noise

        return xt_prev, x0

###==================================================================================================================###

class HyperParamsDDIM(nn.Module):
    """Hyperparameters for DDIM noise schedule with flexible beta computation.

    Manages the noise schedule parameters for DDIM, including beta values, derived
    quantities (alphas, alpha_cumprod, etc.), and a subsampled time step schedule
    (tau schedule), as inspired by Song et al. (2021). Supports trainable or fixed
    schedules and various beta scheduling methods.

    Parameters
    ----------
    eta : float, optional
        Noise scaling factor for the DDIM reverse process (default: 0, deterministic).
    num_steps : int, optional
        Total number of diffusion steps (default: 1000).
    tau_num_steps : int, optional
        Number of subsampled time steps for DDIM sampling (default: 100).
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
    eta : float
        Noise scaling factor for the reverse process.
    num_steps : int
        Total number of diffusion steps.
    tau_num_steps : int
        Number of subsampled time steps.
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
    alpha_cumprod : torch.Tensor, optional
        Cumulative product of alphas, shape (num_steps,). Available if `trainable_beta` is False.
    sqrt_alpha_cumprod : torch.Tensor, optional
        Square root of alpha_cumprod, shape (num_steps,). Available if `trainable_beta` is False.
    sqrt_one_minus_alpha_cumprod : torch.Tensor, optional
        Square root of (1 - alpha_cumprod), shape (num_steps,). Available if `trainable_beta` is False.
    tau_indices : torch.Tensor
        Indices for subsampled time steps, shape (tau_num_steps,).

    Raises
    ------
    ValueError
        If `beta_start` or `beta_end` do not satisfy 0 < beta_start < beta_end < 1,
        or if `num_steps` is not positive.
    """

    def __init__(self, eta=None, num_steps=1000, tau_num_steps=100, beta_start=1e-4, beta_end=0.02,
                 trainable_beta=False, beta_method="linear"):
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
            self.betas = nn.Parameter(betas_init)
        else:
            self.register_buffer('betas', betas_init)
            self.register_buffer('alphas', 1 - self.betas)
            self.register_buffer('alpha_cumprod', torch.cumprod(self.alphas, dim=0))
            self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(self.alpha_cumprod))
            self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1 - self.alpha_cumprod))

        self.register_buffer('tau_indices', torch.linspace(0, num_steps - 1, tau_num_steps, dtype=torch.long))

    def compute_beta_schedule(self, beta_range, num_steps, method):
        """Computes the beta schedule based on the specified method.

        Generates a sequence of beta values for the DDIM noise schedule using the
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
            x = torch.linspace(beta_min ** 0.5, beta_max ** 0.5, num_steps)
            beta = x ** 2
        elif method == "constant":
            beta = torch.full((num_steps,), beta_max)
        elif method == "inverse_time":
            beta = 1.0 / torch.linspace(num_steps, 1, num_steps)
            beta = beta_min + (beta_max - beta_min) * (beta - beta.min()) / (beta.max() - beta.min())
        elif method == "linear":
            beta = torch.linspace(beta_min, beta_max, num_steps)
        else:
            raise ValueError(
                f"Unknown beta_method: {method}. Supported: linear, sigmoid, quadratic, constant, inverse_time")

        beta = torch.clamp(beta, min=beta_min, max=beta_max)
        return beta

    def get_tau_schedule(self):
        """Computes the subsampled (tau) noise schedule for DDIM.

        Returns the noise schedule parameters for the subsampled time steps used in
        DDIM sampling, based on the `tau_indices`.

        Returns
        -------
        tuple
            A tuple containing:
            - tau_betas: Beta values for subsampled steps, shape (tau_num_steps,).
            - tau_alphas: Alpha values for subsampled steps, shape (tau_num_steps,).
            - tau_alpha_cumprod: Cumulative product of alphas for subsampled steps, shape (tau_num_steps,).
            - tau_sqrt_alpha_cumprod: Square root of alpha_cumprod for subsampled steps, shape (tau_num_steps,).
            - tau_sqrt_one_minus_alpha_cumprod: Square root of (1 - alpha_cumprod) for subsampled steps, shape (tau_num_steps,).
        """
        if self.trainable_beta:
            betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod = self.compute_schedule(self.betas)
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

    def compute_schedule(self, betas):
        """Computes noise schedule parameters dynamically from betas.

        Calculates the derived noise schedule parameters (alphas, alpha_cumprod, etc.)
        from the provided beta values, as used in the DDIM forward and reverse processes.

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
            - alpha_cumprod: Cumulative product of alphas, shape (num_steps,).
            - sqrt_alpha_cumprod: Square root of alpha_cumprod, shape (num_steps,).
            - sqrt_one_minus_alpha_cumprod: Square root of (1 - alpha_cumprod), shape (num_steps,).
        """
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_cumprod, torch.sqrt(alpha_cumprod), torch.sqrt(1 - alpha_cumprod)

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

###==================================================================================================================###

class TrainDDIM(nn.Module):
    """Trainer for Denoising Diffusion Implicit Models (DDIM).

    Manages the training process for DDIM, optimizing a noise predictor model to learn
    the noise added by the forward diffusion process. Supports conditional training with
    text prompts, mixed precision training, learning rate scheduling, early stopping, and
    checkpointing, as inspired by Song et al. (2021).

    Parameters
    ----------
    noise_predictor : nn.Module
        Model to predict noise added during the forward diffusion process.
    hyper_params : nn.Module
        Hyperparameter module (e.g., HyperParamsDDIM) defining the noise schedule.
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    optimizer : torch.optim.Optimizer
        Optimizer for training the noise predictor and conditional model (if applicable).
    objective : callable
        Loss function to compute the difference between predicted and actual noise.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    max_epoch : int, optional
        Maximum number of training epochs (default: 1000).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    conditional_model : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    metrics_ : Metrics, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    tokenizer : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    max_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    store_path : str, optional
        Path to save model checkpoints (default: "ddim_model.pth").
    patience : int, optional
        Number of epochs to wait for improvement before early stopping (default: 100).
    warmup_epochs : int, optional
        Number of epochs for learning rate warmup (default: 100).
    val_frequency : int, optional
        Frequency (in epochs) for validation (default: 10).
    output_range : tuple, optional
        Range for clamping generated images (default: (-1, 1)).
    normalize_output : bool, optional
        Whether to normalize generated images to [0, 1] for metrics (default: True).

    Attributes
    ----------
    device : torch.device
        Device used for computation.
    noise_predictor : nn.Module
        Noise prediction model.
    hyper_params : nn.Module
        Hyperparameter module for the noise schedule.
    conditional_model : nn.Module or None
        Conditional model for text-based training, if provided.
    metrics_ : Metrics or None
        Metrics object for evaluation.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    objective : callable
        Loss function for training.
    store_path : str
        Path for saving checkpoints.
    data_loader : torch.utils.data.DataLoader
        Training data loader.
    val_loader : torch.utils.data.DataLoader or None
        Validation data loader, if provided.
    max_epoch : int
        Maximum training epochs.
    max_length : int
        Maximum length for tokenized prompts.
    patience : int
        Patience for early stopping.
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler based on validation or training loss.
    forward_diffusion : ForwardDDIM
        Forward diffusion module for DDIM.
    warmup_lr_scheduler : torch.optim.lr_scheduler.LambdaLR
        Learning rate scheduler for warmup.
    val_frequency : int
        Frequency for validation.
    tokenizer : BertTokenizer
        Tokenizer for text prompts.
    output_range : tuple
        Output range for generated images.
    normalize_output : bool
        Whether to normalize output.

    Raises
    ------
    ValueError
        If the default tokenizer ("bert-base-uncased") fails to load and no tokenizer is provided.
    """
    def __init__(self, noise_predictor, hyper_params, data_loader, optimizer, objective, val_loader=None,
                 max_epoch=1000, device=None, conditional_model=None, metrics_=None, tokenizer=None, max_length=77,
                 store_path=None, patience=100, warmup_epochs=100, val_frequency=10, output_range=(-1, 1), normalize_output=True):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_predictor = noise_predictor.to(self.device)
        self.hyper_params = hyper_params.to(self.device)
        self.forward_diffusion = ForwardDDIM(hyper_params=self.hyper_params).to(self.device)
        self.reverse_diffusion = ReverseDDIM(hyper_params=self.hyper_params).to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or "ddim_model.pth"
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.max_length = max_length
        self.patience = patience
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience, factor=0.5)
        self.forward_diffusion = ForwardDDIM(hyper_params=self.hyper_params).to(self.device)
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)
        self.val_frequency = val_frequency
        self.output_range = output_range
        self.normalize_output = normalize_output
        if tokenizer is None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")

    def load_checkpoint(self, checkpoint_path):
        """Loads a training checkpoint to resume training.

        Restores the state of the noise predictor, conditional model (if applicable),
        and optimizer from a saved checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.

        Returns
        -------
        tuple
            A tuple containing:
            - epoch: The epoch at which the checkpoint was saved (int).
            - loss: The loss at the checkpoint (float).

        Raises
        ------
        FileNotFoundError
            If the checkpoint file is not found.
        KeyError
            If the checkpoint is missing required keys ('model_state_dict_noise_predictor'
            or 'optimizer_state_dict').

        Warns
        -----
        warnings.warn
            If the optimizer state cannot be loaded, if the checkpoint contains a
            conditional model state but none is defined, or if no conditional model
            state is provided when expected.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        if 'model_state_dict_noise_predictor' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict_noise_predictor' key")
        self.noise_predictor.load_state_dict(checkpoint['model_state_dict_noise_predictor'])

        if self.conditional_model is not None:
            if 'model_state_dict_conditional' in checkpoint and checkpoint['model_state_dict_conditional'] is not None:
                self.conditional_model.load_state_dict(checkpoint['model_state_dict_conditional'])
            else:
                warnings.warn(
                    "Checkpoint contains no 'model_state_dict_conditional' or it is None, skipping conditional model loading")
        elif 'model_state_dict_conditional' in checkpoint and checkpoint['model_state_dict_conditional'] is not None:
            warnings.warn(
                "Checkpoint contains conditional model state, but no conditional model is defined in this instance")

        if 'optimizer_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'optimizer_state_dict' key")
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            warnings.warn(f"Optimizer state loading failed: {e}. Continuing without optimizer state.")

        epoch = checkpoint.get('epoch', -1)
        loss = checkpoint.get('loss', float('inf'))

        self.noise_predictor.to(self.device)
        if self.conditional_model is not None:
            self.conditional_model.to(self.device)

        print(f"Loaded checkpoint from {checkpoint_path} at epoch {epoch} with loss {loss:.4f}")
        return epoch, loss

    @staticmethod
    def warmup_scheduler(optimizer, warmup_epochs):
        """Creates a learning rate scheduler for warmup.

        Generates a scheduler that linearly increases the learning rate from 0 to the
        optimizer's initial value over the specified warmup epochs, then maintains it.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to apply the scheduler to.
        warmup_epochs : int, optional
            Number of epochs for the warmup phase.

        Returns
        -------
        torch.optim.lr_scheduler.LambdaLR
            Learning rate scheduler for warmup.
        """
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    def forward(self):
        """Trains the DDIM model to predict noise added by the forward diffusion process.

        Executes the training loop, optimizing the noise predictor and conditional model
        (if applicable) using mixed precision, gradient clipping, and learning rate
        scheduling. Supports validation, early stopping, and checkpointing.

        Returns
        -------
        tuple
            A tuple containing:
            - train_losses: List of mean training losses per epoch (list of float).
            - best_val_loss: Best validation or training loss achieved (float).

        Notes
        -----
        - Checkpoints are saved when the validation (or training) loss improves, and on early stopping.
        - Early stopping is triggered if no improvement occurs for `patience` epochs.
        """
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()

        scaler = GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0
        for epoch in range(self.max_epoch):
            train_losses_ = []
            for x, y in tqdm(self.data_loader):
                x = x.to(self.device)

                if self.conditional_model is not None:
                    y_list = y.cpu().numpy().tolist() if isinstance(y, torch.Tensor) else y
                    y_list = [str(item) for item in y_list]
                    y_encoded = self.tokenizer(
                        y_list,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    input_ids = y_encoded["input_ids"]
                    attention_mask = y_encoded["attention_mask"]
                    y_encoded = self.conditional_model(input_ids, attention_mask)
                else:
                    y_encoded = None

                self.optimizer.zero_grad()
                with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    noise = torch.randn_like(x).to(self.device)
                    t = torch.randint(0, self.hyper_params.num_steps, (x.shape[0],)).to(self.device)
                    assert x.device == noise.device == t.device, "Device mismatch detected"
                    assert t.shape[0] == x.shape[0], "Timestep batch size mismatch"
                    noisy_x = self.forward_diffusion(x, noise, t)
                    p_noise = self.noise_predictor(noisy_x, t, y_encoded)
                    loss = self.objective(p_noise, noise)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.noise_predictor.parameters(), max_norm=1.0)
                if self.conditional_model is not None:
                    nn.utils.clip_grad_norm_(self.conditional_model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                self.warmup_lr_scheduler.step()
                train_losses_.append(loss.item())

            if self.hyper_params.trainable_beta:
                self.hyper_params.constrain_betas()

            mean_train_loss = torch.mean(torch.tensor(train_losses_)).item()
            train_losses.append(mean_train_loss)
            print(f"\nEpoch: {epoch + 1} | Train Loss: {mean_train_loss:.4f}", end="")

            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_loss, fid, mse, psnr, ssim, lpips_score = self.validate()
                print(f" | Val Loss: {val_loss:.4f}", end="")
                if self.metrics_ and self.metrics_.fid:
                    print(f" | FID: {fid:.4f}", end="")
                if self.metrics_ and self.metrics_.metrics:
                    print(f" | MSE: {mse:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}", end="")
                if self.metrics_ and self.metrics_.lpips:
                    print(f" | LPIPS: {lpips_score:.4f}", end="")
                print()

                current_best = val_loss
                self.scheduler.step(val_loss)
            else:
                print()
                current_best = mean_train_loss
                self.scheduler.step(mean_train_loss)

            if current_best < best_val_loss and (epoch + 1) % self.val_frequency == 0:
                best_val_loss = current_best
                wait = 0
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict_noise_predictor': self.noise_predictor.state_dict(),
                        'model_state_dict_conditional': self.conditional_model.state_dict() if self.conditional_model is not None else None,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_val_loss,
                        'hyper_params_model': self.hyper_params.state_dict() if isinstance(self.hyper_params, nn.Module) else self.hyper_params,
                        'max_epoch': self.max_epoch,
                    }, self.store_path)
                    print(f"Model saved at epoch {epoch + 1}")
                except Exception as e:
                    print(f"Failed to save model: {e}")
            else:
                wait += 1
                if wait >= self.patience:
                    print("Early stopping triggered")
                    try:
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict_noise_predictor': self.noise_predictor.state_dict(),
                            'model_state_dict_conditional': self.conditional_model.state_dict() if self.conditional_model is not None else None,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': best_val_loss,
                            'hyper_params_model': self.hyper_params.state_dict() if isinstance(self.hyper_params, nn.Module) else self.hyper_params,
                            'max_epoch': self.max_epoch,
                        }, self.store_path + "_early_stop.pth")
                        print(f"Final model saved at {self.store_path}_early_stop.pth")
                    except Exception as e:
                        print(f"Failed to save final model: {e}")
                    break

        return train_losses, best_val_loss

    def validate(self):
        """Validates the noise predictor and computes evaluation metrics.

        Computes validation loss (MSE between predicted and ground truth noise) and generates
        samples using the reverse diffusion model by manually iterating over timesteps.
        Decodes samples to images and computes image-domain metrics (MSE, PSNR, SSIM, FID, LPIPS)
        if metrics_ is provided.

        Returns
        -------
        tuple
            A tuple containing:
            - val_loss: Mean validation loss (float).
            - fid: Mean FID score (float, or `float('inf')` if not computed).
            - mse: Mean MSE (float, or None if not computed).
            - psnr: Mean PSNR (float, or None if not computed).
            - ssim: Mean SSIM (float, or None if not computed).
            - lpips_score: Mean LPIPS score (float, or None if not computed).
        """
        self.noise_predictor.eval()
        if self.conditional_model is not None:
            self.conditional_model.eval()

        val_losses = []
        fid_, mse_, psnr_, ssim_, lpips_score_ = [], [], [], [], []
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                x_orig = x

                if self.conditional_model is not None:
                    y_list = y.cpu().numpy().tolist() if isinstance(y, torch.Tensor) else y
                    y_list = [str(item) for item in y_list]
                    y_encoded = self.tokenizer(
                        y_list,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    input_ids = y_encoded["input_ids"]
                    attention_mask = y_encoded["attention_mask"]
                    y_encoded = self.conditional_model(input_ids, attention_mask)
                else:
                    y_encoded = None

                # validation loss
                noise = torch.randn_like(x).to(self.device)
                t = torch.randint(0, self.hyper_params.num_steps, (x.shape[0],)).to(self.device)
                assert x.device == noise.device == t.device, "Device mismatch detected"
                assert t.shape[0] == x.shape[0], "Timestep batch size mismatch"
                noisy_x = self.forward_diffusion(x, noise, t)
                p_noise = self.noise_predictor(noisy_x, t, y_encoded)
                loss = self.objective(p_noise, noise)
                val_losses.append(loss.item())

                # generate samples for metrics
                if self.metrics_ is not None and self.reverse_diffusion is not None:
                    xt = torch.randn_like(x).to(self.device)
                    for t in reversed(range(self.hyper_params.tau_num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        prev_time_steps = torch.full((xt.shape[0],), max(t - 1, 0), device=self.device, dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, y_encoded)
                        xt, _ = self.reverse_diffusion(xt, predicted_noise, time_steps, prev_time_steps)

                    x_hat = torch.clamp(xt, min=self.output_range[0], max=self.output_range[1])
                    if self.normalize_output:
                        x_hat = (x_hat - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
                        x_orig = (x_orig - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
                    fid, mse, psnr, ssim, lpips_score = self.metrics_.forward(x_orig, x_hat)
                    if self.metrics_.fid:
                        fid_.append(fid)
                    if self.metrics_.metrics:
                        mse_.append(mse)
                        psnr_.append(psnr)
                        ssim_.append(ssim)
                    if self.metrics_.lpips:
                        lpips_score_.append(lpips_score)

            val_loss = torch.mean(torch.tensor(val_losses)).item()
            fid_ = torch.mean(torch.tensor(fid_)).item() if fid_ else float('inf')
            mse_ = torch.mean(torch.tensor(mse_)).item() if mse_ else None
            psnr_ = torch.mean(torch.tensor(psnr_)).item() if psnr_ else None
            ssim_ = torch.mean(torch.tensor(ssim_)).item() if ssim_ else None
            lpips_score_ = torch.mean(torch.tensor(lpips_score_)).item() if lpips_score_ else None

        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()
        return val_loss, fid_, mse_, psnr_, ssim_, lpips_score_

###==================================================================================================================###

class SampleDDIM(nn.Module):
    """Image generation using a trained DDIM model.

    Implements the sampling process for DDIM, generating images by iteratively denoising
    random noise using a trained noise predictor and reverse diffusion process with a
    subsampled time step schedule. Supports conditional generation with text prompts,
    as inspired by Song et al. (2021).

    Parameters
    ----------
    reverse_diffusion : nn.Module
        Reverse diffusion module (e.g., ReverseDDIM) for the reverse process.
    noise_predictor : nn.Module
        Trained model to predict noise at each time step.
    image_shape : tuple
        Tuple of (height, width) specifying the generated image dimensions.
    conditional_model : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    tokenizer : str, optional
        Pretrained tokenizer name from Hugging Face (default: "bert-base-uncased").
    max_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for generated images (default: 3).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    output_range : tuple, optional
        Tuple of (min, max) for clamping generated images (default: (-1, 1)).

    Attributes
    ----------
    device : torch.device
        Device used for computation.
    reverse : nn.Module
        Reverse diffusion module.
    noise_predictor : nn.Module
        Noise prediction model.
    conditional_model : nn.Module or None
        Conditional model for text-based generation, if provided.
    tokenizer : BertTokenizer
        Tokenizer for processing text prompts.
    max_length : int
        Maximum length for tokenized prompts.
    in_channels : int
        Number of input channels.
    image_shape : tuple
        Shape of generated images (height, width).
    batch_size : int
        Batch size for generation.
    output_range : tuple
        Range for clamping generated images.

    Raises
    ------
    ValueError
        If `image_shape` is not a tuple of two positive integers, `batch_size` is not
        positive, or `output_range` is not a valid (min, max) tuple with min < max.
    """
    def __init__(self, reverse_diffusion, noise_predictor, image_shape, conditional_model=None,
                 tokenizer="bert-base-uncased",
                 max_length=77, batch_size=1, in_channels=3, device=None, output_range=(-1, 1)):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reverse = reverse_diffusion.to(self.device)
        self.noise_predictor = noise_predictor.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.in_channels = in_channels
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.output_range = output_range

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(
                isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(output_range, (tuple, list)) or len(output_range) != 2 or output_range[0] >= output_range[1]:
            raise ValueError("output_range must be a tuple (min, max) with min < max")


    def tokenize(self, prompts):
        """Tokenizes text prompts for conditional generation.

        Converts input prompts into tokenized input IDs and attention masks using the
        specified tokenizer, suitable for use with the conditional model.

        Parameters
        ----------
        prompts : str or list
            A single text prompt or a list of text prompts.

        Returns
        -------
        tuple
            A tuple containing:
            - input_ids: Tokenized input IDs, shape (batch_size, max_length).
            - attention_mask: Attention mask, shape (batch_size, max_length).

        Raises
        ------
        TypeError
            If `prompts` is not a string or a list of strings.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        elif not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise TypeError("prompts must be a string or list of strings")
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def forward(self, conditions=None, normalize_output=True, save_images=True, save_path="ddim_generated"):
        """Generates images using the DDIM sampling process.

        Iteratively denoises random noise to generate images using the reverse diffusion
        process with a subsampled time step schedule and noise predictor. Supports
        conditional generation with text prompts.

        Parameters
        ----------
        conditions : str or list, optional
            Text prompt(s) for conditional generation, default None.
        normalize_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        save_images : bool, optional
            If True, saves generated images to `save_path` (default: True).
        save_path : str, optional
            Directory to save generated images (default: "ddim_generated").

        Returns
        -------
        torch.Tensor
            Generated images, shape (batch_size, in_channels, height, width).
            If `normalize_output` is True, images are normalized to [0, 1]; otherwise,
            they are clamped to `output_range`.

        Raises
        ------
        ValueError
            If `conditions` is provided but no conditional model is specified, or if
            a conditional model is specified but `conditions` is None.
        """

        if conditions is not None and self.conditional_model is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conditions is None and self.conditional_model is not None:
            raise ValueError("Conditions must be provided for conditional model")

        noisy_samples = torch.randn(self.batch_size, self.in_channels, self.image_shape[0], self.image_shape[1]).to(self.device)

        self.noise_predictor.eval()
        self.reverse.eval()
        if self.conditional_model:
            self.conditional_model.eval()

        with torch.no_grad():
            xt = noisy_samples
            for t in reversed(range(self.reverse.hyper_params.tau_num_steps)):
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)
                prev_time_steps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device, dtype=torch.long)

                if self.conditional_model is not None and conditions is not None:
                    input_ids, attention_masks = self.tokenize(conditions)
                    key_padding_mask = (attention_masks == 0)
                    y = self.conditional_model(input_ids, key_padding_mask)
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)

                xt, _ = self.reverse(xt, predicted_noise, time_steps, prev_time_steps)

            generated_imgs = torch.clamp(xt, min=self.output_range[0], max=self.output_range[1])
            if normalize_output:
                generated_imgs = (generated_imgs - self.output_range[0]) / (self.output_range[1] - self.output_range[0])

            if save_images:
                os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
                for i in range(generated_imgs.size(0)):
                    img_path = os.path.join(save_path, f"image_{i}.png")
                    save_image(generated_imgs[i], img_path)

        return generated_imgs

    def to(self, device):
        """Moves the module and its components to the specified device.

        Updates the device attribute and moves the reverse diffusion, noise predictor,
        and conditional model (if present) to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device for the module and its components.

        Returns
        -------
        SampleDDIM
            The module itself, moved to the specified device.
        """
        self.device = device
        self.noise_predictor.to(device)
        self.reverse.to(device)
        if self.conditional_model:
            self.conditional_model.to(device)
        return super().to(device)