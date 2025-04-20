__version__ = "1.0.0"

"""Score-Based Generative Modeling with Stochastic Differential Equations (SDE).

This module implements a complete framework for score-based generative models using SDEs,
as described in Song et al. (2021, "Score-Based Generative Modeling through Stochastic
Differential Equations"). It provides components for forward and reverse diffusion
processes, hyperparameter management, training, and image sampling, supporting Variance
Exploding (VE), Variance Preserving (VP), sub-Variance Preserving (sub-VP), and ODE
methods for flexible noise schedules. Supports both unconditional and conditional
generation with text prompts.

Components:
- ForwardSDE: Forward diffusion process to add noise using SDE methods.
- ReverseSDE: Reverse diffusion process to denoise using SDE methods.
- HyperParamsSDE: Noise schedule and SDE-specific parameter management.
- TrainSDE: Training loop with mixed precision and scheduling.
- SampleSDE: Image generation from trained SDE models.

References:
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
  Score-Based Generative Modeling through Stochastic Differential Equations.

Examples
--------
>>> from torchdiff.sde import HyperParamsSDE, ForwardSDE, ReverseSDE, TrainSDE, SampleSDE
>>> from torchdiff.utils import TextEncoder, NoisePredictor, Metrics
>>> from torch.optim import Adam
>>> import torch.nn as nn
...
>>> hyper_params = HyperParamsSDE(num_steps=1000, beta_start=1e-4, beta_end=0.02)
>>> forward_sde = ForwardSDE(hyper_params, method="vp")
>>> reverse_sde = ReverseSDE(hyper_params, method="vp")
>>> metrics = Metrics(device='cuda', fid=True, metrics=True, lpips=True)
>>> noise_predictor = NoisePredictor(in_channels=3, down_channels=[32, 64, 128], mid_channels=[128, 128, 128],
...                                  up_channels=[128, 64, 32], down_sampling=[True, True, True], time_embed_dim=128,
...                                  y_embed_dim=128, num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2, dropout_rate=0.1,
...                                  down_sampling_factor=2, where_y=True, y_to_all=False)
>>> text_encoder = TextEncoder(use_pretrained_model=True, model_name="bert-base-uncased", vocabulary_size=30522,
...                            num_layers=2, input_dimension=128, output_dimension=128, num_heads=4, context_length=77,
...                            dropout_rate=0.1, qkv_bias=False, scaling_value=4, epsilon=1e-5)
>>> optimizer = Adam(compressor.parameters(), lr=1e-4)
>>> metrics = Metrics(device='cuda', fid=True, metrics=True, lpips=True)
>>> train_sde = TrainSDE(method="vp", noise_predictor=noise_predictor, hyper_params=hyper_params,
...                      data_loader=data_loader, optimizer=optimizer, objective=nn.MSELoss(),
...                      conditional_model=text_encoder, tokenizer=tokenizer, metrics_=metrics)
>>> train_losses, best_val_loss = train_sde()
>>> sampler = SampleSDE(reverse_sde, noise_predictor, image_shape=(64, 64))
>>> images = sampler(conditions="A cat", normalize_output=True, save_images=True, save_path="sde_generated")

License
-------
MIT License.

Version
-------
1.0.0
"""


import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
import warnings
from torchvision.utils import save_image
import os

###==================================================================================================================###

class ForwardSDE(nn.Module):
    """Forward diffusion process for SDE-based generative models.

    Implements the forward diffusion process for score-based generative models using
    Stochastic Differential Equations (SDEs), supporting Variance Exploding (VE),
    Variance Preserving (VP), sub-Variance Preserving (sub-VP), and ODE methods, as
    described in Song et al. (2021).

    Parameters
    ----------
    hyper_params : object
        Hyperparameter object (HyperParamsSDE) containing SDE-specific parameters. Expected to have
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
            #if self.method == "ve":
            #    x0 = x0
            #else:
            betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            x0 = x0 + drift
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return x0

###==================================================================================================================###

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
        Hyperparameter object (HyperParamsSDE) containing SDE-specific parameters. Expected to have
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
            #if self.method == "ve":
            #    sigma_t = self.hyper_params.sigmas[time_steps]
            #    sigma_t_prev = self.hyper_params.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
            #    drift = -0.5 * (sigma_t ** 2 - sigma_t_prev ** 2).view(-1, 1, 1, 1) * predicted_noise * dt
            #else:
            drift = -0.5 * betas * xt * dt - 0.5 * betas * predicted_noise * dt
            xt = xt + drift
            xt = torch.clamp(xt, -1e5, 1e5)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return xt

###==================================================================================================================###

class HyperParamsSDE(nn.Module):
    """Hyperparameters for SDE-based generative models.

    Manages the noise schedule and SDE-specific parameters for score-based generative
    models, including beta and sigma schedules, time steps, and variance computations,
    as described in Song et al. (2021). Supports trainable or fixed beta schedules and
    multiple scheduling methods for flexible noise control.

    Parameters
    ----------
    num_steps : int, optional
        Number of diffusion steps (default: 1000).
    beta_start : float, optional
        Starting value for beta schedule (default: 1e-4).
    beta_end : float, optional
        Ending value for beta schedule (default: 0.02).
    trainable_beta : bool, optional
        Whether the beta schedule is trainable (default: False).
    beta_method : str, optional
        Method for computing the beta schedule (default: "linear").
        Supported methods: "linear", "sigmoid", "quadratic", "constant", "inverse_time".
    sigma_start : float, optional
        Starting value for sigma schedule for VE method (default: 1e-3).
    sigma_end : float, optional
        Ending value for sigma schedule for VE method (default: 10.0).
    start : float, optional
        Start of the time interval for SDE integration (default: 0.0).
    end : float, optional
        End of the time interval for SDE integration (default: 1.0).

    Attributes
    ----------
    num_steps : int
        Number of diffusion steps.
    beta_start : float
        Minimum beta value.
    beta_end : float
        Maximum beta value.
    trainable_beta : bool
        Whether the beta schedule is trainable.
    beta_method : str
        Method used for beta schedule computation.
    sigma_start : float
        Minimum sigma value for VE method.
    sigma_end : float
        Maximum sigma value for VE method.
    start : float
        Start of the time interval.
    end : float
        End of the time interval.
    betas : torch.Tensor
        Beta schedule values, shape (num_steps,). Trainable if `trainable_beta` is True,
        otherwise a fixed buffer.
    cum_betas : torch.Tensor, optional
        Cumulative sum of betas scaled by `dt`, shape (num_steps,). Available if
        `trainable_beta` is False.
    sigmas : torch.Tensor, optional
        Sigma schedule for VE method, shape (num_steps,). Available if
        `trainable_beta` is False.
    time : torch.Tensor
        Time points for SDE integration, shape (num_steps,).
    dt : float
        Time step size for SDE integration, computed as (end - start) / num_steps.

    Raises
    ------
    ValueError
        If `beta_start` or `beta_end` do not satisfy 0 < beta_start < beta_end,
        `sigma_start` or `sigma_end` do not satisfy 0 < sigma_start < sigma_end,
        or `num_steps` is not positive.
    """
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, trainable_beta=False, beta_method="linear",
                 sigma_start=1e-3, sigma_end=10.0, start=0.0, end=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trainable_beta = trainable_beta
        self.beta_method = beta_method
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.start = start
        self.end = end

        if not (0 < self.beta_start < self.beta_end):
            raise ValueError(f"beta_start ({self.beta_start}) and beta_end ({self.beta_end}) must satisfy 0 < start < end")
        if not (0 < self.sigma_start < self.sigma_end):
            raise ValueError(f"sigma_start ({self.sigma_start}) and sigma_end ({self.sigma_end}) must satisfy 0 < start < end")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps ({self.num_steps}) must be positive")

        beta_range = (beta_start, beta_end)
        betas_init = self.compute_beta_schedule(beta_range, num_steps, beta_method)
        self.time = torch.linspace(self.start, self.end, self.num_steps, dtype=torch.float32)
        self.dt = (self.end - self.start) / self.num_steps

        if trainable_beta:
            self.betas = nn.Parameter(betas_init)
        else:
            self.register_buffer('betas', betas_init)
            self.register_buffer('cum_betas', torch.cumsum(betas_init, dim=0) * self.dt)
            self.register_buffer("sigmas", self.sigma_start * (self.sigma_end / self.sigma_start) ** self.time)

    def compute_beta_schedule(self, beta_range, num_steps, method):
        """Computes the beta schedule based on the specified method.

        Generates a sequence of beta values for the SDE noise schedule using the chosen
        method, ensuring values are clamped within the specified range.

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
            raise ValueError(f"Unknown beta_method: {method}. Supported: linear, sigmoid, quadratic, constant, inverse_time")
        beta = torch.clamp(beta, min=beta_min, max=beta_max)
        return beta

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

    def get_variance(self, time_steps, method):
        """Computes the variance for the specified SDE method at given time steps.

        Calculates the variance used in SDE diffusion processes based on the method
        (VE, VP, or sub-VP), leveraging the sigma or cumulative beta schedules.

        Parameters
        ----------
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, num_steps - 1].
        method : str
            SDE method to compute variance for. Supported methods: "ve", "vp", "sub-vp".

        Returns
        -------
        torch.Tensor
            Variance values for the specified time steps, shape (batch_size,).

        Raises
        ------
        ValueError
            If `method` is not one of the supported methods ("ve", "vp", "sub-vp").
        """
        if method == "ve":
            return self.sigmas[time_steps] ** 2
        elif method == "vp":
            return 1 - torch.exp(-self.cum_betas[time_steps])
        elif method == "sub-vp":
            return 1 - torch.exp(-2 * self.cum_betas[time_steps])
        else:
            raise ValueError(f"Unknown method: {method}")

###==================================================================================================================###

class TrainSDE(nn.Module):
    """Trainer for score-based generative models using Stochastic Differential Equations.

    Manages the training process for SDE-based generative models, optimizing a noise
    predictor to learn the noise added by the forward SDE process, as described in Song
    et al. (2021). Supports conditional training with text prompts, mixed precision,
    learning rate scheduling, early stopping, and checkpointing.

    Parameters
    ----------
    method : str
        SDE method to use for forward diffusion. Supported methods: "ve", "vp", "sub-vp", "ode".
    noise_predictor : nn.Module
        Model to predict noise added during the forward SDE process.
    hyper_params : nn.Module
        Hyperparameter module (e.g., HyperParamsSDE) defining the noise schedule and SDE parameters.
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
        Path to save model checkpoints (default: "sde_model.pth").
    patience : int, optional
        Number of epochs to wait for improvement before early stopping (default: 10).
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
    method : str
        Selected SDE method.
    noise_predictor : nn.Module
        Noise prediction model.
    hyper_params : nn.Module
        Hyperparameter module for the noise schedule and SDE parameters.
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
    forward_diffusion : ForwardSDE
        Forward SDE diffusion module.
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
    def __init__(self, method, noise_predictor, hyper_params, data_loader, optimizer, objective, val_loader=None,
                 max_epoch=1000, device=None, conditional_model=None, metrics_=None, tokenizer=None, max_length=77,
                 store_path=None, patience=100, warmup_epochs=100, val_frequency=10, output_range=(-1, 1), normalize_output=True):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.method = method
        self.noise_predictor = noise_predictor.to(self.device)
        self.hyper_params = hyper_params.to(self.device)
        self.forward_diffusion = ForwardSDE(hyper_params=self.hyper_params, method=self.method).to(self.device)
        self.reverse_diffusion = ReverseSDE(hyper_params=self.hyper_params, method=self.method).to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or "sde_model.pth"
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.max_length = max_length
        self.patience = patience
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience, factor=0.5)
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
    def warmup_scheduler(optimizer, warmup_epochs=10):
        """Creates a learning rate scheduler for warmup.

        Generates a scheduler that linearly increases the learning rate from 0 to the
        optimizer's initial value over the specified warmup epochs, then maintains it.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to apply the scheduler to.
        warmup_epochs : int, optional
            Number of epochs for the warmup phase (default: 10).

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
        """Trains the SDE model to predict noise added by the forward diffusion process.

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
        - Training uses mixed precision via `torch.cuda.amp` or `torch.amp` for efficiency.
        - Checkpoints are saved when the validation (or training) loss improves, and on
          early stopping.
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
                        'epoch': epoch+1,
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
                            'epoch': epoch+1,
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
                    for t in reversed(range(self.hyper_params.num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, y_encoded)
                        noise = torch.randn_like(xt) if getattr(self.reverse_diffusion, "method", None) != "ode" else None
                        xt = self.reverse_diffusion(xt, noise, predicted_noise, time_steps)

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

class SampleSDE(nn.Module):
    """Sampler for generating images using SDE-based generative models.

    Generates images by iteratively denoising random noise using the reverse SDE process
    and a trained noise predictor, as described in Song et al. (2021). Supports both
    unconditional and conditional generation with text prompts.

    Parameters
    ----------
    reverse_diffusion : ReverseSDE
        Reverse SDE diffusion module for denoising.
    noise_predictor : nn.Module
        Model to predict noise added during the forward SDE process.
    image_shape : tuple
        Shape of generated images as (height, width).
    conditional_model : nn.Module, optional
        Model for conditional generation (e.g., TextEncoder), default None.
    tokenizer : str or BertTokenizer, optional
        Tokenizer for processing text prompts, default "bert-base-uncased".
    max_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for generated images (default: 3).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    output_range : tuple, optional
        Range for clamping generated images (min, max), default (-1, 1).

    Attributes
    ----------
    device : torch.device
        Device used for computation.
    reverse : ReverseSDE
        Reverse SDE diffusion module.
    noise_predictor : nn.Module
        Noise prediction model.
    conditional_model : nn.Module or None
        Conditional model for text-based generation, if provided.
    tokenizer : BertTokenizer
        Tokenizer for text prompts.
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
        positive, or `output_range` is not a tuple (min, max) with min < max.
    """
    def __init__(self, reverse_diffusion, noise_predictor, image_shape, conditional_model=None,
                 tokenizer="bert-base-uncased", max_length=77, batch_size=1, in_channels=3, device=None, output_range=(-1, 1)):
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

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(output_range, (tuple, list)) or len(output_range) != 2 or output_range[0] >= output_range[1]:
            raise ValueError("output_range must be a tuple (min, max) with min < max")

    def tokenize(self, prompts):
        """Tokenizes text prompts for conditional generation.

        Converts input prompts into tokenized tensors using the specified tokenizer.

        Parameters
        ----------
        prompts : str or list
            Text prompt(s) for conditional generation. Can be a single string or a list
            of strings.

        Returns
        -------
        tuple
            A tuple containing:
            - input_ids: Tokenized input IDs (torch.Tensor, shape (batch_size, max_length)).
            - attention_mask: Attention mask for tokenized inputs (torch.Tensor, same shape).

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

    def forward(self, conditions=None, normalize_output=True, save_images=True, save_path="sde_generated"):
        """Generates images using the reverse SDE sampling process.

        Iteratively denoises random noise to generate images using the reverse SDE process
        and noise predictor. Supports conditional generation with text prompts.

        Parameters
        ----------
        conditions : str or list, optional
            Text prompt(s) for conditional generation, default None.
        normalize_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        save_images : bool, optional
            If True, saves generated images to `save_path` (default: True).
        save_path : str, optional
            Directory to save generated images (default: "sde_generated").

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

        Notes
        -----
        - Sampling is performed with `torch.no_grad()` for efficiency.
        - The noise predictor, reverse SDE, and conditional model (if applicable) are set
          to evaluation mode during sampling.
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
            for t in reversed(range(self.reverse.hyper_params.num_steps)):
                noise = torch.randn_like(xt) if self.reverse.method != "ode" else None
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)

                if self.conditional_model is not None and conditions is not None:
                    input_ids, attention_masks = self.tokenize(conditions)
                    key_padding_mask = (attention_masks == 0)
                    y = self.conditional_model(input_ids, key_padding_mask)
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)

                xt = self.reverse(xt, noise, predicted_noise, time_steps)

            generated_imgs = torch.clamp(xt, min=self.output_range[0], max=self.output_range[1])
            if normalize_output:
                generated_imgs = (generated_imgs - self.output_range[0]) / (self.output_range[1] - self.output_range[0])

            # save images if save_images is True
            if save_images:
                os.makedirs(save_path, exist_ok=True)
                for i in range(generated_imgs.size(0)):
                    img_path = os.path.join(save_path, f"image_{i}.png")
                    save_image(generated_imgs[i], img_path)

        return generated_imgs

    def to(self, device):
        """Moves the module and its components to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device for computation.

        Returns
        -------
        self
            The module moved to the specified device.

        Notes
        -----
        - Moves `noise_predictor`, `reverse`, and `conditional_model` (if applicable) to
          the specified device.
        """
        self.device = device
        self.noise_predictor.to(device)
        self.reverse.to(device)
        if self.conditional_model:
            self.conditional_model.to(device)
        return super().to(device)