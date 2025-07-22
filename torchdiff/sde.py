"""
**Score-Based Generative Modeling with Stochastic Differential Equations (SDE)**

This module implements a complete framework for score-based generative models using SDEs,
as described in Song et al. (2021, "Score-Based Generative Modeling through Stochastic
Differential Equations"). It provides components for forward and reverse diffusion
processes, hyperparameter management, training, and image sampling, supporting Variance
Exploding (VE), Variance Preserving (VP), sub-Variance Preserving (sub-VP), and ODE
methods for flexible noise schedules. Supports both unconditional and conditional
generation with text prompts.

**Components**

- **ForwardSDE**: Forward diffusion process to add noise using SDE methods.
- **ReverseSDE**: Reverse diffusion process to denoise using SDE methods.
- **HyperParamsSDE**: Noise schedule and SDE-specific parameter management.
- **TrainSDE**: Training loop with mixed precision and scheduling.
- **SampleSDE**: Image generation from trained SDE models.

**References**

- Song, Yang, et al. "Score-based generative modeling through stochastic differential equations." arXiv preprint arXiv:2011.13456 (2020).

---------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from typing import Optional, Tuple, Callable, List, Any, Union, Self
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
import warnings
from torchvision.utils import save_image


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
        attributes: `dt`, `sigmas`, `betas`, `cum_betas`.
    method : str
        SDE method to use. Supported methods: "ve", "vp", "sub-vp", "ode".
    """
    def __init__(self, hyper_params: torch.nn.Module, method: str) -> None:
        super().__init__()
        self.hyper_params = hyper_params
        self.method = method

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
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
        xt (torch.Tensor) - Noisy data tensor at the specified time steps, same shape as `x0`.

        """
        dt = self.hyper_params.dt
        if self.method == "ve":
            # Use property to get sigmas (handles trainable case)
            sigma_t = self.hyper_params.sigmas[time_steps]
            sigma_t_prev = self.hyper_params.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
            sigma_diff = torch.sqrt(torch.clamp(sigma_t ** 2 - sigma_t_prev ** 2, min=0))
            x0 = x0 + noise * sigma_diff.view(-1, 1, 1, 1)

        elif self.method == "vp":
            # Use property to get betas (handles trainable case)
            betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            diffusion = torch.sqrt(betas * dt) * noise
            x0 = x0 + drift + diffusion

        elif self.method == "sub-vp":
            # Use properties to get betas and cum_betas (handles trainable case)
            betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
            cum_betas = self.hyper_params._cum_betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            diffusion = torch.sqrt(betas * (1 - torch.exp(-2 * cum_betas)) * dt) * noise
            x0 = x0 + drift + diffusion

        elif self.method == "ode":
            # Use property to get betas (handles trainable case)
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
        attributes: `dt`, `sigmas`, `betas`, `cum_betas`.
    method : str
        SDE method to use. Supported methods: "ve", "vp", "sub-vp", "ode".
    """
    def __init__(self, hyper_params: torch.nn.Module, method: str) -> None:
        super().__init__()
        self.hyper_params = hyper_params
        self.method = method

    def forward(self, xt: torch.Tensor, noise: torch.Tensor, predicted_noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
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
        xt (torch.Tensor) - Denoised tensor at the previous time step, same shape as `xt`.

        **Notes**

        - For the "ve" and "ode" methods, the output is clamped to [-1e5, 1e5] to prevent numerical instability.
        - Stochastic noise (`noise`) is only added if provided and the method supports it (not applicable for "ode" in non-VE cases).
        """
        dt = self.hyper_params.dt
        # Use properties to get betas and cum_betas (handles trainable case)
        betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
        cum_betas = self.hyper_params._cum_betas[time_steps].view(-1, 1, 1, 1)
        if self.method == "ve":
            # Use property to get sigmas (handles trainable case)
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
    """
    def __init__(
            self,
            num_steps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            trainable_beta: bool = False,
            beta_method: str = "linear",
            sigma_start: float = 1e-3,
            sigma_end: float = 10.0,
            start: float = 0.0,
            end: float = 1.0
    ) -> None:
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
            # Use reparameterization trick for trainable betas
            # Initialize unconstrained parameters and transform them to valid beta range
            self.beta_raw = nn.Parameter(torch.logit((betas_init - beta_start) / (beta_end - beta_start)))
        else:
            self.register_buffer('betas_buffer', betas_init)
            self.register_buffer('cum_betas', torch.cumsum(betas_init, dim=0) * self.dt)
            self.register_buffer("sigmas_buffer", self.sigma_start * (self.sigma_end / self.sigma_start) ** self.time)

    @property
    def betas(self) -> torch.Tensor:
        """Returns the beta values, applying reparameterization if trainable."""
        if self.trainable_beta:
            # Transform unconstrained parameters to valid beta range using sigmoid
            return self.beta_start + (self.beta_end - self.beta_start) * torch.sigmoid(self.beta_raw)
        else:
            return self._buffers['betas_buffer']

    @property
    def _cum_betas(self) -> torch.Tensor:
        """Returns the cumulative beta values, computing dynamically if trainable."""
        if self.trainable_beta:
            return torch.cumsum(self.betas, dim=0) * self.dt
        else:
            return self._buffers['cum_betas']

    @property
    def sigmas(self) -> torch.Tensor:
        """Returns the sigma values, computing dynamically if trainable."""
        if self.trainable_beta:
            # For trainable case, sigmas still use the fixed time schedule
            # but could be modified if needed for trainable sigma schedules
            return self.sigma_start * (self.sigma_end / self.sigma_start) ** self.time
        else:
            return self._buffers['sigmas_buffer']

    def compute_beta_schedule(self, beta_range: Tuple[float, float], num_steps: int, method: str) -> torch.Tensor:
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
        betas (torch.Tensor) - Tensor of beta values, shape (num_steps,).
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

    def get_variance(self, time_steps: torch.Tensor, method: str) -> torch.Tensor:
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
        variance_values (torch.Tensor) - Variance values for the specified time steps, shape (batch_size,).
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
    metrics_ : object, optional
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
    ddp : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    num_grad_accumulation : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    progress_frequency : int, optional
        Number of epochs before printing loss.
    """
    def __init__(
            self,
            method: str,
            noise_predictor: torch.nn.Module,
            hyper_params: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epoch: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            conditional_model: Optional[torch.nn.Module] = None,
            metrics_: Optional[Any] = None,
            tokenizer: Optional[BertTokenizer] = None,
            max_length: int = 77,
            store_path: Optional[str] = None,
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            output_range: Tuple[float, float] = (-1.0, 1.0),
            normalize_output: bool = True,
            ddp: bool = False,
            num_grad_accumulation: int = 1,
            progress_frequency: int = 1
    ) -> None:

        super().__init__()
        # Initialize DDP settings first
        self.ddp = ddp
        self.num_grad_accumulation = num_grad_accumulation
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup distributed training if enabled
        if self.ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # Move models to appropriate device
        self.method = method
        self.noise_predictor = noise_predictor.to(self.device)
        self.hyper_params = hyper_params.to(self.device)
        self.forward_diffusion = ForwardSDE(hyper_params=self.hyper_params, method=self.method).to(self.device)
        self.reverse_diffusion = ReverseSDE(hyper_params=self.hyper_params, method=self.method).to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None

        # Training components
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or "ddpm_model"
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.max_length = max_length
        self.patience = patience
        self.val_frequency = val_frequency
        self.output_range = output_range
        self.normalize_output = normalize_output
        self.progress_frequency = progress_frequency

        # Learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)

        # Initialize tokenizer
        if tokenizer is None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")
        else:
            self.tokenizer = tokenizer


    def _setup_ddp(self) -> None:
        """Setup Distributed Data Parallel training configuration.

        Initializes process group, determines rank information, and sets up
        CUDA device for the current process.
        """
        # Check if DDP environment variables are set
        if "RANK" not in os.environ:
            raise ValueError("DDP enabled but RANK environment variable not set")
        if "LOCAL_RANK" not in os.environ:
            raise ValueError("DDP enabled but LOCAL_RANK environment variable not set")
        if "WORLD_SIZE" not in os.environ:
            raise ValueError("DDP enabled but WORLD_SIZE environment variable not set")

        # Ensure CUDA is available for DDP
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA but CUDA is not available")

        # Initialize process group only if not already initialized
        if not torch.distributed.is_initialized():
            init_process_group(backend="nccl")

        # Get rank information
        self.ddp_rank = int(os.environ["RANK"])  # Global rank across all nodes
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])  # Local rank on current node
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes

        # Set device and make it current
        self.device = torch.device(f"cuda:{self.ddp_local_rank}")
        # self.device = f"cuda:{self.ddp_local_rank}"
        torch.cuda.set_device(self.device)

        # Master process handles logging, checkpointing, etc.
        self.master_process = self.ddp_rank == 0

        if self.master_process:
            print(f"DDP initialized with world_size={self.ddp_world_size}")

    def _setup_single_gpu(self) -> None:
        """Setup single GPU or CPU training configuration."""
        self.ddp_rank = 0
        self.ddp_local_rank = 0
        self.ddp_world_size = 1
        self.master_process = True

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """Loads a training checkpoint to resume training.

        Restores the state of the noise predictor, conditional model (if applicable),
        and optimizer from a saved checkpoint. Handles DDP model state dict loading.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.

        Returns
        -------
        epoch : int
            The epoch at which the checkpoint was saved.
        loss : float
             The loss at the checkpoint.
        """
        try:
            # Load checkpoint with proper device mapping
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        # Load noise predictor state
        if 'model_state_dict_noise_predictor' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict_noise_predictor' key")

        # Handle DDP wrapped model state dict
        state_dict = checkpoint['model_state_dict_noise_predictor']
        if self.ddp and not any(key.startswith('module.') for key in state_dict.keys()):
            # If loading non-DDP checkpoint into DDP model, add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not self.ddp and any(key.startswith('module.') for key in state_dict.keys()):
            # If loading DDP checkpoint into non-DDP model, remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.noise_predictor.load_state_dict(state_dict)

        # Load conditional model state if applicable
        if self.conditional_model is not None:
            if 'model_state_dict_conditional' in checkpoint and checkpoint['model_state_dict_conditional'] is not None:
                cond_state_dict = checkpoint['model_state_dict_conditional']
                # Handle DDP wrapping for conditional model
                if self.ddp and not any(key.startswith('module.') for key in cond_state_dict.keys()):
                    cond_state_dict = {f'module.{k}': v for k, v in cond_state_dict.items()}
                elif not self.ddp and any(key.startswith('module.') for key in cond_state_dict.keys()):
                    cond_state_dict = {k.replace('module.', ''): v for k, v in cond_state_dict.items()}
                self.conditional_model.load_state_dict(cond_state_dict)
            else:
                warnings.warn(
                    "Checkpoint contains no 'model_state_dict_conditional' or it is None, "
                    "skipping conditional model loading"
                )

        # Load optimizer state
        if 'optimizer_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'optimizer_state_dict' key")
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            warnings.warn(f"Optimizer state loading failed: {e}. Continuing without optimizer state.")

        epoch = checkpoint.get('epoch', -1)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Loaded checkpoint from {checkpoint_path} at epoch {epoch} with loss {loss:.4f}")
        return epoch, loss

    @staticmethod
    def warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Creates a learning rate scheduler for warmup.

        Generates a scheduler that linearly increases the learning rate from 0 to the
        optimizer's initial value over the specified warmup epochs, then maintains it.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to apply the scheduler to.
        warmup_epochs : int
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

    def _wrap_models_for_ddp(self) -> None:
        """Wrap models with DistributedDataParallel for multi-GPU training."""
        if self.ddp:
            # Wrap noise predictor with DDP
            self.noise_predictor = DDP(
                self.noise_predictor,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

            # Wrap conditional model with DDP if it exists
            if self.conditional_model is not None:
                self.conditional_model = DDP(
                    self.conditional_model,
                    device_ids=[self.ddp_local_rank],
                    find_unused_parameters=True
                )


    def forward(self) -> Tuple[List, float]:
        """Trains the SDE model to predict noise added by the forward diffusion process.

        Executes the training loop, optimizing the noise predictor and conditional model
        (if applicable) using mixed precision, gradient clipping, and learning rate
        scheduling. Supports validation, early stopping, and checkpointing.

        Returns
        -------
        train_losses : list of float
             List of mean training losses per epoch.
        best_val_loss : float
             Best validation or training loss achieved.

        **Notes**

        - Training uses mixed precision via `torch.cuda.amp` or `torch.amp` for efficiency.
        - Checkpoints are saved when the validation (or training) loss improves, and on early stopping.
        - Early stopping is triggered if no improvement occurs for `patience` epochs.
        """
        # Set models to training mode
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()

        # Compile models for optimization (if supported)
        """
        try:
            self.noise_predictor = torch.compile(self.noise_predictor)
            if self.conditional_model is not None:
                self.conditional_model = torch.compile(self.conditional_model)
        except Exception as e:
            if self.master_process:
                print(f"Model compilation failed: {e}. Continuing without compilation.")
        """

        # Wrap models for DDP after compilation
        self._wrap_models_for_ddp()

        # Initialize training components
        scaler = torch.GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        # Main training loop
        for epoch in range(self.max_epoch):
            # Set epoch for distributed sampler if using DDP
            if self.ddp and hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []
            # Training step loop with gradient accumulation
            for step, (x, y) in enumerate(tqdm(self.data_loader, disable=not self.master_process)):
                x = x.to(self.device)

                # Process conditional inputs if conditional model exists
                if self.conditional_model is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None

                # Forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    # Generate noise and timesteps
                    noise = torch.randn_like(x).to(self.device)
                    t = torch.randint(0, self.hyper_params.num_steps, (x.shape[0],)).to(self.device)

                    # Apply forward diffusion
                    noisy_x = self.forward_diffusion(x, noise, t)

                    # Predict noise
                    predicted_noise = self.noise_predictor(noisy_x, t, y_encoded)

                    # Compute loss and scale for gradient accumulation
                    loss = self.objective(predicted_noise, noise) / self.num_grad_accumulation

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient accumulation and optimizer step
                if (step + 1) % self.num_grad_accumulation == 0:
                    # Clip gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.noise_predictor.parameters(), max_norm=1.0)
                    if self.conditional_model is not None:
                        torch.nn.utils.clip_grad_norm_(self.conditional_model.parameters(), max_norm=1.0)

                # Optimizer step
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                # Update learning rate (warmup scheduler)
                self.warmup_lr_scheduler.step()

            # Record loss (unscaled)
            train_losses_epoch.append(loss.item() * self.num_grad_accumulation)

            # Compute mean training loss
            mean_train_loss = torch.tensor(train_losses_epoch).mean().item()

            # All-reduce loss across processes for DDP
            if self.ddp:
                loss_tensor = torch.tensor(mean_train_loss, device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                mean_train_loss = loss_tensor.item()

            # Print training progress (only master process)
            if self.master_process:
                if (epoch + 1) % self.progress_frequency == 0:
                    print(f"\nEpoch: {epoch + 1} | Learning Rate: {self.optimizer.param_groups[0]['lr']} | Train Loss: {mean_train_loss:.4f}", end="")

            # Validation step
            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_metrics = self.validate()
                val_loss, fid, mse, psnr, ssim, lpips_score = val_metrics

                if self.master_process:
                    print(f" | Val Loss: {val_loss:.4f}", end="")
                    if self.metrics_ and hasattr(self.metrics_, 'fid') and self.metrics_.fid:
                        print(f" | FID: {fid:.4f}", end="")
                    if self.metrics_ and hasattr(self.metrics_, 'metrics') and self.metrics_.metrics:
                        print(f" | MSE: {mse:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}", end="")
                    if self.metrics_ and hasattr(self.metrics_, 'lpips') and self.metrics_.lpips:
                        print(f" | LPIPS: {lpips_score:.4f}", end="")
                    print()

                current_best = val_loss
                self.scheduler.step(val_loss)
            else:
                if self.master_process:
                    print()
                current_best = mean_train_loss
                self.scheduler.step(mean_train_loss)

            # Save checkpoint and early stopping (only master process)
            if self.master_process:
                if current_best < best_val_loss and (epoch + 1) % self.val_frequency == 0:
                    best_val_loss = current_best
                    wait = 0
                    self._save_checkpoint(epoch + 1, best_val_loss)
                else:
                    wait += 1
                    if wait >= self.patience:
                        print("Early stopping triggered")
                        self._save_checkpoint(epoch + 1, best_val_loss, "_early_stop")
                        break

        # Clean up DDP
        if self.ddp:
            destroy_process_group()

        return train_losses, best_val_loss

    def _process_conditional_input(self, y: Union[torch.Tensor, List]) -> torch.Tensor:
        """Process conditional input for text-to-image generation.

        Parameters
        ----------
        y : torch.Tensor or list
            Conditional input (text prompts).

        Returns
        -------
        torch.Tensor
            Encoded conditional input.
        """
        # Convert to string list
        y_list = y.cpu().numpy().tolist() if isinstance(y, torch.Tensor) else y
        y_list = [str(item) for item in y_list]

        # Tokenize
        y_encoded = self.tokenizer(
            y_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Get embeddings
        input_ids = y_encoded["input_ids"]
        attention_mask = y_encoded["attention_mask"]
        y_encoded = self.conditional_model(input_ids, attention_mask)

        return y_encoded


    def _save_checkpoint(self, epoch: int, loss: float, suffix: str = "") -> None:
        """Save model checkpoint (only called by master process).

        Parameters
        ----------
        epoch : int
            Current epoch number.
        loss : float
            Current loss value.
        suffix : str, optional
            Suffix to add to checkpoint filename.
        """
        try:
            # Get state dicts, handling DDP wrapping
            noise_predictor_state = (
                self.noise_predictor.module.state_dict() if self.ddp
                else self.noise_predictor.state_dict()
            )
            conditional_state = None
            if self.conditional_model is not None:
                conditional_state = (
                    self.conditional_model.module.state_dict() if self.ddp
                    else self.conditional_model.state_dict()
                )

            checkpoint = {
                'epoch': epoch,
                'model_state_dict_noise_predictor': noise_predictor_state,
                'model_state_dict_conditional': conditional_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'hyper_params_model': (
                    self.hyper_params.state_dict() if isinstance(self.hyper_params, nn.Module)
                    else self.hyper_params
                ),
                'max_epoch': self.max_epoch,
            }

            save_path = self.store_path + suffix + ".pth" if suffix else self.store_path + ".pth"
            torch.save(checkpoint, save_path)
            print(f"Model saved at epoch {epoch} to {save_path}")

        except Exception as e:
            print(f"Failed to save model: {e}")


    def validate(self) -> Tuple[float, float, float, float, float, float]:
        """Validates the noise predictor and computes evaluation Metrics.

        Computes validation loss (MSE between predicted and ground truth noise) and generates
        samples using the reverse diffusion model by manually iterating over timesteps.
        Decodes samples to images and computes image-domain Metrics (MSE, PSNR, SSIM, FID, LPIPS)
        if metrics_ is provided.

        Returns
        -------
        val_loss : float
            Mean validation loss.
        fid : float, or `float('inf')` if not computed
            Mean FID score.
        mse : float, or None if not computed
            Mean MSE
        psnr : float, or None if not computed
             Mean PSNR
        ssim : float, or None if not computed
            Mean SSIM
        lpips_score :  float, or None if not computed
            Mean LPIPS score
        """
        self.noise_predictor.eval()
        if self.conditional_model is not None:
            self.conditional_model.eval()

        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                x_orig = x.clone()

                # Process conditional input
                if self.conditional_model is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None

                # Compute validation loss
                noise = torch.randn_like(x).to(self.device)
                t = torch.randint(0, self.hyper_params.num_steps, (x.shape[0],)).to(self.device)

                noisy_x = self.forward_diffusion(x, noise, t)
                predicted_noise = self.noise_predictor(noisy_x, t, y_encoded)
                loss = self.objective(predicted_noise, noise)
                val_losses.append(loss.item())

                # Generate samples for metrics evaluation
                if self.metrics_ is not None and self.reverse_diffusion is not None:
                    xt = torch.randn_like(x).to(self.device)

                    # Reverse diffusion sampling
                    for t in reversed(range(self.hyper_params.num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, y_encoded)
                        noise = torch.randn_like(xt) if getattr(self.reverse_diffusion, "method", None) != "ode" else None
                        xt = self.reverse_diffusion(xt, noise, predicted_noise, time_steps)

                    # Clamp and normalize generated samples
                    x_hat = torch.clamp(xt, min=self.output_range[0], max=self.output_range[1])
                    if self.normalize_output:
                        x_hat = (x_hat - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
                        x_orig = (x_orig - self.output_range[0]) / (self.output_range[1] - self.output_range[0])

                    # Compute metrics
                    metrics_result = self.metrics_.forward(x_orig, x_hat)
                    fid, mse, psnr, ssim, lpips_score = metrics_result

                    if hasattr(self.metrics_, 'fid') and self.metrics_.fid:
                        fid_scores.append(fid)
                    if hasattr(self.metrics_, 'metrics') and self.metrics_.metrics:
                        mse_scores.append(mse)
                        psnr_scores.append(psnr)
                        ssim_scores.append(ssim)
                    if hasattr(self.metrics_, 'lpips') and self.metrics_.lpips:
                        lpips_scores.append(lpips_score)

        # Compute average metrics
        val_loss = torch.tensor(val_losses).mean().item()

        # All-reduce validation metrics across processes for DDP
        if self.ddp:
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()

        fid_avg = torch.tensor(fid_scores).mean().item() if fid_scores else float('inf')
        mse_avg = torch.tensor(mse_scores).mean().item() if mse_scores else None
        psnr_avg = torch.tensor(psnr_scores).mean().item() if psnr_scores else None
        ssim_avg = torch.tensor(ssim_scores).mean().item() if ssim_scores else None
        lpips_avg = torch.tensor(lpips_scores).mean().item() if lpips_scores else None

        # Return to training mode
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()

        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg


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
    """
    def __init__(
            self,
            reverse_diffusion: torch.nn.Module,
            noise_predictor: torch.nn.Module,
            image_shape: Tuple[int, int],
            conditional_model: Optional[torch.nn.Module] = None,
            tokenizer: str = "bert-base-uncased",
            max_length: int = 77,
            batch_size: int = 1,
            in_channels: int = 3,
            device: Optional[Union[str, torch.device]] = None,
            output_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> None:
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

    def tokenize(self, prompts: Union[str, List]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes text prompts for conditional generation.

        Converts input prompts into tokenized tensors using the specified tokenizer.

        Parameters
        ----------
        prompts : str or list
            Text prompt(s) for conditional generation. Can be a single string or a list
            of strings.

        Returns
        -------
        input_ids : torch.Tensor
             Tokenized input IDs, shape (batch_size, max_length).
        attention_mask : torch.Tensor
            Attention mask, shape (batch_size, max_length).
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

    def forward(
            self,
            conditions: Optional[Union[str, List]] = None,
            normalize_output: bool = True,
            save_images: bool = True,
            save_path: str = "sde_generated"
    ) -> torch.Tensor:
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
        generated_imgs (torch.Tensor) - Generated images, shape (batch_size, in_channels, height, width). If `normalize_output` is True, images are normalized to [0, 1]; otherwise, they are clamped to `output_range`.
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

    def to(self, device: torch.device) -> Self:
        """Moves the module and its components to the specified device.

        Updates the device attribute and moves the reverse diffusion, noise predictor,
        and conditional model (if present) to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device for the module and its components.

        Returns
        -------
        sample_sde (SampleSDE) - moved to the specified device.
        """
        self.device = device
        self.noise_predictor.to(device)
        self.reverse.to(device)
        if self.conditional_model:
            self.conditional_model.to(device)
        return super().to(device)


from utils import NoisePredictor, Metrics, TextEncoder
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For RGB
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use small subset
train_subset = Subset(train_dataset, torch.randperm(len(train_dataset))[:100])
test_subset = Subset(test_dataset, torch.randperm(len(test_dataset))[:10])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_subset, batch_size=10, shuffle=False)

# Initialize the NoisePredictor for the SDE model with parameters for RGB images
noise_predictor = NoisePredictor(
        in_channels=3,  # RGB images
        down_channels=[16, 32],
        mid_channels=[32, 32],
        up_channels=[32, 16],
        down_sampling=[True, True],
        time_embed_dim=64,
        y_embed_dim=64,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        down_sampling_factor=2
)

# label conditional model
text_encoder = TextEncoder(
        use_pretrained_model=True,
        model_name="bert-base-uncased",
        vocabulary_size=30522,
        num_layers=2,
        input_dimension=64,
        output_dimension=64,
        num_heads=2,
        context_length=77
)

# Set up the AdamW optimizer for the NoisePredictor parameters plus TextEncoder trainable parameters with a learning rate of 1e-3
optimizer = torch.optim.Adam(
    [p for p in noise_predictor.parameters() if p.requires_grad] +
    [p for p in text_encoder.parameters() if p.requires_grad], lr=1e-3
)

# Initialize the Mean Squared Error (MSE) loss function
loss = nn.MSELoss()

# Configure the Metrics class for evaluation on CPU (GPUs are recommended for actual training)
metrics = Metrics(
    device="cuda", #"cpu",  # Using CPU for this tutorial, but GPUs are recommended for training diffusion models
    fid=True,
    metrics=True,
    lpips_=True
)


# Initialize SDE hyperparameters for the noise schedule
hyperparams_sde = HyperParamsSDE(
    num_steps=500,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=False,
    sigma_start=1e-3,
    sigma_end=10.0,
    start=0.0,
    end=1.0,
    beta_method="linear"
)

# Set up the reverse diffusion process for sampling
reverse_sde = ReverseSDE(
    hyper_params=hyperparams_sde,
    method="ode" # there are three stochastic methods and one deterministic method available ("ve", "vp", "sub-vp", "ode"). Here we will use a stochastic method
)


# Configure the SDE trainer for model training
trainer = TrainSDE(
    method="ode", # "ve", "vp", "sub-vp", "ode"
    noise_predictor=noise_predictor,
    hyper_params=hyperparams_sde,
    conditional_model=text_encoder,
    metrics_=metrics,
    optimizer=optimizer,
    objective=loss,
    data_loader=train_loader,
    val_loader=val_loader,
    max_epoch=5,
    device=torch.device("cuda"),
    store_path="test_sde",
    val_frequency=3,
    ddp=False,
    num_grad_accumulation=1,
    progress_frequency=1
)

train_losses, best_val_loss = trainer()








