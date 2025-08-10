

import torch
import torch.nn as nn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from transformers import BertTokenizer
import warnings
from torchvision.utils import save_image
from typing import Optional, Tuple, Callable, List, Any, Union, Self
import os





class ForwardDDIM(nn.Module):
    """Forward diffusion process of DDIM.

    Implements the forward diffusion process for Denoising Diffusion Implicit Models (DDIM),
    which perturbs input data by adding Gaussian noise over a series of time steps,
    as defined in Song et al. (2021, "Denoising Diffusion Implicit Models").

    Parameters
    ----------
    `variance_scheduler` : object
        Variance-scheduler object (VarianceSchedulerDDIM) containing the noise schedule parameters.
        Expected to have attributes: `num_steps`, `trainable_beta`, `betas`, `sqrt_alpha_cumprod`,
        `sqrt_one_minus_alpha_cumprod`, `compute_schedule`
    """

    def __init__(self, variance_scheduler: torch.nn.Module) -> None:
        super().__init__()
        self.variance_scheduler = variance_scheduler

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Applies the forward diffusion process to the input data.

        Perturbs the input data `x0` by adding Gaussian noise according to the DDIM
        forward process at specified time steps, using cumulative noise schedule parameters.

        Parameters
        ----------
        `x0` : torch.Tensor
            Input data tensor of shape (batch_size, channels, height, width).
        `noise` : torch.Tensor
            Gaussian noise tensor of the same shape as `x0`.
        `time_steps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,),
            where each value is in the range [0, hyper_params.num_steps - 1].

        Returns
        -------
        xt (torch.Tensor) - Noisy data tensor `xt` at the specified time steps, with the same shape as `x0`.
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
    `variance_scheduler` : object
        Variance-scheduler object (VarianceSchedulerDDIM) containing the noise schedule parameters.
        Expected to have attributes: `tau_num_steps`, `eta`, `get_tau_schedule`.
    """

    def __init__(self, variance_scheduler: torch.nn.Module):
        super().__init__()
        self.variance_scheduler = variance_scheduler

    def forward(self, xt: torch.Tensor, predicted_noise: torch.Tensor, time_steps: torch.Tensor, prev_time_steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the reverse diffusion process to the noisy input.

        Denoises the input `xt` at time step `t` to produce the previous step `xt_prev`
        at `prev_time_steps` using the predicted noise and the DDIM reverse process.
        Optionally includes stochastic noise scaled by `eta`.

        Parameters
        ----------
        `xt` : torch.Tensor
            Noisy input tensor at time step `t`, shape (batch_size, channels, height, width).
        `predicted_noise` : torch.Tensor
            Predicted noise tensor, same shape as `xt`, typically output by a neural network.
        `time_steps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, hyper_params.tau_num_steps - 1].
        `prev_time_steps` : torch.Tensor
            Tensor of previous time step indices (long), shape (batch_size,), where each
            value is in the range [0, hyper_params.tau_num_steps - 1].

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

        _, _, _, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod =self.variance_scheduler.get_tau_schedule()
        tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
        tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)

        eta = self.variance_scheduler.eta
        x0 = (xt - tau_sqrt_one_minus_alpha_cumprod_t * predicted_noise) / tau_sqrt_alpha_cumprod_t
        noise_coeff = eta * ((tau_sqrt_one_minus_alpha_cumprod_t / prev_tau_sqrt_alpha_cumprod_t) *
                             prev_tau_sqrt_one_minus_alpha_cumprod_t / torch.clamp(tau_sqrt_one_minus_alpha_cumprod_t, min=1e-8))
        direction_coeff = torch.clamp(prev_tau_sqrt_one_minus_alpha_cumprod_t ** 2 - noise_coeff ** 2, min=1e-8).sqrt()
        xt_prev = prev_tau_sqrt_alpha_cumprod_t * x0 + noise_coeff * torch.randn_like(xt) + direction_coeff * predicted_noise

        return xt_prev, x0


###==================================================================================================================###

class VarianceSchedulerDDIM(nn.Module):
    """Variance-scheduler for DDIM noise schedule with flexible beta computation.

    Manages the noise schedule parameters for DDIM, including beta values, derived
    quantities (alphas, alpha_cumprod, etc.), and a subsampled time step schedule
    (tau schedule), as inspired by Song et al. (2021). Supports trainable or fixed
    schedules and various beta scheduling methods.

    Parameters
    ----------
    `eta` : float, optional
        Noise scaling factor for the DDIM reverse process (default: 0, deterministic).
    `num_steps` : int, optional
        Total number of diffusion steps (default: 1000).
    `tau_num_steps` : int, optional
        Number of subsampled time steps for DDIM sampling (default: 100).
    `beta_start` : float, optional
        Starting value for beta (default: 1e-4).
    `beta_end` : float, optional
        Ending value for beta (default: 0.02).
    `trainable_beta` : bool, optional
        Whether the beta schedule is trainable (default: False).
    `beta_method` : str, optional
        Method for computing the beta schedule (default: "linear").
        Supported methods: "linear", "sigmoid", "quadratic", "constant", "inverse_time".
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
    ):
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
            # Use reparameterization trick for trainable betas
            # Initialize unconstrained parameters and transform them to valid beta range
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
        """Returns the beta values, applying reparameterization if trainable."""
        if self.trainable_beta:
            # Transform unconstrained parameters to valid beta range using sigmoid
            return self.beta_start + (self.beta_end - self.beta_start) * torch.sigmoid(self.beta_raw)
        # Return the registered buffer directly if it exists
        #return getattr(self, '_buffers', {}).get('betas_buffer', None) or ValueError("Betas buffer not found")
        return self._buffers['betas_buffer']


    def compute_beta_schedule(self, beta_range: Tuple[float, float], num_steps: int, method: str) -> torch.Tensor:
        """Computes the beta schedule based on the specified method.

        Generates a sequence of beta values for the DDIM noise schedule using the
        chosen method, ensuring values are clamped within the specified range.

        Parameters
        ----------
        `beta_range` : tuple
            Tuple of (min_beta, max_beta) specifying the valid range for beta values.
        `num_steps` : int
            Number of diffusion steps.
        `method` : str
            Method for computing the beta schedule. Supported methods:
            "linear", "sigmoid", "quadratic", "constant", "inverse_time".

        Returns
        -------
        beta (torch.Tensor) - Tensor of beta values, shape (num_steps,).
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

    def get_tau_schedule(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the subsampled (tau) noise schedule for DDIM.

        Returns the noise schedule parameters for the subsampled time steps used in
        DDIM sampling, based on the `tau_indices`.

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
            # Use the property to get constrained betas
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
        from the provided beta values, as used in the DDIM forward and reverse processes.

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
        # Use the property to get constrained betas
        betas = self.betas
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

        if time_steps is not None:
            return (betas[time_steps], alphas[time_steps], alpha_cumprod[time_steps],
                    sqrt_alpha_cumprod[time_steps], sqrt_one_minus_alpha_cumprod[time_steps])
        else:
            return betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod


###==================================================================================================================###


class TrainDDIM(nn.Module):
    """Trainer for Denoising Diffusion Implicit Models (DDIM).

    Manages the training process for DDIM, optimizing a noise predictor model to learn
    the noise added by the forward diffusion process. Supports conditional training with
    text prompts, mixed precision training, learning rate scheduling, early stopping, and
    checkpointing, as inspired by Song et al. (2021).

    Parameters
    ----------
    `noise_predictor` : nn.Module
        Model to predict noise added during the forward diffusion process.
    `variance_scheduler` : nn.Module
        Variance-scheduler module (e.g., VarianceSchedulerDDIM) defining the noise schedule.
    `data_loader` : torch.utils.data.DataLoader
        DataLoader for training data.
    `optimizer` : torch.optim.Optimizer
        Optimizer for training the noise predictor and conditional model (if applicable).
    `objective` : callable
        Loss function to compute the difference between predicted and actual noise.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `max_epoch` : int, optional
        Maximum number of training epochs (default: 1000).
    `device` : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    `conditional_model` : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    `metrics_` : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    `tokenizer` : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    `max_length` : int, optional
        Maximum length for tokenized prompts (default: 77).
    `store_path` : str, optional
        Path to save model checkpoints (default: "ddim_model.pth").
    `patience` : int, optional
        Number of epochs to wait for improvement before early stopping (default: 100).
    `warmup_epochs` : int, optional
        Number of epochs for learning rate warmup (default: 100).
    `val_frequency` : int, optional
        Frequency (in epochs) for validation (default: 10).
    `output_range` : tuple, optional
        Range for clamping generated images (default: (-1, 1)).
    `normalize_output` : bool, optional
        Whether to normalize generated images to [0, 1] for metrics (default: True).
    `ddp` : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    `num_grad_accumulation` : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    `progress_frequency` : int, optional
        Number of epochs before printing loss.
    compilation : bool, optional
        whether the model is internally compiled using torch.compile (default: false)
    """
    def __init__(
            self,
            noise_predictor: torch.nn.Module,
            variance_scheduler: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epoch: int = 1000,
            device: str = None,
            conditional_model: torch.nn.Module = None,
            metrics_: Optional[Any] = None,
            tokenizer: Optional[BertTokenizer] = None,
            max_length: int = 77,
            store_path: Optional[str] = None,
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            output_range: Tuple[float, float] = (-1, 1),
            normalize_output: bool = True,
            ddp: bool = False,
            num_grad_accumulation: int = 1,
            progress_frequency: int = 1,
            compilation: bool = False
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
        self.noise_predictor = noise_predictor.to(self.device)
        self.variance_scheduler = variance_scheduler.to(self.device)
        self.forward_diffusion = ForwardDDIM(variance_scheduler=self.variance_scheduler).to(self.device)
        self.reverse_diffusion = ReverseDDIM(variance_scheduler=self.variance_scheduler).to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None

        # Training components
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or "ddim_model"
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.max_length = max_length
        self.patience = patience
        self.val_frequency = val_frequency
        self.output_range = output_range
        self.normalize_output = normalize_output
        self.progress_frequency = progress_frequency
        self.compilation = compilation

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

        # Load hyper_params state
        if 'variance_scheduler_model' not in checkpoint:
            raise KeyError("Checkpoint missing 'variance_scheduler_model' key")
        try:
            if isinstance(self.variance_scheduler, nn.Module):
                self.variance_scheduler.load_state_dict(checkpoint['variance_scheduler_model'])
            else:
                self.variance_scheduler = checkpoint['variance_scheduler_model']
        except Exception as e:
            warnings.warn(f"Variance_scheduler loading failed: {e}. Continuing with current variance_scheduler.")

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
        `optimizer` : torch.optim.Optimizer
            Optimizer to apply the scheduler to.
        `warmup_epochs` : int
            Number of epochs for the warmup phase.

        Returns
        -------
        lr_scheduler (torch.optim.lr_scheduler.LambdaLR) - Learning rate scheduler for warmup.
        """
        def lr_lambda(epoch: int) -> float:
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
        """Trains the DDIM model to predict noise added by the forward diffusion process.

        Executes the training loop, optimizing the noise predictor and conditional model
        (if applicable) using mixed precision, gradient clipping, and learning rate
        scheduling. Supports validation, early stopping, and checkpointing.

        Returns
        -------
        train_losses : list of float
             List of mean training losses per epoch.
        best_val_loss : float
             Best validation or training loss achieved.
        """

        # Set models to training mode
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()

        # Compile models for optimization (if supported)
        if self.compilation:
            try:
                self.noise_predictor = torch.compile(self.noise_predictor)
                if self.conditional_model is not None:
                    self.conditional_model = torch.compile(self.conditional_model)
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

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
                    t = torch.randint(0, self.variance_scheduler.num_steps, (x.shape[0],)).to(self.device)
                    assert x.device == noise.device == t.device, "Device mismatch detected"
                    assert t.shape[0] == x.shape[0], "Timestep batch size mismatch"

                    # Apply forward diffusion
                    noisy_x = self.forward_diffusion(x, noise, t)

                    # Predict noise
                    p_noise = self.noise_predictor(x=noisy_x, t=t, y=y_encoded)

                    # Compute loss and scale for gradient accumulation
                    loss = self.objective(p_noise, noise) / self.num_grad_accumulation

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

            train_losses.append(mean_train_loss)

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
                'variance_scheduler_model': (
                    self.variance_scheduler.state_dict() if isinstance(self.variance_scheduler, nn.Module)
                    else self.variance_scheduler
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
                t = torch.randint(0, self.variance_scheduler.num_steps, (x.shape[0],)).to(self.device)

                noisy_x = self.forward_diffusion(x, noise, t)
                predicted_noise = self.noise_predictor(noisy_x, t, y_encoded)
                loss = self.objective(predicted_noise, noise)
                val_losses.append(loss.item())

                # Generate samples for metrics evaluation
                if self.metrics_ is not None and self.reverse_diffusion is not None:
                    xt = torch.randn_like(x).to(self.device)

                    # Reverse diffusion sampling
                    for t in reversed(range(self.variance_scheduler.tau_num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        prev_time_steps = torch.full((xt.shape[0],), max(t - 1, 0), device=self.device, dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, y_encoded)
                        xt, _ = self.reverse_diffusion(xt, predicted_noise, time_steps, prev_time_steps)

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

class SampleDDIM(nn.Module):
    """Image generation using a trained DDIM model.

    Implements the sampling process for DDIM, generating images by iteratively denoising
    random noise using a trained noise predictor and reverse diffusion process with a
    subsampled time step schedule. Supports conditional generation with text prompts,
    as inspired by Song et al. (2021).

    Parameters
    ----------
    `reverse_diffusion` : nn.Module
        Reverse diffusion module (e.g., ReverseDDIM) for the reverse process.
    `noise_predictor` : nn.Module
        Trained model to predict noise at each time step.
    `image_shape` : tuple
        Tuple of (height, width) specifying the generated image dimensions.
    `conditional_model` : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    `tokenizer` : str, optional
        Pretrained tokenizer name from Hugging Face (default: "bert-base-uncased").
    `max_length` : int, optional
        Maximum length for tokenized prompts (default: 77).
    `batch_size` : int, optional
        Number of images to generate per batch (default: 1).
    `in_channels` : int, optional
        Number of input channels for generated images (default: 3).
    `device` : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    `output_range` : tuple, optional
        Tuple of (min, max) for clamping generated images (default: (-1, 1)).
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
            device: Optional[str] = None,
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

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(
                isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(output_range, (tuple, list)) or len(output_range) != 2 or output_range[0] >= output_range[1]:
            raise ValueError("output_range must be a tuple (min, max) with min < max")


    def tokenize(self, prompts: Union[List, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes text prompts for conditional generation.

        Converts input prompts into tokenized input IDs and attention masks using the
        specified tokenizer, suitable for use with the conditional model.

        Parameters
        ----------
        `prompts` : str or list
            A single text prompt or a list of text prompts.

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

    def forward(self, conditions: Optional[Union[str, List]] = None, normalize_output: bool = True, save_images: bool = True, save_path: str = "ddim_generated") -> torch.Tensor:
        """Generates images using the DDIM sampling process.

        Iteratively denoises random noise to generate images using the reverse diffusion
        process with a subsampled time step schedule and noise predictor. Supports
        conditional generation with text prompts.

        Parameters
        ----------
        `conditions` : str or list, optional
            Text prompt(s) for conditional generation, default None.
        `normalize_output` : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        `save_images` : bool, optional
            If True, saves generated images to `save_path` (default: True).
        `save_path` : str, optional
            Directory to save generated images (default: "ddim_generated").

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

    def to(self, device: torch.device) -> Self:
        """Moves the module and its components to the specified device.

        Updates the device attribute and moves the reverse diffusion, noise predictor,
        and conditional model (if present) to the specified device.

        Parameters
        ----------
        `device` : torch.device
            Target device for the module and its components.

        Returns
        -------
        sample_ddim (SampleDDIM) - moved to the specified device.
        """
        self.device = device
        self.noise_predictor.to(device)
        self.reverse.to(device)
        if self.conditional_model:
            self.conditional_model.to(device)
        return super().to(device)


"""
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the FashionMNIST training dataset (28x28 grayscale images)
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Load the FashionMNIST test dataset (28x28 grayscale images)
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Define subset sizes for training (200 samples) and validation (20 samples)
train_subset_indices = torch.randperm(len(train_dataset))[:100]
test_subset_indices = torch.randperm(len(test_dataset))[:10]

# Create subsets from the FashionMNIST training and test datasets
train_subset = Subset(train_dataset, train_subset_indices)
test_subset = Subset(test_dataset, test_subset_indices)

# Create DataLoaders for the training and validation subsets
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_subset, batch_size=10, shuffle=False, drop_last=False)

# Initialize the NoisePredictor for the DDPM model with parameters for grayscale images
noise_predictor = NoisePredictor(
        in_channels=1,  # Single channel for grayscale images in the training data
        down_channels=[16, 32],
        mid_channels=[32, 32],
        up_channels=[32, 16],
        down_sampling=[True, True],
        time_embed_dim=32,
        y_embed_dim=32,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        down_sampling_factor=2
).to(device)

# label conditional model
text_encoder = TextEncoder(
    use_pretrained_model=True,
    model_name="bert-base-uncased",
    vocabulary_size=30522,
    num_layers=2,
    input_dimension=32,
    output_dimension=32,
    num_heads=2,
    context_length=77
).to(device)

# Set up the AdamW optimizer for the NoisePredictor parameters with a learning rate of 1e-3
optimizer = torch.optim.AdamW(
    [p for p in noise_predictor.parameters() if p.requires_grad] +
    [p for p in text_encoder.parameters() if p.requires_grad], lr=1e-3
)

# Initialize the Mean Squared Error (MSE) loss function
loss = nn.MSELoss()

# Configure the Metrics class for evaluation on CPU (GPUs are recommended for actual training)
metrics = Metrics(
    device="cpu",  # Using CPU for this tutorial, but GPUs are recommended for training diffusion models
    fid=True,
    metrics=True,
    lpips_=True
)


# Initialize DDPM hyperparameters for the noise schedule
hyperparams_ddim = HyperParamsDDIM(
    num_steps=500,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=False,
    beta_method="linear"
)

# Set up the reverse diffusion process for sampling
reverse_ddim = ReverseDDIM(hyperparams_ddim)

# Configure the DDPM trainer for model training
train_ddim = TrainDDIM(
    noise_predictor=noise_predictor,
    hyper_params=hyperparams_ddim,
    conditional_model=text_encoder,
    metrics_=metrics,
    optimizer=optimizer,
    objective=loss,
    data_loader=train_loader,
    val_loader=val_loader,
    max_epoch=5,
    device="cuda",
    store_path="test_ddim",
    val_frequency=3,
    ddp=False,
    num_grad_accumulation=3,
    progress_frequency=3,
    compilation=True
)

train_losses, best_val_loss = train_ddim()
"""


