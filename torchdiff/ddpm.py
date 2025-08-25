"""
**Denoising Diffusion Probabilistic Models (DDPM) implementation**

This module provides a complete implementation of DDPM, as described in Ho et al.
(2020, "Denoising Diffusion Probabilistic Models"). It includes components for forward
and reverse diffusion processes, hyperparameter management, training, and image
sampling. Supports both unconditional and conditional generation with text prompts.

**Components**

- **ForwardDDPM**: Forward diffusion process to add noise.
- **ReverseDDPM**: Reverse diffusion process to denoise.
- **VarianceSchedulerDDPM**: Noise schedule management.
- **TrainDDPM**: Training loop with mixed precision and scheduling.
- **SampleDDPM**: Image generation from trained models.

**References**

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.

- Salimans, Tim, et al. "Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications."
arXiv preprint arXiv:1701.05517 (2017).

-------------------------------------------------------------------------------
"""



import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, List, Any, Union, Self
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from transformers import BertTokenizer
import warnings
from torchvision.utils import save_image
import os


###==================================================================================================================###


class ForwardDDPM(nn.Module):
    """Forward diffusion process for Denoising Diffusion Probabilistic Models (DDPM).

    Implements the forward diffusion process for DDPM, which perturbs input data by
    adding Gaussian noise over a series of time steps, as defined in Ho et al. (2020).
    The noise schedule can be either fixed or trainable, depending on the provided
    hyperparameters.

    Parameters
    ----------
    variance_scheuler : object
        Hyperparameter object (VarianceSchedulerDDPM) containing the noise schedule parameters. Expected to have
        attributes: `num_steps`, `trainable_beta`, `betas`, `sqrt_alpha_bars`, `sqrt_one_minus_alpha_bars`, `compute_schedule`.
    """
    def __init__(self, variance_scheduler: torch.nn.Module) -> None:
        super().__init__()
        self.variance_scheduler = variance_scheduler

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Applies the forward diffusion process to the input data.

        Perturbs the input data `x0` by adding Gaussian noise according to the DDPM
        forward process at specified time steps. Uses the reparameterization trick:
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε.

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
        xt : torch.Tensor
            Noisy data tensor `xt` at the specified time steps, with the same shape as `x0`.
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.variance_scheduler.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.variance_scheduler.num_steps - 1}")

        if self.variance_scheduler.trainable_beta:
            _, _, _, sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = self.variance_scheduler.compute_schedule(time_steps)
            sqrt_alpha_bar_t = sqrt_alpha_bar_t.to(x0.device)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.to(x0.device)
        else:
            sqrt_alpha_bar_t = self.variance_scheduler.sqrt_alpha_bars[time_steps].to(x0.device)
            sqrt_one_minus_alpha_bar_t = self.variance_scheduler.sqrt_one_minus_alpha_bars[time_steps].to(x0.device)

        sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(-1, 1, 1, 1)
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return xt

###==================================================================================================================###


class ReverseDDPM(nn.Module):
    """Reverse diffusion process for Denoising Diffusion Probabilistic Models (DDPM).

    Implements the reverse diffusion process for DDPM, which iteratively denoises a
    noisy input `xt` using a predicted noise component, as defined in Ho et al. (2020).
    The process relies on a noise schedule that can be either fixed or trainable,
    specified through the provided hyperparameters.

    Parameters
    ----------
    variance_scheduler : object
        Hyperparameter object (VarianceSchedulerDDPM) containing the noise schedule parameters. Expected to have
        attributes: `num_steps`, `trainable_beta`, `betas`, `alphas`, `alpha_bars`, `compute_schedule`.
    """
    def __init__(self, variance_scheduler: torch.nn.Module) -> None:
        super().__init__()
        self.variance_scheduler = variance_scheduler

    def forward(self, xt: torch.Tensor, predicted_noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Applies the reverse diffusion process to the noisy input.

        Denoises the input `xt` by computing the mean of the reverse process
        distribution using the predicted noise and optionally adding stochastic noise
        for time steps greater than 0, as per the DDPM reverse process.

        Parameters
        ----------
        xt : torch.Tensor
            Noisy input tensor at time step `t`, of shape (batch_size, channels, height, width).
        predicted_noise : torch.Tensor
            Predicted noise tensor, of the same shape as `xt`, typically output by a neural network.
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, hyper_params.num_steps - 1].

        Returns
        -------
        xt_minus_1 : torch.Tensor
            Denoised tensor `xt_minus_1` at time step `t-1`, with the same shape as `xt`.
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.variance_scheduler.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.variance_scheduler.num_steps - 1}")

        if self.variance_scheduler.trainable_beta:
            betas_t, alphas_t, alpha_bars_t, _, _ = self.variance_scheduler.compute_schedule(time_steps)
            betas_t = betas_t.to(xt.device)
            alphas_t = alphas_t.to(xt.device)
            alpha_bars_t = alpha_bars_t.to(xt.device)
            alpha_bars_t_minus_1 = torch.zeros_like(alpha_bars_t).to(xt.device)
            non_zero_mask = time_steps > 0
            if non_zero_mask.any():
                _, _, alpha_bars_t_minus_1_tmp, _, _ = self.variance_scheduler.compute_schedule(time_steps[non_zero_mask] - 1)
                alpha_bars_t_minus_1[non_zero_mask] = alpha_bars_t_minus_1_tmp.to(xt.device)
        else:
            betas_t = self.variance_scheduler.betas[time_steps].to(xt.device)
            alphas_t = self.variance_scheduler.alphas[time_steps].to(xt.device)
            alpha_bars_t = self.variance_scheduler.alpha_bars[time_steps].to(xt.device)
            alpha_bars_t_minus_1 = torch.zeros_like(alpha_bars_t).to(xt.device)
            non_zero_mask = time_steps > 0
            if non_zero_mask.any():
                alpha_bars_t_minus_1[non_zero_mask] = self.variance_scheduler.alpha_bars[time_steps[non_zero_mask] - 1].to(xt.device)

        sqrt_alphas_t = torch.sqrt(alphas_t).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bars_t = torch.sqrt(1 - alpha_bars_t).view(-1, 1, 1, 1)
        betas_t = betas_t.view(-1, 1, 1, 1)

        mu = (xt - (betas_t / sqrt_one_minus_alpha_bars_t) * predicted_noise) / sqrt_alphas_t

        mask = (time_steps == 0)
        if mask.all():
            return mu

        variance = (1 - alpha_bars_t_minus_1) / (1 - alpha_bars_t) * betas_t.squeeze()
        std = torch.sqrt(variance).view(-1, 1, 1, 1)

        z = torch.randn_like(xt).to(xt.device)
        xt_minus_1 = mu + (~mask).float().view(-1, 1, 1, 1) * std * z
        return xt_minus_1


###==================================================================================================================###


class VarianceSchedulerDDPM(nn.Module):
    """Hyperparameters for Denoising Diffusion Probabilistic Models (DDPM) noise schedule.

    Manages the noise schedule parameters for DDPM, including the computation of beta
    values and derived quantities (alphas, alpha_bars, etc.), with support for
    trainable or fixed schedules and various beta scheduling methods, as inspired by
    Ho et al. (2020).

    Parameters
    ----------
    num_steps : int, optional
        Number of diffusion steps (default: 1000).
    beta_start : float, optional
        Starting value for beta (default: 1e-4).
    beta_end : float, optional
        Ending value for beta (default: 0.02).
    trainable_beta : bool, optional
        Whether the beta schedule is trainable (default: False).
    beta_method : str, optional
        Method for computing the beta schedule (default: "linear").
        Supported methods: "linear", "sigmoid", "quadratic", "constant", "inverse_time".
    """
    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, trainable_beta: bool = False, beta_method: str = "linear") -> None:
        super().__init__()
        self.num_steps = num_steps
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
            self.betas = nn.Parameter(torch.log(betas_init))
        else:
            self.register_buffer('betas', betas_init)
            self.register_buffer('alphas', 1 - self.betas)
            self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
            self.register_buffer('sqrt_alpha_bars', torch.sqrt(self.alpha_bars))
            self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1 - self.alpha_bars))

    def compute_beta_schedule(self, beta_range: Tuple[float, float], num_steps: int, method: str) -> torch.Tensor:
        """Computes the beta schedule based on the specified method.

        Generates a sequence of beta values for the DDPM noise schedule using the
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
        beta (torch.Tensor) - Tensor of beta values, shape (num_steps,).
        """
        beta_min, beta_max = beta_range
        if method == "sigmoid":
            x = torch.linspace(-6, 6, num_steps)
            beta = torch.sigmoid(x) * (beta_max - beta_min) + beta_min
        elif method == "quadratic":
            x = torch.linspace(beta_min**0.5, beta_max**0.5, num_steps)
            beta = x**2
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

    def compute_schedule(self, time_steps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes noise schedule parameters dynamically from betas.

        Parameters-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ----------
        time_steps : torch.Tensor, optional
            Tensor of time step indices (long), shape (batch_size,). If None, returns parameters for all steps.

        Returns
        -------
        betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars : torch.Tensor
            Schedule parameters, shape (batch_size,) if time_steps is provided, else (num_steps,).
        """
        if self.trainable_beta:
            # Compute betas from trainable log_betas using sigmoid
            betas = torch.sigmoid(self.betas) * (self.beta_end - self.beta_start) + self.beta_start
        else:
            betas = self.betas

        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        if time_steps is not None:
            betas = betas[time_steps]
            alphas = alphas[time_steps]
            alpha_bars = alpha_bars[time_steps]

        return betas, alphas, alpha_bars, torch.sqrt(alpha_bars), torch.sqrt(1 - alpha_bars)


###==================================================================================================================###


class TrainDDPM(nn.Module):
    """Trainer for Denoising Diffusion Probabilistic Models (DDPM) with Multi-GPU Support.

    Manages the training process for DDPM, optimizing a noise predictor model to learn
    the noise added by the forward diffusion process. Supports conditional training with
    text prompts, mixed precision training, learning rate scheduling, early stopping,
    checkpointing, and distributed data parallel (DDP) training across multiple GPUs.

    Parameters
    ----------
    noise_predictor : nn.Module
        Model to predict noise added during the forward diffusion process.
    forward_diffusion : nn.Module
        Forward DDPM diffusion module for adding noise.
    reverse_diffusion: nn.Module
        Reverse DDPM diffusion module for denoising.
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data. Should be wrapped with DistributedSampler for DDP.
    optimizer : torch.optim.Optimizer
        Optimizer for training the noise predictor and conditional model (if applicable).
    objective : callable
        Loss function to compute the difference between predicted and actual noise.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    max_epochs : int, optional
        Maximum number of training epochs (default: 1000).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    conditional_model : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    metrics_ : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    tokenizer : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    store_path : str, optional
        Path to save model checkpoints (default: "ddpm_model").
    patience : int, optional
        Number of epochs to wait for improvement before early stopping (default: 100).
    warmup_epochs : int, optional
        Number of epochs for learning rate warmup (default: 100).
    val_frequency : int, optional
        Frequency (in epochs) for validation (default: 10).
    image_output_range : tuple, optional
        Range for clamping generated images (default: (-1, 1)).
    normalize_output : bool, optional
        Whether to normalize generated images to [0, 1] for metrics (default: True).
    use_ddp : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    grad_accumulation_steps : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    log_frequency : int, optional
        Number of epochs before printing loss.
    use_compilation : bool, optional
        whether the model is internally compiled using torch.compile (default: false)
    """

    def __init__(
            self,
            noise_predictor: torch.nn.Module,
            forward_diffusion: torch.nn.Module,
            reverse_diffusion: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            conditional_model: torch.nn.Module = None,
            metrics_: Optional[Any] = None,
            bert_tokenizer: Optional[BertTokenizer] = None,
            max_token_length: int = 77,
            store_path: Optional[str] = None,
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            image_output_range: Tuple[float, float] = (-1.0, 1.0),
            normalize_output: bool = True,
            use_ddp: bool = False,
            grad_accumulation_steps: int = 1,
            log_frequency: int = 1,
            use_compilation: bool = False
    ) -> None:
        super().__init__()

        # initialize DDP settings first
        self.use_ddp = use_ddp
        self.grad_accumulation_steps = grad_accumulation_steps
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # setup distributed training if enabled
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # move models to appropriate device
        self.noise_predictor = noise_predictor.to(self.device)
        self.forward_diffusion = forward_diffusion.to(self.device)
        self.reverse_diffusion = reverse_diffusion.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None

        # training components
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or "ddpm_model"
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.max_token_length = max_token_length
        self.patience = patience
        self.val_frequency = val_frequency
        self.image_output_range = image_output_range
        self.normalize_output = normalize_output
        self.log_frequency = log_frequency
        self.use_compilation = use_compilation

        # learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)

        # initialize tokenizer
        if bert_tokenizer is None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")
        else:
            self.tokenizer = bert_tokenizer

    def _setup_ddp(self) -> None:
        """Setup Distributed Data Parallel training configuration.

        Initializes process group, determines rank information, and sets up
        CUDA device for the current process.
        """
        # check if DDP environment variables are set
        if "RANK" not in os.environ:
            raise ValueError("DDP enabled but RANK environment variable not set")
        if "LOCAL_RANK" not in os.environ:
            raise ValueError("DDP enabled but LOCAL_RANK environment variable not set")
        if "WORLD_SIZE" not in os.environ:
            raise ValueError("DDP enabled but WORLD_SIZE environment variable not set")

        # ensure CUDA is available for DDP
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA but CUDA is not available")

        # initialize process group only if not already initialized
        if not torch.distributed.is_initialized():
            init_process_group(backend="nccl")

        # get rank information
        self.ddp_rank = int(os.environ["RANK"])  # global rank across all nodes
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])  # local rank on current node
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])  # total number of processes

        # set device and make it current
        self.device = torch.device(f"cuda:{self.ddp_local_rank}")
        torch.cuda.set_device(self.device)

        # master process handles logging, checkpointing, etc.
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
            # load checkpoint with proper device mapping
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        # load noise predictor state
        if 'model_state_dict_noise_predictor' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict_noise_predictor' key")

        # handle DDP wrapped model state dict
        state_dict = checkpoint['model_state_dict_noise_predictor']
        if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
            # if loading non-DDP checkpoint into DDP model, add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
            # if loading DDP checkpoint into non-DDP model, remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.noise_predictor.load_state_dict(state_dict)

        # load conditional model state if applicable
        if self.conditional_model is not None:
            if 'model_state_dict_conditional' in checkpoint and checkpoint['model_state_dict_conditional'] is not None:
                cond_state_dict = checkpoint['model_state_dict_conditional']
                # handle DDP wrapping for conditional model
                if self.use_ddp and not any(key.startswith('module.') for key in cond_state_dict.keys()):
                    cond_state_dict = {f'module.{k}': v for k, v in cond_state_dict.items()}
                elif not self.use_ddp and any(key.startswith('module.') for key in cond_state_dict.keys()):
                    cond_state_dict = {k.replace('module.', ''): v for k, v in cond_state_dict.items()}
                self.conditional_model.load_state_dict(cond_state_dict)
            else:
                warnings.warn(
                    "Checkpoint contains no 'model_state_dict_conditional' or it is None, "
                    "skipping conditional model loading"
                )

        # load variance_scheduler state
        if 'variance_scheduler_model' not in checkpoint:
            raise KeyError("Checkpoint missing 'variance_scheduler_model' key")
        try:
            if isinstance(self.forward_diffusion.variance_scheduler, nn.Module):
                self.forward_diffusion.variance_scheduler.load_state_dict(
                    checkpoint['variance_scheduler_model'])
            if isinstance(self.reverse_diffusion.variance_scheduler, nn.Module):
                self.reverse_diffusion.variance_scheduler.load_state_dict(
                    checkpoint['variance_scheduler_model'])
            else:
                self.forward_diffusion.variance_scheduler = checkpoint['variance_scheduler_model']
                self.reverse_diffusion.variance_scheduler = checkpoint['variance_scheduler_model']
        except Exception as e:
            warnings.warn(
                f"Variance_scheduler loading failed: {e}. Continuing with current variance_scheduler.")

        # load optimizer state
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
        if self.use_ddp:
            # wrap noise predictor with DDP
            self.noise_predictor = DDP(
                self.noise_predictor,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

            # wrap conditional model with DDP if it exists
            if self.conditional_model is not None:
                self.conditional_model = DDP(
                    self.conditional_model,
                    device_ids=[self.ddp_local_rank],
                    find_unused_parameters=True
                )

    def forward(self) -> Tuple[List, float]:
        """Trains the DDPM model to predict noise added by the forward diffusion process.

        Executes the training loop with support for distributed training, gradient accumulation,
        mixed precision, gradient clipping, and learning rate scheduling. Includes validation,
        early stopping, and checkpointing functionality.

        Returns
        -------
        train_losses : list of float
             List of mean training losses per epoch.
        best_val_loss : float
             Best validation or training loss achieved.
        """
        # set models to training mode
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()
        if self.forward_diffusion.variance_scheduler.trainable_beta:
            self.reverse_diffusion.train()
            self.forward_diffusion.train()
        else:
            self.reverse_diffusion.eval()
            self.forward_diffusion.eval()

        # compile models for optimization (if supported)
        if self.use_compilation:
            try:
                self.noise_predictor = torch.compile(self.noise_predictor)
                if self.conditional_model is not None:
                    self.conditional_model = torch.compile(self.conditional_model)
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")


        # wrap models for DDP after compilation
        self._wrap_models_for_ddp()

        # initialize training components
        scaler = torch.GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        # main training loop
        for epoch in range(self.max_epochs):
            # set epoch for distributed sampler if using DDP
            if self.use_ddp and hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []

            # training step loop with gradient accumulation
            for step, (x, y) in enumerate(tqdm(self.data_loader, disable=not self.master_process)):
                x = x.to(self.device)

                # process conditional inputs if conditional model exists
                if self.conditional_model is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None

                # forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    # generate noise and timesteps
                    noise = torch.randn_like(x).to(self.device)
                    t = torch.randint(0, self.forward_diffusion.variance_scheduler.num_steps, (x.shape[0],)).to(self.device)

                    # apply forward diffusion
                    noisy_x = self.forward_diffusion(x, noise, t)

                    # predict noise
                    predicted_noise = self.noise_predictor(noisy_x, t, y_encoded, None)

                    # compute loss and scale for gradient accumulation
                    loss = self.objective(predicted_noise, noise) / self.grad_accumulation_steps

                # backward pass
                scaler.scale(loss).backward()

                # gradient accumulation and optimizer step
                if (step + 1) % self.grad_accumulation_steps == 0:
                    # clip gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.noise_predictor.parameters(), max_norm=1.0)
                    if self.conditional_model is not None:
                        torch.nn.utils.clip_grad_norm_(self.conditional_model.parameters(), max_norm=1.0)

                    # optimizer step
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                    # update learning rate (warmup scheduler)
                    self.warmup_lr_scheduler.step()

                # record loss (unscaled)
                train_losses_epoch.append(loss.item() * self.grad_accumulation_steps)

            # compute mean training loss
            mean_train_loss = torch.tensor(train_losses_epoch).mean().item()

            # all-reduce loss across processes for DDP
            if self.use_ddp:
                loss_tensor = torch.tensor(mean_train_loss, device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                mean_train_loss = loss_tensor.item()

            train_losses.append(mean_train_loss)

            # print training progress (only master process)
            if self.master_process and (epoch + 1) % self.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\nEpoch: {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}")

            # validation step
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

            # save checkpoint and early stopping (only master process)
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

        # clean up DDP
        if self.use_ddp:
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
            max_length=self.max_token_length,
            return_tensors="pt"
        ).to(self.device)

        # get embeddings
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
            # get state dicts, handling DDP wrapping
            noise_predictor_state = (
                self.noise_predictor.module.state_dict() if self.use_ddp
                else self.noise_predictor.state_dict()
            )
            conditional_state = None
            if self.conditional_model is not None:
                conditional_state = (
                    self.conditional_model.module.state_dict() if self.use_ddp
                    else self.conditional_model.state_dict()
                )

            checkpoint = {
                'epoch': epoch,
                'model_state_dict_noise_predictor': noise_predictor_state,
                'model_state_dict_conditional': conditional_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'variance_scheduler_model': (
                    self.forward_diffusion.variance_scheduler.state_dict() if isinstance(self.forward_diffusion.variance_scheduler, nn.Module)
                    else self.forward_diffusion.variance_scheduler
                ),
                'max_epochs': self.max_epochs,
            }

            filename = f"ddpm_epoch_{epoch}{suffix}.pth"
            filepath = os.path.join(self.store_path, filename)
            os.makedirs(self.store_path, exist_ok=True)
            torch.save(checkpoint, filepath)

            print(f"Model saved at epoch {epoch}")

        except Exception as e:
            print(f"Failed to save model: {e}")



    def validate(self) -> Tuple[float, float, float, float, float, float]:
        """Validates the noise predictor and computes evaluation metrics.

        Computes validation loss (MSE between predicted and ground truth noise) and generates
        samples using the reverse diffusion model. Evaluates image quality metrics if available.

        Returns
        -------
        tuple
            (val_loss, fid, mse, psnr, ssim, lpips_score) where metrics may be None if not computed.
        """
        self.noise_predictor.eval()
        if self.conditional_model is not None:
            self.conditional_model.eval()
        if self.forward_diffusion.variance_scheduler.trainable_beta:
            self.forward_diffusion.eval()
            self.reverse_diffusion.eval()

        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                x_orig = x.clone()

                # process conditional input
                if self.conditional_model is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None

                # compute validation loss
                noise = torch.randn_like(x).to(self.device)
                t = torch.randint(0, self.forward_diffusion.variance_scheduler.num_steps, (x.shape[0],)).to(self.device)

                noisy_x = self.forward_diffusion(x, noise, t)
                predicted_noise = self.noise_predictor(noisy_x, t, y_encoded, None)
                loss = self.objective(predicted_noise, noise)
                val_losses.append(loss.item())

                # generate samples for metrics evaluation
                if self.metrics_ is not None and self.reverse_diffusion is not None:
                    xt = torch.randn_like(x).to(self.device)

                    # reverse diffusion sampling
                    for t in reversed(range(self.forward_diffusion.variance_scheduler.num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, y_encoded, None)
                        xt = self.reverse_diffusion(xt, predicted_noise, time_steps)

                    # clamp and normalize generated samples
                    x_hat = torch.clamp(xt, min=self.image_output_range[0], max=self.image_output_range[1])
                    if self.normalize_output:
                        x_hat = (x_hat - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])
                        x_orig = (x_orig - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])

                    # compute metrics
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

        # compute average metrics
        val_loss = torch.tensor(val_losses).mean().item()

        # all-reduce validation metrics across processes for DDP
        if self.use_ddp:
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()

        fid_avg = torch.tensor(fid_scores).mean().item() if fid_scores else float('inf')
        mse_avg = torch.tensor(mse_scores).mean().item() if mse_scores else None
        psnr_avg = torch.tensor(psnr_scores).mean().item() if psnr_scores else None
        ssim_avg = torch.tensor(ssim_scores).mean().item() if ssim_scores else None
        lpips_avg = torch.tensor(lpips_scores).mean().item() if lpips_scores else None

        # return to training mode
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()
        if self.forward_diffusion.variance_scheduler.trainable_beta:
            self.reverse_diffusion.train()
            self.forward_diffusion.train()

        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg


###==================================================================================================================###


class SampleDDPM(nn.Module):
    """mage generation using a trained Denoising Diffusion Probabilistic Model (DDPM).

    Implements the sampling process for DDPM, generating images by iteratively
    denoising random noise using a trained noise predictor and reverse diffusion
    process. Supports conditional generation with text prompts via a conditional
    model, as inspired by Ho et al. (2020).

    Parameters
    ----------
    reverse_diffusion : nn.Module
        Reverse diffusion module (e.g., ReverseDDPM) for the reverse process.
    noise_predictor : nn.Module
        Trained model to predict noise at each time step.
    image_shape : tuple
        Tuple of (height, width) specifying the generated image dimensions.
    conditional_model : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    tokenizer : str, optional
        Pretrained tokenizer name from Hugging Face (default: "bert-base-uncased").
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for generated images (default: 3).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    image_output_range : tuple, optional
        Tuple of (min, max) for clamping generated images (default: (-1, 1)).
    """
    def __init__(
            self,
            reverse_diffusion: torch.nn.Module,
            noise_predictor: torch.nn.Module,
            image_shape: Tuple[int, int],
            conditional_model: Optional[torch.nn.Module] = None,
            tokenizer: str = "bert-base-uncased",
            max_token_length: int = 77,
            batch_size: int = 1,
            in_channels: int = 3,
            device: Optional[str] = None,
            image_output_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> None:
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.reverse = reverse_diffusion.to(self.device)
        self.noise_predictor = noise_predictor.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_token_length = max_token_length
        self.in_channels = in_channels
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.image_output_range = image_output_range

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(
                isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(image_output_range, (tuple, list)) or len(image_output_range) != 2 or image_output_range[0] >= image_output_range[1]:
            raise ValueError("output_range must be a tuple (min, max) with min < max")

    def tokenize(self, prompts: Union[List, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes text prompts for conditional generation.

        Converts input prompts into tokenized input IDs and attention masks using the
        specified tokenizer, suitable for use with the conditional model.

        Parameters
        ----------
        prompts : str or list
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
            max_length=self.max_token_length,
            return_tensors="pt"
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def forward(
            self,
            conditions: Optional[Union[str, List]] = None,
            normalize_output: bool = True,
            save_images: bool = True,
            save_path: str = "ddpm_generated"
    ) -> torch.Tensor:
        """Generates images using the DDPM sampling process.

        Iteratively denoises random noise to generate images using the reverse diffusion
        process and noise predictor. Supports conditional generation with text prompts.
        Optionally saves generated images to a specified directory.

        Parameters
        ----------
        conditions : str or list, optional
            Text prompt(s) for conditional generation, default None.
        normalize_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        save_images : bool, optional
            If True, saves generated images to `save_path` (default: True).
        save_path : str, optional
            Directory to save generated images (default: "ddpm_generated").

        Returns
        -------
        generated_imgs (torch.Tensor) - Generated images, shape (batch_size, in_channels, height, width). If `normalize_output` is True, images are normalized to [0, 1]; otherwise, they are clamped to `output_range`.
        """
        if conditions is not None and self.conditional_model is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conditions is None and self.conditional_model is not None:
            raise ValueError("Conditions must be provided for conditional model")

        noisy_samples = torch.randn(self.batch_size, self.in_channels, self.image_shape[0], self.image_shape[1]).to(
            self.device)

        self.noise_predictor.eval()
        self.reverse.eval()
        if self.conditional_model:
            self.conditional_model.eval()

        with torch.no_grad():
            xt = noisy_samples
            for t in reversed(range(self.reverse.variance_scheduler.num_steps)):
                time_steps = torch.full((self.batch_size,), t, device=self.device)#, dtype=torch.long)
                if self.conditional_model is not None and conditions is not None:
                    input_ids, attention_masks = self.tokenize(conditions)
                    key_padding_mask = (attention_masks == 0)
                    y = self.conditional_model(input_ids, key_padding_mask)
                    predicted_noise = self.noise_predictor(xt, time_steps, y, None)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps, None, None)
                xt = self.reverse(xt, predicted_noise, time_steps)

            generated_imgs = torch.clamp(xt, min=self.image_output_range[0], max=self.image_output_range[1])
            if normalize_output:
                generated_imgs = (generated_imgs - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])

            # save images if save_images is True
            if save_images:
                os.makedirs(save_path, exist_ok=True)  # create directory if it doesn't exist
                for i in range(generated_imgs.size(0)):
                    img_path = os.path.join(save_path, f"image_{i+1}.png")
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
        sample_ddpm (SampleDDPM) - moved to the specified device.
        """
        self.device = device
        self.noise_predictor.to(device)
        self.reverse.to(device)
        if self.conditional_model:
            self.conditional_model.to(device)
        return super().to(device)