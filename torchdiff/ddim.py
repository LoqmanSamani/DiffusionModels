"""
**Denoising Diffusion Implicit Models (DDIM)**

This module provides a complete implementation of DDIM, as described in Song et al.
(2021, "Denoising Diffusion Implicit Models"). It includes components for forward and
reverse diffusion processes, hyperparameter management, training, and image sampling.
Supports both unconditional and conditional generation with text prompts, using a
subsampled time step schedule for faster sampling compared to DDPM.

**Components**

- **ForwardDDIM**: Forward diffusion process to add noise.
- **ReverseDDIM**: Reverse diffusion process to denoise with subsampled steps.
- **SchedulerDDIM**: Noise schedule management with subsampled (tau) schedule.
- **TrainDDIM**: Training loop with mixed precision and scheduling.
- **SampleDDIM**: Image generation from trained models with subsampled steps.

**Notes**

- The subsampled time step schedule (tau) enables faster sampling, controlled by the
  `tau_num_steps` parameter in VarianceSchedulerDDIM.

**References**:

- Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020).

-------------------------------------------------------------------------------
"""


###==================================================================================================================###


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
from typing import Optional, Tuple, Callable, List, Any, Union, Dict
from typing_extensions import Self
import os


###==================================================================================================================###


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


###==================================================================================================================###


class ReverseDDIM(nn.Module):
    """
    Implements the reverse (denoising) process of DDIM.

    Computes x_{t_prev} from x_t using the DDIM update rule:

        x_{t_prev} = sqrt(alphā_{t_prev}) * x̂_0
                     + sqrt(1 - alphā_{t_prev} - σ_t²) * ε̂_t
                     + σ_t * z

    where σ_t controls stochasticity via eta.
    Setting eta=0 results in deterministic DDIM sampling.

    Args:
        scheduler: Noise scheduler containing diffusion coefficients.
        pred_type: Model prediction type ["noise", "x0", "v"].
        eta: Controls stochasticity of sampling (0 = deterministic).
        clip_: Whether to clip predicted x0 to [-1, 1].
    """

    def __init__(self, scheduler: nn.Module, pred_type: str = "noise", eta: float = 0.0, clip_: bool = True):
        super().__init__()
        valid_pred_types = ["noise", "x0", "v"]
        if pred_type not in valid_pred_types:
            raise ValueError(f"prediction_type must be one of {valid_pred_types}")
        self.vs = scheduler
        self.pred_type = pred_type
        self.eta = eta
        self.clip_ = clip_

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Convert model output into a prediction of the clean sample x0.

        The conversion depends on the chosen prediction parameterization
        (noise, x0, or v-prediction).

        Args:
            xt: Noisy input at timestep t of shape (batch, ...).
            t: Current timesteps of shape (batch,).
            pred: Model output of shape (batch, ...).

        Returns:
            x0_pred: Predicted clean sample x0.
        """
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, xt.shape)
        if self.pred_type == "noise":
            x0_pred = (xt - sqrt_one_minus_alpha_cumprod_t * pred) / sqrt_alpha_cumprod_t
        elif self.pred_type == "x0":
            x0_pred = pred
        elif self.pred_type == "v":
            x0_pred = sqrt_alpha_cumprod_t * xt - sqrt_one_minus_alpha_cumprod_t * pred
        if self.clip_:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        return x0_pred

    def predict_noise(self, xt: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the predicted noise ε̂_t from x_t and predicted x0.

        Uses the identity:
            ε̂_t = (x_t - sqrt(alphā_t) * x̂_0) / sqrt(1 - alphā_t)

        Args:
            xt: Noisy input at timestep t.
            t: Current timesteps.
            x0_pred: Predicted clean sample x0.

        Returns:
            pred_noise: Predicted noise ε̂_t.
        """
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, xt.shape)
        pred_noise = (xt - sqrt_alpha_cumprod_t * x0_pred) / sqrt_one_minus_alpha_cumprod_t
        return pred_noise

    def forward(
            self,
            xt: torch.Tensor,
            t: torch.Tensor,
            t_prev: torch.Tensor,
            pred: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform one DDIM reverse diffusion step.

        Computes x_{t_prev} from x_t using the DDIM update equation.
        Allows non-adjacent timesteps, enabling accelerated sampling.

        Args:
            xt: Current noisy sample x_t of shape (batch, ...).
            t: Current timestep indices of shape (batch,).
            t_prev: Previous timestep indices of shape (batch,).
            pred: Model prediction at timestep t.

        Returns:
            x_prev: Sample at timestep t_prev.
            pred_x0: Predicted clean sample x0.
        """
        pred_x0 = self.predict_x0(xt, t, pred)
        # predict noise from x_0
        pred_noise = self.predict_noise(xt, t, pred_x0)
        alpha_cumprod_t = self.vs.alphas_cumprod[t]
        alpha_cumprod_t_prev = self.vs.alphas_cumprod[t_prev]
        alpha_cumprod_t = self.vs.get_index(alpha_cumprod_t, xt.shape)
        alpha_cumprod_t_prev = self.vs.get_index(alpha_cumprod_t_prev, xt.shape)

        # compute variance σ_t
        # eta=0: fully deterministic (σ_t=0)
        # eta=1: maximum stochasticity (similar to ddpm)
        sigma_t = self.eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )
        # dir_xt = √(1 - ᾱ_{t_prev} - σ_t²) * ε̂_t
        sqrt_one_minus_alpha_cumprod_t_prev_minus_sigma = torch.sqrt(
            1.0 - alpha_cumprod_t_prev - sigma_t ** 2
        )
        dir_xt = sqrt_one_minus_alpha_cumprod_t_prev_minus_sigma * pred_noise
        noise = torch.randn_like(xt)
        mask = (t_prev != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        # x_{t_prev} = √ᾱ_{t_prev} * x̂_0 + dir_xt + σ_t * z
        x_prev = (torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + sigma_t * mask * noise)
        return x_prev, pred_x0

###==================================================================================================================###

class SchedulerDDIM(nn.Module):
    """
    Noise scheduler for DDIM.

    Responsible for constructing the diffusion noise schedule and
    precomputing all coefficients required for both training and sampling.

    Supports multiple beta schedules and allows using fewer inference
    steps than training steps.

    Args:
        schedule_type: Type of beta schedule ("linear", "cosine", etc.).
        train_steps: Number of diffusion steps used during training.
        sample_steps: Number of steps used during inference.
        beta_min: Minimum beta value.
        beta_max: Maximum beta value.
        cosine_s: Offset parameter for cosine schedule.
        clip_min: Minimum clipping value for betas.
        clip_max: Maximum clipping value for betas.
        learn_var: Whether posterior variance is learnable.
    """
    def __init__(
            self,
            schedule_type: str = "linear",
            train_steps: int = 1000,  # total timesteps for training
            sample_steps: Optional[int] = None,  # can use fewer steps for sampling
            beta_min: float = 0.0001,
            beta_max: float = 0.02,
            cosine_s: float = 0.008,
            clip_min: float = 0.0001,
            clip_max: float = 0.9999,
            learn_var: bool = False
    ):
        super().__init__()
        valid_schedules = ["linear", "cosine", "quadratic", "sigmoid"]
        if schedule_type not in valid_schedules:
            raise ValueError(f"schedule_type must be one of {valid_schedules}, got {schedule_type}")

        self.schedule_type = schedule_type
        self.train_steps = train_steps
        self.sample_steps = sample_steps or train_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cosine_s = cosine_s
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.learn_var = learn_var
        self._setup_schedule()
        self._setup_inference_timesteps()

    def _setup_schedule(self):
        """
        Construct the diffusion noise schedule and precompute coefficients.

        Computes betas, alphas, cumulative products, square roots, and
        posterior variances required for forward and reverse diffusion.
        """
        if self.schedule_type == "linear":
            betas = torch.linspace(self.beta_min, self.beta_max, self.train_steps)
        elif self.schedule_type == "cosine":
            steps = self.train_steps + 1
            t = torch.linspace(0, self.train_steps, steps)
            alphas_cumprod = torch.cos(((t / self.train_steps) + self.cosine_s) / (1 + self.cosine_s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, self.clip_min, self.clip_max)
        elif self.schedule_type == "quadratic":
            betas = torch.linspace(self.beta_min ** 0.5, self.beta_max ** 0.5, self.train_steps) ** 2
        elif self.schedule_type == "sigmoid":
            betas = torch.linspace(-6, 6, self.train_steps)
            betas = torch.sigmoid(betas) * (self.beta_max - self.beta_min) + self.beta_min
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))
        if self.learn_var:
            self.register_parameter('log_variance', nn.Parameter(posterior_log_variance.clone()))
        else:
            self.register_buffer('log_variance', posterior_log_variance)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', posterior_log_variance)

    def _setup_inference_timesteps(self):
        """
        Create the set of timesteps used during inference.

        DDIM allows skipping timesteps for faster sampling by selecting
        a subset of training timesteps.
        """
        step_ratio = self.train_steps // self.sample_steps
        inference_timesteps = torch.arange(0, self.train_steps, step_ratio)
        self.register_buffer('inference_timesteps', inference_timesteps)

    def set_inference_timesteps(self, num_inference_timesteps: int):
        """
        Update the number of inference timesteps dynamically.

        Allows changing sampling speed and quality trade-offs at inference
        time without retraining the model.

        Args:
            num_inference_timesteps: Number of timesteps to use for sampling.
        """
        self.sample_steps = num_inference_timesteps
        self._setup_inference_timesteps()

    def get_index(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Reshape timestep-dependent coefficients for broadcasting.

        Extracts values indexed by t and reshapes them to match the
        dimensionality of a given tensor shape.

        Args:
            t: Timesteps tensor of shape (batch,).
            x_shape: Shape of the target tensor.

        Returns:
            Reshaped tensor suitable for broadcasting.
        """
        batch_size = t.shape[0]
        out = t.to(t.device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


###==================================================================================================================###


class TrainDDIM(nn.Module):
    """Trainer for Denoising Diffusion Implicit Models (DDIM).

    Manages the training process for DDIM, optimizing a noise predictor model to learn
    the noise added by the forward diffusion process. Supports conditional training with
    text prompts, mixed precision training, learning rate scheduling, early stopping, and
    checkpointing, as inspired by Song et al. (2021).

    Parameters
    ----------
    `diff_net` : nn.Module
        Main model to predict noise/v/x0
    fwd_ddim : nn.Module
        Forward DDIM diffusion module for adding noise.
    rwd_ddim: nn.Module
        Reverse DDIM diffusion module for denoising.
    `data_loader` : torch.utils.data.DataLoader
        DataLoader for training data.
    `optim` : torch.optim.Optimizer
        Optimizer for training the noise predictor and conditional model (if applicable).
    `loss_fn` : callable
        Loss function to compute the difference between predicted and actual noise.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `max_epochs` : int, optional
        Maximum number of training epochs (default: 100).
    `device` : str
        Device for computation (default: CUDA).
    `cond_net` : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    `metrics_` : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    `tokenizer` : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    `max_token_length` : int, optional
        Maximum length for tokenized prompts (default: 77).
    `store_path` : str, optional
        Path to save model checkpoints (default: "ddim_train").
    `patience` : int, optional
        Number of epochs to wait for improvement before early stopping (default: 20).
    `warmup_steps` : int, optional
        Number of epochs for learning rate warmup (default: 1000).
    `val_freq` : int, optional
        Frequency (in epochs) for validation (default: 10).
    `norm_range` : tuple, optional
        Range for clamping generated images (default: (-1, 1)).
    `norm_output` : bool, optional
        Whether to normalize generated images to [0, 1] for metrics (default: True).
    `use_ddp` : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    `grad_acc` : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    `log_freq` : int, optional
        Number of epochs before printing loss.
    use_comp : bool, optional
        whether the model is internally compiled using torch.compile (default: false)
    """
    def __init__(
            self,
            diff_net: torch.nn.Module,
            fwd_ddim: torch.nn.Module,
            rwd_ddim: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 100,
            device: str = 'cuda',
            cond_net: torch.nn.Module = None,
            metrics_: Optional[Any] = None,
            tokenizer: Optional[BertTokenizer] = None,
            max_token_length: int = 77,
            store_path: Optional[str] = None,
            patience: int = 20,
            warmup_steps: int = 1000,
            val_freq: int = 10,
            norm_range: Tuple[float, float] = (-1, 1),
            norm_output: bool = True,
            use_ddp: bool = False,
            grad_acc: int = 1,
            log_freq: int = 1,
            use_comp: bool = False,
            *args
    ) -> None:
        super().__init__()
        self.use_ddp = use_ddp
        self.grad_acc = grad_acc
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()
        self.diff_net = diff_net.to(self.device)
        self.fwd_ddim = fwd_ddim.to(self.device)
        self.rwd_ddim = rwd_ddim.to(self.device)
        self.cond_net = cond_net.to(self.device) if cond_net else None
        self.metrics_ = metrics_
        self.optim = optim
        self.loss_fn = loss_fn
        self.store_path = store_path or "ddim_train"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.max_token_length = max_token_length
        self.patience = patience
        self.val_freq = val_freq
        self.norm_range = norm_range
        self.norm_output = norm_output
        self.log_freq = log_freq
        self.use_comp = use_comp
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.best_loss = float('inf')
        self.losses = {'train_losses': [], 'val_losses': []}
        self.scheduler = ReduceLROnPlateau(
            self.optim,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optim, warmup_steps)
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
        if "RANK" not in os.environ:
            raise ValueError("DDP enabled but RANK environment variable not set")
        if "LOCAL_RANK" not in os.environ:
            raise ValueError("DDP enabled but LOCAL_RANK environment variable not set")
        if "WORLD_SIZE" not in os.environ:
            raise ValueError("DDP enabled but WORLD_SIZE environment variable not set")
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA but CUDA is not available")
        if not torch.distributed.is_initialized():
            init_process_group(backend="nccl")
        self.ddp_rank = int(os.environ["RANK"])  # global rank across all nodes
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])  # local rank on current node
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])  # total number of processes
        self.device = torch.device(f"cuda:{self.ddp_local_rank}")
        torch.cuda.set_device(self.device)
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
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        if 'model_state_dict_diff_net' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict_diff_net' key")
        state_dict = checkpoint['model_state_dict_diff_net']
        if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.diff_net.load_state_dict(state_dict)
        if self.cond_net is not None:
            if 'model_state_dict_cond' in checkpoint and checkpoint['model_state_dict_cond'] is not None:
                cond_state_dict = checkpoint['model_state_dict_cond']
                if self.use_ddp and not any(key.startswith('module.') for key in cond_state_dict.keys()):
                    cond_state_dict = {f'module.{k}': v for k, v in cond_state_dict.items()}
                elif not self.use_ddp and any(key.startswith('module.') for key in cond_state_dict.keys()):
                    cond_state_dict = {k.replace('module.', ''): v for k, v in cond_state_dict.items()}
                self.cond_net.load_state_dict(cond_state_dict)
            else:
                warnings.warn(
                    "Checkpoint contains no 'model_state_dict_cond' or it is None, "
                    "skipping conditional model loading"
                )
        if 'scheduler_model' not in checkpoint:
            raise KeyError("Checkpoint missing 'scheduler_model' key")
        try:
            if isinstance(self.fwd_ddim.vs, nn.Module):
                self.fwd_ddim.vs.load_state_dict(
                    checkpoint['scheduler_model'])
            if isinstance(self.rwd_ddim.vs, nn.Module):
                self.rwd_ddim.vs.load_state_dict(
                    checkpoint['scheduler_model'])
            else:
                self.fwd_ddim.vs = checkpoint['scheduler_model']
                self.rwd_ddim.vs = checkpoint['scheduler_model']
        except Exception as e:
            warnings.warn(f"Scheduler loading failed: {e}. Continuing with current scheduler.")

        if 'optim_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'optim_state_dict' key")
        try:
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
        except ValueError as e:
            warnings.warn(f"Optimizer state loading failed: {e}. Continuing without optimizer state.")
        epoch = checkpoint.get('epoch', -1)
        loss = checkpoint.get('loss', float('inf'))
        if self.master_process:
            print(f"Loaded checkpoint from {checkpoint_path} at epoch {epoch} with loss {loss:.4f}")
        return epoch, loss

    @staticmethod
    def warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Creates a learning rate scheduler for warmup.

        Generates a scheduler that linearly increases the learning rate from 0 to the
        optimizer's initial value over the specified warmup epochs, then maintains it.

        Parameters
        ----------
        `optimizer` : torch.optim.Optimizer
            Optimizer to apply the scheduler to.
        `warmup_steps` : int
            Number of steps for the warmup phase.

        Returns
        -------
        lr_scheduler (torch.optim.lr_scheduler.LambdaLR) - Learning rate scheduler for warmup.
        """
        def lr_lambda(step):
            if step < warmup_steps:
                return 0.1 + (0.9 * step / warmup_steps)
            return 1.0
        return LambdaLR(optimizer, lr_lambda)

    def _wrap_models_for_ddp(self) -> None:
        """Wrap models with DistributedDataParallel for multi-GPU training."""
        if self.use_ddp:
            self.diff_net = DDP(
                self.diff_net,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )
            if self.cond_net is not None:
                self.cond_net = DDP(
                    self.cond_net,
                    device_ids=[self.ddp_local_rank],
                    find_unused_parameters=True
                )

    def forward(self) -> Dict:
        """Trains the DDIM model to predict noise added by the forward diffusion process.

        Executes the training loop, optimizing the noise predictor and conditional model
        (if applicable) using mixed precision, gradient clipping, and learning rate
        scheduling. Supports validation, early stopping, and checkpointing.

        Returns
        -------
        losses: dictionlary contains train and validation losses
        """
        self.diff_net.train()
        if self.cond_net is not None:
            self.cond_net.train()

        if self.use_comp:
            try:
                self.diff_net = torch.compile(self.diff_net)
                if self.cond_net is not None:
                    self.cond_net = torch.compile(self.cond_net)
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

        self._wrap_models_for_ddp()
        scaler = torch.GradScaler()
        wait = 0
        for epoch in range(self.max_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}", disable=not self.master_process)
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            train_losses_epoch = []
            for step, (x, y) in enumerate(pbar):
                x = x.to(self.device)
                if self.cond_net is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    noise = torch.randn_like(x)
                    t = torch.randint(0, self.fwd_ddim.vs.train_steps, (x.shape[0],), device=x.device)
                    xt, target = self.fwd_ddim(x, t, noise)
                    pred = self.diff_net(xt, t, y_encoded, clip_embeddings=None)
                    loss = self.loss_fn(pred, target) / self.grad_acc

                scaler.scale(loss).backward()
                if (step + 1) % self.grad_acc == 0:
                    scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.diff_net.parameters(), max_norm=1.0)
                    if self.cond_net is not None:
                        torch.nn.utils.clip_grad_norm_(self.cond_net.parameters(), max_norm=1.0)
                    scaler.step(self.optim)
                    scaler.update()
                    self.optim.zero_grad()
                    if self.global_step > 0 and self.global_step < self.warmup_steps:
                        self.warmup_lr_scheduler.step()
                    self.global_step += 1
                pbar.set_postfix({'Loss': f'{loss.item() * self.grad_acc:.4f}'})
                train_losses_epoch.append(loss.item() * self.grad_acc)
            mean_train_loss = torch.tensor(train_losses_epoch).mean().item()
            self.losses['train_losses'].append(mean_train_loss)
            if self.use_ddp:
                loss_tensor = torch.tensor(mean_train_loss, device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                mean_train_loss = loss_tensor.item()
            if self.master_process and (epoch + 1) % self.log_freq == 0:
                current_lr = self.optim.param_groups[0]['lr']
                print(f"\nEpoch: {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}")
            if self.val_loader is not None and (epoch + 1) % self.val_freq == 0:
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

                self.scheduler.step(val_loss)
                self.losses['val_losses'].append((val_loss, fid, mse, psnr, ssim, lpips_score))
            else:
                if self.master_process:
                    print()
                self.scheduler.step(mean_train_loss)
            if self.master_process:
                if mean_train_loss < self.best_loss:
                    self.best_loss = mean_train_loss
                    wait = 0
                    self._save_checkpoint(epoch + 1, self.best_loss, "best_")
                else:
                    wait += 1
                    if wait >= self.patience:
                        print("Early stopping triggered")
                        self._save_checkpoint(epoch + 1, mean_train_loss, "early_stop_")
                        break
                if (epoch + 1) % self.val_freq == 0:
                    self._save_checkpoint(epoch + 1, mean_train_loss, "")
        if self.use_ddp:
            destroy_process_group()
        return self.losses

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
        # convert to string list
        y_list = y.cpu().numpy().tolist() if isinstance(y, torch.Tensor) else y
        y_list = [str(item) for item in y_list]
        y_encoded = self.tokenizer(
            y_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt"
        ).to(self.device)
        input_ids = y_encoded["input_ids"]
        attention_mask = y_encoded["attention_mask"]
        y_encoded = self.cond_net(input_ids, attention_mask)
        return y_encoded

    def _save_checkpoint(self, epoch: int, loss: float, pref: str = "") -> None:
        """Save model checkpoint (only called by master process).

        Parameters
        ----------
        epoch : int
            Current epoch number.
        loss : float
            Current loss value.
        pref : str, optional
            prefix to add to checkpoint filename.
        """
        try:
            diff_net_state = (
                self.diff_net.module.state_dict() if self.use_ddp
                else self.diff_net.state_dict()
            )
            cond_state = None
            if self.cond_net is not None:
                cond_state = (
                    self.cond_net.module.state_dict() if self.use_ddp
                    else self.cond_net.state_dict()
                )
            checkpoint = {
                'epoch': epoch,
                'model_state_dict_diff_net': diff_net_state,
                'model_state_dict_cond': cond_state,
                'optim_state_dict': self.optim.state_dict(),
                'loss': loss,
                'losses': self.losses,
                'scheduler_model': self.fwd_ddim.vs.state_dict(),
                'max_epochs': self.max_epochs,
            }
            filename = f"{pref}model_epoch_{epoch}.pth"
            filepath = os.path.join(self.store_path, filename)
            os.makedirs(self.store_path, exist_ok=True)
            torch.save(checkpoint, filepath)
            print(f"Model saved at epoch {epoch} with loss: {loss:.4f}")
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

        self.diff_net.eval()
        if self.cond_net is not None:
            self.cond_net.eval()

        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                x_orig = x.clone()
                if self.cond_net is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None
                noise = torch.randn_like(x)
                t = torch.randint(0, self.fwd_ddim.vs.train_steps, (x.shape[0],), device=x.device)
                xt, target = self.fwd_ddim(x, t, noise)
                pred = self.diff_net(xt, t, y_encoded, clip_embeddings=None)
                loss = self.loss_fn(pred, target)
                val_losses.append(loss.item())

                if self.metrics_ is not None and self.rwd_ddim is not None:
                    xt = torch.randn_like(x)
                    timesteps = self.fwd_ddim.vs.inference_timesteps.flip(0)
                    for i in range(len(timesteps) - 1):
                        t_ = timesteps[i].item()
                        t_pre = timesteps[i + 1].item()
                        time = torch.full((xt.shape[0],), t_, device=self.device, dtype=torch.long)
                        prev_time = torch.full((xt.shape[0],), t_pre, device=self.device, dtype=torch.long)
                        pred = self.diff_net(xt, time, y_encoded, clip_embeddings=None)
                        xt, _ = self.rwd_ddim(xt, time, prev_time, pred)
                    x_hat = torch.clamp(xt, min=self.norm_range[0], max=self.norm_range[1])
                    if self.norm_output:
                        x_hat = (x_hat - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])
                        x_orig = (x_orig - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])

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

        val_loss = torch.tensor(val_losses).mean().item()
        if self.use_ddp:
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()

        fid_avg = torch.tensor(fid_scores).mean().item() if fid_scores else float('inf')
        mse_avg = torch.tensor(mse_scores).mean().item() if mse_scores else None
        psnr_avg = torch.tensor(psnr_scores).mean().item() if psnr_scores else None
        ssim_avg = torch.tensor(ssim_scores).mean().item() if ssim_scores else None
        lpips_avg = torch.tensor(lpips_scores).mean().item() if lpips_scores else None

        self.diff_net.train()
        if self.cond_net is not None:
            self.cond_net.train()
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
    `rwd_ddim` : nn.Module
        Reverse diffusion module (e.g., ReverseDDIM) for the reverse process.
    `diff_net` : nn.Module
        Trained model to predict noise/v/x0 at each time step.
    `img_size` : tuple
        Tuple of (height, width) specifying the generated image dimensions.
    `cond_net` : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    `tokenizer` : str, optional
        Pretrained tokenizer name from Hugging Face (default: "bert-base-uncased").
    `max_length` : int, optional
        Maximum length for tokenized prompts (default: 77).
    `batch_size` : int, optional
        Number of images to generate per batch (default: 1).
    `in_channels` : int, optional
        Number of input channels for generated images (default: 3).
    `device` : str
        Device for computation (default: CUDA).
    `norm_range` : tuple, optional
        Tuple of (min, max) for clamping generated images (default: (-1, 1)).
    """
    def __init__(
            self,
            rwd_ddim: torch.nn.Module,
            diff_net: torch.nn.Module,
            img_size: Tuple[int, int],
            cond_net: Optional[torch.nn.Module] = None,
            tokenizer: str = "bert-base-uncased",
            max_token_length: int = 77,
            batch_size: int = 1,
            in_channels: int = 3,
            device: Optional[str] = None,
            norm_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.rwd_ddim = rwd_ddim.to(self.device)
        self.diff_net = diff_net.to(self.device)
        self.cond_net = cond_net.to(self.device) if cond_net else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_token_length = max_token_length
        self.in_channels = in_channels
        self.img_size = img_size
        self.batch_size = batch_size
        self.norm_range = norm_range
        if not isinstance(img_size, (tuple, list)) or len(img_size) != 2 or not all(
                isinstance(s, int) and s > 0 for s in img_size):
            raise ValueError("img_size must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(norm_range, (tuple, list)) or len(norm_range) != 2 or norm_range[0] >= norm_range[1]:
            raise ValueError("norm_range must be a tuple (min, max) with min < max")

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
            max_length=self.max_token_length,
            return_tensors="pt"
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def forward(self, conds: Optional[Union[str, List]] = None, norm_output: bool = True, save_imgs: bool = True,
                save_path: str = "ddim_samples") -> torch.Tensor:
        """Generates images using the DDIM sampling process.

        Iteratively denoises random noise to generate images using the reverse diffusion
        process with a subsampled time step schedule and noise predictor. Supports
        conditional generation with text prompts.

        Parameters
        ----------
        `conds` : str or list, optional
            Text prompt(s) for conditional generation, default None.
        `norm_output` : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        `save_imgs` : bool, optional
            If True, saves generated images to `save_path` (default: True).
        `save_path` : str, optional
            Directory to save generated images (default: "ddim_samples").

        Returns
        -------
        samps (torch.Tensor) - Generated images, shape (batch_size, in_channels, height, width).
        """
        if conds is not None and self.cond_net is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conds is None and self.cond_net is not None:
            raise ValueError("Conditions must be provided for conditional model")

        init_samps = torch.randn(self.batch_size, self.in_channels, self.img_size[0], self.img_size[1]).to(self.device)
        self.diff_net.eval()
        if self.cond_net:
            self.cond_net.eval()
        timesteps = self.rwd_ddim.vs.inference_timesteps
        timesteps = timesteps.flip(0)
        iterator = tqdm(
            range(len(timesteps) - 1),
            total=len(timesteps) - 1,
            desc="Sampling",
            dynamic_ncols=True,
            leave=True,
        )
        if self.cond_net is not None and conds is not None:
            input_ids, attention_masks = self.tokenize(conds)
            key_padding_mask = (attention_masks == 0)
            y = self.cond_net(input_ids, key_padding_mask)
        else:
            y = None

        with torch.no_grad():
            xt = init_samps
            for i in iterator:
                t_current = timesteps[i].item()
                t_prev = timesteps[i + 1].item()
                #assert t_current > t_prev or t_prev == 0
                time = torch.full((self.batch_size,), t_current, device=self.device, dtype=torch.long)
                prev_time = torch.full((self.batch_size,), t_prev, device=self.device, dtype=torch.long)
                pred = self.diff_net(xt, time, y, clip_embeddings=None)
                xt, _ = self.rwd_ddim(xt, time, prev_time, pred)
            samps = torch.clamp(xt, min=self.norm_range[0], max=self.norm_range[1])
            if norm_output:
                samps = (samps - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])
            if save_imgs:
                os.makedirs(save_path, exist_ok=True)
                for i in range(samps.size(0)):
                    img_path = os.path.join(save_path, f"img_{i+1}.png")
                    save_image(samps[i], img_path)
        return samps

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
        self.diff_net.to(device)
        if self.cond_net:
            self.cond_net.to(device)
        return super().to(device)