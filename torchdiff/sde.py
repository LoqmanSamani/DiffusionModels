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
- **VarianceSchedulerSDE**: Noise schedule and SDE-specific parameter management.
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
import os


###==================================================================================================================###

class ForwardDiffusion(nn.Module):
    """
    Unified forward diffusion process for continuous-time diffusion models.

    This module implements the marginal forward noising process
    p(x_t | x_0) for several commonly used stochastic differential equation
    (SDE) formulations, including:

        • Variance Preserving (VP-SDE)
        • Variance Exploding (VE-SDE)
        • Sub-Variance Preserving (Sub-VP-SDE)
        • Probability Flow ODE (ODE)

    Given clean data x₀, Gaussian noise ε ~ N(0, I), and continuous time
    t ∈ [0, 1], the forward process samples x_t and provides the *true score*
    ∇ₓ log p(x_t | x₀), which is commonly used for score matching objectives.

    Supported forward marginals:

    1. VP-SDE:
        p(x_t | x_0) = N(α(t) x_0, σ²(t) I)

    2. VE-SDE:
        p(x_t | x_0) = N(x_0, σ²(t) I),
        where σ(t) = σ_min (σ_max / σ_min)^t

    3. Sub-VP-SDE:
        p(x_t | x_0) = N(x_0, σ²(t) I),
        where σ²(t) = 1 - exp(-∫₀ᵗ β(s) ds)

    4. Probability Flow ODE:
        Shares the same marginals as VP-SDE but corresponds to a
        deterministic dynamics during sampling.

    The returned score is analytically computed as:

        ∇ₓ log p(x_t | x₀) = -(x_t - μ(t)) / σ²(t) = -ε / σ(t)

    where μ(t) is the mean of the forward transition.

    Parameters
    ----------
    variance_scheduler : VarianceSchedulerSDE
        Scheduler providing β(t), α(t), and σ(t) for VP and Sub-VP processes.

    sde_method : str, default="vp"
        Forward process type. Must be one of:
        {"vp", "ve", "sub-vp", "ode"}.

    sigma_min : float, default=0.01
        Minimum noise scale for the VE-SDE.

    sigma_max : float, default=50.0
        Maximum noise scale for the VE-SDE.

    eps : float, default=1e-8
        Small constant for numerical stability when computing the score.

    Notes
    -----
    • Time t is assumed to be normalized to [0, 1].
    • All operations are vectorized and support arbitrary data dimensions.
    • Broadcasting is handled automatically to match the shape of x₀.
    • For the ODE method, noise is still used to compute the analytical
      score during training, even though sampling is deterministic.

    References
    ----------
    - Song et al., "Score-Based Generative Modeling through SDEs", ICLR 2021
    - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
    - Kingma et al., "Variational Diffusion Models", NeurIPS 2021
    """
    def __init__(
            self,
            variance_scheduler: nn.Module,
            sde_method: str = "vp",
            sigma_min: float = 0.01,
            sigma_max: float = 50.0,
            eps: float = 1e-8
    ):
        super().__init__()

        valid_methods = ["vp", "ve", "sub-vp", "ode"]
        if sde_method not in valid_methods:
            raise ValueError(f"sde_method must be one of {valid_methods}, got {sde_method}")

        self.vs = variance_scheduler
        self.sde_method = sde_method
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = eps

    def _broadcast_to_shape(self, tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Broadcast tensor to target shape by adding trailing dimensions"""
        while tensor.dim() < len(target_shape):
            tensor = tensor.unsqueeze(-1)
        return tensor

    def get_forward_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean coefficient and std for the forward process based on SDE method

        Returns:
            mean_coeff: coefficient for clean data x_0
            std: standard deviation of noise
        """
        mean_coeff = None
        std = None
        if self.sde_method == "vp":
            # VP-SDE: p(x_t | x_0) = N(α(t)x_0, σ²(t)I)
            mean_coeff = self.vs.alpha(t)
            std = self.vs.std(t)

        elif self.sde_method == "ve":
            # VE-SDE: p(x_t | x_0) = N(x_0, σ²(t)I)
            # σ(t) grows from sigma_min to sigma_max
            mean_coeff = torch.ones_like(t)
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            std = sigma_t

        elif self.sde_method == "sub-vp":
            # Sub-VP-SDE: p(x_t | x_0) = N(x_0, σ²(t)I) where σ²(t) = 1 - e^(-∫β(s)ds)
            mean_coeff = torch.ones_like(t)
            std = self.vs.std(t)

        elif self.sde_method == "ode":
            # Probability flow ODE: same marginals as VP-SDE but deterministic
            mean_coeff = self.vs.alpha(t)
            std = self.vs.std(t)

        return mean_coeff, std

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from transition kernel and compute true score

        Args:
            x0: (batch, ..., dims) clean data
            noise: (batch, ..., dims) standard Gaussian noise
            t: (batch,) continuous time in [0, 1]

        Returns:
            xt: (batch, ..., dims) noised data
            score: (batch, ..., dims) true score ∇_x log p(x_t | x_0)
        """
        mean_coeff, std = self.get_forward_params(t)
        # broadcast to match x0 shape
        mean_coeff = self._broadcast_to_shape(mean_coeff, x0.shape)
        std = self._broadcast_to_shape(std, x0.shape)
        # x_t = mean_coeff * x_0 + std * ε
        xt = mean_coeff * x0 + std * noise
        # ∇_x log p(x_t | x_0) = -(x_t - mean_coeff*x_0) / σ²(t) = -ε / σ(t)
        score = -noise / (std + self.eps)
        return xt, score

###==================================================================================================================###

class ReverseSDE(nn.Module):
    """
    Unified reverse-time diffusion process for continuous-time diffusion models.

    This module implements a single-step numerical solver for the *reverse-time*
    stochastic differential equation (SDE) or probability flow ordinary
    differential equation (ODE) corresponding to a trained score-based model.

    Given a noisy sample x_t at time t and an estimate of the score
    ∇ₓ log p_t(x), the reverse process evolves the system backward in time
    (t → 0) using an Euler–Maruyama discretization.

    Supported reverse dynamics:

        • Variance Preserving (VP-SDE)
        • Variance Exploding (VE-SDE)
        • Sub-Variance Preserving (Sub-VP-SDE)
        • Probability Flow ODE (ODE)

    General reverse SDE form:
        dx = [f(x, t) - g²(t) ∇ₓ log p_t(x)] dt + g(t) dW̄_t

    where:
        • f(x, t) is the forward drift
        • g(t) is the diffusion coefficient
        • dW̄_t denotes reverse-time Brownian motion

    For the probability flow ODE, the diffusion term vanishes and the dynamics
    become deterministic while preserving the same marginals as the VP-SDE.

    Parameters
    ----------
    variance_scheduler : nn.Module
        Scheduler providing β(t) and related quantities for VP and Sub-VP
        dynamics. Typically an instance of `VarianceSchedulerSDE`.

    sde_method : str, default="vp"
        Type of reverse-time dynamics. Must be one of:
        {"vp", "ve", "sub-vp", "ode"}.

    sigma_min : float, default=0.01
        Minimum noise scale for the VE-SDE.

    sigma_max : float, default=50.0
        Maximum noise scale for the VE-SDE.

    Notes
    -----
    • Time t is assumed to be normalized to [0, 1].
    • Reverse integration proceeds with a *negative* time step dt < 0.
    • The score ∇ₓ log p_t(x) is typically predicted by a neural network.
    • For the final step or ODE-based sampling, stochastic noise can be disabled.
    • All tensor operations support broadcasting over arbitrary data shapes.

    Numerical Integration
    ---------------------
    The update rule implemented is the Euler–Maruyama scheme:

        x_{t+dt} = x_t
                   + [f(x_t, t) - g²(t)·score(x_t, t)] dt
                   + g(t) √|dt| ε

    where ε ~ N(0, I). For ODE sampling, the stochastic term is omitted.

    References
    ----------
    - Anderson, "Reverse-Time Diffusion Equation Models", 1982
    - Song et al., "Score-Based Generative Modeling through SDEs", ICLR 2021
    - Kingma et al., "Variational Diffusion Models", NeurIPS 2021
    """
    def __init__(
            self,
            variance_scheduler: nn.Module,
            sde_method: str = "vp",
            sigma_min: float = 0.01,
            sigma_max: float = 50.0
    ):
        super().__init__()

        valid_methods = ["vp", "ve", "sub-vp", "ode"]
        if sde_method not in valid_methods:
            raise ValueError(f"sde_method must be one of {valid_methods}, got {sde_method}")

        self.vs = variance_scheduler
        self.sde_method = sde_method
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _broadcast_to_shape(self, tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Broadcast tensor to target shape by adding trailing dimensions"""
        while tensor.dim() < len(target_shape):
            tensor = tensor.unsqueeze(-1)
        return tensor

    def get_reverse_coeffs(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get drift and diffusion coefficients for reverse SDE

        Returns:
            drift_coeff: coefficient for drift term
            g_squared: squared diffusion coefficient (for score term)
            diffusion_coeff: coefficient for diffusion term
        """
        if self.sde_method == "vp":
            # VP-SDE: dx = [-½β(t)x - β(t)∇log p_t(x)]dt + √β(t)dw̄
            drift_coeff = -0.5 * self.vs.beta(t)
            g_squared = self.vs.beta(t)
            diffusion_coeff = torch.sqrt(self.vs.beta(t))

        elif self.sde_method == "ve":
            # VE-SDE: dx = [-σ(t)dσ/dt ∇log p_t(x)]dt + √(2σ(t)dσ/dt)dw̄
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            dsigma_dt = sigma_t * torch.log(torch.tensor(self.sigma_max / self.sigma_min))
            drift_coeff = torch.zeros_like(t)
            g_squared = 2 * sigma_t * dsigma_dt
            diffusion_coeff = torch.sqrt(g_squared)

        elif self.sde_method == "sub-vp":
            # Sub-VP-SDE: dx = [-β(t)∇log p_t(x)]dt + √β(t)dw̄
            drift_coeff = torch.zeros_like(t)
            g_squared = self.vs.beta(t)
            diffusion_coeff = torch.sqrt(self.vs.beta(t))

        elif self.sde_method == "ode":
            # Probability flow ODE: deterministic (no diffusion)
            drift_coeff = -0.5 * self.vs.beta(t)
            g_squared = self.vs.beta(t)
            diffusion_coeff = torch.zeros_like(t)

        return drift_coeff, g_squared, diffusion_coeff

    def forward(self, xt: torch.Tensor, score: torch.Tensor, t: torch.Tensor, dt: float, last_step: bool = False) -> torch.Tensor:
        """Single reverse Euler-Maruyama step
        Args:
            xt: (batch, ..., dims) current state
            score: (batch, ..., dims) score estimate ∇_x log p_t(x)
            t: (batch,) current time
            dt: scalar time step (negative for reverse)
            last_step: if True, skip noise for deterministic final step

        Returns:
            x_prev: (batch, ..., dims) previous state
        """
        if not torch.is_tensor(dt):
            assert dt < 0.0, "dt must be negative for reverse diffusion!"
            dt = torch.tensor(dt, device=xt.device, dtype=xt.dtype)

        drift_coeff, g_squared, diffusion_coeff = self.get_reverse_coeffs(t)
        # Broadcast to match xt shape
        drift_coeff = self._broadcast_to_shape(drift_coeff, xt.shape)
        g_squared = self._broadcast_to_shape(g_squared, xt.shape)
        diffusion_coeff = self._broadcast_to_shape(diffusion_coeff, xt.shape)
        # Reverse drift: f(x,t) - g²(t)·score
        drift = drift_coeff * xt - g_squared * score
        # Diffusion term
        if last_step or self.sde_method == "ode":
            noise = torch.zeros_like(xt)
        else:
            noise = torch.randn_like(xt)
        diffusion = diffusion_coeff * noise
        # Euler-Maruyama step
        x_prev = xt + drift * dt + diffusion * torch.sqrt(torch.abs(dt))
        return x_prev

###==================================================================================================================###

class VarianceSchedulerSDE(nn.Module):
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

###==================================================================================================================###

class TrainSDE(nn.Module):
    """Trainer for score-based generative models using Stochastic Differential Equations.

    Manages the training process for SDE-based generative models, optimizing a noise
    predictor to learn the noise added by the forward SDE process, as described in Song
    et al. (2021). Supports conditional training with text prompts, mixed precision,
    learning rate scheduling, early stopping, and checkpointing.

    Parameters
    ----------

    noise_predictor : nn.Module
        Model to predict noise added during the forward SDE process.
    forward_diffusion : nn.Module
        Forward SDE diffusion module for adding noise.
    reverse_diffusion: nn.Module
        Reverse SDE diffusion module for denoising.
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data.
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
    bert_tokenizer : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    store_path : str, optional
        Path to save model checkpoints (default: "sde_model.pth").
    patience : int, optional
        Number of epochs to wait for improvement before early stopping (default: 10).
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
            conditional_model: Optional[torch.nn.Module] = None,
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
            use_compilation: bool = False,
            time_eps: float = 1e-8,
            pred_noise: bool = True,
            sampling_steps: int = 400,
            *args
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
        self.store_path = store_path or "sde_model"
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
        self.time_eps = time_eps
        self.pred_noise = pred_noise
        self.sampling_steps = sampling_steps

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
                self.forward_diffusion.variance_scheduler.load_state_dict(checkpoint['variance_scheduler_model'])
            if isinstance(self.reverse_diffusion.variance_scheduler, nn.Module):
                self.reverse_diffusion.variance_scheduler.load_state_dict(checkpoint['variance_scheduler_model'])
            else:
                self.forward_diffusion.variance_scheduler = checkpoint['variance_scheduler_model']
                self.reverse_diffusion.variance_scheduler = checkpoint['variance_scheduler_model']
        except Exception as e:
            warnings.warn(f"Variance_scheduler loading failed: {e}. Continuing with current variance_scheduler.")

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
                    t = self.time_sample(x.shape[0], self.time_eps)

                    # apply forward diffusion
                    xt, score = self.forward_diffusion(x, noise, t)

                    # predict noise
                    predicted = self.noise_predictor(xt, t, y_encoded, None)

                    if self.pred_noise: # if model predicts noise
                        loss = self.objective(predicted, noise) / self.grad_accumulation_steps
                    else: # if model predicts score
                        loss = self.objective(predicted, score) / self.grad_accumulation_steps

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
            train_losses.append(mean_train_loss)

            # all-reduce loss across processes for DDP
            if self.use_ddp:
                loss_tensor = torch.tensor(mean_train_loss, device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                mean_train_loss = loss_tensor.item()

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

    def sample_time(self, batch_size: int, eps: float = 1e-8) -> torch.Tensor:
        return eps + (1 - eps) * torch.rand(batch_size, device=self.device)

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

        # tokenize
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

            filename = f"sde_epoch_{epoch}{suffix}.pth"
            filepath = os.path.join(self.store_path, filename)
            os.makedirs(self.store_path, exist_ok=True)
            torch.save(checkpoint, filepath)

            print(f"Model saved at epoch {epoch}")

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
                t = self.time_sample(x.shape[0], self.time_eps)

                # apply forward diffusion
                xt, score = self.forward_diffusion(x, noise, t)

                # predict noise
                predicted = self.noise_predictor(xt, t, y_encoded, None)

                if self.pred_noise:  # if model predicts noise
                    loss = self.objective(predicted, noise) / self.grad_accumulation_steps
                else:  # if model predicts score
                    loss = self.objective(predicted, score) / self.grad_accumulation_steps
                val_losses.append(loss.item())

                # generate samples for metrics evaluation
                if self.metrics_ is not None and self.reverse_diffusion is not None:
                    xt = torch.randn_like(x).to(self.device)
                    # reverse diffusion sampling
                    t_schedule = torch.linspace(1.0, self.eps_time, self.sampling_steps + 1)
                    dt = torch.tensor((1.0 - self.time_eps)/self.sampling_steps, device=xt.device, dtype=xt.dtype)
                    for t in reversed(range(self.sampling_steps)):
                        t_current = float(t_schedule[t])
                        predicted = self.noise_predictor(xt, t_current, y_encoded, None)
                        if self.pred_noise:
                            std_ = self.forward_diffusion.vs.std(t)
                            while std_.dim() < len(xt.shape):
                                std = std_.unsqueeze(-1)
                            score = -predicted / (std + self.forward_diffusion.eps)
                        else:
                            score = predicted
                        if t == 0:
                            xt = self.reverse_diffusion(xt, score, t_current, dt, last_step=True)
                        else:
                            xt = self.reverse_diffusion(xt, score, t_current, dt)

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
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for generated images (default: 3).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    image_output_range : tuple, optional
        Range for clamping generated images (min, max), default (-1, 1).
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
            device: Optional[Union[str, torch.device]] = None,
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

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(image_output_range, (tuple, list)) or len(image_output_range) != 2 or image_output_range[0] >= image_output_range[1]:
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
             Tokenized input IDs, shape (batch_size, max_token_length).
        attention_mask : torch.Tensor
            Attention mask, shape (batch_size, max_token_length).
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
            for t in reversed(range(self.reverse.variance_scheduler.num_steps)):
                noise = torch.randn_like(xt) if self.reverse.sde_method != "ode" else None
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)

                if self.conditional_model is not None and conditions is not None:
                    input_ids, attention_masks = self.tokenize(conditions)
                    key_padding_mask = (attention_masks == 0)
                    y = self.conditional_model(input_ids, key_padding_mask)
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)

                xt = self.reverse(xt, noise, predicted_noise, time_steps)

            generated_imgs = torch.clamp(xt, min=self.image_output_range[0], max=self.image_output_range[1])
            if normalize_output:
                generated_imgs = (generated_imgs - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])

            # save images if save_images is True
            if save_images:
                os.makedirs(save_path, exist_ok=True)
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
        sample_sde (SampleSDE) - moved to the specified device.
        """
        self.device = device
        self.noise_predictor.to(device)
        self.reverse.to(device)
        if self.conditional_model:
            self.conditional_model.to(device)
        return super().to(device)