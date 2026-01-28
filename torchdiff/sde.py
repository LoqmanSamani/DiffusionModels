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
- **SchedulerSDE**: Noise schedule and SDE-specific parameter management.
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
from typing import Optional, Tuple, Callable, List, Any, Union, Dict
from typing_extensions import Self
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
import warnings
from torchvision.utils import save_image
import os


###==================================================================================================================###

class ForwardSDE(nn.Module):
    """
    Forward diffusion process for continuous-time diffusion models.

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
    scheduler : SchedulerSDE
        Scheduler providing β(t), α(t), and σ(t) for VP and Sub-VP processes.

    method : str, default="vp"
        Forward process type. Must be one of:
        {"vp", "ve", "sub-vp", "ode"}.

    pred_type: Prediction parameterization.
                One of {"noise", "x0", "score"}.

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
            scheduler: nn.Module,
            method: str = "vp",
            pred_type = 'noise',
            sigma_min: float = 0.01,
            sigma_max: float = 50.0,
            eps: float = 1e-8
    ):
        super().__init__()

        valid_methods = ["vp", "ve", "sub-vp", "ode"]
        if method not in valid_methods:
            raise ValueError(f"sde_method must be one of {valid_methods}, got {method}")

        valid_types = ["noise", "score"]
        if pred_type not in valid_types:
            raise ValueError(f"pred_type must be one of {valid_types}, got {pred_type}")

        self.vs = scheduler
        self.method = method
        self.pred_type = pred_type
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
        if self.method == "vp":
            # vp-sde: p(x_t | x_0) = N(α(t)x_0, σ²(t)I)
            mean_coeff = self.vs.alpha(t)
            std = self.vs.std(t)

        elif self.method == "ve":
            # ve-sde: p(x_t | x_0) = N(x_0, σ²(t)I)
            # σ(t) grows from sigma_min to sigma_max
            mean_coeff = torch.ones_like(t)
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            std = sigma_t

        elif self.method == "sub-vp":
            # sub-vp-sde: p(x_t | x_0) = N(x_0, σ²(t)I) where σ²(t) = 1 - e^(-∫β(s)ds)
            mean_coeff = torch.ones_like(t)
            std = self.vs.std(t)

        elif self.method == "ode":
            # probability flow ode: same marginals as vp-sde but deterministic
            mean_coeff = self.vs.alpha(t)
            std = self.vs.std(t)

        return mean_coeff, std

    def forward(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from transition kernel and compute true score

        Arguments:
            x0: (batch, ..., dims) clean data
            t: (batch, ) continuous time in [0, 1]
            noise: (batch, ..., dims) standard Gaussian noise

        Returns:
            xt: (batch, ..., dims) noised data
            target: (batch, ..., dims) true score/added noise
        """
        mean_coeff, std = self.get_forward_params(t)
        # broadcast to match x0 shape
        mean_coeff = self._broadcast_to_shape(mean_coeff, x0.shape)
        std = self._broadcast_to_shape(std, x0.shape)
        # x_t = mean_coeff * x_0 + std * ε
        xt = mean_coeff * x0 + std * noise
        if self.pred_type == 'noise':
            target = noise
        elif self.pred_type == "score":
            # ∇_x log p(x_t | x_0) = -(x_t - mean_coeff*x_0) / σ²(t) = -ε / σ(t)
            target = -noise / (std + self.eps)
        return xt, target

###==================================================================================================================###

class ReverseSDE(nn.Module):
    """
    Reverse-time diffusion process for continuous-time sde diffusion models

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
    scheduler : nn.Module
        Scheduler providing β(t) and related quantities for VP and Sub-VP
        dynamics. Typically an instance of `SchedulerSDE`.

    method : str, default="vp"
        Type of reverse-time dynamics. Must be one of:
        {"vp", "ve", "sub-vp", "ode"}.

    pred_type: Prediction parameterization.
                One of {"noise", "score"}.

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
            scheduler: nn.Module,
            method: str = "vp",
            pred_type: str = 'noise',
            sigma_min: float = 0.01,
            sigma_max: float = 50.0,
            eps: float = 1e-8
    ):
        super().__init__()
        valid_methods = ["vp", "ve", "sub-vp", "ode"]
        if method not in valid_methods:
            raise ValueError(f"sde_method must be one of {valid_methods}, got {method}")
        valid_types = ["noise", "score"]
        if pred_type not in valid_types:
            raise ValueError(f"pred_type must be one of {valid_types}, got {pred_type}")

        self.vs = scheduler
        self.method = method
        self.pred_type = pred_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = eps

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
        if self.method == "vp":
            # vp-sde: dx = [-½β(t)x - β(t)∇log p_t(x)]dt + √β(t)dw̄
            drift_coeff = -0.5 * self.vs.beta(t)
            g_squared = self.vs.beta(t)
            diffusion_coeff = torch.sqrt(self.vs.beta(t))

        elif self.method == "ve":
            # ve-sde: dx = [-σ(t)dσ/dt ∇log p_t(x)]dt + √(2σ(t)dσ/dt)dw̄
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            dsigma_dt = sigma_t * torch.log(torch.tensor(self.sigma_max / self.sigma_min))
            drift_coeff = torch.zeros_like(t)
            g_squared = 2 * sigma_t * dsigma_dt
            diffusion_coeff = torch.sqrt(g_squared)

        elif self.method == "sub-vp":
            # sub-vp-sde: dx = [-β(t)∇log p_t(x)]dt + √β(t)dw̄
            drift_coeff = torch.zeros_like(t)
            g_squared = self.vs.beta(t)
            diffusion_coeff = torch.sqrt(self.vs.beta(t))

        elif self.method == "ode":
            # probability flow ode: deterministic
            drift_coeff = -0.5 * self.vs.beta(t)
            g_squared = self.vs.beta(t)
            diffusion_coeff = torch.zeros_like(t) # no diffusion in ode

        return drift_coeff, g_squared, diffusion_coeff

    def forward(self, xt: torch.Tensor, pred: torch.Tensor, t: torch.Tensor, dt: float, last_step: bool = False) -> torch.Tensor:
        """Single reverse Euler-Maruyama step
        Args:
            xt: (batch, ..., dims) current state
            pred: (batch, ..., dims) output (prediction of diffusion model)
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
        # broadcast to match xt shape
        drift_coeff = self._broadcast_to_shape(drift_coeff, xt.shape)
        g_squared = self._broadcast_to_shape(g_squared, xt.shape)
        diffusion_coeff = self._broadcast_to_shape(diffusion_coeff, xt.shape)
        if self.method == "ve":
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            std = sigma_t
        else:
            std = self.vs.std(t)
        while std.dim() < len(xt.shape):
            std = std.unsqueeze(-1)
        if self.pred_type == "noise":
            score = -pred / (std + self.eps)
        elif self.pred_type == "score":
            score = pred
        # [-½β(t)x - β(t)∇log p_t(x)]dt + √β(t)dw̄
        # reverse drift: f(x,t) - g²(t)·score
        drift = drift_coeff * xt - g_squared * score
        if last_step or self.method == "ode":
            noise = torch.zeros_like(xt)
        else:
            noise = torch.randn_like(xt)
        diffusion = diffusion_coeff * noise
        # Euler-Maruyama step
        x_prev = xt + drift * dt + diffusion * torch.sqrt(torch.abs(dt))
        return x_prev

###==================================================================================================================###

class SchedulerSDE(nn.Module):
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

    score_net : nn.Module
        Model to predict score/noise.
    fwd_sde : nn.Module
        Forward SDE diffusion module for adding noise.
    rwd_sde: nn.Module
        Reverse SDE diffusion module for denoising.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    optim : torch.optim.Optimizer
        Optimizer for training the noise predictor and conditional model (if applicable).
    loss_fn : callable
        Loss function to compute the difference between predicted and actual noise.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    max_epochs : int, optional
        Maximum number of training epochs (default: 1000).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    cond_net : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    metrics_ : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    tokenizer : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    store_path : str, optional
        Path to save model checkpoints (default: "sde_train").
    patience : int, optional
        Number of epochs to wait for improvement before early stopping (default: 20).
    warmup_steps : int, optional
        Number of steps for learning rate warmup (default: 1000).
    val_freq : int, optional
        Frequency (in epochs) for validation (default: 10).
    norm_range : tuple, optional
        Range for clamping generated images (default: (-1, 1)).
    norm_output : bool, optional
        Whether to normalize generated images to [0, 1] for metrics (default: True).
    use_ddp : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    grad_acc : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    log_freq : int, optional
        Number of epochs before printing loss.
    use_comp : bool, optional
        whether the model is internally compiled using torch.compile (default: false)
    time_eps: float, optional
        lower bound for diffusion time sampling (time_eps, 1.0) (default: 1e-5)
    num_steps: int, optional
        number of time staps for sampling during validation (default: 400)
    """
    def __init__(
            self,
            score_net: torch.nn.Module,
            fwd_sde: torch.nn.Module,
            rwd_sde: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 1000,
            device: str = 'cuda',
            cond_net: Optional[torch.nn.Module] = None,
            metrics_: Optional[Any] = None,
            tokenizer: Optional[BertTokenizer] = None,
            max_token_length: int = 77,
            store_path: Optional[str] = None,
            patience: int = 20,
            warmup_steps: int = 1000,
            val_freq: int = 10,
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            norm_output: bool = True,
            use_ddp: bool = False,
            grad_acc: int = 1,
            log_freq: int = 1,
            use_comp: bool = False,
            time_eps: float = 1e-5,
            num_steps: int = 400,
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

        self.score_net = score_net.to(self.device)
        self.fwd_sde = fwd_sde.to(self.device)
        self.rwd_sde = rwd_sde.to(self.device)
        self.cond_net = cond_net.to(self.device) if cond_net else None

        self.metrics_ = metrics_
        self.optim = optim
        self.loss_fn = loss_fn
        self.store_path = store_path or "sde_train"
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
        self.time_eps = time_eps
        self.num_steps = num_steps
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
        else:
            self.tokenizer = tokenizer

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

        # get rank info
        self.ddp_rank = int(os.environ["RANK"])
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])

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
        if 'model_state_dict_score_net' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict_score_net' key")

        state_dict = checkpoint['model_state_dict_score_net']
        if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.score_net.load_state_dict(state_dict)
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
            if isinstance(self.fwd_sde.vs, nn.Module):
                self.fwd_sde.vs.load_state_dict(checkpoint['scheduler_model'])
            if isinstance(self.rwd_sde.vs, nn.Module):
                self.rwd_sde.vs.load_state_dict(checkpoint['scheduler_model'])
            else:
                self.fwd_sde.vs = checkpoint['scheduler_model']
                self.rwd_sde.vs = checkpoint['scheduler_model']
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
        optimizer : torch.optim.Optimizer
            Optimizer to apply the scheduler to.
        warmup_steps : int
            Number of steps for the warmup phase.

        Returns
        -------
        torch.optim.lr_scheduler.LambdaLR
            Learning rate scheduler for warmup.
        """
        def lr_lambda(step):
            if step < warmup_steps:
                return 0.1 + (0.9 * step / warmup_steps)
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    def _wrap_models_for_ddp(self) -> None:
        """Wrap models with DistributedDataParallel for multi-GPU training."""
        if self.use_ddp:
            self.score_net = DDP(
                self.score_net,
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
        """Trains the SDE model to predict noise added by the forward diffusion process.

        Executes the training loop, optimizing the noise predictor and conditional model
        (if applicable) using mixed precision, gradient clipping, and learning rate
        scheduling. Supports validation, early stopping, and checkpointing.

        Returns
        -------
        losses : dictionary of train and validation losses.

        **Notes**

        - Training uses mixed precision via `torch.cuda.amp` or `torch.amp` for efficiency.
        - Checkpoints are saved when the validation (or training) loss improves, and on early stopping.
        - Early stopping is triggered if no improvement occurs for `patience` epochs.
        """
        self.score_net.train()
        if self.cond_net is not None:
            self.cond_net.train()

        if self.use_comp:
            try:
                self.score_net = torch.compile(self.score_net)
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
                    t = self.sample_time(x.shape[0], self.time_eps)
                    xt, target = self.fwd_sde(x, t, noise)
                    pred = self.score_net(xt, t, y_encoded, clip_embeddings=None)
                    var = self.fwd_sde.vs.variance(t)
                    if self.fwd_sde.method == "ve":
                        sigma = self.fwd_sde.sigma_min * (self.fwd_sde.sigma_max / self.fwd_sde.sigma_min) ** t
                        loss = self.loss_fn(pred, target, sigma) / self.grad_acc
                    else:
                        loss = self.loss_fn(pred, target, var) / self.grad_acc
                scaler.scale(loss).backward()
                if (step + 1) % self.grad_acc == 0:
                    scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
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

    def sample_time(self, batch_size: int, eps: float = 1e-5) -> torch.Tensor:
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
            pref to add to checkpoint filename.
        """
        try:
            score_net_state = (
                self.score_net.module.state_dict() if self.use_ddp
                else self.score_net.state_dict()
            )
            cond_state = None
            if self.cond_net is not None:
                cond_state = (
                    self.cond_net.module.state_dict() if self.use_ddp
                    else self.cond_net.state_dict()
                )
            checkpoint = {
                'epoch': epoch,
                'model_state_dict_score_net': score_net_state,
                'model_state_dict_cond': cond_state,
                'optim_state_dict': self.optim.state_dict(),
                'loss': loss,
                'losses': self.losses,
                'scheduler_model': self.fwd_sde.vs.state_dict(),
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
        self.score_net.eval()
        if self.cond_net is not None:
            self.cond_net.eval()

        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []
        with torch.no_grad():
            with torch.no_grad():
                for x, y in self.val_loader:
                    x = x.to(self.device)
                    x_orig = x.clone()
                    if self.cond_net is not None:
                        y_encoded = self._process_conditional_input(y)
                    else:
                        y_encoded = None

                    noise = torch.randn_like(x)
                    t = self.sample_time(x.shape[0], self.time_eps)
                    xt, target = self.fwd_sde(x, t, noise)
                    pred = self.score_net(xt, t, y_encoded, clip_embeddings=None)
                    var = self.fwd_sde.vs.variance(t)
                    if self.fwd_sde.method == "ve":
                        sigma = self.fwd_sde.sigma_min * (self.fwd_sde.sigma_max / self.fwd_sde.sigma_min) ** t
                        loss = self.loss_fn(pred, target, sigma) / self.grad_acc
                    else:
                        loss = self.loss_fn(pred, target, var) / self.grad_acc
                    val_losses.append(loss.item())
                    if self.metrics_ is not None and self.rwd_sde is not None:
                        xt = torch.randn_like(x).to(self.device)
                        # reverse diffusion sampling
                        t_schedule = torch.linspace(1.0, self.time_eps, self.num_steps + 1)
                        dt = torch.tensor(-(1.0 - self.time_eps) / self.num_steps, device=xt.device, dtype=xt.dtype)
                        for t in range(self.num_steps):
                            t_current = float(t_schedule[t])
                            t_batch = torch.full((xt.shape[0],), t_current, dtype=xt.dtype, device=self.device)
                            pred = self.score_net(xt, t_batch, y_encoded, None)
                            last_step = (t == self.num_steps - 1)
                            xt = self.rwd_sde(xt, pred, t_batch, dt, last_step = last_step)

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

        self.score_net.train()
        if self.cond_net is not None:
            self.cond_net.train()
        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg


###==================================================================================================================###

class SampleSDE(nn.Module):
    """Sampler for generating images using SDE-based generative models.

    Generates images by iteratively denoising random noise using the reverse SDE process
    and a trained noise predictor, as described in Song et al. (2021). Supports both
    unconditional and conditional generation with text prompts.

    Parameters
    ----------
    rwd_sde : ReverseSDE
        Reverse SDE diffusion module for denoising.
    score_net : nn.Module
        Model to predict noise added during the forward SDE process.
    img_size : tuple
        Shape of generated images as (height, width).
    cond_net : nn.Module, optional
        Model for conditional generation (e.g., TextEncoder), default None.
    tokenizer : str or BertTokenizer, optional
        Tokenizer for processing text prompts, default "bert-base-uncased".
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for generated images (default: 3).
    device : srt, optional
        Device for computation (default: CUDA).
    norm_range : tuple, optional
        Range for clamping generated images (min, max), default (-1, 1).
    """
    def __init__(
            self,
            rwd_sde: torch.nn.Module,
            score_net: torch.nn.Module,
            img_size: Tuple[int, int],
            cond_net: Optional[torch.nn.Module] = None,
            tokenizer: str = "bert-base-uncased",
            max_token_length: int = 77,
            batch_size: int = 1,
            in_channels: int = 3,
            device: str = 'cuda',
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            time_eps: float =  1e-5
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.rwd_sde = rwd_sde.to(self.device)
        self.score_net = score_net.to(self.device)
        self.cond_net = cond_net.to(self.device) if cond_net else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_token_length = max_token_length
        self.in_channels = in_channels
        self.img_size = img_size
        self.batch_size = batch_size
        self.norm_range = norm_range
        self.time_eps = time_eps

        if not isinstance(img_size, (tuple, list)) or len(img_size) != 2 or not all(isinstance(s, int) and s > 0 for s in img_size):
            raise ValueError("img_size must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(norm_range, (tuple, list)) or len(norm_range) != 2 or norm_range[0] >= norm_range[1]:
            raise ValueError("norm_range must be a tuple (min, max) with min < max")

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
            num_steps: int,
            conds: Optional[Union[str, List]] = None,
            norm_output: bool = True,
            save_imgs: bool = True,
            save_path: str = "sde_samples"
    ) -> torch.Tensor:
        """Generates images using the reverse SDE sampling process.

        Iteratively denoises random noise to generate images using the reverse SDE process
        and noise predictor. Supports conditional generation with text prompts.

        Parameters
        ----------
        conds : str or list, optional
            Text prompt(s) for conditional generation, default None.
        norm_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        save_imgs : bool, optional
            If True, saves generated images to `save_path` (default: True).
        save_path : str, optional
            Directory to save generated images (default: "sde_samples").

        Returns
        -------
        samps (torch.Tensor) - Generated images, shape (batch_size, in_channels, height, width).
        If `norm_output` is True, images are normalized to [0, 1]; otherwise, they are clamped to `norm_range`.
        """
        if conds is not None and self.cond_net is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conds is None and self.cond_net is not None:
            raise ValueError("Conditions must be provided for conditional model")

        init_samps = torch.randn(self.batch_size, self.in_channels, self.img_size[0], self.img_size[1], device=self.device)
        self.score_net.eval()
        self.rwd_sde.eval()
        if self.cond_net:
            self.cond_net.eval()

        if self.cond_net is not None and conds is not None:
            input_ids, attention_masks = self.tokenize(conds)
            key_padding_mask = (attention_masks == 0)
            y = self.cond_net(input_ids, key_padding_mask)
        else:
            y = None
        t_schedule = torch.linspace(1.0, self.time_eps, num_steps + 1, device=self.device)
        dt = -(1.0 - self.time_eps) / num_steps
        iterator = tqdm(
            range(num_steps),
            total=num_steps,
            desc="Sampling",
            dynamic_ncols=True,
            leave=True,
        )
        #iterator = tqdm(range(num_steps), desc="Sampling")
        with torch.no_grad():
            xt = init_samps
            for step in iterator:
                t_current = float(t_schedule[step])
                t_batch = torch.full((self.batch_size,), t_current, dtype=xt.dtype, device=self.device)
                pred = self.score_net(xt, t_batch, y)
                last_step = (step == num_steps - 1)
                xt = self.rwd_sde(xt, pred, t_batch, dt, last_step = last_step)
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
        device : torch.device
            Target device for the module and its components.

        Returns
        -------
        sample_sde (SampleSDE) - moved to the specified device.
        """
        self.device = device
        self.score_net.to(device)
        self.rwd_sde.to(device)
        if self.cond_net:
            self.cond_net.to(device)
        return super().to(device)