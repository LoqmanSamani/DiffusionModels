
import torch.nn.functional as F
import math
import random

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Callable, Any
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
import os
import warnings


class UpsamplerTrainer(nn.Module):
    """Trainer for upsampler models."""

    def __init__(
            self,
            upsampler_model: nn.Module,
            noise_scheduler: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            metrics_: Optional[Any] = None,
            max_epoch: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            store_path: str = "unclip_upsampler",
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            use_ddp: bool = False,
            num_grad_accumulation: int = 1,
            progress_frequency: int = 1,
            compilation: bool = False,
            output_range: Tuple[float, float] = (-1.0, 1.0),
            normalize: bool = True
    ):
        super().__init__()
        # training configuration
        self.use_ddp = use_ddp
        self.num_grad_accumulation = num_grad_accumulation
        self.compilation = compilation
        self.device = torch.device(device) or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # setup distributed training
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # compile and wrap models
        self._compile_models()
        self._wrap_models_for_ddp()

        # core model
        self.upsampler_model = upsampler_model.to(device)
        self.noise_scheduler = noise_scheduler.to(device)

        # training components
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.train_loader = train_loader
        self.val_loader = val_loader

        # training parameters
        self.max_epoch = max_epoch
        self.patience = patience
        self.val_frequency = val_frequency
        self.progress_frequency = progress_frequency
        self.output_range = output_range
        self.normalize = normalize

        # checkpoint management
        self.store_path = store_path

        # learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)

    def forward(self):#"-> Tuple[List[float], float]:

        # set models to training mode
        self.upsampler_model.train()

        if self.variance_scheduler.trainable_beta:
            self.variance_scheduler.train()
        else:
            self.variance_scheduler.eval()


        # initialize training components
        scaler = torch.GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0
        # main training loop
        for epoch in range(self.max_epoch):
            # set epoch for distributed sampler if using DDP
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []

            # training step loop with gradient accumulation
            for step, (images, texts) in enumerate(tqdm(self.train_loader, disable=not self.master_process)):
                images = images.to(self.device, non_blocking=True)
        pass

    def _setup_ddp(self) -> None:
        required_env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"DDP enabled but {var} environment variable not set")

        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA but CUDA is not available")

        if not torch.distributed.is_initialized():
            init_process_group(backend="nccl")

        self.ddp_rank = int(os.environ["RANK"])
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])

        self.device = torch.device(f"cuda:{self.ddp_local_rank}")
        torch.cuda.set_device(self.device)

        self.master_process = self.ddp_rank == 0

        if self.master_process:
            print(f"DDP initialized with world_size={self.ddp_world_size}")

    def _setup_single_gpu(self) -> None:
        """setup single GPU or CPU training configuration."""
        self.ddp_rank = 0
        self.ddp_local_rank = 0
        self.ddp_world_size = 1
        self.master_process = True

    @staticmethod
    def warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
        def lr_lambda(epoch):
            return min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0

        return LambdaLR(optimizer, lr_lambda)

    def _wrap_models_for_ddp(self) -> None:
        """Wrap models with DistributedDataParallel for multi-GPU training."""
        if self.use_ddp:
            self.upsampler_model = self.upsampler_model.to(self.ddp_local_rank)
            self.upsampler_model = DDP(
                self.upsampler_model,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

    def _compile_models(self) -> None:
        """Compile models for optimization if supported."""
        if self.compilation:
            try:
                self.upsampler_model = self.upsampler_model.to(self.device)
                self.upsampler_model = torch.compile(self.upsampler_model, mode="reduce-overhead")

                if self.master_process:
                    print("Models compiled successfully")
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")




    def corrupt_conditioning_image(self, x_low: torch.Tensor, corruption_type: str = "gaussian_blur") -> torch.Tensor:
        """Corrupt conditioning image for robustness (as mentioned in paper)."""
        if corruption_type == "gaussian_blur":
            # Apply Gaussian blur
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            return self._gaussian_blur(x_low, kernel_size, sigma)
        elif corruption_type == "bsr_degradation":
            # More diverse BSR degradation for second upsampler
            # Simplified version - in practice, you'd use more sophisticated degradation
            return self._bsr_degradation(x_low)
        else:
            return x_low

    def _gaussian_blur(self, x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur."""
        # Create Gaussian kernel
        kernel = self._get_gaussian_kernel(kernel_size, sigma).to(x.device)
        kernel = kernel.expand(x.shape[1], 1, kernel_size, kernel_size)

        # Apply convolution
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

    def _get_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Generate Gaussian kernel."""
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g[:, None] * g[None, :]

    def _bsr_degradation(self, x: torch.Tensor) -> torch.Tensor:
        """Simplified BSR degradation."""
        # Add noise
        noise_level = random.uniform(0.0, 0.1)
        noise = torch.randn_like(x) * noise_level

        # Apply blur
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 3.0)
        x_degraded = self._gaussian_blur(x + noise, kernel_size, sigma)

        return torch.clamp(x_degraded, -1.0, 1.0)

    def compute_loss(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        """Compute training loss for upsampler."""
        batch_size = x_high.shape[0]

        # Sample timesteps
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

        # Sample noise
        noise = torch.randn_like(x_high)

        # Add noise to high-resolution image
        x_high_noisy = self.add_noise(x_high, noise, timesteps)

        # Corrupt conditioning image for robustness
        corruption_type = "gaussian_blur" if self.upsampler_model.low_res_size == 64 else "bsr_degradation"
        x_low_corrupted = self.corrupt_conditioning_image(x_low, corruption_type)

        # Predict noise
        predicted_noise = self.upsampler_model(x_high_noisy, timesteps, x_low_corrupted)

        # Compute MSE loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss