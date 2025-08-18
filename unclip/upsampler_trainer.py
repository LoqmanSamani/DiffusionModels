
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Callable, List
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
import os
import warnings




class TrainUpsamplerUnCLIP(nn.Module):
    """Trainer for the UnCLIP upsampler model.

    Orchestrates the training of the UnCLIP upsampler model, integrating forward diffusion,
    noise prediction, and low-resolution image conditioning with optional corruption (Gaussian
    blur or BSR degradation). Supports mixed precision, gradient accumulation, DDP, and
    comprehensive training utilities.

    Parameters
    ----------
    `upsampler_model` : nn.Module
        The UnCLIP upsampler model (e.g., UpsamplerUnCLIP) to be trained.
    `train_loader` : torch.utils.data.DataLoader
        DataLoader for training data, providing low- and high-resolution image pairs.
    `optimizer` : torch.optim.Optimizer
        Optimizer for training the upsampler model.
    `objective` : Callable
        Loss function to compute the difference between predicted and target noise.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `max_epochs` : int, optional
        Maximum number of training epochs (default: 1000).
    `device` : Union[str, torch.device], optional
        Device for computation (default: CUDA if available, else CPU).
    `store_path` : str, optional
        Directory to save model checkpoints (default: "unclip_upsampler").
    `patience` : int, optional
        Number of epochs to wait for improvement before early stopping (default: 100).
    `warmup_epochs` : int, optional
        Number of epochs for learning rate warmup (default: 100).
    `val_frequency` : int, optional
        Frequency (in epochs) for validation (default: 10).
    `use_ddp` : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    `grad_accumulation_steps` : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    `log_frequency` : int, optional
        Frequency (in epochs) for printing progress (default: 1).
    `use_compilation` : bool, optional
        Whether to compile the model using torch.compile (default: False).
    `image_output_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `normalize_image_outputs` : bool, optional
        Whether to normalize inputs/outputs (default: True).
    `use_autocast` : bool, optional
        Whether to use automatic mixed precision training (default: True).
    """

    def __init__(
            self,
            upsampler_model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            store_path: str = "unclip_upsampler",
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            use_ddp: bool = False,
            grad_accumulation_steps: int = 1,
            log_frequency: int = 1,
            use_compilation: bool = False,
            image_output_range: Tuple[float, float] = (-1.0, 1.0),
            normalize_image_outputs: bool = True,
            use_autocast: bool = True
    ) -> None:
        super().__init__()
        # Training configuration
        self.use_ddp = use_ddp
        self.grad_accumulation_steps = grad_accumulation_steps
        self.use_compilation = use_compilation
        self.use_autocast = use_autocast  # Store autocast flag

        # Device initialization
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Setup distributed training
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # Compile and wrap models
        self._compile_models()
        self._wrap_models_for_ddp()

        # Core model
        self.upsampler_model = upsampler_model.to(self.device)
        self.num_timesteps = self.upsampler_model.forward_diffusion.variance_scheduler.num_steps

        # Training components
        self.optimizer = optimizer
        self.objective = objective
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frequency = val_frequency
        self.log_frequency = log_frequency
        self.image_output_range = image_output_range
        self.normalize_image_outputs = normalize_image_outputs

        # Checkpoint management
        self.store_path = store_path

        # Learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)

    def forward(self) -> Tuple[List[float], float]:
        """Trains the UnCLIP upsampler model to predict noise for denoising.

        Executes the training loop, optimizing the upsampler model using low- and high-resolution
        image pairs, mixed precision, gradient clipping, and learning rate scheduling. Supports
        validation, early stopping, and checkpointing.

        Returns
        -------
        train_losses : List[float]
            List of mean training losses per epoch.
        best_val_loss : float
            Best validation or training loss achieved.
        """
        # Set models to training mode
        self.upsampler_model.train()
        if self.upsampler_model.forward_diffusion.variance_scheduler.trainable_beta:
            self.upsampler_model.forward_diffusion.variance_scheduler.train()
        else:
            self.upsampler_model.forward_diffusion.variance_scheduler.eval()

        # Initialize training components
        scaler = torch.GradScaler() if self.use_autocast else None  # Only use scaler with autocast
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        # Main training loop
        for epoch in range(self.max_epochs):
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []

            # Training step loop with gradient accumulation
            for step, (low_res_images, high_res_images) in enumerate(tqdm(self.train_loader, disable=not self.master_process)):
                low_res_images = low_res_images.to(self.device, non_blocking=True)
                high_res_images = high_res_images.to(self.device, non_blocking=True)

                # Forward pass with optional autocast
                if self.use_autocast:
                    with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                        batch_size = high_res_images.shape[0]
                        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
                        noise = torch.randn_like(high_res_images)
                        # Force FP32 for forward_diffusion to avoid NaN in variance scheduling
                        with torch.autocast(device_type='cuda', enabled=False):
                            high_res_images_noisy = self.upsampler_model.forward_diffusion(high_res_images, noise, timesteps)
                        corruption_type = "gaussian_blur" if self.upsampler_model.low_res_size == 64 else "bsr_degradation"
                        low_res_images_corrupted = self.corrupt_conditioning_image(low_res_images, corruption_type)
                        predicted_noise = self.upsampler_model(high_res_images_noisy, timesteps, low_res_images_corrupted)
                        loss = self.objective(predicted_noise, noise) / self.grad_accumulation_steps
                else:
                    batch_size = high_res_images.shape[0]
                    timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
                    noise = torch.randn_like(high_res_images)
                    high_res_images_noisy = self.upsampler_model.forward_diffusion(high_res_images, noise, timesteps)
                    corruption_type = "gaussian_blur" if self.upsampler_model.low_res_size == 64 else "bsr_degradation"
                    low_res_images_corrupted = self.corrupt_conditioning_image(low_res_images, corruption_type)
                    predicted_noise = self.upsampler_model(high_res_images_noisy, timesteps, low_res_images_corrupted)
                    loss = self.objective(predicted_noise, noise) / self.grad_accumulation_steps

                # Backward pass
                if self.use_autocast:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.grad_accumulation_steps == 0:
                    # Clip gradients
                    if self.use_autocast:
                        scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.upsampler_model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.upsampler_model.forward_diffusion.parameters(), max_norm=1.0)

                    # Optimizer step
                    if self.use_autocast:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()  # Clear memory after optimizer step

                train_losses_epoch.append(loss.item() * self.grad_accumulation_steps)

            # Changed: Moved warmup_lr_scheduler.step() here to ensure it is called after optimizer.step()
            # and only once per epoch, matching the intent of warmup_epochs.
            self.warmup_lr_scheduler.step()

            mean_train_loss = self._compute_mean_loss(train_losses_epoch)
            train_losses.append(mean_train_loss)

            if self.master_process and (epoch + 1) % self.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}")

            current_loss = mean_train_loss

            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_loss = self.validate()
                if self.master_process:
                    print(f" | Val Loss: {val_loss:.4f}")
                    print()
                current_loss = val_loss

            self.scheduler.step(current_loss)

            if self.master_process:
                if current_loss < best_val_loss and (epoch + 1) % self.val_frequency == 0:
                    best_val_loss = current_loss
                    wait = 0
                    self._save_checkpoint(epoch + 1, best_val_loss, is_best=True)
                else:
                    wait += 1
                    if wait >= self.patience:
                        print("Early stopping triggered")
                        self._save_checkpoint(epoch + 1, current_loss, suffix="_early_stop")
                        break

        if self.use_ddp:
            destroy_process_group()

        return train_losses, best_val_loss

    def _compute_mean_loss(self, losses: List[float]) -> float:
        """Computes mean loss with DDP synchronization if needed.

        Calculates the mean of the provided losses and synchronizes the result across
        processes in DDP mode.

        Parameters
        ----------
        `losses` : List[float]
            List of loss values for the current epoch.

        Returns
        -------
        mean_loss : float
            Mean loss value, synchronized if using DDP.
        """
        if not losses:
            return 0.0
        mean_loss = sum(losses) / len(losses)
        if self.use_ddp:
            # synchronize loss across all processes
            loss_tensor = torch.tensor(mean_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            mean_loss = (loss_tensor / self.ddp_world_size).item()

        return mean_loss

    def _setup_ddp(self) -> None:
        """Sets up Distributed Data Parallel training configuration.

        Initializes the process group, sets up rank information, and configures the CUDA
        device for the current process in DDP mode.
        """
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
        """Sets up single GPU or CPU training configuration.

        Configures the training setup for single-device operation, setting rank and process
        information for non-DDP training.
        """
        self.ddp_rank = 0
        self.ddp_local_rank = 0
        self.ddp_world_size = 1
        self.master_process = True

    @staticmethod
    def warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Creates a learning rate scheduler for warmup.

        Generates a scheduler that linearly increases the learning rate from 0 to the
        optimizer's initial value over the specified warmup epochs.

        Parameters
        ----------
        `optimizer` : torch.optim.Optimizer
            Optimizer to apply the scheduler to.
        `warmup_epochs` : int
            Number of epochs for the warmup phase.

        Returns
        -------
        lr_scheduler : torch.optim.lr_scheduler.LambdaLR
            Learning rate scheduler for warmup.
        """
        def lr_lambda(epoch):
            return min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0

        return LambdaLR(optimizer, lr_lambda)

    def _wrap_models_for_ddp(self) -> None:
        """Wraps models with DistributedDataParallel for multi-GPU training.

        Configures the upsampler model for DDP training by wrapping it with DistributedDataParallel.
        """
        if self.use_ddp:
            self.upsampler_model = self.upsampler_model.to(self.ddp_local_rank)
            self.upsampler_model = DDP(
                self.upsampler_model,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

    def _compile_models(self) -> None:
        """Compiles models for optimization if supported.

        Attempts to compile the upsampler model using torch.compile for optimization,
        falling back to uncompiled execution if compilation fails.
        """
        if self.use_compilation:
            try:
                self.upsampler_model = self.upsampler_model.to(self.device)
                self.upsampler_model = torch.compile(self.upsampler_model, mode="reduce-overhead")

                if self.master_process:
                    print("Models compiled successfully")
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

    def corrupt_conditioning_image(self, x_low: torch.Tensor, corruption_type: str = "gaussian_blur") -> torch.Tensor:
        """Corrupts the low-resolution conditioning image for robustness.

        Applies Gaussian blur or BSR degradation to the low-resolution image to simulate
        real-world degradation, as specified in the UnCLIP paper.

        Parameters
        ----------
        `x_low` : torch.Tensor
            Low-resolution input image, shape (batch_size, channels, low_res_size, low_res_size).
        `corruption_type` : str, optional
            Type of corruption to apply: "gaussian_blur" or "bsr_degradation" (default: "gaussian_blur").

        Returns
        -------
        x_degraded : torch.Tensor
            Corrupted low-resolution image, same shape as input.
        """
        if corruption_type == "gaussian_blur":
            # apply Gaussian blur
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            return self._gaussian_blur(x_low, kernel_size, sigma)
        elif corruption_type == "bsr_degradation":
            # more diverse BSR degradation for second upsampler
            return self._bsr_degradation(x_low)
        else:
            return x_low

    def _gaussian_blur(self, x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """Applies Gaussian blur to the input image.

        Parameters
        ----------
        `x` : torch.Tensor
            Input image tensor, shape (batch_size, channels, height, width).
        `kernel_size` : int
            Size of the Gaussian kernel.
        `sigma` : float
            Standard deviation of the Gaussian distribution.

        Returns
        -------
        x_blurred : torch.Tensor
            Blurred image tensor, same shape as input.
        """
        # create Gaussian kernel
        kernel = self._get_gaussian_kernel(kernel_size, sigma).to(x.device)
        kernel = kernel.expand(x.shape[1], 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

    def _get_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Generates a 2D Gaussian kernel.

        Parameters
        ----------
        `kernel_size` : int
            Size of the Gaussian kernel.
        `sigma` : float
            Standard deviation of the Gaussian distribution.

        Returns
        -------
        kernel : torch.Tensor
            2D Gaussian kernel, shape (kernel_size, kernel_size).
        """
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g[:, None] * g[None, :]

    def _bsr_degradation(self, x: torch.Tensor) -> torch.Tensor:
        """Applies BSR degradation to the input image.

        Simulates degradation with noise and Gaussian blur, as used in the UnCLIP paper
        for the second upsampler.

        Parameters
        ----------
        `x` : torch.Tensor
            Input image tensor, shape (batch_size, channels, height, width).

        Returns
        -------
        x_degraded : torch.Tensor
            Degraded image tensor, same shape as input, clamped to [-1, 1].
        """
        # add noise
        noise_level = random.uniform(0.0, 0.1)
        noise = torch.randn_like(x) * noise_level

        # apply blur
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 3.0)
        x_degraded = self._gaussian_blur(x + noise, kernel_size, sigma)

        return torch.clamp(x_degraded, -1.0, 1.0)

    def validate(self) -> float:
        """Validates the UnCLIP upsampler model.

        Computes the validation loss by applying forward diffusion to high-resolution images,
        predicting noise with the upsampler model conditioned on corrupted low-resolution images,
        and comparing predicted noise to ground truth.

        Returns
        -------
        val_loss : float
            Mean validation loss.
        """
        # set models to eval mode for evaluation
        self.upsampler_model.eval()
        self.upsampler_model.forward_diffusion.eval()

        val_losses = []

        with torch.no_grad():
            for low_res_images, high_res_images in self.val_loader:
                low_res_images = low_res_images.to(self.device, non_blocking=True)
                high_res_images = high_res_images.to(self.device, non_blocking=True)
                batch_size = high_res_images.shape[0]
                timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
                noise = torch.randn_like(high_res_images)
                high_res_images_noisy = self.upsampler_model.forward_diffusion(high_res_images, noise, timesteps)
                corruption_type = "gaussian_blur" if self.upsampler_model.low_res_size == 64 else "bsr_degradation"
                low_res_images_corrupted = self.corrupt_conditioning_image(low_res_images, corruption_type)
                predicted_noise = self.upsampler_model(high_res_images_noisy, timesteps, low_res_images_corrupted)
                # compute loss
                loss = self.objective(predicted_noise, noise)
                val_losses.append(loss.item())

        # compute average loss
        val_loss = torch.tensor(val_losses).mean().item()

        if self.use_ddp:
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()

        # return to training mode
        self.upsampler_model.train()
        if not self.upsampler_model.forward_diffusion.variance_scheduler.trainable_beta:
            self.upsampler_model.forward_diffusion.variance_scheduler.eval()

        return val_loss

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False, suffix: str = ""):
        """Saves model checkpoint.

        Saves the state of the upsampler model, its variance scheduler, optimizer, and
        schedulers, with options for best model and epoch-specific checkpoints.

        Parameters
        ----------
        `epoch` : int
            Current epoch number.
        `loss` : float
            Current loss value.
        `is_best` : bool, optional
            Whether to save as the best model checkpoint (default: False).
        `suffix` : str, optional
            Suffix to add to checkpoint filename, default "".
        """
        if not self.master_process:
            return
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            # Core model
            'upsampler_model_state_dict': self.upsampler_model.module.state_dict() if self.use_ddp else self.upsampler_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Training configuration
            'model_channels': self.upsampler_model.model_channels,
            'num_res_blocks': self.upsampler_model.num_res_blocks,
            'normalize': self.normalize_image_outputs,
            'output_range': self.image_output_range
        }

        # Save variance scheduler (submodule of forward_diffusion)
        checkpoint['variance_scheduler_state_dict'] = (
            self.upsampler_model.module.forward_diffusion.variance_scheduler.state_dict() if self.use_ddp
            else self.upsampler_model.forward_diffusion.variance_scheduler.state_dict()
        )

        # Save schedulers state
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        checkpoint['warmup_scheduler_state_dict'] = self.warmup_lr_scheduler.state_dict()

        filename = f"unclip_upsampler_epoch_{epoch}{suffix}.pth"
        if is_best:
            filename = f"unclip_upsampler_best{suffix}.pth"

        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)

        if is_best:
            print(f"Best model saved: {filepath}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """Loads model checkpoint.

        Restores the state of the upsampler model, its variance scheduler, optimizer, and
        schedulers from a saved checkpoint, handling DDP compatibility.

        Parameters
        ----------
        `checkpoint_path` : str
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
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        def _load_model_state_dict(model: nn.Module, state_dict: dict, model_name: str) -> None:
            """Helper function to load state dict with DDP compatibility."""
            try:
                # Handle DDP state dict compatibility
                if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                model.load_state_dict(state_dict)
                if self.master_process:
                    print(f"✓ Loaded {model_name}")
            except Exception as e:
                warnings.warn(f"Failed to load {model_name}: {e}")

        # Load core upsampler model
        if 'upsampler_model_state_dict' in checkpoint:
            _load_model_state_dict(self.upsampler_model, checkpoint['upsampler_model_state_dict'],
                                   'upsampler_model')

        # Load variance scheduler (submodule of forward_diffusion)
        if 'variance_scheduler_state_dict' in checkpoint or 'hyper_params_state_dict' in checkpoint:
            state_dict = checkpoint.get('variance_scheduler_state_dict', checkpoint.get('hyper_params_state_dict'))
            try:
                _load_model_state_dict(self.upsampler_model.forward_diffusion.variance_scheduler, state_dict, 'variance_scheduler')
            except Exception as e:
                warnings.warn(f"Failed to load variance scheduler: {e}")

        # Load optimizer
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.master_process:
                    print("✓ Loaded optimizer")
            except Exception as e:
                warnings.warn(f"Failed to load optimizer state: {e}")

        # Load schedulers
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if self.master_process:
                    print("✓ Loaded main scheduler")
            except Exception as e:
                warnings.warn(f"Failed to load scheduler state: {e}")

        if 'warmup_scheduler_state_dict' in checkpoint:
            try:
                self.warmup_lr_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
                if self.master_process:
                    print("✓ Loaded warmup scheduler")
            except Exception as e:
                warnings.warn(f"Failed to load warmup scheduler state: {e}")

        # Verify configuration compatibility
        if 'model_channels' in checkpoint:
            if checkpoint['model_channels'] != self.upsampler_model.model_channels:
                warnings.warn(
                    f"Model channels mismatch: checkpoint={checkpoint['model_channels']}, current={self.upsampler_model.model_channels}")

        if 'num_res_blocks' in checkpoint:
            if checkpoint['num_res_blocks'] != self.upsampler_model.num_res_blocks:
                warnings.warn(
                    f"Num res blocks mismatch: checkpoint={checkpoint['num_res_blocks']}, current={self.upsampler_model.num_res_blocks}")

        if 'normalize' in checkpoint:
            if checkpoint['normalize'] != self.normalize_image_outputs:
                warnings.warn(
                    f"Normalize setting mismatch: checkpoint={checkpoint['normalize']}, current={self.normalize_image_outputs}")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        return epoch, loss


"""
from prior_diff import VarianceSchedulerUnCLIP, ForwardUnCLIP
from upsampler import UpsamplerUnCLIP
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define a dummy dataset for example purposes (replace with real dataset in practice)
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, low_res_size=64, high_res_size=256):
        self.num_samples = num_samples
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random low-res and high-res images (in practice, load from disk or augment)
        low_res_image = torch.rand(3, self.low_res_size, self.low_res_size) * 2 - 1  # Normalize to [-1, 1]
        high_res_image = torch.rand(3, self.high_res_size, self.high_res_size) * 2 - 1  # Normalize to [-1, 1]
        return low_res_image, high_res_image

# Instantiate the variance scheduler
hyp = VarianceSchedulerUnCLIP(
    num_steps=400,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=True,
    beta_method="linear"
)

# Instantiate the forward diffusion process
forward = ForwardUnCLIP(hyp)

# Instantiate the upsampler model
model = UpsamplerUnCLIP(
    forward_diffusion=forward,
    in_channels=3,
    out_channels=3,
    model_channels=32,
    num_res_blocks=2,
    channel_mult=(1, 2, 4, 8),
    dropout=0.1,
    time_embed_dim=32,
    low_res_size=64,
    high_res_size=256
)

# Create train loader with dummy dataset (replace with real DataLoader for your dataset)
train_dataset = DummyDataset(num_samples=4)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

# Optional validation loader (using same dummy for example)
val_dataset = DummyDataset(num_samples=2)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Define objective (loss function, e.g., MSE for noise prediction)
objective = nn.MSELoss()

# Instantiate the trainer
trainer = TrainUpsamplerUnCLIP(
    upsampler_model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    objective=objective,
    val_loader=val_loader,  # Optional
    max_epoch=10,  # Small number for example; increase for real training
    device='cuda' if torch.cuda.is_available() else 'cpu',
    store_path="upsampler",
    patience=10,
    warmup_epochs=2,
    val_frequency=5,
    use_ddp=False,  # Set to True if using distributed training
    num_grad_accumulation=2,
    progress_frequency=1,
    compilation=True,  # Set to True if torch.compile is desired and supported
    output_range=(-1.0, 1.0),
    normalize=True,
    use_autocast=False
)

# Run the training
train_losses, best_val_loss = trainer()

# Print results
print(f"Training losses: {train_losses}")
print(f"Best validation loss: {best_val_loss}")

"""

