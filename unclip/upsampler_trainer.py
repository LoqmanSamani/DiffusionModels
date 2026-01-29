import torch.nn.functional as F
import random
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Dict
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
    `up_net` : nn.Module
        The UnCLIP upsampler model (e.g., UpsamplerUnCLIP) to be trained.
    `train_loader` : torch.utils.data.DataLoader
        DataLoader for training data, providing low- and high-resolution image pairs.
    `optim` : torch.optim.Optimizer
        Optimizer for training the upsampler model.
    `loss_fn` : Callable
        Loss function to compute the difference between predicted and target noise.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `max_epochs` : int, optional
        Maximum number of training epochs (default: 100).
    `device` : str, optional
        Device for computation (default: CUDA).
    `store_path` : str, optional
        Directory to save model checkpoints (default: "unclip_upsampler").
    `patience` : int, optional
        Number of epochs to wait for improvement before early stopping (default: 20).
    `warmup_steps` : int, optional
        Number of epochs for learning rate warmup (default: 10000).
    `val_freq` : int, optional
        Frequency (in epochs) for validation (default: 10).
    `use_ddp` : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    `grad_acc` : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    `log_freq` : int, optional
        Frequency (in epochs) for printing progress (default: 1).
    `use_comp` : bool, optional
        Whether to compile the model using torch.compile (default: False).
    `norm_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `norm_out` : bool, optional
        Whether to normalize inputs/outputs (default: True).
    `use_autocast` : bool, optional
        Whether to use automatic mixed precision training (default: True).
    """
    def __init__(
            self,
            up_net: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 1000,
            device: str = 'cuda',
            store_path: str = "unclip_upsampler",
            patience: int = 100,
            warmup_steps: int = 10000,
            val_freq: int = 10,
            use_ddp: bool = False,
            grad_acc: int = 1,
            log_freq: int = 1,
            use_comp: bool = False,
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            norm_out: bool = True,
            use_autocast: bool = True
    ) -> None:
        super().__init__()
        # training configuration
        self.use_ddp = use_ddp
        self.grad_acc = grad_acc
        self.use_comp = use_comp
        self.use_autocast = use_autocast
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()
        self._compile_models()
        self._wrap_models_for_ddp()

        self.up_net = up_net.to(self.device)
        self.optim = optim
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_freq = val_freq
        self.log_freq = log_freq
        self.norm_range = norm_range
        self.norm_out = norm_out
        self.store_path = store_path
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.best_loss = float('inf')
        self.losses = {'train_losses': [], 'val_losses': []}
        # learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optim,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optim, warmup_steps)

    def forward(self) -> Dict:
        """Trains the UnCLIP upsampler model to predict noise for denoising.

        Executes the training loop, optimizing the upsampler model using low- and high-resolution
        image pairs, mixed precision, gradient clipping, and learning rate scheduling. Supports
        validation, early stopping, and checkpointing.

        Returns
        -------
        losses: dictionary contaions train and validation losses.
        """
        self.up_net.train()
        scaler = torch.GradScaler() if self.use_autocast else None
        wait = 0
        for epoch in range(self.max_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}", disable=not self.master_process)
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            train_losses_epoch = []
            for step, (low_imgs, high_imgs) in enumerate(pbar):
                low_imgs = low_imgs.to(self.device, non_blocking=True)
                high_imgs = high_imgs.to(self.device, non_blocking=True)
                with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', enabled=self.use_autocast):
                    batch_size = high_imgs.shape[0]
                    timesteps = torch.randint(0, self.up_net.fwd_unclip.vs.train_steps, (batch_size,), device=self.device)
                    noise = torch.randn_like(high_imgs)
                    high_imgs_noisy, target = self.up_net.fwd_unclip(high_imgs, noise, timesteps)
                    corr_type = "gaussian_blur" if self.up_net.low_res_size == 64 else "bsr_degradation"
                    low_imgs_corr = self.corrupt_cond_img(low_imgs, corr_type)
                    pred = self.up_net(high_imgs_noisy, timesteps, low_imgs_corr)
                    loss = self.loss_fn(pred, target) / self.grad_acc
                if self.use_autocast:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % self.grad_acc == 0:
                    if self.use_autocast:
                        scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.up_net.parameters(), max_norm=1.0)
                    if self.use_autocast:
                        scaler.step(self.optim)
                        scaler.update()
                    else:
                        self.optim.step()
                    self.optim.zero_grad()
                    # torch.cuda.empty_cache()  # clear memory after optimizer step
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
                print(f"Epoch {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}")

            if self.val_loader is not None and (epoch + 1) % self.val_freq == 0:
                val_loss = self.validate()
                if self.master_process:
                    print(f" | Val Loss: {val_loss:.4f}")
                    print()
                self.scheduler.step(val_loss)
                self.losses['val_losses'].append(val_loss)
            else:
                if self.master_process:
                    print()
                self.scheduler.step(mean_train_loss)

            if self.master_process:
                if mean_train_loss < self.best_loss:
                    self.best_loss = mean_train_loss
                    wait = 0
                    self._save_checkpoint(epoch + 1, self.best_loss)
                else:
                    wait += 1
                    if wait >= self.patience:
                        print("Early stopping triggered")
                        self._save_checkpoint(epoch + 1, mean_train_loss)
                        break
                if (epoch + 1) % self.val_freq == 0:
                    self._save_checkpoint(epoch + 1, mean_train_loss)
        if self.use_ddp:
            destroy_process_group()
        return self.losses

    def _compile_models(self) -> None:
        """Compiles models for optimization if supported.

        Attempts to compile the prior model using torch.compile for performance optimization,
        with fallback to uncompiled models if compilation fails.
        """
        if self.use_comp:
            try:
                self.prior_net = torch.compile(self.prior_net)
                if self.master_process:
                    print("Models compiled successfully")
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

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
    def warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Creates a learning rate scheduler for warmup.

        Generates a scheduler that linearly increases the learning rate from 0 to the
        optimizer's initial value over the specified warmup epochs.

        Parameters
        ----------
        `optimizer` : torch.optim.Optimizer
            Optimizer to apply the scheduler to.
        `warmup_steps` : int
            Number of steps for the warmup phase.

        Returns
        -------
        lr_scheduler : torch.optim.lr_scheduler.LambdaLR
            Learning rate scheduler for warmup.
        """
        def lr_lambda(step):
            if step < warmup_steps:
                return 0.1 + (0.9 * step / warmup_steps)
            return 1.0
        return LambdaLR(optimizer, lr_lambda)

    def _wrap_models_for_ddp(self) -> None:
        """Wraps models with DistributedDataParallel for multi-GPU training.

        Configures the upsampler model for DDP training by wrapping it with DistributedDataParallel.
        """
        if self.use_ddp:
            self.up_net = self.up_net.to(self.ddp_local_rank)
            self.up_net = DDP(
                self.up_net,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

    def corrupt_cond_img(self, x_low: torch.Tensor, corr_type: str = "gaussian_blur") -> torch.Tensor:
        """Corrupts the low-resolution conditioning image for robustness.

        Applies Gaussian blur or BSR degradation to the low-resolution image to simulate
        real-world degradation, as specified in the UnCLIP paper.

        Parameters
        ----------
        `x_low` : torch.Tensor
            Low-resolution input image, shape (batch_size, channels, low_res_size, low_res_size).
        `corr_type` : str, optional
            Type of corruption to apply: "gaussian_blur" or "bsr_degradation" (default: "gaussian_blur").

        Returns
        -------
        x_degraded : torch.Tensor
            Corrupted low-resolution image, same shape as input.
        """
        if corr_type == "gaussian_blur":
            # apply Gaussian blur
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            return self._gaussian_blur(x_low, kernel_size, sigma)
        elif corr_type == "bsr_degradation":
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
        kernel = self._gaussian_kernel(kernel_size, sigma).to(x.device)
        kernel = kernel.expand(x.shape[1], 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

    def _gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
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
        self.up_net.eval()
        val_losses = []
        with torch.no_grad():
            for low_imgs, high_imgs in self.val_loader:
                low_imgs = low_imgs.to(self.device, non_blocking=True)
                high_imgs = high_imgs.to(self.device, non_blocking=True)
                batch_size = high_imgs.shape[0]
                timesteps = torch.randint(0, self.up_net.fwd_unclip.vs.train_steps, (batch_size,), device=self.device)
                noise = torch.randn_like(high_imgs)
                high_imgs_noisy, target = self.up_net.fwd_unclip(high_imgs, noise, timesteps)
                corr_type = "gaussian_blur" if self.up_net.low_res_size == 64 else "bsr_degradation"
                low_imgs_corr = self.corrupt_cond_img(low_imgs, corr_type)
                pred = self.up_net(high_imgs_noisy, timesteps, low_imgs_corr)
                loss = self.loss_fn(pred, target)
                val_losses.append(loss.item())
        val_loss = torch.tensor(val_losses).mean().item()
        if self.use_ddp:
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()
        self.up_net.train()
        return val_loss

    def _save_checkpoint(self, epoch: int, loss: float, pref: str = ""):
        """Saves model checkpoint.

        Saves the state of the upsampler model, its variance scheduler, optimizer, and
        schedulers, with options for best model and epoch-specific checkpoints.

        Parameters
        ----------
        `epoch` : int
            Current epoch number.
        `loss` : float
            Current loss value.
        `prefix` : str, optional
            prefix to add to checkpoint filename, default "".
        """
        if not self.master_process:
            return
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'losses': self.losses,
            'up_net_state_dict': self.up_net.module.state_dict() if self.use_ddp else self.up_net.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'model_channels': self.up_net.model_channels,
            'num_res_blocks': self.up_net.num_res_blocks,
            'normalize': self.norm_out,
            'norm_range': self.norm_range
        }

        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        checkpoint['warmup_scheduler_state_dict'] = self.warmup_lr_scheduler.state_dict()
        try:
            filename = f"{pref}model_epoch_{epoch}.pth"
            filepath = os.path.join(self.store_path, filename)
            os.makedirs(self.store_path, exist_ok=True)
            torch.save(checkpoint, filepath)
            print(f"Model saved at epoch {epoch} with loss: {loss: .4f}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def load_checkpoint(self, check_path: str) -> Tuple[int, float]:
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
            checkpoint = torch.load(check_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found: {check_path}")
        def _load_model_state_dict(model: nn.Module, state_dict: dict, model_name: str) -> None:
            """Helper function to load state dict with DDP compatibility."""
            try:
                if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                if self.master_process:
                    print(f"✓ Loaded {model_name}")
            except Exception as e:
                warnings.warn(f"Failed to load {model_name}: {e}")

        # load core upsampler model
        if 'up_net_state_dict' in checkpoint:
            _load_model_state_dict(self.up_net, checkpoint['up_net_state_dict'],'up_net')
        # load optimizer
        if 'optim_state_dict' in checkpoint:
            try:
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
                if self.master_process:
                    print("✓ Loaded optimizer")
            except Exception as e:
                warnings.warn(f"Failed to load optimizer state: {e}")
        # load schedulers
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

        # verify configuration compatibility
        if 'model_channels' in checkpoint:
            if checkpoint['model_channels'] != self.up_net.model_channels:
                warnings.warn(
                    f"Model channels mismatch: checkpoint={checkpoint['model_channels']}, current={self.up_net.model_channels}")

        if 'num_res_blocks' in checkpoint:
            if checkpoint['num_res_blocks'] != self.up_net.num_res_blocks:
                warnings.warn(
                    f"Num res blocks mismatch: checkpoint={checkpoint['num_res_blocks']}, current={self.up_net.num_res_blocks}")

        if 'normalize' in checkpoint:
            if checkpoint['normalize'] != self.normalize_image_outputs:
                warnings.warn(
                    f"Normalize setting mismatch: checkpoint={checkpoint['normalize']}, current={self.normalize_image_outputs}")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Successfully loaded checkpoint from {check_path}")
            print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        return epoch, loss