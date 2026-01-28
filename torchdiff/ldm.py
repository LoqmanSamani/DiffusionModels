"""
**Latent Diffusion Models (LDM)**

This module provides a framework for training and sampling Latent Diffusion Models, as
described in Rombach et al. (2022, "High-Resolution Image Synthesis with Latent Diffusion
Models"). It supports diffusion in the latent space using a variational autoencoder
(compressor model), includes utilities for training the autoencoder, noise predictor, and
conditional model, and provides metrics for evaluating generated images. The framework is
compatible with DDPM, DDIM, and SDE diffusion models, supporting both unconditional and
conditional generation with text prompts.

**Components**

- **AutoencoderLDM**: Variational autoencoder for compressing images to latent space and
  decoding back to image space.
- **TrainAE**: Trainer for AutoencoderLDM, optimizing reconstruction and regularization
  losses with evaluation metrics.
- **TrainLDM**: Training loop with mixed precision, warmup, and scheduling for the noise
  predictor and conditional model (e.g., TextEncoder with projection layers) in latent
  space, with image-domain evaluation metrics using a reverse diffusion model.
- **SampleLDM**: Image generation from trained models, decoding from latent to image space.


**Notes**


- The `scheduler` parameter expects an external hyperparameter module (e.g.,
  SchedulerDDPM, SchedulerSDE) as an nn.Module for noise schedule management.
- AutoencoderLDM serves as the `comp_net` in TrainLDM and SampleLDM, providing
  `encode` and `decode` methods for latent space conversion. It supports KL-divergence or
  vector quantization (VQ) regularization, using internal components (DownBlock, UpBlock,
  Conv3, DownSampling, UpSampling, Attention, VectorQuantizer).
- TrainAE trains AutoencoderLDM, optimizing reconstruction (MSE), regularization (KL or
  VQ), and optional perceptual (LPIPS) losses, with metrics (MSE, PSNR, SSIM, FID, LPIPS)
  computed via the Metrics class, KL warmup, early stopping, and learning rate scheduling.
- TrainLDM trains the noise predictor and conditional model, optimizing MSE between
  predicted and ground truth noise, with optional validation metrics (MSE, PSNR, SSIM, FID,
  LPIPS) on generated images decoded from latents sampled using a reverse diffusion model
  (e.g., ReverseDDPM).
- SampleLDM supports multiple diffusion models ("ddpm", "ddim", "sde") via the `model`
  parameter, requiring compatible `reverse_diffusion` modules (e.g., ReverseDDPM,
  ReverseDDIM, ReverseSDE).


**References**

- Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models."
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.


- Esser, Patrick, Robin Rombach, and Bjorn Ommer. "Taming transformers for high-resolution image synthesis."
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

---------------------------------------------------------------------------------
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any, Callable, List, Union, Dict
from typing_extensions import Self
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
import torch.utils.checkpoint as checkpoint
import warnings
from tqdm import tqdm
from torchvision.utils import save_image
import os


###==================================================================================================================###

class TrainLDM(nn.Module):
    """Trainer for the noise/score/v predictor in Latent Diffusion Models.

    Optimizes the noise predictor and conditional model (e.g., TextEncoder)
    to predict noise in the latent space of AutoencoderLDM, using a diffusion model (e.g., DDPM, DDIM, SDE).
    Supports mixed precision, conditional generation with text prompts, and evaluation metrics
    (MSE, PSNR, SSIM, FID, LPIPS) for generated images during validation, using a specified reverse
    diffusion model.

    Parameters
    ----------
    diff_type : str
        Diffusion model type ("ddpm", "ddim", "sde").
    fwd_diff : ForwardDDPM, ForwardDDIM, or ForwardSDE
        Forward diffusion model defining the noise schedule.
    rwd_diff : ReverseDDPM, ReverseDDIM, or ReverseSDE
        Reverse diffusion model for sampling during validation (default: None).
    diff_net : torch.nn.Module
        Model to predict noise/score/v in the latent space (e.g., DiffusionNetwork).
    comp_net : torch.nn.Module
        Variational autoencoder for encoding/decoding latents.
    optim : torch.optim.Optimizer
        Optimizer for the noise predictor and conditional model (e.g., Adam).
    loss_fn : Callable
        Loss function for noise prediction (e.g., MSELoss).
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data (default: None).
    cond_net : TextEncoder, optional
        Text encoder with projection layers for conditional generation (default: None).

    metrics_ : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    max_epochs : int, optional
        Maximum number of training epochs (default: 100).
    device : str, optional
        Device for computation (e.g., 'cuda', 'cpu') (default: 'cuda').
    store_path : str, optional
        Path to save model checkpoints (default: None, uses 'ldm_train').
    patience : int, optional
        Number of epochs to wait for early stopping if validation loss doesn’t improve
        (default: 20).
    warmup_steps : int, optional
        Number of steps for learning rate warmup (default: 1000).
    tokenizer : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    max_token_length : int, optional
        Maximum sequence length for tokenized text (default: 77).
    val_freq : int, optional
        Frequency (in epochs) for validation and metric computation (default: 10).
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
            diff_type: str,
            fwd_diff: torch.nn.Module,
            rwd_diff: torch.nn.Module,
            diff_net: torch.nn.Module,
            comp_net: torch.nn.Module,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            cond_net: Optional[torch.nn.Module] = None,
            metrics_: Optional[Any] = None,
            max_epochs: int = 100,
            device: str = 'cuda',
            store_path: Optional[str] = None,
            patience: int = 20,
            warmup_steps: int = 1000,
            tokenizer: Optional[BertTokenizer] = None,
            max_token_length: int = 77,
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
        if diff_type not in ["ddpm", "ddim", "sde"]:
            raise ValueError(f"Unknown model: {diff_type}. Supported: ddpm, ddim, sde")
        self.diff_type = diff_type
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

        self.fwd_diff = fwd_diff.to(self.device)
        self.rwd_diff = rwd_diff.to(self.device)
        self.diff_net = diff_net.to(self.device)
        self.comp_net = comp_net.to(self.device)
        self.cond_net = cond_net.to(self.device) if cond_net else None
        self.metrics_ = metrics_
        self.optim = optim
        self.loss_fn = loss_fn
        self.store_path = store_path or "ldm_train"
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
                if isinstance(self.fwd_diff.vs, nn.Module):
                    self.fwd_diff.vs.load_state_dict(checkpoint['scheduler_model'])
                if isinstance(self.rwd_diff.vs, nn.Module):
                    self.rwd_diff.vs.load_state_dict(checkpoint['scheduler_model'])
                else:
                    self.fwd_diff.vs = checkpoint['scheduler_model']
                    self.rwd_diff.vs = checkpoint['scheduler_model']
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
        """Trains the noise/score/v/x0 predictor and conditional model with mixed precision and evaluation metrics.

        Optimizes the noise predictor and conditional model (e.g., TextEncoder with projection layers)
        using the forward diffusion model’s noise schedule, with text conditioning. Performs validation
        with image-domain metrics (MSE, PSNR, SSIM, FID, LPIPS) using the reverse diffusion model,
        saves checkpoints for the best validation loss, and supports early stopping.

        Returns
        -------
        losses : dictionary of train and validation losses
        """
        self.diff_net.train()
        if self.cond_net is not None:
            self.cond_net.train()
        self.comp_net.eval()  # pre-trained compressor model
        if self.use_comp:
            try:
                self.diff_net = torch.compile(self.diff_net)
                if self.cond_net is not None:
                    self.cond_net = torch.compile(self.cond_net)
                self.comp_net = torch.compile(self.comp_net)
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

        self._wrap_models_for_ddp()
        scaler = torch.GradScaler()
        wait = 0
        diff_steps = 0
        if self.diff_type == "ddpm":
            diff_steps = self.fwd_diff.vs.time_steps
        elif self.diff_type == "ddim":
            diff_steps = self.fwd_diff.vs.train_steps
        for epoch in range(self.max_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}", disable=not self.master_process)
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            train_losses_epoch = []
            for step, (x, y) in enumerate(pbar):
                x = x.to(self.device)
                with torch.no_grad():
                    x, _ = self.comp_net.encode(x)
                if self.cond_net is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    noise = torch.randn_like(x)
                    if self.diff_type == 'sde':
                        t = self.sample_time(x.shape[0], self.time_eps)
                    else:
                        t = torch.randint(0, diff_steps, (x.shape[0],), device=x.device)
                    xt, target = self.fwd_diff(x, t, noise)
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
                'scheduler_model': self.fwd_diff.vs.state_dict(),
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
        """Validates the noise predictor and computes evaluation metrics.

        Computes validation loss (MSE between predicted and ground truth noise) and generates
        samples using the reverse diffusion model. Evaluates image quality metrics if available.

        Returns
        -------
        tuple
            (val_loss, fid, mse, psnr, ssim, lpips_score) where metrics may be None if not computed.
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
                x, _ = self.comp_net.encode(x)
                if self.cond_net is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None

                noise = torch.randn_like(x)
                if self.diff_type == 'sde':
                    t = self.sample_time(x.shape[0], self.time_eps)
                elif self.diff_type == 'ddpm':
                    t = torch.randint(0, self.fwd_diff.vs.time_steps, (x.shape[0],), device=x.device)
                else:
                    t = torch.randint(0, self.fwd_diff.vs.train_steps, (x.shape[0],), device=x.device)
                xt, target = self.fwd_diff(x, t, noise)
                pred = self.diff_net(xt, t, y_encoded, clip_embeddings=None)
                loss = self.loss_fn(pred, target)
                val_losses.append(loss.item())

                if self.metrics_ is not None and self.rwd_diff is not None:
                    xt = torch.randn_like(x)
                    if self.diff_type == 'ddpm':
                        for t in reversed(range(self.fwd_diff.vs.time_steps)):
                            time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                            pred = self.diff_net(xt, time_steps, y_encoded, clip_embeddings=None)
                            xt, _ = self.rwd_diff(xt, pred, time_steps)
                    elif self.diff_type == 'ddim':
                        timesteps = self.fwd_diff.vs.inference_timesteps.flip(0)
                        for i in range(len(timesteps) - 1):
                            t_current = timesteps[i].item()
                            t_next = timesteps[i + 1].item()
                            time = torch.full((xt.shape[0],), t_current, device=self.device, dtype=torch.long)
                            prev_time = torch.full((xt.shape[0],), t_next, device=self.device, dtype=torch.long)
                            pred = self.diff_net(xt, time, y_encoded, clip_embeddings=None)
                            xt, _ = self.rwd_diff(xt, time, prev_time, pred)
                    else:
                        t_schedule = torch.linspace(1.0, self.time_eps, self.num_steps + 1)
                        dt = torch.tensor(-(1.0 - self.time_eps) / self.num_steps, device=xt.device, dtype=xt.dtype)
                        for t in range(self.num_steps):
                            t_current = float(t_schedule[t])
                            t_batch = torch.full((xt.shape[0],), t_current, dtype=xt.dtype, device=self.device)
                            pred = self.diff_net(xt, t_batch, y_encoded, None)
                            last_step = (t == self.num_steps - 1)
                            xt = self.rwd_diff(xt, pred, t_batch, dt, last_step=last_step)

                    x_hat = self.comp_net.decode(xt)
                    x_hat = torch.clamp(x_hat, min=self.norm_range[0], max=self.norm_range[1])
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

class SampleLDM(nn.Module):
    """Sampler for generating images using Latent Diffusion Models (LDM).

    Generates images by iteratively denoising random noise in the latent space using a
    reverse diffusion process, decoding the result back to the image space with a
    pre-trained compressor, as described in Rombach et al. (2022). Supports DDPM, DDIM,
    and SDE diffusion models, as well as conditional generation with text prompts.

    Parameters
    ----------
    diff_type : str
        Diffusion model type. Supported: "ddpm", "ddim", "sde".
    rwd_diff : nn.Module
        Reverse diffusion module (e.g., ReverseDDPM, ReverseDDIM, ReverseSDE).
    diff_net : nn.Module
        Model to predict noise added during the forward diffusion process.
    comp_net : nn.Module
        Pre-trained model to encode/decode between image and latent spaces (e.g., AutoencoderLDM).
    img_size : tuple
        Shape of generated images as (height, width).
    cond_net : nn.Module, optional
        Model for conditional generation (e.g., TextEncoder), default None.
    tokenizer : str or BertTokenizer, optional
        Tokenizer for processing text prompts, default "bert-base-uncased".
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for latent representations (default: 3).
    device : str
        Device for computation (default: CUDA).
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    norm_range : tuple, optional
        Range for clamping generated images (min, max), default (-1, 1).
    """
    def __init__(
            self,
            diff_type: str,
            rwd_diff: torch.nn.Module,
            diff_net: torch.nn.Module,
            comp_net: torch.nn.Module,
            num_steps: int,
            img_size: Tuple[float, float],
            cond_net: Optional[torch.nn.Module] = None,
            tokenizer: str = "bert-base-uncased",
            batch_size: int = 1,
            in_channels: int = 3,
            device: str = 'cuda',
            max_token_length: int = 77,
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            time_eps: float = 1e-5,
            *args
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.diff_type = diff_type
        self.num_steps = num_steps
        self.diff_net = diff_net.to(self.device)
        self.rwd_diff = rwd_diff.to(self.device)
        self.comp_net = comp_net.to(self.device)
        self.cond_net = cond_net.to(self.device) if cond_net else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.in_channels = in_channels
        self.img_size = img_size
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.norm_range = norm_range
        self.time_eps = time_eps
        if not isinstance(img_size, (tuple, list)) or len(img_size) != 2 or not all(isinstance(s, int) and s > 0 for s in img_size):
            raise ValueError("img_size must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not isinstance(norm_range, (tuple, list)) or len(norm_range) != 2 or norm_range[0] >= norm_range[1]:
            raise ValueError("norm_range must be a tuple (min, max) with min < max")

    def tokenize(self, prompts: Union[List, str]):
        """Tokenizes text prompts for conditional generation.

        Converts input prompts into tokenized tensors using the specified tokenizer.

        Parameters
        ----------
        prompts : str or list
            Text prompt(s) for conditional generation. Can be a single string or a list of strings.

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
            conds: Optional[Union[List, str]] = None,
            norm_output: bool = True,
            save_imgs: bool = True,
            save_path: str = "ldm_samples"
    ) -> torch.Tensor:
        """Generates images using the reverse diffusion process in the latent space.

        Iteratively denoises random noise in the latent space using the specified reverse
        diffusion model (DDPM, DDIM, SDE), then decodes the result to the image space
        with the compressor model. Supports conditional generation with text prompts.

        Parameters
        ----------
        conds : str or list, optional
            Text prompt(s) for conditional generation, default None.
        norm_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        save_imgs : bool, optional
            If True, saves generated images to `save_path` (default: True).
        save_path : str, optional
            Directory to save generated images (default: "ldm_samples").

        Returns
        -------
        samps (torch.Tensor) - Generated images, shape (batch_size, channels, height, width).
        If `norm_output` is True, images are normalized to [0, 1]; otherwise, they are clamped to `norm_range`.
        """
        if conds is not None and self.cond_net is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conds is None and self.cond_net is not None:
            raise ValueError("Conditions must be provided for conditional model")
        init_samps = torch.randn(self.batch_size, self.in_channels, self.img_size[0], self.img_size[1]).to(self.device)
        self.diff_net.eval()
        self.comp_net.eval()
        if self.cond_net:
            self.cond_net.eval()

        with torch.no_grad():
            xt = init_samps
            xt, _ = self.comp_net.encode(xt)
            if self.cond_net is not None and conds is not None:
                input_ids, attention_masks = self.tokenize(conds)
                key_padding_mask = (attention_masks == 0)
                y = self.cond_net(input_ids, key_padding_mask)
            else:
                y = None
            if self.diff_type == 'ddpm':
                iterator = tqdm(
                    reversed(range(self.rwd_diff.vs.time_steps)),
                    total=self.rwd_diff.vs.time_steps,
                    desc="Sampling",
                    dynamic_ncols=True,
                    leave=True
                )
                for t in iterator:
                    time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                    pred = self.diff_net(xt, time_steps, y, clip_embeddings=None)
                    xt, _ = self.rwd_diff(xt, pred, time_steps)
            elif self.diff_type == 'ddim':
                timesteps = self.rwd_diff.vs.inference_timesteps.flip(0)
                iterator = tqdm(
                    range(len(timesteps) - 1),
                    total=len(timesteps) - 1,
                    desc="Sampling",
                    dynamic_ncols=True,
                    leave=True
                )
                for t in iterator:
                    t_current = timesteps[t].item()
                    t_next = timesteps[t + 1].item()
                    time = torch.full((xt.shape[0],), t_current, device=self.device, dtype=torch.long)
                    prev_time = torch.full((xt.shape[0],), t_next, device=self.device, dtype=torch.long)
                    pred = self.diff_net(xt, time, y, clip_embeddings=None)
                    xt, _ = self.rwd_diff(xt, time, prev_time, pred)
            else:
                iterator = tqdm(
                    range(self.num_steps),
                    total=self.num_steps,
                    desc="Sampling",
                    dynamic_ncols=True,
                    leave=True
                )
                t_schedule = torch.linspace(1.0, self.time_eps, self.num_steps + 1)
                dt = torch.tensor(-(1.0 - self.time_eps) / self.num_steps, device=xt.device, dtype=xt.dtype)
                for t in iterator:
                    t_current = float(t_schedule[t])
                    t_batch = torch.full((xt.shape[0],), t_current, dtype=xt.dtype, device=self.device)
                    pred = self.diff_net(xt, t_batch, y, None)
                    last_step = (t == self.num_steps - 1)
                    xt = self.rwd_diff(xt, pred, t_batch, dt, last_step=last_step)

            x = self.comp_net.decode(xt)
            samps = torch.clamp(x, min=self.norm_range[0], max=self.norm_range[1])
            if norm_output:
                samps = (samps - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])
            if save_imgs:
                os.makedirs(save_path, exist_ok=True)
                for t in range(samps.size(0)):
                    img_path = os.path.join(save_path, f"img_{t + 1}.png")
                    save_image(samps[t], img_path)
        return samps

    def to(self, device: torch.device) -> Self:
        """Moves the module and its components to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device for computation.

        Returns
        -------
        sample (SampleDDIM, SampleDDIM or SampleSDE) - The module moved to the specified device.
        """
        self.device = device
        self.diff_net.to(device)
        self.rwd_diff.to(device)
        self.comp_net.to(device)
        if self.cond_net:
            self.cond_net.to(device)
        return super().to(device)

###==================================================================================================================###

class AutoencoderLDM(nn.Module):
    """Variational autoencoder for latent space compression in Latent Diffusion Models.

    Encodes images into a latent space and decodes them back to the image space, used as
    the `compressor_model` in LDM’s `TrainLDM` and `SampleLDM`. Supports KL-divergence
    or vector quantization (VQ) regularization for the latent representation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB images).
    down_channels : list
        List of channel sizes for encoder downsampling blocks (e.g., [32, 64, 128, 256]).
    up_channels : list
        List of channel sizes for decoder upsampling blocks (e.g., [256, 128, 64, 16]).
    out_channels : int
        Number of output channels, typically equal to `in_channels`.
    dropout_rate : float
        Dropout rate for regularization in convolutional and attention layers.
    num_heads : int
        Number of attention heads in self-attention layers.
    num_groups : int
        Number of groups for group normalization in attention layers.
    num_layers_per_block : int
        Number of convolutional layers in each downsampling and upsampling block.
    total_down_sampling_factor : int
        Total downsampling factor across the encoder (e.g., 8 for 8x reduction).
    latent_channels : int
        Number of channels in the latent representation for diffusion models.
    num_embeddings : int
        Number of discrete embeddings in the VQ codebook (if `use_vq=True`).
    use_vq : bool, optional
        If True, uses vector quantization (VQ) regularization; otherwise, uses
        KL-divergence (default: False).
    beta : float, optional
        Weight for KL-divergence loss (if `use_vq=False`) (default: 1.0).
    use_flash: bool, optional
        if true and available flash attention is used to improve training efficiency (default: True)
    use_grad_check: bool, optional
        if true, gradient checkpoint is used (default: False)
    """
    def __init__(
            self,
            in_channels: int,
            down_channels: List[int],
            up_channels: List[int],
            out_channels: int,
            dropout_rate: float,
            num_heads: int,
            num_groups: int,
            num_layers_per_block: int,
            total_down_sampling_factor: int,
            latent_channels: int,
            num_embeddings: int,
            use_vq: bool = False,
            beta: float = 1.0,
            use_flash: bool = True,
            use_grad_check: bool = False,
            *args
    ) -> None:
        super().__init__()
        assert in_channels == out_channels, "Input and output channels must match for auto-encoding"
        self.use_vq = use_vq
        self.beta = beta
        self.current_beta = beta
        self.use_flash = use_flash
        self.use_grad_check = use_grad_check
        num_down_blocks = len(down_channels) - 1
        self.down_sampling_factor = int(total_down_sampling_factor ** (1 / num_down_blocks))

        # encoder
        self.conv1 = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels=down_channels[i],
                out_channels=down_channels[i + 1],
                num_layers=num_layers_per_block,
                down_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate,
                use_grad_check=self.use_grad_check
            ) for i in range(num_down_blocks)
        ])
        self.attention1 = Attention(down_channels[-1], num_heads, num_groups, dropout_rate, use_flash)

        # latent projection
        if use_vq:
            self.vq_layer = VectorQuantizer(num_embeddings, down_channels[-1])
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)
        else:
            self.conv_mu_logvar = nn.Conv2d(down_channels[-1], down_channels[-1] * 2, kernel_size=3, padding=1)
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)

        # decoder
        self.conv2 = nn.Conv2d(latent_channels, up_channels[0], kernel_size=3, padding=1)
        self.attention2 = Attention(up_channels[0], num_heads, num_groups, dropout_rate, use_flash)
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=up_channels[i],
                out_channels=up_channels[i + 1],
                num_layers=num_layers_per_block,
                up_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate,
                use_grad_check=use_grad_check
            ) for i in range(len(up_channels) - 1)
        ])
        self.conv3 = Conv3(up_channels[-1], out_channels, dropout_rate)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies reparameterization trick for variational autoencoding.

        Samples from a Gaussian distribution using the mean and log-variance to enable
        differentiable training.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution, shape (batch_size, channels, height, width).
        logvar : torch.Tensor
            Log-variance of the latent distribution, same shape as `mu`.

        Returns
        -------
        reparam (torch.Tensor) - Sampled latent representation, same shape as `mu`.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Encodes images into a latent representation.

        Processes input images through the encoder, applying convolutions, downsampling,
        self-attention, and latent projection (VQ or KL-based).

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (batch_size, in_channels, height, width).

        Returns
        -------
        z : (torch.Tensor)
            Latent representation, shape (batch_size, latent_channels, height/down_sampling_factor, width/down_sampling_factor).
        reg_loss : float
            Regularization loss (VQ loss if `use_vq=True`, KL-divergence loss if `use_vq=False`).

        **Notes**

        - The VQ loss is computed by `VectorQuantizer` if `use_vq=True`.
        - The KL-divergence loss is normalized by batch size and latent size, weighted
          by `current_beta`.
        """
        x = self.conv1(x)
        for block in self.down_blocks:
            x = block(x)

        if self.use_grad_check and self.training:
            x = x + checkpoint.checkpoint(self.attention1, x, use_reentrant=False)
        else:
            x = x + self.attention1(x)
        if self.use_vq:
            z, vq_loss = self.vq_layer(x)
            z = self.quant_conv(z)
            return z, vq_loss
        else:
            mu_logvar = self.conv_mu_logvar(x)
            mu, logvar = mu_logvar.chunk(2, dim=1)
            z = self.reparameterize(mu, logvar)
            z = self.quant_conv(z)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * self.current_beta
            return z, kl_loss

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent representations back to images.

        Processes latent representations through the decoder, applying convolutions,
        self-attention, upsampling, and final reconstruction.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation, shape (batch_size, latent_channels,
            height/down_sampling_factor, width/down_sampling_factor).

        Returns
        -------
        x (torch.Tensor) - Reconstructed images, shape (batch_size, out_channels, height, width).
        """
        x = self.conv2(z)
        if self.use_grad_check and self.training:
            x = x + checkpoint.checkpoint(self.attention2, x, use_reentrant=False)
        else:
            x = x + self.attention2(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv3(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float, torch.Tensor]:
        """Encodes images to latent space and decodes them, computing reconstruction and regularization losses.

        Performs a full autoencoding pass, encoding images to the latent space, decoding
        them back, and calculating MSE reconstruction loss and regularization loss (VQ
        or KL-based).

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (batch_size, in_channels, height, width).

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed images, shape (batch_size, out_channels, height, width).
        total_loss : float
            Sum of reconstruction (MSE) and regularization losses.
        reg_loss : float
            Regularization loss (VQ or KL-divergence).
        z : torch.Tensor
            Latent representation, shape (batch_size, latent_channels, height/down_sampling_factor, width/down_sampling_factor).

        **Notes**

        - The reconstruction loss is computed as the mean squared error between `x_hat` and `x`.
        - The regularization loss depends on `use_vq` (VQ loss or KL-divergence).
        """
        z, reg_loss = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_loss.item() + reg_loss
        return x_hat, total_loss, reg_loss, z



class VectorQuantizer(nn.Module):
    """Vector quantization layer for discretizing latent representations.

    Quantizes input latent vectors to the nearest embedding in a learned codebook,
    used in `AutoencoderLDM` when `use_vq=True` to enable discrete latent spaces for
    Latent Diffusion Models. Computes commitment and codebook losses to train the
    codebook embeddings.

    Parameters
    ----------
    num_embed : int
        Number of discrete embeddings in the codebook.
    embed_dim : int
        Dimensionality of each embedding vector (matches input channel dimension).
    commit_cost : float, optional
        Weight for the commitment loss, encouraging inputs to be close to quantized values (default: 0.25).

    **Notes**

    - The codebook embeddings are initialized uniformly in the range [-1/num_embeddings, 1/num_embeddings].
    - The forward pass flattens input latents, computes Euclidean distances to codebook embeddings, and selects the nearest embedding for quantization.
    - The commitment loss encourages input latents to be close to their quantized versions, while the codebook loss updates embeddings to match inputs.
    - A straight-through estimator is used to pass gradients from the quantized output to the input.
    """
    def __init__(self, num_embed: int, embed_dim: int, commit_cost: float = 0.25) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_embed = num_embed
        self.commit_cost = commit_cost
        self.embed = nn.Embedding(num_embed, embed_dim)
        self.embed.weight.data.uniform_(-1.0 / num_embed, 1.0 / num_embed)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantizes latent representations to the nearest codebook embedding.

        Computes the closest embedding for each input vector, applies quantization,
        and calculates commitment and codebook losses for training.

        Parameters
        ----------
        z : torch.Tensor
            Input latent representation, shape (batch_size, embedding_dim, height,
            width).

        Returns
        -------
        quantized : torch.Tensor
            Quantized latent representation, same shape as `z`.
        vq_loss : torch.Tensor
            Sum of commitment and codebook losses.

        **Notes**

        - The input is flattened to (batch_size * height * width, embedding_dim) for distance computation.
        - Euclidean distances are computed efficiently using vectorized operations.
        - The commitment loss is scaled by `commitment_cost`, and the total VQ loss combines commitment and codebook losses.
        """
        batch_size, channels, height, width = z.shape
        assert channels == self.embed_dim, f"Expected channel dim {self.embed_dim}, got {channels}"
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, self.embed_dim)
        z_sq = torch.sum(z_flat ** 2, dim=1, keepdim=True)
        e_sq = torch.sum(self.embed.weight ** 2, dim=1)
        dist = z_sq + e_sq - 2 * torch.matmul(z_flat, self.embed.weight.t())
        encode_idx = torch.argmin(dist, dim=1)
        quantized = self.embed(encode_idx).view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        commit_loss = self.commit_cost * F.mse_loss(z.detach(), quantized)
        codebook_loss = F.mse_loss(z, quantized.detach())
        quantized = z + (quantized - z).detach()
        return quantized, commit_loss + codebook_loss

class DownBlock(nn.Module):
    """Downsampling block for the encoder in AutoencoderLDM.

    Applies multiple convolutional layers with residual connections followed by
    downsampling to reduce spatial dimensions in the encoder of the variational
    autoencoder used in Latent Diffusion Models.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for convolutional layers.
    num_layers : int
        Number of convolutional layer pairs (Conv3) per block.
    down_sampling_factor : int
        Factor by which to downsample spatial dimensions.
    dropout_rate : float
        Dropout rate for Conv3 layers.
    use_grad_check: bool, optional
        if true, gradient checkpoint is used (default: False)

    **Notes**

    - Each layer pair consists of two Conv3 modules with a residual connection using a 1x1 convolution to match dimensions.
    - The downsampling is applied after all convolutional layers, reducing spatial dimensions by `down_sampling_factor`.
    """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
                 down_sampling_factor: int, dropout_rate: float, use_grad_check: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.use_grad_check = use_grad_check
        self.res_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResidualBlock(in_ch, out_channels, dropout_rate)
            )
        self.down_sampling = DownSampling(out_channels, out_channels, down_sampling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through convolutional layers and downsampling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        output (torch.Tensor) - Output tensor, shape (batch_size, out_channels, height/down_sampling_factor, width/down_sampling_factor).
        """
        for block in self.res_blocks:
            if self.use_grad_check and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.down_sampling(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block"""
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = Conv3(in_channels, out_channels, dropout_rate)
        self.conv2 = Conv3(out_channels, out_channels, dropout_rate)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through residual connection"""
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class Conv3(nn.Module):
    """Convolutional layer with group normalization, SiLU activation, and dropout.

    Used in DownBlock and UpBlock of AutoencoderLDM for feature extraction and
    transformation in the encoder and decoder.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : float
        Dropout rate for regularization.

    **Notes**

    - The layer applies group normalization, SiLU activation, dropout, and a 3x3 convolution in sequence.
    - Spatial dimensions are preserved due to padding=1 in the convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through group normalization, activation, dropout, and convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        x (torch.Tensor) - Output tensor, shape (batch_size, out_channels, height, width).
        """
        x = self.group_norm(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class DownSampling(nn.Module):
    """Downsampling module for reducing spatial dimensions in AutoencoderLDM’s encoder.

    Combines convolutional downsampling and max pooling, concatenating their outputs
    to preserve feature information during downsampling in DownBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (sum of conv and pool paths).
    down_sampling_factor : int
        Factor by which to downsample spatial dimensions.

    **Notes**

    - The module splits the output channels evenly between convolutional and pooling paths, concatenating them along the channel dimension.
    - The convolutional path uses a stride equal to `down_sampling_factor`, while the pooling path uses max pooling with the same factor.
    """
    def __init__(self, in_channels: int, out_channels: int, down_sampling_factor: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=down_sampling_factor,
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsamples input by combining convolutional and pooling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        x (torch.Tensor) - Downsampled tensor, shape (batch_size, out_channels, height/down_sampling_factor, width/down_sampling_factor).
        """
        return self.conv(x)


class Attention(nn.Module):
    """Self-attention module for feature enhancement in AutoencoderLDM.

    Applies multi-head self-attention to enhance features in the encoder and decoder,
    used after downsampling (in DownBlock) and before upsampling (in UpBlock).

    Parameters
    ----------
    num_channels : int
        Number of input and output channels (embedding dimension for attention).
    num_heads : int
        Number of attention heads.
    num_groups : int
        Number of groups for group normalization.
    dropout_rate : float
        Dropout rate for attention outputs.
    use_flash: bool, optional
        if true and available flash attention is used to improve training efficiency (default: True)

    **Notes**

    - The input is reshaped to (batch_size, height * width, num_channels) for attention processing, then restored to (batch_size, num_channels, height, width).
    - Group normalization is applied before attention to stabilize training.
    """
    def __init__(self, num_channels: int, num_heads: int, num_groups: int,
                 dropout_rate: float, use_flash: bool = True) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.use_flash = use_flash

        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

        if use_flash and hasattr(F, 'scaled_dot_product_attention'):
            self.qkv = nn.Linear(num_channels, num_channels * 3)
            self.proj = nn.Linear(num_channels, num_channels)
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=num_channels,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout_rate
            )
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies self-attention to input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, num_channels, height, width).

        Returns
        -------
        x (torch.Tensor) - Output tensor, same shape as input.
        """
        batch_size, channels, h, w = x.shape
        x_norm = self.group_norm(x)
        x_norm = x_norm.reshape(batch_size, channels, h * w).transpose(1, 2)
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            qkv = self.qkv(x_norm).reshape(batch_size, h * w, 3, self.num_heads, channels // self.num_heads)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, h * w, channels)
            x_attn = self.proj(attn_output)
        else:
            x_attn, _ = self.attention(x_norm, x_norm, x_norm)
        x_attn = self.dropout(x_attn)
        x_attn = x_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x_attn


class UpBlock(nn.Module):
    """Upsampling block for the decoder in AutoencoderLDM.

    Applies upsampling followed by multiple convolutional layers with residual
    connections to increase spatial dimensions in the decoder of the variational
    autoencoder used in Latent Diffusion Models.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for convolutional layers.
    num_layers : int
        Number of convolutional layer pairs (Conv3) per block.
    up_sampling_factor : int
        Factor by which to upsample spatial dimensions.
    dropout_rate : float
        Dropout rate for Conv3 layers.
    use_grad_check: bool, optional
        if true, gradient checkpoint is used (default: False)

    **Notes**

    - Upsampling is applied first, followed by convolutional layer pairs with residual connections using 1x1 convolutions.
    - Each layer pair consists of two Conv3 modules.
    """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
                 up_sampling_factor: int, dropout_rate: float, use_grad_check: bool = False) -> None:
        super().__init__()
        self.up_sampling = UpSampling(in_channels, in_channels, up_sampling_factor)
        self.use_grad_check = use_grad_check
        self.res_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResidualBlock(in_ch, out_channels, dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through upsampling and convolutional layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        output (torch.Tensor) - Output tensor, shape (batch_size, out_channels, height * up_sampling_factor, width * up_sampling_factor).
        """
        x = self.up_sampling(x)
        for block in self.res_blocks:
            if self.use_grad_check and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class UpSampling(nn.Module):
    """Upsampling module for increasing spatial dimensions in AutoencoderLDM’s decoder.

    Combines transposed convolution and nearest-neighbor upsampling, concatenating
    their outputs to preserve feature information during upsampling in UpBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (sum of conv and upsample paths).
    up_sampling_factor : int
        Factor by which to upsample spatial dimensions.

    **Notes**

    - The module splits the output channels evenly between transposed convolution and upsampling paths, concatenating them along the channel dimension.
    - If the spatial dimensions of the two paths differ, the upsampling path is interpolated to match the convolutional path’s size.
    """
    def __init__(self, in_channels: int, out_channels: int, up_sampling_factor: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=up_sampling_factor,
            padding=1,
            output_padding=up_sampling_factor - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsamples input by combining transposed convolution and upsampling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        x (torch.Tensor) - Upsampled tensor, shape
        (batch_size, out_channels, height * up_sampling_factor, width * up_sampling_factor).

        **Notes**

        - Interpolation is applied if the spatial dimensions of the
          convolutional and upsampling paths differ, using nearest-neighbor mode.
        """
        return self.conv(x)

###==================================================================================================================###

class TrainAE(nn.Module):
    """Trainer for the AutoencoderLDM variational autoencoder in Latent Diffusion Models.

    Optimizes the AutoencoderLDM model to compress images into latent space and reconstruct
    them, using reconstruction loss (MSE), regularization (KL or VQ), and optional
    perceptual loss (LPIPS). Supports mixed precision, KL warmup, early stopping, and
    learning rate scheduling, with evaluation metrics (MSE, PSNR, SSIM, FID, LPIPS).

    Parameters
    ----------
    model : nn.Module
        The variational autoencoder model (AutoencoderLDM) to train.
    optim : torch.optim.Optimizer
        Optimizer for training (e.g., Adam).
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data (default: None).
    max_epochs : int, optional
        Maximum number of training epochs (default: 100).
    metrics_ : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    device : str
        Device for computation (e.g., 'cuda', 'cpu').
    store_path : str, optional
        Path to save model checkpoints (default: 'vlc_model.pth').
    checkpoint : int, optional
        Frequency (in epochs) to save model checkpoints (default: 10).
    kl_warmup_epochs : int, optional
        Number of epochs for KL loss warmup (default: 10).
    patience : int, optional
        Number of epochs to wait for early stopping if validation loss doesn’t improve
        (default: 10).
    val_freq : int, optional
        Frequency (in epochs) for validation and metric computation (default: 5).
    warmup_steps: int, optional
        learinig rate warmup steps (default: 1000)
    use_ddp : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    grad_acc : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    log_freq : int, optional
        Number of epochs before printing loss.
    use_comp: bool, optional
        if true, model is compiled (default: False)
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optim: torch.optim.Optimizer,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 100,
            metrics_: Optional[Any] = None,
            device: str = 'cuda',
            store_path: str = "vlc_model",
            checkpoint: int = 10,
            kl_warmup_epochs: int = 10,
            patience: int = 10,
            val_freq: int = 5,
            warmup_steps: int = 1000,
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

        self.model = model.to(self.device)
        self.optim = optim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.metrics_ = metrics_  
        self.store_path = store_path
        self.checkpoint = checkpoint
        self.kl_warmup_epochs = kl_warmup_epochs
        self.patience = patience
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
        self.val_freq = val_freq
        self.log_freq = log_freq

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


    def load_checkpoint(self, checkpoint_path: str) -> Tuple[float, float]:
        """Loads a training checkpoint to resume training.

        Restores the state of the noise predictor, conditional model (if applicable),
        and optimizer from a saved checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.

        Returns
        -------
        epoch : float
            The epoch at which the checkpoint was saved (int).
        loss : float
            The loss at the checkpoint (float).
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict' key")
        state_dict = checkpoint['model_state_dict']
        if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
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
            Number of steps for the warm phase.

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
        """Wrap models with DistributedDataParallel for multi-GPU training"""
        if self.use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

    def forward(self) -> Dict:
        """Trains the AutoencoderLDM model with mixed precision and evaluation metrics.

        Performs training with reconstruction and regularization losses, KL warmup, gradient
        clipping, and learning rate scheduling. Saves checkpoints for the best validation
        loss and supports early stopping.

        Returns
        -------
        losses : dictionlary contains train and validation losses
        """
        if self.use_comp:
            try:
                self.model = torch.compile(self.model)
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
            if self.model.use_vq:
                beta = 1.0
            else:
                beta = min(1.0, epoch / self.kl_warmup_epochs) * self.model.beta
                self.model.current_beta = beta
            train_losses_epoch = []
            for step, (x, y) in enumerate(pbar):
                x = x.to(self.device)
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    x_hat, loss, reg_loss, z = self.model(x)
                    loss = loss / self.grad_acc

                scaler.scale(loss).backward()
                if (step + 1) % self.grad_acc == 0:
                    scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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

    def _save_checkpoint(self, epoch: int, loss: float, pref: str = "") -> None:
        """Save model checkpoint (only called by master process).

        Parameters
        ----------
        epoch : int
            Current epoch number.
        loss : float
            Current loss value.
        pref : str, optional
            Prefix to add to checkpoint filename.
        """
        try:
            model_state = (
                self.model.module.state_dict() if self.use_ddp else self.model.state_dict()
            )
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optim_state_dict': self.optim.state_dict(),
                'loss': loss,
                'losses': self.losses,
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
        """Validates the AutoencoderLDM model and computes evaluation Metrics.

        Computes validation loss and optional Metrics (MSE, PSNR, SSIM, FID, LPIPS) using
        the provided Metrics object.

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
        self.model.eval()
        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []
        with torch.no_grad():
            for x, _ in self.val_loader:
                x = x.to(self.device)
                x_hat, loss, reg_loss, z = self.model(x)
                val_losses.append(loss.item())
                if self.metrics_ is not None:
                    metrics_result = self.metrics_.forward(x, x_hat)
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
        self.model.train()
        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg