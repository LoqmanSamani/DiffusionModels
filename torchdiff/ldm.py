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


- The `varinace_scheduler` parameter expects an external hyperparameter module (e.g.,
  VarianceSchedulerDDPM, VarianceSchedulerSDE) as an nn.Module for noise schedule management.
- AutoencoderLDM serves as the `compressor_model` in TrainLDM and SampleLDM, providing
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
from typing import Optional, Tuple, Any, Callable, List, Union, Self
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
import warnings
from tqdm import tqdm
from torchvision.utils import save_image
import os



###==================================================================================================================###

class TrainLDM(nn.Module):
    """Trainer for the noise predictor in Latent Diffusion Models.

    Optimizes the noise predictor and conditional model (e.g., TextEncoder)
    to predict noise in the latent space of AutoencoderLDM, using a diffusion model (e.g., DDPM, DDIM, SDE).
    Supports mixed precision, conditional generation with text prompts, and evaluation metrics
    (MSE, PSNR, SSIM, FID, LPIPS) for generated images during validation, using a specified reverse
    diffusion model.

    Parameters
    ----------
    diffusion_model : str
        Diffusion model type ("ddpm", "ddim", "sde").
    forward_diffusion : ForwardDDPM, ForwardDDIM, or ForwardSDE
        Forward diffusion model defining the noise schedule.
    reverse_diffusion : ReverseDDPM, ReverseDDIM, or ReverseSDE
        Reverse diffusion model for sampling during validation (default: None).
    noise_predictor : torch.nn.Module
        Model to predict noise in the latent space (e.g., NoisePredictor).
    compressor_model : torch.nn.Module
        Variational autoencoder for encoding/decoding latents.
    optimizer : torch.optim.Optimizer
        Optimizer for the noise predictor and conditional model (e.g., Adam).
    objective : Callable
        Loss function for noise prediction (e.g., MSELoss).
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data (default: None).
    conditional_model : TextEncoder, optional
        Text encoder with projection layers for conditional generation (default: None).

    metrics_ : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    max_epochs : int, optional
        Maximum number of training epochs (default: 1000).
    device : str, optional
        Device for computation (e.g., 'cuda', 'cpu') (default: None).
    store_path : str, optional
        Path to save model checkpoints (default: None, uses 'ldm_model.pth').
    patience : int, optional
        Number of epochs to wait for early stopping if validation loss doesn’t improve
        (default: 100).
    warmup_epochs : int, optional
        Number of epochs for learning rate warmup (default: 100).
    bert_tokenizer : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    max_token_length : int, optional
        Maximum sequence length for tokenized text (default: 77).
    val_frequency : int, optional
        Frequency (in epochs) for validation and metric computation (default: 10).
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
            diffusion_model: str,
            forward_diffusion: torch.nn.Module,
            reverse_diffusion: torch.nn.Module,
            noise_predictor: torch.nn.Module,
            compressor_model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            data_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            conditional_model: Optional[torch.nn.Module] = None,
            metrics_: Optional[Any] = None,
            max_epochs: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            store_path: Optional[str] = None,
            patience: int = 100,
            warmup_epochs: int = 100,
            bert_tokenizer: Optional[BertTokenizer] = None,
            max_token_length: int = 77,
            val_frequency: int = 10,
            image_output_range: Tuple[float, float] = (-1.0, 1.0),
            normalize_output: bool = True,
            use_ddp: bool = False,
            grad_accumulation_steps: int = 1,
            log_frequency: int = 1,
            use_compilation: bool = False
    ) -> None:
        super().__init__()
        if diffusion_model not in ["ddpm", "ddim", "sde"]:
            raise ValueError(f"Unknown model: {diffusion_model}. Supported: ddpm, ddim, sde")
        self.diffusion_model = diffusion_model

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
        self.forward_diffusion = forward_diffusion.to(self.device)
        self.reverse_diffusion = reverse_diffusion.to(self.device)
        self.noise_predictor = noise_predictor.to(self.device)
        self.compressor_model = compressor_model.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None

        # Training components
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or "ldm_model"
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
            # If loading DDP checkpoint into non-DDP model, remove 'module.' prefix
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
        """Trains the noise predictor and conditional model with mixed precision and evaluation metrics.

        Optimizes the noise predictor and conditional model (e.g., TextEncoder with projection layers)
        using the forward diffusion model’s noise schedule, with text conditioning. Performs validation
        with image-domain metrics (MSE, PSNR, SSIM, FID, LPIPS) using the reverse diffusion model,
        saves checkpoints for the best validation loss, and supports early stopping.

        Returns
        -------
        train_losses : List of float
            List of mean training losses per epoch.
        best_val_loss : float
            Best validation loss achieved (or best training loss if no validation).
        """
        # set models to training mode
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()
        self.compressor_model.eval()  # pre-trained compressor model
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
                self.compressor_model = torch.compile(self.compressor_model)
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

                with torch.no_grad():
                    x, _ = self.compressor_model.encode(x)

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

            filename = f"ldm_epoch_{epoch}{suffix}.pth"
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

        num_steps = self.forward_diffusion.variance_scheduler.tau_num_steps if self.diffusion_model == "ddim" else self.forward_diffusion.variance_scheduler.num_steps

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                x_orig = x.clone()
                x, _ = self.compressor_model.encode(x)

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
                    for t in reversed(range(num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device)#, dtype=torch.long)
                        prev_time_steps = torch.full((xt.shape[0],), max(t - 1, 0), device=self.device)#, dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, y_encoded, None)

                        if self.diffusion_model == "sde":
                            noise = torch.randn_like(xt) if getattr(self.reverse_diffusion, "sde_method", None) != "ode" else None
                            xt = self.reverse_diffusion(xt, noise, predicted_noise, time_steps)
                        elif self.diffusion_model == "ddim":
                            xt, _ = self.reverse_diffusion(xt, predicted_noise, time_steps, prev_time_steps)
                        elif self.diffusion_model == "ddpm":
                            xt = self.reverse_diffusion(xt, predicted_noise, time_steps)
                        else:
                            raise ValueError(f"Unknown model: {self.diffusion_model}. Supported: ddpm, ddim, sde")

                    x_hat = self.compressor_model.decode(xt)

                    # clamp and normalize generated samples
                    x_hat = torch.clamp(x_hat, min=self.image_output_range[0], max=self.image_output_range[1])
                    if self.normalize_output:
                        x_hat = (x_hat - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])
                        x_orig = (x_orig - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])

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


class SampleLDM(nn.Module):
    """Sampler for generating images using Latent Diffusion Models (LDM).

    Generates images by iteratively denoising random noise in the latent space using a
    reverse diffusion process, decoding the result back to the image space with a
    pre-trained compressor, as described in Rombach et al. (2022). Supports DDPM, DDIM,
    and SDE diffusion models, as well as conditional generation with text prompts.

    Parameters
    ----------
    diffusion_model : str
        Diffusion model type. Supported: "ddpm", "ddim", "sde".
    reverse_diffusion : nn.Module
        Reverse diffusion module (e.g., ReverseDDPM, ReverseDDIM, ReverseSDE).
    noise_predictor : nn.Module
        Model to predict noise added during the forward diffusion process.
    compressor_model : nn.Module
        Pre-trained model to encode/decode between image and latent spaces (e.g., AutoencoderLDM).
    image_shape : tuple
        Shape of generated images as (height, width).
    conditional_model : nn.Module, optional
        Model for conditional generation (e.g., TextEncoder), default None.
    bert_tokenizer : str or BertTokenizer, optional
        Tokenizer for processing text prompts, default "bert-base-uncased".
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for latent representations (default: 3).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    image_output_range : tuple, optional
        Range for clamping generated images (min, max), default (-1, 1).
    """
    def __init__(
            self,
            diffusion_model: str,
            reverse_diffusion: torch.nn.Module,
            noise_predictor: torch.nn.Module,
            compressor_model: torch.nn.Module,
            image_shape: Tuple[float, float],
            conditional_model: Optional[torch.nn.Module] = None,
            bert_tokenizer: str = "bert-base-uncased",
            batch_size: int = 1,
            in_channels: int = 3,
            device: Optional[Union[str, torch.device]] = None,
            max_token_length: int = 77,
            image_output_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> None:
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.diffusion_model = diffusion_model
        self.noise_predictor = noise_predictor.to(self.device)
        self.reverse = reverse_diffusion.to(self.device)
        self.compressor = compressor_model.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.in_channels = in_channels
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.image_output_range = image_output_range

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not isinstance(image_output_range, (tuple, list)) or len(image_output_range) != 2 or image_output_range[0] >= image_output_range[1]:
            raise ValueError("output_range must be a tuple (min, max) with min < max")

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
            conditions: Optional[Union[List, str]] = None,
            normalize_output: bool = True,
            save_images: bool = True,
            save_path: str = "ldm_generated"
    ) -> torch.Tensor:
        """Generates images using the reverse diffusion process in the latent space.

        Iteratively denoises random noise in the latent space using the specified reverse
        diffusion model (DDPM, DDIM, SDE), then decodes the result to the image space
        with the compressor model. Supports conditional generation with text prompts.

        Parameters
        ----------
        conditions : str or list, optional
            Text prompt(s) for conditional generation, default None.
        normalize_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        save_images : bool, optional
            If True, saves generated images to `save_path` (default: True).
        save_path : str, optional
            Directory to save generated images (default: "ldm_generated").

        Returns
        -------
        generated_imgs (torch.Tensor) - Generated images, shape (batch_size, channels, height, width). If `normalize_output` is True, images are normalized to [0, 1]; otherwise, they are clamped to `output_range`.
        """
        if conditions is not None and self.conditional_model is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conditions is None and self.conditional_model is not None:
            raise ValueError("Conditions must be provided for conditional model")

        noisy_samples = torch.randn(self.batch_size, self.in_channels, self.image_shape[0], self.image_shape[1]).to(self.device)

        self.noise_predictor.eval()
        self.compressor.eval()
        self.reverse.eval()
        if self.conditional_model:
            self.conditional_model.eval()

        with torch.no_grad():
            xt = noisy_samples
            xt, _ = self.compressor.encode(xt)

            if self.diffusion_model == "ddim":
                num_steps = self.reverse.variance_scheduler.tau_num_steps
            elif self.diffusion_model == "ddpm" or self.diffusion_model == "sde":
                num_steps = self.reverse.variance_scheduler.num_steps
            else:
                raise ValueError(f"Unknown model: {self.diffusion_model}. Supported: ddpm, ddim, sde")

            for t in reversed(range(num_steps)):
                time_steps = torch.full((self.batch_size,), t, device=self.device)#, dtype=torch.long)
                prev_time_steps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)#, dtype=torch.long)

                if self.diffusion_model == "sde":
                    noise = torch.randn_like(xt) if getattr(self.reverse, "sde_method", None) != "ode" else None

                if self.conditional_model is not None and conditions is not None:
                    input_ids, attention_masks = self.tokenize(conditions)
                    key_padding_mask = (attention_masks == 0)
                    y = self.conditional_model(input_ids, key_padding_mask)
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)

                if self.diffusion_model == "sde":
                    xt = self.reverse(xt, noise, predicted_noise, time_steps)
                elif self.diffusion_model == "ddim":
                    xt, _ = self.reverse(xt, predicted_noise, time_steps, prev_time_steps)
                elif self.diffusion_model == "ddpm":
                    xt = self.reverse(xt, predicted_noise, time_steps)
                else:
                    raise ValueError(f"Unknown model: {self.diffusion_model}. Supported: ddpm, ddim, sde")

            x = self.compressor.decode(xt)
            generated_imgs = torch.clamp(x, min=self.image_output_range[0], max=self.image_output_range[1])
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

        Parameters
        ----------
        device : torch.device
            Target device for computation.

        Returns
        -------
        sample (SampleDDIM, SampleDDIM or SampleSDE) - The module moved to the specified device.
        """
        self.device = device
        self.noise_predictor.to(device)
        self.reverse.to(device)
        self.compressor.to(device)
        if self.conditional_model:
            self.conditional_model.to(device)
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
            beta: float = 1.0
    ) -> None:
        super().__init__()
        assert in_channels == out_channels, "Input and output channels must match for auto-encoding"
        self.use_vq = use_vq
        self.beta = beta
        self.current_beta = beta
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
                dropout_rate=dropout_rate
            ) for i in range(num_down_blocks)
        ])
        self.attention1 = Attention(down_channels[-1], num_heads, num_groups, dropout_rate)

        # latent projection
        if use_vq:
            self.vq_layer = VectorQuantizer(num_embeddings, down_channels[-1])
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)
        else:
            self.conv_mu = nn.Conv2d(down_channels[-1], down_channels[-1], kernel_size=3, padding=1)
            self.conv_logvar = nn.Conv2d(down_channels[-1], down_channels[-1], kernel_size=3, padding=1)
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)

        # decoder
        self.conv2 = nn.Conv2d(latent_channels, up_channels[0], kernel_size=3, padding=1)
        self.attention2 = Attention(up_channels[0], num_heads, num_groups, dropout_rate)
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=up_channels[i],
                out_channels=up_channels[i + 1],
                num_layers=num_layers_per_block,
                up_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate
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
        res_x = x
        x = self.attention1(x)
        x = x + res_x
        if self.use_vq:
            z, vq_loss = self.vq_layer(x)
            z = self.quant_conv(z)
            return z, vq_loss
        else:
            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            z = self.reparameterize(mu, logvar)
            z = self.quant_conv(z)
            kl_unnormalized = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            batch_size = x.size(0)
            latent_size = torch.prod(torch.tensor(mu.shape[1:])).item()
            kl_loss = kl_unnormalized / (batch_size * latent_size) * self.current_beta
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
        res_x = x
        x = self.attention2(x)
        x = x + res_x
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

###==================================================================================================================###

class VectorQuantizer(nn.Module):
    """Vector quantization layer for discretizing latent representations.

    Quantizes input latent vectors to the nearest embedding in a learned codebook,
    used in `AutoencoderLDM` when `use_vq=True` to enable discrete latent spaces for
    Latent Diffusion Models. Computes commitment and codebook losses to train the
    codebook embeddings.

    Parameters
    ----------
    num_embeddings : int
        Number of discrete embeddings in the codebook.
    embedding_dim : int
        Dimensionality of each embedding vector (matches input channel dimension).
    commitment_cost : float, optional
        Weight for the commitment loss, encouraging inputs to be close to quantized values (default: 0.25).


    **Notes**

    - The codebook embeddings are initialized uniformly in the range [-1/num_embeddings, 1/num_embeddings].
    - The forward pass flattens input latents, computes Euclidean distances to codebook embeddings, and selects the nearest embedding for quantization.
    - The commitment loss encourages input latents to be close to their quantized versions, while the codebook loss updates embeddings to match inputs.
    - A straight-through estimator is used to pass gradients from the quantized output to the input.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

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
        z = z.contiguous()
        assert z.size(1) == self.embedding_dim, f"Expected channel dim {self.embedding_dim}, got {z.size(1)}"
        z_flattened = z.reshape(-1, self.embedding_dim)
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float().squeeze(1)
        quantized = torch.matmul(encodings, self.embedding.weight).view_as(z)
        commitment_loss = self.commitment_cost * torch.mean((z.detach() - quantized) ** 2)
        codebook_loss = torch.mean((z - quantized.detach()) ** 2)
        quantized = z + (quantized - z).detach()
        return quantized, commitment_loss + codebook_loss

###==================================================================================================================###

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

    **Notes**

    - Each layer pair consists of two Conv3 modules with a residual connection using a 1x1 convolution to match dimensions.
    - The downsampling is applied after all convolutional layers, reducing spatial dimensions by `down_sampling_factor`.
    """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, down_sampling_factor: int, dropout_rate: float) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])

        self.down_sampling = DownSampling(
            in_channels=out_channels,
            out_channels=out_channels,
            down_sampling_factor=down_sampling_factor
        )
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers)

        ])

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
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
        output = self.down_sampling(output)
        return output

###==================================================================================================================###

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
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.activation = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)

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
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

###==================================================================================================================###

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
        self.down_sampling_factor = down_sampling_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                      kernel_size=3, stride=down_sampling_factor, padding=1)
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=down_sampling_factor, stride=down_sampling_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                      kernel_size=1, stride=1, padding=0)
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Downsamples input by combining convolutional and pooling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        x (torch.Tensor) - Downsampled tensor, shape (batch_size, out_channels, height/down_sampling_factor, width/down_sampling_factor).
        """
        return torch.cat(tensors=[self.conv(batch), self.pool(batch)], dim=1)

###==================================================================================================================###

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

    **Notes**

    - The input is reshaped to (batch_size, height * width, num_channels) for attention processing, then restored to (batch_size, num_channels, height, width).
    - Group normalization is applied before attention to stabilize training.
    """
    def __init__(self, num_channels: int, num_heads: int, num_groups: int, dropout_rate: float) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)

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
        x = x.reshape(batch_size, channels, h * w)
        x = self.group_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x

###==================================================================================================================###

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

    **Notes**

    - Upsampling is applied first, followed by convolutional layer pairs with residual connections using 1x1 convolutions.
    - Each layer pair consists of two Conv3 modules.
    """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, up_sampling_factor: int, dropout_rate: float) -> None:
        super().__init__()
        self.num_layers = num_layers
        effective_in_channels = in_channels

        self.up_sampling = UpSampling(
            in_channels=in_channels,
            out_channels=in_channels,
            up_sampling_factor=up_sampling_factor
        )

        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=effective_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=effective_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(self.num_layers)
        ])

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
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
        return output

###==================================================================================================================###

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
        half_out_channels = out_channels // 2
        self.up_sampling_factor = up_sampling_factor
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=half_out_channels,
                kernel_size=3,
                stride=up_sampling_factor,
                padding=1,
                output_padding=up_sampling_factor - 1
            ),
            nn.Conv2d(
                in_channels=half_out_channels,
                out_channels=half_out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=up_sampling_factor, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=half_out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Upsamples input by combining transposed convolution and upsampling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        x (torch.Tensor) - Upsampled tensor, shape (batch_size, out_channels, height * up_sampling_factor, width * up_sampling_factor).

        **Notes**

        - Interpolation is applied if the spatial dimensions of the convolutional and upsampling paths differ, using nearest-neighbor mode.
        """
        conv_output = self.conv(batch)
        up_sample_output = self.up_sample(batch)
        if conv_output.shape[2:] != up_sample_output.shape[2:]:
            _, _, h, w = conv_output.shape
            up_sample_output = torch.nn.functional.interpolate(
                up_sample_output,
                size=(h, w),
                mode='nearest'
            )
        return torch.cat(tensors=[conv_output, up_sample_output], dim=1)

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
    optimizer : torch.optim.Optimizer
        Optimizer for training (e.g., Adam).
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data (default: None).
    max_epochs : int, optional
        Maximum number of training epochs (default: 100).
    metrics_ : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    device : None, optional
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
    val_frequency : int, optional
        Frequency (in epochs) for validation and metric computation (default: 5).
    use_ddp : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    grad_accumulation_steps : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    log_frequency : int, optional
        Number of epochs before printing loss.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            data_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 100,
            metrics_: Optional[Any] = None,
            device: Optional[Union[str, torch.device]] = None,
            store_path: str = "vlc_model",
            checkpoint: int = 10,
            kl_warmup_epochs: int = 10,
            patience: int = 10,
            val_frequency: int = 5,
            warmup_epochs: int = 100,
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

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.metrics_ = metrics_  
        self.store_path = store_path
        self.checkpoint = checkpoint
        self.kl_warmup_epochs = kl_warmup_epochs
        self.patience = patience
        self.use_compilation = use_compilation

        # Learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)
        self.val_frequency = val_frequency
        self.log_frequency = log_frequency

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
            # load checkpoint with proper device mapping
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")


        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict' key")

        # Handle DDP wrapped model state dict
        state_dict = checkpoint['model_state_dict']
        if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
            # if loading non-DDP checkpoint into DDP model, add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
            # if loading DDP checkpoint into non-DDP model, remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

        if 'optimizer_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'optimizer_state_dict' key")
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            warnings.warn(f"Optimizer state loading failed: {e}. Continuing without optimizer state.")

        epoch = checkpoint.get('epoch', -1)
        loss = checkpoint.get('loss', float('inf'))

        self.noise_predictor.to(self.device)
        if self.conditional_model is not None:
            self.conditional_model.to(self.device)

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


    def forward(self) -> Tuple[List[float], float]:
        """Trains the AutoencoderLDM model with mixed precision and evaluation metrics.

        Performs training with reconstruction and regularization losses, KL warmup, gradient
        clipping, and learning rate scheduling. Saves checkpoints for the best validation
        loss and supports early stopping.

        Returns
        -------
        train_losses : list
            List of mean training losses per epoch.
        best_val_loss :  float
            Best validation loss achieved (or best training loss if no validation).
        """
        # compile models for optimization (if supported)
        if self.use_compilation:
            try:
                self.model = torch.compile(self.model)
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

            if self.model.use_vq:
                beta = 1.0  # no warmup for VQ
            else:
                beta = min(1.0, epoch / self.kl_warmup_epochs) * self.model.beta
                self.model.current_beta = beta

            train_losses_epoch = []

            # training step loop with gradient accumulation
            for step, (x, y) in enumerate(tqdm(self.data_loader, disable=not self.master_process)):
                x = x.to(self.device)

                # forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    x_hat, loss, reg_loss, z = self.model(x)
                    # compute loss and scale for gradient accumulation
                    loss = loss / self.grad_accumulation_steps

                # backward pass
                scaler.scale(loss).backward()

                # gradient accumulation and optimizer step
                if (step + 1) % self.grad_accumulation_steps == 0:
                    # clip gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
            model_state = (
                self.model.module.state_dict() if self.use_ddp else self.model.state_dict()
            )

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'max_epochs': self.max_epochs,
            }

            filename = f"ldm_epoch_{epoch}{suffix}.pth"
            filepath = os.path.join(self.store_path, filename)
            os.makedirs(self.store_path, exist_ok=True)
            torch.save(checkpoint, filepath)
            print(f"Model saved at epoch {epoch}")
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

                # compute metrics
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

        self.model.train()

        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg