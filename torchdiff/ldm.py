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


- The `hyper_params` parameter expects an external hyperparameter module (e.g.,
  HyperParamsDDPM, HyperParamsSDE) as an nn.Module for noise schedule management.
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
# multi-GPU processor module
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
import warnings
from tqdm import tqdm
from torchvision.utils import save_image



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
    model : str
        Diffusion model type ("ddpm", "ddim", "sde").
    forward_diffusion : ForwardDDPM, ForwardDDIM, or ForwardSDE
        Forward diffusion model defining the noise schedule.
    hyper_params : HyperParamsDDPM, HyperParamsDDIM, or HyperParamsSDE
        Hyperparameters for the diffusion process (nn.Module).
    noise_predictor : NoisePredictor
        Model to predict noise in the latent space (e.g., NoisePredictor).
    compressor_model : AutoencoderLDM
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
    reverse_diffusion : ReverseDDPM, ReverseDDIM, or ReverseSDE, optional
        Reverse diffusion model for sampling during validation (default: None).
    metrics_ : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    max_epoch : int, optional
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
    max_length : int, optional
        Maximum sequence length for tokenized text (default: 77).
    val_frequency : int, optional
        Frequency (in epochs) for validation and metric computation (default: 10).
    output_range : tuple, optional
        Range for clamping generated images (default: (-1, 1)).
    normalize_output : bool, optional
        Whether to normalize generated images to [0, 1] for metrics (default: True).
    ddp : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    num_grad_accumulation : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    progress_frequency : int, optional
        Number of epochs before printing loss.
    """

    def __init__(
            self,
            model: str,
            forward_diffusion: torch.nn.Module,
            hyper_params: torch.nn.Module,
            noise_predictor: torch.nn.Module,
            compressor_model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            data_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            conditional_model: Optional[torch.nn.Module] = None,
            reverse_diffusion: Optional[torch.nn.Module] = None,
            metrics_: Optional[Any] = None,
            max_epoch: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            store_path: Optional[str] = None,
            patience: int = 100,
            warmup_epochs: int = 100,
            tokenizer: Optional[BertTokenizer] = None,
            max_length: int = 77,
            val_frequency: int = 10,
            output_range: Tuple[float, float] = (-1.0, 1.0),
            normalize_output: bool = True,
            ddp: bool = False,
            num_grad_accumulation: int = 1,
            progress_frequency: int = 1
    ) -> None:
        super().__init__()
        if model not in ["ddpm", "ddim", "sde"]:
            raise ValueError(f"Unknown model: {model}. Supported: ddpm, ddim, sde")
        self.model = model

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
        self.forward_diffusion = forward_diffusion.to(self.device)
        self.hyper_params = hyper_params.to(self.device)
        self.noise_predictor = noise_predictor.to(self.device)
        self.compressor_model = compressor_model.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.reverse_diffusion = reverse_diffusion.to(self.device) if reverse_diffusion else None

        # Training components
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or "ldm_model"
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.max_length = max_length
        self.patience = patience
        self.val_frequency = val_frequency
        self.output_range = output_range
        self.normalize_output = normalize_output
        self.progress_frequency = progress_frequency

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
        else:
            self.tokenizer = tokenizer

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
        # Set models to training mode
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()
        self.compressor_model.eval()  # pre-trained, not trained here

        # Compile models for optimization (if supported)
        """
        try:
            self.noise_predictor = torch.compile(self.noise_predictor)
            if self.conditional_model is not None:
                self.conditional_model = torch.compile(self.conditional_model)
            self.compressor_model = torch.compile(self.compressor_model)
        except Exception as e:
            if self.master_process:
                print(f"Model compilation failed: {e}. Continuing without compilation.")
        """

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

                with torch.no_grad():
                    x, _ = self.compressor_model.encode(x)

                # Process conditional inputs if conditional model exists
                if self.conditional_model is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None

                # Forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    # Generate noise and timesteps
                    noise = torch.randn_like(x).to(self.device)
                    t = torch.randint(0, self.hyper_params.num_steps, (x.shape[0],)).to(self.device)

                    # Apply forward diffusion
                    noisy_x = self.forward_diffusion(x, noise, t)

                    # Predict noise
                    predicted_noise = self.noise_predictor(noisy_x, t, y_encoded)

                    # Compute loss and scale for gradient accumulation
                    loss = self.objective(predicted_noise, noise) / self.num_grad_accumulation

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
                'hyper_params_model': (
                    self.hyper_params.state_dict() if isinstance(self.hyper_params, nn.Module)
                    else self.hyper_params
                ),
                'max_epoch': self.max_epoch,
            }

            save_path = self.store_path + suffix + ".pth" if suffix else self.store_path + ".pth"
            torch.save(checkpoint, save_path)
            print(f"Model saved at epoch {epoch} to {save_path}")

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

        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []

        num_steps = self.hyper_params.tau_num_steps if self.model == "ddim" else self.hyper_params.num_steps

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                x_orig = x.clone()
                x, _ = self.compressor_model.encode(x)

                # Process conditional input
                if self.conditional_model is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None

                # Compute validation loss
                noise = torch.randn_like(x).to(self.device)
                t = torch.randint(0, self.hyper_params.num_steps, (x.shape[0],)).to(self.device)

                noisy_x = self.forward_diffusion(x, noise, t)
                predicted_noise = self.noise_predictor(noisy_x, t, y_encoded)
                loss = self.objective(predicted_noise, noise)
                val_losses.append(loss.item())

                # Generate samples for metrics evaluation
                if self.metrics_ is not None and self.reverse_diffusion is not None:
                    xt = torch.randn_like(x).to(self.device)

                    # Reverse diffusion sampling
                    for t in reversed(range(num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        prev_time_steps = torch.full((xt.shape[0],), max(t - 1, 0), device=self.device, dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, y_encoded)

                        if self.model == "sde":
                            noise = torch.randn_like(xt) if getattr(self.reverse_diffusion, "method", None) != "ode" else None
                            xt = self.reverse_diffusion(xt, noise, predicted_noise, time_steps)
                        elif self.model == "ddim":
                            xt, _ = self.reverse_diffusion(xt, predicted_noise, time_steps, prev_time_steps)
                        elif self.model == "ddpm":
                            xt = self.reverse_diffusion(xt, predicted_noise, time_steps)
                        else:
                            raise ValueError(f"Unknown model: {self.model}. Supported: ddpm, ddim, sde")

                    x_hat = self.compressor_model.decode(xt)

                    # Clamp and normalize generated samples
                    x_hat = torch.clamp(x_hat, min=self.output_range[0], max=self.output_range[1])
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


class SampleLDM(nn.Module):
    """Sampler for generating images using Latent Diffusion Models (LDM).

    Generates images by iteratively denoising random noise in the latent space using a
    reverse diffusion process, decoding the result back to the image space with a
    pre-trained compressor, as described in Rombach et al. (2022). Supports DDPM, DDIM,
    and SDE diffusion models, as well as conditional generation with text prompts.

    Parameters
    ----------
    model : str
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
    tokenizer : str or BertTokenizer, optional
        Tokenizer for processing text prompts, default "bert-base-uncased".
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for latent representations (default: 3).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    max_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    output_range : tuple, optional
        Range for clamping generated images (min, max), default (-1, 1).
    """
    def __init__(
            self,
            model: str,
            reverse_diffusion: torch.nn.Module,
            noise_predictor: torch.nn.Module,
            compressor_model: torch.nn.Module,
            image_shape: Tuple[float, float],
            conditional_model: Optional[torch.nn.Module] = None,
            tokenizer: str = "bert-base-uncased",
            batch_size: int = 1,
            in_channels: int = 3,
            device: Optional[Union[str, torch.device]] = None,
            max_length: int = 77,
            output_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.noise_predictor = noise_predictor.to(self.device)
        self.reverse = reverse_diffusion.to(self.device)
        self.compressor = compressor_model.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.in_channels = in_channels
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_range = output_range

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not isinstance(output_range, (tuple, list)) or len(output_range) != 2 or output_range[0] >= output_range[1]:
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
            max_length=self.max_length,
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

            if self.model == "ddim":
                num_steps = self.reverse.hyper_params.tau_num_steps
            elif self.model == "ddpm" or self.model == "sde":
                num_steps = self.reverse.hyper_params.num_steps
            else:
                raise ValueError(f"Unknown model: {self.model}. Supported: ddpm, ddim, sde")

            for t in reversed(range(num_steps)):
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)
                prev_time_steps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device, dtype=torch.long)

                if self.model == "sde":
                    noise = torch.randn_like(xt) if getattr(self.reverse, "method", None) != "ode" else None

                if self.conditional_model is not None and conditions is not None:
                    input_ids, attention_masks = self.tokenize(conditions)
                    key_padding_mask = (attention_masks == 0)
                    y = self.conditional_model(input_ids, key_padding_mask)
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)

                if self.model == "sde":
                    xt = self.reverse(xt, noise, predicted_noise, time_steps)
                elif self.model == "ddim":
                    xt, _ = self.reverse(xt, predicted_noise, time_steps, prev_time_steps)
                elif self.model == "ddpm":
                    xt = self.reverse(xt, predicted_noise, time_steps)
                else:
                    raise ValueError(f"Unknown model: {self.model}. Supported: ddpm, ddim, sde")

            x = self.compressor.decode(xt)
            generated_imgs = torch.clamp(x, min=self.output_range[0], max=self.output_range[1])
            if normalize_output:
                generated_imgs = (generated_imgs - self.output_range[0]) / (self.output_range[1] - self.output_range[0])

            # save images if save_images is True
            if save_images:
                os.makedirs(save_path, exist_ok=True)
                for i in range(generated_imgs.size(0)):
                    img_path = os.path.join(save_path, f"image_{i}.png")
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
        total_loss = recon_loss + reg_loss
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
    model : AutoencoderLDM
        The variational autoencoder model (AutoencoderLDM) to train.
    optimizer : torch.optim.Optimizer
        Optimizer for training (e.g., Adam).
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data (default: None).
    max_epoch : int, optional
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
    ddp : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    num_grad_accumulation : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    progress_frequency : int, optional
        Number of epochs before printing loss.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            data_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epoch: int = 100,
            metrics_: Optional[Any] = None,
            device: Optional[Union[str, torch.device]] = None,
            store_path: str = "vlc_model",
            checkpoint: int = 10,
            kl_warmup_epochs: int = 10,
            patience: int = 10,
            val_frequency: int = 5,
            warmup_epochs: int = 100,
            ddp: bool = False,
            num_grad_accumulation: int = 1,
            progress_frequency: int = 1
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

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.metrics_ = metrics_  
        self.store_path = store_path
        self.checkpoint = checkpoint
        self.kl_warmup_epochs = kl_warmup_epochs
        self.patience = patience

        # Learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)
        self.val_frequency = val_frequency
        self.progress_frequency = progress_frequency

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
            # Load checkpoint with proper device mapping
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")


        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict' key")

        # Handle DDP wrapped model state dict
        state_dict = checkpoint['model_state_dict']
        if self.ddp and not any(key.startswith('module.') for key in state_dict.keys()):
            # If loading non-DDP checkpoint into DDP model, add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not self.ddp and any(key.startswith('module.') for key in state_dict.keys()):
            # If loading DDP checkpoint into non-DDP model, remove 'module.' prefix
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

        # Compile models for optimization (if supported)
        """
        try:
            self.model = torch.compile(self.model)
        except Exception as e:
            if self.master_process:
                print(f"Model compilation failed: {e}. Continuing without compilation.")
        """

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

            if self.model.use_vq:
                beta = 1.0  # no warmup for VQ
            else:
                beta = min(1.0, epoch / self.kl_warmup_epochs) * self.model.beta
                self.model.current_beta = beta

            train_losses_epoch = []

            # Training step loop with gradient accumulation
            for step, (x, y) in enumerate(tqdm(self.data_loader, disable=not self.master_process)):
                x = x.to(self.device)

                # Forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    x_hat, loss, reg_loss, z = self.model(x)
                    # Compute loss and scale for gradient accumulation
                    loss = loss / self.num_grad_accumulation

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient accumulation and optimizer step
                if (step + 1) % self.num_grad_accumulation == 0:
                    # Clip gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
            model_state = (
                self.model.module.state_dict() if self.ddp else self.model.state_dict()
            )

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'max_epoch': self.max_epoch,
            }

            save_path = self.store_path + suffix + ".pth" if suffix else self.store_path + ".pth"
            torch.save(checkpoint, save_path)
            print(f"Model saved at epoch {epoch} to {save_path}")

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

                # Compute metrics
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

        self.model.train()

        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg


#########################################################################################################################

class ForwardSDE(nn.Module):
    """Forward diffusion process for SDE-based generative models.

    Implements the forward diffusion process for score-based generative models using
    Stochastic Differential Equations (SDEs), supporting Variance Exploding (VE),
    Variance Preserving (VP), sub-Variance Preserving (sub-VP), and ODE methods, as
    described in Song et al. (2021).

    Parameters
    ----------
    hyper_params : object
        Hyperparameter object (HyperParamsSDE) containing SDE-specific parameters. Expected to have
        attributes: `dt`, `sigmas`, `betas`, `cum_betas`.
    method : str
        SDE method to use. Supported methods: "ve", "vp", "sub-vp", "ode".
    """
    def __init__(self, hyper_params: torch.nn.Module, method: str) -> None:
        super().__init__()
        self.hyper_params = hyper_params
        self.method = method

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Applies the forward SDE diffusion process to the input data.

        Perturbs the input data `x0` by adding noise according to the specified SDE
        method at given time steps, incorporating drift and diffusion terms as applicable.

        Parameters
        ----------
        x0 : torch.Tensor
            Input data tensor, shape (batch_size, channels, height, width).
        noise : torch.Tensor
            Gaussian noise tensor, same shape as `x0`.
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, hyper_params.num_steps - 1].

        Returns
        -------
        xt (torch.Tensor) - Noisy data tensor at the specified time steps, same shape as `x0`.

        """
        dt = self.hyper_params.dt
        if self.method == "ve":
            # Use property to get sigmas (handles trainable case)
            sigma_t = self.hyper_params.sigmas[time_steps]
            sigma_t_prev = self.hyper_params.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
            sigma_diff = torch.sqrt(torch.clamp(sigma_t ** 2 - sigma_t_prev ** 2, min=0))
            x0 = x0 + noise * sigma_diff.view(-1, 1, 1, 1)

        elif self.method == "vp":
            # Use property to get betas (handles trainable case)
            betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            diffusion = torch.sqrt(betas * dt) * noise
            x0 = x0 + drift + diffusion

        elif self.method == "sub-vp":
            # Use properties to get betas and cum_betas (handles trainable case)
            betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
            cum_betas = self.hyper_params._cum_betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            diffusion = torch.sqrt(betas * (1 - torch.exp(-2 * cum_betas)) * dt) * noise
            x0 = x0 + drift + diffusion

        elif self.method == "ode":
            # Use property to get betas (handles trainable case)
            betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            x0 = x0 + drift
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return x0

###==================================================================================================================###

class ReverseSDE(nn.Module):
    """Reverse diffusion process for SDE-based generative models.

    Implements the reverse diffusion process for score-based generative models using
    Stochastic Differential Equations (SDEs), supporting Variance Exploding (VE),
    Variance Preserving (VP), sub-Variance Preserving (sub-VP), and ODE methods, as
    described in Song et al. (2021). The reverse process denoises a noisy input using
    predicted noise estimates.

    Parameters
    ----------
    hyper_params : object
        Hyperparameter object (HyperParamsSDE) containing SDE-specific parameters. Expected to have
        attributes: `dt`, `sigmas`, `betas`, `cum_betas`.
    method : str
        SDE method to use. Supported methods: "ve", "vp", "sub-vp", "ode".
    """
    def __init__(self, hyper_params: torch.nn.Module, method: str) -> None:
        super().__init__()
        self.hyper_params = hyper_params
        self.method = method

    def forward(self, xt: torch.Tensor, noise: torch.Tensor, predicted_noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Applies the reverse SDE diffusion process to the noisy input.

        Denoises the input `xt` by applying the reverse SDE process, using predicted
        noise estimates and optional stochastic noise, according to the specified SDE
        method at given time steps. Incorporates drift and diffusion terms as applicable.

        Parameters
        ----------
        xt : torch.Tensor
            Noisy input tensor at time step `t`, shape (batch_size, channels, height, width).
        noise : torch.Tensor or None
            Gaussian noise tensor, same shape as `xt`, used for stochasticity. If None,
            no stochastic noise is added (e.g., for deterministic ODE).
        predicted_noise : torch.Tensor
            Predicted noise tensor, same shape as `xt`, typically output by a neural network.
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, hyper_params.num_steps - 1].

        Returns
        -------
        xt (torch.Tensor) - Denoised tensor at the previous time step, same shape as `xt`.

        **Notes**

        - For the "ve" and "ode" methods, the output is clamped to [-1e5, 1e5] to prevent numerical instability.
        - Stochastic noise (`noise`) is only added if provided and the method supports it (not applicable for "ode" in non-VE cases).
        """
        dt = self.hyper_params.dt
        # Use properties to get betas and cum_betas (handles trainable case)
        betas = self.hyper_params.betas[time_steps].view(-1, 1, 1, 1)
        cum_betas = self.hyper_params._cum_betas[time_steps].view(-1, 1, 1, 1)
        if self.method == "ve":
            # Use property to get sigmas (handles trainable case)
            sigma_t = self.hyper_params.sigmas[time_steps]
            sigma_t_prev = self.hyper_params.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
            sigma_diff = torch.sqrt(torch.clamp(sigma_t ** 2 - sigma_t_prev ** 2, min=0))
            drift = -(sigma_t ** 2 - sigma_t_prev ** 2).view(-1, 1, 1, 1) * predicted_noise * dt
            diffusion = sigma_diff.view(-1, 1, 1, 1) * noise if noise is not None else 0
            xt = xt + drift + diffusion
            xt = torch.clamp(xt, -1e5, 1e5)

        elif self.method == "vp":
            drift = -0.5 * betas * xt * dt - betas * predicted_noise * dt
            diffusion = torch.sqrt(betas * dt) * noise if noise is not None else 0
            xt = xt + drift + diffusion

        elif self.method == "sub-vp":
            drift = -0.5 * betas * xt * dt - betas * (1 - torch.exp(-2 * cum_betas)) * predicted_noise * dt
            diffusion = torch.sqrt(betas * (1 - torch.exp(-2 * cum_betas)) * dt) * noise if noise is not None else 0
            xt = xt + drift + diffusion

        elif self.method == "ode":
            drift = -0.5 * betas * xt * dt - 0.5 * betas * predicted_noise * dt
            xt = xt + drift
            xt = torch.clamp(xt, -1e5, 1e5)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return xt

###==================================================================================================================###

class HyperParamsSDE(nn.Module):
    """Hyperparameters for SDE-based generative models.

    Manages the noise schedule and SDE-specific parameters for score-based generative
    models, including beta and sigma schedules, time steps, and variance computations,
    as described in Song et al. (2021). Supports trainable or fixed beta schedules and
    multiple scheduling methods for flexible noise control.

    Parameters
    ----------
    num_steps : int, optional
        Number of diffusion steps (default: 1000).
    beta_start : float, optional
        Starting value for beta schedule (default: 1e-4).
    beta_end : float, optional
        Ending value for beta schedule (default: 0.02).
    trainable_beta : bool, optional
        Whether the beta schedule is trainable (default: False).
    beta_method : str, optional
        Method for computing the beta schedule (default: "linear").
        Supported methods: "linear", "sigmoid", "quadratic", "constant", "inverse_time".
    sigma_start : float, optional
        Starting value for sigma schedule for VE method (default: 1e-3).
    sigma_end : float, optional
        Ending value for sigma schedule for VE method (default: 10.0).
    start : float, optional
        Start of the time interval for SDE integration (default: 0.0).
    end : float, optional
        End of the time interval for SDE integration (default: 1.0).
    """
    def __init__(
            self,
            num_steps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            trainable_beta: bool = False,
            beta_method: str = "linear",
            sigma_start: float = 1e-3,
            sigma_end: float = 10.0,
            start: float = 0.0,
            end: float = 1.0
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trainable_beta = trainable_beta
        self.beta_method = beta_method
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.start = start
        self.end = end

        if not (0 < self.beta_start < self.beta_end):
            raise ValueError(f"beta_start ({self.beta_start}) and beta_end ({self.beta_end}) must satisfy 0 < start < end")
        if not (0 < self.sigma_start < self.sigma_end):
            raise ValueError(f"sigma_start ({self.sigma_start}) and sigma_end ({self.sigma_end}) must satisfy 0 < start < end")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps ({self.num_steps}) must be positive")

        beta_range = (beta_start, beta_end)
        betas_init = self.compute_beta_schedule(beta_range, num_steps, beta_method)
        self.time = torch.linspace(self.start, self.end, self.num_steps, dtype=torch.float32)
        self.dt = (self.end - self.start) / self.num_steps

        if trainable_beta:
            # Use reparameterization trick for trainable betas
            # Initialize unconstrained parameters and transform them to valid beta range
            self.beta_raw = nn.Parameter(torch.logit((betas_init - beta_start) / (beta_end - beta_start)))
        else:
            self.register_buffer('betas_buffer', betas_init)
            self.register_buffer('cum_betas', torch.cumsum(betas_init, dim=0) * self.dt)
            self.register_buffer("sigmas_buffer", self.sigma_start * (self.sigma_end / self.sigma_start) ** self.time)

    @property
    def betas(self) -> torch.Tensor:
        """Returns the beta values, applying reparameterization if trainable."""
        if self.trainable_beta:
            # Transform unconstrained parameters to valid beta range using sigmoid
            return self.beta_start + (self.beta_end - self.beta_start) * torch.sigmoid(self.beta_raw)
        else:
            return self._buffers['betas_buffer']

    @property
    def _cum_betas(self) -> torch.Tensor:
        """Returns the cumulative beta values, computing dynamically if trainable."""
        if self.trainable_beta:
            return torch.cumsum(self.betas, dim=0) * self.dt
        else:
            return self._buffers['cum_betas']

    @property
    def sigmas(self) -> torch.Tensor:
        """Returns the sigma values, computing dynamically if trainable."""
        if self.trainable_beta:
            # For trainable case, sigmas still use the fixed time schedule
            # but could be modified if needed for trainable sigma schedules
            return self.sigma_start * (self.sigma_end / self.sigma_start) ** self.time
        else:
            return self._buffers['sigmas_buffer']

    def compute_beta_schedule(self, beta_range: Tuple[float, float], num_steps: int, method: str) -> torch.Tensor:
        """Computes the beta schedule based on the specified method.

        Generates a sequence of beta values for the SDE noise schedule using the chosen
        method, ensuring values are clamped within the specified range.

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
        betas (torch.Tensor) - Tensor of beta values, shape (num_steps,).
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
            raise ValueError(f"Unknown beta_method: {method}. Supported: linear, sigmoid, quadratic, constant, inverse_time")
        beta = torch.clamp(beta, min=beta_min, max=beta_max)
        return beta

    def get_variance(self, time_steps: torch.Tensor, method: str) -> torch.Tensor:
        """Computes the variance for the specified SDE method at given time steps.

        Calculates the variance used in SDE diffusion processes based on the method
        (VE, VP, or sub-VP), leveraging the sigma or cumulative beta schedules.

        Parameters
        ----------
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, num_steps - 1].
        method : str
            SDE method to compute variance for. Supported methods: "ve", "vp", "sub-vp".

        Returns
        -------
        variance_values (torch.Tensor) - Variance values for the specified time steps, shape (batch_size,).
        """
        if method == "ve":
            return self.sigmas[time_steps] ** 2
        elif method == "vp":
            return 1 - torch.exp(-self.cum_betas[time_steps])
        elif method == "sub-vp":
            return 1 - torch.exp(-2 * self.cum_betas[time_steps])
        else:
            raise ValueError(f"Unknown method: {method}")


#########################################################################################################################



import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
# Import required classes from the LDM module of TorchDiff
# Import utility functions from the TorchDiff utils module
from utils import TextEncoder, NoisePredictor, Metrics

transform = transforms.Compose([
    transforms.Resize(96),
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)



# Use small subset
train_subset = Subset(train_dataset, torch.randperm(len(train_dataset))[:100])
test_subset = Subset(test_dataset, torch.randperm(len(test_dataset))[:20])

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_subset, batch_size=20, shuffle=False)



# train auto-encoder of ldm models
compressor_model = AutoencoderLDM(
    in_channels=3,
    down_channels=[8, 16],
    up_channels=[16, 8],
    out_channels=3,
    dropout_rate=.1,
    latent_channels=2, # number of channels in latent space
    num_heads=1,
    num_groups=8,
    num_layers_per_block=3,
    total_down_sampling_factor=3,
    num_embeddings=32
)

compressor_metrics = Metrics(device="cpu", fid=True, metrics=True, lpips_=True)

compressor_optimizer = torch.optim.Adam(compressor_model.parameters(), lr=1e-3)
# loss function
obj = nn.MSELoss()

autoencoder_trainer = TrainAE(
    model=compressor_model,
    optimizer=compressor_optimizer,
    data_loader=train_loader,
    val_loader=val_loader,
    max_epoch=5,
    metrics_=compressor_metrics,
    device="cuda",
    store_path="vae_model",
    checkpoint=5,
    kl_warmup_epochs=2,
    patience=5,
    val_frequency=3,
    ddp=False,
    num_grad_accumulation=2,
    progress_frequency=3
)


#train_losses, best_val_loss = autoencoder_trainer()

noise_predictor = NoisePredictor(
    in_channels=2,  # number of latent space channels
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
)

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
)

# Set up the AdamW optimizer for the NoisePredictor parameters plus TextEncoder trainable parameters with a learning rate of 1e-3
optimizer = torch.optim.Adam(
    [p for p in noise_predictor.parameters() if p.requires_grad] +
    [p for p in text_encoder.parameters() if p.requires_grad], lr=1e-5
)

# Initialize SDE hyperparameters for the noise schedule
hyperparams_sde = HyperParamsSDE(
    num_steps=500,
    beta_start=1e-4,
    beta_end=0.02,
    sigma_start=1e-3,
    sigma_end=10.0,
    start=0.0,
    end=1.0,
    beta_method="linear"
)

# Set up the reverse diffusion process for sampling
reverse_sde = ReverseSDE(
    hyper_params=hyperparams_sde,
    method="ode" # there are three stochastic methods and one deterministic method available ("ve", "vp", "sub-vp", "ode"). Here we will use the deterministic (ode) method
)

# forward sde
forward_sde = ForwardSDE(
    hyper_params=hyperparams_sde,
    method="ode"
)

loss = nn.MSELoss()


# Configure the LDM trainer for model training
ldm_trainer = TrainLDM(
    model="sde", # "ddpm", "ddim", "sde"
    forward_diffusion=forward_sde,
    noise_predictor=noise_predictor,
    hyper_params=hyperparams_sde,
    compressor_model=compressor_model,# trained vae to send images to latent space (encode) and bring them back to original space (decode)
    conditional_model=text_encoder,
    reverse_diffusion=reverse_sde,
    metrics_=compressor_metrics,
    optimizer=optimizer,
    objective=loss,
    data_loader=train_loader,
    val_loader=val_loader,
    max_epoch=10,
    device=torch.device("cuda"),
    store_path="test_ldm",
    val_frequency=5,
    ddp=False,
    num_grad_accumulation=1,
    progress_frequency=1
)

# train_losses, best_val_loss = ldm_trainer()

sampler = SampleLDM(
    model="sde",
    reverse_diffusion=reverse_sde,
    noise_predictor=noise_predictor,
    compressor_model=compressor_model,
    image_shape=(96, 96),
    conditional_model=text_encoder,
    tokenizer="bert-base-uncased",
    max_length=77,
    batch_size=10,
    in_channels=3,
    device="cuda",
    output_range=(-1, 1)
)

#gen_imgs = sampler(
#    conditions=['airplane', 'bird', 'car', 'cat', 'deer',
#                'dog', 'horse', 'monkey', 'ship', 'truck'], # 10 labels as conditions
#    save_images=True,
#    save_path="ldm_generated"
#)