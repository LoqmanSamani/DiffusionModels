import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Callable, Any
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
# multi-GPU processor module
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
import warnings
from torchvision.utils import save_image
import os



class TrainPrior(nn.Module):
    def __init__(
            self,
            prior_model: nn.Module,
            clip_model: nn.Module,
            hyper_params: nn.Module,
            forward_diffusion: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            text_projection: Optional[nn.Module] = None,  # used instead of PCA in the main paper
            image_projection: Optional[nn.Module] = None,  # used instead of PCA in the main paper
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epoch: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            metrics_: Optional[Any] = None,
            store_path: Optional[str] = None,
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            ddp: bool = False,
            num_grad_accumulation: int = 1,
            progress_frequency: int = 1,
            compilation: bool = False,
            output_range: Tuple[float, float] = (-1.0, 1.0),
            reduce_dim: bool = True,
            output_dim: int = 319,
            normalize: bool = True
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

        # Move models to appropriate device
        self.prior_model = prior_model.to(self.device)
        self.clip_model = clip_model.to(self.device)
        if text_projection is not None and image_projection is not None:
            self.text_projection = text_projection.to(self.device)
            self.image_projection = image_projection.to(self.device)

        self.hyper_params = hyper_params.to(self.device)
        self.forward_diffusion = forward_diffusion.to(self.device)

        # Training components
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or "prior_model"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.patience = patience
        self.val_frequency = val_frequency
        self.progress_frequency = progress_frequency
        self.compilation = compilation
        self.output_range = output_range
        self.reduce_dim = reduce_dim
        self.output_dim = output_dim
        self.normalize = normalize

        # Learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)

    def _setup_ddp(self) -> None:
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
        try:
            # Load checkpoint with proper device mapping
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        # Load prior model state
        if 'model_state_dict_prior' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict_prior' key")

        # Handle DDP wrapped model state dict
        state_dict = checkpoint['model_state_dict_prior']
        if self.ddp and not any(key.startswith('module.') for key in state_dict.keys()):
            # If loading non-DDP checkpoint into DDP model, add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not self.ddp and any(key.startswith('module.') for key in state_dict.keys()):
            # If loading DDP checkpoint into non-DDP model, remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.prior_model.load_state_dict(state_dict)

        # Load hyper_params state
        if 'hyper_params_model' not in checkpoint:
            raise KeyError("Checkpoint missing 'hyper_params_model' key")
        try:
            if isinstance(self.hyper_params, nn.Module):
                self.hyper_params.load_state_dict(checkpoint['hyper_params_model'])
            else:
                self.hyper_params = checkpoint['hyper_params_model']
        except Exception as e:
            warnings.warn(f"Hyper_params loading failed: {e}. Continuing with current hyper_params.")

        # Load projection models state
        if self.reduce_dim:
            if 'text_projection_model' not in checkpoint:
                raise KeyError("Checkpoint missing 'text_projection_model' key")
            try:
                if isinstance(self.text_projection, nn.Module):
                    self.text_projection.load_state_dict(checkpoint['text_projection_model'])
                else:
                    self.text_projection = checkpoint['text_projection_model']
            except Exception as e:
                warnings.warn(f"text_projection_model loading failed: {e}. Continuing with current text_projection_model.")

            if 'image_projection_model' not in checkpoint:
                raise KeyError("Checkpoint missing 'image_projection_model' key")
            try:
                if isinstance(self.image_projection, nn.Module):
                    self.image_projection.load_state_dict(checkpoint['image_projection_model'])
                else:
                    self.image_projection = checkpoint['image_projection_model']
            except Exception as e:
                warnings.warn(f"image_projection_model loading failed: {e}. Continuing with current image_projection_model.")

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
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    def _wrap_models_for_ddp(self) -> None:
        """Wrap models with DistributedDataParallel for multi-GPU training."""
        if self.ddp:
            # Wrap prior with DDP
            self.prior_model = DDP(
                self.prior_model,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

    def forward(self) -> Tuple[List, float]:

        # Set models to training mode
        self.prior_model.train()
        if self.reduce_dim:
            self.text_projection.train()
            self.image_projection.train()

        # Compile models for optimization (if supported)
        if self.compilation:
            try:
                self.prior_model = torch.compile(self.prior_model)
                if self.reduce_dim:
                    self.text_projection = torch.compile(self.text_projection)
                    self.image_projection = torch.compile(self.image_projection)
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

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

                # Forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):

                    with torch.no_grad():
                        text_embeddings = self.clip_model(data=y, data_type="text", normalize=self.normalize)
                        image_embeddings = self.clip_model(data=x, data_type="img", normalize=self.normalize)

                    if self.reduce_dim:
                        text_embeddings = self.text_projection(text_embeddings)
                        image_embeddings = self.image_projection(image_embeddings)

                    # Generate noise and timesteps
                    noise = torch.randn_like(image_embeddings).to(self.device)
                    t = torch.randint(0, self.hyper_params.num_steps, (image_embeddings.shape[0],)).to(self.device)

                    # Apply forward diffusion
                    noisy_image_embeddings = self.forward_diffusion(image_embeddings, noise, t)

                    # sequence_input = torch.cat([y, text_embeddings, noisy_image_embeddings], dim=-1)

                    # Predict un-noisy encoded images
                    pred_image_embeddings = self.prior_model(text_embeddings, noisy_image_embeddings, t)

                    if self.reduce_dim:
                        pred_image_embeddings = self.image_projection.inverse_transform(pred_image_embeddings)

                    # Compute loss and scale for gradient accumulation
                    loss = self.objective(pred_image_embeddings, image_embeddings) / self.num_grad_accumulation

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient accumulation and optimizer step
                if (step + 1) % self.num_grad_accumulation == 0:
                    # Clip gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.prior_model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.image_projection.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.text_projection.parameters(), max_norm=1.0)

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
                val_loss, mse, psnr, ssim = val_metrics

                if self.master_process:
                    print(f" | Val Loss: {val_loss:.4f}", end="")
                    if self.metrics_ and hasattr(self.metrics_, 'metrics') and self.metrics_.metrics:
                        print(f" | MSE: {mse:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}", end="")
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
        try:
            # Get state dicts, handling DDP wrapping
            prior_model_state = (
                self.prior_model.module.state_dict() if self.ddp
                else self.prior_model.state_dict()
            )
            image_proj_model_state = None
            text_proj_model_state = None
            if self.reduce_dim:
                image_proj_model_state = (
                    self.image_projection.module.state_dict() if self.ddp
                    else self.image_projection.state_dict()
                )
                text_proj_model_state = (
                    self.text_projection.module.state_dict() if self.ddp
                    else self.text_projection.state_dict()
                )

            checkpoint = {
                'epoch': epoch,
                'model_state_dict_prior': prior_model_state,
                'image_projection_model' : image_proj_model_state,
                'text_projection_model': text_proj_model_state,
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



    def validate(self) -> Tuple[float, float, float, float]:

        self.prior_model.eval()
        if self.reduce_dim:
            self.text_projection.eval()
            self.image_projection.eval()

        # set fid and lpips of metrics to false! we do not need to compute them.
        self.metrics_.fid = False
        self.metrics_.lpips_ = False

        val_losses = []
        mse_scores, psnr_scores, ssim_scores = [], [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)

                text_embeddings = self.clip_model(data=y, data_type="text", normalize=self.normalize)
                image_embeddings = self.clip_model(data=x, data_type="img", normalize=self.normalize)

                orig_image_embeddings = image_embeddings.clone()

                if self.reduce_dim:
                    text_embeddings = self.text_projection(text_embeddings)
                    image_embeddings = self.image_projection(image_embeddings)

                # Generate noise and timesteps
                noise = torch.randn_like(image_embeddings).to(self.device)
                t = torch.randint(0, self.hyper_params.num_steps, (image_embeddings.shape[0],)).to(self.device)

                # Apply forward diffusion
                noisy_image_embeddings = self.forward_diffusion(image_embeddings, noise, t)

                sequence_input = torch.cat([y, text_embeddings, noisy_image_embeddings], dim=-1)

                # Predict un-noisy encoded images
                pred_image_embeddings = self.prior_model(sequence_input, t)

                if self.reduce_dim:
                    pred_image_embeddings = self.text_projection.inverse_transform(pred_image_embeddings)

                # Compute loss and scale for gradient accumulation
                loss = self.objective(pred_image_embeddings, image_embeddings)
                val_losses.append(loss.item())

                # Generate samples for metrics evaluation
                if self.metrics_ is not None:
                    if self.normalize:
                        pred_image_embeddings = F.normalize(pred_image_embeddings, p=2, dim=-1)
                        orig_image_embeddings = F.normalize(orig_image_embeddings, p=2, dim=-1)

                    # Compute metrics
                    metrics_result = self.metrics_.forward(orig_image_embeddings, pred_image_embeddings)
                    _, mse, psnr, ssim, _ = metrics_result

                    if hasattr(self.metrics_, 'metrics') and self.metrics_.metrics:
                        mse_scores.append(mse)
                        psnr_scores.append(psnr)
                        ssim_scores.append(ssim)

        # Compute average metrics
        val_loss = torch.tensor(val_losses).mean().item()

        # All-reduce validation metrics across processes for DDP
        if self.ddp:
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()

        mse_avg = torch.tensor(mse_scores).mean().item() if mse_scores else None
        psnr_avg = torch.tensor(psnr_scores).mean().item() if psnr_scores else None
        ssim_avg = torch.tensor(ssim_scores).mean().item() if ssim_scores else None

        # Return to training mode
        self.prior_model.train()
        if self.reduce_dim:
            self.text_projection.train()
            self.image_projection.train()

        return val_loss, mse_avg, psnr_avg, ssim_avg