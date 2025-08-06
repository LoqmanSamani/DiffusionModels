import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Callable, Any
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
import warnings
import os



class TrainUnCLIPPrior(nn.Module):
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
            max_epochs: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            metrics_: Optional[Any] = None,
            store_path: Optional[str] = None,
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            use_ddp: bool = False,
            num_grad_accumulation: int = 1,
            progress_frequency: int = 1,
            compilation: bool = False,
            output_range: Tuple[float, float] = (-1.0, 1.0),
            reduce_dim: bool = True,
            output_dim: int = 319,
            normalize: bool = True
    ) -> None:
        super().__init__()

        # Training configuration
        self.use_ddp = use_ddp
        self.num_grad_accumulation = num_grad_accumulation
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup distributed training
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # Core models
        self.prior_model = prior_model.to(self.device)
        self.clip_model = clip_model.to(self.device)
        self.hyper_params = hyper_params.to(self.device)
        self.forward_diffusion = forward_diffusion.to(self.device)

        # Projection models (for dimensionality reduction)
        self.reduce_dim = reduce_dim
        if self.reduce_dim and text_projection is not None and image_projection is not None:
            self.text_projection = text_projection.to(self.device)
            self.image_projection = image_projection.to(self.device)
        else:
            self.text_projection = None
            self.image_projection = None

        # Training components
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frequency = val_frequency
        self.progress_frequency = progress_frequency
        self.compilation = compilation
        self.output_range = output_range
        self.reduce_dim = reduce_dim
        self.normalize = normalize
        self.output_dim = output_dim

        # Checkpoint management
        self.store_path = store_path
        # os.makedirs(self.store_path, exist_ok=True)

        # Learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)


    def _setup_ddp(self) -> None:

        required_env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"DDP enabled but {var} environment variable not set")

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

    @staticmethod
    def warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
        def lr_lambda(epoch):
            return min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
        return LambdaLR(optimizer, lr_lambda)

    def _wrap_models_for_ddp(self) -> None:
        """Wrap models with DistributedDataParallel for multi-GPU training."""
        if self.use_ddp:
            # Wrap prior with DDP
            self.prior_model = DDP(
                self.prior_model,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )
            if self.reduce_dim:
                self.text_projection = DDP(self.text_projection, device_ids=[self.ddp_local_rank])
                self.image_projection = DDP(self.image_projection, device_ids=[self.ddp_local_rank])

    def _compile_models(self) -> None:
        """Compile models for optimization if supported."""
        if self.compilation:
            try:
                self.prior_model = torch.compile(self.prior_model)
                if self.reduce_dim:
                    self.text_projection = torch.compile(self.text_projection)
                    self.image_projection = torch.compile(self.image_projection)
                if self.master_process:
                    print("Models compiled successfully")
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

    def forward(self) -> Tuple[List[float], float]:

        # Set models to training mode
        self.prior_model.train()
        if self.reduce_dim:
            self.text_projection.train()
            self.image_projection.train()

        # Compile and wrap models
        self._compile_models()
        self._wrap_models_for_ddp()

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
            if self.use_ddp and hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []

            # Training step loop with gradient accumulation
            for step, (x, y) in enumerate(tqdm(self.train_loader, disable=not self.master_process)):
                x = x.to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    loss = self._compute_training_loss(x, y)
                    loss = loss / self.num_grad_accumulation

                # Backward pass
                scaler.scale(loss).backward()

                # Optimizer step with gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self._optimizer_step(scaler)
                    # Update learning rate (warmup scheduler)
                    self.warmup_lr_scheduler.step()

                # Record loss (unscaled)
                train_losses_epoch.append(loss.item() * self.num_grad_accumulation)

            # Compute and sync training loss
            mean_train_loss = self._compute_mean_loss(train_losses_epoch)
            train_losses.append(mean_train_loss)

            # Print training progress (only master process)
            if self.master_process and (epoch + 1) % self.progress_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}", end="")

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
            if self.use_ddp:
                loss_tensor = torch.tensor(mean_train_loss, device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                mean_train_loss = loss_tensor.item()

            train_losses.append(mean_train_loss)

            # Print training progress
            if self.master_process:
                if (epoch + 1) % self.progress_frequency == 0:
                    print(f"\nEpoch: {epoch + 1} | Learning Rate: {self.optimizer.param_groups[0]['lr']} | Train Loss: {mean_train_loss:.4f}", end="")

            # Validation and checkpointing
            current_loss = mean_train_loss
            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_metrics = self.validate()
                val_loss, mse, psnr, ssim = val_metrics
                current_loss = val_loss

                if self.master_process:
                    print(f" | Val Loss: {val_loss:.4f}")
                    if self.metrics_ and hasattr(self.metrics_, 'metrics') and self.metrics_.metrics:
                        print(f" | MSE: {mse:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}", end="")
            elif self.master_process:
                print()

            # Learning rate scheduling
            self.scheduler.step(current_loss)

            # Save checkpoint and early stopping
            if self.master_process:
                if current_loss < best_val_loss:
                    best_val_loss = current_loss
                    wait = 0
                    self._save_checkpoint(epoch + 1, best_val_loss, is_best=True)
                else:
                    wait += 1

                    if wait >= self.patience:
                        print("Early stopping triggered")
                        self._save_checkpoint(epoch + 1, current_loss, suffix="_early_stop")
                        break
        # Cleanup
        if self.use_ddp:
            destroy_process_group()

        return train_losses, best_val_loss


    def _compute_training_loss(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:

        with torch.no_grad():
            # Encode text and image with CLIP
            text_embeddings = self.clip_model(data=texts, data_type="text", normalize=self.normalize)
            image_embeddings = self.clip_model(data=images, data_type="img", normalize=self.normalize)

        # Reduce dimensionality (optional)
        if self.reduce_dim:
            text_embeddings = self.text_projection(text_embeddings)
            image_embeddings = self.image_projection(image_embeddings)

        # Sample timestep t ~ Uniform(1, T)
        batch_size = image_embeddings.shape[0]
        timesteps = torch.randint(0, self.hyper_params.num_steps, (batch_size,), device=self.device)

        # Sample noise ε ~ N(0, I)
        noise = torch.randn_like(image_embeddings)

        # Compute noised embedding z_{i,t}
        noisy_image_embeddings = self.forward_diffusion(image_embeddings, noise, timesteps)

        # Predict unnoised embedding ẑ_i
        predicted_image_embeddings = self.prior_model(text_embeddings, noisy_image_embeddings, timesteps)

        # Transform back to original space if using dimension reduction
        if self.reduce_dim:
            predicted_image_embeddings = self.image_projection.inverse_transform(predicted_image_embeddings)
            target_embeddings = self.image_projection.inverse_transform(image_embeddings)
        else:
            target_embeddings = image_embeddings

        # Compute loss L = ||ẑ_i - z_i||²
        loss = self.loss_fn(predicted_image_embeddings, target_embeddings)
        return loss

    def _optimizer_step(self, scaler: torch.cuda.amp.GradScaler) -> None:
        """Perform optimizer step with gradient clipping."""
        scaler.unscale_(self.optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.prior_model.parameters(), max_norm=1.0)
        if self.use_dimension_reduction:
            torch.nn.utils.clip_grad_norm_(self.image_projection.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.text_projection.parameters(), max_norm=1.0)

        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad()

    def _compute_mean_loss(self, losses: List[float]) -> float:
        """Compute mean loss and sync across processes if using DDP."""
        mean_loss = torch.tensor(losses).mean().item()

        if self.use_ddp:
            loss_tensor = torch.tensor(mean_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            mean_loss = loss_tensor.item()

        return mean_loss


    def validate(self) -> Tuple[float, ...]:

        self.prior_model.eval()
        if self.reduce_dim:
            self.text_projection.eval()
            self.image_projection.eval()

        # set fid and lpips of metrics to false! we do not need to compute them.
        self.metrics_.fid = False
        self.metrics_.lpips_ = False

        val_losses = []
        metrics_scores = {'mse': [], 'psnr': [], 'ssim': []}

        with torch.no_grad():
            for images, texts in self.val_loader:
                images = images.to(self.device, non_blocking=True)

                # Get embeddings
                text_embeddings = self.clip_model(data=texts, data_type="text", normalize=self.normalize)
                image_embeddings = self.clip_model(data=images, data_type="img", normalize=self.normalize)
                original_image_embeddings = image_embeddings.clone()

                if self.reduce_dim:
                    text_embeddings = self.text_projection(text_embeddings)
                    image_embeddings = self.image_projection(image_embeddings)

                # Forward diffusion
                batch_size = image_embeddings.shape[0]
                timesteps = torch.randint(0, self.hyper_params.num_steps, (batch_size,), device=self.device)
                noise = torch.randn_like(image_embeddings)
                noisy_image_embeddings = self.forward_diffusion(image_embeddings, noise, timesteps)

                # Predict
                predicted_embeddings = self.prior_model(text_embeddings, noisy_image_embeddings, timesteps)

                if self.reduce_dim:
                    predicted_embeddings = self.image_projection.inverse_transform(predicted_embeddings)

                # Compute loss
                loss = self.objective(predicted_embeddings, original_image_embeddings)
                val_losses.append(loss.item())

                # Compute metrics if available
                if self.metrics_ is not None:
                    if self.normalize:
                        predicted_embeddings = F.normalize(predicted_embeddings, p=2, dim=-1)
                        original_image_embeddings = F.normalize(original_image_embeddings, p=2, dim=-1)

                    metrics_result = self.metrics_.forward(original_image_embeddings, predicted_embeddings)
                    if hasattr(self.metrics_, 'metrics') and self.metrics_.metrics:
                        _, mse, psnr, ssim, _ = metrics_result
                        metrics_scores['mse'].append(mse)
                        metrics_scores['psnr'].append(psnr)
                        metrics_scores['ssim'].append(ssim)

        # Compute averages
        val_loss = self._compute_mean_loss(val_losses)

        results = [val_loss]
        for metric_name, scores in metrics_scores.items():
            if scores:
                avg_score = torch.tensor(scores).mean().item()
                results.append(avg_score)

        # Return to training mode
        self.prior_model.train()
        if self.use_dimension_reduction:
            self.text_projection.train()
            self.image_projection.train()

        return tuple(results)


    def _save_checkpoint(self, epoch: int, loss: float, suffix: str = "", is_best: bool = False) -> None:
        """Save model checkpoint."""
        try:
            # Get state dicts
            prior_state = (
                self.prior_model.module.state_dict() if self.use_ddp
                else self.prior_model.state_dict()
            )

            checkpoint = {
                'epoch': epoch,
                'prior_model_state_dict': prior_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'hyper_params_model': (
                    self.hyper_params.state_dict()
                    if isinstance(self.hyper_params, nn.Module)
                    else self.hyper_params
                ),
                'max_epochs': self.max_epochs,
            }

            # Add projection models if used
            if self.reduce_dim:
                checkpoint['image_projection_state_dict'] = (
                    self.image_projection.module.state_dict() if self.use_ddp
                    else self.image_projection.state_dict()
                )
                checkpoint['text_projection_state_dict'] = (
                    self.text_projection.module.state_dict() if self.use_ddp
                    else self.text_projection.state_dict()
                )

            # Save checkpoint
            if is_best:
                save_path = os.path.join(self.store_path, "best_model.pth")
            else:
                filename = f"checkpoint_epoch_{epoch}{suffix}.pth"
                save_path = os.path.join(self.store_path, filename)

            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved: {save_path}")

        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load prior model
        if 'prior_model_state_dict' in checkpoint:
            state_dict = checkpoint['prior_model_state_dict']

            # Handle DDP state dict compatibility
            if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.prior_model.load_state_dict(state_dict)

        # Load projection models
        if self.reduce_dim:
            for model_name, model in [('image_projection', self.image_projection),
                                      ('text_projection', self.text_projection)]:
                key = f'{model_name}_state_dict'
                if key in checkpoint:
                    try:
                        model.load_state_dict(checkpoint[key])
                    except Exception as e:
                        warnings.warn(f"Failed to load {model_name}: {e}")

        # Load optimizer
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                warnings.warn(f"Failed to load optimizer state: {e}")

        # Load scheduler
        if 'hyper_params_model' in checkpoint:
            try:
                if isinstance(self.hyper_params, nn.Module):
                    self.hyper_params.load_state_dict(checkpoint['hyper_params_model'])
                else:
                    self.hyper_params = checkpoint['hyper_params_model']
            except Exception as e:
                warnings.warn(f"Failed to load hyperparams model: {e}")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, loss {loss:.4f})")

        return epoch, loss