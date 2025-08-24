import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Callable, Any
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
import os
import warnings




class TrainUnClipDecoder(nn.Module):
    """Trainer for the UnCLIP decoder model.

    Orchestrates the training of the UnCLIP decoder model, integrating CLIP embeddings, forward
    and reverse diffusion processes, and optional dimensionality reduction. Supports mixed
    precision, gradient accumulation, DDP, and comprehensive evaluation metrics.

    Parameters
    ----------
    `clip_embedding_dim` : int
        Dimensionality of the input embeddings.
    `decoder_model` : nn.Module
        The UnCLIP decoder model (e.g., UnClipDecoder) to be trained.
    `clip_model` : nn.Module
        CLIP model for generating text and image embeddings.
    `train_loader` : torch.utils.data.DataLoader
        DataLoader for training data.
    `optimizer` : torch.optim.Optimizer
        Optimizer for training the decoder model.
    `objective` : Callable
        Loss function to compute the difference between predicted and target noise.
    `clip_text_projection` : nn.Module, optional
        Projection module for text embeddings, default None.
    `clip_image_projection` : nn.Module, optional
        Projection module for image embeddings, default None.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `metrics_` : Any, optional
        Object providing evaluation metrics (e.g., FID, MSE, PSNR, SSIM, LPIPS), default None.
    `max_epochs` : int, optional
        Maximum number of training epochs (default: 1000).
    `device` : Union[str, torch.device], optional
        Device for computation (default: CUDA if available, else CPU).
    `store_path` : str, optional
        Directory to save model checkpoints (default: "unclip_decoder").
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
    `reduce_clip_embedding_dim` : bool, optional
        Whether to apply dimensionality reduction to embeddings (default: True).
    `transformer_embedding_dim` : int, optional
        Output dimensionality for reduced embeddings (default: 312).
    `normalize_clip_embeddings` : bool, optional
        Whether to normalize CLIP embeddings (default: True).
    `finetune_clip_projections` : bool, optional
        Whether to fine-tune projection layers (default: False).
    """
    def __init__(
            self,
            clip_embedding_dim: int,
            decoder_model: nn.Module,
            clip_model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            clip_text_projection: Optional[nn.Module] = None,
            clip_image_projection: Optional[nn.Module] = None,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            metrics_: Optional[Any] = None,
            max_epochs: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            store_path: str = "unclip_decoder",
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            use_ddp: bool = False,
            grad_accumulation_steps: int = 1,
            log_frequency: int = 1,
            use_compilation: bool = False,
            image_output_range: Tuple[float, float] = (-1.0, 1.0),
            reduce_clip_embedding_dim: bool = True,
            transformer_embedding_dim: int = 312,
            normalize_clip_embeddings: bool = True,
            finetune_clip_projections: bool = False # if text_projection and image_projection model should be finetune
    ):
        super().__init__()
        # training configuration
        self.use_ddp = use_ddp
        self.grad_accumulation_steps = grad_accumulation_steps
        self.use_compilation = use_compilation
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # core models
        self.decoder_model = decoder_model.to(self.device)
        self.clip_model = clip_model.to(self.device)

        self.reduce_clip_embedding_dim = reduce_clip_embedding_dim

        # setup distributed training
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # compile and wrap models
        self._compile_models()
        self._wrap_models_for_ddp()

        # projection models (PCA equivalent in the paper)
        if self.reduce_clip_embedding_dim and clip_text_projection is not None and clip_image_projection is not None:
            self.clip_text_projection = clip_text_projection.to(self.device)
            self.clip_image_projection = clip_image_projection.to(self.device)
        else:
            self.clip_text_projection = None
            self.clip_image_projection = None

        # training components
        self.clip_embedding_dim = transformer_embedding_dim if self.reduce_clip_embedding_dim else clip_embedding_dim
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.train_loader = train_loader
        self.val_loader = val_loader

        # training parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frequency = val_frequency
        self.log_frequency = log_frequency
        self.image_output_range = image_output_range
        self.reduce_clip_embedding_dim = reduce_clip_embedding_dim
        self.normalize_clip_embeddings = normalize_clip_embeddings
        self.transformer_embedding_dim = transformer_embedding_dim
        self.finetune_clip_projections = finetune_clip_projections


        # checkpoint management
        self.store_path = store_path

        # learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)

    def forward(self) -> Tuple[List[float], float]:
        """Trains the UnCLIP decoder model to predict noise for denoising.

        Executes the training loop, optimizing the decoder model using CLIP embeddings, mixed
        precision, gradient clipping, and learning rate scheduling. Supports validation, early
        stopping, and checkpointing.

        Returns
        -------
        train_losses : List[float]
            List of mean training losses per epoch.
        best_val_loss : float
            Best validation or training loss achieved.
        """
        # set models to training mode
        self.decoder_model.train()  # sets noise_predictor, conditional_model, variance_scheduler, clip_time_proj to train mode
        if not self.decoder_model.forward_diffusion.variance_scheduler.trainable_beta:  # ff beta is not trainable
            self.decoder_model.forward_diffusion.variance_scheduler.eval()

        # set text_projection and image_projection to train mode if fine-tuning
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            if self.finetune_clip_projections:
                self.clip_text_projection.train()
                self.clip_image_projection.train()
            else:
                self.clip_text_projection.eval()
                self.clip_image_projection.eval()

        # set CLIP model to eval mode (frozen)
        if self.clip_model is not None:
            self.clip_model.eval()

        # initialize training components
        scaler = torch.GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        # main training loop
        for epoch in range(self.max_epochs):
            # set epoch for distributed sampler if using DDP
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []

            # training step loop with gradient accumulation
            for step, (images, texts) in enumerate(tqdm(self.train_loader, disable=not self.master_process)):
                images = images.to(self.device, non_blocking=True)

                # forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    # encode text and image with CLIP
                    text_embeddings, image_embeddings = self._get_clip_embeddings(images, texts)

                    # reduce dimensionality (PCA equivalent)
                    text_embeddings, image_embeddings = self._apply_dimensionality_reduction(
                        text_embeddings, image_embeddings
                    )

                    # use decoder model to predict noise
                    p_classifier_free = torch.rand(1).item()
                    p_text_drop = torch.rand(1).item()
                    predicted_noise, noise = self.decoder_model(
                        image_embeddings,
                        text_embeddings,
                        images,
                        texts,
                        p_classifier_free,
                        p_text_drop
                    )

                    # compute loss
                    loss = self.objective(predicted_noise, noise) / self.num_grad_accumulation

                scaler.scale(loss).backward()

                if (step + 1) % self.num_grad_accumulation == 0:
                    # clip gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.decoder_model.parameters(), max_norm=1.0)  # covers all submodules
                    if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None and self.finetune_clip_projections:
                        torch.nn.utils.clip_grad_norm_(self.clip_text_projection.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.clip_image_projection.parameters(), max_norm=1.0)

                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    self.warmup_lr_scheduler.step()
                    torch.cuda.empty_cache()  # clear memory after optimizer step

                train_losses_epoch.append(loss.item() * self.num_grad_accumulation)

            mean_train_loss = self._compute_mean_loss(train_losses_epoch)
            train_losses.append(mean_train_loss)

            if self.master_process and (epoch + 1) % self.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}")

            current_loss = mean_train_loss

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

        Configures the decoder model and, if fine-tuning, the projection models for DDP training.
        """
        if self.use_ddp:
            self.decoder_model = self.decoder_model.to(self.ddp_local_rank)
            self.decoder_model = DDP(
                self.decoder_model,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )
            # only wrap text_projection and image_projection if they are trainable
            if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None and self.finetune_clip_projections:
                self.clip_text_projection = self.clip_text_projection.to(self.ddp_local_rank)
                self.clip_image_projection = self.clip_image_projection.to(self.ddp_local_rank)
                self.clip_text_projection = DDP(self.clip_text_projection, device_ids=[self.ddp_local_rank])
                self.clip_image_projection = DDP(self.clip_image_projection, device_ids=[self.ddp_local_rank])

    def _compile_models(self) -> None:
        """Compiles models for optimization if supported.

        Attempts to compile the decoder model and, if fine-tuning, the projection models using
        torch.compile for optimization, falling back to uncompiled execution if compilation fails.
        """
        if self.use_compilation:
            try:
                self.decoder_model = self.decoder_model.to(self.device)
                self.decoder_model = torch.compile(self.decoder_model, mode="reduce-overhead")
                # only compile text_projection and image_projection if they are trainable
                if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None and self.finetune_clip_projections:
                    self.clip_text_projection = self.clip_text_projection.to(self.device)
                    self.clip_image_projection = self.clip_image_projection.to(self.device)
                    self.clip_text_projection = torch.compile(self.clip_text_projection, mode="reduce-overhead")
                    self.clip_image_projection = torch.compile(self.clip_image_projection, mode="reduce-overhead")
                if self.master_process:
                    print("Models compiled successfully")
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

    def _get_clip_embeddings(
            self,
            images: torch.Tensor,
            texts: Union[List, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes images and texts using the CLIP model.

        Generates text and image embeddings using the CLIP model, with optional normalization.

        Parameters
        ----------
        `images` : torch.Tensor
            Input images, shape (batch_size, channels, height, width).
        `texts` : Union[List, torch.Tensor]
            Text prompts for conditional generation.

        Returns
        -------
        text_embeddings : torch.Tensor
            CLIP text embeddings, shape (batch_size, embedding_dim).
        image_embeddings : torch.Tensor
            CLIP image embeddings, shape (batch_size, embedding_dim).
        """
        with torch.no_grad():
            # encode text y with CLIP text encoder: z_t ← CLIP_text(y)
            text_embeddings = self.clip_model(data=texts, data_type="text", normalize=self.normalize)
            # encode image x with CLIP image encoder: z_i ← CLIP_image(x)
            image_embeddings = self.clip_model(data=images, data_type="img", normalize=self.normalize)
        return text_embeddings, image_embeddings

    def _apply_dimensionality_reduction(
            self,
            text_embeddings: torch.Tensor,
            image_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies dimensionality reduction to embeddings if enabled.

        Projects text and image embeddings to a lower-dimensional space using learned
        projection layers, mimicking PCA as used in the UnCLIP paper.

        Parameters
        ----------
        `text_embeddings` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embedding_dim).
        `image_embeddings` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embedding_dim).

        Returns
        -------
        text_embeddings : torch.Tensor
            Projected text embeddings, shape (batch_size, output_dim) if reduced, else unchanged.
        image_embeddings : torch.Tensor
            Projected image embeddings, shape (batch_size, output_dim) if reduced, else unchanged.
        """
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            if not self.finetune_clip_projections:
                with torch.no_grad():
                    text_embeddings = self.clip_text_projection(text_embeddings.to(self.device))
                    image_embeddings = self.clip_image_projection(image_embeddings.to(self.device))
            else:
                text_embeddings = self.clip_text_projection(text_embeddings.to(self.device))
                image_embeddings = self.clip_image_projection(image_embeddings.to(self.device))
        return text_embeddings.to(self.device), image_embeddings.to(self.device)

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

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False, suffix: str = ""):
        """Saves model checkpoint.

        Saves the state of the decoder model, its submodules, optimizer, and schedulers,
        with options for best model and epoch-specific checkpoints.

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
            # Core models (submodules of decoder_model)
            'noise_predictor_state_dict': self.decoder_model.module.noise_predictor.state_dict() if self.use_ddp else self.decoder_model.noise_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Training configuration
            'embedding_dim': self.clip_embedding_dim,
            'output_dim': self.transformer_embedding_dim,
            'reduce_dim': self.reduce_clip_embedding_dim,
            'normalize': self.normalize
        }

        # Save conditional model (submodule of decoder_model)
        if self.decoder_model.conditional_model is not None:
            checkpoint['conditional_model_state_dict'] = (
                self.decoder_model.module.conditional_model.state_dict() if self.use_ddp
                else self.decoder_model.conditional_model.state_dict()
            )

        # Save variance scheduler (submodule of decoder_model, always saved)
        checkpoint['variance_scheduler_state_dict'] = (
            self.decoder_model.forward_diffusion.module.variance_scheduler.state_dict() if self.use_ddp
            else self.decoder_model.forward_diffusion.variance_scheduler.state_dict()
        )

        # Save CLIP time projection layer (submodule of decoder_model)
        checkpoint['clip_time_proj_state_dict'] = (
            self.decoder_model.module.clip_time_proj.state_dict() if self.use_ddp
            else self.decoder_model.clip_time_proj.state_dict()
        )

        # Save decoder projection layer (submodule of decoder_model)
        checkpoint['decoder_projection_state_dict'] = (
            self.decoder_model.module.decoder_projection.state_dict() if self.use_ddp
            else self.decoder_model.decoder_projection.state_dict()
        )

        # Save projection models (PCA equivalent)
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            checkpoint['text_projection_state_dict'] = (
                self.clip_text_projection.module.state_dict() if self.use_ddp
                else self.clip_text_projection.state_dict()
            )
            checkpoint['image_projection_state_dict'] = (
                self.clip_image_projection.module.state_dict() if self.use_ddp
                else self.clip_image_projection.state_dict()
            )

        # Save schedulers state
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        checkpoint['warmup_scheduler_state_dict'] = self.warmup_lr_scheduler.state_dict()

        filename = f"unclip_decoder_epoch_{epoch}{suffix}.pth"
        if is_best:
            filename = f"unclip_decoder_best{suffix}.pth"

        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)

        if is_best:
            print(f"Best model saved: {filepath}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """Loads model checkpoint.

        Restores the state of the decoder model, its submodules, optimizer, and schedulers
        from a saved checkpoint, handling DDP compatibility.

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

        # Load core noise predictor model (submodule of decoder_model)
        if 'noise_predictor_state_dict' in checkpoint:
            _load_model_state_dict(self.decoder_model.noise_predictor, checkpoint['noise_predictor_state_dict'],
                                   'noise_predictor')

        # Load conditional model (submodule of decoder_model)
        if self.decoder_model.conditional_model is not None and 'conditional_model_state_dict' in checkpoint:
            _load_model_state_dict(self.decoder_model.conditional_model, checkpoint['conditional_model_state_dict'],
                                   'conditional_model')

        # Load variance scheduler (submodule of decoder_model)
        if 'variance_scheduler_state_dict' in checkpoint:
            state_dict = checkpoint.get('variance_scheduler_state_dict')
            try:
                _load_model_state_dict(self.decoder_model.forward_diffusion.variance_scheduler, state_dict, 'variance_scheduler')
            except Exception as e:
                warnings.warn(f"Failed to load variance scheduler: {e}")

        # Load CLIP time projection layer (submodule of decoder_model)
        if 'clip_time_proj_state_dict' in checkpoint:
            try:
                _load_model_state_dict(self.decoder_model.clip_time_proj, checkpoint['clip_time_proj_state_dict'],
                                       'clip_time_proj')
            except Exception as e:
                warnings.warn(f"Failed to load CLIP time projection: {e}")

        # Load decoder projection layer (submodule of decoder_model)
        if 'decoder_projection_state_dict' in checkpoint:
            try:
                _load_model_state_dict(self.decoder_model.decoder_projection,
                                       checkpoint['decoder_projection_state_dict'], 'decoder_projection')
            except Exception as e:
                warnings.warn(f"Failed to load decoder projection: {e}")

        # Load projection models (PCA equivalent)
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            if 'text_projection_state_dict' in checkpoint:
                _load_model_state_dict(self.clip_text_projection, checkpoint['text_projection_state_dict'],
                                       'text_projection')
            if 'image_projection_state_dict' in checkpoint:
                _load_model_state_dict(self.clip_image_projection, checkpoint['image_projection_state_dict'],
                                       'image_projection')

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
        if 'embedding_dim' in checkpoint:
            if checkpoint['embedding_dim'] != self.clip_embedding_dim:
                warnings.warn(
                    f"Embedding dimension mismatch: checkpoint={checkpoint['embedding_dim']}, current={self.clip_embedding_dim}")

        if 'reduce_dim' in checkpoint:
            if checkpoint['reduce_dim'] != self.reduce_clip_embedding_dim:
                warnings.warn(
                    f"Reduce dimension setting mismatch: checkpoint={checkpoint['reduce_dim']}, current={self.reduce_clip_embedding_dim}")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        return epoch, loss

    def validate(self) -> Tuple[float, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Validates the UnCLIP decoder model.

        Computes validation loss and optional metrics (FID, MSE, PSNR, SSIM, LPIPS) by
        encoding images and texts, applying forward diffusion, predicting noise, and
        reconstructing images through reverse diffusion.

        Returns
        -------
        val_loss : float
            Mean validation loss.
        fid_avg : float or None
            Average FID score, if computed.
        mse_avg : float or None
            Average MSE score, if computed.
        psnr_avg : float or None
            Average PSNR score, if computed.
        ssim_avg : float or None
            Average SSIM score, if computed.
        lpips_avg : float or None
            Average LPIPS score, if computed.
        """

        # set models to eval mode for evaluation
        self.decoder_model.eval()  # sets noise_predictor, conditional_model, variance_scheduler, clip_time_proj, decoder_projection to eval mode
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            self.clip_text_projection.eval()
            self.clip_image_projection.eval()
        if self.clip_model is not None:
            self.clip_model.eval()

        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []

        with torch.no_grad():
            for images, texts in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                images_orig = images.clone()
                text_embeddings, image_embeddings = self._get_clip_embeddings(images, texts)
                text_embeddings, image_embeddings = self._apply_dimensionality_reduction(
                    text_embeddings, image_embeddings
                )
                p_classifier_free = torch.rand(1).item()
                p_text_drop = torch.rand(1).item()
                predicted_noise, noise = self.decoder_model(
                    image_embeddings,
                    text_embeddings,
                    images,
                    texts,
                    p_classifier_free,
                    p_text_drop
                )
                loss = self.objective(predicted_noise, noise)
                val_losses.append(loss.item())

                if self.metrics_ is not None and self.decoder_model.reverse_diffusion is not None:
                    xt = torch.randn_like(images).to(self.device)
                    for t in reversed(range(self.decoder_model.forward_diffusion.variance_scheduler.tau_num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        prev_time_steps = torch.full((xt.shape[0],), max(t - 1, 0), device=self.device, dtype=torch.long)
                        image_embeddings = self.decoder_model._apply_classifier_free_guidance(image_embeddings, p_classifier_free)
                        text_embeddings = self.decoder_model._apply_text_dropout(text_embeddings, p_text_drop)
                        c = self.decoder_model.decoder_projection(image_embeddings)  # updated to submodule
                        y_encoded = self.decoder_model._encode_text_with_glide(texts if text_embeddings is not None else None)
                        context = self.decoder_model._concatenate_embeddings(y_encoded, c)
                        clip_image_embedding = self.decoder_model.clip_time_proj(image_embeddings)
                        predicted_noise = self.decoder_model.noise_predictor(xt, time_steps, context, clip_image_embedding)
                        xt, _ = self.decoder_model.reverse_diffusion(xt, predicted_noise, time_steps, prev_time_steps)

                    x_hat = torch.clamp(xt, min=self.image_output_range[0], max=self.image_output_range[1])

                    if self.normalize:
                        x_hat = (x_hat - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])
                        x_orig = (images_orig - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])

                    metrics_result = self.metrics_.forward(x_orig, x_hat)
                    fid = metrics_result[0] if getattr(self.metrics_, 'fid', False) else float('inf')
                    mse = metrics_result[1] if getattr(self.metrics_, 'metrics', False) else None
                    psnr = metrics_result[2] if getattr(self.metrics_, 'metrics', False) else None
                    ssim = metrics_result[3] if getattr(self.metrics_, 'metrics', False) else None
                    lpips_score = metrics_result[4] if getattr(self.metrics_, 'lpips', False) else None

                    if fid != float('inf'):
                        fid_scores.append(fid)
                    if mse is not None:
                        mse_scores.append(mse)
                    if psnr is not None:
                        psnr_scores.append(psnr)
                    if ssim is not None:
                        ssim_scores.append(ssim)
                    if lpips_score is not None:
                        lpips_scores.append(lpips_score)

        # compute averages
        val_loss = torch.tensor(val_losses).mean().item()
        fid_avg = torch.tensor(fid_scores).mean().item() if fid_scores else float('inf')
        mse_avg = torch.tensor(mse_scores).mean().item() if mse_scores else None
        psnr_avg = torch.tensor(psnr_scores).mean().item() if psnr_scores else None
        ssim_avg = torch.tensor(ssim_scores).mean().item() if ssim_scores else None
        lpips_avg = torch.tensor(lpips_scores).mean().item() if lpips_scores else None

        # synchronize metrics across GPUs in DDP mode
        if self.use_ddp:
            metrics = [val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg]
            metrics_tensors = [torch.tensor(m, device=self.device) if m is not None else torch.tensor(float('inf'), device=self.device) for m in metrics]
            for tensor in metrics_tensors:
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg = [t.item() if t.item() != float('inf') else (None if i > 1 else float('inf')) for i, t in enumerate(metrics_tensors)]

        # return to training mode
        self.decoder_model.train()  # sets noise_predictor, conditional_model, variance_scheduler, clip_time_proj, decoder_projection to train mode
        if not self.decoder_model.variance_scheduler.trainable_beta:
            self.decoder_model.variance_scheduler.eval()
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            if self.finetune_clip_projections:
                self.clip_text_projection.train()
                self.clip_image_projection.train()
            else:
                self.clip_text_projection.eval()
                self.clip_image_projection.eval()
        if self.clip_model is not None:
            self.clip_model.eval()

        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg


"""
from utils import NoisePredictor, TextEncoder, Metrics
from clip_model import CLIPEncoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from project_prior import Projection
import torch
from prior_diff import VarianceSchedulerUnCLIP, ForwardUnCLIP, ReverseUnCLIP
from decoder_model import UnClipDecoder


class CIFAR10WithCaptions(Dataset):
    def __init__(self, cifar_dataset):
        self.dataset = cifar_dataset
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        # More descriptive templates
        self.templates = [
            "A photo of a {}",
            "An image of a {}",
            "A picture of a {}",
            "This is a {}",
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        class_name = self.class_names[label]
        # Use different templates for variety
        template = self.templates[idx % len(self.templates)]
        caption = template.format(class_name)
        return image, caption



# Updated transforms for CLIP
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 with captions
cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_dataset = CIFAR10WithCaptions(cifar_train)
test_dataset = CIFAR10WithCaptions(cifar_test)

# Small subset for testing
train_subset_indices = torch.randperm(len(train_dataset))[:4]
test_subset_indices = torch.randperm(len(test_dataset))[:2]
train_subset = Subset(train_dataset, train_subset_indices)
test_subset = Subset(test_dataset, test_subset_indices)

# DataLoaders
t_loader = DataLoader(train_subset, batch_size=2, shuffle=True, pin_memory=True)
v_loader = DataLoader(test_subset, batch_size=1, shuffle=False, pin_memory=True)

d = torch.device("cuda")

n_model = NoisePredictor(
        in_channels=3,
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


c_model = CLIPEncoder(
    model_name="openai/clip-vit-base-patch32",
    device="cuda",
    use_fast=False
)


t_proj = Projection(
    input_dim=512,
    output_dim=32,
    hidden_dim=128,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
)
i_proj = Projection(
    input_dim=512,
    output_dim=32,
    hidden_dim=128,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
)

h_model = VarianceSchedulerUnCLIP(
    num_steps=500,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=False,
    beta_method="linear"
)
for_ = ForwardUnCLIP(h_model)
rev_ = ReverseUnCLIP(h_model)

cond = TextEncoder(
    use_pretrained_model=True,
    model_name="bert-base-uncased",
    vocabulary_size=30522,
    num_layers=2,
    input_dimension=32,
    output_dimension=32,
    num_heads=2,
    context_length=77
).to(d)

decoder = UnClipDecoder(
    embedding_dim=32,
    noise_predictor=n_model,
    forward_diffusion=for_,
    reverse_diffusion=rev_,
    conditional_model=cond,  # GLIDE text encoder
    tokenizer=None,
    device="cpu",
    output_range=(-1.0, 1.0),
    normalize=True,
    classifier_free=0.1,  # paper specifies 10%
    drop_caption=0.5,  # paper specifies 50%
    max_length=77
)

opt = torch.optim.AdamW([p for p in decoder.parameters() if p.requires_grad], lr=1e-3)


obj = nn.MSELoss()

mets = Metrics(
    device="cpu",
    fid=True,
    metrics=True,
    lpips_=True
)


model = TrainUnClipDecoder(
    embedding_dim=512,
    decoder_model=decoder,
    clip_model=c_model,
    train_loader=t_loader,
    optimizer=opt,
    objective=obj,
    text_projection=t_proj,
    image_projection=i_proj,
    val_loader=v_loader,
    metrics_=mets,
    max_epoch=5,
    device="cuda",
    store_path="unclip_decoder",
    patience=5,
    warmup_epochs=2,
    val_frequency=10,
    use_ddp=False,
    num_grad_accumulation=1,
    progress_frequency=1,
    compilation=False,
    output_range=(-1.0, 1.0),
    reduce_dim=True,
    output_dim=32,
    normalize=True,
    finetune_projections=False
)

# Ensure requires_grad is set correctly
for p in model.clip_model.parameters():
    p.requires_grad = False
if not model.finetune_projections:
    for p in model.text_projection.parameters():
        p.requires_grad = False
    for p in model.image_projection.parameters():
        p.requires_grad = False
if not model.decoder_model.forward_diffusion.variance_scheduler.trainable_beta:
    for p in model.decoder_model.forward_diffusion.variance_scheduler.parameters():
        p.requires_grad = False

# Run training
one, two = model()

# Count trainable parameters
def count_trainable_parameters(model, finetune_projections=False):
    total_params = 0
    total_params += sum(p.numel() for p in model.decoder_model.parameters() if p.requires_grad)
    if finetune_projections and model.text_projection is not None and model.image_projection is not None:
        total_params += sum(p.numel() for p in model.text_projection.parameters() if p.requires_grad)
        total_params += sum(p.numel() for p in model.image_projection.parameters() if p.requires_grad)
    return total_params

# Case 1: finetune_projections=False, train_projection=False
print("Trainable parameters (finetune_projections=False, train_projection=False):")
total_params_false = count_trainable_parameters(model, finetune_projections=False)
print(f"Total trainable parameters: {total_params_false}")

# Case 2: finetune_projections=True, train_projection=True
model.finetune_projections = True
for p in model.text_projection.parameters():
    p.requires_grad = True
for p in model.image_projection.parameters():
    p.requires_grad = True

print("\nTrainable parameters (finetune_projections=True, train_projection=True):")
total_params_true = count_trainable_parameters(model, finetune_projections=True)
print(f"Total trainable parameters: {total_params_true}")

print("After parameters count")
"""
