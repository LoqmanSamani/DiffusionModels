import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from typing import Optional, List, Tuple, Union, Callable, Any, Dict
from tqdm.auto import tqdm
import os
import warnings




class TrainUnClipDecoder(nn.Module):
    """Trainer for the UnCLIP decoder model.

    Orchestrates the training of the UnCLIP decoder model, integrating CLIP embeddings, forward
    and reverse diffusion processes, and optional dimensionality reduction. Supports mixed
    precision, gradient accumulation, DDP, and comprehensive evaluation metrics.

    Parameters
    ----------
    `clip_embed_dim` : int
        Dimensionality of the input embeddings.
    `decoder_net` : nn.Module
        The UnCLIP decoder model (e.g., UnClipDecoder) to be trained.
    `clip_net` : nn.Module
        CLIP model for generating text and image embeddings.
    `train_loader` : torch.utils.data.DataLoader
        DataLoader for training data.
    `optim` : torch.optim.Optimizer
        Optimizer for training the decoder model.
    `loss_fn` : Callable
        Loss function to compute the difference between predicted and target noise.
    `clip_text_proj` : nn.Module, optional
        Projection module for text embeddings, default None.
    `clip_img_proj` : nn.Module, optional
        Projection module for image embeddings, default None.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `metrics_` : Any, optional
        Object providing evaluation metrics (e.g., FID, MSE, PSNR, SSIM, LPIPS), default None.
    `max_epochs` : int, optional
        Maximum number of training epochs (default: 100).
    `device` : str, optional
        Device for computation (default: CUDA).
    `store_path` : str, optional
        Directory to save model checkpoints (default: "unclip_decoder").
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
    `reduce_clip_embed_dim` : bool, optional
        Whether to apply dimensionality reduction to embeddings (default: True).
    `trans_embed_dim` : int, optional
        Output dimensionality for reduced embeddings (default: 312).
    `norm_clip_embed` : bool, optional
        Whether to normalize CLIP embeddings (default: True).
    `finetune_clip_proj` : bool, optional
        Whether to fine-tune projection layers (default: False).
    `use_autocast`: bool
        Whether use mix percision for efficienty (default: True)
    """
    def __init__(
            self,
            clip_embed_dim: int,
            decoder_net: nn.Module,
            clip_net: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            clip_text_proj: Optional[nn.Module] = None,
            clip_img_proj: Optional[nn.Module] = None,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            metrics_: Optional[Any] = None,
            max_epochs: int = 100,
            device: str = 'cuda',
            store_path: str = "unclip_decoder",
            patience: int = 20,
            warmup_steps: int = 10000,
            val_freq: int = 10,
            use_ddp: bool = False,
            grad_acc: int = 1,
            log_freq: int = 1,
            use_comp: bool = False,
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            reduce_clip_embed_dim: bool = True,
            trans_embed_dim: int = 312,
            norm_clip_embed: bool = True,
            finetune_clip_proj: bool = False, # if text_projection and image_projection model should be finetune
            use_autocast: bool =  True
    ):
        super().__init__()
        # training configuration
        self.use_ddp = use_ddp
        self.grad_acc = grad_acc
        self.use_comp = use_comp
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        # core models
        self.decoder_net = decoder_net.to(self.device)
        self.clip_net = clip_net.to(self.device)
        self.reduce_clip_embed_dim = reduce_clip_embed_dim
        # setup distributed training
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()
        # compile and wrap models
        self._compile_models()
        self._wrap_models_for_ddp()
        # projection models (PCA equivalent in the paper)
        if self.reduce_clip_embed_dim and clip_text_proj is not None and clip_img_proj is not None:
            self.clip_text_proj = clip_text_proj.to(self.device)
            self.clip_img_proj = clip_img_proj.to(self.device)
        else:
            self.clip_text_proj = None
            self.clip_img_proj = None
        # training components
        self.clip_embed_dim = trans_embed_dim if self.reduce_clip_embed_dim else clip_embed_dim
        self.metrics_ = metrics_
        self.optim = optim
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        # training parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_freq = val_freq
        self.log_freq = log_freq
        self.norm_range = norm_range
        self.norm_clip_embed = norm_clip_embed
        self.trans_embed_dim = trans_embed_dim
        self.finetune_clip_proj = finetune_clip_proj
        self.use_autocast = use_autocast
        # checkpoint management
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
        """Trains the UnCLIP decoder model to predict noise for denoising.

        Executes the training loop, optimizing the decoder model using CLIP embeddings, mixed
        precision, gradient clipping, and learning rate scheduling. Supports validation, early
        stopping, and checkpointing.

        Returns
        -------
        loses: a ductionlaty of losses (train and validation losses)
        """
        self.decoder_net.train()
        # set text_projection and image_projection to train mode if fine-tuning
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            if self.finetune_clip_proj:
                self.clip_text_proj.train()
                self.clip_img_proj.train()
            else:
                self.clip_text_proj.eval()
                self.clip_img_proj.eval()
        # set clip model to eval mode (frozen)
        if self.clip_net is not None:
            self.clip_net.eval()

        scaler = torch.GradScaler() if self.use_autocast else None
        wait = 0
        for epoch in range(self.max_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}", disable=not self.master_process)
            # set epoch for distributed sampler if using ddp
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            train_losses_epoch = []
            for step, (imgs, texts) in enumerate(pbar):
                imgs = imgs.to(self.device, non_blocking=True)
                with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', dtype=torch.bfloat16, enabled=self.use_autocast):
                    # encode text and image with clip
                    text_embed, img_embed = self._clip_embed(imgs, texts)
                    # reduce dimensionality (pca equivalent)
                    text_embed, img_embed = self._dim_reduction(text_embed, img_embed)
                    # use decoder model to predict noise
                    pred, target = self.decoder_net(
                        img_embed,
                        text_embed,
                        imgs,
                        texts
                    )
                    loss = self.loss_fn(pred, target) / self.grad_acc
                if self.use_autocast:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % self.grad_acc == 0:
                    if self.use_autocast:
                        scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.decoder_net.parameters(), max_norm=1.0)  # covers all submodules
                    if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None and self.finetune_clip_proj:
                        torch.nn.utils.clip_grad_norm_(self.clip_text_proj.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.clip_img_proj.parameters(), max_norm=1.0)
                    if self.use_autocast:
                        scaler.step(self.optim)
                        scaler.update()
                    else:
                        self.optim.step()
                    self.optim.zero_grad()
                    if self.global_step > 0 and self.global_step < self.warmup_steps:
                        self.warmup_lr_scheduler.step()
                    self.global_step += 1
                    #torch.cuda.empty_cache()  # clear memory after optimizer step
                pbar.set_postfix({'Loss': f'{loss.item() * self.grad_acc:.4f}'})
                train_losses_epoch.append(loss.item() * self.grad_acc)
            mean_train_loss = self._mean_loss(train_losses_epoch)
            self.losses['train_losses'].append(mean_train_loss)
            if self.master_process and (epoch + 1) % self.log_freq == 0:
                current_lr = self.optim.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}")

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

        Configures the decoder model and, if fine-tuning, the projection models for DDP training.
        """
        if self.use_ddp:
            self.decoder_net = self.decoder_net.to(self.ddp_local_rank)
            self.decoder_net = DDP(
                self.decoder_net,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )
            if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None and self.finetune_clip_proj:
                self.clip_text_proj = self.clip_text_proj.to(self.ddp_local_rank)
                self.clip_img_proj = self.clip_img_proj.to(self.ddp_local_rank)
                self.clip_text_proj = DDP(self.clip_text_proj, device_ids=[self.ddp_local_rank])
                self.clip_img_proj = DDP(self.clip_img_proj, device_ids=[self.ddp_local_rank])

    def _compile_models(self) -> None:
        """Compiles models for optimization if supported.

        Attempts to compile the decoder model and, if fine-tuning, the projection models using
        torch.compile for optimization, falling back to uncompiled execution if compilation fails.
        """
        if self.use_comp:
            try:
                self.decoder_net = self.decoder_net.to(self.device)
                self.decoder_net = torch.compile(self.decoder_net, mode="reduce-overhead")
                # only compile text_projection and image_projection if they are trainable
                if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None and self.finetune_clip_proj:
                    self.clip_text_proj = self.clip_text_proj.to(self.device)
                    self.clip_img_proj = self.clip_img_proj.to(self.device)
                    self.clip_text_proj = torch.compile(self.clip_text_proj, mode="reduce-overhead")
                    self.clip_img_proj = torch.compile(self.clip_img_proj, mode="reduce-overhead")
                if self.master_process:
                    print("Models compiled successfully")
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

    def _clip_embed(
            self,
            imgs: torch.Tensor,
            txts: Union[List, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes images and texts using the CLIP model.

        Generates text and image embeddings using the CLIP model, with optional normalization.

        Parameters
        ----------
        `imge` : torch.Tensor
            Input images, shape (batch_size, channels, height, width).
        `txts` : Union[List, torch.Tensor]
            Text prompts for conditional generation.

        Returns
        -------
        txt_embed : torch.Tensor
            CLIP text embeddings, shape (batch_size, embed_dim).
        img_embed : torch.Tensor
            CLIP image embeddings, shape (batch_size, embed_dim).
        """
        with torch.no_grad():
            # z_t ← CLIP_text(y)
            txt_embed = self.clip_net(data=txts, data_type="text", normalize=self.norm_clip_embed)
            # z_i ← CLIP_image(x)
            img_embed = self.clip_net(data=imgs, data_type="img", normalize=self.norm_clip_embed)
        return txt_embed, img_embed

    def _dim_reduction(
            self,
            txt_embed: torch.Tensor,
            img_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies dimensionality reduction to embeddings if enabled.

        Projects text and image embeddings to a lower-dimensional space using learned
        projection layers, mimicking PCA as used in the UnCLIP paper.

        Parameters
        ----------
        `txt_embed` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embed_dim).
        `img_embed` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embed_dim).

        Returns
        -------
        txt_embed : torch.Tensor
            Projected text embeddings, shape (batch_size, output_dim) if reduced, else unchanged.
        img_embed : torch.Tensor
            Projected image embeddings, shape (batch_size, output_dim) if reduced, else unchanged.
        """
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            if not self.finetune_clip_proj:
                with torch.no_grad():
                    txt_embed = self.clip_text_proj(txt_embed.to(self.device))
                    img_embed = self.clip_img_proj(img_embed.to(self.device))
            else:
                txt_embed = self.clip_text_proj(txt_embed.to(self.device))
                img_embed = self.clip_img_proj(img_embed.to(self.device))
        return txt_embed.to(self.device), img_embed.to(self.device)

    def _mean_loss(self, losses: List[float]) -> float:
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

    def _save_checkpoint(self, epoch: int, loss: float, pref: str = ""):
        """Saves model checkpoint.

        Saves the state of the decoder model, its submodules, optimizer, and schedulers,
        with options for best model and epoch-specific checkpoints.

        Parameters
        ----------
        `epoch` : int
            Current epoch number.
        `loss` : float
            Current loss value.

        `pref` : str, optional
            Prefix to add to checkpoint filename, default "".
        """
        if not self.master_process:
            return
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'losses': self.losses,
            # core models (submodules of decoder_model)
            'diff_net_state_dict': self.decoder_net.module.diff_net.state_dict() if self.use_ddp else self.decoder_net.diff_net.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            # training configuration
            'embedding_dim': self.clip_embed_dim,
            'output_dim': self.trans_embed_dim,
            'reduce_dim': self.reduce_clip_embed_dim,
            'normalize': self.norm_clip_embed
        }
        # save conditional model (submodule of decoder_model)
        if self.decoder_net.glide_text_encoder is not None:
            checkpoint['cond_model_state_dict'] = (
                self.decoder_net.module.glide_text_encoder.state_dict() if self.use_ddp
                else self.decoder_net.glide_text_encoder.state_dict()
            )
        # save scheduler (submodule of decoder_model, always saved)
        checkpoint['variance_scheduler_state_dict'] = (
            self.decoder_net.fwd_unclip.module.vs.state_dict() if self.use_ddp
            else self.decoder_net.fwd_unclip.vs.state_dict()
        )
        # save clip time projection layer (submodule of decoder_net)
        checkpoint['clip_time_proj_state_dict'] = (
            self.decoder_net.module.clip_time_proj.state_dict() if self.use_ddp
            else self.decoder_net.clip_time_proj.state_dict()
        )
        # save decoder projection layer (submodule of decoder_net)
        checkpoint['decoder_proj_state_dict'] = (
            self.decoder_net.module.clip_decoder_proj.state_dict() if self.use_ddp
            else self.decoder_net.clip_decoder_proj.state_dict()
        )
        # a nn.Linear projection layer
        checkpoint['clip_time_proj_state_dict'] = (
            self.decoder_net.module.clip_time_proj.state_dict() if self.use_ddp
            else self.decoder_net.clip_time_proj.state_dict()
        )
        # save projection models (pca equivalent)
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            checkpoint['text_proj_state_dict'] = (
                self.clip_text_proj.module.state_dict() if self.use_ddp
                else self.clip_text_proj.state_dict()
            )
            checkpoint['img_proj_state_dict'] = (
                self.clip_img_proj.module.state_dict() if self.use_ddp
                else self.clip_img_proj.state_dict()
            )
        # save schedulers state
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        checkpoint['warmup_scheduler_state_dict'] = self.warmup_lr_scheduler.state_dict()
        try:
            filename = f"{pref}model_epoch_{epoch}.pth"
            filepath = os.path.join(self.store_path, filename)
            os.makedirs(self.store_path, exist_ok=True)
            torch.save(checkpoint, filepath)
            print(f"Model saved at epoch {epoch} with loss: {loss:.4f}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def load_checkpoint(self, check_path: str) -> Tuple[int, float]:
        """Loads model checkpoint.

        Restores the state of the decoder model, its submodules, optimizer, and schedulers
        from a saved checkpoint, handling DDP compatibility.

        Parameters
        ----------
        `check_path` : str
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

        def _load_model(model: nn.Module, state_dict: dict, model_name: str) -> None:
            """Helper function to load state dict with DDP compatibility."""
            try:
                # handle ddp state dict compatibility
                if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                if self.master_process:
                    print(f"✓ Loaded {model_name}")
            except Exception as e:
                warnings.warn(f"Failed to load {model_name}: {e}")

        # load core diffusion network model (submodule of decoder_model)
        if 'diff_net_state_dict' in checkpoint:
            _load_model(self.decoder_net.diff_net, checkpoint['diff_net_state_dict'], 'diff_net')
        # load conditional model (submodule of decoder_model) - matches your save logic
        if self.decoder_net.glide_text_encoder is not None and 'cond_model_state_dict' in checkpoint:
            _load_model(self.decoder_net.glide_text_encoder, checkpoint['cond_model_state_dict'], 'glide_text_encoder')

        # load scheduler (submodule of decoder_model)
        if 'variance_scheduler_state_dict' in checkpoint:
            try:
                _load_model(self.decoder_net.fwd_unclip.vs, checkpoint['variance_scheduler_state_dict'], 'variance_scheduler')
            except Exception as e:
                warnings.warn(f"Failed to load variance scheduler: {e}")

        # load CLIP time projection layer (submodule of decoder_model)
        if 'clip_time_proj_state_dict' in checkpoint:
            try:
                _load_model(self.decoder_net.clip_time_proj, checkpoint['clip_time_proj_state_dict'],'clip_time_proj')
            except Exception as e:
                warnings.warn(f"Failed to load CLIP time projection: {e}")

        # load decoder projection layer (submodule of decoder_model)
        if 'decoder_proj_state_dict' in checkpoint:
            try:
                _load_model(self.decoder_net.clip_decoder_proj, checkpoint['decoder_proj_state_dict'], 'clip_decoder_proj')
            except Exception as e:
                warnings.warn(f"Failed to load decoder projection: {e}")

        if 'clip_time_proj_state_dict' in checkpoint and self.master_process:
            warnings.warn("Found duplicate 'clip_time_proj_state_dict' in checkpoint - skipping to avoid conflict")

        # load projection models (pca equivalent)
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            if 'text_proj_state_dict' in checkpoint:
                _load_model(self.clip_text_proj, checkpoint['text_proj_state_dict'], 'text_proj')
            if 'image_proj_state_dict' in checkpoint:
                _load_model(self.clip_img_proj, checkpoint['image_proj_state_dict'], 'image_proj')

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
        if 'embedding_dim' in checkpoint:
            if checkpoint['embedding_dim'] != self.clip_embed_dim:
                warnings.warn(
                    f"Embedding dimension mismatch: checkpoint={checkpoint['embedding_dim']}, current={self.clip_embed_dim}")

        if 'reduce_dim' in checkpoint:
            if checkpoint['reduce_dim'] != self.reduce_clip_embed_dim:
                warnings.warn(
                    f"Reduce dimension setting mismatch: checkpoint={checkpoint['reduce_dim']}, current={self.reduce_clip_embed_dim}")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Successfully loaded checkpoint from {check_path}")
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
        self.decoder_net.eval()
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            self.clip_text_proj.eval()
            self.clip_img_proj.eval()
        if self.clip_net is not None:
            self.clip_net.eval()
        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []
        with torch.no_grad():
            for imgs, txts in self.val_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                img_orig = imgs.clone()
                txt_embed, img_embed = self._clip_embed(imgs, txts)
                txt_embed, img_embed = self._dim_reduction(txt_embed, img_embed)
                p_classifier_free = torch.rand(1).item()
                p_text_drop = torch.rand(1).item()
                pred, target = self.decoder_net(
                    img_embed,
                    txt_embed,
                    imgs,
                    txts
                )
                loss = self.loss_fn(pred, target)
                val_losses.append(loss.item())
                if self.metrics_ is not None and self.decoder_net.rwd_unclip is not None:
                    xt = torch.randn_like(imgs).to(self.device)
                    timesteps = self.decoder_net.fwd_unclip.vs.inference_timesteps.flip(0)
                    for t in range(len(timesteps) - 1):
                        t_ = timesteps[t].item()
                        t_pre = timesteps[t+1].item()
                        time = torch.full((xt.shape[0],), t_, device=self.device, dtype=torch.long)
                        prev_time = torch.full((xt.shape[0],), t_pre, device=self.device, dtype=torch.long)
                        img_embed = self.decoder_net._classifier_free_guidance(img_embed)
                        txt_embed = self.decoder_net._text_dropout(txt_embed)
                        c = self.decoder_net.clip_decoder_proj(img_embed)
                        y = self.decoder_net._encode_text_with_glide(txts if txt_embed is not None else None)
                        context = self.decoder_net._conc_embed(y, c)
                        clip_img_embed = self.decoder_net.clip_time_proj(img_embed)
                        pred = self.decoder_net.diff_net(xt, time, context, clip_img_embed)
                        xt, _ = self.decoder_net.rwd_unclip(xt, time, prev_time, pred)

                    x_hat = torch.clamp(xt, min=self.norm_range[0], max=self.norm_range[1])
                    if self.norm_clip_embed:
                        x_hat = (x_hat - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])
                        x_orig = (img_orig - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])

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

        val_loss = torch.tensor(val_losses).mean().item()
        fid_avg = torch.tensor(fid_scores).mean().item() if fid_scores else float('inf')
        mse_avg = torch.tensor(mse_scores).mean().item() if mse_scores else None
        psnr_avg = torch.tensor(psnr_scores).mean().item() if psnr_scores else None
        ssim_avg = torch.tensor(ssim_scores).mean().item() if ssim_scores else None
        lpips_avg = torch.tensor(lpips_scores).mean().item() if lpips_scores else None

        if self.use_ddp:
            metrics = [val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg]
            metrics_tensors = [torch.tensor(m, device=self.device) if m is not None else torch.tensor(float('inf'), device=self.device) for m in metrics]
            for tensor in metrics_tensors:
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg = [t.item() if t.item() != float('inf') else (None if i > 1 else float('inf')) for i, t in enumerate(metrics_tensors)]

        self.decoder_net.train()
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            if self.finetune_clip_proj:
                self.clip_text_proj.train()
                self.clip_img_proj.train()
            else:
                self.clip_text_proj.eval()
                self.clip_img_proj.eval()
        if self.clip_net is not None:
            self.clip_net.eval()

        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg