import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from typing import Optional, List, Tuple, Callable, Dict
from tqdm.auto import tqdm
import os
import warnings




class TrainUnCLIPPrior(nn.Module):
    """Trainer for the UnCLIPTransformerPrior model.

    Handles the training of the UnCLIP prior model to predict clean image embeddings from
    noisy image embeddings and text embeddings, with support for dimension reduction,
    mixed precision training, and distributed training.

    Parameters
    ----------
    `prior_net` : nn.Module
        The UnCLIP prior model to be trained (e.g., UnCLIPTransformerPrior).
    `clip_net` : nn.Module
        CLIP model for encoding text and images.
    `train_loader` : torch.utils.data.DataLoader
        DataLoader for training data.
    `optim` : torch.optim.Optimizer
        Optimizer for training the prior model.
    `loss_fn` : Callable
        Loss function to compute the difference between predicted and target embeddings.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `max_epochs` : int, optional
        Maximum number of training epochs (default: 100).
    `device` : str, optional
        Device for computation (default: CUDA).
    `store_path` : str, optional
        Directory path to save model checkpoints, default 'unclip_prior_train'".
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
        Frequency (in epochs) for printing training progress (default: 1).
    `use_comp` : bool, optional
        Whether to compile models for optimization (default: False).
    `nor_range` : Tuple[float, float], optional
        Range for clamping output embeddings (default: (-1.0, 1.0)).
    `reduce_clip_embed_dim` : bool, optional
        Whether to apply dimension reduction to embeddings (default: True).
    `trans_embed_dim` : int, optional
        Target dimensionality for reduced embeddings (default: 319).
    `norm_clip_embed`: bool
        Whether clip embedding are normalized (default: True)
    `use_autocast`: bool
        Whether mix percision is applied (default: True)
    """

    def __init__(
            self,
            prior_net: nn.Module,
            clip_net: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 100,
            device: str = 'cuda',
            store_path: str = 'unclip_prier_train',
            patience: int = 20,
            warmup_steps: int = 10000,
            val_freq: int = 10,
            use_ddp: bool = False,
            grad_acc: int = 1,
            log_freq: int = 1,
            use_comp: bool = False,
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            reduce_clip_embed_dim: bool = True,
            trans_embed_dim: int = 319,
            norm_clip_embed: bool = True,
            use_autocast: bool = True
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
        self.prior_net = prior_net.to(self.device)
        self.clip_net = clip_net.to(self.device)
        self.optim = optim
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_freq = val_freq
        self.log_freq = log_freq
        self.use_comp = use_comp
        self.norm_range = norm_range
        self.reduce_clip_embed_dim = reduce_clip_embed_dim
        self.norm_clip_embed = norm_clip_embed
        self.trans_embed_dim = trans_embed_dim
        self.store_path = store_path
        self.use_autocast = use_autocast
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

    def _setup_ddp(self) -> None:
        """Sets up Distributed Data Parallel training configuration.

        Initializes the process group, sets up rank information, and configures the CUDA
        device for the current process.

        Raises
        ------
        ValueError
            If required DDP environment variables (RANK, LOCAL_RANK, WORLD_SIZE) are not set.
        RuntimeError
            If CUDA is not available when DDP is enabled.
        """
        required_env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"DDP enabled but {var} environment variable not set")
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
        """Wraps the prior model with DistributedDataParallel for multi-GPU training.

        Configures the prior model for DDP, setting device IDs and handling unused parameters.
        """
        if self.use_ddp:
            self.prior_net = DDP(
                self.prior_net,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

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

    def forward(self) -> Dict:
        """Trains the UnCLIP prior model.

        Executes the training loop, optimizing the prior model to predict clean image embeddings
        from noisy embeddings and text conditions, with support for validation, early stopping,
        and checkpointing.

        Returns
        -------
        losses: dictionlaty contains train and validation losses
        """
        self.prior_net.train()
        self._compile_models()
        self._wrap_models_for_ddp()
        scaler = torch.GradScaler() if self.use_autocast else None
        wait = 0
        for epoch in range(self.max_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}", disable=not self.master_process)
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            train_losses_epoch = []
            for step, (x, y) in enumerate(pbar):
                x = x.to(self.device, non_blocking=True)
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu', enabled=self.use_autocast):
                    loss = self._train_loss(x, y)
                    loss = loss / self.grad_acc
                if self.use_autocast:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % self.grad_acc == 0:
                    if self.use_autocast:
                        self._optim_step(scaler)
                    if self.global_step > 0 and self.global_step < self.warmup_steps:
                        self.warmup_lr_scheduler.step()
                    self.global_step += 1
                pbar.set_postfix({'Loss': f'{loss.item() * self.grad_acc:.4f}'})
                train_losses_epoch.append(loss.item() * self.grad_acc)
            mean_train_loss = self._mean_loss(train_losses_epoch)
            self.losses['train_losses'].append(mean_train_loss)
            if self.master_process and (epoch + 1) % self.log_freq == 0:
                current_lr = self.optim.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}", end="")

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


    def _train_loss(self, imgs: torch.Tensor, txts: List[str]) -> torch.Tensor:
        """Computes the training loss for the UnCLIP prior model.

        Calculates the loss by encoding images and text with CLIP, applying forward diffusion,
        predicting clean embeddings, and comparing with target embeddings.

        Parameters
        ----------
        `imgs` : torch.Tensor
            Input images, shape (batch_size, channels, height, width).
        `txts` : List[str]
            List of text prompts for conditioning.

        Returns
        -------
        loss : torch.Tensor
            Loss value computed between predicted and target embeddings.
        """
        with torch.no_grad():
            # encode text and image with clip
            txt_embed = self.clip_net(data=txts, data_type="text", normalize=self.norm_clip_embed)
            img_embed = self.clip_net(data=imgs, data_type="img", normalize=self.norm_clip_embed)
        # reduce dimensionality
        if self.reduce_clip_embed_dim:
            txt_embed = self.prior_net.clip_text_proj(txt_embed)
            img_embed = self.prior_net.clip_img_proj(img_embed)
        # t ~ Uniform(1, T)
        batch_size = img_embed.shape[0]
        timesteps = torch.randint(0, self.prior_net.fwd_unclip.vs.train_steps, (batch_size,), device=self.device)
        # ε ~ N(0, I)
        noise = torch.randn_like(img_embed)
        # z_{i,t}
        noisy_img_embed, target = self.prior_net.fwd_unclip(img_embed, noise, timesteps)
        # ẑ_i
        pred_img_embed = self.prior_net(txt_embed, noisy_img_embed, timesteps)
        # transform back to original space if using dimension reduction
        if self.reduce_clip_embed_dim:
            pred_img_embed = self.prior_net.clip_img_proj.inverse_transform(pred_img_embed)
            target = self.prior_net.clip_img_proj.inverse_transform(target)
        # L = ||ẑ_i - z_i||²
        loss = self.loss_fn(pred_img_embed, target)
        return loss

    def _optim_step(self, scaler: torch.GradScaler) -> None:
        """Performs an optimizer step with gradient clipping.

        Applies gradient clipping, updates the optimizer with scaled gradients, and resets
        gradients for the next iteration.

        Parameters
        ----------
        `scaler` : torch.GradScaler
            Gradient scaler for mixed precision training.
        """
        if self.use_autocast:
            scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.prior_net.parameters(), max_norm=1.0)
        if self.use_autocast:
            scaler.step(self.optim)
            scaler.update()
        else:
            self.optim.step()
        self.optim.zero_grad()

    def _mean_loss(self, losses: List[float]) -> float:
        """Computes the mean loss and synchronizes across processes if using DDP.

        Calculates the mean of the provided loss values and performs an all-reduce operation
        in DDP mode to synchronize the loss across processes.

        Parameters
        ----------
        `losses` : List[float]
            List of loss values from a training or validation epoch.

        Returns
        -------
        mean_loss : float
            Mean loss value, synchronized across processes if DDP is enabled.
        """
        mean_loss = torch.tensor(losses).mean().item()
        if self.use_ddp:
            loss_tensor = torch.tensor(mean_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            mean_loss = loss_tensor.item()
        return mean_loss


    def validate(self) -> float:
        """Validates the UnCLIP prior model.

        Computes the validation loss by encoding images and text, applying forward diffusion,
        predicting clean embeddings, and comparing with target embeddings.

        Returns
        -------
        val_loss : float
            Mean validation loss, synchronized across processes if DDP is enabled.
        """
        self.prior_net.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, txts in self.val_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                txt_embed = self.clip_net(data=txts, data_type="text", normalize=self.norm_clip_embed)
                img_embed = self.clip_net(data=imgs, data_type="img", normalize=self.norm_clip_embed)
                if self.reduce_clip_embed_dim:
                    txt_embed = self.prior_net.clip_text_proj(txt_embed)
                    img_embed = self.prior_net.clip_img_proj(img_embed)
                batch_size = img_embed.shape[0]
                timesteps = torch.randint(0, self.prior_net.fwd_unclip.vs.train_steps, (batch_size,), device=self.device)
                noise = torch.randn_like(img_embed)
                noisy_img_embed, target = self.prior_net.fwd_unclip(img_embed, noise, timesteps)
                pred_embed = self.prior_net(txt_embed, noisy_img_embed, timesteps)
                if self.reduce_clip_embed_dim:
                    pred_embed = self.prior_net.clip_img_proj.inverse_transform(pred_embed)
                    target = self.prior_net.clip_img_proj.inverse_transform(target)
                loss = self.loss_fn(pred_embed, target)
                val_losses.append(loss.item())
        val_loss = self._mean_loss(val_losses)
        self.prior_net.train()
        return val_loss


    def _save_checkpoint(self, epoch: int, loss: float, pref: str = "") -> None:
        """Saves a model checkpoint.

        Saves the state of the prior model and optimizer to a checkpoint file, with options
        for best model or early stopping checkpoints.

        Parameters
        ----------
        `epoch` : int
            Current epoch number.
        `loss` : float
            Current loss value.
        `pref` : str, optional
            prefix to append to the checkpoint filename, default "".

        """
        try:
            prior_state = (
                self.prior_net.module.state_dict() if self.use_ddp
                else self.prior_net.state_dict()
            )
            checkpoint = {
                'epoch': epoch,
                'prior_net_state_dict': prior_state,
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

    def load_checkpoint(self, check_path: str) -> Tuple[int, float]:
        """Loads a model checkpoint to resume training.

        Restores the prior model and optimizer states from a saved checkpoint, handling
        DDP compatibility for state dictionaries.

        Parameters
        ----------
        `checkpoint_path` : str
            Path to the checkpoint file.

        Returns
        -------
        epoch : int
            The epoch at which the checkpoint was saved.
        loss : float
            The loss value at the checkpoint.
        """
        try:
            checkpoint = torch.load(check_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found: {check_path}")
        if 'prior_net_state_dict' in checkpoint:
            state_dict = checkpoint['prior_net_state_dict']
            if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.prior_net.load_state_dict(state_dict)
        if 'optim_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            except Exception as e:
                warnings.warn(f"Failed to load optimizer state: {e}")
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))
        if self.master_process:
            print(f"Loaded checkpoint from {check_path} (epoch {epoch}, loss {loss:.4f})")
        return epoch, loss