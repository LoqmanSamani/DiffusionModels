import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from transformers import BertTokenizer
import warnings
from typing import Callable, Optional, Dict, Any, Tuple, List, Union
import os



class TrainSDE(nn.Module):
    """Trainer for score-based generative models using Stochastic Differential Equations.

    Manages the training process for SDE-based generative models, optimizing a noise
    predictor to learn the noise added by the forward SDE process, as described in Song
    et al. (2021). Supports conditional training with text prompts, mixed precision,
    learning rate scheduling, early stopping, and checkpointing.

    Parameters
    ----------

    score_net : nn.Module
        Model to predict score/noise.
    fwd_sde : nn.Module
        Forward SDE diffusion module for adding noise.
    rwd_sde: nn.Module
        Reverse SDE diffusion module for denoising.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    optim : torch.optim.Optimizer
        Optimizer for training the noise predictor and conditional model (if applicable).
    loss_fn : callable
        Loss function to compute the difference between predicted and actual noise.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    max_epochs : int, optional
        Maximum number of training epochs (default: 1000).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    cond_model : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    metrics_ : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    bert_tokenizer : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    store_path : str, optional
        Path to save model checkpoints (default: "sde_model.pth").
    patience : int, optional
        Number of epochs to wait for improvement before early stopping (default: 10).
    warmup_epochs : int, optional
        Number of epochs for learning rate warmup (default: 100).
    val_freq : int, optional
        Frequency (in epochs) for validation (default: 10).
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
    """
    def __init__(
            self,
            score_net: torch.nn.Module,
            fwd_sde: torch.nn.Module,
            rwd_sde: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 1000,
            device: str = 'cuda',
            cond_model: Optional[torch.nn.Module] = None,
            metrics_: Optional[Any] = None,
            bert_tokenizer: Optional[BertTokenizer] = None,
            max_token_length: int = 77,
            store_path: Optional[str] = None,
            patience: int = 100,
            warmup_steps: int = 10000,
            val_freq: int = 10,
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            norm_output: bool = True,
            use_ddp: bool = False,
            grad_acc: int = 1,
            log_freq: int = 1,
            use_comp: bool = False,
            time_eps: float = 1e-5,
            pred_noise: bool = True,
            num_steps: int = 400,
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

        self.score_net = score_net.to(self.device)
        self.fwd_sde = fwd_sde.to(self.device)
        self.rwd_sde = rwd_sde.to(self.device)
        self.cond_model = cond_model.to(self.device) if cond_model else None

        self.metrics_ = metrics_
        self.optim = optim
        self.loss_fn = loss_fn
        self.store_path = store_path or "sde_model"
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
        self.pred_noise = pred_noise
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

        # get rank info
        self.ddp_rank = int(os.environ["RANK"])
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])

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
        if 'model_state_dict_score_net' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict_score_net' key")

        state_dict = checkpoint['model_state_dict_score_net']
        if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.score_net.load_state_dict(state_dict)
        if self.cond_model is not None:
            if 'model_state_dict_cond' in checkpoint and checkpoint['model_state_dict_cond'] is not None:
                cond_state_dict = checkpoint['model_state_dict_cond']
                if self.use_ddp and not any(key.startswith('module.') for key in cond_state_dict.keys()):
                    cond_state_dict = {f'module.{k}': v for k, v in cond_state_dict.items()}
                elif not self.use_ddp and any(key.startswith('module.') for key in cond_state_dict.keys()):
                    cond_state_dict = {k.replace('module.', ''): v for k, v in cond_state_dict.items()}
                self.cond_model.load_state_dict(cond_state_dict)
            else:
                warnings.warn(
                    "Checkpoint contains no 'model_state_dict_cond' or it is None, "
                    "skipping conditional model loading"
                )

        if 'scheduler_model' not in checkpoint:
            raise KeyError("Checkpoint missing 'scheduler_model' key")
        try:
            if isinstance(self.fwd_sde.vs, nn.Module):
                self.fwd_sde.vs.load_state_dict(checkpoint['scheduler_model'])
            if isinstance(self.rwd_sde.vs, nn.Module):
                self.rwd_sde.vs.load_state_dict(checkpoint['scheduler_model'])
            else:
                self.fwd_sde.vs = checkpoint['scheduler_model']
                self.rwd_sde.vs = checkpoint['scheduler_model']
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
        def lr_lambda(step):
            if step < warmup_epochs:
                return 0.1 + (0.9 * step / warmup_epochs)
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    def _wrap_models_for_ddp(self) -> None:
        """Wrap models with DistributedDataParallel for multi-GPU training."""
        if self.use_ddp:
            self.score_net = DDP(
                self.score_net,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )
            if self.cond_model is not None:
                self.cond_model = DDP(
                    self.cond_model,
                    device_ids=[self.ddp_local_rank],
                    find_unused_parameters=True
                )

    def forward(self) -> Dict:
        """Trains the SDE model to predict noise added by the forward diffusion process.

        Executes the training loop, optimizing the noise predictor and conditional model
        (if applicable) using mixed precision, gradient clipping, and learning rate
        scheduling. Supports validation, early stopping, and checkpointing.

        Returns
        -------
        train_losses : list of float
             List of mean training losses per epoch.
        best_val_loss : float
             Best validation or training loss achieved.

        **Notes**

        - Training uses mixed precision via `torch.cuda.amp` or `torch.amp` for efficiency.
        - Checkpoints are saved when the validation (or training) loss improves, and on early stopping.
        - Early stopping is triggered if no improvement occurs for `patience` epochs.
        """
        self.score_net.train()
        if self.cond_model is not None:
            self.cond_model.train()

        if self.use_comp:
            try:
                self.score_net = torch.compile(self.score_net)
                if self.cond_model is not None:
                    self.cond_model = torch.compile(self.cond_model)
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
            train_losses_epoch = []
            for step, (x, y) in enumerate(pbar):
                x = x.to(self.device)
                if self.cond_model is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None

                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    noise = torch.randn_like(x)
                    t = self.sample_time(x.shape[0], self.time_eps)
                    xt, score = self.fwd_sde(x, noise, t)
                    pred = self.score_net(xt, t, y_encoded, clip_embeddings=None)
                    var = self.fwd_sde.vs.variance(t)
                    if self.pred_noise:  # if model predicts noise
                        if self.fwd_sde.method == "ve":
                            sigma = self.fwd_sde.sigma_min * (self.fwd_sde.sigma_max / self.fwd_sde.sigma_min) ** t
                            loss = self.loss_fn(pred, noise, sigma) / self.grad_acc
                        else:
                            loss = self.loss_fn(pred, noise, var) / self.grad_acc
                    else:  # if model predicts score
                        if self.fwd_sde.method == "ve":
                            sigma = self.fwd_sde.sigma_min * (self.fwd_sde.sigma_max / self.fwd_sde.sigma_min) ** t
                            loss = self.loss_fn(pred, score, sigma) / self.grad_acc
                        else:
                            loss = self.loss_fn(pred, score, var) / self.grad_acc
                scaler.scale(loss).backward()
                if (step + 1) % self.grad_acc == 0:
                    scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
                    if self.cond_model is not None:
                        torch.nn.utils.clip_grad_norm_(self.cond_model.parameters(), max_norm=1.0)
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
        y_encoded = self.cond_model(input_ids, attention_mask)
        return y_encoded


    def _save_checkpoint(self, epoch: int, loss: float, preff: str = "") -> None:
        """Save model checkpoint (only called by master process).

        Parameters
        ----------
        epoch : int
            Current epoch number.
        loss : float
            Current loss value.
        preff : str, optional
            Suffix to add to checkpoint filename.
        """
        try:
            score_net_state = (
                self.score_net.module.state_dict() if self.use_ddp
                else self.score_net.state_dict()
            )
            cond_state = None
            if self.cond_model is not None:
                cond_state = (
                    self.cond_model.module.state_dict() if self.use_ddp
                    else self.cond_model.state_dict()
                )
            checkpoint = {
                'epoch': epoch,
                'model_state_dict_score_net': score_net_state,
                'model_state_dict_cond': cond_state,
                'optim_state_dict': self.optim.state_dict(),
                'loss': loss,
                'losses': self.losses,
                'scheduler_model': self.fwd_sde.vs.state_dict(),
                'max_epochs': self.max_epochs,
            }
            filename = f"{preff}model_epoch_{epoch}.pth"
            filepath = os.path.join(self.store_path, filename)
            os.makedirs(self.store_path, exist_ok=True)
            torch.save(checkpoint, filepath)

            print(f"Model saved at epoch {epoch}")
        except Exception as e:
            print(f"Failed to save model: {e}")


    def validate(self) -> Tuple[float, float, float, float, float, float]:
        """Validates the noise predictor and computes evaluation Metrics.

        Computes validation loss (MSE between predicted and ground truth noise) and generates
        samples using the reverse diffusion model by manually iterating over timesteps.
        Decodes samples to images and computes image-domain Metrics (MSE, PSNR, SSIM, FID, LPIPS)
        if metrics_ is provided.

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
        self.score_net.eval()
        if self.cond_model is not None:
            self.cond_model.eval()

        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []
        with torch.no_grad():
            with torch.no_grad():
                for x, y in self.val_loader:
                    x = x.to(self.device)
                    x_orig = x.clone()
                    if self.cond_model is not None:
                        y_encoded = self._process_conditional_input(y)
                    else:
                        y_encoded = None

                    noise = torch.randn_like(x)
                    t = self.sample_time(x.shape[0], self.time_eps)
                    xt, score = self.fwd_sde(x, noise, t)
                    pred = self.score_net(xt, t, y_encoded, clip_embeddings=None)
                    var = self.fwd_sde.vs.variance(t)
                    if self.pred_noise:  # if model predicts noise
                        if self.fwd_sde.method == "ve":
                            sigma = self.fwd_sde.sigma_min * (self.fwd_sde.sigma_max / self.fwd_sde.sigma_min) ** t
                            loss = self.loss_fn(pred, noise, sigma) / self.grad_acc
                        else:
                            loss = self.loss_fn(pred, noise, var) / self.grad_acc
                    else:  # if model predicts score
                        if self.fwd_sde.method == "ve":
                            sigma = self.fwd_sde.sigma_min * (self.fwd_sde.sigma_max / self.fwd_sde.sigma_min) ** t
                            loss = self.loss_fn(pred, score, sigma) / self.grad_acc
                        else:
                            loss = self.loss_fn(pred, score, var) / self.grad_acc
                    val_losses.append(loss.item())
                    if self.metrics_ is not None and self.rwd_sde is not None:
                        xt = torch.randn_like(x).to(self.device)
                        # reverse diffusion sampling
                        t_schedule = torch.linspace(1.0, self.time_eps, self.num_steps + 1)
                        dt = torch.tensor(-(1.0 - self.time_eps) / self.num_steps, device=xt.device, dtype=xt.dtype)
                        for t in range(self.num_steps):
                            t_current = float(t_schedule[t])
                            t_batch = torch.full((xt.shape[0],), t_current, dtype=xt.dtype, device=self.device)
                            pred = self.score_net(xt, t_batch, y_encoded, None)
                            if self.pred_noise:
                                std = self.fwd_sde.vs.std(t_batch)
                                while std.dim() < len(xt.shape):
                                    std = std.unsqueeze(-1)
                                score = -pred / (std + self.fwd_sde.eps)
                            else:
                                score = pred
                            last_step = (t == self.num_steps - 1)
                            xt = self.rwd_sde(xt, score, t_batch, dt, last_step = last_step)

                    x_hat = torch.clamp(xt, min=self.norm_range[0], max=self.norm_range[1])
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

        self.score_net.train()
        if self.cond_model is not None:
            self.cond_model.train()
        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg