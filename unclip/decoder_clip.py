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
from project_decoder import ProjectDecoder
from ddim_model import ReverseDDIM, ForwardDDIM
from transformers import BertTokenizer



class TrainUnClipDecoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            noise_predictor: nn.Module,
            clip_model: nn.Module,
            hyper_params: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            text_projection: Optional[nn.Module] = None,
            image_projection: Optional[nn.Module] = None,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            conditional_model: torch.nn.Module = None,
            metrics_: Optional[Any] = None,
            tokenizer: Optional[BertTokenizer] = None,
            max_epochs: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            store_path: str = "unclip_decoder",
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
            normalize: bool = True,
            classifier_free: float = 0.1,
            drop_caption: float = 0.5
    ):
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
        self.noise_predictor = noise_predictor.to(self.device)
        self.clip_model = clip_model.to(self.device)
        self.hyper_params = hyper_params.to(self.device)
        self.forward_diffusion = ForwardDDIM(hyper_params=self.hyper_params).to(self.device)
        self.reverse_diffusion = ReverseDDIM(hyper_params=self.hyper_params).to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None


        # Projection models (for dimensionality reduction)
        self.reduce_dim = reduce_dim
        if self.reduce_dim and text_projection is not None and image_projection is not None:
            self.text_projection = text_projection.to(self.device)
            self.image_projection = image_projection.to(self.device)
        else:
            self.text_projection = None
            self.image_projection = None
        self.embedding_dim = output_dim if self.reduce_dim else embedding_dim
        self.decoder_projection = ProjectDecoder(input_dim=self.embedding_dim, num_tokens=4)

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
        self.classifier_free = classifier_free
        self.drop_caption = drop_caption

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

        # Initialize tokenizer
        if tokenizer is None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")

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

        # Initialize training components
        scaler = torch.GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        # Main training loop
        for epoch in range(self.max_epochs):
            # Set epoch for distributed sampler if using DDP
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []

            # Training step loop with gradient accumulation
            for step, (x, y) in enumerate(tqdm(self.train_loader, disable=not self.master_process)):
                x = x.to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    with torch.no_grad():
                        # Encode text and image with CLIP
                        text_embeddings = self.clip_model(data=y, data_type="text", normalize=self.normalize)
                        image_embeddings = self.clip_model(data=x, data_type="img", normalize=self.normalize)

                    # Reduce dimensionality (optional)
                    if self.reduce_dim:
                        text_embeddings = self.text_projection(text_embeddings)
                        image_embeddings = self.image_projection(image_embeddings)

                    rand1 = torch.rand().item()
                    if rand1 < self.classifier_free:
                        image_embeddings = torch.zeros_like(image_embeddings)
                    rand2 = torch.rand().item()
                    if rand2 < self.drop_caption:
                        text_embeddings = None

                    # Process conditional inputs if conditional model exists
                    if self.conditional_model is not None:
                        y_encoded = self._process_conditional_input(y)
                    else:
                        y_encoded = None

                    c = self.decoder_projection(image_embeddings)
                    if y_encoded is not None:
                        y_encoded = y_encoded.unsqueeze(1)  # Shape: [batch_size, 1, embed_dim]
                        # Concatenate along the sequence dimension (dim=1)
                        s = torch.cat([y_encoded, c], dim=1)
                    else:
                        s = c

                    # Generate noise and timesteps
                    noise = torch.randn_like(x).to(self.device)
                    t = torch.randint(0, self.hyper_params.num_steps, (x.shape[0],)).to(self.device)
                    # Apply forward diffusion
                    noisy_x = self.forward_diffusion(x, noise, t)

                    # Predict noise
                    predicted_noise = self.noise_predictor(noisy_x, t, s)

                    # Compute loss and scale for gradient accumulation
                    loss = self.objective(predicted_noise, noise) / self.num_grad_accumulation

                    # Backward pass - ONLY ONCE!
                scaler.scale(loss).backward()

                # Optimizer step with gradient accumulation
                if (step + 1) % self.num_grad_accumulation == 0:
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

            # Validation and checkpointing
            current_loss = mean_train_loss
            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_loss = self.validate()
                current_loss = val_loss

                if self.master_process:
                    print(f" | Val Loss: {val_loss:.4f}")
            elif self.master_process:
                print()

            # Learning rate scheduling
            self.scheduler.step(current_loss)

            # Save checkpoint and early stopping
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

            # Cleanup
        if self.use_ddp:
            destroy_process_group()

        return train_losses, best_val_loss

    def _process_conditional_input(self, y: Union[torch.Tensor, List]) -> torch.Tensor:

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

    def validate(self) -> Tuple[float, ...]:
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

        self.noise_predictor.eval()
        if self.conditional_model is not None:
            self.conditional_model.eval()

        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                x_orig = x.clone()

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
                    for t in reversed(range(self.hyper_params.tau_num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        prev_time_steps = torch.full((xt.shape[0],), max(t - 1, 0), device=self.device, dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, y_encoded)
                        xt, _ = self.reverse_diffusion(xt, predicted_noise, time_steps, prev_time_steps)

                    # Clamp and normalize generated samples
                    x_hat = torch.clamp(xt, min=self.output_range[0], max=self.output_range[1])
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


