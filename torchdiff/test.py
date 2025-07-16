import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
# multi-GPU processor module
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from transformers import BertTokenizer
import warnings
from torchvision.utils import save_image
import os


class TrainDDPM(nn.Module):
    """Trainer for Denoising Diffusion Probabilistic Models (DDPM) with Multi-GPU Support.

    Manages the training process for DDPM, optimizing a noise predictor model to learn
    the noise added by the forward diffusion process. Supports conditional training with
    text prompts, mixed precision training, learning rate scheduling, early stopping,
    checkpointing, and distributed data parallel (DDP) training across multiple GPUs.

    Parameters
    ----------
    noise_predictor : nn.Module
        Model to predict noise added during the forward diffusion process.
    hyper_params : nn.Module
        Hyperparameter module (e.g., HyperParamsDDPM) defining the noise schedule.
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data. Should be wrapped with DistributedSampler for DDP.
    optimizer : torch.optim.Optimizer
        Optimizer for training the noise predictor and conditional model (if applicable).
    objective : callable
        Loss function to compute the difference between predicted and actual noise.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    max_epoch : int, optional
        Maximum number of training epochs (default: 1000).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    conditional_model : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    metrics_ : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    tokenizer : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    max_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    store_path : str, optional
        Path to save model checkpoints (default: "ddpm_model.pth").
    patience : int, optional
        Number of epochs to wait for improvement before early stopping (default: 100).
    warmup_epochs : int, optional
        Number of epochs for learning rate warmup (default: 100).
    val_frequency : int, optional
        Frequency (in epochs) for validation (default: 10).
    output_range : tuple, optional
        Range for clamping generated images (default: (-1, 1)).
    normalize_output : bool, optional
        Whether to normalize generated images to [0, 1] for metrics (default: True).
    ddp : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    num_grad_accumulation : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    """

    def __init__(self, noise_predictor, hyper_params, data_loader, optimizer, objective, val_loader=None,
                 max_epoch=1000, device=None, conditional_model=None, metrics_=None, tokenizer=None, max_length=77,
                 store_path=None, patience=100, warmup_epochs=100, val_frequency=10, output_range=(-1, 1),
                 normalize_output=True, ddp=False, num_grad_accumulation=1):
        super().__init__()

        # Initialize DDP settings first
        self.ddp = ddp
        self.num_grad_accumulation = num_grad_accumulation

        # Setup distributed training if enabled
        if self.ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # Move models to appropriate device
        self.noise_predictor = noise_predictor.to(self.device)
        self.hyper_params = hyper_params.to(self.device)
        self.forward_diffusion = ForwardDDPM(hyper_params=self.hyper_params).to(self.device)
        self.reverse_diffusion = ReverseDDPM(hyper_params=self.hyper_params).to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None

        # Training components
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or "ddpm_model.pth"
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.max_length = max_length
        self.patience = patience
        self.val_frequency = val_frequency
        self.output_range = output_range
        self.normalize_output = normalize_output

        # Learning rate scheduling
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=self.patience, factor=0.5
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

    def _setup_ddp(self):
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

        # Initialize process group
        init_process_group(backend="nccl")

        # Get rank information
        self.ddp_rank = int(os.environ["RANK"])  # Global rank across all nodes
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])  # Local rank on current node
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes

        # Set device and make it current
        self.device = torch.device(f"cuda:{self.ddp_local_rank}")
        torch.cuda.set_device(self.device)

        # Master process handles logging, checkpointing, etc.
        self.master_process = self.ddp_rank == 0

        if self.master_process:
            print(f"DDP initialized with world_size={self.ddp_world_size}")

    def _setup_single_gpu(self):
        """Setup single GPU or CPU training configuration."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ddp_rank = 0
        self.ddp_local_rank = 0
        self.ddp_world_size = 1
        self.master_process = True

    def load_checkpoint(self, checkpoint_path):
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
    def warmup_scheduler(optimizer, warmup_epochs):
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

    def _wrap_models_for_ddp(self):
        """Wrap models with DistributedDataParallel for multi-GPU training."""
        if self.ddp:
            # Wrap noise predictor with DDP
            self.noise_predictor = DDP(
                self.noise_predictor,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=False  # Set to True if you have unused parameters
            )

            # Wrap conditional model with DDP if it exists
            if self.conditional_model is not None:
                self.conditional_model = DDP(
                    self.conditional_model,
                    device_ids=[self.ddp_local_rank],
                    find_unused_parameters=False
                )

    def forward(self):
        """Trains the DDPM model to predict noise added by the forward diffusion process.

        Executes the training loop with support for distributed training, gradient accumulation,
        mixed precision, gradient clipping, and learning rate scheduling. Includes validation,
        early stopping, and checkpointing functionality.

        Returns
        -------
        train_losses : list of float
             List of mean training losses per epoch.
        best_val_loss : float
             Best validation or training loss achieved.
        """
        # Set models to training mode
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()

        # Compile models for optimization (if supported)
        try:
            self.noise_predictor = torch.compile(self.noise_predictor)
            if self.conditional_model is not None:
                self.conditional_model = torch.compile(self.conditional_model)
        except Exception as e:
            if self.master_process:
                print(f"Model compilation failed: {e}. Continuing without compilation.")

        # Wrap models for DDP after compilation
        self._wrap_models_for_ddp()

        # Initialize training components
        scaler = GradScaler()
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

                # Process conditional inputs if conditional model exists
                if self.conditional_model is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None

                # Forward pass with mixed precision
                with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
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

            # Constrain betas if trainable
            if hasattr(self.hyper_params, 'trainable_beta') and self.hyper_params.trainable_beta:
                self.hyper_params.constrain_betas()

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
                print(f"\nEpoch: {epoch + 1} | Train Loss: {mean_train_loss:.4f}", end="")

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

    def _process_conditional_input(self, y):
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

    def _save_checkpoint(self, epoch, loss, suffix=""):
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

            save_path = self.store_path + suffix + ".pth" if suffix else self.store_path
            torch.save(checkpoint, save_path)
            print(f"Model saved at epoch {epoch} to {save_path}")

        except Exception as e:
            print(f"Failed to save model: {e}")

    def validate(self):
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
                    for t in reversed(range(self.hyper_params.num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, y_encoded)
                        xt = self.reverse_diffusion(xt, predicted_noise, time_steps)

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




import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import tempfile
import shutil


def setup_cpu_ddp(rank, world_size, master_port=12355):
    """Setup DDP for CPU testing"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Initialize process group with gloo backend for CPU
    torch.distributed.init_process_group(
        backend='gloo',
        init_method=f'tcp://localhost:{master_port}',
        rank=rank,
        world_size=world_size
    )


def create_dummy_components(device='cpu'):
    """Create dummy components for testing"""
    import torch.nn as nn

    # Dummy noise predictor
    class DummyNoisePredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
            self.time_embed = nn.Linear(1, 32)
            self.cond_embed = nn.Linear(768, 32)  # BERT embedding size

        def forward(self, x, t, cond=None):
            # Simple dummy forward pass
            t_embed = self.time_embed(t.float().unsqueeze(-1))
            out = self.conv(x)
            return out

    # Dummy conditional model (simulates BERT)
    class DummyConditionalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 768)

        def forward(self, input_ids, attention_mask):
            return self.embedding(input_ids).mean(dim=1)

    # Dummy hyperparameters
    class DummyHyperParams(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_steps = 100
            self.trainable_beta = False

        def constrain_betas(self):
            pass

    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return torch.randn(3, 32, 32), f"dummy prompt {idx}"

    # Create components
    noise_predictor = DummyNoisePredictor()
    conditional_model = DummyConditionalModel()
    hyper_params = DummyHyperParams()

    dataset = DummyDataset(200)
    val_dataset = DummyDataset(50)

    return noise_predictor, conditional_model, hyper_params, dataset, val_dataset


def test_ddp_process(rank, world_size, temp_dir):
    """Test function that runs in each process"""
    print(f"Process {rank} starting...")

    # Setup DDP
    setup_cpu_ddp(rank, world_size)

    # Create dummy components
    noise_predictor, conditional_model, hyper_params, dataset, val_dataset = create_dummy_components()

    # Create distributed samplers
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, sampler=val_sampler, num_workers=0)

    # Create optimizer and loss
    optimizer = torch.optim.Adam(list(noise_predictor.parameters()) + list(conditional_model.parameters()), lr=1e-4)
    objective = torch.nn.MSELoss()



    trainer = TrainDDPM(
        noise_predictor=noise_predictor,
        hyper_params=hyper_params,
        data_loader=train_loader,
        optimizer=optimizer,
        objective=objective,
        val_loader=val_loader,
        max_epoch=3,  # Short test
        device=torch.device('cpu'),
        conditional_model=conditional_model,
        store_path=os.path.join(temp_dir, f"test_model_rank_{rank}.pth"),
        ddp=True,
        num_grad_accumulation=2,
        val_frequency=1,
        progress_frequency=1
    )

    # Run training
    train_losses, best_val_loss = trainer.forward()

    print(f"Process {rank} completed. Best val loss: {best_val_loss:.4f}")

    # Cleanup
    torch.distributed.destroy_process_group()


def test_ddp_cpu():
    """Main test function for CPU-based DDP testing"""
    world_size = 2  # Simulate 2 GPUs

    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()

    try:
        # Spawn processes
        mp.spawn(
            test_ddp_process,
            args=(world_size, temp_dir),
            nprocs=world_size,
            join=True
        )
        print("CPU DDP test completed successfully!")

    except Exception as e:
        print(f"CPU DDP test failed: {e}")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_ddp_cpu()