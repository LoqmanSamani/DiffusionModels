import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Callable
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
import warnings
import os



class TrainUnCLIPPrior(nn.Module):
    """Trainer for the UnCLIPTransformerPrior model.

    Handles the training of the UnCLIP prior model to predict clean image embeddings from
    noisy image embeddings and text embeddings, with support for dimension reduction,
    mixed precision training, and distributed training.

    Parameters
    ----------
    `prior_model` : nn.Module
        The UnCLIP prior model to be trained (e.g., UnCLIPTransformerPrior).
    `clip_model` : nn.Module
        CLIP model for encoding text and images.
    `train_loader` : torch.utils.data.DataLoader
        DataLoader for training data.
    `optimizer` : torch.optim.Optimizer
        Optimizer for training the prior model.
    `objective` : Callable
        Loss function to compute the difference between predicted and target embeddings.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `max_epochs` : int, optional
        Maximum number of training epochs (default: 1000).
    `device` : Union[str, torch.device], optional
        Device for computation (default: CUDA if available, else CPU).
    `store_path` : str, optional
        Directory path to save model checkpoints, default None.
    `patience` : int, optional
        Number of epochs to wait for improvement before early stopping (default: 100).
    `warmup_epochs` : int, optional
        Number of epochs for learning rate warmup (default: 100).
    `val_frequency` : int, optional
        Frequency (in epochs) for validation (default: 10).
    `use_ddp` : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    `num_grad_accumulation` : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    `log_frequency` : int, optional
        Frequency (in epochs) for printing training progress (default: 1).
    `use_compilation` : bool, optional
        Whether to compile models for optimization (default: False).
    `embedding_output_range` : Tuple[float, float], optional
        Range for clamping output embeddings (default: (-1.0, 1.0)).
    `reduce_clip_embedding_dim` : bool, optional
        Whether to apply dimension reduction to embeddings (default: True).
    `transformer_embedding_dim` : int, optional
        Target dimensionality for reduced embeddings (default: 319).
    `normalize` : bool, optional
        Whether to normalize CLIP embeddings (default: True).
    """

    def __init__(
            self,
            prior_model: nn.Module,
            clip_model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            store_path: Optional[str] = None,
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            use_ddp: bool = False,
            grad_accumulation_steps: int = 1,
            log_frequency: int = 1,
            use_compilation: bool = False,
            embedding_output_range: Tuple[float, float] = (-1.0, 1.0),
            reduce_clip_embedding_dim: bool = True,
            transformer_embedding_dim: int = 319,
            normalize_clip_embeddings: bool = True
    ) -> None:
        super().__init__()

        # Training configuration
        self.use_ddp = use_ddp
        self.grad_accumulation_steps = grad_accumulation_steps
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Setup distributed training
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # Core models
        self.prior_model = prior_model.to(self.device)
        self.clip_model = clip_model.to(self.device)

        # Training components
        self.optimizer = optimizer
        self.objective = objective
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frequency = val_frequency
        self.log_frequency = log_frequency
        self.use_compilation = use_compilation
        self.embedding_output_range = embedding_output_range
        self.reduce_clip_embedding_dim = reduce_clip_embedding_dim
        self.normalize_clip_embeddings = normalize_clip_embeddings
        self.transformer_embedding_dim = transformer_embedding_dim

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
        """Wraps the prior model with DistributedDataParallel for multi-GPU training.

        Configures the prior model for DDP, setting device IDs and handling unused parameters.
        """
        if self.use_ddp:
            # Wrap prior with DDP
            self.prior_model = DDP(
                self.prior_model,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

    def _compile_models(self) -> None:
        """Compiles models for optimization if supported.

        Attempts to compile the prior model using torch.compile for performance optimization,
        with fallback to uncompiled models if compilation fails.
        """
        if self.use_compilation:
            try:
                self.prior_model = torch.compile(self.prior_model)

                if self.master_process:
                    print("Models compiled successfully")
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

    def forward(self) -> Tuple[List[float], float]:
        """Trains the UnCLIP prior model.

        Executes the training loop, optimizing the prior model to predict clean image embeddings
        from noisy embeddings and text conditions, with support for validation, early stopping,
        and checkpointing.

        Returns
        -------
        train_losses : List[float]
            List of mean training losses per epoch.
        best_val_loss : float
            Best validation or training loss achieved.
        """
        # Set models to training mode
        self.prior_model.train()

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
                    loss = self._compute_training_loss(x, y)
                    loss = loss / self.grad_accumulation_steps

                # Backward pass - ONLY ONCE!
                scaler.scale(loss).backward()

                # Optimizer step with gradient accumulation
                if (step + 1) % self.grad_accumulation_steps == 0:
                    self._optimizer_step(scaler)
                    # Update learning rate (warmup scheduler)
                    self.warmup_lr_scheduler.step()

                # Record loss (unscaled)
                train_losses_epoch.append(loss.item() * self.grad_accumulation_steps)

            # Compute and sync training loss
            mean_train_loss = self._compute_mean_loss(train_losses_epoch)
            train_losses.append(mean_train_loss)

            # Print training progress (only master process)
            if self.master_process and (epoch + 1) % self.log_frequency == 0:
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


    def _compute_training_loss(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """Computes the training loss for the UnCLIP prior model.

        Calculates the loss by encoding images and text with CLIP, applying forward diffusion,
        predicting clean embeddings, and comparing with target embeddings.

        Parameters
        ----------
        `images` : torch.Tensor
            Input images, shape (batch_size, channels, height, width).
        `texts` : List[str]
            List of text prompts for conditioning.

        Returns
        -------
        loss : torch.Tensor
            Loss value computed between predicted and target embeddings.
        """

        with torch.no_grad():
            # Encode text and image with CLIP
            text_embeddings = self.clip_model(data=texts, data_type="text", normalize=self.normalize_clip_embeddings)
            image_embeddings = self.clip_model(data=images, data_type="img", normalize=self.normalize_clip_embeddings)

            #print("encoded images: ", image_embeddings.size())
            #print("encoded text: ", text_embeddings.size())

        # Reduce dimensionality (optional)
        if self.reduce_clip_embedding_dim:
            text_embeddings = self.prior_model.clip_text_projection(text_embeddings)
            image_embeddings = self.prior_model.clip_image_projection(image_embeddings)
            #print("encoded images: ", image_embeddings.size())
            #print("encoded text: ", text_embeddings.size())

        # Sample timestep t ~ Uniform(1, T)
        batch_size = image_embeddings.shape[0]
        timesteps = torch.randint(0, self.prior_model.forward_diffusion.variance_scheduler.num_steps, (batch_size,), device=self.device)
        #print("time ", timesteps.size())

        # Sample noise ε ~ N(0, I)
        noise = torch.randn_like(image_embeddings)
        #print("noise ", noise.size())

        # Compute noised embedding z_{i,t}
        noisy_image_embeddings = self.prior_model.forward_diffusion(image_embeddings, noise, timesteps)
        #print("noisy image: ", noisy_image_embeddings.size())

        # Predict unnoised embedding ẑ_i
        predicted_image_embeddings = self.prior_model(text_embeddings, noisy_image_embeddings, timesteps)

        # Transform back to original space if using dimension reduction
        if self.reduce_clip_embedding_dim:
            predicted_image_embeddings = self.prior_model.image_projection.inverse_transform(predicted_image_embeddings)
            target_embeddings = self.prior_model.image_projection.inverse_transform(image_embeddings)
        else:
            target_embeddings = image_embeddings

        # Compute loss L = ||ẑ_i - z_i||²
        loss = self.objective(predicted_image_embeddings, target_embeddings)
        return loss

    def _optimizer_step(self, scaler: torch.GradScaler) -> None:
        """Performs an optimizer step with gradient clipping.

        Applies gradient clipping, updates the optimizer with scaled gradients, and resets
        gradients for the next iteration.

        Parameters
        ----------
        `scaler` : torch.GradScaler
            Gradient scaler for mixed precision training.
        """
        scaler.unscale_(self.optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.prior_model.parameters(), max_norm=1.0)

        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad()

    def _compute_mean_loss(self, losses: List[float]) -> float:
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

        self.prior_model.eval()

        val_losses = []

        with torch.no_grad():
            for images, texts in self.val_loader:
                images = images.to(self.device, non_blocking=True)

                # Get embeddings
                text_embeddings = self.clip_model(data=texts, data_type="text", normalize=self.normalize_clip_embeddings)
                image_embeddings = self.clip_model(data=images, data_type="img", normalize=self.normalize_clip_embeddings)
                original_image_embeddings = image_embeddings.clone()

                if self.reduce_clip_embedding_dim:
                    text_embeddings = self.prior_model.text_projection(text_embeddings)
                    image_embeddings = self.prior_model.image_projection(image_embeddings)

                # Forward diffusion
                batch_size = image_embeddings.shape[0]
                timesteps = torch.randint(0, self.prior_model.forward_diffusion.variance_scheduler.num_steps, (batch_size,), device=self.device)
                noise = torch.randn_like(image_embeddings)
                noisy_image_embeddings = self.prior_model.forward_diffusion(image_embeddings, noise, timesteps)

                # Predict
                predicted_embeddings = self.prior_model(text_embeddings, noisy_image_embeddings, timesteps)

                if self.reduce_clip_embedding_dim:
                    predicted_embeddings = self.prior_model.image_projection.inverse_transform(predicted_embeddings)

                # Compute loss
                loss = self.objective(predicted_embeddings, original_image_embeddings)
                val_losses.append(loss.item())


        # Compute averages
        val_loss = self._compute_mean_loss(val_losses)

        # Return to training mode
        self.prior_model.train()

        return val_loss


    def _save_checkpoint(self, epoch: int, loss: float, suffix: str = "", is_best: bool = False) -> None:
        """Saves a model checkpoint.

        Saves the state of the prior model and optimizer to a checkpoint file, with options
        for best model or early stopping checkpoints.

        Parameters
        ----------
        `epoch` : int
            Current epoch number.
        `loss` : float
            Current loss value.
        `suffix` : str, optional
            Suffix to append to the checkpoint filename, default "".
        `is_best` : bool, optional
            Whether to save the checkpoint as the best model, default False.
        """
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
                'max_epochs': self.max_epochs,
            }

            # Create the directory if it doesn't exist
            os.makedirs(self.store_path, exist_ok=True)

            # Define the checkpoint filename
            if is_best:
                filename = "best_model.pth"
            else:
                filename = f"checkpoint_epoch_{epoch}{suffix}.pth"

            # Construct the full save path
            save_path = os.path.join(self.store_path, filename)

            # Save checkpoint
            torch.save(checkpoint, save_path)
            if self.master_process:  # Only print from the master process in DDP
                print(f"Checkpoint saved: {save_path}")

        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
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

        # Load optimizer
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                warnings.warn(f"Failed to load optimizer state: {e}")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, loss {loss:.4f})")

        return epoch, loss



"""
from prior_diff import ForwardUnCLIP, ReverseUnCLIP, VarianceSchedulerUnCLIP
from prior_model import UnCLIPTransformerPrior
from clip_model import CLIPEncoder
from project_prior import Projection
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import torch


# Option 2A: Use CIFAR-10 with descriptive captions
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
train_subset_indices = torch.randperm(len(train_dataset))[:100]
test_subset_indices = torch.randperm(len(test_dataset))[:20]

train_subset = Subset(train_dataset, train_subset_indices)
test_subset = Subset(test_dataset, test_subset_indices)

# DataLoaders
t_loader = DataLoader(train_subset, batch_size=32, shuffle=True, pin_memory=True)
val = DataLoader(test_subset, batch_size=10, shuffle=False, pin_memory=True)

h_model = VarianceSchedulerUnCLIP(
    num_steps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=True,
    beta_method="cosine"
)

c_model = CLIPEncoder(model_name="openai/clip-vit-base-patch32")
tp = Projection(
    input_dim=512,
    output_dim=320,
    hidden_dim=480,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
)
ip = Projection(
    input_dim=512,
    output_dim=320,
    hidden_dim=480,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
)

d_model = ForwardUnCLIP(h_model)
r_model = ReverseUnCLIP(h_model)

p_model = UnCLIPTransformerPrior(
    forward_diffusion=d_model,
    reverse_diffusion=r_model, # will be used during training
    text_projection=tp,  # used during training instead of PCA in the main paper
    image_projection=ip,
    embedding_dim=320,
    num_layers=12,
    num_attention_heads=8,
    feedforward_dim=512,
    max_sequence_length=2,
    dropout_rate=0.3
)



opt = torch.optim.AdamW([p for p in p_model.parameters() if p.requires_grad], lr=1e-3)

models = [h_model, p_model, tp, ip]

total_params = 0
for model in models:
    total_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_params)

obj = nn.MSELoss()



train = TrainUnCLIPPrior(
    prior_model=p_model,
    clip_model=c_model,
    train_loader=t_loader,
    optimizer=opt,
    objective=obj,
    val_loader=val,
    max_epochs=5,
    device="cuda",
    store_path="prior",
    patience=3,
    warmup_epochs=2,
    val_frequency=3,
    use_ddp=False,
    num_grad_accumulation=2,
    progress_frequency=1,
    compilation=False,
    output_range=(-1.0, 1.0),
    reduce_dim=True,
    output_dim=320,
    normalize=True
)

train_losses, best_val_loss = train()
"""

