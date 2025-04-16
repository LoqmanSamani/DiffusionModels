"""Training module for Denoising Diffusion Implicit Models (DDIM).

This module implements the training process for DDIM, as described in Song et al. (2021,
"Denoising Diffusion Implicit Models"). It supports both unconditional and conditional
training with text prompts, using mixed precision and learning rate scheduling.
"""

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
import warnings
from forward_ddim import ForwardDDIM




class TrainDDIM(nn.Module):
    """Trainer for Denoising Diffusion Implicit Models (DDIM).

    Manages the training process for DDIM, optimizing a noise predictor model to learn
    the noise added by the forward diffusion process. Supports conditional training with
    text prompts, mixed precision training, learning rate scheduling, early stopping, and
    checkpointing, as inspired by Song et al. (2021).

    Parameters
    ----------
    noise_predictor : nn.Module
        Model to predict noise added during the forward diffusion process.
    hyper_params_model : nn.Module
        Hyperparameter module (e.g., HyperParamsDDIM) defining the noise schedule.
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data.
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
    tokenizer : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    max_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    store_path : str, optional
        Path to save model checkpoints (default: "ddim_model.pth").
    patience : int, optional
        Number of epochs to wait for improvement before early stopping (default: 10).
    warmup_epochs : int, optional
        Number of epochs for learning rate warmup (default: 100).
    val_frequency : int, optional
        Frequency (in epochs) for validation (default: 10).

    Attributes
    ----------
    device : torch.device
        Device used for computation.
    noise_predictor : nn.Module
        Noise prediction model.
    hyper_params_model : nn.Module
        Hyperparameter module for the noise schedule.
    conditional_model : nn.Module or None
        Conditional model for text-based training, if provided.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    objective : callable
        Loss function for training.
    store_path : str
        Path for saving checkpoints.
    data_loader : torch.utils.data.DataLoader
        Training data loader.
    val_loader : torch.utils.data.DataLoader or None
        Validation data loader, if provided.
    max_epoch : int
        Maximum training epochs.
    max_length : int
        Maximum length for tokenized prompts.
    patience : int
        Patience for early stopping.
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler based on validation or training loss.
    forward_diffusion : ForwardDDIM
        Forward diffusion module for DDIM.
    warmup_lr_scheduler : torch.optim.lr_scheduler.LambdaLR
        Learning rate scheduler for warmup.
    val_frequency : int
        Frequency for validation.
    tokenizer : BertTokenizer
        Tokenizer for text prompts.

    Raises
    ------
    ValueError
        If the default tokenizer ("bert-base-uncased") fails to load and no tokenizer is provided.
    """
    def __init__(self, noise_predictor, hyper_params_model, data_loader, optimizer, objective, val_loader=None,
                 max_epoch=1000, device=None, conditional_model=None, tokenizer=None, max_length=77,
                 store_path=None, patience=10, warmup_epochs=100, val_frequency=10):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_predictor = noise_predictor
        self.hyper_params_model = hyper_params_model.to(self.device)
        self.conditional_model = conditional_model
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or "ddim_model.pth"
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.max_length = max_length
        self.patience = patience
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience, factor=0.5)
        self.forward_diffusion = ForwardDDIM(hyper_params=self.hyper_params_model).to(self.device)
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)
        self.val_frequency = val_frequency
        if tokenizer is None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")

    def load_checkpoint(self, checkpoint_path):
        """Loads a training checkpoint to resume training.

        Restores the state of the noise predictor, conditional model (if applicable),
        and optimizer from a saved checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.

        Returns
        -------
        tuple
            A tuple containing:
            - epoch: The epoch at which the checkpoint was saved (int).
            - loss: The loss at the checkpoint (float).

        Raises
        ------
        FileNotFoundError
            If the checkpoint file is not found.
        KeyError
            If the checkpoint is missing required keys ('model_state_dict_noise_predictor'
            or 'optimizer_state_dict').

        Warns
        -----
        warnings.warn
            If the optimizer state cannot be loaded, if the checkpoint contains a
            conditional model state but none is defined, or if no conditional model
            state is provided when expected.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        if 'model_state_dict_noise_predictor' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict_noise_predictor' key")
        self.noise_predictor.load_state_dict(checkpoint['model_state_dict_noise_predictor'])

        if self.conditional_model is not None:
            if 'model_state_dict_conditional' in checkpoint and checkpoint['model_state_dict_conditional'] is not None:
                self.conditional_model.load_state_dict(checkpoint['model_state_dict_conditional'])
            else:
                warnings.warn(
                    "Checkpoint contains no 'model_state_dict_conditional' or it is None, skipping conditional model loading")
        elif 'model_state_dict_conditional' in checkpoint and checkpoint['model_state_dict_conditional'] is not None:
            warnings.warn(
                "Checkpoint contains conditional model state, but no conditional model is defined in this instance")

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

        print(f"Loaded checkpoint from {checkpoint_path} at epoch {epoch} with loss {loss:.4f}")
        return epoch, loss

    @staticmethod
    def warmup_scheduler(optimizer, warmup_epochs=10):
        """Creates a learning rate scheduler for warmup.

        Generates a scheduler that linearly increases the learning rate from 0 to the
        optimizer's initial value over the specified warmup epochs, then maintains it.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to apply the scheduler to.
        warmup_epochs : int, optional
            Number of epochs for the warmup phase (default: 10).

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

    def forward(self):
        """Trains the DDIM model to predict noise added by the forward diffusion process.

        Executes the training loop, optimizing the noise predictor and conditional model
        (if applicable) using mixed precision, gradient clipping, and learning rate
        scheduling. Supports validation, early stopping, and checkpointing.

        Returns
        -------
        tuple
            A tuple containing:
            - train_losses: List of mean training losses per epoch (list of float).
            - best_val_loss: Best validation or training loss achieved (float).

        Notes
        -----
        - Training uses mixed precision via `torch.cuda.amp` or `torch.amp` for efficiency.
        - Checkpoints are saved when the validation (or training) loss improves, and on early stopping.
        - Early stopping is triggered if no improvement occurs for `patience` epochs.
        """
        self.noise_predictor.train()
        self.noise_predictor.to(self.device)
        if self.conditional_model is not None:
            self.conditional_model.train()
            self.conditional_model.to(self.device)

        scaler = GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0
        for epoch in range(self.max_epoch):
            train_losses_ = []
            for x, y in tqdm(self.data_loader):
                x = x.to(self.device)

                if self.conditional_model is not None:
                    y_list = y.cpu().numpy().tolist() if isinstance(y, torch.Tensor) else y
                    y_list = [str(item) for item in y_list]
                    y_encoded = self.tokenizer(
                        y_list,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    input_ids = y_encoded["input_ids"]
                    attention_mask = y_encoded["attention_mask"]
                    y_encoded = self.conditional_model(input_ids, attention_mask)
                else:
                    y_encoded = None

                self.optimizer.zero_grad()
                with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    noise = torch.randn_like(x).to(self.device)
                    t = torch.randint(0, self.hyper_params_model.num_steps, (x.shape[0],)).to(self.device)
                    assert x.device == noise.device == t.device, "Device mismatch detected"
                    assert t.shape[0] == x.shape[0], "Timestep batch size mismatch"
                    noisy_x = self.forward_diffusion(x, noise, t)
                    p_noise = self.noise_predictor(noisy_x, t, y_encoded)
                    loss = self.objective(p_noise, noise)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.noise_predictor.parameters(), max_norm=1.0)
                if self.conditional_model is not None:
                    nn.utils.clip_grad_norm_(self.conditional_model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                self.warmup_lr_scheduler.step()
                train_losses_.append(loss.item())

            mean_train_loss = torch.mean(torch.tensor(train_losses_)).item()
            train_losses.append(mean_train_loss)
            print(f"\nEpoch: {epoch + 1} | Train Loss: {mean_train_loss:.4f}", end="")

            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_loss = self.validate()
                print(f" | Val Loss: {val_loss:.4f}")
                current_best = val_loss
                self.scheduler.step(val_loss)
            else:
                print()
                current_best = mean_train_loss
                self.scheduler.step(mean_train_loss)

            if current_best < best_val_loss:
                best_val_loss = current_best
                wait = 0
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict_noise_predictor': self.noise_predictor.state_dict(),
                        'model_state_dict_conditional': self.conditional_model.state_dict() if self.conditional_model is not None else None,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_val_loss,
                        'hyper_params_model': self.hyper_params_model,
                        'max_epoch': self.max_epoch,
                    }, self.store_path)
                    print(f"Model saved at epoch {epoch + 1}")
                except Exception as e:
                    print(f"Failed to save model: {e}")
            else:
                wait += 1
                if wait >= self.patience:
                    print("Early stopping triggered")
                    try:
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict_noise_predictor': self.noise_predictor.state_dict(),
                            'model_state_dict_conditional': self.conditional_model.state_dict() if self.conditional_model is not None else None,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': best_val_loss,
                            'hyper_params_model': self.hyper_params_model,
                            'max_epoch': self.max_epoch,
                        }, self.store_path + "_early_stop.pth")
                        print(f"Final model saved at {self.store_path}_early_stop.pth")
                    except Exception as e:
                        print(f"Failed to save final model: {e}")
                    break

        return train_losses, best_val_loss

    def validate(self):
        """Validates the DDIM model on the validation dataset.

        Computes the validation loss using the noise predictor and forward diffusion
        process, with optional conditional inputs.

        Returns
        -------
        float
            Mean validation loss across the validation dataset.
        """
        self.noise_predictor.eval()
        if self.conditional_model is not None:
            self.conditional_model.eval()

        val_losses = []
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)

                if self.conditional_model is not None:
                    y_list = y.cpu().numpy().tolist() if isinstance(y, torch.Tensor) else y
                    y_list = [str(item) for item in y_list]
                    y_encoded = self.tokenizer(
                        y_list,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    input_ids = y_encoded["input_ids"]
                    attention_mask = y_encoded["attention_mask"]
                    y_encoded = self.conditional_model(input_ids, attention_mask)
                else:
                    y_encoded = None

                noise = torch.randn_like(x).to(self.device)
                t = torch.randint(0, self.hyper_params_model.num_steps, (x.shape[0],)).to(self.device)
                assert x.device == noise.device == t.device, "Device mismatch detected"
                assert t.shape[0] == x.shape[0], "Timestep batch size mismatch"
                noisy_x = self.forward_diffusion(x, noise, t)
                p_noise = self.noise_predictor(noisy_x, t, y_encoded)
                loss = self.objective(p_noise, noise)
                val_losses.append(loss.item())

        mean_val_loss = torch.mean(torch.tensor(val_losses)).item()
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()
        return mean_val_loss