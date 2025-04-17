__version__ = "1.0.0"

"""Latent Diffusion Models (LDM) implementation.

This module provides a framework for training and sampling Latent Diffusion Models, as
described in Rombach et al. (2022, "High-Resolution Image Synthesis with Latent Diffusion
Models"). It supports diffusion in the latent space using a pre-trained compressor model,
with compatibility for DDPM, DDIM, and SDE diffusion models. Includes components for
training a noise predictor and generating images, supporting both unconditional and
conditional generation with text prompts.

Components:
- TrainLDM: Training loop with mixed precision and scheduling in the latent space.
- SampleLDM: Image generation from trained models, decoding from latent to image space.

Notes
-----
- The `hyper_params` parameter expects an external hyperparameter module (e.g.,
  HyperParamsDDPM, HyperParamsSDE) for noise schedule management.
- The `compressor_model` parameter expects a pre-trained model (e.g., AutoencoderLDM) with `encode`
  and `decode` methods for latent space conversion.
- SampleLDM supports multiple diffusion models ("ddpm", "ddim", "sde") via the `model`
  parameter, requiring compatible `reverse_diffusion` modules (e.g., ReverseDDPM,
  ReverseDDIM, ReverseSDE).

References:
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).
  High-Resolution Image Synthesis with Latent Diffusion Models.

Examples
--------
>>> from torchdiff.ddpm import HyperParamsDDPM, ForwardDDPM, ReverseDDPM  # example using DDPM
>>> from torchdiff.ldm import TrainLDM, SampleLDM
>>> from torch.diff.nets import TextEncoder, AutoencoderLDM
...
>>> hyper_params = HyperParamsDDPM(num_steps=1000, beta_start=1e-4, beta_end=0.02, beta_method="linear")
>>> forward_ddpm = ForwardDDPM(hyper_params)
>>> reverse_ddpm = ReverseDDPM(hyper_params)
>>> noise_predictor = NoisePredictor(in_channels=3, down_channels=[32, 64, 128], mid_channels=[128, 128, 128],
...                                  up_channels=[128, 64, 32], down_sampling=[True, True, True], time_embed_dim=128,
...                                  y_embed_dim=128, num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2, dropout_rate=0.1,
...                                  down_sampling_factor=2, where_y=True, y_to_all=False)
>>> text_encoder = TextEncoder(use_pretrained_model=True, model_name="bert-base-uncased", vocabulary_size=30522,
...                            num_layers=2, input_dimension=128, output_dimension=128, num_heads=4, context_length=77,
...                            dropout_rate=0.1, qkv_bias=False, scaling_value=4, epsilon=1e-5)
>>> compressor = AutoencoderLDM(in_channels=3, down_channels=[16, 32], up_channels=[32, 16], out_channels=3,
...                             latent_channels=3, dropout_rate=0.1, num_heads=4, num_groups=8, num_layers_per_block=2,
...                             total_down_sampling_factor=2, use_vq=False, num_embeddings=32, beta=1e-4)  # Pre-trained autoencoder with encode/decode methods
>>> train_ldm = TrainLDM(forward_model=forward_ddpm, hyper_params=hyper_params,
...                      noise_predictor=noise_predictor, compressor_model=compressor,
...                      optimizer=optimizer, objective=nn.MSELoss(), data_loader=data_loader,
...                      conditional_model=text_encoder, tokenizer=tokenizer)
>>> train_losses, best_val_loss = train_ldm()
>>> sampler = SampleLDM(model="ddpm", reverse_diffusion=reverse_ddpm,
...                     noise_predictor=noise_predictor, compressor_model=compressor,
...                     image_shape=(256, 256), conditional_model=text_encoder, tokenizer=tokenizer)
>>> images = sampler(conditions="A cat", normalize_output=True)

License
-------
MIT License.

Version
-------
1.0.0
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
import warnings
from tqdm import tqdm

###==================================================================================================================###

class TrainLDM(nn.Module):
    """Trainer for Latent Diffusion Models (LDM).

    Manages the training process for LDMs, optimizing a noise predictor to learn the noise
    added by the forward diffusion process in the latent space, as described in Rombach
    et al. (2022). Uses a pre-trained compressor model to encode images into a latent
    space, supports conditional training with text prompts, mixed precision, learning rate
    scheduling, early stopping, and checkpointing.

    Parameters
    ----------
    forward_model : nn.Module
        Forward diffusion module (e.g., ForwardDDPM, ForwardSDE) to add noise in the latent space.
    hyper_params : nn.Module
        Hyperparameter module (e.g., HyperParamsDDPM, HyperParamsSDE) defining the noise schedule.
    noise_predictor : nn.Module
        Model to predict noise added during the forward diffusion process.
    compressor_model : nn.Module
        Pre-trained model to encode images into the latent space and decode back (e.g., AutoencoderLDM).
    optimizer : torch.optim.Optimizer
        Optimizer for training the noise predictor and conditional model (if applicable).
    objective : callable
        Loss function to compute the difference between predicted and actual noise.
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    conditional_model : nn.Module, optional
        Model for conditional generation (e.g., TextEncoder), default None.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    max_epoch : int, optional
        Maximum number of training epochs (default: 1000).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    store_path : str, optional
        Path to save model checkpoints (default: "ldm_model.pth").
    patience : int, optional
        Number of epochs to wait for improvement before early stopping (default: 100).
    warmup_epochs : int, optional
        Number of epochs for learning rate warmup (default: 100).
    tokenizer : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    max_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    val_frequency : int, optional
        Frequency (in epochs) for validation (default: 10).

    Attributes
    ----------
    device : torch.device
        Device used for computation.
    forward_diffusion : nn.Module
        Forward diffusion module.
    hyper_params : nn.Module
        Hyperparameter module for the noise schedule.
    noise_predictor : nn.Module
        Noise prediction model.
    compressor_model : nn.Module
        Compressor model for latent space encoding/decoding.
    conditional_model : nn.Module or None
        Conditional model for text-based training, if provided.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    objective : callable
        Loss function for training.
    data_loader : torch.utils.data.DataLoader
        Training data loader.
    val_loader : torch.utils.data.DataLoader or None
        Validation data loader, if provided.
    max_epoch : int
        Maximum training epochs.
    store_path : str
        Path for saving checkpoints.
    max_length : int
        Maximum length for tokenized prompts.
    patience : int
        Patience for early stopping.
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler based on validation or training loss.
    warmup_lr_scheduler : torch.optim.lr_scheduler.LambdaLR
        Learning rate scheduler for warmup.
    tokenizer : BertTokenizer
        Tokenizer for text prompts.
    val_frequency : int
        Frequency for validation.

    Raises
    ------
    ValueError
        If the default tokenizer ("bert-base-uncased") fails to load and no tokenizer is provided.
    """
    def __init__(self, forward_model, hyper_params, noise_predictor, compressor_model, optimizer, objective, data_loader,
                 conditional_model=None, val_loader=None, max_epoch=1000, device=None, store_path=None,
                 patience=100, warmup_epochs=100, tokenizer=None, max_length=77, val_frequency=10):
        super( ).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.forward_diffusion = forward_model.to(device)
        self.hyper_params = hyper_params.to(device)
        self.noise_predictor = noise_predictor
        self.compressor_model = compressor_model
        self.conditional_model = conditional_model
        self.optimizer = optimizer
        self.objective = objective
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.store_path = store_path or "ldm_model.pth"
        self.max_length = max_length
        self.patience = patience
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience, factor=0.5)
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
    def warmup_scheduler(optimizer, warmup_epochs):
        """Creates a learning rate scheduler for warmup.

        Generates a scheduler that linearly increases the learning rate from 0 to the
        optimizer's initial value over the specified warmup epochs, then maintains it.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to apply the scheduler to.
        warmup_epochs : int, optional
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

    def forward(self):
        """Trains the LDM to predict noise added by the forward diffusion process in the latent space.

        Executes the training loop, optimizing the noise predictor and conditional model
        (if applicable) using mixed precision, gradient clipping, and learning rate
        scheduling. Uses a pre-trained compressor model to encode images into the latent
        space. Supports validation, early stopping, and checkpointing.

        Returns
        -------
        tuple
            A tuple containing:
            - train_losses: List of mean training losses per epoch (list of float).
            - best_val_loss: Best validation or training loss achieved (float).

        Notes
        -----
        - Training uses mixed precision via `torch.cuda.amp` for efficiency.
        - The compressor model is assumed pre-trained and set to evaluation mode.
        - Checkpoints are saved when the validation (or training) loss improves, and on
          early stopping.
        - Early stopping is triggered if no improvement occurs for `patience` epochs.
        """
        self.noise_predictor.train()
        self.noise_predictor.to(self.device)
        if self.conditional_model is not None:
            self.conditional_model.train()
            self.conditional_model.to(self.device)
        if self.compressor_model is not None:
            self.compressor_model.eval() # the model is already trained
            self.compressor_model.to(self.device)

        scaler = GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0
        for epoch in range(self.max_epoch):
            train_losses_ = []
            for x, y in tqdm(self.data_loader):
                x = x.to(self.device)
                with torch.no_grad():
                    x, _ = self.compressor_model.encode(x)
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
                    t = torch.randint(0, self.hyper_params.num_steps, (x.shape[0],)).to(self.device)
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

            if self.hyper_params.trainable_beta:
                self.hyper_params.constrain_betas() # constrains trainable betas

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
                        'hyper_params_model': self.hyper_params,
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
                            'hyper_params_model': self.hyper_params,
                            'max_epoch': self.max_epoch,
                        }, self.store_path + "_early_stop.pth")
                        print(f"Final model saved at {self.store_path}_early_stop.pth")
                    except Exception as e:
                        print(f"Failed to save final model: {e}")
                    break

        return train_losses, best_val_loss

    def validate(self):
        """Validates the LDM on the validation dataset.

        Computes the validation loss using the noise predictor and forward diffusion
        process in the latent space, with optional conditional inputs.

        Returns
        -------
        float
            Mean validation loss across the validation dataset.

        Notes
        -----
        - Validation is performed with `torch.no_grad()` for efficiency.
        - The compressor model is used to encode validation data into the latent space.
        - The noise predictor and conditional model (if applicable) are set to evaluation
          mode during validation and restored to training mode afterward.
        """
        self.noise_predictor.eval()
        if self.conditional_model is not None:
            self.conditional_model.eval()

        val_losses = []
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                with torch.no_grad():
                    x, _ = self.compressor_model.encode(x)
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
                t = torch.randint(0, self.hyper_params.num_steps, (x.shape[0],)).to(self.device)
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

###==================================================================================================================###

class SampleLDM(nn.Module):
    """Sampler for generating images using Latent Diffusion Models (LDM).

    Generates images by iteratively denoising random noise in the latent space using a
    reverse diffusion process, decoding the result back to the image space with a
    pre-trained compressor, as described in Rombach et al. (2022). Supports DDPM, DDIM,
    and SDE diffusion models, as well as conditional generation with text prompts.

    Parameters
    ----------
    model : str
        Diffusion model type. Supported: "ddpm", "ddim", "sde".
    reverse_diffusion : nn.Module
        Reverse diffusion module (e.g., ReverseDDPM, ReverseDDIM, ReverseSDE).
    noise_predictor : nn.Module
        Model to predict noise added during the forward diffusion process.
    compressor_model : nn.Module
        Pre-trained model to encode/decode between image and latent spaces (e.g., AutoencoderLDM).
    image_shape : tuple
        Shape of generated images as (height, width).
    conditional_model : nn.Module, optional
        Model for conditional generation (e.g., TextEncoder), default None.
    tokenizer : str or BertTokenizer, optional
        Tokenizer for processing text prompts, default "bert-base-uncased".
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for latent representations (default: 3).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    max_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    output_range : tuple, optional
        Range for clamping generated images (min, max), default (-1, 1).

    Attributes
    ----------
    device : torch.device
        Device used for computation.
    model : str
        Diffusion model type ("ddpm", "ddim", "sde").
    noise_predictor : nn.Module
        Noise prediction model.
    reverse : nn.Module
        Reverse diffusion module.
    compressor : nn.Module
        Compressor model for latent space encoding/decoding.
    conditional_model : nn.Module or None
        Conditional model for text-based generation, if provided.
    tokenizer : BertTokenizer
        Tokenizer for text prompts.
    in_channels : int
        Number of input channels for latent representations.
    image_shape : tuple
        Shape of generated images (height, width).
    batch_size : int
        Batch size for generation.
    max_length : int
        Maximum length for tokenized prompts.
    output_range : tuple
        Range for clamping generated images.

    Raises
    ------
    ValueError
        If `image_shape` is not a tuple of two positive integers, `batch_size` is not
        positive, `in_channels` is not positive, or `output_range` is not a tuple
        (min, max) with min < max.
    """
    def __init__(self, model, reverse_diffusion, noise_predictor, compressor_model, image_shape, conditional_model=None,
                 tokenizer="bert-base-uncased", batch_size=1, in_channels=3, device=None, max_length=77, output_range=(-1, 1)):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.noise_predictor = noise_predictor.to(self.device)
        self.reverse = reverse_diffusion.to(self.device)
        self.compressor = compressor_model.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.in_channels = in_channels
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_range = output_range

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not isinstance(output_range, (tuple, list)) or len(output_range) != 2 or output_range[0] >= output_range[1]:
            raise ValueError("output_range must be a tuple (min, max) with min < max")

    def tokenize(self, prompts):
        """Tokenizes text prompts for conditional generation.

        Converts input prompts into tokenized tensors using the specified tokenizer.

        Parameters
        ----------
        prompts : str or list
            Text prompt(s) for conditional generation. Can be a single string or a list
            of strings.

        Returns
        -------
        tuple
            A tuple containing:
            - input_ids: Tokenized input IDs (torch.Tensor, shape (batch_size, max_length)).
            - attention_mask: Attention mask for tokenized inputs (torch.Tensor, same shape).

        Raises
        ------
        TypeError
            If `prompts` is not a string or a list of strings.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        elif not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise TypeError("prompts must be a string or list of strings")

        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def forward(self, conditions=None, normalize_output=True):
        """Generates images using the reverse diffusion process in the latent space.

        Iteratively denoises random noise in the latent space using the specified reverse
        diffusion model (DDPM, DDIM, or SDE), then decodes the result to the image space
        with the compressor model. Supports conditional generation with text prompts.

        Parameters
        ----------
        conditions : str or list, optional
            Text prompt(s) for conditional generation, default None.
        normalize_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).

        Returns
        -------
        torch.Tensor
            Generated images, shape (batch_size, channels, height, width).
            If `normalize_output` is True, images are normalized to [0, 1]; otherwise,
            they are clamped to `output_range`.

        Raises
        ------
        ValueError
            If `conditions` is provided but no conditional model is specified, if a
            conditional model is specified but `conditions` is None, or if `model` is not
            one of "ddpm", "ddim", "sde".

        Notes
        -----
        - Sampling is performed with `torch.no_grad()` for efficiency.
        - The noise predictor, reverse diffusion, compressor, and conditional model
          (if applicable) are set to evaluation mode during sampling.
        - For DDIM, uses the subsampled tau schedule (`tau_num_steps`); for DDPM/SDE,
          uses the full number of steps (`num_steps`).
        - The compressor model is assumed to have `encode` and `decode` methods for
          latent space conversion.
        """
        if conditions is not None and self.conditional_model is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conditions is None and self.conditional_model is not None:
            raise ValueError("Conditions must be provided for conditional model")

        noisy_samples = torch.randn(self.batch_size, self.in_channels, self.image_shape[0], self.image_shape[1]).to(self.device)

        self.noise_predictor.eval()
        self.compressor.eval()
        self.reverse.eval()
        if self.conditional_model:
            self.conditional_model.eval()

        with torch.no_grad():
            xt = noisy_samples
            xt, _ = self.compressor.encode(xt)

            if self.model == "ddim":
                num_steps = self.reverse.hyper_params.tau_num_steps
            elif self.model == "ddpm" or self.model == "sde":
                num_steps = self.reverse.hyper_params.num_steps
            else:
                raise ValueError(f"Unknown model: {self.model}. Supported: ddpm, ddim, sde")

            for t in reversed(range(num_steps)):
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)
                prev_time_steps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device, dtype=torch.long)

                if self.model == "sde":
                    noise = torch.randn_like(xt) if getattr(self.reverse, "method", None) != "ode" else None

                if self.conditional_model is not None and conditions is not None:
                    input_ids, attention_masks = self.tokenize(conditions)
                    key_padding_mask = (attention_masks == 0)
                    y = self.conditional_model(input_ids, key_padding_mask)
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)

                if self.model == "sde":
                    xt = self.reverse(xt, noise, predicted_noise, time_steps)
                elif self.model == "ddim":
                    xt, _ = self.reverse(xt, predicted_noise, time_steps, prev_time_steps)
                elif self.model == "ddpm":
                    xt = self.reverse(xt, predicted_noise, time_steps)
                else:
                    raise ValueError(f"Unknown model: {self.model}. Supported: ddpm, ddim, sde")

            x = self.compressor.decode(xt)
            generated_imgs = torch.clamp(x, min=self.output_range[0], max=self.output_range[1])
            if normalize_output:
                generated_imgs = (generated_imgs - self.output_range[0]) / (self.output_range[1] - self.output_range[0])

        return generated_imgs

    def to(self, device):
        """Moves the module and its components to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device for computation.

        Returns
        -------
        self
            The module moved to the specified device.

        Notes
        -----
        - Moves `noise_predictor`, `reverse`, `compressor`, and `conditional_model`
          (if applicable) to the specified device.
        """
        self.device = device
        self.noise_predictor.to(device)
        self.reverse.to(device)
        self.compressor.to(device)
        if self.conditional_model:
            self.conditional_model.to(device)
        return super().to(device)

###==================================================================================================================###