__version__ = "1.0.0"

"""Latent Diffusion Models (LDM) implementation.

This module provides a framework for training and sampling Latent Diffusion Models, as
described in Rombach et al. (2022, "High-Resolution Image Synthesis with Latent Diffusion
Models"). It supports diffusion in the latent space using a variational autoencoder
(compressor model) and includes utilities for training the autoencoder and noise predictor.
The framework is compatible with DDPM, DDIM, and SDE diffusion models, supporting both
unconditional and conditional generation with text prompts.

Components:
- AutoencoderLDM: Variational autoencoder for compressing images to latent space and
  decoding back to image space.
- TrainAE: Trainer for the AutoencoderLDM, optimizing reconstruction and regularization
  losses.
- TrainLDM: Training loop with mixed precision and scheduling for the noise predictor in
  latent space.
- SampleLDM: Image generation from trained models, decoding from latent to image space.

Notes
-----
- The `hyper_params` parameter expects an external hyperparameter module (e.g.,
  HyperParamsDDPM, HyperParamsSDE) for noise schedule management.
- AutoencoderLDM serves as the `compressor_model` in TrainLDM and SampleLDM, providing
  `encode` and `decode` methods for latent space conversion. It supports KL-divergence or
  vector quantization (VQ) regularization, using internal components (DownBlock, UpBlock,
  Conv3, DownSampling, UpSampling, Attention, VectorQuantizer).
- TrainAE trains AutoencoderLDM, optimizing reconstruction (MSE), regularization (KL or
  VQ), and optional perceptual (LPIPS) losses, with metrics (MSE, PSNR, SSIM, FID), KL
  warmup, early stopping, and learning rate scheduling.
- The `noise_predictor` parameter expects a model (e.g., UNet from utils module) operating
  on latent representations, predicting noise in the diffusion process.
- The `conditional_model` parameter expects a text encoder (e.g., TextEncoder from utils
  module) for conditional generation, with tokenized text inputs compatible with its
  attention mask convention (0 for padding in custom transformer, 1 for BERT).
- SampleLDM supports multiple diffusion models ("ddpm", "ddim", "sde") via the `model`
  parameter, requiring compatible `reverse_diffusion` modules (e.g., ReverseDDPM,
  ReverseDDIM, ReverseSDE).
- Ensure `image_shape` in SampleLDM matches the input resolution expected by the
  compressor and noise predictor.

References
----------
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).
High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.

Examples
--------
>>> from torchdiff.ddpm import HyperParamsDDPM, ForwardDDPM, ReverseDDPM
>>> from torchdiff.ldm import AutoencoderLDM, TrainAE, TrainLDM, SampleLDM
>>> from torchdiff.utils import TextEncoder, UNet
>>> from torch.optim import Adam
>>> import torch.nn as nn
>>> hyper_params = HyperParamsDDPM(num_steps=1000, beta_start=1e-4, beta_end=0.02, beta_method="linear")
>>> forward_ddpm = ForwardDDPM(hyper_params)
>>> reverse_ddpm = ReverseDDPM(hyper_params)
>>> text_encoder = TextEncoder(use_pretrained_model=True, model_name="bert-base-uncased")
>>> compressor = AutoencoderLDM(in_channels=3, down_channels=[16, 32], up_channels=[32, 16], out_channels=3,
...                             dropout_rate=0.1, num_heads=4, num_groups=8, num_layers_per_block=2,
...                             total_down_sampling_factor=2, latent_channels=3, num_embeddings=32, use_vq=False,
...                             beta=1e-4)
>>> optimizer_ae = Adam(compressor.parameters(), lr=1e-4)
>>> train_ae = TrainAE(model=compressor, optimizer=optimizer_ae, data_loader=data_loader, val_loader=val_loader,
...                    max_epoch=100, device='cuda', save_path='vlc_model.pth')
>>> ae_losses, best_ae_loss = train_ae.train()
>>> noise_predictor = UNet(in_channels=3, attention=True)  # Placeholder until implementation provided
>>> optimizer_ldm = Adam(noise_predictor.parameters(), lr=1e-4)
>>> train_ldm = TrainLDM(forward_model=forward_ddpm, hyper_params=hyper_params,
...                      noise_predictor=noise_predictor, compressor_model=compressor,
...                      optimizer=optimizer_ldm, objective=nn.MSELoss(), data_loader=data_loader,
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
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
import warnings
from tqdm import tqdm
import lpips
from pytorch_fid import fid_score
import os
import shutil
from torchvision.utils import save_image

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

class AutoencoderLDM(nn.Module):
    """Variational autoencoder for latent space compression in Latent Diffusion Models.

    Encodes images into a latent space and decodes them back to the image space, used as
    the `compressor_model` in LDM’s `TrainLDM` and `SampleLDM`. Supports KL-divergence
    or vector quantization (VQ) regularization for the latent representation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB images).
    down_channels : list
        List of channel sizes for encoder downsampling blocks (e.g., [32, 64, 128, 256]).
    up_channels : list
        List of channel sizes for decoder upsampling blocks (e.g., [256, 128, 64, 16]).
    out_channels : int
        Number of output channels, typically equal to `in_channels`.
    dropout_rate : float
        Dropout rate for regularization in convolutional and attention layers.
    num_heads : int
        Number of attention heads in self-attention layers.
    num_groups : int
        Number of groups for group normalization in attention layers.
    num_layers_per_block : int
        Number of convolutional layers in each downsampling and upsampling block.
    total_down_sampling_factor : int
        Total downsampling factor across the encoder (e.g., 8 for 8x reduction).
    latent_channels : int
        Number of channels in the latent representation for diffusion models.
    num_embeddings : int
        Number of discrete embeddings in the VQ codebook (if `use_vq=True`).
    use_vq : bool, optional
        If True, uses vector quantization (VQ) regularization; otherwise, uses
        KL-divergence (default: False).
    beta : float, optional
        Weight for KL-divergence loss (if `use_vq=False`) (default: 1.0).

    Attributes
    ----------
    use_vq : bool
        Whether VQ regularization is used.
    beta : float
        Fixed weight for KL-divergence loss.
    current_beta : float
        Current weight for KL-divergence loss (modifiable during training).
    down_sampling_factor : int
        Downsampling factor per block, derived from `total_down_sampling_factor`.
    conv1 : torch.nn.Conv2d
        Initial convolutional layer for encoding.
    down_blocks : torch.nn.ModuleList
        List of DownBlock modules for encoder downsampling.
    attention1 : Attention
        Self-attention layer after encoder downsampling.
    vq_layer : VectorQuantizer or None
        Vector quantization layer (if `use_vq=True`).
    conv_mu : torch.nn.Conv2d or None
        Convolutional layer for mean of latent distribution (if `use_vq=False`).
    conv_logvar : torch.nn.Conv2d or None
        Convolutional layer for log-variance of latent distribution (if `use_vq=False`).
    quant_conv : torch.nn.Conv2d
        Convolutional layer to project latent representation to `latent_channels`.
    conv2 : torch.nn.Conv2d
        Initial convolutional layer for decoding.
    attention2 : Attention
        Self-attention layer after decoder’s initial convolution.
    up_blocks : torch.nn.ModuleList
        List of UpBlock modules for decoder upsampling.
    conv3 : Conv3
        Final convolutional layer for output reconstruction.

    Raises
    ------
    AssertionError
        If `in_channels` does not equal `out_channels`.

    Notes
    -----
    - The encoder downsamples images using `DownBlock` modules, followed by self-attention
      and latent projection (VQ or KL-based).
    - The decoder upsamples the latent representation using `UpBlock` modules, with
      self-attention and final convolution.
    - The `down_sampling_factor` is computed as `total_down_sampling_factor` raised to
      the power of `1 / (len(down_channels) - 1)`, applied per downsampling block.
    - The latent representation has `latent_channels` channels, suitable for LDM’s
      diffusion process.
    """
    def __init__(
            self,
            in_channels,
            down_channels,
            up_channels,
            out_channels,
            dropout_rate,
            num_heads,
            num_groups,
            num_layers_per_block,
            total_down_sampling_factor,
            latent_channels,
            num_embeddings,
            use_vq=False,
            beta=1.0

    ):
        super().__init__()
        assert in_channels == out_channels, "Input and output channels must match for auto-encoding"
        self.use_vq = use_vq
        self.beta = beta
        self.current_beta = beta
        num_down_blocks = len(down_channels) - 1
        self.down_sampling_factor = int(total_down_sampling_factor ** (1 / num_down_blocks))

        # encoder
        self.conv1 = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels=down_channels[i],
                out_channels=down_channels[i + 1],
                num_layers=num_layers_per_block,
                down_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate
            ) for i in range(num_down_blocks)
        ])
        self.attention1 = Attention(down_channels[-1], num_heads, num_groups, dropout_rate)

        # latent projection
        if use_vq:
            self.vq_layer = VectorQuantizer(num_embeddings, down_channels[-1])
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)
        else:
            self.conv_mu = nn.Conv2d(down_channels[-1], down_channels[-1], kernel_size=3, padding=1)
            self.conv_logvar = nn.Conv2d(down_channels[-1], down_channels[-1], kernel_size=3, padding=1)
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)

        # decoder
        self.conv2 = nn.Conv2d(latent_channels, up_channels[0], kernel_size=3, padding=1)
        self.attention2 = Attention(up_channels[0], num_heads, num_groups, dropout_rate)
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=up_channels[i],
                out_channels=up_channels[i + 1],
                num_layers=num_layers_per_block,
                up_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate
            ) for i in range(len(up_channels) - 1)
        ])
        self.conv3 = Conv3(up_channels[-1], out_channels, dropout_rate)

    def reparameterize(self, mu, logvar):
        """Applies reparameterization trick for variational autoencoding.

        Samples from a Gaussian distribution using the mean and log-variance to enable
        differentiable training.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution, shape (batch_size, channels, height, width).
        logvar : torch.Tensor
            Log-variance of the latent distribution, same shape as `mu`.

        Returns
        -------
        torch.Tensor
            Sampled latent representation, same shape as `mu`.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encodes images into a latent representation.

        Processes input images through the encoder, applying convolutions, downsampling,
        self-attention, and latent projection (VQ or KL-based).

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (batch_size, in_channels, height, width).

        Returns
        -------
        tuple
            A tuple containing:
            - z: Latent representation, shape (batch_size, latent_channels,
              height/down_sampling_factor, width/down_sampling_factor).
            - reg_loss: Regularization loss (VQ loss if `use_vq=True`, KL-divergence
              loss if `use_vq=False`).

        Notes
        -----
        - The VQ loss is computed by `VectorQuantizer` if `use_vq=True`.
        - The KL-divergence loss is normalized by batch size and latent size, weighted
          by `current_beta`.
        """
        x = self.conv1(x)
        for block in self.down_blocks:
            x = block(x)
        res_x = x
        x = self.attention1(x)
        x = x + res_x
        if self.use_vq:
            z, vq_loss = self.vq_layer(x)
            z = self.quant_conv(z)
            return z, vq_loss
        else:
            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            z = self.reparameterize(mu, logvar)
            z = self.quant_conv(z)
            kl_unnormalized = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            batch_size = x.size(0)
            latent_size = torch.prod(torch.tensor(mu.shape[1:])).item()
            kl_loss = kl_unnormalized / (batch_size * latent_size) * self.current_beta
            return z, kl_loss

    def decode(self, z):
        """Decodes latent representations back to images.

        Processes latent representations through the decoder, applying convolutions,
        self-attention, upsampling, and final reconstruction.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation, shape (batch_size, latent_channels,
            height/down_sampling_factor, width/down_sampling_factor).

        Returns
        -------
        torch.Tensor
            Reconstructed images, shape (batch_size, out_channels, height, width).
        """
        x = self.conv2(z)
        res_x = x
        x = self.attention2(x)
        x = x + res_x
        for block in self.up_blocks:
            x = block(x)
        x = self.conv3(x)
        return x

    def forward(self, x):
        """Encodes images to latent space and decodes them, computing reconstruction and regularization losses.

        Performs a full autoencoding pass, encoding images to the latent space, decoding
        them back, and calculating MSE reconstruction loss and regularization loss (VQ
        or KL-based).

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (batch_size, in_channels, height, width).

        Returns
        -------
        tuple
            A tuple containing:
            - x_hat: Reconstructed images, shape (batch_size, out_channels, height,
              width).
            - total_loss: Sum of reconstruction (MSE) and regularization losses.
            - reg_loss: Regularization loss (VQ or KL-divergence).
            - z: Latent representation, shape (batch_size, latent_channels,
              height/down_sampling_factor, width/down_sampling_factor).

        Notes
        -----
        - The reconstruction loss is computed as the mean squared error between `x_hat`
          and `x`.
        - The regularization loss depends on `use_vq` (VQ loss or KL-divergence).
        """
        z, reg_loss = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_loss + reg_loss
        return x_hat, total_loss, reg_loss, z

###==================================================================================================================###

class VectorQuantizer(nn.Module):
    """Vector quantization layer for discretizing latent representations.

    Quantizes input latent vectors to the nearest embedding in a learned codebook,
    used in `AutoencoderLDM` when `use_vq=True` to enable discrete latent spaces for
    Latent Diffusion Models. Computes commitment and codebook losses to train the
    codebook embeddings.

    Parameters
    ----------
    num_embeddings : int
        Number of discrete embeddings in the codebook.
    embedding_dim : int
        Dimensionality of each embedding vector (matches input channel dimension).
    commitment_cost : float, optional
        Weight for the commitment loss, encouraging inputs to be close to quantized
        values (default: 0.25).

    Attributes
    ----------
    embedding_dim : int
        Dimensionality of embedding vectors.
    num_embeddings : int
        Number of embeddings in the codebook.
    commitment_cost : float
        Weight for commitment loss.
    embedding : torch.nn.Embedding
        Embedding layer containing the codebook, shape (num_embeddings,
        embedding_dim).

    Notes
    -----
    - The codebook embeddings are initialized uniformly in the range
      [-1/num_embeddings, 1/num_embeddings].
    - The forward pass flattens input latents, computes Euclidean distances to
      codebook embeddings, and selects the nearest embedding for quantization.
    - The commitment loss encourages input latents to be close to their quantized
      versions, while the codebook loss updates embeddings to match inputs.
    - A straight-through estimator is used to pass gradients from the quantized output
      to the input.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        """Quantizes latent representations to the nearest codebook embedding.

        Computes the closest embedding for each input vector, applies quantization,
        and calculates commitment and codebook losses for training.

        Parameters
        ----------
        z : torch.Tensor
            Input latent representation, shape (batch_size, embedding_dim, height,
            width).

        Returns
        -------
        tuple
            A tuple containing:
            - quantized: Quantized latent representation, same shape as `z`.
            - vq_loss: Sum of commitment and codebook losses.

        Raises
        ------
        AssertionError
            If the channel dimension of `z` does not match `embedding_dim`.

        Notes
        -----
        - The input is flattened to (batch_size * height * width, embedding_dim) for
          distance computation.
        - Euclidean distances are computed efficiently using vectorized operations.
        - The commitment loss is scaled by `commitment_cost`, and the total VQ loss
          combines commitment and codebook losses.
        """
        z = z.contiguous()
        assert z.size(1) == self.embedding_dim, f"Expected channel dim {self.embedding_dim}, got {z.size(1)}"
        z_flattened = z.reshape(-1, self.embedding_dim)
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float().squeeze(1)
        quantized = torch.matmul(encodings, self.embedding.weight).view_as(z)
        commitment_loss = self.commitment_cost * torch.mean((z.detach() - quantized) ** 2)
        codebook_loss = torch.mean((z - quantized.detach()) ** 2)
        quantized = z + (quantized - z).detach()
        return quantized, commitment_loss + codebook_loss

###==================================================================================================================###

class DownBlock(nn.Module):
    """Downsampling block for the encoder in AutoencoderLDM.

    Applies multiple convolutional layers with residual connections followed by
    downsampling to reduce spatial dimensions in the encoder of the variational
    autoencoder used in Latent Diffusion Models.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for convolutional layers.
    num_layers : int
        Number of convolutional layer pairs (Conv3) per block.
    down_sampling_factor : int
        Factor by which to downsample spatial dimensions.
    dropout_rate : float
        Dropout rate for Conv3 layers.

    Attributes
    ----------
    num_layers : int
        Number of convolutional layer pairs.
    conv1 : torch.nn.ModuleList
        List of Conv3 layers for the first convolution in each pair.
    conv2 : torch.nn.ModuleList
        List of Conv3 layers for the second convolution in each pair.
    down_sampling : DownSampling
        Downsampling module to reduce spatial dimensions.
    resnet : torch.nn.ModuleList
        List of 1x1 convolutional layers for residual connections.

    Notes
    -----
    - Each layer pair consists of two Conv3 modules with a residual connection using a
      1x1 convolution to match dimensions.
    - The downsampling is applied after all convolutional layers, reducing spatial
      dimensions by `down_sampling_factor`.
    """
    def __init__(self, in_channels, out_channels, num_layers, down_sampling_factor, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])

        self.down_sampling = DownSampling(
            in_channels=out_channels,
            out_channels=out_channels,
            down_sampling_factor=down_sampling_factor
        )
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers)

        ])

    def forward(self, x):
        """Processes input through convolutional layers and downsampling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch_size, out_channels,
            height/down_sampling_factor, width/down_sampling_factor).
        """
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
        output = self.down_sampling(output)
        return output

###==================================================================================================================###

class Conv3(nn.Module):
    """Convolutional layer with group normalization, SiLU activation, and dropout.

    Used in DownBlock and UpBlock of AutoencoderLDM for feature extraction and
    transformation in the encoder and decoder.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : float
        Dropout rate for regularization.

    Attributes
    ----------
    group_norm : torch.nn.GroupNorm
        Group normalization with 8 groups.
    activation : torch.nn.SiLU
        SiLU (Swish) activation function.
    conv : torch.nn.Conv2d
        3x3 convolutional layer with padding to maintain spatial dimensions.
    dropout : torch.nn.Dropout
        Dropout layer for regularization.

    Notes
    -----
    - The layer applies group normalization, SiLU activation, dropout, and a 3x3
      convolution in sequence.
    - Spatial dimensions are preserved due to padding=1 in the convolution.
    """
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.activation = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """Processes input through group normalization, activation, dropout, and convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch_size, out_channels, height, width).
        """
        x = self.group_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

###==================================================================================================================###

class DownSampling(nn.Module):
    """Downsampling module for reducing spatial dimensions in AutoencoderLDM’s encoder.

    Combines convolutional downsampling and max pooling, concatenating their outputs
    to preserve feature information during downsampling in DownBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (sum of conv and pool paths).
    down_sampling_factor : int
        Factor by which to downsample spatial dimensions.

    Attributes
    ----------
    down_sampling_factor : int
        Downsampling factor.
    conv : torch.nn.Sequential
        Convolutional path with 1x1 and 3x3 convolutions, outputting out_channels/2.
    pool : torch.nn.Sequential
        Max pooling path with 1x1 convolution, outputting out_channels/2.

    Notes
    -----
    - The module splits the output channels evenly between convolutional and pooling
      paths, concatenating them along the channel dimension.
    - The convolutional path uses a stride equal to `down_sampling_factor`, while the
      pooling path uses max pooling with the same factor.
    """
    def __init__(self, in_channels, out_channels, down_sampling_factor):
        super().__init__()
        self.down_sampling_factor = down_sampling_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                      kernel_size=3, stride=down_sampling_factor, padding=1)
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=down_sampling_factor, stride=down_sampling_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                      kernel_size=1, stride=1, padding=0)
        )

    def forward(self, batch):
        """Downsamples input by combining convolutional and pooling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Downsampled tensor, shape (batch_size, out_channels,
            height/down_sampling_factor, width/down_sampling_factor).
        """
        return torch.cat(tensors=[self.conv(batch), self.pool(batch)], dim=1)

###==================================================================================================================###

class Attention(nn.Module):
    """Self-attention module for feature enhancement in AutoencoderLDM.

    Applies multi-head self-attention to enhance features in the encoder and decoder,
    used after downsampling (in DownBlock) and before upsampling (in UpBlock).

    Parameters
    ----------
    num_channels : int
        Number of input and output channels (embedding dimension for attention).
    num_heads : int
        Number of attention heads.
    num_groups : int
        Number of groups for group normalization.
    dropout_rate : float
        Dropout rate for attention outputs.

    Attributes
    ----------
    group_norm : torch.nn.GroupNorm
        Group normalization before attention.
    attention : torch.nn.MultiheadAttention
        Multi-head self-attention with `batch_first=True`.
    dropout : torch.nn.Dropout
        Dropout layer for regularization.

    Notes
    -----
    - The input is reshaped to (batch_size, height * width, num_channels) for
      attention processing, then restored to (batch_size, num_channels, height, width).
    - Group normalization is applied before attention to stabilize training.
    """
    def __init__(self, num_channels, num_heads, num_groups, dropout_rate):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """Applies self-attention to input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, num_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor, same shape as input.
        """
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h * w)
        x = self.group_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x

###==================================================================================================================###

class UpBlock(nn.Module):
    """Upsampling block for the decoder in AutoencoderLDM.

    Applies upsampling followed by multiple convolutional layers with residual
    connections to increase spatial dimensions in the decoder of the variational
    autoencoder used in Latent Diffusion Models.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for convolutional layers.
    num_layers : int
        Number of convolutional layer pairs (Conv3) per block.
    up_sampling_factor : int
        Factor by which to upsample spatial dimensions.
    dropout_rate : float
        Dropout rate for Conv3 layers.

    Attributes
    ----------
    num_layers : int
        Number of convolutional layer pairs.
    up_sampling : UpSampling
        Upsampling module to increase spatial dimensions.
    conv1 : torch.nn.ModuleList
        List of Conv3 layers for the first convolution in each pair.
    conv2 : torch.nn.ModuleList
        List of Conv3 layers for the second convolution in each pair.
    resnet : torch.nn.ModuleList
        List of 1x1 convolutional layers for residual connections.

    Notes
    -----
    - Upsampling is applied first, followed by convolutional layer pairs with residual
      connections using 1x1 convolutions.
    - Each layer pair consists of two Conv3 modules.
    """
    def __init__(self, in_channels, out_channels, num_layers, up_sampling_factor, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        effective_in_channels = in_channels

        self.up_sampling = UpSampling(
            in_channels=in_channels,
            out_channels=in_channels,
            up_sampling_factor=up_sampling_factor
        )

        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=effective_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=effective_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(self.num_layers)
        ])

    def forward(self, x):
        """Processes input through upsampling and convolutional layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch_size, out_channels,
            height * up_sampling_factor, width * up_sampling_factor).
        """
        x = self.up_sampling(x)
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
        return output

###==================================================================================================================###

class UpSampling(nn.Module):
    """Upsampling module for increasing spatial dimensions in AutoencoderLDM’s decoder.

    Combines transposed convolution and nearest-neighbor upsampling, concatenating
    their outputs to preserve feature information during upsampling in UpBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (sum of conv and upsample paths).
    up_sampling_factor : int
        Factor by which to upsample spatial dimensions.

    Attributes
    ----------
    up_sampling_factor : int
        Upsampling factor.
    conv : torch.nn.Sequential
        Transposed convolutional path, outputting out_channels/2.
    up_sample : torch.nn.Sequential
        Nearest-neighbor upsampling path with 1x1 convolution, outputting
        out_channels/2.

    Notes
    -----
    - The module splits the output channels evenly between transposed convolution and
      upsampling paths, concatenating them along the channel dimension.
    - If the spatial dimensions of the two paths differ, the upsampling path is
      interpolated to match the convolutional path’s size.
    """
    def __init__(self, in_channels, out_channels, up_sampling_factor):
        super().__init__()
        half_out_channels = out_channels // 2
        self.up_sampling_factor = up_sampling_factor
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=half_out_channels,
                kernel_size=3,
                stride=up_sampling_factor,
                padding=1,
                output_padding=up_sampling_factor - 1
            ),
            nn.Conv2d(
                in_channels=half_out_channels,
                out_channels=half_out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=up_sampling_factor, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=half_out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, batch):
        """Upsamples input by combining transposed convolution and upsampling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Upsampled tensor, shape (batch_size, out_channels,
            height * up_sampling_factor, width * up_sampling_factor).

        Notes
        -----
        - Interpolation is applied if the spatial dimensions of the convolutional and
          upsampling paths differ, using nearest-neighbor mode.
        """
        conv_output = self.conv(batch)
        up_sample_output = self.up_sample(batch)
        if conv_output.shape[2:] != up_sample_output.shape[2:]:
            _, _, h, w = conv_output.shape
            up_sample_output = torch.nn.functional.interpolate(
                up_sample_output,
                size=(h, w),
                mode='nearest'
            )
        return torch.cat(tensors=[conv_output, up_sample_output], dim=1)

###==================================================================================================================###

class TrainAE:
    """Trainer for the variational autoencoder in Latent Diffusion Models.

    Manages training of the `AutoencoderLDM` compressor model, optimizing for
    reconstruction and regularization losses (KL-divergence or VQ), with optional
    perceptual loss, metrics (MSE, PSNR, SSIM, FID), KL warmup, early stopping, and
    learning rate scheduling.

    Parameters
    ----------
    model : AutoencoderLDM
        The variational autoencoder model to train (compressor model for LDM).
    optimizer : torch.optim.Optimizer
        Optimizer for training the model.
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data, yielding (images, labels) batches.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data (default: None).
    max_epoch : int, optional
        Maximum number of training epochs (default: 100).
    device : str, optional
        Device for training (e.g., 'cuda', 'cpu') (default: 'cuda').
    save_path : str, optional
        File path to save the best model checkpoint (default: 'vlc_model.pth').
    checkpoint : int, optional
        Frequency (in epochs) to save model checkpoints (default: 10).
    kl_warmup_epochs : int, optional
        Number of epochs for KL-divergence loss warmup (default: 10).
    patience : int, optional
        Number of epochs to wait for early stopping if validation loss does not
        improve (default: 10).
    per_loss : bool, optional
        Whether to include perceptual loss using LPIPS (default: True).
    metrics : bool, optional
        Whether to compute MSE, PSNR, and SSIM metrics (default: True).
    fid : bool, optional
        Whether to compute FID score (default: False).
    perceptual_weight : float, optional
        Weight for the perceptual loss term (default: 0.1).

    Attributes
    ----------
    model : AutoencoderLDM
        The autoencoder model being trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    data_loader : torch.utils.data.DataLoader
        Training DataLoader.
    val_loader : torch.utils.data.DataLoader or None
        Validation DataLoader, if provided.
    max_epoch : int
        Maximum training epochs.
    device : str
        Training device.
    save_path : str
        Path for saving model checkpoints.
    checkpoint : int
        Epoch frequency for saving checkpoints.
    kl_warmup_epochs : int
        Epochs for KL loss warmup.
    patience : int
        Epochs for early stopping patience.
    per_loss : bool
        Flag for perceptual loss computation.
    metrics : bool
        Flag for MSE, PSNR, SSIM computation.
    fid : bool
        Flag for FID computation.
    perceptual_loss : lpips.LPIPS
        LPIPS model for perceptual loss (VGG backbone).
    perceptual_weight : float
        Weight for perceptual loss.
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler based on validation loss.
    temp_dir_real : str
        Temporary directory for real images during FID computation.
    temp_dir_fake : str
        Temporary directory for fake (reconstructed) images during FID computation.

    Notes
    -----
    - The total loss includes reconstruction (MSE), regularization (KL or VQ), and
      optional perceptual (LPIPS) losses.
    - KL warmup linearly increases the KL loss weight (`model.current_beta`) from 0 to
      `model.beta` over `kl_warmup_epochs` when `model.use_vq=False`.
    - Metrics (MSE, PSNR, SSIM) and FID are computed if enabled, with FID requiring
      temporary disk storage for images.
    - Early stopping is based on validation loss (or training loss if `val_loader` is
      None), and the learning rate is reduced if validation loss plateaus.
    - The model is saved when the best validation (or training) loss is achieved.
    """
    def __init__(self, model, optimizer, data_loader, val_loader=None, max_epoch=100,
                 device="cuda", save_path="vlc_model.pth", checkpoint=10, kl_warmup_epochs=10,
                 patience=10, per_loss=True, metrics=True, fid=False, perceptual_weight=0.1):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.device = device
        self.save_path = save_path
        self.checkpoint = checkpoint
        self.kl_warmup_epochs = kl_warmup_epochs
        self.patience = patience
        self.per_loss = per_loss
        self.metrics = metrics
        self.fid = fid
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(device)
        self.perceptual_weight = perceptual_weight
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.temp_dir_real = "temp_real"
        self.temp_dir_fake = "temp_fake"

    def compute_metrics(self, x, x_hat):
        """Computes image quality metrics (MSE, PSNR, SSIM) for reconstructed images.

        Parameters
        ----------
        x : torch.Tensor
            Ground truth images, shape (batch_size, channels, height, width).
        x_hat : torch.Tensor
            Reconstructed images, same shape as `x`.

        Returns
        -------
        dict
            Dictionary containing:
            - mse: Mean squared error (float).
            - psnr: Peak signal-to-noise ratio (float).
            - ssim: Structural similarity index (float, mean over batch).
        """
        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse)
        c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
        mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(x_hat, kernel_size=3, stride=1, padding=1)
        mu_xy = mu_x * mu_y
        sigma_x_sq = F.avg_pool2d(x.pow(2), kernel_size=3, stride=1, padding=1) - mu_x.pow(2)
        sigma_y_sq = F.avg_pool2d(x_hat.pow(2), kernel_size=3, stride=1, padding=1) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(x * x_hat, kernel_size=3, stride=1, padding=1) - mu_xy
        ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x_sq + sigma_y_sq + c2))
        return {"mse": mse.item(), "psnr": psnr.item(), "ssim": ssim.mean().item()}


    def compute_fid(self, real_images, fake_images):
        """Computes the Fréchet Inception Distance (FID) between real and reconstructed images.

        Saves images to temporary directories and uses the Inception V3 model to compute
        FID, cleaning up directories afterward.

        Parameters
        ----------
        real_images : torch.Tensor
            Real images, shape (batch_size, channels, height, width), in [-1, 1] range.
        fake_images : torch.Tensor
            Reconstructed images, same shape, in [-1, 1] range.

        Returns
        -------
        float
            FID score, or `float('inf')` if computation fails.

        Notes
        -----
        - Images are normalized to [0, 1] and saved as PNG files for FID computation.
        - The Inception V3 model uses 2048-dimensional features (`dims=2048`).
        - Temporary directories (`temp_dir_real`, `temp_dir_fake`) are created and
          removed automatically.
        """
        real_images = (real_images + 1) / 2
        fake_images = (fake_images + 1) / 2
        real_images = real_images.clamp(0, 1).cpu()
        fake_images = fake_images.clamp(0, 1).cpu()

        os.makedirs(self.temp_dir_real, exist_ok=True)
        os.makedirs(self.temp_dir_fake, exist_ok=True)

        try:
            for i, (real, fake) in enumerate(zip(real_images, fake_images)):
                save_image(real, f"{self.temp_dir_real}/{i}.png")
                save_image(fake, f"{self.temp_dir_fake}/{i}.png")

            fid = fid_score.calculate_fid_given_paths(
                paths=[self.temp_dir_real, self.temp_dir_fake],
                batch_size=50,
                device=self.device,
                dims=2048
            )
        except Exception as e:
            print(f"Error computing FID: {e}")
            fid = float('inf')
        finally:
            shutil.rmtree(self.temp_dir_real, ignore_errors=True)
            shutil.rmtree(self.temp_dir_fake, ignore_errors=True)

        return fid

    def train(self):
        """Trains the autoencoder model for the specified number of epochs.

        Optimizes the model using training data, with optional validation, metrics
        computation, and FID scoring. Saves the best model based on validation (or
        training) loss.

        Returns
        -------
        tuple
            A tuple containing:
            - train_losses: List of mean training losses per epoch.
            - best_val_loss: Best validation (or training) loss achieved.

        Notes
        -----
        - The training loss includes reconstruction, regularization, and optional
          perceptual losses.
        - KL warmup adjusts `model.current_beta` for KL-divergence loss if
          `model.use_vq=False`.
        - Early stopping halts training if the best loss does not improve for
          `patience` epochs.
        - The learning rate is adjusted via `scheduler` based on validation loss.
        - Metrics and FID are computed if enabled via `metrics` and `fid` flags.
        """
        self.model.train()
        self.model.to(self.device)
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(self.max_epoch):
            if self.model.use_vq:
                beta = 1.0
            else:
                beta = min(1.0, epoch / self.kl_warmup_epochs) * self.model.beta
                self.model.current_beta = beta

            train_losses_ = []
            metrics_epoch = {"mse": [], "psnr": [], "ssim": []}
            all_real, all_fake = [], []

            for x, _ in tqdm(self.data_loader):
                x = x.to(self.device)
                x_hat, total_loss, reg_loss, z = self.model(x)
                if self.per_loss:
                    percep_loss = self.perceptual_loss(x_hat, x).mean()
                    loss = total_loss + self.perceptual_weight * percep_loss
                else:
                    loss = total_loss
                train_losses_.append(loss.item())

                if self.metrics or self.fid:
                    with torch.no_grad():
                        if self.metrics:
                            batch_metrics = self.compute_metrics(x, x_hat)
                            for k, v in batch_metrics.items():
                                metrics_epoch[k].append(v)
                        if self.fid:
                            all_real.append(x)
                            all_fake.append(x_hat)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            mean_train_loss = torch.mean(torch.tensor(train_losses_)).item()
            train_losses.append(mean_train_loss)
            metrics_summary = {k: sum(v) / len(v) for k, v in metrics_epoch.items()} if self.metrics else {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
            fid = self.compute_fid(torch.cat(all_real), torch.cat(all_fake)) if self.fid and all_real else float('inf')

            print(f"\nEpoch: {epoch + 1} | Loss: {mean_train_loss:.4f} | Reg Weight: {beta:.4f}", end="")
            if self.metrics:
                print(f" | PSNR: {metrics_summary['psnr']:.2f} | SSIM: {metrics_summary['ssim']:.4f}", end="")
            if self.fid:
                print(f" | FID: {fid:.2f}", end="")
            print()
            if self.val_loader is not None:
                val_loss, val_metrics, val_fid = self.validate()
                print(f"Val Loss: {val_loss:.4f}", end="")
                if self.metrics:
                    print(f" | Val PSNR: {val_metrics['psnr']:.2f} | Val SSIM: {val_metrics['ssim']:.4f}", end="")
                if self.fid:
                    print(f" | Val FID: {val_fid:.2f}", end="")
                print()
                current_best = val_loss
                self.scheduler.step(val_loss)
            else:
                current_best = mean_train_loss

            if current_best < best_val_loss:
                best_val_loss = current_best
                wait = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                }, self.save_path)
                print(f"Model saved at epoch {epoch + 1}")
            else:
                wait += 1
                if wait >= self.patience:
                    print("Early stopping triggered")
                    break

        return train_losses, best_val_loss

    def validate(self):
        """Evaluates the model on the validation dataset.

        Computes validation loss, metrics (MSE, PSNR, SSIM), and FID score without
        updating model parameters.

        Returns
        -------
        tuple
            A tuple containing:
            - mean_val_loss: Mean validation loss (float).
            - metrics_summary: Dictionary of mean MSE, PSNR, SSIM (or zeros if
              `metrics=False`).
            - fid: FID score (or `float('inf')` if `fid=False` or computation fails).
        """
        self.model.eval()
        val_losses = []
        metrics_val = {"mse": [], "psnr": [], "ssim": []}
        all_real, all_fake = [], []

        with torch.no_grad():
            for x, _ in self.val_loader:
                x = x.to(self.device)
                x_hat, total_loss, reg_loss, z = self.model(x)
                if self.per_loss:
                    percep_loss = self.perceptual_loss(x_hat, x).mean()
                    loss = total_loss + self.perceptual_weight * percep_loss
                else:
                    loss = total_loss
                val_losses.append(loss.item())
                if self.metrics or self.fid:
                    if self.metrics:
                        batch_metrics = self.compute_metrics(x, x_hat)
                        for k, v in batch_metrics.items():
                            metrics_val[k].append(v)
                    if self.fid:
                        all_real.append(x)
                        all_fake.append(x_hat)

        mean_val_loss = torch.mean(torch.tensor(val_losses)).item()
        metrics_summary = {k: sum(v) / len(v) for k, v in metrics_val.items()} if self.metrics else {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
        fid = self.compute_fid(torch.cat(all_real), torch.cat(all_fake)) if self.fid and all_real else float('inf')

        self.model.train()
        return mean_val_loss, metrics_summary, fid