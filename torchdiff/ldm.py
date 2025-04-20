__version__ = "1.0.0"

"""Latent Diffusion Models (LDM) implementation.

This module provides a framework for training and sampling Latent Diffusion Models, as
described in Rombach et al. (2022, "High-Resolution Image Synthesis with Latent Diffusion
Models"). It supports diffusion in the latent space using a variational autoencoder
(compressor model), includes utilities for training the autoencoder, noise predictor, and
conditional model, and provides metrics for evaluating generated images. The framework is
compatible with DDPM, DDIM, and SDE diffusion models, supporting both unconditional and
conditional generation with text prompts.

Components:
- AutoencoderLDM: Variational autoencoder for compressing images to latent space and
  decoding back to image space.
- TrainAE: Trainer for AutoencoderLDM, optimizing reconstruction and regularization
  losses with evaluation metrics.
- TrainLDM: Training loop with mixed precision, warmup, and scheduling for the noise
  predictor and conditional model (e.g., TextEncoder with projection layers) in latent
  space, with image-domain evaluation metrics using a reverse diffusion model.
- SampleLDM: Image generation from trained models, decoding from latent to image space.


Notes
-----
- The `hyper_params` parameter expects an external hyperparameter module (e.g.,
  HyperParamsDDPM, HyperParamsSDE) as an nn.Module for noise schedule management.
- AutoencoderLDM serves as the `compressor_model` in TrainLDM and SampleLDM, providing
  `encode` and `decode` methods for latent space conversion. It supports KL-divergence or
  vector quantization (VQ) regularization, using internal components (DownBlock, UpBlock,
  Conv3, DownSampling, UpSampling, Attention, VectorQuantizer).
- TrainAE trains AutoencoderLDM, optimizing reconstruction (MSE), regularization (KL or
  VQ), and optional perceptual (LPIPS) losses, with metrics (MSE, PSNR, SSIM, FID, LPIPS)
  computed via the Metrics class, KL warmup, early stopping, and learning rate scheduling.
- TrainLDM trains the noise predictor and conditional model, optimizing MSE between
  predicted and ground truth noise, with optional validation metrics (MSE, PSNR, SSIM, FID,
  LPIPS) on generated images decoded from latents sampled using a reverse diffusion model
  (e.g., ReverseDDPM).
- Metrics computes MSE, PSNR, SSIM, FID, and LPIPS for evaluating generated images,
  assuming inputs in [-1, 1] or [0, 1] based on normalization, returning individual metric values.
- The `conditional_model` parameter expects a text encoder (e.g., TextEncoder from utils
  module) with a BERT tokenizer and trainable projection layers for conditional generation,
  with tokenized text inputs compatible with its attention mask convention (1 for valid tokens).
- SampleLDM supports multiple diffusion models ("ddpm", "ddim", "sde") via the `model`
  parameter, requiring compatible `reverse_diffusion` modules (e.g., ReverseDDPM,
  ReverseDDIM, ReverseSDE).


References
----------
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).
High-Resolution Image Synthesis with Latent Diffusion Models.

Examples
--------
>>> from torchdiff.ddpm import HyperParamsDDPM, ForwardDDPM, ReverseDDPM
>>> from torchdiff.ldm import AutoencoderLDM, TrainAE, TrainLDM, SampleLDM, Metrics
>>> from torchdiff.nets import TextEncoder
>>> from torchdiff.noise_predictor import NoisePredictor
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
>>> metrics = Metrics(device='cuda', fid=True, metrics=True, lpips=True)
>>> train_ae = TrainAE(model=compressor, optimizer=optimizer_ae, data_loader=data_loader, val_loader=val_loader,
...                    max_epoch=100, metrics_=metrics, device='cuda', save_path='vlc_model.pth')
>>> ae_losses, best_ae_loss = train_ae.train()
>>> noise_predictor = NoisePredictor(in_channels=3, down_channels=[16, 32], mid_channels=[32, 32],
...                                 up_channels=[32, 16], down_sampling=[True, False], time_embed_dim=128,
...                                 y_embed_dim=128, num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2,
...                                 dropout_rate=0.1, down_sampling_factor=2, where_y=True, y_to_all=False)
>>> optimizer_ldm = Adam(list(noise_predictor.parameters()) + list(text_encoder.parameters()), lr=1e-4)
>>> train_ldm = TrainLDM(model="ddpm", forward_model=forward_ddpm, reverse_diffusion=reverse_ddpm,
...                      hyper_params=hyper_params, noise_predictor=noise_predictor, compressor_model=compressor,
...                      optimizer=optimizer_ldm, objective=nn.MSELoss(), data_loader=data_loader,
...                      val_loader=val_loader, conditional_model=text_encoder, metrics_=metrics,
...                      device='cuda', store_path='ldm_model.pth')
>>> train_losses, best_val_loss = train_ldm.train()
>>> sampler = SampleLDM(model="ddpm", reverse_diffusion=reverse_ddpm,
...                     noise_predictor=noise_predictor, compressor_model=compressor,
...                     image_shape=(256, 256), conditional_model=text_encoder)
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
from torchvision.utils import save_image
import os

###==================================================================================================================###

class TrainLDM(nn.Module):
    """Trainer for the noise predictor in Latent Diffusion Models.

    Optimizes the noise predictor and conditional model (e.g., TextEncoder)
    to predict noise in the latent space of AutoencoderLDM, using a diffusion model (e.g., DDPM, DDIM, SDE).
    Supports mixed precision, conditional generation with text prompts, and evaluation metrics
    (MSE, PSNR, SSIM, FID, LPIPS) for generated images during validation, using a specified reverse
    diffusion model.

    Parameters
    ----------
    model : str
        Diffusion model type ("ddpm", "ddim", "sde").
    forward_model : ForwardDDPM, ForwardDDIM, or ForwardSDE
        Forward diffusion model defining the noise schedule.
    hyper_params : HyperParamsDDPM, HyperParamsDDIM, or HyperParamsSDE
        Hyperparameters for the diffusion process (nn.Module).
    noise_predictor : NoisePredictor
        Model to predict noise in the latent space (e.g., NoisePredictor).
    compressor_model : AutoencoderLDM
        Variational autoencoder for encoding/decoding latents.
    optimizer : torch.optim.Optimizer
        Optimizer for the noise predictor and conditional model (e.g., Adam).
    objective : torch.nn.Module
        Loss function for noise prediction (e.g., MSELoss).
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data (default: None).
    conditional_model : TextEncoder, optional
        Text encoder with projection layers for conditional generation (default: None).
    reverse_diffusion : ReverseDDPM, ReverseDDIM, or ReverseSDE, optional
        Reverse diffusion model for sampling during validation (default: None).
    metrics_ : Metrics, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    max_epoch : int, optional
        Maximum number of training epochs (default: 1000).
    device : str, optional
        Device for computation (e.g., 'cuda', 'cpu') (default: None).
    store_path : str, optional
        Path to save model checkpoints (default: None, uses 'ldm_model.pth').
    patience : int, optional
        Number of epochs to wait for early stopping if validation loss doesn’t improve
        (default: 100).
    warmup_epochs : int, optional
        Number of epochs for learning rate warmup (default: 100).
    max_length : int, optional
        Maximum sequence length for tokenized text (default: 77).
    val_frequency : int, optional
        Frequency (in epochs) for validation and metric computation (default: 10).
    output_range : tuple, optional
        Range for clamping generated images (default: (-1, 1)).
    normalize_output : bool, optional
        Whether to normalize generated images to [0, 1] for metrics (default: True).

    Attributes
    ----------
    device : torch.device
        Computation device.
    model : str
        Diffusion model type.
    forward_model : ForwardDDPM, ForwardDDIM, or ForwardSDE
        Forward diffusion model.
    reverse_diffusion : ReverseDDPM, ReverseDDIM, or ReverseSDE or None
        Reverse diffusion model.
    hyper_params : HyperParamsDDPM, HyperParamsDDIM, or HyperParamsSDE
        Diffusion hyperparameters.
    noise_predictor : NoisePredictor
        Noise prediction model.
    compressor_model : AutoencoderLDM
        Autoencoder for latent space.
    optimizer : torch.optim.Optimizer
        Training optimizer.
    objective : torch.nn.Module
        Loss function.
    data_loader : torch.utils.data.DataLoader
        Training DataLoader.
    val_loader : torch.utils.data.DataLoader or None
        Validation DataLoader.
    conditional_model : TextEncoder or None
        Text encoder for conditioning.
    tokenizer : callable
        Text tokenizer.
    metrics_ : Metrics or None
        Metrics object for evaluation.
    max_epoch : int
        Maximum training epochs.
    store_path : str
        Checkpoint save path.
    patience : int
        Early stopping patience.
    warmup_epochs : int
        Warmup epochs for learning rate.
    max_length : int
        Maximum text sequence length.
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler.
    warmup_lr_scheduler : torch.optim.lr_scheduler.LambdaLR
        Warmup learning rate scheduler.
    val_frequency : int
        Validation frequency.
    output_range : tuple
        Output range for generated images.
    normalize_output : bool
        Whether to normalize output.

    Raises
    ------
    ValueError
        If the default tokenizer ("bert-base-uncased") fails to load and no tokenizer is provided.
        If model is not one of these models ["ddpm", "ddim", "sde"].
    """

    def __init__(self, model, forward_model, hyper_params, noise_predictor, compressor_model,
                 optimizer, objective, data_loader, val_loader=None, conditional_model=None,
                 reverse_diffusion=None, metrics_=None, max_epoch=1000, device=None,
                 store_path=None, patience=100, warmup_epochs=100, tokenizer=None, max_length=77,
                 val_frequency=10, output_range=(-1, 1), normalize_output=True):
        super().__init__()
        if model not in ["ddpm", "ddim", "sde"]:
            raise ValueError(f"Unknown model: {model}. Supported: ddpm, ddim, sde")
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.forward_model = forward_model.to(self.device)
        self.reverse_diffusion = reverse_diffusion.to(self.device) if reverse_diffusion else None
        self.hyper_params = hyper_params.to(self.device)  # nn.Module, move to device
        self.noise_predictor = noise_predictor.to(self.device)
        self.compressor_model = compressor_model.to(self.device)
        self.optimizer = optimizer
        self.objective = objective
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.metrics_ = metrics_
        self.max_epoch = max_epoch
        self.store_path = store_path or "ldm_model.pth"
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.max_length = max_length
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=0.5)
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)
        self.val_frequency = val_frequency
        self.output_range = output_range
        self.normalize_output = normalize_output
        if tokenizer is None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")

    def load_checkpoint(self, checkpoint_path):
        """Loads a checkpoint for the noise predictor and optional conditional model.

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
            If required state dictionaries are missing.
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

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to schedule.
        warmup_epochs : int
            Number of epochs for warmup.

        Returns
        -------
        torch.optim.lr_scheduler.LambdaLR
            Scheduler that scales learning rate linearly during warmup.
        """
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 1.0
        return LambdaLR(optimizer, lr_lambda)

    def forward(self):
        """Trains the noise predictor and conditional model with mixed precision and evaluation metrics.

        Optimizes the noise predictor and conditional model (e.g., TextEncoder with projection layers)
        using the forward diffusion model’s noise schedule, with text conditioning. Performs validation
        with image-domain metrics (MSE, PSNR, SSIM, FID, LPIPS) using the reverse diffusion model,
        saves checkpoints for the best validation loss, and supports early stopping.

        Returns
        -------
        tuple
            A tuple containing:
            - train_losses: List of mean training losses per epoch.
            - best_val_loss: Best validation loss achieved (or best training loss if no validation).
        """
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()
        self.compressor_model.eval()  # pre-trained, not trained here

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
                with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    noise = torch.randn_like(x).to(self.device)
                    t = torch.randint(0, self.hyper_params.num_steps, (x.shape[0],), device=self.device)
                    assert x.device == noise.device == t.device, "Device mismatch detected"
                    assert t.shape[0] == x.shape[0], "Timestep batch size mismatch"
                    noisy_x = self.forward_model(x, noise, t)
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
                self.hyper_params.constrain_betas()

            mean_train_loss = torch.mean(torch.tensor(train_losses_)).item()
            train_losses.append(mean_train_loss)
            print(f"Epoch: {epoch + 1} | Train Loss: {mean_train_loss:.4f}", end="")

            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_loss, fid, mse, psnr, ssim, lpips_score = self.validate()
                print(f" | Val Loss: {val_loss:.4f}", end="")
                if self.metrics_ and self.metrics_.fid:
                    print(f" | FID: {fid:.4f}", end="")
                if self.metrics_ and self.metrics_.metrics:
                    print(f" | MSE: {mse:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}", end="")
                if self.metrics_ and self.metrics_.lpips:
                    print(f" | LPIPS: {lpips_score:.4f}", end="")
                print()

                current_best = val_loss
                self.scheduler.step(val_loss)
            else:
                print()
                current_best = mean_train_loss
                self.scheduler.step(mean_train_loss)

            if current_best < best_val_loss and (epoch + 1) % self.val_frequency == 0:
                best_val_loss = current_best
                wait = 0
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict_noise_predictor': self.noise_predictor.state_dict(),
                        'model_state_dict_conditional': self.conditional_model.state_dict() if self.conditional_model is not None else None,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_val_loss,
                        'hyper_params_model': self.hyper_params.state_dict() if isinstance(self.hyper_params, nn.Module) else self.hyper_params,
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
                            'hyper_params_model': self.hyper_params.state_dict() if isinstance(self.hyper_params, nn.Module) else self.hyper_params,
                        }, self.store_path + "_early_stop.pth")
                        print(f"Final model saved at {self.store_path}_early_stop.pth")
                    except Exception as e:
                        print(f"Failed to save final model: {e}")
                    break

        return train_losses, best_val_loss

    def validate(self):
        """Validates the noise predictor and computes evaluation metrics.

        Computes validation loss (MSE between predicted and ground truth noise) and generates
        samples using the reverse diffusion model by manually iterating over timesteps.
        Decodes samples to images and computes image-domain metrics (MSE, PSNR, SSIM, FID, LPIPS)
        if metrics_ is provided.

        Returns
        -------
        tuple
            A tuple containing:
            - val_loss: Mean validation loss (float).
            - fid: Mean FID score (float, or `float('inf')` if not computed).
            - mse: Mean MSE (float, or None if not computed).
            - psnr: Mean PSNR (float, or None if not computed).
            - ssim: Mean SSIM (float, or None if not computed).
            - lpips_score: Mean LPIPS score (float, or None if not computed).
        """
        self.noise_predictor.eval()
        if self.conditional_model is not None:
            self.conditional_model.eval()
        val_losses = []
        fid_, mse_, psnr_, ssim_, lpips_score_ = [], [], [], [], []
        num_steps = self.hyper_params.tau_num_steps if self.model == "ddim" else self.hyper_params.num_steps
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                x_orig = x  # store original images for metrics
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

                # validation loss
                noise = torch.randn_like(x).to(self.device)
                t = torch.randint(0, self.hyper_params.num_steps, (x.shape[0],), device=self.device)
                assert x.device == noise.device == t.device, "Device mismatch detected"
                assert t.shape[0] == x.shape[0], "Timestep batch size mismatch"
                noisy_x = self.forward_model(x, noise, t)
                p_noise = self.noise_predictor(noisy_x, t, y_encoded)
                loss = self.objective(p_noise, noise)
                val_losses.append(loss.item())

                # generate samples for metrics
                if self.metrics_ is not None and self.reverse_diffusion is not None:
                    xt = torch.randn_like(x).to(self.device)
                    for t in reversed(range(num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        prev_time_steps = torch.full((xt.shape[0],), max(t - 1, 0), device=self.device, dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, y_encoded)
                        if self.model == "sde":
                            noise = torch.randn_like(xt) if getattr(self.reverse_diffusion, "method", None) != "ode" else None
                            xt = self.reverse_diffusion(xt, noise, predicted_noise, time_steps)
                        elif self.model == "ddim":
                            xt, _ = self.reverse_diffusion(xt, predicted_noise, time_steps, prev_time_steps)
                        elif self.model == "ddpm":
                            xt = self.reverse_diffusion(xt, predicted_noise, time_steps)
                        else:
                            raise ValueError(f"Unknown model: {self.model}. Supported: ddpm, ddim, sde")

                    x_hat = self.compressor_model.decode(xt)
                    x_hat = torch.clamp(x_hat, min=self.output_range[0], max=self.output_range[1])
                    if self.normalize_output:
                        x_hat = (x_hat - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
                        x_orig = (x_orig - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
                    fid, mse, psnr, ssim, lpips_score = self.metrics_.forward(x_orig, x_hat)
                    if self.metrics_.fid:
                        fid_.append(fid)
                    if self.metrics_.metrics:
                        mse_.append(mse)
                        psnr_.append(psnr)
                        ssim_.append(ssim)
                    if self.metrics_.lpips:
                        lpips_score_.append(lpips_score)

        val_loss = torch.mean(torch.tensor(val_losses)).item()
        fid_ = torch.mean(torch.tensor(fid_)).item() if fid_ else float('inf')
        mse_ = torch.mean(torch.tensor(mse_)).item() if mse_ else None
        psnr_ = torch.mean(torch.tensor(psnr_)).item() if psnr_ else None
        ssim_ = torch.mean(torch.tensor(ssim_)).item() if ssim_ else None
        lpips_score_ = torch.mean(torch.tensor(lpips_score_)).item() if lpips_score_ else None

        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()
        return val_loss, fid_, mse_, psnr_, ssim_, lpips_score_

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

    def forward(self, conditions=None, normalize_output=True, save_images=True, save_path="ldm_generated"):
        """Generates images using the reverse diffusion process in the latent space.

        Iteratively denoises random noise in the latent space using the specified reverse
        diffusion model (DDPM, DDIM, SDE), then decodes the result to the image space
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

            # save images if save_images is True
            if save_images:
                os.makedirs(save_path, exist_ok=True)
                for i in range(generated_imgs.size(0)):
                    img_path = os.path.join(save_path, f"image_{i}.png")
                    save_image(generated_imgs[i], img_path)

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

class TrainAE(nn.Module):
    """Trainer for the AutoencoderLDM variational autoencoder in Latent Diffusion Models.

    Optimizes the AutoencoderLDM model to compress images into latent space and reconstruct
    them, using reconstruction loss (MSE), regularization (KL or VQ), and optional
    perceptual loss (LPIPS). Supports mixed precision, KL warmup, early stopping, and
    learning rate scheduling, with evaluation metrics (MSE, PSNR, SSIM, FID, LPIPS).

    Parameters
    ----------
    model : AutoencoderLDM
        The variational autoencoder model (AutoencoderLDM) to train.
    optimizer : torch.optim.Optimizer
        Optimizer for training (e.g., Adam).
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data (default: None).
    max_epoch : int, optional
        Maximum number of training epochs (default: 100).
    metrics_ : Metrics, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    device : str, optional
        Device for computation (e.g., 'cuda', 'cpu') (default: 'cuda').
    save_path : str, optional
        Path to save model checkpoints (default: 'vlc_model.pth').
    checkpoint : int, optional
        Frequency (in epochs) to save model checkpoints (default: 10).
    kl_warmup_epochs : int, optional
        Number of epochs for KL loss warmup (default: 10).
    patience : int, optional
        Number of epochs to wait for early stopping if validation loss doesn’t improve
        (default: 10).
    val_frequency : int, optional
        Frequency (in epochs) for validation and metric computation (default: 5).

    Attributes
    ----------
    device : torch.device
        Computation device.
    model : AutoencoderLDM
        Autoencoder model being trained.
    optimizer : torch.optim.Optimizer
        Training optimizer.
    data_loader : torch.utils.data.DataLoader
        Training DataLoader.
    val_loader : torch.utils.data.DataLoader or None
        Validation DataLoader.
    max_epoch : int
        Maximum training epochs.
    metrics_ : Metrics or None
        Metrics object for evaluation.
    save_path : str
        Checkpoint save path.
    checkpoint : int
        Checkpoint frequency.
    kl_warmup_epochs : int
        KL warmup epochs.
    patience : int
        Early stopping patience.
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler.
    val_frequency : int
        Validation frequency.
    """

    def __init__(self, model, optimizer, data_loader, val_loader=None, max_epoch=100, metrics_=None,
                 device="cuda", save_path="vlc_model.pth", checkpoint=10, kl_warmup_epochs=10,
                 patience=10, val_frequency=5):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.metrics_ = metrics_  
        self.save_path = save_path
        self.checkpoint = checkpoint
        self.kl_warmup_epochs = kl_warmup_epochs
        self.patience = patience
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.val_frequency = val_frequency

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

        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict' key")
        self.model.load_state_dict(checkpoint['model_state_dict'])

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

    def forward(self):
        """Trains the AutoencoderLDM model with mixed precision and evaluation metrics.

        Performs training with reconstruction and regularization losses, KL warmup, gradient
        clipping, and learning rate scheduling. Saves checkpoints for the best validation
        loss and supports early stopping.

        Returns
        -------
        tuple
            A tuple containing:
            - train_losses: List of mean training losses per epoch.
            - best_val_loss: Best validation loss achieved (or best training loss if no validation).
        """
        scaler = GradScaler()
        self.model.train()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(self.max_epoch):
            if self.model.use_vq:
                beta = 1.0  # no warmup for VQ
            else:
                beta = min(1.0, epoch / self.kl_warmup_epochs) * self.model.beta
                self.model.current_beta = beta

            train_losses_ = []
            for x, _ in tqdm(self.data_loader):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    x_hat, loss, reg_loss, z = self.model(x)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                train_losses_.append(loss.item())

            mean_train_loss = torch.mean(torch.tensor(train_losses_)).item()
            train_losses.append(mean_train_loss)
            print(f"Epoch: {epoch + 1} | Train Loss: {mean_train_loss:.4f}", end="")

            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_loss, fid, mse, psnr, ssim, lpips_score = self.validate()
                print(f" | Val Loss: {val_loss:.4f}", end="")
                if self.metrics_ and self.metrics_.fid:
                    print(f" | FID: {fid:.4f}", end="")
                if self.metrics_ and self.metrics_.metrics:
                    print(f" | MSE: {mse:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}", end="")
                if self.metrics_ and self.metrics_.lpips:
                    print(f" | LPIPS: {lpips_score:.4f}", end="")
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
                print(f" | Model saved at epoch {epoch + 1}")
            else:
                wait += 1
                if wait >= self.patience:
                    print("Early stopping triggered")
                    break

        return train_losses, best_val_loss

    def validate(self):
        """Validates the AutoencoderLDM model and computes evaluation metrics.

        Computes validation loss and optional metrics (MSE, PSNR, SSIM, FID, LPIPS) using
        the provided Metrics object.

        Returns
        -------
        tuple
            A tuple containing:
            - val_loss: Mean validation loss (float).
            - fid: Mean FID score (float, or `float('inf')` if not computed).
            - mse: Mean MSE (float, or None if not computed).
            - psnr: Mean PSNR (float, or None if not computed).
            - ssim: Mean SSIM (float, or None if not computed).
            - lpips_score: Mean LPIPS score (float, or None if not computed).
        """
        self.model.eval()
        val_losses = []
        fid_, mse_, psnr_, ssim_, lpips_score_ = [], [], [], [], []

        with torch.no_grad():
            for x, _ in self.val_loader:
                x = x.to(self.device)
                x_hat, loss, reg_loss, z = self.model(x)
                val_losses.append(loss.item())
                if self.metrics_ is not None:
                    fid, mse, psnr, ssim, lpips_score = self.metrics_.forward(x, x_hat)
                    if self.metrics_.fid:
                        fid_.append(fid)
                    if self.metrics_.metrics:
                        mse_.append(mse)
                        psnr_.append(psnr)
                        ssim_.append(ssim)
                    if self.metrics_.lpips:
                        lpips_score_.append(lpips_score)

        val_loss = torch.mean(torch.tensor(val_losses)).item()
        fid_ = torch.mean(torch.tensor(fid_)).item() if fid_ else float('inf')
        mse_ = torch.mean(torch.tensor(mse_)).item() if mse_ else None
        psnr_ = torch.mean(torch.tensor(psnr_)).item() if psnr_ else None
        ssim_ = torch.mean(torch.tensor(ssim_)).item() if ssim_ else None
        lpips_score_ = torch.mean(torch.tensor(lpips_score_)).item() if lpips_score_ else None

        self.model.train()
        return val_loss, fid_, mse_, psnr_, ssim_, lpips_score_