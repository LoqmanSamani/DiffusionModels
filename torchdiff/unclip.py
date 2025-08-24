"""
**UnCLIP Diffusion Model**

This module provides a comprehensive implementation of the UnCLIP diffusion model,
as described in Ramesh et al. (2022, "Hierarchical Text-Conditional Image Generation with CLIP Latents").
It integrates CLIP embeddings with diffusion processes for high-quality image generation conditioned on text prompts or image embeddings.
The module supports training, sampling, and upsampling processes, leveraging components from CLIP, GLIDE, and DDIM,
with classifier-free guidance and text dropout for robust generation.

**Components**

- **VarianceSchedulerUnCLIP**: Manages noise schedules with support for linear, sigmoid, quadratic, constant, inverse_time,
                               and cosine beta schedules, including subsampled (tau) schedules for efficient sampling.
- **ForwardUnCLIP**: Forward diffusion process to add noise to image or latent embeddings.
- **ReverseUnCLIP**: Reverse diffusion process for denoising, supporting noise or clean image predictions with subsampled steps.
- **CLIPEncoder**: Encodes images or text into embeddings using a pre-trained CLIP model.
- **UnClipDecoder**: Generates low-resolution images (64x64) from CLIP embeddings, incorporating GLIDE text encoding and classifier-free guidance.
- **UnCLIPTransformerPrior**: Transformer-based prior to predict clean image embeddings from noisy embeddings and text conditions.
- **CLIPContextProjection**: Projects CLIP image embeddings into context tokens for the decoder.
- **CLIPEmbeddingProjection**: Reduces and reconstructs embedding dimensionality for efficient processing.
- **TrainUnClipDecoder**: Orchestrates training of the decoder with mixed precision, gradient accumulation, and DDP support.
- **SampleUnCLIP**: Generates images from text prompts or noise, scaling from 64x64 to 256x256 or 1024x1024 with upsamplers.
- **UpsamplerUnCLIP**: U-Net-based upsampler for scaling images (64x64 to 256x256 or 256x256 to 1024x1024), conditioned on low-resolution inputs.
- **TrainUpsamplerUnCLIP**: Trains the upsampler with noise prediction, low-resolution conditioning, and optional image corruption (Gaussian blur or BSR degradation).

**Notes**

- The model uses a subsampled time step schedule (tau) for faster sampling, controlled by the `tau_num_steps` parameter in VarianceSchedulerUnCLIP.
- Classifier-free guidance and text dropout enhance generation quality, with tunable parameters `classifier_free_prop` and `drop_caption`.
- Upsampling stages use corrupted low-resolution inputs (Gaussian blur for 64x64→256x256, BSR degradation for 256x256→1024x1024) to improve robustness.
- Supports distributed training with DDP, mixed precision via autocast, and learning rate scheduling with warmup and plateau reduction.

**References**

- Ramesh, Aditya, et al. "Hierarchical Text-Conditional Image Generation with CLIP Latents." arXiv preprint arXiv:2204.06125 (2022).
- Radford, Alec, et al. "Learning Transferable Visual Models From Natural Language Supervision." arXiv preprint arXiv:2103.00020 (2021).
- Nichol, Alexander, et al. "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models." arXiv preprint arXiv:2112.10741 (2021).
- Song, Jiaming, et al. "Denoising Diffusion Implicit Models." arXiv preprint arXiv:2010.02502 (2020).

-------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torchvision
from PIL import Image
from transformers import BertTokenizer, CLIPProcessor, CLIPModel
from typing import Optional, List, Tuple, Union, Callable, Any, Self
from tqdm import tqdm
import os
import warnings
import random
import math


###==================================================================================================================###


class VarianceSchedulerUnCLIP(nn.Module):
    """Manages noise schedule parameters for UnCLIP diffusion models.

    Handles beta values, derived noise schedule quantities, and a subsampled time step schedule
    (tau schedule) for UnCLIP diffusion processes. Supports trainable or fixed beta schedules
    and multiple scheduling methods, including linear, sigmoid, quadratic, constant, inverse_time,
    and cosine schedules.

    Parameters
    ----------
    `eta` : float, optional
        Noise scaling factor for the reverse process (default: 0, deterministic).
    `num_steps` : int, optional
        Total number of diffusion steps (default: 1000).
    `tau_num_steps` : int, optional
        Number of subsampled time steps for sampling (default: 100).
    `beta_start` : float, optional
        Starting value for beta (default: 1e-4).
    `beta_end` : float, optional
        Ending value for beta (default: 0.02).
    `trainable_beta` : bool, optional
        Whether the beta schedule is trainable (default: False).
    `beta_method` : str, optional
        Method for computing the beta schedule (default: "linear").
        Supported methods: "linear", "sigmoid", "quadratic", "constant", "inverse_time", "cosine".
    """
    def __init__(
            self,
            eta: Optional[float] = None,
            num_steps: int = 1000,
            tau_num_steps: int = 100,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            trainable_beta: bool = False,
            beta_method: str = "linear"
    ) -> None:
        super().__init__()
        self.eta = eta or 0
        self.num_steps = num_steps
        self.tau_num_steps = tau_num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trainable_beta = trainable_beta
        self.beta_method = beta_method

        if not (0 < beta_start < beta_end < 1):
            raise ValueError(f"beta_start ({beta_start}) and beta_end ({beta_end}) must satisfy 0 < start < end < 1")
        if num_steps <= 0:
            raise ValueError(f"num_steps ({num_steps}) must be positive")

        beta_range = (beta_start, beta_end)
        betas_init = self.compute_beta_schedule(beta_range, num_steps, beta_method)

        if trainable_beta:
            self.beta_raw = nn.Parameter(torch.logit((betas_init - beta_start) / (beta_end - beta_start)))
        else:
            self.register_buffer('betas_buffer', betas_init)
            self.register_buffer('alphas', 1 - self.betas)
            self.register_buffer('alpha_cumprod', torch.cumprod(self.alphas, dim=0))
            self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(self.alpha_cumprod))
            self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1 - self.alpha_cumprod))

        self.register_buffer('tau_indices', torch.linspace(0, num_steps - 1, tau_num_steps, dtype=torch.long))

    @property
    def betas(self) -> torch.Tensor:
        """Returns the beta values, applying reparameterization if trainable.

        Returns the beta values, using sigmoid reparameterization for trainable betas
        or directly accessing the stored buffer for fixed betas.

        Returns
        -------
        betas : torch.Tensor
            Beta values, shape (num_steps,).
        """
        if self.trainable_beta:
            return self.beta_start + (self.beta_end - self.beta_start) * torch.sigmoid(self.beta_raw)
        return self._buffers['betas_buffer']

    def compute_beta_schedule(self, beta_range: Tuple[float, float], num_steps: int, method: str) -> torch.Tensor:
        """Computes the beta schedule based on the specified method.

        Generates a sequence of beta values for the noise schedule using the chosen method,
        ensuring values are clamped within the specified range. Supports linear, sigmoid,
        quadratic, constant, inverse_time, and cosine schedules.

        Parameters
        ----------
        `beta_range` : tuple
            Tuple of (min_beta, max_beta) specifying the valid range for beta values.
        `num_steps` : int
            Number of diffusion steps.
        `method` : str
            Method for computing the beta schedule. Supported methods:
            "linear", "sigmoid", "quadratic", "constant", "inverse_time", "cosine".

        Returns
        -------
        beta : torch.Tensor
            Tensor of beta values, shape (num_steps,).
        """
        beta_min, beta_max = beta_range
        if method == "sigmoid":
            x = torch.linspace(-6, 6, num_steps)
            beta = torch.sigmoid(x) * (beta_max - beta_min) + beta_min
        elif method == "quadratic":
            x = torch.linspace(beta_min ** 0.5, beta_max ** 0.5, num_steps)
            beta = x ** 2
        elif method == "constant":
            beta = torch.full((num_steps,), beta_max)
        elif method == "inverse_time":
            beta = 1.0 / torch.linspace(num_steps, 1, num_steps)
            beta = beta_min + (beta_max - beta_min) * (beta - beta.min()) / (beta.max() - beta.min())
        elif method == "linear":
            beta = torch.linspace(beta_min, beta_max, num_steps)
        elif method == "cosine":
            s = 0.008
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        else:
            raise ValueError(f"Unknown beta_method: {method}")
        beta = torch.clamp(beta, min=beta_min, max=beta_max)
        return beta

    def get_tau_schedule(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the subsampled (tau) noise schedule for UnCLIP.

        Returns the noise schedule parameters for the subsampled time steps used in
        UnCLIP sampling, based on the `tau_indices`.

        Returns
        -------
        tau_betas : torch.Tensor
            Beta values for subsampled steps, shape (tau_num_steps,).
        tau_alphas : torch.Tensor
            Alpha values for subsampled steps, shape (tau_num_steps,).
        tau_alpha_cumprod : torch.Tensor
            Cumulative product of alphas for subsampled steps, shape (tau_num_steps,).
        tau_sqrt_alpha_cumprod : torch.Tensor
            Square root of alpha_cumprod for subsampled steps, shape (tau_num_steps,).
        tau_sqrt_one_minus_alpha_cumprod : torch.Tensor
            Square root of (1 - alpha_cumprod) for subsampled steps, shape (tau_num_steps,).
        """
        if self.trainable_beta:
            betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod = self.compute_schedule()
        else:
            betas = self.betas
            alphas = self.alphas
            alpha_cumprod = self.alpha_cumprod
            sqrt_alpha_cumprod = self.sqrt_alpha_cumprod
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod

        tau_betas = betas[self.tau_indices]
        tau_alphas = alphas[self.tau_indices]
        tau_alpha_cumprod = alpha_cumprod[self.tau_indices]
        tau_sqrt_alpha_cumprod = sqrt_alpha_cumprod[self.tau_indices]
        tau_sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod[self.tau_indices]

        return tau_betas, tau_alphas, tau_alpha_cumprod, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod

    def compute_schedule(self, time_steps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes noise schedule parameters dynamically from betas.

        Calculates the derived noise schedule parameters (alphas, alpha_cumprod, etc.)
        from the provided beta values for the UnCLIP diffusion process.

        Parameters
        ----------
        `time_steps` : torch.Tensor, optional
            If provided, returns parameters only for specified time steps.
            If None, returns parameters for all time steps.

        Returns
        -------
        betas : torch.Tensor
            Beta values, shape (num_steps,) or (len(time_steps),).
        alphas : torch.Tensor
            1 - betas, shape (num_steps,) or (len(time_steps),).
        alpha_cumprod : torch.Tensor
            Cumulative product of alphas, shape (num_steps,) or (len(time_steps),).
        sqrt_alpha_cumprod : torch.Tensor
            Square root of alpha_cumprod, shape (num_steps,) or (len(time_steps),).
        sqrt_one_minus_alpha_cumprod : torch.Tensor
            Square root of (1 - alpha_cumprod), shape (num_steps,) or (len(time_steps),).
        """
        betas = self.betas
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
        if time_steps is not None:
            return (betas[time_steps], alphas[time_steps], alpha_cumprod[time_steps],
                    sqrt_alpha_cumprod[time_steps], sqrt_one_minus_alpha_cumprod[time_steps])
        return betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod

###==================================================================================================================###

class ForwardUnCLIP(nn.Module):
    """Forward diffusion process for UnCLIP diffusion models.

    Applies Gaussian noise to input data (2D or 4D tensors) according to the UnCLIP
    forward diffusion process at specified time steps, using cumulative noise schedule
    parameters from the variance scheduler.

    Parameters
    ----------
    `variance_scheduler` : torch.nn.Module
        Variance scheduler module (e.g., VarianceSchedulerUnCLIP) containing the noise
        schedule parameters.
    """
    def __init__(self, variance_scheduler: torch.nn.Module) -> None:
        super().__init__()
        self.variance_scheduler = variance_scheduler

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Applies the forward diffusion process to the input data.

        Perturbs the input data `x0` by adding Gaussian noise at specified time steps,
        supporting both 2D (e.g., latent embeddings) and 4D (e.g., image) inputs.

        Parameters
        ----------
        `x0` : torch.Tensor
            Input data tensor, shape (batch_size, embedding_dim) for 2D or
            (batch_size, channels, height, width) for 4D.
        `noise` : torch.Tensor
            Gaussian noise tensor, same shape as `x0`.
        `time_steps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,),
            where each value is in the range [0, variance_scheduler.num_steps - 1].

        Returns
        -------
        xt : torch.Tensor
            Noisy data tensor at the specified time steps, same shape as `x0`.
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.variance_scheduler.num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.variance_scheduler.num_steps - 1}")

        if self.variance_scheduler.trainable_beta:
            _, _, _, sqrt_alpha_cumprod_t, sqrt_one_minus_alpha_cumprod_t = self.variance_scheduler.compute_schedule(
                time_steps
            )
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.to(x0.device)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.to(x0.device)
        else:
            sqrt_alpha_cumprod_t = self.variance_scheduler.sqrt_alpha_cumprod[time_steps].to(x0.device)
            sqrt_one_minus_alpha_cumprod_t = self.variance_scheduler.sqrt_one_minus_alpha_cumprod[time_steps].to(x0.device)

        # check input dimensions and adjust reshaping for 2D or 4D tensors
        is_2d = x0.dim() == 2  # check if input is 2D (batch_size, embedding_dim)
        if is_2d:
            # for 2D inputs, reshape to [batch_size, 1]
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1)
        else:
            # for 4D inputs, reshape to [batch_size, 1, 1, 1]
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1, 1)

        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt

###==================================================================================================================###

class ReverseUnCLIP(nn.Module):
    """Reverse diffusion process for UnCLIP diffusion models.

    Denoises a noisy input `xt` using either a predicted noise component or predicted clean image
    and a subsampled time step schedule, supporting both 2D (e.g., latent embeddings) and 4D (e.g., image) inputs.

    Parameters
    ----------
    `variance_scheduler` : torch.nn.Module
        Variance scheduler module (e.g., VarianceSchedulerUnCLIP) containing the noise
        schedule parameters.
    `prediction_type` : str, default "noise"
        Type of prediction the model makes. Either "noise" (predicts noise like DDIM) or
        "x0" (predicts clean image like UnCLIP prior).
    """

    def __init__(self, variance_scheduler: torch.nn.Module, prediction_type: str = "noise"):
        super().__init__()
        self.variance_scheduler = variance_scheduler
        if prediction_type not in ["noise", "x0"]:
            raise ValueError(f"prediction_type must be either 'noise' or 'x0', got {prediction_type}")
        self.prediction_type = prediction_type

    def forward(
            self,
            xt: torch.Tensor,
            model_prediction: torch.Tensor,
            time_steps: torch.Tensor,
            prev_time_steps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the reverse diffusion process to the noisy input.

        Denoises the input `xt` at time step `t` to produce the previous step `xt_prev`
        at `prev_time_steps` using either the predicted noise or predicted clean image
        and the UnCLIP reverse process. Supports both 2D and 4D inputs.

        Parameters
        ----------
        `xt` : torch.Tensor
            Noisy input tensor at time step `t`, shape (batch_size, embedding_dim) for 2D
            or (batch_size, channels, height, width) for 4D.
        `model_prediction` : torch.Tensor
            Model prediction tensor, same shape as `xt`. Can be either predicted noise
            or predicted clean image depending on `prediction_type`.
        `time_steps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, variance_scheduler.tau_num_steps - 1].
        `prev_time_steps` : torch.Tensor
            Tensor of previous time step indices (long), shape (batch_size,), where each
            value is in the range [0, variance_scheduler.tau_num_steps - 1].

        Returns
        -------
        xt_prev : torch.Tensor
            Denoised tensor at `prev_time_steps`, same shape as `xt`.
        x0 : torch.Tensor
            Estimated original data (t=0), same shape as `xt`.
        """
        if not torch.all((time_steps >= 0) & (time_steps < self.variance_scheduler.tau_num_steps)):
            raise ValueError(f"time_steps must be between 0 and {self.variance_scheduler.tau_num_steps - 1}")
        if not torch.all((prev_time_steps >= 0) & (prev_time_steps < self.variance_scheduler.tau_num_steps)):
            raise ValueError(f"prev_time_steps must be between 0 and {self.variance_scheduler.tau_num_steps - 1}")

        _, _, _, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod = self.variance_scheduler.get_tau_schedule()

        # check input dimensions and adjust reshaping for 2D or 4D tensors
        is_2d = xt.dim() == 2  # check if input is 2D (batch_size, embedding_dim)
        if is_2d:
            # for 2D inputs, reshape to [batch_size, 1]
            tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[time_steps].to(xt.device).view(-1, 1)
            tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[time_steps].to(xt.device).view(-1, 1)
            prev_tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1)
            prev_tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1)
        else:
            # for 4D inputs, reshape to [batch_size, 1, 1, 1]
            tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
            tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[time_steps].to(xt.device).view(-1, 1, 1, 1)
            prev_tau_sqrt_alpha_cumprod_t = tau_sqrt_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)
            prev_tau_sqrt_one_minus_alpha_cumprod_t = tau_sqrt_one_minus_alpha_cumprod[prev_time_steps].to(xt.device).view(-1, 1, 1, 1)

        eta = self.variance_scheduler.eta

        predicted_noise = None
        x0 = None
        # handle different prediction types
        if self.prediction_type == "noise":
            # model predicts noise
            predicted_noise = model_prediction
            x0 = (xt - tau_sqrt_one_minus_alpha_cumprod_t * predicted_noise) / tau_sqrt_alpha_cumprod_t
        elif self.prediction_type == "x0":
            # model predicts clean image
            x0 = model_prediction
            # calculate implied noise from the predicted clean image
            predicted_noise = (xt - tau_sqrt_alpha_cumprod_t * x0) / tau_sqrt_one_minus_alpha_cumprod_t

        # DDIM sampling step (same for both prediction types)
        noise_coeff = eta * ((tau_sqrt_one_minus_alpha_cumprod_t / prev_tau_sqrt_alpha_cumprod_t) *
                             prev_tau_sqrt_one_minus_alpha_cumprod_t / torch.clamp(tau_sqrt_one_minus_alpha_cumprod_t, min=1e-8))
        direction_coeff = torch.clamp(prev_tau_sqrt_one_minus_alpha_cumprod_t ** 2 - noise_coeff ** 2, min=1e-8).sqrt()
        xt_prev = prev_tau_sqrt_alpha_cumprod_t * x0 + noise_coeff * torch.randn_like(xt) + direction_coeff * predicted_noise

        return xt_prev, x0

    def set_prediction_type(self, prediction_type: str):
        """Change the prediction type after initialization.

        Parameters
        ----------
        prediction_type : str
            Type of prediction the model makes. Either "noise" or "x0".
        """
        if prediction_type not in ["noise", "x0"]:
            raise ValueError(f"prediction_type must be either 'noise' or 'x0', got {prediction_type}")
        self.prediction_type = prediction_type

###==================================================================================================================###

class CLIPEncoder(nn.Module):
    """Encodes images or text using a pre-trained CLIP model.

    Loads a CLIP model and processor from the transformers library, providing methods to
    encode images or text into embeddings and compute similarity scores between them.

    Parameters
    ----------
    `model_name` : str, optional
        Name of the CLIP model to load (default: 'openai/clip-vit-base-patch32').
    `device` : str, optional
        Device to run the model on (default: 'cuda' if available, else 'cpu').
    `use_fast` : bool, optional
        Whether to use the fast image processor (torchvision-based) (default: False).
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        use_fast: bool = False,
    ) -> None:
        super().__init__()

        # set model name and device
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # load CLIP model and processor
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=use_fast)
            self.model = self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model or processor for {self.model_name}: {e}")

        # set model to evaluation mode by default
        self.model.eval()

    def forward(
            self,
            data: Union[torch.Tensor, List[str], str, Image.Image, List[Image.Image]],
            data_type: str,
            normalize: bool = True
    ) -> torch.Tensor:
        """Encodes input data (image or text) using the CLIP model.

        Processes input data (images or text) to produce embeddings, with optional L2
        normalization.

        Parameters
        ----------
        `data` : Union[torch.Tensor, List[str], str, Image.Image, List[Image.Image]]
            Input data to encode:
                - torch.Tensor: Preprocessed image tensor (batch_size, channels, height, width).
                - List[str] or str: Text or list of texts.
                - PIL.Image.Image or List[PIL.Image.Image]: Single or list of PIL images.
        `data_type` : str
            Type of input data ('img' or 'text').
        `normalize` : bool, optional
            Whether to L2-normalize the output embeddings (default: True).

        Returns
        -------
        outputs : torch.Tensor
            Encoded embeddings, shape (batch_size, embedding_dim).
        """
        if data_type not in ["img", "text"]:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'img' or 'text'.")

        with torch.no_grad():
            if data_type == "img":
                outputs = self._encode_images(data)
            else:
                outputs = self._encode_texts(data)

            # normalize embeddings if requested
            if normalize:
                outputs = F.normalize(outputs, p=2, dim=-1)

            return outputs

    def _encode_images(self, data: Union[torch.Tensor, Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Encodes images into embeddings using the CLIP model.

        Processes image inputs (tensors or PIL images) to produce image embeddings.

        Parameters
        ----------
        `data` : Union[torch.Tensor, Image.Image, List[Image.Image]]
            Input images as a tensor or PIL image(s).

        Returns
        -------
        image_features : torch.Tensor
            Image embeddings, shape (batch_size, embedding_dim).
        """
        if isinstance(data, torch.Tensor):
            if data.dim() == 3:
                data = data.unsqueeze(0)
            inputs = {"pixel_values": data.to(self.device)}
        elif isinstance(data, (Image.Image, list)):
            if isinstance(data, Image.Image):
                data = [data]
            inputs = self.processor(images=data, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            raise ValueError(f"Invalid image data type: {type(data)}. Expected torch.Tensor, PIL.Image.Image, or List[PIL.Image.Image].")
        return self.model.get_image_features(**inputs)

    def _encode_texts(self, data: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        """Encodes texts into embeddings using the CLIP model.

        Processes text inputs (strings or tokenized tensors) to produce text embeddings.

        Parameters
        ----------
        `data` : Union[str, List[str], torch.Tensor]
            Input texts as strings or tokenized tensor.

        Returns
        -------
        text_features : torch.Tensor
            Text embeddings, shape (batch_size, embedding_dim).
        """
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
            if data.dim() == 2:
                return data
            if data.dim() == 1:
                data = data.unsqueeze(0)
            attention_mask = torch.ones_like(data)
            return self.model.get_text_features(input_ids=data, attention_mask=attention_mask)

        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list) and all(isinstance(t, str) for t in data):
            pass
        else:
            raise ValueError(
                f"Invalid text data type: {type(data)}. Expected str, List[str], or torch.Tensor."
            )

        inputs = self.processor(text=data, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.get_text_features(**inputs)

    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity between image and text embeddings.

        Calculates the cosine similarity matrix between batches of image and text embeddings.

        Parameters
        ----------
        `image_features` : torch.Tensor
            Image embeddings, shape (batch_size, embedding_dim).
        `text_features` : torch.Tensor
            Text embeddings, shape (batch_size, embedding_dim).

        Returns
        -------
        similarity : torch.Tensor
            Cosine similarity scores, shape (batch_size, batch_size).
        """
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return torch.matmul(image_features, text_features.T)

###==================================================================================================================###

class UnClipDecoder(nn.Module):
    """Decoder for UnCLIP diffusion models.

    Combines CLIP image embeddings and text embeddings to guide the denoising process,
    using a noise predictor and diffusion processes. Incorporates classifier-free guidance,
    text caption dropout, and projection of CLIP embeddings into context tokens.

    Parameters
    ----------
    `clip_embedding_dim` : int
        Dimensionality of the input embeddings.
    `noise_predictor` : nn.Module
        Model to predict noise during the denoising process.
    `forward_diffusion` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise.
    `reverse_diffusion` : nn.Module
        Reverse diffusion module (e.g., ReverseUnCLIP) for denoising.
    `glide_text_encoder` : nn.Module, optional
        GLIDE text encoder for processing text prompts, default None.
    `bert_tokenizer` : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    `device` : Union[str, torch.device], optional
        Device for computation (default: CUDA if available, else CPU).
    `image_output_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `normalize_clip_embeddings` : bool, optional
        Whether to normalize outputs (default: True).
    `classifier_free_prop` : float, optional
        Probability for classifier-free guidance (default: 0.1, per paper).
    `drop_caption` : float, optional
        Probability for text caption dropout (default: 0.5, per paper).
    `max_token_length` : int, optional
        Maximum length for tokenized prompts (default: 77).
    """
    def __init__(
            self,
            clip_embedding_dim: int,
            noise_predictor: nn.Module,
            forward_diffusion: nn.Module,
            reverse_diffusion: nn.Module,
            glide_text_encoder: torch.nn.Module = None,  # GLIDE text encoder
            bert_tokenizer: Optional[BertTokenizer] = None,
            device: Optional[Union[str, torch.device]] = None,
            image_output_range: Tuple[float, float] = (-1.0, 1.0),
            normalize_clip_embeddings: bool = True,
            classifier_free_prop: float = 0.1,  # paper specifies 10%
            drop_caption: float = 0.5,  # paper specifies 50%
            max_token_length: int = 77  # max_token_length for tokenization
    ) -> None:
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.clip_embedding_dim = clip_embedding_dim

        # core models
        self.noise_predictor = noise_predictor.to(self.device)
        self.forward_diffusion = forward_diffusion.to(self.device)
        self.reverse_diffusion = reverse_diffusion.to(self.device)
        self.glide_text_encoder = glide_text_encoder.to(self.device) if glide_text_encoder else None

        # paper: "projecting CLIP embeddings into four extra tokens of context"
        self.clip_decoder_projection = CLIPContextProjection(
            clip_embedding_dim=self.clip_embedding_dim,
            num_tokens=4
        ).to(self.device)
        self.clip_time_projection = nn.Linear(self.clip_embedding_dim, self.clip_embedding_dim).to(self.device)

        # training parameters
        self.image_output_range = image_output_range
        self.normalize_clip_embeddings = normalize_clip_embeddings
        self.classifier_free_prop = classifier_free_prop
        self.drop_caption = drop_caption
        self.max_token_length = max_token_length

        # initialize tokenizer
        if bert_tokenizer is None:
            try:
                self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")


    def forward(
            self,
            image_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor,
            images: torch.Tensor,
            texts: torch.Tensor,
            p_classifier_free: float,
            p_text_drop: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes embeddings and images to predict noise for training.

        Applies classifier-free guidance and text dropout, projects CLIP image embeddings
        into context tokens, encodes text with GLIDE, and predicts noise for the diffusion process.

        Parameters
        ----------
        `image_embeddings` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embedding_dim).
        `text_embeddings` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embedding_dim).
        `images` : torch.Tensor
            Input images, shape (batch_size, channels, height, width).
        `texts` : torch.Tensor
            Text prompts for conditional generation.
        `p_classifier_free` : float
            Probability for applying classifier-free guidance.
        `p_text_drop` : float
            Probability for applying text caption dropout.

        Returns
        -------
        predicted_noise : torch.Tensor
            Predicted noise tensor, shape (batch_size, channels, height, width).
        noise : torch.Tensor
            Ground truth noise tensor, shape (batch_size, channels, height, width).
        """

        image_embeddings = self._apply_classifier_free_guidance(image_embeddings, p_classifier_free)
        text_embeddings = self._apply_text_dropout(text_embeddings, p_text_drop)
        # project z_i to 4 tokens
        c = self.clip_decoder_projection(image_embeddings)
        # encode text with GLIDE
        y_encoded = self._encode_text_with_glide(texts if text_embeddings is not None else None)
        # concatenate embeddings
        context = self._concatenate_embeddings(y_encoded, c)
        # sample timestep and noise
        t, noise = self._sample_timestep_and_noise(images.shape[0], images.shape)
        # compute noisy image
        noisy_images = self.forward_diffusion(images, noise, t)
        clip_image_embedding = self.clip_time_projection(image_embeddings)
        predicted_noise = self.noise_predictor(noisy_images, t, context, clip_image_embedding)
        return predicted_noise, noise

    def inference_forward(self, image_embeddings, prompt_embeddings):
        pass

    def _apply_classifier_free_guidance(self, image_embeddings: torch.Tensor, p_value: float) -> torch.Tensor:
        """Applies classifier-free guidance to image embeddings.

        Sets image embeddings to zero with a specified probability to implement
        classifier-free guidance, as described in the UnCLIP paper.

        Parameters
        ----------
        `image_embeddings` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embedding_dim).
        `p_value` : float
            Probability for applying classifier-free guidance.

        Returns
        -------
        image_embeddings : torch.Tensor
            Modified image embeddings, shape (batch_size, embedding_dim).
        """
        if p_value < self.classifier_free_prop:
            # set z_i ← 0 {classifier-free guidance}
            image_embeddings = torch.zeros_like(image_embeddings)

        return image_embeddings

    def _apply_text_dropout(self, text_embeddings: torch.Tensor, p_value: float) -> Optional[torch.Tensor]:
        """Applies text caption dropout to text embeddings.

        Drops text embeddings with a specified probability to implement text dropout,
        as described in the UnCLIP paper.

        Parameters
        ----------
        `text_embeddings` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embedding_dim).
        `p_value` : float
            Probability for applying text caption dropout.

        Returns
        -------
        text_embeddings : torch.Tensor or None
            Modified text embeddings or None if dropped, shape (batch_size, embedding_dim).
        """
        if p_value < self.drop_caption:
            # set y ← ∅ {drop text caption}
            return None

        return text_embeddings


    def _encode_text_with_glide(self, texts: Union[List, torch.Tensor]) -> Optional[torch.Tensor]:
        """Encodes text prompts using the GLIDE text encoder.

        Tokenizes and encodes text prompts into embeddings using the GLIDE text encoder,
        returning None if no text or conditional model is provided.

        Parameters
        ----------
        `texts` : Union[List, torch.Tensor]
            Text prompts or tensor of text data.

        Returns
        -------
        y_encoded : torch.Tensor or None
            Encoded text embeddings, shape (batch_size, seq_len, embedding_dim), or None.
        """
        if texts is None:
            return None

        if self.glide_text_encoder is None:
            return None

        # convert to string list if needed
        if isinstance(texts, torch.Tensor):
            texts = texts.cpu().numpy().tolist()
        texts = [str(item) for item in texts]

        # tokenize
        tokenized = self.bert_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt"
        ).to(self.device)

        # get embeddings from GLIDE text encoder
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        y_encoded = self.glide_text_encoder(input_ids, attention_mask)
        # print("y shape: ", y_encoded.size())

        return y_encoded

    def _concatenate_embeddings(self, y_encoded: Optional[torch.Tensor], c: torch.Tensor) -> torch.Tensor:
        """Concatenates GLIDE text embeddings and context tokens.

        Combines encoded text embeddings (if available) with projected context tokens
        along the sequence dimension, as specified in the UnCLIP paper.

        Parameters
        ----------
        `y_encoded` : torch.Tensor or None
            Encoded text embeddings from GLIDE, shape (batch_size, seq_len, embedding_dim).
        `c` : torch.Tensor
            Projected context tokens, shape (batch_size, num_tokens, embedding_dim).

        Returns
        -------
        s : torch.Tensor
            Concatenated embeddings, shape (batch_size, seq_len + num_tokens, embedding_dim).
        """
        if y_encoded is not None:
            # ensure y_encoded has sequence dimension
            if len(y_encoded.shape) == 2:  # [batch_size, embed_dim]
                y_encoded = y_encoded.unsqueeze(1)  # [batch_size, 1, embed_dim]

            # concatenate along the sequence dimension
            s = torch.cat([y_encoded, c], dim=1)  # [batch_size, seq_len + 4, embed_dim]
        else:
            s = c  # [batch_size, 4, embed_dim]

        return s

    def _sample_timestep_and_noise(self, batch_size: int, image_shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples timesteps and noise for the diffusion process.

        Generates random timesteps and Gaussian noise for use in the forward diffusion process.

        Parameters
        ----------
        `batch_size` : int
            Number of samples in the batch.
        `image_shape` : torch.Size
            Shape of the images, typically (batch_size, channels, height, width).

        Returns
        -------
        t : torch.Tensor
            Sampled timestep indices, shape (batch_size,).
        noise : torch.Tensor
            Sampled Gaussian noise, shape (batch_size, channels, height, width).
        """
        # sample timestep t ~ Uniform(1, T)
        t = torch.randint(0, self.forward_diffusion.variance_scheduler.num_steps, (batch_size,), device=self.device)
        # sample noise ε ~ N(0, I)
        noise = torch.randn(image_shape, device=self.device)
        return t, noise

###==================================================================================================================###

class UnCLIPTransformerPrior(nn.Module):
    """Transformer-based prior model for UnCLIP diffusion.

    Predicts clean image embeddings from noisy image embeddings and text embeddings using
    a Transformer architecture, incorporating time embeddings and optional projection
    layers for text and image inputs.

    Parameters
    ----------
    `forward_diffusion` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise during training.
    `reverse_diffusion` : nn.Module
        Reverse diffusion module (e.g., ReverseUnCLIP) for denoising during training.
    `clip_text_projection` : nn.Module, optional
        Projection module for text embeddings, default None.
    `clip_image_projection` : nn.Module, optional
        Projection module for image embeddings, default None.
    `transformer_embedding_dim` : int, optional
        Dimensionality of embeddings (default: 320).
    `num_layers` : int, optional
        Number of Transformer layers (default: 12).
    `num_attention_heads` : int, optional
        Number of attention heads in each Transformer layer (default: 8).
    `feedforward_dim` : int, optional
        Dimensionality of the feedforward network in Transformer layers (default: 768).
    `max_sequence_length` : int, optional
        Maximum sequence length for input embeddings (default: 2).
    `dropout_rate` : float, optional
        Dropout probability for regularization (default: 0.2).
    """
    def __init__(
        self,
        forward_diffusion: nn.Module, # will be used during training
        reverse_diffusion: nn.Module, # will be used during training
        clip_text_projection: Optional[nn.Module] = None,  # used during training instead of PCA in the main paper
        clip_image_projection: Optional[nn.Module] = None,  # used during training instead of PCA in the main paper
        transformer_embedding_dim: int = 320,
        num_layers: int = 12,
        num_attention_heads: int = 8,
        feedforward_dim: int = 768,
        max_sequence_length: int = 2,
        dropout_rate: float = 0.2
    ) -> None:
        super().__init__()

        self.forward_diffusion = forward_diffusion
        self.reverse_diffusion = reverse_diffusion
        self.clip_text_projection = clip_text_projection
        self.clip_image_projection = clip_image_projection

        self.transformer_embedding_dim = transformer_embedding_dim
        self.max_sequence_length = max_sequence_length

        # time embedding network
        self.time_embedding_net = nn.Sequential(
            nn.Linear(transformer_embedding_dim, transformer_embedding_dim),
            nn.GELU(),
            nn.Linear(transformer_embedding_dim, transformer_embedding_dim)
        )

        # positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(max_sequence_length, transformer_embedding_dim))

        # transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_embedding_dim, num_attention_heads, feedforward_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        # final output projection
        self.output_projection = nn.Linear(transformer_embedding_dim, transformer_embedding_dim)

    def forward(
            self,
            text_embeddings: torch.Tensor,
            noisy_image_embeddings: torch.Tensor,
            timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Predicts clean image embeddings from noisy inputs and text embeddings.

        Processes text and noisy image embeddings through a Transformer architecture,
        conditioned on time embeddings, to predict the clean image embeddings.

        Parameters
        ----------
        `text_embeddings` : torch.Tensor
            Text embeddings, shape (batch_size, embedding_dim).
        `noisy_image_embeddings` : torch.Tensor
            Noisy image embeddings, shape (batch_size, embedding_dim).
        `timesteps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,).

        Returns
        -------
        predicted_clean_embeddings : torch.Tensor
            Predicted clean image embeddings, shape (batch_size, embedding_dim).
        """
        device = text_embeddings.device
        # create sinusoidal time embeddings
        time_embeddings = self._get_sinusoidal_embeddings(timesteps, self.transformer_embedding_dim, device)
        time_embeddings = self.time_embedding_net(time_embeddings)
        # add time information to image embeddings
        conditioned_image_embeddings = noisy_image_embeddings + time_embeddings
        # create sequence: [text_embeddings, conditioned_image_embeddings]
        sequence = torch.stack([text_embeddings, conditioned_image_embeddings], dim=1)  # [B, 2, D]
        # add positional embeddings
        sequence = sequence + self.positional_embeddings.unsqueeze(0)
        # pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            sequence = transformer_block(sequence)
        # extract predicted clean image embedding (second position in sequence)
        predicted_clean_embeddings = sequence[:, 1, :]  # [B, D]
        # apply final projection
        predicted_clean_embeddings = self.output_projection(predicted_clean_embeddings)

        return predicted_clean_embeddings

    def _get_sinusoidal_embeddings(
            self,
            timesteps: torch.Tensor,
            embedding_dim: int,
            device: Union[torch.device, str]
    ) -> torch.Tensor:
        """Generates sinusoidal positional embeddings for timesteps.

        Creates sinusoidal embeddings for the given timesteps to condition the Transformer
        on the diffusion process time steps.

        Parameters
        ----------
        `timesteps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,).
        `embedding_dim` : int
            Dimensionality of the embeddings.
        `device` : Union[torch.device, str]
            Device to place the embeddings on.

        Returns
        -------
        embeddings : torch.Tensor
            Sinusoidal time embeddings, shape (batch_size, embedding_dim).
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # handle odd embedding dimensions
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

        return emb


class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head attention and feedforward layers.

    Implements a Transformer block with multi-head self-attention, layer normalization,
    and a feedforward network with residual connections for processing sequences in
    the UnCLIPTransformerPrior model.

    Parameters
    ----------
    `embedding_dim` : int
        Dimensionality of input and output embeddings.
    `num_heads` : int
        Number of attention heads in the multi-head attention layer.
    `feedforward_dim` : int
        Dimensionality of the feedforward network.
    `dropout` : float
        Dropout probability for regularization.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            feedforward_dim: int,
            dropout: float
    ) -> None:
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feedforward_norm = nn.LayerNorm(embedding_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input sequence through the Transformer block.

        Applies multi-head self-attention followed by a feedforward network, with residual
        connections and layer normalization.

        Parameters
        ----------
        `x` : torch.Tensor
            Input sequence tensor, shape (batch_size, sequence_length, embedding_dim).

        Returns
        -------
        output : torch.Tensor
            Processed sequence tensor, shape (batch_size, sequence_length, embedding_dim).
        """
        # self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x)
        x = self.attention_norm(x + attn_output)

        # feedforward with residual connection
        ff_output = self.feedforward(x)
        x = self.feedforward_norm(x + ff_output)

        return x

###==================================================================================================================###

class CLIPContextProjection(nn.Module):
    """Projects CLIP image embeddings into multiple context tokens.

    Transforms a single CLIP image embedding into a specified number of context tokens
    using a linear projection followed by layer normalization.

    Parameters
    ----------
    `clip_embedding_dim` : int
        Dimensionality of the input CLIP embedding (e.g., 319 or 512).
    `num_tokens` : int, optional
        Number of context tokens to generate (default: 4).
    """
    def __init__(self, clip_embedding_dim, num_tokens=4):
        super().__init__()
        self.clip_embedding_dim = clip_embedding_dim
        self.num_tokens = num_tokens
        self.clip_projection = nn.Linear(clip_embedding_dim, clip_embedding_dim * num_tokens)
        self.clip_embedding_norm = nn.LayerNorm(clip_embedding_dim)

    def forward(self, z_i):
        """Projects CLIP image embedding into context tokens.

        Applies a linear projection to transform the input embedding into multiple tokens,
        reshapes the output, and applies layer normalization.

        Parameters
        ----------
        `z_i` : torch.Tensor
            Input CLIP image embedding, shape (batch_size, input_dim).

        Returns
        -------
        c : torch.Tensor
            Context tokens, shape (batch_size, num_tokens, input_dim).
        """
        batch_size = z_i.shape[0]
        projected = self.clip_projection(z_i)
        c = projected.view(batch_size, self.num_tokens, self.clip_embedding_dim)
        c = self.clip_embedding_norm(c)
        return c

###==================================================================================================================###

class CLIPEmbeddingProjection(nn.Module):
    """Projection module for dimensionality reduction and reconstruction.

    Implements a neural network with forward and inverse projections to reduce and
    restore input dimensionality, supporting customizable hidden layers, dropout, and
    layer normalization.

    Parameters
    ----------
    `clip_embedding_dim` : int, optional
        Input dimensionality (default: 1024).
    `transformer_embedding_dim` : int, optional
        Output dimensionality for forward projection (default: 320).
    `hidden_dim` : int, optional
        Hidden layer dimensionality (default: 512).
    `num_layers` : int, optional
        Number of layers in the projection network (default: 2).
    `dropout_rate` : float, optional
        Dropout probability for regularization (default: 0.2).
    `use_layer_norm` : bool, optional
        Whether to apply layer normalization after hidden layers (default: True).
    """
    def __init__(
        self,
        clip_embedding_dim: int = 1024,
        transformer_embedding_dim: int = 320,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        use_layer_norm: bool = True
    ) -> None:
        super().__init__()

        self.clip_embedding_dim = clip_embedding_dim
        self.transformer_embedding_dim = transformer_embedding_dim

        # Forward projection: input_dim -> output_dim
        self.forward_projection = self._build_projection_network(
            clip_embedding_dim, transformer_embedding_dim, hidden_dim, num_layers, dropout_rate, use_layer_norm
        )

        # Inverse projection: output_dim -> input_dim
        self.inverse_projection = self._build_projection_network(
            transformer_embedding_dim, clip_embedding_dim, hidden_dim, num_layers, dropout_rate, use_layer_norm
        )
    def _build_projection_network(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            use_layer_norm: bool
    ) -> nn.Sequential:
        """Builds a projection network with customizable layers.

        Constructs a neural network with linear layers, optional layer normalization,
        GELU activation, and dropout for either forward or inverse projection.

        Parameters
        ----------
        `input_dim` : int
            Input dimensionality for the network.
        `output_dim` : int
            Output dimensionality for the network.
        `hidden_dim` : int
            Hidden layer dimensionality.
        `num_layers` : int
            Number of layers in the network.
        `dropout` : float
            Dropout probability for regularization.
        `use_layer_norm` : bool
            Whether to apply layer normalization after hidden layers.

        Returns
        -------
        network : nn.Sequential
            Sequential container of the projection network layers.
        """
        layers = []
        current_dim = input_dim

        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input to a lower-dimensional space.

        Applies the forward projection network to reduce the dimensionality of the input tensor.

        Parameters
        ----------
        `x` : torch.Tensor
            Input tensor to be projected, shape (batch_size, input_dim).

        Returns
        -------
        x_reduced : torch.Tensor
            Projected tensor, shape (batch_size, output_dim).
        """
        return self.forward_projection(x)

    def inverse_transform(self, x_reduced: torch.Tensor) -> torch.Tensor:
        """Reconstructs input from lower-dimensional space.

        Applies the inverse projection network to restore the original dimensionality
        of the input tensor.

        Parameters
        ----------
        `x_reduced` : torch.Tensor
            Reduced-dimensionality tensor, shape (batch_size, output_dim).

        Returns
        -------
        x_reconstructed : torch.Tensor
            Reconstructed tensor, shape (batch_size, input_dim).
        """
        return self.inverse_projection(x_reduced)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the reconstruction loss for the projection.

        Calculates the mean squared error between the original input and its reconstruction
        after forward and inverse projections.

        Parameters
        ----------
        `x` : torch.Tensor
            Original input tensor, shape (batch_size, input_dim).

        Returns
        -------
        loss : torch.Tensor
            Mean squared error loss between the original and reconstructed tensors.
        """
        x_reduced = self.forward(x)
        x_reconstructed = self.inverse_transform(x_reduced)
        return F.mse_loss(x_reconstructed, x)

###==================================================================================================================###

class TrainUnClipDecoder(nn.Module):
    """Trainer for the UnCLIP decoder model.

    Orchestrates the training of the UnCLIP decoder model, integrating CLIP embeddings, forward
    and reverse diffusion processes, and optional dimensionality reduction. Supports mixed
    precision, gradient accumulation, DDP, and comprehensive evaluation metrics.

    Parameters
    ----------
    `clip_embedding_dim` : int
        Dimensionality of the input embeddings.
    `decoder_model` : nn.Module
        The UnCLIP decoder model (e.g., UnClipDecoder) to be trained.
    `clip_model` : nn.Module
        CLIP model for generating text and image embeddings.
    `train_loader` : torch.utils.data.DataLoader
        DataLoader for training data.
    `optimizer` : torch.optim.Optimizer
        Optimizer for training the decoder model.
    `objective` : Callable
        Loss function to compute the difference between predicted and target noise.
    `clip_text_projection` : nn.Module, optional
        Projection module for text embeddings, default None.
    `clip_image_projection` : nn.Module, optional
        Projection module for image embeddings, default None.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `metrics_` : Any, optional
        Object providing evaluation metrics (e.g., FID, MSE, PSNR, SSIM, LPIPS), default None.
    `max_epochs` : int, optional
        Maximum number of training epochs (default: 1000).
    `device` : Union[str, torch.device], optional
        Device for computation (default: CUDA if available, else CPU).
    `store_path` : str, optional
        Directory to save model checkpoints (default: "unclip_decoder").
    `patience` : int, optional
        Number of epochs to wait for improvement before early stopping (default: 100).
    `warmup_epochs` : int, optional
        Number of epochs for learning rate warmup (default: 100).
    `val_frequency` : int, optional
        Frequency (in epochs) for validation (default: 10).
    `use_ddp` : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    `grad_accumulation_steps` : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    `log_frequency` : int, optional
        Frequency (in epochs) for printing progress (default: 1).
    `use_compilation` : bool, optional
        Whether to compile the model using torch.compile (default: False).
    `image_output_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `reduce_clip_embedding_dim` : bool, optional
        Whether to apply dimensionality reduction to embeddings (default: True).
    `transformer_embedding_dim` : int, optional
        Output dimensionality for reduced embeddings (default: 312).
    `normalize_clip_embeddings` : bool, optional
        Whether to normalize CLIP embeddings (default: True).
    `finetune_clip_projections` : bool, optional
        Whether to fine-tune projection layers (default: False).
    """
    def __init__(
            self,
            clip_embedding_dim: int,
            decoder_model: nn.Module,
            clip_model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            clip_text_projection: Optional[nn.Module] = None,
            clip_image_projection: Optional[nn.Module] = None,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            metrics_: Optional[Any] = None,
            max_epochs: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            store_path: str = "unclip_decoder",
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            use_ddp: bool = False,
            grad_accumulation_steps: int = 1,
            log_frequency: int = 1,
            use_compilation: bool = False,
            image_output_range: Tuple[float, float] = (-1.0, 1.0),
            reduce_clip_embedding_dim: bool = True,
            transformer_embedding_dim: int = 312,
            normalize_clip_embeddings: bool = True,
            finetune_clip_projections: bool = False # if text_projection and image_projection model should be finetune
    ):
        super().__init__()
        # training configuration
        self.use_ddp = use_ddp
        self.grad_accumulation_steps = grad_accumulation_steps
        self.use_compilation = use_compilation
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # core models
        self.decoder_model = decoder_model.to(self.device)
        self.clip_model = clip_model.to(self.device)

        self.reduce_clip_embedding_dim = reduce_clip_embedding_dim

        # setup distributed training
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # compile and wrap models
        self._compile_models()
        self._wrap_models_for_ddp()

        # projection models (PCA equivalent in the paper)
        if self.reduce_clip_embedding_dim and clip_text_projection is not None and clip_image_projection is not None:
            self.clip_text_projection = clip_text_projection.to(self.device)
            self.clip_image_projection = clip_image_projection.to(self.device)
        else:
            self.clip_text_projection = None
            self.clip_image_projection = None

        # training components
        self.clip_embedding_dim = transformer_embedding_dim if self.reduce_clip_embedding_dim else clip_embedding_dim
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.train_loader = train_loader
        self.val_loader = val_loader

        # training parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frequency = val_frequency
        self.log_frequency = log_frequency
        self.image_output_range = image_output_range
        self.reduce_clip_embedding_dim = reduce_clip_embedding_dim
        self.normalize_clip_embeddings = normalize_clip_embeddings
        self.transformer_embedding_dim = transformer_embedding_dim
        self.finetune_clip_projections = finetune_clip_projections


        # checkpoint management
        self.store_path = store_path

        # learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)

    def forward(self) -> Tuple[List[float], float]:
        """Trains the UnCLIP decoder model to predict noise for denoising.

        Executes the training loop, optimizing the decoder model using CLIP embeddings, mixed
        precision, gradient clipping, and learning rate scheduling. Supports validation, early
        stopping, and checkpointing.

        Returns
        -------
        train_losses : List[float]
            List of mean training losses per epoch.
        best_val_loss : float
            Best validation or training loss achieved.
        """
        # set models to training mode
        self.decoder_model.train()  # sets noise_predictor, conditional_model, variance_scheduler, clip_time_proj to train mode
        if not self.decoder_model.forward_diffusion.variance_scheduler.trainable_beta:  # ff beta is not trainable
            self.decoder_model.forward_diffusion.variance_scheduler.eval()

        # set text_projection and image_projection to train mode if fine-tuning
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            if self.finetune_clip_projections:
                self.clip_text_projection.train()
                self.clip_image_projection.train()
            else:
                self.clip_text_projection.eval()
                self.clip_image_projection.eval()

        # set CLIP model to eval mode (frozen)
        if self.clip_model is not None:
            self.clip_model.eval()

        # initialize training components
        scaler = torch.GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        # main training loop
        for epoch in range(self.max_epochs):
            # set epoch for distributed sampler if using DDP
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []

            # training step loop with gradient accumulation
            for step, (images, texts) in enumerate(tqdm(self.train_loader, disable=not self.master_process)):
                images = images.to(self.device, non_blocking=True)

                # forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    # encode text and image with CLIP
                    text_embeddings, image_embeddings = self._get_clip_embeddings(images, texts)

                    # reduce dimensionality (PCA equivalent)
                    text_embeddings, image_embeddings = self._apply_dimensionality_reduction(
                        text_embeddings, image_embeddings
                    )

                    # use decoder model to predict noise
                    p_classifier_free = torch.rand(1).item()
                    p_text_drop = torch.rand(1).item()
                    predicted_noise, noise = self.decoder_model(
                        image_embeddings,
                        text_embeddings,
                        images,
                        texts,
                        p_classifier_free,
                        p_text_drop
                    )

                    # compute loss
                    loss = self.objective(predicted_noise, noise) / self.grad_accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % self.grad_accumulation_steps == 0:
                    # clip gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.decoder_model.parameters(), max_norm=1.0)  # covers all submodules
                    if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None and self.finetune_clip_projections:
                        torch.nn.utils.clip_grad_norm_(self.clip_text_projection.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.clip_image_projection.parameters(), max_norm=1.0)

                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    self.warmup_lr_scheduler.step()
                    torch.cuda.empty_cache()  # clear memory after optimizer step

                train_losses_epoch.append(loss.item() * self.grad_accumulation_steps)

            mean_train_loss = self._compute_mean_loss(train_losses_epoch)
            train_losses.append(mean_train_loss)

            if self.master_process and (epoch + 1) % self.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}")

            current_loss = mean_train_loss

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

            self.scheduler.step(current_loss)

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

        if self.use_ddp:
            destroy_process_group()

        return train_losses, best_val_loss

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
        """Wraps models with DistributedDataParallel for multi-GPU training.

        Configures the decoder model and, if fine-tuning, the projection models for DDP training.
        """
        if self.use_ddp:
            self.decoder_model = self.decoder_model.to(self.ddp_local_rank)
            self.decoder_model = DDP(
                self.decoder_model,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )
            # only wrap text_projection and image_projection if they are trainable
            if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None and self.finetune_clip_projections:
                self.clip_text_projection = self.clip_text_projection.to(self.ddp_local_rank)
                self.clip_image_projection = self.clip_image_projection.to(self.ddp_local_rank)
                self.clip_text_projection = DDP(self.clip_text_projection, device_ids=[self.ddp_local_rank])
                self.clip_image_projection = DDP(self.clip_image_projection, device_ids=[self.ddp_local_rank])

    def _compile_models(self) -> None:
        """Compiles models for optimization if supported.

        Attempts to compile the decoder model and, if fine-tuning, the projection models using
        torch.compile for optimization, falling back to uncompiled execution if compilation fails.
        """
        if self.use_compilation:
            try:
                self.decoder_model = self.decoder_model.to(self.device)
                self.decoder_model = torch.compile(self.decoder_model, mode="reduce-overhead")
                # only compile text_projection and image_projection if they are trainable
                if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None and self.finetune_clip_projections:
                    self.clip_text_projection = self.clip_text_projection.to(self.device)
                    self.clip_image_projection = self.clip_image_projection.to(self.device)
                    self.clip_text_projection = torch.compile(self.clip_text_projection, mode="reduce-overhead")
                    self.clip_image_projection = torch.compile(self.clip_image_projection, mode="reduce-overhead")
                if self.master_process:
                    print("Models compiled successfully")
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

    def _get_clip_embeddings(
            self,
            images: torch.Tensor,
            texts: Union[List, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes images and texts using the CLIP model.

        Generates text and image embeddings using the CLIP model, with optional normalization.

        Parameters
        ----------
        `images` : torch.Tensor
            Input images, shape (batch_size, channels, height, width).
        `texts` : Union[List, torch.Tensor]
            Text prompts for conditional generation.

        Returns
        -------
        text_embeddings : torch.Tensor
            CLIP text embeddings, shape (batch_size, embedding_dim).
        image_embeddings : torch.Tensor
            CLIP image embeddings, shape (batch_size, embedding_dim).
        """
        with torch.no_grad():
            # encode text y with CLIP text encoder: z_t ← CLIP_text(y)
            text_embeddings = self.clip_model(data=texts, data_type="text", normalize=self.normalize_clip_embeddings)
            # encode image x with CLIP image encoder: z_i ← CLIP_image(x)
            image_embeddings = self.clip_model(data=images, data_type="img", normalize=self.normalize_clip_embeddings)
        return text_embeddings, image_embeddings

    def _apply_dimensionality_reduction(
            self,
            text_embeddings: torch.Tensor,
            image_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies dimensionality reduction to embeddings if enabled.

        Projects text and image embeddings to a lower-dimensional space using learned
        projection layers, mimicking PCA as used in the UnCLIP paper.

        Parameters
        ----------
        `text_embeddings` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embedding_dim).
        `image_embeddings` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embedding_dim).

        Returns
        -------
        text_embeddings : torch.Tensor
            Projected text embeddings, shape (batch_size, output_dim) if reduced, else unchanged.
        image_embeddings : torch.Tensor
            Projected image embeddings, shape (batch_size, output_dim) if reduced, else unchanged.
        """
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            if not self.finetune_clip_projections:
                with torch.no_grad():
                    text_embeddings = self.clip_text_projection(text_embeddings.to(self.device))
                    image_embeddings = self.clip_image_projection(image_embeddings.to(self.device))
            else:
                text_embeddings = self.clip_text_projection(text_embeddings.to(self.device))
                image_embeddings = self.clip_image_projection(image_embeddings.to(self.device))
        return text_embeddings.to(self.device), image_embeddings.to(self.device)

    def _compute_mean_loss(self, losses: List[float]) -> float:
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

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False, suffix: str = ""):
        """Saves model checkpoint.

        Saves the state of the decoder model, its submodules, optimizer, and schedulers,
        with options for best model and epoch-specific checkpoints.

        Parameters
        ----------
        `epoch` : int
            Current epoch number.
        `loss` : float
            Current loss value.
        `is_best` : bool, optional
            Whether to save as the best model checkpoint (default: False).
        `suffix` : str, optional
            Suffix to add to checkpoint filename, default "".
        """
        if not self.master_process:
            return
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            # core models (submodules of decoder_model)
            'noise_predictor_state_dict': self.decoder_model.module.noise_predictor.state_dict() if self.use_ddp else self.decoder_model.noise_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # training configuration
            'embedding_dim': self.clip_embedding_dim,
            'output_dim': self.transformer_embedding_dim,
            'reduce_dim': self.reduce_clip_embedding_dim,
            'normalize': self.normalize_clip_embeddings
        }

        # save conditional model (submodule of decoder_model)
        if self.decoder_model.glide_text_encoder is not None:
            checkpoint['conditional_model_state_dict'] = (
                self.decoder_model.module.glide_text_encoder.state_dict() if self.use_ddp
                else self.decoder_model.glide_text_encoder.state_dict()
            )

        # save variance scheduler (submodule of decoder_model, always saved)
        checkpoint['variance_scheduler_state_dict'] = (
            self.decoder_model.forward_diffusion.module.variance_scheduler.state_dict() if self.use_ddp
            else self.decoder_model.forward_diffusion.variance_scheduler.state_dict()
        )

        # save CLIP time projection layer (submodule of decoder_model)
        checkpoint['clip_time_proj_state_dict'] = (
            self.decoder_model.module.clip_time_projection.state_dict() if self.use_ddp
            else self.decoder_model.clip_time_projection.state_dict()
        )

        # save decoder projection layer (submodule of decoder_model)
        checkpoint['decoder_projection_state_dict'] = (
            self.decoder_model.module.clip_decoder_projection.state_dict() if self.use_ddp
            else self.decoder_model.clip_decoder_projection.state_dict()
        )
        # a nn.Linear projection layer
        checkpoint['clip_time_projection_state_dict'] = (
            self.decoder_model.module.clip_time_projection.state_dict() if self.use_ddp
            else self.decoder_model.clip_time_projection.state_dict()
        )

        # save projection models (PCA equivalent)
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            checkpoint['text_projection_state_dict'] = (
                self.clip_text_projection.module.state_dict() if self.use_ddp
                else self.clip_text_projection.state_dict()
            )
            checkpoint['image_projection_state_dict'] = (
                self.clip_image_projection.module.state_dict() if self.use_ddp
                else self.clip_image_projection.state_dict()
            )

        # save schedulers state
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        checkpoint['warmup_scheduler_state_dict'] = self.warmup_lr_scheduler.state_dict()

        filename = f"unclip_decoder_epoch_{epoch}{suffix}.pth"
        if is_best:
            filename = f"unclip_decoder_best{suffix}.pth"

        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)

        if is_best:
            print(f"Best model saved: {filepath}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """Loads model checkpoint.

        Restores the state of the decoder model, its submodules, optimizer, and schedulers
        from a saved checkpoint, handling DDP compatibility.

        Parameters
        ----------
        `checkpoint_path` : str
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
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        def _load_model_state_dict(model: nn.Module, state_dict: dict, model_name: str) -> None:
            """Helper function to load state dict with DDP compatibility."""
            try:
                # handle DDP state dict compatibility
                if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                model.load_state_dict(state_dict)
                if self.master_process:
                    print(f"✓ Loaded {model_name}")
            except Exception as e:
                warnings.warn(f"Failed to load {model_name}: {e}")

        # load core noise predictor model (submodule of decoder_model)
        if 'noise_predictor_state_dict' in checkpoint:
            _load_model_state_dict(self.decoder_model.noise_predictor, checkpoint['noise_predictor_state_dict'],
                                   'noise_predictor')

        # load conditional model (submodule of decoder_model) - matches your save logic
        if self.decoder_model.glide_text_encoder is not None and 'conditional_model_state_dict' in checkpoint:
            _load_model_state_dict(self.decoder_model.glide_text_encoder, checkpoint['conditional_model_state_dict'],
                                   'glide_text_encoder')

        # load variance scheduler (submodule of decoder_model)
        if 'variance_scheduler_state_dict' in checkpoint:
            try:
                _load_model_state_dict(self.decoder_model.forward_diffusion.variance_scheduler,
                                       checkpoint['variance_scheduler_state_dict'], 'variance_scheduler')
            except Exception as e:
                warnings.warn(f"Failed to load variance scheduler: {e}")

        # load CLIP time projection layer (submodule of decoder_model)
        if 'clip_time_proj_state_dict' in checkpoint:
            try:
                _load_model_state_dict(self.decoder_model.clip_time_projection,
                                       checkpoint['clip_time_proj_state_dict'], 'clip_time_projection')
            except Exception as e:
                warnings.warn(f"Failed to load CLIP time projection: {e}")

        # load decoder projection layer (submodule of decoder_model)
        if 'decoder_projection_state_dict' in checkpoint:
            try:
                _load_model_state_dict(self.decoder_model.clip_decoder_projection,
                                       checkpoint['decoder_projection_state_dict'], 'clip_decoder_projection')
            except Exception as e:
                warnings.warn(f"Failed to load decoder projection: {e}")

        # handle the duplicate clip_time_projection_state_dict (from your save function)
        # This loads the same thing as clip_time_proj_state_dict above, so we'll skip it
        # to avoid overwriting, but add a warning if it exists
        if 'clip_time_projection_state_dict' in checkpoint and self.master_process:
            warnings.warn(
                "Found duplicate 'clip_time_projection_state_dict' in checkpoint - skipping to avoid conflict")

        # load projection models (PCA equivalent)
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            if 'text_projection_state_dict' in checkpoint:
                _load_model_state_dict(self.clip_text_projection, checkpoint['text_projection_state_dict'],
                                       'text_projection')
            if 'image_projection_state_dict' in checkpoint:
                _load_model_state_dict(self.clip_image_projection, checkpoint['image_projection_state_dict'],
                                       'image_projection')

        # load optimizer
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            if checkpoint['embedding_dim'] != self.clip_embedding_dim:
                warnings.warn(
                    f"Embedding dimension mismatch: checkpoint={checkpoint['embedding_dim']}, current={self.clip_embedding_dim}")

        if 'reduce_dim' in checkpoint:
            if checkpoint['reduce_dim'] != self.reduce_clip_embedding_dim:
                warnings.warn(
                    f"Reduce dimension setting mismatch: checkpoint={checkpoint['reduce_dim']}, current={self.reduce_clip_embedding_dim}")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
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

        # set models to eval mode for evaluation
        self.decoder_model.eval()  # sets noise_predictor, conditional_model, variance_scheduler, clip_time_proj, decoder_projection to eval mode
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            self.clip_text_projection.eval()
            self.clip_image_projection.eval()
        if self.clip_model is not None:
            self.clip_model.eval()

        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []

        with torch.no_grad():
            for images, texts in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                images_orig = images.clone()
                text_embeddings, image_embeddings = self._get_clip_embeddings(images, texts)
                text_embeddings, image_embeddings = self._apply_dimensionality_reduction(
                    text_embeddings, image_embeddings
                )
                p_classifier_free = torch.rand(1).item()
                p_text_drop = torch.rand(1).item()
                predicted_noise, noise = self.decoder_model(
                    image_embeddings,
                    text_embeddings,
                    images,
                    texts,
                    p_classifier_free,
                    p_text_drop
                )
                loss = self.objective(predicted_noise, noise)
                val_losses.append(loss.item())

                if self.metrics_ is not None and self.decoder_model.reverse_diffusion is not None:
                    xt = torch.randn_like(images).to(self.device)
                    for t in reversed(range(self.decoder_model.forward_diffusion.variance_scheduler.tau_num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        prev_time_steps = torch.full((xt.shape[0],), max(t - 1, 0), device=self.device, dtype=torch.long)
                        image_embeddings = self.decoder_model._apply_classifier_free_guidance(image_embeddings, p_classifier_free)
                        text_embeddings = self.decoder_model._apply_text_dropout(text_embeddings, p_text_drop)
                        c = self.decoder_model.clip_decoder_projection(image_embeddings)
                        y_encoded = self.decoder_model._encode_text_with_glide(texts if text_embeddings is not None else None)
                        context = self.decoder_model._concatenate_embeddings(y_encoded, c)
                        clip_image_embedding = self.decoder_model.clip_time_projection(image_embeddings)
                        predicted_noise = self.decoder_model.noise_predictor(xt, time_steps, context, clip_image_embedding)
                        xt, _ = self.decoder_model.reverse_diffusion(xt, predicted_noise, time_steps, prev_time_steps)

                    x_hat = torch.clamp(xt, min=self.image_output_range[0], max=self.image_output_range[1])

                    if self.normalize_clip_embeddings:
                        x_hat = (x_hat - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])
                        x_orig = (images_orig - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])

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

        # compute averages
        val_loss = torch.tensor(val_losses).mean().item()
        fid_avg = torch.tensor(fid_scores).mean().item() if fid_scores else float('inf')
        mse_avg = torch.tensor(mse_scores).mean().item() if mse_scores else None
        psnr_avg = torch.tensor(psnr_scores).mean().item() if psnr_scores else None
        ssim_avg = torch.tensor(ssim_scores).mean().item() if ssim_scores else None
        lpips_avg = torch.tensor(lpips_scores).mean().item() if lpips_scores else None

        # synchronize metrics across GPUs in DDP mode
        if self.use_ddp:
            metrics = [val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg]
            metrics_tensors = [torch.tensor(m, device=self.device) if m is not None else torch.tensor(float('inf'), device=self.device) for m in metrics]
            for tensor in metrics_tensors:
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg = [t.item() if t.item() != float('inf') else (None if i > 1 else float('inf')) for i, t in enumerate(metrics_tensors)]

        # return to training mode
        self.decoder_model.train()  # sets noise_predictor, conditional_model, variance_scheduler, clip_time_proj, decoder_projection to train mode
        if not self.decoder_model.forward_diffusion.variance_scheduler.trainable_beta:
            self.decoder_model.forward_diffusion.variance_scheduler.eval()
            self.decoder_model.reverse_diffusion.variance_scheduler.eval()
        if self.reduce_clip_embedding_dim and self.clip_text_projection is not None and self.clip_image_projection is not None:
            if self.finetune_clip_projections:
                self.clip_text_projection.train()
                self.clip_image_projection.train()
            else:
                self.clip_text_projection.eval()
                self.clip_image_projection.eval()
        if self.clip_model is not None:
            self.clip_model.eval()

        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg

###==================================================================================================================###

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

        # training configuration
        self.use_ddp = use_ddp
        self.grad_accumulation_steps = grad_accumulation_steps
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # setup distributed training
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # core models
        self.prior_model = prior_model.to(self.device)
        self.clip_model = clip_model.to(self.device)

        # training components
        self.optimizer = optimizer
        self.objective = objective
        self.train_loader = train_loader
        self.val_loader = val_loader

        # training parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frequency = val_frequency
        self.log_frequency = log_frequency
        self.use_compilation = use_compilation
        self.embedding_output_range = embedding_output_range
        self.reduce_clip_embedding_dim = reduce_clip_embedding_dim
        self.normalize_clip_embeddings = normalize_clip_embeddings
        self.transformer_embedding_dim = transformer_embedding_dim

        # checkpoint management
        self.store_path = store_path

        # learning rate scheduling
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

        # ensure CUDA is available for DDP
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA but CUDA is not available")

        # initialize process group only if not already initialized
        if not torch.distributed.is_initialized():
            init_process_group(backend="nccl")

        # get rank information
        self.ddp_rank = int(os.environ["RANK"])  # global rank across all nodes
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])  # local rank on current node
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])  # total number of processes

        # set device and make it current
        self.device = torch.device(f"cuda:{self.ddp_local_rank}")
        torch.cuda.set_device(self.device)

        # master process handles logging, checkpointing, etc.
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
            # wrap prior with DDP
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
        # set models to training mode
        self.prior_model.train()

        # compile and wrap models
        self._compile_models()
        self._wrap_models_for_ddp()

        # initialize training components
        scaler = torch.GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        # main training loop
        for epoch in range(self.max_epochs):
            # set epoch for distributed sampler if using DDP
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []

            # training step loop with gradient accumulation
            for step, (x, y) in enumerate(tqdm(self.train_loader, disable=not self.master_process)):
                x = x.to(self.device, non_blocking=True)

                # forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    loss = self._compute_training_loss(x, y)
                    loss = loss / self.grad_accumulation_steps

                # backward pass
                scaler.scale(loss).backward()

                # optimizer step with gradient accumulation
                if (step + 1) % self.grad_accumulation_steps == 0:
                    self._optimizer_step(scaler)
                    # update learning rate (warmup scheduler)
                    self.warmup_lr_scheduler.step()

                # record loss (unscaled)
                train_losses_epoch.append(loss.item() * self.grad_accumulation_steps)

            # compute and sync training loss
            mean_train_loss = self._compute_mean_loss(train_losses_epoch)
            train_losses.append(mean_train_loss)

            # print training progress (only master process)
            if self.master_process and (epoch + 1) % self.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}", end="")

            # validation and checkpointing
            current_loss = mean_train_loss
            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_loss = self.validate()
                current_loss = val_loss

                if self.master_process:
                    print(f" | Val Loss: {val_loss:.4f}")
            elif self.master_process:
                print()

            # learning rate scheduling
            self.scheduler.step(current_loss)

            # save checkpoint and early stopping
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

        # cleanup
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
            # encode text and image with CLIP
            text_embeddings = self.clip_model(data=texts, data_type="text", normalize=self.normalize_clip_embeddings)
            image_embeddings = self.clip_model(data=images, data_type="img", normalize=self.normalize_clip_embeddings)

        # reduce dimensionality (optional)
        if self.reduce_clip_embedding_dim:
            text_embeddings = self.prior_model.clip_text_projection(text_embeddings)
            image_embeddings = self.prior_model.clip_image_projection(image_embeddings)

        # sample timestep t ~ Uniform(1, T)
        batch_size = image_embeddings.shape[0]
        timesteps = torch.randint(0, self.prior_model.forward_diffusion.variance_scheduler.num_steps, (batch_size,), device=self.device)

        # sample noise ε ~ N(0, I)
        noise = torch.randn_like(image_embeddings)

        # compute noised embedding z_{i,t}
        noisy_image_embeddings = self.prior_model.forward_diffusion(image_embeddings, noise, timesteps)

        # Predict unnoised embedding ẑ_i
        predicted_image_embeddings = self.prior_model(text_embeddings, noisy_image_embeddings, timesteps)

        # transform back to original space if using dimension reduction
        if self.reduce_clip_embedding_dim:
            predicted_image_embeddings = self.prior_model.clip_image_projection.inverse_transform(predicted_image_embeddings)
            target_embeddings = self.prior_model.clip_image_projection.inverse_transform(image_embeddings)
        else:
            target_embeddings = image_embeddings

        # compute loss L = ||ẑ_i - z_i||²
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

        # gradient clipping
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

                # get embeddings
                text_embeddings = self.clip_model(data=texts, data_type="text", normalize=self.normalize_clip_embeddings)
                image_embeddings = self.clip_model(data=images, data_type="img", normalize=self.normalize_clip_embeddings)
                original_image_embeddings = image_embeddings.clone()

                if self.reduce_clip_embedding_dim:
                    text_embeddings = self.prior_model.clip_text_projection(text_embeddings)
                    image_embeddings = self.prior_model.clip_image_projection(image_embeddings)

                # forward diffusion
                batch_size = image_embeddings.shape[0]
                timesteps = torch.randint(0, self.prior_model.forward_diffusion.variance_scheduler.num_steps, (batch_size,), device=self.device)
                noise = torch.randn_like(image_embeddings)
                noisy_image_embeddings = self.prior_model.forward_diffusion(image_embeddings, noise, timesteps)

                # predict
                predicted_embeddings = self.prior_model(text_embeddings, noisy_image_embeddings, timesteps)

                if self.reduce_clip_embedding_dim:
                    predicted_embeddings = self.prior_model.clip_image_projection.inverse_transform(predicted_embeddings)

                # compute loss
                loss = self.objective(predicted_embeddings, original_image_embeddings)
                val_losses.append(loss.item())


        # compute averages
        val_loss = self._compute_mean_loss(val_losses)

        # return to training mode
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

            # create the directory if it doesn't exist
            os.makedirs(self.store_path, exist_ok=True)

            # define the checkpoint filename
            if is_best:
                filename = "best_model.pth"
            else:
                filename = f"checkpoint_epoch_{epoch}{suffix}.pth"

            # construct the full save path
            save_path = os.path.join(self.store_path, filename)

            # save checkpoint
            torch.save(checkpoint, save_path)
            if self.master_process:  # only print from the master process in DDP
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

        # load prior model
        if 'prior_model_state_dict' in checkpoint:
            state_dict = checkpoint['prior_model_state_dict']

            # handle DDP state dict compatibility
            if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.prior_model.load_state_dict(state_dict)

        # load optimizer
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

###==================================================================================================================###

class SampleUnCLIP(nn.Module):
    """Generates images using the UnCLIP model pipeline.

    Combines a prior model, decoder model, CLIP model, and upsampler models to generate
    images from text prompts or noise. Performs diffusion-based sampling with classifier-free
    guidance in both prior and decoder stages, followed by upsampling to higher resolutions.

    Parameters
    ----------
    `prior_model` : nn.Module
        The UnCLIP prior model for generating image embeddings from text.
    `decoder_model` : nn.Module
        The UnCLIP decoder model for generating low-resolution images from embeddings.
    `clip_model` : nn.Module
        CLIP model for encoding text prompts into embeddings.
    `low_res_upsampler` : nn.Module
        First upsampler model for scaling images from 64x64 to 256x256.
    `high_res_upsampler` : nn.Module, optional
        Second upsampler model for scaling images from 256x256 to 1024x1024, default None.
    `device` : Union[torch.device, str], optional
        Device for computation (default: CUDA if available, else CPU).
    `clip_embedding_dim` : int, optional
        Dimensionality of CLIP embeddings (default: 512).
    `prior_guidance_scale` : float, optional
        Classifier-free guidance scale for the prior model (default: 4.0).
    `decoder_guidance_scale` : float, optional
        Classifier-free guidance scale for the decoder model (default: 8.0).
    `batch_size` : int, optional
        Number of images to generate per batch (default: 1).
    `normalize` : bool, optional
        Whether to normalize CLIP embeddings (default: True).
    `prior_dim_reduction` : bool, optional
        Whether to apply dimensionality reduction in the prior model (default: True).
    `image_size` : Tuple[int, int, int], optional
        Size of the initial generated images (default: (3, 64, 64) for RGB 64x64).
    `use_high_res_upsampler` : bool, optional
        Whether to use the second upsampler for 1024x1024 output (default: True).
    `image_output_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    """
    def __init__(
            self,
            prior_model: nn.Module,
            decoder_model: nn.Module,
            clip_model: nn.Module,
            low_res_upsampler: nn.Module,
            high_res_upsampler: Optional[nn.Module] = None,
            device: Optional[Union[torch.device, str]] = None,
            clip_embedding_dim: int = 512,  # CLIP embedding dimension
            prior_guidance_scale: float = 4.0,
            decoder_guidance_scale: float = 8.0,
            batch_size: int = 1,
            normalize_clip_embeddings: bool = True,
            prior_dim_reduction: bool = True,
            initial_image_size: Tuple[int, int, int] = (3, 64, 64),
            use_high_res_upsampler: bool = True,
            image_output_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> None:
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.prior_model = prior_model.to(self.device).eval()
        self.decoder_model = decoder_model.to(self.device).eval()
        self.clip_model = clip_model.to(self.device).eval()
        self.low_res_upsampler = low_res_upsampler.to(self.device).eval()
        self.high_res_upsampler = high_res_upsampler.to(self.device).eval() if high_res_upsampler else None

        self.prior_guidance_scale = prior_guidance_scale
        self.decoder_guidance_scale = decoder_guidance_scale
        self.batch_size = batch_size
        self.normalize_clip_embeddings = normalize_clip_embeddings
        self.prior_dim_reduction = prior_dim_reduction
        self.clip_embedding_dim = clip_embedding_dim
        self.initial_image_size = initial_image_size
        self.use_high_res_upsampler = use_high_res_upsampler
        self.image_output_range = image_output_range
        self.images_256 = None
        self.images_1024 = None

    def forward(
            self,
            prompts: Optional[Union[str, List]] = None,
            normalize_output: bool = True,
                save_images: bool = True,
            save_path: str = "unclip_generated"
    ):
        """Generates images from text prompts or noise using the UnCLIP pipeline.

        Executes the full UnCLIP generation process: prior model generates image embeddings,
        decoder model generates 64x64 images, first upsampler scales to 256x256, and optional
        second upsampler scales to 1024x1024. Supports classifier-free guidance and saves
        generated images if requested.

        Parameters
        ----------
        `prompts` : Union[str, List], optional
            Text prompt(s) for conditional generation, default None (unconditional).
        `normalize_output` : bool, optional
            Whether to normalize output images to [0, 1] range (default: True).
        `save_images` : bool, optional
            Whether to save generated images to disk (default: True).
        `save_path` : str, optional
            Directory to save generated images (default: "unclip_generated").

        Returns
        -------
        final_images : torch.Tensor
            Generated images, shape (batch_size, channels, height, width), either 256x256
            or 1024x1024 depending on use_second_upsampler.
        """
        # initialize noise for prior sampling (image embedding space)
        embedding_noise = torch.randn((self.batch_size, self.clip_embedding_dim), device=self.device)

        with torch.no_grad():

            # ====== PRIOR STAGE: generate image embeddings from text ======
            # encode text prompt using CLIP
            text_embeddings = self.clip_model(data=prompts, data_type="text", normalize=self.normalize_clip_embeddings)
            current_embeddings = embedding_noise.clone()

            # optionally reduce dimensionality for prior model
            if self.prior_dim_reduction:
                text_embeddings_reduced = self.prior_model.clip_text_projection(text_embeddings)
                current_embeddings_reduced = self.prior_model.clip_image_projection(current_embeddings)
            else:
                text_embeddings_reduced = text_embeddings
                current_embeddings_reduced = current_embeddings

            # prior diffusion sampling loop
            for t in reversed(range(self.prior_model.forward_diffusion.variance_scheduler.tau_num_steps)):
                timesteps = torch.full((self.batch_size,), t, device=self.device)
                prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                # predict embeddings
                predicted_embeddings = self.prior_model(text_embeddings_reduced, current_embeddings_reduced, timesteps)

                # apply guidance
                guided_embeddings = self.compute_prior_guided_prediction(
                    predicted_embeddings, text_embeddings_reduced, current_embeddings_reduced, timesteps
                )

                # update embeddings using reverse diffusion
                current_embeddings_reduced, _ = self.prior_model.reverse_diffusion(
                    current_embeddings_reduced, guided_embeddings, timesteps, prev_timesteps
                )

            # convert back to full embedding dimension if needed
            if self.prior_dim_reduction:
                final_image_embeddings = self.prior_model.clip_image_projection.inverse_transform(current_embeddings_reduced)
            else:
                final_image_embeddings = current_embeddings_reduced

            # ====== DECODER STAGE: generate 64x64 images from embeddings ======
            # initialize noise for decoder sampling
            decoder_noise = torch.randn((self.batch_size, self.initial_image_size[0], self.initial_image_size[1], self.initial_image_size[2]), device=self.device)

            # project image embeddings to 4 tokens
            projected_embeddings = self.decoder_model.clip_decoder_projection(final_image_embeddings)

            # encode text with GLIDE/decoder's text encoder
            glide_text_embeddings = self.decoder_model._encode_text_with_glide(prompts)

            # concatenate embeddings for context
            context = self.decoder_model._concatenate_embeddings(glide_text_embeddings, projected_embeddings)

            current_images = decoder_noise

            for t in reversed(range(self.decoder_model.forward_diffusion.variance_scheduler.tau_num_steps)):

                timesteps = torch.full((self.batch_size,), t, device=self.device)
                prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                # predict noise
                predicted_noise = self.decoder_model.noise_predictor(current_images, timesteps, context, None)

                # apply guidance
                guided_noise = self.compute_decoder_guided_prediction(
                    predicted_noise, current_images, timesteps, context
                )

                # update images using reverse diffusion
                current_images, _ = self.decoder_model.reverse_diffusion(
                    current_images, guided_noise, timesteps, prev_timesteps
                )

            generated_64x64 = current_images

            # ====== FIRST UPSAMPLER: 64x64 -> 256x256 ======
            upsampled_256_noise = torch.randn((self.batch_size, self.initial_image_size[0], 256, 256), device=self.device)
            current_256_images = upsampled_256_noise

            for t in reversed(range(self.low_res_upsampler.forward_diffusion.variance_scheduler.tau_num_steps)):
                timesteps = torch.full((self.batch_size,), t, device=self.device)
                prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                # predict noise for upsampling (conditioned on low-res image)
                predicted_noise = self.low_res_upsampler(current_256_images, timesteps, generated_64x64)

                # update using reverse diffusion
                current_256_images, _ = self.low_res_upsampler.reverse_diffusion(
                    current_256_images, predicted_noise, timesteps, prev_timesteps
                )

            self.images_256 = current_256_images

            # ====== SECOND UPSAMPLER: 256x256 -> 1024x1024 (if enabled) ======
            if self.use_high_res_upsampler and self.high_res_upsampler:
                upsampled_1024_noise = torch.randn((self.batch_size, self.initial_image_size[0], 1024, 1024), device=self.device)
                current_1024_images = upsampled_1024_noise

                for t in reversed(range(self.high_res_upsampler.forward_diffusion.variance_scheduler.tau_num_steps)):
                    timesteps = torch.full((self.batch_size,), t, device=self.device)
                    prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                    # predict noise for upsampling (conditioned on 256x256 image)
                    predicted_noise = self.high_res_upsampler(current_1024_images, timesteps, self.images_256)

                    # update using reverse diffusion
                    current_1024_images, _ = self.high_res_upsampler.reverse_diffusion(
                        current_1024_images, predicted_noise, timesteps, prev_timesteps
                    )

                self.images_1024 = current_1024_images

            # ====== POST-PROCESSING ======
            # normalize output to [0, 1] range if requested
            if normalize_output:
                final_256 = (self.images_256 - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])
                final_1024 = None
                if self.images_1024 is not None:
                    final_1024 = (self.images_1024 - self.image_output_range[0]) / (
                            self.image_output_range[1] - self.image_output_range[0])
            else:
                final_256 = self.images_256
                final_1024 = self.images_1024

            # save images if requested
            if save_images:
                os.makedirs(save_path, exist_ok=True)
                os.makedirs(os.path.join(save_path, "images_256"), exist_ok=True)
                if final_1024 is not None:
                    os.makedirs(os.path.join(save_path, "images_1024"), exist_ok=True)

                for i in range(self.batch_size):
                    img_path_256 = os.path.join(save_path, "images_256", f"image_{i+1}.png")
                    torchvision.utils.save_image(final_256[i], img_path_256)

                    if final_1024 is not None:
                        img_path_1024 = os.path.join(save_path, "images_1024", f"image_{i+1}.png")
                        torchvision.utils.save_image(final_1024[i], img_path_1024)

        # return final images
        if final_1024 is not None:
            return final_1024
        else:
            return final_256

    def compute_prior_guided_prediction(
            self,
            predicted_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor,
            current_embeddings: torch.Tensor,
            timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Computes classifier-free guidance for the prior model.

        Combines conditioned and unconditioned predictions using the classifier-free guidance
        formula to enhance the quality of generated image embeddings.

        Parameters
        ----------
        `predicted_embeddings` : torch.Tensor
            Conditioned predicted embeddings, shape (batch_size, embedding_dim).
        `text_embeddings` : torch.Tensor
            Text embeddings from CLIP, shape (batch_size, embedding_dim).
        `current_embeddings` : torch.Tensor
            Current noisy embeddings, shape (batch_size, embedding_dim).
        `timesteps` : torch.Tensor
            Timestep indices, shape (batch_size,).

        Returns
        -------
        guided_embeddings : torch.Tensor
            Guided embeddings, shape (batch_size, embedding_dim).
        """
        # use zero embeddings for unconditional generation
        zero_text_embeddings = torch.zeros_like(text_embeddings)
        unconditioned_pred = self.prior_model(zero_text_embeddings, current_embeddings, timesteps)

        # CFG formula: (1 + guidance_scale) * conditioned - guidance_scale * unconditioned
        return (1.0 + self.prior_guidance_scale) * predicted_embeddings - self.prior_guidance_scale * unconditioned_pred

    def compute_decoder_guided_prediction(
            self,
            predicted_noise: torch.Tensor,
            current_images: torch.Tensor,
            timesteps: torch.Tensor,
            context: torch.Tensor
    ) -> torch.Tensor:
        """Computes classifier-free guidance for the decoder model.

        Combines conditioned and unconditioned noise predictions using the classifier-free
        guidance formula to enhance the quality of generated images.

        Parameters
        ----------
        `predicted_noise` : torch.Tensor
            Conditioned predicted noise, shape (batch_size, channels, height, width).
        `current_images` : torch.Tensor
            Current noisy images, shape (batch_size, channels, height, width).
        `timesteps` : torch.Tensor
            Timestep indices, shape (batch_size,).
        `context` : torch.Tensor
            Context embeddings (concatenated GLIDE text and projected image embeddings),
            shape (batch_size, seq_len, embedding_dim).

        Returns
        -------
        guided_noise : torch.Tensor
            Guided noise prediction, shape (batch_size, channels, height, width).
        """
        zero_context = torch.zeros_like(context)
        unconditioned_noise = self.decoder_model.noise_predictor(current_images, timesteps, zero_context, None)

        # CFG formula: (1 + guidance_scale) * conditioned - guidance_scale * unconditioned
        return (1.0 + self.decoder_guidance_scale) * predicted_noise - self.decoder_guidance_scale * unconditioned_noise

    def to(self, device: Union[torch.device, str]) -> Self:
        """Moves the module and all its components to the specified device.

        Updates the device attribute and moves all sub-models (prior, decoder, CLIP,
        and upsamplers) to the specified device.

        Parameters
        ----------
        device : Union[torch.device, str]
            Target device for the module and its components.

        Returns
        -------
        SampleUnCLIP
            The module moved to the specified device.
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device

        # move all sub-models to the specified device
        self.prior_model.to(device)
        self.decoder_model.to(device)
        self.clip_model.to(device)
        self.low_res_upsampler.to(device)

        if self.second_upsampler_model is not None:
            self.second_upsampler_model.to(device)

        return super().to(device)

###==================================================================================================================###

class UpsamplerUnCLIP(nn.Module):
    """Diffusion-based upsampler for UnCLIP models.

    A U-Net-like model that upsamples low-resolution images to high-resolution images,
    conditioned on noisy high-resolution images and timesteps, using residual blocks,
    downsampling, and upsampling layers.

    Parameters
    ----------
    `forward_diffusion` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise during training.
    `in_channels` : int, optional
        Number of input channels (default: 3, for RGB images).
    `out_channels` : int, optional
        Number of output channels (default: 3, for RGB noise prediction).
    `model_channels` : int, optional
        Base number of channels in the model (default: 192).
    `num_res_blocks` : int, optional
        Number of residual blocks per resolution level (default: 2).
    `channel_mult` : Tuple[int, ...], optional
        Channel multiplier for each resolution level (default: (1, 2, 4, 8)).
    `dropout` : float, optional
        Dropout probability for regularization (default: 0.1).
    `time_embed_dim` : int, optional
        Dimensionality of time embeddings (default: 768).
    `low_res_size` : int, optional
        Spatial size of low-resolution input (default: 64).
    `high_res_size` : int, optional
        Spatial size of high-resolution output (default: 256).
    """

    def __init__(
            self,
            forward_diffusion: nn.Module,
            reverse_diffusion: nn.Module,
            in_channels: int = 3,
            out_channels: int = 3,
            model_channels: int = 192,
            num_res_blocks: int = 2,
            channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
            dropout_rate: float = 0.1,
            time_embed_dim: int = 768,
            low_res_size: int = 64,
            high_res_size: int = 256,
    ) -> None:
        super().__init__()

        self.forward_diffusion = forward_diffusion # this will be used on training time inside 'TrainUpsamplerUnCLIP'
        self.reverse_diffusion = reverse_diffusion # this module will be used in inference time
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size

        # time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input projection
        # concatenate noisy high-res and upsampled low-res
        self.input_proj = nn.Conv2d(in_channels * 2, model_channels, 3, padding=1)

        # encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock(ch, model_channels * mult, time_embed_dim, dropout_rate)
                )
                ch = model_channels * mult

            if level != len(channel_mult) - 1:
                self.downsample_blocks.append(DownsampleBlock(ch, ch))

        # middle blocks
        self.middle_blocks = nn.ModuleList([
            ResBlock(ch, ch, time_embed_dim, dropout_rate),
            ResBlock(ch, ch, time_embed_dim, dropout_rate),
        ])

        # decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                # skip connections double the input channels
                in_ch = ch + (model_channels * mult if i == 0 else 0)
                out_ch = model_channels * mult

                self.decoder_blocks.append(
                    ResBlock(in_ch, out_ch, time_embed_dim, dropout_rate)
                )
                ch = out_ch

            if level != 0:
                self.upsample_blocks.append(UpsampleBlock(ch, ch))

        # output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x_high: torch.Tensor, t: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        """Predicts noise for the upsampling process.

        Processes a noisy high-resolution image and a low-resolution conditioning image,
        conditioned on timesteps, to predict the noise component for denoising.

        Parameters
        ----------
        `x_high` : torch.Tensor
            Noisy high-resolution image, shape (batch_size, in_channels, high_res_size, high_res_size).
        `t` : torch.Tensor
            Timestep indices, shape (batch_size,).
        `x_low` : torch.Tensor
            Low-resolution conditioning image, shape (batch_size, in_channels, low_res_size, low_res_size).

        Returns
        -------
        out : torch.Tensor
            Predicted noise, shape (batch_size, out_channels, high_res_size, high_res_size).
        """
        # upsample low-resolution image to match high-resolution
        x_low_upsampled = F.interpolate(
            x_low,
            size=(x_high.shape[-2], x_high.shape[-1]),
            mode='bicubic',
            align_corners=False
        )

        # concatenate noisy high-res and upsampled low-res
        x = torch.cat([x_high, x_low_upsampled], dim=1)

        # time embedding
        time_emb = self.time_embed(t.float())  # Ensure float for embedding

        # input projection
        h = self.input_proj(x)

        # store skip connections
        skip_connections = []

        # encoder
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, time_emb)
            if (i + 1) % self.num_res_blocks == 0:
                skip_connections.append(h)
                downsample_idx = (i + 1) // self.num_res_blocks - 1
                if downsample_idx < len(self.downsample_blocks):
                    h = self.downsample_blocks[downsample_idx](h)

        # middle
        for i, block in enumerate(self.middle_blocks):
            h = block(h, time_emb)

        # decoder
        upsample_idx = 0
        for i, block in enumerate(self.decoder_blocks):
            # add skip connection
            if i % (self.num_res_blocks + 1) == 0 and skip_connections:
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)

            h = block(h, time_emb)

            # upsample at the end of each resolution level
            if ((i + 1) % (self.num_res_blocks + 1) == 0 and
                    upsample_idx < len(self.upsample_blocks)):
                h = self.upsample_blocks[upsample_idx](h)
                upsample_idx += 1

        # output projection
        out = self.output_proj(h)

        return out



class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps.

    Generates sinusoidal embeddings for timesteps to condition the upsampler on the
    diffusion process stage.

    Parameters
    ----------
    `dim` : int
        Dimensionality of the embedding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generates sinusoidal embeddings for timesteps.

        Parameters
        ----------
        `timesteps` : torch.Tensor
            Timestep indices, shape (batch_size,).

        Returns
        -------
        embeddings : torch.Tensor
            Sinusoidal embeddings, shape (batch_size, dim).
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time embedding and conditioning.

    A convolutional residual block with group normalization, time embedding conditioning,
    and optional scale-shift normalization, used in the UnCLIP upsampler.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels.
    `out_channels` : int
        Number of output channels.
    `time_embed_dim` : int
        Dimensionality of time embeddings.
    `dropout` : float, optional
        Dropout probability (default: 0.1).
    `use_scale_shift_norm` : bool, optional
        Whether to use scale-shift normalization for time embeddings (default: True).
    """
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int,
                 dropout: float = 0.1, use_scale_shift_norm: bool = True):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels * 2 if use_scale_shift_norm else out_channels)
        )

        self.out_norm = nn.GroupNorm(8, out_channels)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Processes input through the residual block with time conditioning.

        Parameters
        ----------
        `x` : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).
        `time_emb` : torch.Tensor
            Time embeddings, shape (batch_size, time_embed_dim).

        Returns
        -------
        out : torch.Tensor
            Output tensor, shape (batch_size, out_channels, height, width).
        """
        h = self.in_layers(x)

        # apply time embedding
        emb_out = self.time_emb_proj(time_emb)[:, :, None, None]

        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.out_norm(h) * (1 + scale) + shift
            h = self.out_rest(h)
        else:
            h = h + emb_out
            h = self.out_norm(h)
            h = self.out_rest(h)

        return h + self.skip_connection(x)


class UpsampleBlock(nn.Module):
    """Upsampling block using transposed convolution.

    Increases the spatial resolution of the input tensor using a transposed convolution.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels.
    `out_channels` : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsamples the input tensor.

        Parameters
        ----------
        `x` : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        out : torch.Tensor
            Upsampled tensor, shape (batch_size, out_channels, height*2, width*2).
        """
        return self.conv(x)


class DownsampleBlock(nn.Module):
    """Downsampling block using strided convolution.

    Reduces the spatial resolution of the input tensor using a strided convolution.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels.
    `out_channels` : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsamples the input tensor.

        Parameters
        ----------
        `x` : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        out : torch.Tensor
            Downsampled tensor, shape (batch_size, out_channels, height//2, width//2).
        """
        return self.conv(x)

###==================================================================================================================###

class TrainUpsamplerUnCLIP(nn.Module):
    """Trainer for the UnCLIP upsampler model.

    Orchestrates the training of the UnCLIP upsampler model, integrating forward diffusion,
    noise prediction, and low-resolution image conditioning with optional corruption (Gaussian
    blur or BSR degradation). Supports mixed precision, gradient accumulation, DDP, and
    comprehensive training utilities.

    Parameters
    ----------
    `upsampler_model` : nn.Module
        The UnCLIP upsampler model (e.g., UpsamplerUnCLIP) to be trained.
    `train_loader` : torch.utils.data.DataLoader
        DataLoader for training data, providing low- and high-resolution image pairs.
    `optimizer` : torch.optim.Optimizer
        Optimizer for training the upsampler model.
    `objective` : Callable
        Loss function to compute the difference between predicted and target noise.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `max_epochs` : int, optional
        Maximum number of training epochs (default: 1000).
    `device` : Union[str, torch.device], optional
        Device for computation (default: CUDA if available, else CPU).
    `store_path` : str, optional
        Directory to save model checkpoints (default: "unclip_upsampler").
    `patience` : int, optional
        Number of epochs to wait for improvement before early stopping (default: 100).
    `warmup_epochs` : int, optional
        Number of epochs for learning rate warmup (default: 100).
    `val_frequency` : int, optional
        Frequency (in epochs) for validation (default: 10).
    `use_ddp` : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    `grad_accumulation_steps` : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    `log_frequency` : int, optional
        Frequency (in epochs) for printing progress (default: 1).
    `use_compilation` : bool, optional
        Whether to compile the model using torch.compile (default: False).
    `image_output_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `normalize_image_outputs` : bool, optional
        Whether to normalize inputs/outputs (default: True).
    `use_autocast` : bool, optional
        Whether to use automatic mixed precision training (default: True).
    """

    def __init__(
            self,
            upsampler_model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            store_path: str = "unclip_upsampler",
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            use_ddp: bool = False,
            grad_accumulation_steps: int = 1,
            log_frequency: int = 1,
            use_compilation: bool = False,
            image_output_range: Tuple[float, float] = (-1.0, 1.0),
            normalize_image_outputs: bool = True,
            use_autocast: bool = True
    ) -> None:
        super().__init__()

        # training configuration
        self.use_ddp = use_ddp
        self.grad_accumulation_steps = grad_accumulation_steps
        self.use_compilation = use_compilation
        self.use_autocast = use_autocast  # Store autocast flag

        # device initialization
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # setup distributed training
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # compile and wrap models
        self._compile_models()
        self._wrap_models_for_ddp()

        # core model
        self.upsampler_model = upsampler_model.to(self.device)
        self.num_timesteps = self.upsampler_model.forward_diffusion.variance_scheduler.num_steps

        # training components
        self.optimizer = optimizer
        self.objective = objective
        self.train_loader = train_loader
        self.val_loader = val_loader

        # training parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frequency = val_frequency
        self.log_frequency = log_frequency
        self.image_output_range = image_output_range
        self.normalize_image_outputs = normalize_image_outputs

        # checkpoint management
        self.store_path = store_path

        # learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)

    def forward(self) -> Tuple[List[float], float]:
        """Trains the UnCLIP upsampler model to predict noise for denoising.

        Executes the training loop, optimizing the upsampler model using low- and high-resolution
        image pairs, mixed precision, gradient clipping, and learning rate scheduling. Supports
        validation, early stopping, and checkpointing.

        Returns
        -------
        train_losses : List[float]
            List of mean training losses per epoch.
        best_val_loss : float
            Best validation or training loss achieved.
        """
        # set models to training mode
        self.upsampler_model.train()
        if self.upsampler_model.forward_diffusion.variance_scheduler.trainable_beta:
            self.upsampler_model.forward_diffusion.variance_scheduler.train()
        else:
            self.upsampler_model.forward_diffusion.variance_scheduler.eval()

        # initialize training components
        scaler = torch.GradScaler() if self.use_autocast else None
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        # main training loop
        for epoch in range(self.max_epochs):
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []

            # training step loop with gradient accumulation
            for step, (low_res_images, high_res_images) in enumerate(tqdm(self.train_loader, disable=not self.master_process)):
                low_res_images = low_res_images.to(self.device, non_blocking=True)
                high_res_images = high_res_images.to(self.device, non_blocking=True)

                # forward pass with optional autocast
                if self.use_autocast:
                    with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                        batch_size = high_res_images.shape[0]
                        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
                        noise = torch.randn_like(high_res_images)
                        # force FP32 for forward_diffusion to avoid NaN in variance scheduling
                        with torch.autocast(device_type='cuda', enabled=False):
                            high_res_images_noisy = self.upsampler_model.forward_diffusion(high_res_images, noise, timesteps)
                        corruption_type = "gaussian_blur" if self.upsampler_model.low_res_size == 64 else "bsr_degradation"
                        low_res_images_corrupted = self.corrupt_conditioning_image(low_res_images, corruption_type)
                        predicted_noise = self.upsampler_model(high_res_images_noisy, timesteps, low_res_images_corrupted)
                        loss = self.objective(predicted_noise, noise) / self.grad_accumulation_steps
                else:
                    batch_size = high_res_images.shape[0]
                    timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
                    noise = torch.randn_like(high_res_images)
                    high_res_images_noisy = self.upsampler_model.forward_diffusion(high_res_images, noise, timesteps)
                    corruption_type = "gaussian_blur" if self.upsampler_model.low_res_size == 64 else "bsr_degradation"
                    low_res_images_corrupted = self.corrupt_conditioning_image(low_res_images, corruption_type)
                    predicted_noise = self.upsampler_model(high_res_images_noisy, timesteps, low_res_images_corrupted)
                    loss = self.objective(predicted_noise, noise) / self.grad_accumulation_steps

                # backward pass
                if self.use_autocast:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.grad_accumulation_steps == 0:
                    # clip gradients
                    if self.use_autocast:
                        scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.upsampler_model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.upsampler_model.forward_diffusion.parameters(), max_norm=1.0)

                    # optimizer step
                    if self.use_autocast:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()  # clear memory after optimizer step

                train_losses_epoch.append(loss.item() * self.grad_accumulation_steps)

            self.warmup_lr_scheduler.step()

            mean_train_loss = self._compute_mean_loss(train_losses_epoch)
            train_losses.append(mean_train_loss)

            if self.master_process and (epoch + 1) % self.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}")

            current_loss = mean_train_loss

            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_loss = self.validate()
                if self.master_process:
                    print(f" | Val Loss: {val_loss:.4f}")
                    print()
                current_loss = val_loss

            self.scheduler.step(current_loss)

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

        if self.use_ddp:
            destroy_process_group()

        return train_losses, best_val_loss

    def _compute_mean_loss(self, losses: List[float]) -> float:
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
        """Wraps models with DistributedDataParallel for multi-GPU training.

        Configures the upsampler model for DDP training by wrapping it with DistributedDataParallel.
        """
        if self.use_ddp:
            self.upsampler_model = self.upsampler_model.to(self.ddp_local_rank)
            self.upsampler_model = DDP(
                self.upsampler_model,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

    def _compile_models(self) -> None:
        """Compiles models for optimization if supported.

        Attempts to compile the upsampler model using torch.compile for optimization,
        falling back to uncompiled execution if compilation fails.
        """
        if self.use_compilation:
            try:
                self.upsampler_model = self.upsampler_model.to(self.device)
                self.upsampler_model = torch.compile(self.upsampler_model, mode="reduce-overhead")

                if self.master_process:
                    print("Models compiled successfully")
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

    def corrupt_conditioning_image(self, x_low: torch.Tensor, corruption_type: str = "gaussian_blur") -> torch.Tensor:
        """Corrupts the low-resolution conditioning image for robustness.

        Applies Gaussian blur or BSR degradation to the low-resolution image to simulate
        real-world degradation, as specified in the UnCLIP paper.

        Parameters
        ----------
        `x_low` : torch.Tensor
            Low-resolution input image, shape (batch_size, channels, low_res_size, low_res_size).
        `corruption_type` : str, optional
            Type of corruption to apply: "gaussian_blur" or "bsr_degradation" (default: "gaussian_blur").

        Returns
        -------
        x_degraded : torch.Tensor
            Corrupted low-resolution image, same shape as input.
        """
        if corruption_type == "gaussian_blur":
            # apply Gaussian blur
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            return self._gaussian_blur(x_low, kernel_size, sigma)
        elif corruption_type == "bsr_degradation":
            # more diverse BSR degradation for second upsampler
            return self._bsr_degradation(x_low)
        else:
            return x_low

    def _gaussian_blur(self, x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """Applies Gaussian blur to the input image.

        Parameters
        ----------
        `x` : torch.Tensor
            Input image tensor, shape (batch_size, channels, height, width).
        `kernel_size` : int
            Size of the Gaussian kernel.
        `sigma` : float
            Standard deviation of the Gaussian distribution.

        Returns
        -------
        x_blurred : torch.Tensor
            Blurred image tensor, same shape as input.
        """
        # create Gaussian kernel
        kernel = self._get_gaussian_kernel(kernel_size, sigma).to(x.device)
        kernel = kernel.expand(x.shape[1], 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

    def _get_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Generates a 2D Gaussian kernel.

        Parameters
        ----------
        `kernel_size` : int
            Size of the Gaussian kernel.
        `sigma` : float
            Standard deviation of the Gaussian distribution.

        Returns
        -------
        kernel : torch.Tensor
            2D Gaussian kernel, shape (kernel_size, kernel_size).
        """
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g[:, None] * g[None, :]

    def _bsr_degradation(self, x: torch.Tensor) -> torch.Tensor:
        """Applies BSR degradation to the input image.

        Simulates degradation with noise and Gaussian blur, as used in the UnCLIP paper
        for the second upsampler.

        Parameters
        ----------
        `x` : torch.Tensor
            Input image tensor, shape (batch_size, channels, height, width).

        Returns
        -------
        x_degraded : torch.Tensor
            Degraded image tensor, same shape as input, clamped to [-1, 1].
        """
        # add noise
        noise_level = random.uniform(0.0, 0.1)
        noise = torch.randn_like(x) * noise_level

        # apply blur
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 3.0)
        x_degraded = self._gaussian_blur(x + noise, kernel_size, sigma)

        return torch.clamp(x_degraded, -1.0, 1.0)

    def validate(self) -> float:
        """Validates the UnCLIP upsampler model.

        Computes the validation loss by applying forward diffusion to high-resolution images,
        predicting noise with the upsampler model conditioned on corrupted low-resolution images,
        and comparing predicted noise to ground truth.

        Returns
        -------
        val_loss : float
            Mean validation loss.
        """
        # set models to eval mode for evaluation
        self.upsampler_model.eval()
        self.upsampler_model.forward_diffusion.eval()

        val_losses = []

        with torch.no_grad():
            for low_res_images, high_res_images in self.val_loader:
                low_res_images = low_res_images.to(self.device, non_blocking=True)
                high_res_images = high_res_images.to(self.device, non_blocking=True)
                batch_size = high_res_images.shape[0]
                timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
                noise = torch.randn_like(high_res_images)
                high_res_images_noisy = self.upsampler_model.forward_diffusion(high_res_images, noise, timesteps)
                corruption_type = "gaussian_blur" if self.upsampler_model.low_res_size == 64 else "bsr_degradation"
                low_res_images_corrupted = self.corrupt_conditioning_image(low_res_images, corruption_type)
                predicted_noise = self.upsampler_model(high_res_images_noisy, timesteps, low_res_images_corrupted)
                # compute loss
                loss = self.objective(predicted_noise, noise)
                val_losses.append(loss.item())

        # compute average loss
        val_loss = torch.tensor(val_losses).mean().item()

        if self.use_ddp:
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()

        # return to training mode
        self.upsampler_model.train()
        if not self.upsampler_model.forward_diffusion.variance_scheduler.trainable_beta:
            self.upsampler_model.forward_diffusion.variance_scheduler.eval()

        return val_loss

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False, suffix: str = ""):
        """Saves model checkpoint.

        Saves the state of the upsampler model, its variance scheduler, optimizer, and
        schedulers, with options for best model and epoch-specific checkpoints.

        Parameters
        ----------
        `epoch` : int
            Current epoch number.
        `loss` : float
            Current loss value.
        `is_best` : bool, optional
            Whether to save as the best model checkpoint (default: False).
        `suffix` : str, optional
            Suffix to add to checkpoint filename, default "".
        """
        if not self.master_process:
            return
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            # core model
            'upsampler_model_state_dict': self.upsampler_model.module.state_dict() if self.use_ddp else self.upsampler_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # training configuration
            'model_channels': self.upsampler_model.model_channels,
            'num_res_blocks': self.upsampler_model.num_res_blocks,
            'normalize': self.normalize_image_outputs,
            'output_range': self.image_output_range
        }

        # save variance scheduler (submodule of forward_diffusion)
        checkpoint['variance_scheduler_state_dict'] = (
            self.upsampler_model.module.forward_diffusion.variance_scheduler.state_dict() if self.use_ddp
            else self.upsampler_model.forward_diffusion.variance_scheduler.state_dict()
        )

        # save schedulers state
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        checkpoint['warmup_scheduler_state_dict'] = self.warmup_lr_scheduler.state_dict()

        filename = f"unclip_upsampler_epoch_{epoch}{suffix}.pth"
        if is_best:
            filename = f"unclip_upsampler_best{suffix}.pth"

        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)

        if is_best:
            print(f"Best model saved: {filepath}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """Loads model checkpoint.

        Restores the state of the upsampler model, its variance scheduler, optimizer, and
        schedulers from a saved checkpoint, handling DDP compatibility.

        Parameters
        ----------
        `checkpoint_path` : str
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
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        def _load_model_state_dict(model: nn.Module, state_dict: dict, model_name: str) -> None:
            """Helper function to load state dict with DDP compatibility."""
            try:
                # handle DDP state dict compatibility
                if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                model.load_state_dict(state_dict)
                if self.master_process:
                    print(f"✓ Loaded {model_name}")
            except Exception as e:
                warnings.warn(f"Failed to load {model_name}: {e}")

        # load core upsampler model
        if 'upsampler_model_state_dict' in checkpoint:
            _load_model_state_dict(self.upsampler_model, checkpoint['upsampler_model_state_dict'],
                                   'upsampler_model')

        # load variance scheduler (submodule of forward_diffusion)
        if 'variance_scheduler_state_dict' in checkpoint or 'hyper_params_state_dict' in checkpoint:
            state_dict = checkpoint.get('variance_scheduler_state_dict', checkpoint.get('hyper_params_state_dict'))
            try:
                _load_model_state_dict(self.upsampler_model.forward_diffusion.variance_scheduler, state_dict, 'variance_scheduler')
            except Exception as e:
                warnings.warn(f"Failed to load variance scheduler: {e}")

        # load optimizer
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
        if 'model_channels' in checkpoint:
            if checkpoint['model_channels'] != self.upsampler_model.model_channels:
                warnings.warn(
                    f"Model channels mismatch: checkpoint={checkpoint['model_channels']}, current={self.upsampler_model.model_channels}")

        if 'num_res_blocks' in checkpoint:
            if checkpoint['num_res_blocks'] != self.upsampler_model.num_res_blocks:
                warnings.warn(
                    f"Num res blocks mismatch: checkpoint={checkpoint['num_res_blocks']}, current={self.upsampler_model.num_res_blocks}")

        if 'normalize' in checkpoint:
            if checkpoint['normalize'] != self.normalize_image_outputs:
                warnings.warn(
                    f"Normalize setting mismatch: checkpoint={checkpoint['normalize']}, current={self.normalize_image_outputs}")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        return epoch, loss