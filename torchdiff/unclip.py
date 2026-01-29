"""
**UnCLIP Diffusion Model**

This module provides a comprehensive implementation of the UnCLIP diffusion model,
as described in Ramesh et al. (2022, "Hierarchical Text-Conditional Image Generation with CLIP Latents").
It integrates CLIP embeddings with diffusion processes for high-quality image generation conditioned on text prompts or image embeddings.
The module supports training, sampling, and upsampling processes, leveraging components from CLIP, GLIDE, and DDIM,
with classifier-free guidance and text dropout for robust generation.

**Components**

- **SchedulerUnCLIP**: Manages noise schedules with support for linear, sigmoid, quadratic, constant, inverse_time,
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
from torch.utils.checkpoint import checkpoint
import torchvision
from PIL import Image
from transformers import BertTokenizer, CLIPProcessor, CLIPModel
from typing import Optional, List, Tuple, Union, Callable, Any, Dict
from tqdm.auto import tqdm
import os
import warnings
import random
import math


###==================================================================================================================###


class SchedulerUnCLIP(nn.Module):
    """Variance scheduler for UnCLIP supporting multiple schedule types

    Manages noise schedule parameters with support for both full training schedule
    and subsampled inference schedule  for faster sampling.
    """
    def __init__(
            self,
            schedule_type: str = "linear",
            train_steps: int = 1000,
            sample_steps: Optional[int] = None,
            beta_min: float = 1e-4,
            beta_max: float = 0.02,
            cosine_s: float = 0.008,
            clip_min: float = 1e-4,
            clip_max: float = 0.9999,
            learn_var: bool = False
    ):
        super().__init__()
        valid_schedules = ["linear", "cosine", "quadratic", "sigmoid", "constant", "inverse_time"]
        if schedule_type not in valid_schedules:
            raise ValueError(f"schedule_type must be one of {valid_schedules}, got {schedule_type}")
        if not (0 < beta_min < beta_max < 1):
            raise ValueError(f"beta_start and beta_end must satisfy 0 < beta_start < beta_end < 1")

        self.schedule_type = schedule_type
        self.train_steps = train_steps
        self.sample_steps = sample_steps or train_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cosine_s = cosine_s
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.learn_var = learn_var
        self._setup_schedule()
        self._setup_inf_timesteps()

    def _setup_schedule(self):
        """Setup the noise schedule and precompute all coefficients"""
        if self.schedule_type == "linear":
            betas = torch.linspace(self.beta_min, self.beta_max, self.train_steps)
        elif self.schedule_type == "cosine":
            steps = self.train_steps + 1
            t = torch.linspace(0, self.train_steps, steps)
            alphas_cumprod = torch.cos(((t / self.train_steps) + self.cosine_s) / (1 + self.cosine_s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, self.clip_min, self.clip_max)

        elif self.schedule_type == "quadratic":
            betas = torch.linspace(
                self.beta_min ** 0.5,
                self.beta_max ** 0.5,
                self.train_steps
            ) ** 2

        elif self.schedule_type == "sigmoid":
            x = torch.linspace(-6, 6, self.train_steps)
            betas = torch.sigmoid(x) * (self.beta_max - self.beta_min) + self.beta_min

        elif self.schedule_type == "constant":
            betas = torch.full((self.train_steps,), self.beta_max)

        elif self.schedule_type == "inverse_time":
            beta = 1.0 / torch.linspace(self.train_steps, 1, self.train_steps)
            betas = self.beta_min + (self.beta_max - self.beta_min) * (
                    (beta - beta.min()) / (beta.max() - beta.min())
            )
        betas = torch.clamp(betas, min=self.clip_min, max=self.clip_max)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))

        if self.learn_var:
            self.register_parameter('log_variance', nn.Parameter(posterior_log_variance.clone()))
        else:
            self.register_buffer('log_variance', posterior_log_variance)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', posterior_log_variance)

    def _setup_inf_timesteps(self):
        """Setup inference timesteps (tau schedule for UnCLIP)

        Creates a uniform subset of timesteps for faster sampling.
        Similar to DDIM but called 'tau schedule' in UnCLIP literature.
        """
        step_ratio = self.train_steps // self.sample_steps
        # Create uniform spacing: [0, step_ratio, 2*step_ratio, ...]
        inf_timesteps = torch.arange(0, self.train_steps, step_ratio, dtype=torch.long)
        self.register_buffer('inference_timesteps', inf_timesteps)

    def set_inf_timesteps(self, num_inf_timesteps: int):
        """Dynamically change the number of inference steps

        Allows using different numbers of steps at inference time.
        """
        self.sample_steps = num_inf_timesteps
        self._setup_inf_timesteps()

    def get_index(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract coefficients at timestep t and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = t.to(t.device)
        if len(x_shape) == 2:
            return out.reshape(batch_size, 1)
        else:
            return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

###==================================================================================================================###

class ForwardUnCLIP(nn.Module):
    """Forward diffusion process for UnCLIP

    Applies Gaussian noise to input data according to the forward diffusion process.
    Supports both 2D (latent embeddings) and 4D (images) inputs.

    q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1 - ᾱ_t)I)
    """
    def __init__(self, scheduler: nn.Module, pred_type: str = "noise"):
        super().__init__()
        valid_types = ["noise", "x0"]
        if pred_type not in valid_types:
            raise ValueError(f"pred_type must be one of {valid_types}, got {pred_type}")
        self.vs = scheduler
        self.pred_type = pred_type

    def forward(
            self,
            x0: torch.Tensor,
            noise: torch.Tensor,
            t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from q(x_t | x_0) and compute prediction target

        Args:
            x0: (batch, ...) clean data (2D or 4D)
            t: (batch,) discrete timesteps in [0, train_steps-1]
            noise: (batch, ...) gaussian noise

        Returns:
            xt: (batch, ...) noised data
            target: (batch, ...) prediction target (noise or x0)
        """
        if not torch.all((t >= 0) & (t < self.vs.train_steps)):
            raise ValueError(f"t must be in [0, {self.vs.train_steps - 1}]")
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, x0.shape)
        # x_t ~ q(x_t | x_0)
        # x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        if self.pred_type == "noise":
            # predict noise ε (ddim-style)
            target = noise
        elif self.pred_type == "x0":
            # predict original data x_0 (unclip prior style)
            target = x0
        return xt, target

###==================================================================================================================###

class ReverseUnCLIP(nn.Module):
    """Reverse diffusion process for UnCLIP

    Denoises input using DDIM-style sampling with the tau (subsampled) schedule.
    Supports both noise prediction and x0 prediction modes.
    Works with both 2D (latent embeddings) and 4D (images) inputs.
    """
    def __init__(self, scheduler: nn.Module, pred_type: str = "noise", eta: float = 0.0, clip_: bool = True):
        super().__init__()
        valid_pred_types = ["noise", "x0"]
        if pred_type not in valid_pred_types:
            raise ValueError(f"pred_type must be one of {valid_pred_types}")

        self.vs = scheduler
        self.pred_type = pred_type
        self.eta = eta  # noise scaling factor (0 = deterministic)
        self.clip_ = clip_

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Convert model output to x0 prediction based on prediction type"""
        actual_t = self.vs.inference_timesteps[t]
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[actual_t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[actual_t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, xt.shape)
        if self.pred_type == "noise":
            # x_0 = (x_t - √(1 - ᾱ_t) * ε_θ) / √ᾱ_t
            x0_pred = (xt - sqrt_one_minus_alpha_cumprod_t * pred) / sqrt_alpha_cumprod_t
        elif self.pred_type == "x0":
            # directly predict x_0
            x0_pred = pred
        if self.clip_:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        return x0_pred

    def predict_noise(self, xt: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor) -> torch.Tensor:
        """Predict noise from x0

        ε̂ = (x_t - √ᾱ_t * x̂_0) / √(1 - ᾱ_t)
        """
        actual_t = self.vs.inference_timesteps[t]
        sqrt_alpha_cumprod_t = self.vs.sqrt_alphas_cumprod[actual_t]
        sqrt_one_minus_alpha_cumprod_t = self.vs.sqrt_one_minus_alphas_cumprod[actual_t]
        sqrt_alpha_cumprod_t = self.vs.get_index(sqrt_alpha_cumprod_t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self.vs.get_index(sqrt_one_minus_alpha_cumprod_t, xt.shape)
        pred_noise = (xt - sqrt_alpha_cumprod_t * x0_pred) / sqrt_one_minus_alpha_cumprod_t
        return pred_noise

    def forward(self, xt: torch.Tensor, t: torch.Tensor, t_pre: torch.Tensor, pred: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """UnCLIP reverse step from x_t to x_{t_prev}

        Uses tau schedule (subsampled timesteps) for faster sampling.

        Args:
            xt: (batch, ...) current state (2D or 4D)
            t: (batch,) current tau timestep indices [0, sample_steps-1]
            t_pre: (batch,) previous tau timestep indices
            pred: (batch, ...) model prediction

        Returns:
            x_prev: (batch, ...) previous state x_{t_prev}
            pred_x0: (batch, ...) predicted x0 (if return_pred_x0=True)
        """
        if not torch.all((t >= 0) & (t < self.vs.sample_steps)):
            raise ValueError(f"t must be in [0, {self.vs.sample_steps - 1}]")
        if not torch.all((t_pre >= 0) & (t_pre < self.vs.sample_steps)):
            raise ValueError(f"t_prev must be in [0, {self.vs.sample_steps - 1}]")

        pred_x0 = self.predict_x0(xt, t, pred)
        pred_noise = self.predict_noise(xt, t, pred_x0)
        actual_t = self.vs.inference_timesteps[t]
        actual_t_prev = self.vs.inference_timesteps[t_pre]
        alpha_cumprod_t = self.vs.alphas_cumprod[actual_t]
        alpha_cumprod_t_prev = self.vs.alphas_cumprod[actual_t_prev]
        alpha_cumprod_t = self.vs.get_index(alpha_cumprod_t, xt.shape)
        alpha_cumprod_t_prev = self.vs.get_index(alpha_cumprod_t_prev, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
        sqrt_one_minus_alpha_cumprod_t_prev = torch.sqrt(1.0 - alpha_cumprod_t_prev)

        noise_coeff = self.eta * (
                (sqrt_one_minus_alpha_cumprod_t / sqrt_alpha_cumprod_t_prev) *
                sqrt_one_minus_alpha_cumprod_t_prev /
                torch.clamp(sqrt_one_minus_alpha_cumprod_t, min=1e-8)
        )
        direction_coeff = torch.sqrt(
            torch.clamp(
                sqrt_one_minus_alpha_cumprod_t_prev ** 2 - noise_coeff ** 2,
                min=1e-8
            )
        )
        noise = torch.randn_like(xt)
        mask = (actual_t_prev != 0).float()
        mask = self.vs.get_index(mask, xt.shape)
        # x_{t_prev} = √ᾱ_{t_prev} * x̂_0 + noise_coeff * z + direction_coeff * ε̂
        x_prev = (
                sqrt_alpha_cumprod_t_prev * pred_x0 +
                noise_coeff * mask * noise +
                direction_coeff * pred_noise
        )
        return x_prev, pred_x0

    def set_pred_type(self, pred_type: str):
        """Change the prediction type after initialization"""
        if pred_type not in ["noise", "x0"]:
            raise ValueError(f"pred_type must be 'noise' or 'x0'")
        self.pred_type = pred_type

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
        device: str = 'cuda',
        use_fast: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        try:
            # load clip model and processor
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
    `clip_embed_dim` : int
        Dimensionality of the input embeddings.
    `diff_net` : nn.Module
        Model to predict noise/x0 during the denoising process.
    `fwd_unclip` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise.
    `rwd_unclip` : nn.Module
        Reverse diffusion module (e.g., ReverseUnCLIP) for denoising.
    `glide_text_encoder` : nn.Module, optional
        GLIDE text encoder for processing text prompts, default None.
    `tokenizer` : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    `device` : str, optional
        Device for computation (default: CUDA).
    `norm_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `norm_clip_embed` : bool, optional
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
            clip_embed_dim: int,
            diff_net: nn.Module,
            fwd_unclip: nn.Module,
            rwd_unclip: nn.Module,
            glide_text_encoder: torch.nn.Module = None,  # GLIDE text encoder
            tokenizer: Optional[BertTokenizer] = None,
            device: str = 'cuda',
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            norm_clip_embed: bool = True,
            classifier_free_prop: float = 0.1,  # paper specifies 10%
            drop_caption: float = 0.5,  # paper specifies 50%
            max_token_length: int = 77  # max_token_length for tokenization
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.clip_embed_dim = clip_embed_dim
        # core models
        self.diff_net = diff_net.to(self.device)
        self.fwd_unclip = fwd_unclip.to(self.device)
        self.rwd_unclip = rwd_unclip.to(self.device)
        self.glide_text_encoder = glide_text_encoder.to(self.device) if glide_text_encoder else None

        # projecting CLIP embeddings into four extra tokens of context
        self.clip_decoder_proj = CLIPContextProjection(clip_embed_dim=self.clip_embed_dim, num_tokens=4).to(self.device)
        self.clip_time_proj = nn.Linear(self.clip_embed_dim, self.clip_embed_dim).to(self.device)

        # training parameters
        self.norm_range = norm_range
        self.norm_clip_embed = norm_clip_embed
        self.classifier_free_prop = classifier_free_prop
        self.drop_caption = drop_caption
        self.max_token_length = max_token_length

        # initialize tokenizer
        if tokenizer is None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")

    def forward(
            self,
            img_embed: torch.Tensor,
            text_embed: torch.Tensor,
            imgs: torch.Tensor,
            texts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes embeddings and images to predict noise for training.

        Applies classifier-free guidance and text dropout, projects CLIP image embeddings
        into context tokens, encodes text with GLIDE, and predicts noise for the diffusion process.

        Parameters
        ----------
        `img_embed` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embed_dim).
        `text_embed` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embed_dim).
        `imgs` : torch.Tensor
            Input images, shape (batch_size, channels, height, width).
        `texts` : torch.Tensor
            Text prompts for conditional generation.

        Returns
        -------
        pred : torch.Tensor
            Predicted noise/x0 tensor, shape (batch_size, channels, height, width).
        target : torch.Tensor
            Ground truth noise/x0 tensor, shape (batch_size, channels, height, width).
        """
        if self.norm_clip_embed:
            img_embed = F.normalize(img_embed, dim=-1)
        img_embed = self._classifier_free_guidance(img_embed)
        text_embed = self._text_dropout(text_embed)
        # project z_i to 4 tokens
        c = self.clip_decoder_proj(img_embed)
        # encode text with GLIDE
        y = self._encode_text_with_glide(texts if text_embed is not None else None)
        # concatenate embeddings
        context = self._conc_embed(y, c)
        # sample timestep and noise
        t, noise = self._sample_time_and_noise(imgs.shape[0], imgs.shape)
        # compute noisy image
        noisy_imgs, target = self.fwd_unclip(imgs, noise, t)
        clip_img_embed = self.clip_time_proj(img_embed)
        pred = self.diff_net(noisy_imgs, t, context, clip_img_embed)
        return pred, target

    def inference_forward(self, img_embed, prompt_embed):
        pass

    def _classifier_free_guidance(self, img_embed: torch.Tensor) -> torch.Tensor:
        """Applies classifier-free guidance to image embeddings.

        Sets image embeddings to zero with a specified probability to implement
        classifier-free guidance, as described in the UnCLIP paper.

        Parameters
        ----------
        `img_embed` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embed_dim).

        Returns
        -------
        img_embed : torch.Tensor
            Modified image embeddings, shape (batch_size, embed_dim).
        """
        batch_size = img_embed.shape[0]
        mask = torch.rand(batch_size, 1, device=self.device) < self.classifier_free_prop
        return img_embed * (~mask).float()

    def _text_dropout(self, text_embed: torch.Tensor) -> torch.Tensor:
        """Applies text caption dropout to text embeddings.

        Drops text embeddings with a specified probability to implement text dropout,
        as described in the UnCLIP paper.

        Parameters
        ----------
        `text_embed` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embed_dim).

        Returns
        -------
        text_embed : torch.Tensor or None
            Modified text embeddings or None if dropped, shape (batch_size, embed_dim).
        """
        if text_embed is None:
            return None
        batch_size = text_embed.shape[0]
        mask = torch.rand(batch_size, 1, device=self.device) < self.drop_caption
        return text_embed * (~mask).float()


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
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt"
        ).to(self.device)

        # get embeddings from GLIDE text encoder
        input_ids = tokenized["input_ids"]
        att_mask = tokenized["attention_mask"]
        y_encoded = self.glide_text_encoder(input_ids, att_mask)
        return y_encoded

    def _conc_embed(self, y: Optional[torch.Tensor], c: torch.Tensor) -> torch.Tensor:
        """Concatenates GLIDE text embeddings and context tokens.

        Combines encoded text embeddings (if available) with projected context tokens
        along the sequence dimension, as specified in the UnCLIP paper.

        Parameters
        ----------
        `y` : torch.Tensor or None
            Encoded text embeddings from GLIDE, shape (batch_size, seq_len, embed_dim).
        `c` : torch.Tensor
            Projected context tokens, shape (batch_size, num_tokens, embed_dim).

        Returns
        -------
        s : torch.Tensor
            Concatenated embeddings, shape (batch_size, seq_len + num_tokens, embed_dim).
        """
        if y is not None:
            if len(y.shape) == 2:  # [batch_size, embed_dim]
                y = y.unsqueeze(1)  # [batch_size, 1, embed_dim]
            # concatenate along the sequence dimension
            s = torch.cat([y, c], dim=1)  # [batch_size, seq_len + 4, embed_dim]
        else:
            s = c  # [batch_size, 4, embed_dim]
        return s

    def _sample_time_and_noise(self, batch_size: int, img_shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples timesteps and noise for the diffusion process.

        Generates random timesteps and Gaussian noise for use in the forward diffusion process.

        Parameters
        ----------
        `batch_size` : int
            Number of samples in the batch.
        `img_size` : torch.Size
            Shape of the images, typically (batch_size, channels, height, width).

        Returns
        -------
        t : torch.Tensor
            Sampled timestep indices, shape (batch_size,).
        noise : torch.Tensor
            Sampled Gaussian noise, shape (batch_size, channels, height, width).
        """
        # sample timestep t ~ Uniform(1, T)
        t = torch.randint(0, self.fwd_unclip.vs.train_steps, (batch_size,), device=self.device)
        # sample noise ε ~ N(0, I)
        noise = torch.randn(img_shape, device=self.device)
        return t, noise

###==================================================================================================================###

class UnCLIPTransformerPrior(nn.Module):
    """Transformer-based prior model for UnCLIP diffusion.

    Predicts clean image embeddings from noisy image embeddings and text embeddings using
    a Transformer architecture, incorporating time embeddings and optional projection
    layers for text and image inputs.

    Parameters
    ----------
    `fwd_unclip` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise during training.
    `rwd_unclip` : nn.Module
        Reverse diffusion module (e.g., ReverseUnCLIP) for denoising during training.
    `clip_text_proj` : nn.Module, optional
        Projection module for text embeddings, default None.
    `clip_img_proj` : nn.Module, optional
        Projection module for image embeddings, default None.
    `trans_embed_dim` : int, optional
        Dimensionality of embeddings (default: 320).
    `num_layers` : int, optional
        Number of Transformer layers (default: 12).
    `num_att_heads` : int, optional
        Number of attention heads in each Transformer layer (default: 8).
    `ff_dim` : int, optional
        Dimensionality of the feedforward network in Transformer layers (default: 768).
    `max_sequence_length` : int, optional
        Maximum sequence length for input embeddings (default: 2).
    `dropoute` : float, optional
        Dropout probability for regularization (default: 0.2).
    `use_flash`: bool, optional
        Enable flash attention if available (default: True).
    `grad_check`: bool, optional
        Apply gradinet checkpointing (default: False).
    `check_every_n_layers`: int, optional
        Frequency of applying gradient checkpoint (default: 2 layers)
    """
    def __init__(
            self,
            fwd_unclip: nn.Module, # will be used during training
            rwd_unclip: nn.Module, # will be used during training
            clip_text_proj: Optional[nn.Module] = None,  # used during training instead of PCA in the main paper
            clip_img_proj: Optional[nn.Module] = None,  # used during training instead of PCA in the main paper
            trans_embed_dim: int = 320,
            num_layers: int = 12,
            num_att_heads: int = 8,
            ff_dim: int = 768,
            max_sequence_length: int = 2,
            dropout: float = 0.2,
            use_flash: bool = True,
            grad_check: bool = False,
            check_every_n_layers: int = 2
    ) -> None:
        super().__init__()

        self.fwd_unclip = fwd_unclip
        self.rwd_unclip = rwd_unclip
        self.clip_text_proj = clip_text_proj
        self.clip_img_proj = clip_img_proj
        self.trans_embed_dim = trans_embed_dim
        self.max_sequence_length = max_sequence_length
        self.grad_check = grad_check
        self.check_every_n_layers = check_every_n_layers
        self.use_flash = use_flash and self._check_flash_attention()
        # time embedding network
        self.time_embed_net = nn.Sequential(
            nn.Linear(trans_embed_dim, trans_embed_dim),
            nn.GELU(),
            nn.Linear(trans_embed_dim, trans_embed_dim)
        )
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(max_sequence_length, trans_embed_dim))
        # transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(trans_embed_dim, num_att_heads, ff_dim, dropout, self.use_flash)
            for _ in range(num_layers)
        ])
        # final output projection
        self.out_proj = nn.Linear(trans_embed_dim, trans_embed_dim)
        # cache for sinusoidal embeddings (reuse across batches)
        self._cached_sinusoidal_embeds = {}

    def forward(
            self,
            text_embed: torch.Tensor,
            noisy_img_embed: torch.Tensor,
            timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Predicts clean image embeddings from noisy inputs and text embeddings.

        Processes text and noisy image embeddings through a Transformer architecture,
        conditioned on time embeddings, to predict the clean image embeddings.

        Parameters
        ----------
        `text_embed` : torch.Tensor
            Text embeddings, shape (batch_size, embed_dim).
        `noisy_img_embed` : torch.Tensor
            Noisy image embeddings, shape (batch_size, embed_dim).
        `timesteps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,).

        Returns
        -------
        pred_clean_embed : torch.Tensor
            Predicted clean image embeddings, shape (batch_size, embed_dim).
        """
        device = text_embed.device
        # create sinusoidal time embeddings
        time_embed = self._sinusoidal_embed_cached(timesteps, self.trans_embed_dim, device)
        time_embed = self.time_embed_net(time_embed)
        # add time information to image embeddings
        cond_img_embed = noisy_img_embed + time_embed
        # create sequence: [text_embed, cond_img_embed]
        seq = torch.stack([text_embed, cond_img_embed], dim=1)  # [B, 2, D]
        # add positional embeddings
        seq = seq + self.pos_embed.unsqueeze(0)
        # pass through transformer blocks
        if self.grad_check and self.training:
            seq = self._forward_with_check(seq)
        else:
            for transformer_block in self.transformer_blocks:
                seq = transformer_block(seq)
        # extract predicted clean image embedding (second position in sequence)
        pred_clean_embed = seq[:, 1, :]  # [B, D]
        # apply final projection
        pred_clean_embed = self.out_proj(pred_clean_embed)

        return pred_clean_embed

    def _forward_with_check(self, seq: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing every N layers"""
        for i in range(0, len(self.transformer_blocks), self.check_every_n_layers):
            end_idx = min(i + self.check_every_n_layers, len(self.transformer_blocks))
            layers_to_checkpoint = self.transformer_blocks[i:end_idx]
            def create_forward_func(layers):
                def forward_func(x):
                    for layer in layers:
                        x = layer(x)
                    return x
                return forward_func
            # apply checkpointing
            seq = checkpoint(
                create_forward_func(layers_to_checkpoint),
                seq,
                use_reentrant=False
            )
        return seq

    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            if hasattr(nn.functional, 'scaled_dot_product_attention'):
                return True
        except:
            pass
        return False
    def enable_grad_check(self):
        """Enable gradient checkpointing for memory savings"""
        self.use_grad_check = True

    def disable_grad_check(self):
        """Disable gradient checkpointing"""
        self.use_grad_check = False

    def _sinusoidal_embed_cached(
            self,
            timesteps: torch.Tensor,
            embed_dim: int,
            device: Union[torch.device, str]
    ) -> torch.Tensor:
        """Generates sinusoidal positional embeddings with caching.

        Caches the sinusoidal embedding computation to avoid recomputation
        for the same timesteps across different batches.
        """
        max_timestep = timesteps.max().item()
        cache_key = (embed_dim, device, max_timestep)
        if cache_key not in self._cached_sinusoidal_embeds:
            half_dim = embed_dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            all_timesteps = torch.arange(max_timestep + 1, device=device).float()
            emb = all_timesteps[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            if embed_dim % 2 == 1:
                emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
            self._cached_sinusoidal_embeds[cache_key] = emb
        cached_emb = self._cached_sinusoidal_embeds[cache_key]
        return cached_emb[timesteps]

    def _sinusoidal_embed(
            self,
            timesteps: torch.Tensor,
            embed_dim: int,
            device: Union[torch.device, str]
    ) -> torch.Tensor:
        """Generates sinusoidal positional embeddings for timesteps.

        Creates sinusoidal embeddings for the given timesteps to condition the Transformer
        on the diffusion process time steps.

        Parameters
        ----------
        `timesteps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,).
        `embed_dim` : int
            Dimensionality of the embeddings.
        `device` : Union[torch.device, str]
            Device to place the embeddings on.

        Returns
        -------
        emb : torch.Tensor
            Sinusoidal time embeddings, shape (batch_size, embed_dim).
        """
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # handle odd embedding dimensions
        if embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

        return emb


class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head attention and feedforward layers.

    Implements a Transformer block with multi-head self-attention, layer normalization,
    and a feedforward network with residual connections for processing sequences in
    the UnCLIPTransformerPrior model.

    Parameters
    ----------
    `embed_dim` : int
        Dimensionality of input and output embeddings.
    `num_heads` : int
        Number of attention heads in the multi-head attention layer.
    `ff_dim` : int
        Dimensionality of the feedforward network.
    `dropout` : float
        Dropout probability for regularization.
    `use_falsh`: bool
        Whethere use flash attention (default: True)
    """
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout: float,
            use_flash: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_flash = use_flash

        self.att_norm = nn.LayerNorm(embed_dim)
        self.ff_norm = nn.LayerNorm(embed_dim)

        # multi-head attention
        if use_flash and hasattr(nn.functional, 'scaled_dot_product_attention'):
            # use manual qkv projection for flash attention
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.dropout_p = dropout
        else:
            # fall back to standard MultiheadAttention
            self.self_att = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                batch_first=True
            )
        # feed forward net
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input sequence through the Transformer block.

        Applies multi-head self-attention followed by a feedforward network, with residual
        connections and layer normalization.

        Parameters
        ----------
        `x` : torch.Tensor
            Input sequence tensor, shape (batch_size, sequence_length, embed_dim).

        Returns
        -------
        `x` : torch.Tensor
            Processed sequence tensor, shape (batch_size, sequence_length, embed_dim).
        """
        n_x = self.att_norm(x)
        if self.use_flash and hasattr(nn.functional, 'scaled_dot_product_attention'):
            att_out = self._flash_attention(n_x)
        else:
            att_out, _ = self.self_att(n_x, n_x, n_x)
        x = x + att_out
        n_x = self.ff_norm(x)
        ff_out = self.ff(n_x)
        x = x + ff_out
        return x

    def _flash_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Flash Attention

        Parameters
        ----------
        `x` : torch.Tensor
            Input tensor, shape (batch_size, seq_len, embed_dim).

        Returns
        -------
        att_out : torch.Tensor
            Attention output, shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape
        # project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, S, 3*D]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.embed_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D//H]
        q, k, v = qkv[0], qkv[1], qkv[2]
        att_out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        att_out = att_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        att_out = self.out_proj(att_out)
        return att_out

class FusedGELU(nn.Module):
    """Fused GELU activation for better efficiency on some hardware"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use approximate GELU for speed
        return nn.functional.gelu(x, approximate='tanh')

###==================================================================================================================###

class CLIPContextProjection(nn.Module):
    """Projects CLIP image embeddings into multiple context tokens.

    Transforms a single CLIP image embedding into a specified number of context tokens
    using a linear projection followed by layer normalization.

    Parameters
    ----------
    `clip_embed_dim` : int
        Dimensionality of the input CLIP embedding (e.g., 319 or 512).
    `num_tokens` : int, optional
        Number of context tokens to generate (default: 4).
    """
    def __init__(self, clip_embed_dim, num_tokens=4):
        super().__init__()
        self.clip_embed_dim = clip_embed_dim
        self.num_tokens = num_tokens
        self.clip_proj = nn.Linear(clip_embed_dim, clip_embed_dim * num_tokens)
        self.clip_embed_norm = nn.LayerNorm(clip_embed_dim)

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
        proj = self.clip_proj(z_i)
        c = proj.view(batch_size, self.num_tokens, self.clip_embed_dim)
        c = self.clip_embed_norm(c)
        return c

###==================================================================================================================###

class CLIPEmbeddingProjection(nn.Module):
    """Projection module for dimensionality reduction and reconstruction.

    Implements a neural network with forward and inverse projections to reduce and
    restore input dimensionality, supporting customizable hidden layers, dropout, and
    layer normalization.

    Parameters
    ----------
    `clip_embed_dim` : int, optional
        Input dimensionality (default: 1024).
    `trans_embed_dim` : int, optional
        Output dimensionality for forward projection (default: 320).
    `hidden_dim` : int, optional
        Inner dimension of projection (default: 512).
    `dropout`: float
        Dropout rate (default: 0.2)
    `use_layer_norm`: bool
        If normalize output (default: True)
    """
    def __init__(
        self,
        clip_embed_dim: int = 1024,
        trans_embed_dim: int = 320,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_layer_norm: bool = True
    ) -> None:
        super().__init__()
        self.clip_embed_dim = clip_embed_dim
        self.trans_embed_dim = trans_embed_dim
        # input_dim -> output_dim
        self.fwd_proj = self._build_proj_net(
            clip_embed_dim, trans_embed_dim, hidden_dim, num_layers, dropout, use_layer_norm
        )
        # output_dim -> input_dim
        self.inv_proj = self._build_proj_net(
            trans_embed_dim, clip_embed_dim, hidden_dim, num_layers, dropout, use_layer_norm
        )
    def _build_proj_net(
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
        return self.fwd_proj(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstructs input from lower-dimensional space.

        Applies the inverse projection network to restore the original dimensionality
        of the input tensor.

        Parameters
        ----------
        `x` : torch.Tensor
            Reduced-dimensionality tensor, shape (batch_size, output_dim).

        Returns
        -------
        x : torch.Tensor
            Reconstructed tensor, shape (batch_size, input_dim).
        """
        return self.inv_proj(x)

    def rec_loss(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.forward(x)
        x_rec = self.inverse_transform(x)
        return F.mse_loss(x_rec, x)

###==================================================================================================================###

class TrainUnClipDecoder(nn.Module):
    """Trainer for the UnCLIP decoder model.

    Orchestrates the training of the UnCLIP decoder model, integrating CLIP embeddings, forward
    and reverse diffusion processes, and optional dimensionality reduction. Supports mixed
    precision, gradient accumulation, DDP, and comprehensive evaluation metrics.

    Parameters
    ----------
    `clip_embed_dim` : int
        Dimensionality of the input embeddings.
    `decoder_net` : nn.Module
        The UnCLIP decoder model (e.g., UnClipDecoder) to be trained.
    `clip_net` : nn.Module
        CLIP model for generating text and image embeddings.
    `train_loader` : torch.utils.data.DataLoader
        DataLoader for training data.
    `optim` : torch.optim.Optimizer
        Optimizer for training the decoder model.
    `loss_fn` : Callable
        Loss function to compute the difference between predicted and target noise.
    `clip_text_proj` : nn.Module, optional
        Projection module for text embeddings, default None.
    `clip_img_proj` : nn.Module, optional
        Projection module for image embeddings, default None.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `metrics_` : Any, optional
        Object providing evaluation metrics (e.g., FID, MSE, PSNR, SSIM, LPIPS), default None.
    `max_epochs` : int, optional
        Maximum number of training epochs (default: 100).
    `device` : str, optional
        Device for computation (default: CUDA).
    `store_path` : str, optional
        Directory to save model checkpoints (default: "unclip_decoder").
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
        Frequency (in epochs) for printing progress (default: 1).
    `use_comp` : bool, optional
        Whether to compile the model using torch.compile (default: False).
    `norm_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `reduce_clip_embed_dim` : bool, optional
        Whether to apply dimensionality reduction to embeddings (default: True).
    `trans_embed_dim` : int, optional
        Output dimensionality for reduced embeddings (default: 312).
    `norm_clip_embed` : bool, optional
        Whether to normalize CLIP embeddings (default: True).
    `finetune_clip_proj` : bool, optional
        Whether to fine-tune projection layers (default: False).
    `use_autocast`: bool
        Whether use mix percision for efficienty (default: True)
    """
    def __init__(
            self,
            clip_embed_dim: int,
            decoder_net: nn.Module,
            clip_net: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            clip_text_proj: Optional[nn.Module] = None,
            clip_img_proj: Optional[nn.Module] = None,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            metrics_: Optional[Any] = None,
            max_epochs: int = 100,
            device: str = 'cuda',
            store_path: str = "unclip_decoder",
            patience: int = 20,
            warmup_steps: int = 10000,
            val_freq: int = 10,
            use_ddp: bool = False,
            grad_acc: int = 1,
            log_freq: int = 1,
            use_comp: bool = False,
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            reduce_clip_embed_dim: bool = True,
            trans_embed_dim: int = 312,
            norm_clip_embed: bool = True,
            finetune_clip_proj: bool = False, # if text_projection and image_projection model should be finetune
            use_autocast: bool =  True
    ):
        super().__init__()
        # training configuration
        self.use_ddp = use_ddp
        self.grad_acc = grad_acc
        self.use_comp = use_comp
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        # core models
        self.decoder_net = decoder_net.to(self.device)
        self.clip_net = clip_net.to(self.device)
        self.reduce_clip_embed_dim = reduce_clip_embed_dim
        # setup distributed training
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()
        # compile and wrap models
        self._compile_models()
        self._wrap_models_for_ddp()
        # projection models (PCA equivalent in the paper)
        if self.reduce_clip_embed_dim and clip_text_proj is not None and clip_img_proj is not None:
            self.clip_text_proj = clip_text_proj.to(self.device)
            self.clip_img_proj = clip_img_proj.to(self.device)
        else:
            self.clip_text_proj = None
            self.clip_img_proj = None
        # training components
        self.clip_embed_dim = trans_embed_dim if self.reduce_clip_embed_dim else clip_embed_dim
        self.metrics_ = metrics_
        self.optim = optim
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        # training parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_freq = val_freq
        self.log_freq = log_freq
        self.norm_range = norm_range
        self.norm_clip_embed = norm_clip_embed
        self.trans_embed_dim = trans_embed_dim
        self.finetune_clip_proj = finetune_clip_proj
        self.use_autocast = use_autocast
        # checkpoint management
        self.store_path = store_path
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.best_loss = float('inf')
        self.losses = {'train_losses': [], 'val_losses': []}
        # learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optim,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optim, warmup_steps)

    def forward(self) -> Dict:
        """Trains the UnCLIP decoder model to predict noise for denoising.

        Executes the training loop, optimizing the decoder model using CLIP embeddings, mixed
        precision, gradient clipping, and learning rate scheduling. Supports validation, early
        stopping, and checkpointing.

        Returns
        -------
        loses: a ductionlaty of losses (train and validation losses)
        """
        self.decoder_net.train()
        # set text_projection and image_projection to train mode if fine-tuning
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            if self.finetune_clip_proj:
                self.clip_text_proj.train()
                self.clip_img_proj.train()
            else:
                self.clip_text_proj.eval()
                self.clip_img_proj.eval()
        # set clip model to eval mode (frozen)
        if self.clip_net is not None:
            self.clip_net.eval()

        scaler = torch.GradScaler() if self.use_autocast else None
        wait = 0
        for epoch in range(self.max_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}", disable=not self.master_process)
            # set epoch for distributed sampler if using ddp
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            train_losses_epoch = []
            for step, (imgs, texts) in enumerate(pbar):
                imgs = imgs.to(self.device, non_blocking=True)
                with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', dtype=torch.bfloat16, enabled=self.use_autocast):
                    # encode text and image with clip
                    text_embed, img_embed = self._clip_embed(imgs, texts)
                    # reduce dimensionality (pca equivalent)
                    text_embed, img_embed = self._dim_reduction(text_embed, img_embed)
                    # use decoder model to predict noise
                    pred, target = self.decoder_net(
                        img_embed,
                        text_embed,
                        imgs,
                        texts
                    )
                    loss = self.loss_fn(pred, target) / self.grad_acc
                if self.use_autocast:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % self.grad_acc == 0:
                    if self.use_autocast:
                        scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.decoder_net.parameters(), max_norm=1.0)  # covers all submodules
                    if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None and self.finetune_clip_proj:
                        torch.nn.utils.clip_grad_norm_(self.clip_text_proj.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.clip_img_proj.parameters(), max_norm=1.0)
                    if self.use_autocast:
                        scaler.step(self.optim)
                        scaler.update()
                    else:
                        self.optim.step()
                    self.optim.zero_grad()
                    if self.global_step > 0 and self.global_step < self.warmup_steps:
                        self.warmup_lr_scheduler.step()
                    self.global_step += 1
                    #torch.cuda.empty_cache()  # clear memory after optimizer step
                pbar.set_postfix({'Loss': f'{loss.item() * self.grad_acc:.4f}'})
                train_losses_epoch.append(loss.item() * self.grad_acc)
            mean_train_loss = self._mean_loss(train_losses_epoch)
            self.losses['train_losses'].append(mean_train_loss)
            if self.master_process and (epoch + 1) % self.log_freq == 0:
                current_lr = self.optim.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}")

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
        """Wraps models with DistributedDataParallel for multi-GPU training.

        Configures the decoder model and, if fine-tuning, the projection models for DDP training.
        """
        if self.use_ddp:
            self.decoder_net = self.decoder_net.to(self.ddp_local_rank)
            self.decoder_net = DDP(
                self.decoder_net,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )
            if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None and self.finetune_clip_proj:
                self.clip_text_proj = self.clip_text_proj.to(self.ddp_local_rank)
                self.clip_img_proj = self.clip_img_proj.to(self.ddp_local_rank)
                self.clip_text_proj = DDP(self.clip_text_proj, device_ids=[self.ddp_local_rank])
                self.clip_img_proj = DDP(self.clip_img_proj, device_ids=[self.ddp_local_rank])

    def _compile_models(self) -> None:
        """Compiles models for optimization if supported.

        Attempts to compile the decoder model and, if fine-tuning, the projection models using
        torch.compile for optimization, falling back to uncompiled execution if compilation fails.
        """
        if self.use_comp:
            try:
                self.decoder_net = self.decoder_net.to(self.device)
                self.decoder_net = torch.compile(self.decoder_net, mode="reduce-overhead")
                # only compile text_projection and image_projection if they are trainable
                if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None and self.finetune_clip_proj:
                    self.clip_text_proj = self.clip_text_proj.to(self.device)
                    self.clip_img_proj = self.clip_img_proj.to(self.device)
                    self.clip_text_proj = torch.compile(self.clip_text_proj, mode="reduce-overhead")
                    self.clip_img_proj = torch.compile(self.clip_img_proj, mode="reduce-overhead")
                if self.master_process:
                    print("Models compiled successfully")
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

    def _clip_embed(
            self,
            imgs: torch.Tensor,
            txts: Union[List, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes images and texts using the CLIP model.

        Generates text and image embeddings using the CLIP model, with optional normalization.

        Parameters
        ----------
        `imge` : torch.Tensor
            Input images, shape (batch_size, channels, height, width).
        `txts` : Union[List, torch.Tensor]
            Text prompts for conditional generation.

        Returns
        -------
        txt_embed : torch.Tensor
            CLIP text embeddings, shape (batch_size, embed_dim).
        img_embed : torch.Tensor
            CLIP image embeddings, shape (batch_size, embed_dim).
        """
        with torch.no_grad():
            # z_t ← CLIP_text(y)
            txt_embed = self.clip_net(data=txts, data_type="text", normalize=self.norm_clip_embed)
            # z_i ← CLIP_image(x)
            img_embed = self.clip_net(data=imgs, data_type="img", normalize=self.norm_clip_embed)
        return txt_embed, img_embed

    def _dim_reduction(
            self,
            txt_embed: torch.Tensor,
            img_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies dimensionality reduction to embeddings if enabled.

        Projects text and image embeddings to a lower-dimensional space using learned
        projection layers, mimicking PCA as used in the UnCLIP paper.

        Parameters
        ----------
        `txt_embed` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embed_dim).
        `img_embed` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embed_dim).

        Returns
        -------
        txt_embed : torch.Tensor
            Projected text embeddings, shape (batch_size, output_dim) if reduced, else unchanged.
        img_embed : torch.Tensor
            Projected image embeddings, shape (batch_size, output_dim) if reduced, else unchanged.
        """
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            if not self.finetune_clip_proj:
                with torch.no_grad():
                    txt_embed = self.clip_text_proj(txt_embed.to(self.device))
                    img_embed = self.clip_img_proj(img_embed.to(self.device))
            else:
                txt_embed = self.clip_text_proj(txt_embed.to(self.device))
                img_embed = self.clip_img_proj(img_embed.to(self.device))
        return txt_embed.to(self.device), img_embed.to(self.device)

    def _mean_loss(self, losses: List[float]) -> float:
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

    def _save_checkpoint(self, epoch: int, loss: float, pref: str = ""):
        """Saves model checkpoint.

        Saves the state of the decoder model, its submodules, optimizer, and schedulers,
        with options for best model and epoch-specific checkpoints.

        Parameters
        ----------
        `epoch` : int
            Current epoch number.
        `loss` : float
            Current loss value.

        `pref` : str, optional
            Prefix to add to checkpoint filename, default "".
        """
        if not self.master_process:
            return
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'losses': self.losses,
            # core models (submodules of decoder_model)
            'diff_net_state_dict': self.decoder_net.module.diff_net.state_dict() if self.use_ddp else self.decoder_net.diff_net.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            # training configuration
            'embedding_dim': self.clip_embed_dim,
            'output_dim': self.trans_embed_dim,
            'reduce_dim': self.reduce_clip_embed_dim,
            'normalize': self.norm_clip_embed
        }
        # save conditional model (submodule of decoder_model)
        if self.decoder_net.glide_text_encoder is not None:
            checkpoint['cond_model_state_dict'] = (
                self.decoder_net.module.glide_text_encoder.state_dict() if self.use_ddp
                else self.decoder_net.glide_text_encoder.state_dict()
            )
        # save scheduler (submodule of decoder_model, always saved)
        checkpoint['variance_scheduler_state_dict'] = (
            self.decoder_net.fwd_unclip.module.vs.state_dict() if self.use_ddp
            else self.decoder_net.fwd_unclip.vs.state_dict()
        )
        # save clip time projection layer (submodule of decoder_net)
        checkpoint['clip_time_proj_state_dict'] = (
            self.decoder_net.module.clip_time_proj.state_dict() if self.use_ddp
            else self.decoder_net.clip_time_proj.state_dict()
        )
        # save decoder projection layer (submodule of decoder_net)
        checkpoint['decoder_proj_state_dict'] = (
            self.decoder_net.module.clip_decoder_proj.state_dict() if self.use_ddp
            else self.decoder_net.clip_decoder_proj.state_dict()
        )
        # a nn.Linear projection layer
        checkpoint['clip_time_proj_state_dict'] = (
            self.decoder_net.module.clip_time_proj.state_dict() if self.use_ddp
            else self.decoder_net.clip_time_proj.state_dict()
        )
        # save projection models (pca equivalent)
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            checkpoint['text_proj_state_dict'] = (
                self.clip_text_proj.module.state_dict() if self.use_ddp
                else self.clip_text_proj.state_dict()
            )
            checkpoint['img_proj_state_dict'] = (
                self.clip_img_proj.module.state_dict() if self.use_ddp
                else self.clip_img_proj.state_dict()
            )
        # save schedulers state
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        checkpoint['warmup_scheduler_state_dict'] = self.warmup_lr_scheduler.state_dict()
        try:
            filename = f"{pref}model_epoch_{epoch}.pth"
            filepath = os.path.join(self.store_path, filename)
            os.makedirs(self.store_path, exist_ok=True)
            torch.save(checkpoint, filepath)
            print(f"Model saved at epoch {epoch} with loss: {loss:.4f}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def load_checkpoint(self, check_path: str) -> Tuple[int, float]:
        """Loads model checkpoint.

        Restores the state of the decoder model, its submodules, optimizer, and schedulers
        from a saved checkpoint, handling DDP compatibility.

        Parameters
        ----------
        `check_path` : str
            Path to the checkpoint file.

        Returns
        -------
        epoch : int
            The epoch at which the checkpoint was saved.
        loss : float
            The loss at the checkpoint.
        """
        try:
            checkpoint = torch.load(check_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found: {check_path}")

        def _load_model(model: nn.Module, state_dict: dict, model_name: str) -> None:
            """Helper function to load state dict with DDP compatibility."""
            try:
                # handle ddp state dict compatibility
                if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                if self.master_process:
                    print(f"✓ Loaded {model_name}")
            except Exception as e:
                warnings.warn(f"Failed to load {model_name}: {e}")

        # load core diffusion network model (submodule of decoder_model)
        if 'diff_net_state_dict' in checkpoint:
            _load_model(self.decoder_net.diff_net, checkpoint['diff_net_state_dict'], 'diff_net')
        # load conditional model (submodule of decoder_model) - matches your save logic
        if self.decoder_net.glide_text_encoder is not None and 'cond_model_state_dict' in checkpoint:
            _load_model(self.decoder_net.glide_text_encoder, checkpoint['cond_model_state_dict'], 'glide_text_encoder')

        # load scheduler (submodule of decoder_model)
        if 'variance_scheduler_state_dict' in checkpoint:
            try:
                _load_model(self.decoder_net.fwd_unclip.vs, checkpoint['variance_scheduler_state_dict'], 'variance_scheduler')
            except Exception as e:
                warnings.warn(f"Failed to load variance scheduler: {e}")

        # load CLIP time projection layer (submodule of decoder_model)
        if 'clip_time_proj_state_dict' in checkpoint:
            try:
                _load_model(self.decoder_net.clip_time_proj, checkpoint['clip_time_proj_state_dict'],'clip_time_proj')
            except Exception as e:
                warnings.warn(f"Failed to load CLIP time projection: {e}")

        # load decoder projection layer (submodule of decoder_model)
        if 'decoder_proj_state_dict' in checkpoint:
            try:
                _load_model(self.decoder_net.clip_decoder_proj, checkpoint['decoder_proj_state_dict'], 'clip_decoder_proj')
            except Exception as e:
                warnings.warn(f"Failed to load decoder projection: {e}")

        if 'clip_time_proj_state_dict' in checkpoint and self.master_process:
            warnings.warn("Found duplicate 'clip_time_proj_state_dict' in checkpoint - skipping to avoid conflict")

        # load projection models (pca equivalent)
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            if 'text_proj_state_dict' in checkpoint:
                _load_model(self.clip_text_proj, checkpoint['text_proj_state_dict'], 'text_proj')
            if 'image_proj_state_dict' in checkpoint:
                _load_model(self.clip_img_proj, checkpoint['image_proj_state_dict'], 'image_proj')

        # load optimizer
        if 'optim_state_dict' in checkpoint:
            try:
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
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
            if checkpoint['embedding_dim'] != self.clip_embed_dim:
                warnings.warn(
                    f"Embedding dimension mismatch: checkpoint={checkpoint['embedding_dim']}, current={self.clip_embed_dim}")

        if 'reduce_dim' in checkpoint:
            if checkpoint['reduce_dim'] != self.reduce_clip_embed_dim:
                warnings.warn(
                    f"Reduce dimension setting mismatch: checkpoint={checkpoint['reduce_dim']}, current={self.reduce_clip_embed_dim}")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Successfully loaded checkpoint from {check_path}")
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
        self.decoder_net.eval()
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            self.clip_text_proj.eval()
            self.clip_img_proj.eval()
        if self.clip_net is not None:
            self.clip_net.eval()
        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []
        with torch.no_grad():
            for imgs, txts in self.val_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                img_orig = imgs.clone()
                txt_embed, img_embed = self._clip_embed(imgs, txts)
                txt_embed, img_embed = self._dim_reduction(txt_embed, img_embed)
                p_classifier_free = torch.rand(1).item()
                p_text_drop = torch.rand(1).item()
                pred, target = self.decoder_net(
                    img_embed,
                    txt_embed,
                    imgs,
                    txts
                )
                loss = self.loss_fn(pred, target)
                val_losses.append(loss.item())
                if self.metrics_ is not None and self.decoder_net.rwd_unclip is not None:
                    xt = torch.randn_like(imgs).to(self.device)
                    timesteps = self.decoder_net.fwd_unclip.vs.inference_timesteps.flip(0)
                    for t in range(len(timesteps) - 1):
                        t_ = timesteps[t].item()
                        t_pre = timesteps[t+1].item()
                        time = torch.full((xt.shape[0],), t_, device=self.device, dtype=torch.long)
                        prev_time = torch.full((xt.shape[0],), t_pre, device=self.device, dtype=torch.long)
                        img_embed = self.decoder_net._classifier_free_guidance(img_embed)
                        txt_embed = self.decoder_net._text_dropout(txt_embed)
                        c = self.decoder_net.clip_decoder_proj(img_embed)
                        y = self.decoder_net._encode_text_with_glide(txts if txt_embed is not None else None)
                        context = self.decoder_net._conc_embed(y, c)
                        clip_img_embed = self.decoder_net.clip_time_proj(img_embed)
                        pred = self.decoder_net.diff_net(xt, time, context, clip_img_embed)
                        xt, _ = self.decoder_net.rwd_unclip(xt, time, prev_time, pred)

                    x_hat = torch.clamp(xt, min=self.norm_range[0], max=self.norm_range[1])
                    if self.norm_clip_embed:
                        x_hat = (x_hat - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])
                        x_orig = (img_orig - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])

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

        val_loss = torch.tensor(val_losses).mean().item()
        fid_avg = torch.tensor(fid_scores).mean().item() if fid_scores else float('inf')
        mse_avg = torch.tensor(mse_scores).mean().item() if mse_scores else None
        psnr_avg = torch.tensor(psnr_scores).mean().item() if psnr_scores else None
        ssim_avg = torch.tensor(ssim_scores).mean().item() if ssim_scores else None
        lpips_avg = torch.tensor(lpips_scores).mean().item() if lpips_scores else None

        if self.use_ddp:
            metrics = [val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg]
            metrics_tensors = [torch.tensor(m, device=self.device) if m is not None else torch.tensor(float('inf'), device=self.device) for m in metrics]
            for tensor in metrics_tensors:
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg = [t.item() if t.item() != float('inf') else (None if i > 1 else float('inf')) for i, t in enumerate(metrics_tensors)]

        self.decoder_net.train()
        if self.reduce_clip_embed_dim and self.clip_text_proj is not None and self.clip_img_proj is not None:
            if self.finetune_clip_proj:
                self.clip_text_proj.train()
                self.clip_img_proj.train()
            else:
                self.clip_text_proj.eval()
                self.clip_img_proj.eval()
        if self.clip_net is not None:
            self.clip_net.eval()

        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg

###==================================================================================================================###

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

###==================================================================================================================###

class SampleUnCLIP(nn.Module):
    """Generates images using the UnCLIP model pipeline.

    Combines a prior model, decoder model, CLIP model, and upsampler models to generate
    images from text prompts or noise. Performs diffusion-based sampling with classifier-free
    guidance in both prior and decoder stages, followed by upsampling to higher resolutions.

    Parameters
    ----------
    `prior_net` : nn.Module
        The UnCLIP prior model for generating image embeddings from text.
    `decoder_net` : nn.Module
        The UnCLIP decoder model for generating low-resolution images from embeddings.
    `clip_net` : nn.Module
        CLIP model for encoding text prompts into embeddings.
    `low_res_upsampler` : nn.Module
        First upsampler model for scaling images from 64x64 to 256x256.
    `high_res_upsampler` : nn.Module, optional
        Second upsampler model for scaling images from 256x256 to 1024x1024, default None.
    `device` : str, optional
        Device for computation (default: CUDA).
    `offload_device`: str
        Device for offloading (default: CPU)
    `clip_embed_dim` : int, optional
        Dimensionality of CLIP embeddings (default: 512).
    `prior_guidance_scale` : float, optional
        Classifier-free guidance scale for the prior model (default: 4.0).
    `decoder_guidance_scale` : float, optional
        Classifier-free guidance scale for the decoder model (default: 8.0).
    `batch_size` : int, optional
        Number of images to generate per batch (default: 1).
    `norm_clip_embed` : bool, optional
        Whether to normalize CLIP embeddings (default: True).
    `prior_dim_reduction` : bool, optional
        Whether to apply dimensionality reduction in the prior model (default: True).
    `init_img_size` : Tuple[int, int, int], optional
        Size of the initial generated images (default: (3, 64, 64) for RGB 64x64).
    `use_high_res_upsampler` : bool, optional
        Whether to use the second upsampler for 1024x1024 output (default: True).
    `norm_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `use_model_offloading`: bool
        Whether model offloading is used (default: True)
    """
    def __init__(
            self,
            prior_net: nn.Module,
            decoder_net: nn.Module,
            clip_net: nn.Module,
            low_res_upsampler: nn.Module,
            high_res_upsampler: Optional[nn.Module] = None,
            device: str = 'cuda',
            offload_device: str = 'cpu',
            clip_embed_dim: int = 512,
            prior_guidance_scale: float = 4.0,
            decoder_guidance_scale: float = 8.0,
            batch_size: int = 1,
            norm_clip_embed: bool = True,
            prior_dim_reduction: bool = True,
            init_img_size: Tuple[int, int, int] = (3, 64, 64),
            use_high_res_upsampler: bool = True,
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            use_model_offloading: bool = True,
            *args
    ) -> None:
        super().__init__()

        self.device = torch.device(device) if isinstance(device, str) else device
        self.offload_device = torch.device(offload_device)
        self.use_model_offloading = use_model_offloading

        # keep models on CPU initially if offloading is enabled
        init_device = self.offload_device if use_model_offloading else self.device

        self.prior_net = prior_net.to(init_device).eval()
        self.decoder_net = decoder_net.to(init_device).eval()
        self.clip_net = clip_net.to(init_device).eval()
        self.low_res_upsampler = low_res_upsampler.to(init_device).eval()
        self.high_res_upsampler = high_res_upsampler.to(init_device).eval() if high_res_upsampler else None

        self.prior_guidance_scale = prior_guidance_scale
        self.decoder_guidance_scale = decoder_guidance_scale
        self.batch_size = batch_size
        self.norm_clip_embed = norm_clip_embed
        self.prior_dim_reduction = prior_dim_reduction
        self.clip_embed_dim = clip_embed_dim
        self.init_img_size = init_img_size
        self.use_high_res_upsampler = use_high_res_upsampler
        self.norm_range = norm_range
        self.imgs_256 = None
        self.imgs_1024 = None

    def _move_model_to_device(self, model: nn.Module, target_device: torch.device):
        """Helper to move model to device if offloading is enabled."""
        if self.use_model_offloading:
            model.to(target_device)
            if target_device == self.device:
                torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(
            self,
            prompts: Optional[Union[str, List]] = None,
            norm_output: bool = True,
            save_imgs: bool = True,
            save_path: str = "unclip_samples"
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
        `norm_output` : bool, optional
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
        # ====== PRIOR STAGE: generate image embeddings from text ======
        self._move_model_to_device(self.clip_net, self.device)
        self._move_model_to_device(self.prior_net, self.device)
        # encode text prompt using CLIP
        txt_embed = self.clip_net(data=prompts, data_type="text", normalize=self.norm_clip_embed)
        # free CLIP immediately after use
        self._move_model_to_device(self.clip_net, self.offload_device)
        # initialize noise for prior sampling
        embed_noise = torch.randn((self.batch_size, self.clip_embed_dim), device=self.device)
        curr_embed = embed_noise
        if self.prior_dim_reduction:
            txt_embed_reduced = self.prior_net.clip_text_proj(txt_embed)
            curr_embed_reduced = self.prior_net.clip_img_proj(curr_embed)
        else:
            txt_embed_reduced = txt_embed
            curr_embed_reduced = curr_embed
        # prior diffusion sampling with batched CFG
        timesteps = self.decoder_net.fwd_unclip.vs.inference_timesteps.flip(0)
        for t in tqdm(range(len(timesteps) - 1), desc="Prior diffusion", leave=True):
            t_ = timesteps[t].item()
            t_pre = timesteps[t + 1].item()
            time = torch.full((self.batch_size,), t_, device=self.device, dtype=torch.long)
            prev_time = torch.full((self.batch_size,), t_pre, device=self.device, dtype=torch.long)
            guided_embed = self._prior_guided_pred(
                txt_embed_reduced, curr_embed_reduced, time
            )
            curr_embed_reduced, _ = self.prior_net.rwd_unclip(
                curr_embed_reduced, time, prev_time, guided_embed
            )
        # convert back to full embedding dimension
        if self.prior_dim_reduction:
            f_img_embed = self.prior_net.clip_img_proj.inverse_transform(curr_embed_reduced)
        else:
            f_img_embed = curr_embed_reduced
        # free prior model and intermediate tensors
        self._move_model_to_device(self.prior_net, self.offload_device)
        del embed_noise, curr_embed, curr_embed_reduced, txt_embed
        if self.prior_dim_reduction:
            del txt_embed_reduced
        torch.cuda.empty_cache()

        # ====== DECODER STAGE: generate 64x64 images from embeddings ======
        self._move_model_to_device(self.decoder_net, self.device)
        decoder_noise = torch.randn(
            (self.batch_size, *self.init_img_size), device=self.device
        )
        proj_embed = self.decoder_net.clip_decoder_proj(f_img_embed)
        glide_txt_embed = self.decoder_net._encode_text_with_glide(prompts)
        context = self.decoder_net._conc_embed(glide_txt_embed, proj_embed)
        curr_imgs = decoder_noise
        # decoder diffusion with batched CFG
        timesteps = self.decoder_net.fwd_unclip.vs.inference_timesteps.flip(0)
        for t in tqdm(range(len(timesteps) - 1), desc="Decoder 64x64", leave=True):
            t_ = timesteps[t].item()
            t_pre = timesteps[t + 1].item()
            time = torch.full((self.batch_size,), t_, device=self.device, dtype=torch.long)
            prev_time = torch.full((self.batch_size,), t_pre, device=self.device, dtype=torch.long)
            guided_pred = self._decoder_guided_pred(curr_imgs, time, context)
            curr_imgs, _ = self.decoder_net.rwd_unclip(
                curr_imgs, time, prev_time, guided_pred
            )
        samps_64x64 = curr_imgs
        # free decoder
        self._move_model_to_device(self.decoder_net, self.offload_device)
        del decoder_noise, curr_imgs, context, glide_txt_embed, proj_embed, f_img_embed
        torch.cuda.empty_cache()

        # ====== FIRST UPSAMPLER: 64x64 -> 256x256 ======
        self._move_model_to_device(self.low_res_upsampler, self.device)
        up_256_noise = torch.randn(
            (self.batch_size, self.init_img_size[0], 256, 256), device=self.device
        )
        curr_256_imgs = up_256_noise
        timesteps = self.low_res_upsampler.rwd_unclip.vs.inference_timesteps.flip(0)
        for t in tqdm(range(len(timesteps) - 1), desc="Upsampler 256x256", leave=True):
            t_ = timesteps[t].item()
            t_pre = timesteps[t + 1].item()
            time = torch.full((self.batch_size,), t_, device=self.device, dtype=torch.long)
            prev_time = torch.full((self.batch_size,), t_pre, device=self.device, dtype=torch.long)
            pred = self.low_res_upsampler(curr_256_imgs, time, samps_64x64)
            curr_256_imgs, _ = self.low_res_upsampler.rwd_unclip(
                curr_256_imgs, time, prev_time, pred
            )
        self.imgs_256 = curr_256_imgs
        # free low-res upsampler
        self._move_model_to_device(self.low_res_upsampler, self.offload_device)
        del up_256_noise, curr_256_imgs, samps_64x64
        torch.cuda.empty_cache()

        # ====== SECOND UPSAMPLER: 256x256 -> 1024x1024 ======
        if self.use_high_res_upsampler and self.high_res_upsampler:
            self._move_model_to_device(self.high_res_upsampler, self.device)
            up_1024_noise = torch.randn(
                (self.batch_size, self.init_img_size[0], 1024, 1024), device=self.device
            )
            curr_1024_imgs = up_1024_noise
            timesteps = self.high_res_upsampler.rwd_unclip.vs.inference_timesteps.flip(0)
            for t in tqdm(range(len(timesteps) - 1), desc="Upsampler 1024x1024", leave=True):
                t_ = timesteps[t].item()
                t_pre = timesteps[t + 1].item()
                time = torch.full((self.batch_size,), t_, device=self.device, dtype=torch.long)
                prev_time = torch.full((self.batch_size,), t_pre, device=self.device, dtype=torch.long)
                pred = self.high_res_upsampler(curr_1024_imgs, time, self.imgs_256)
                curr_1024_imgs, _ = self.high_res_upsampler.rwd_unclip(
                    curr_1024_imgs, time, prev_time, pred
                )
            self.imgs_1024 = curr_1024_imgs
            # free high-res upsampler
            self._move_model_to_device(self.high_res_upsampler, self.offload_device)
            del up_1024_noise, curr_1024_imgs
            torch.cuda.empty_cache()

        # ====== POST-PROCESSING ======
        if norm_output:
            f_256 = (self.imgs_256 - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])
            f_1024 = None
            if self.imgs_1024 is not None:
                f_1024 = (self.imgs_1024 - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])
        else:
            f_256 = self.imgs_256
            f_1024 = self.imgs_1024

        if save_imgs:
            self._save_images(f_256, f_1024, save_path)
        return f_1024 if f_1024 is not None else f_256

    def _prior_guided_pred(
            self,
            txt_embed: torch.Tensor,
            curr_embed: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """Batched CFG for prior"""
        batch_size = txt_embed.shape[0]
        txt_embed_batched = torch.cat([txt_embed, torch.zeros_like(txt_embed)], dim=0)
        curr_embed_batched = torch.cat([curr_embed, curr_embed], dim=0)
        t_batched = torch.cat([t, t], dim=0)
        pred_batched = self.prior_net(txt_embed_batched, curr_embed_batched, t_batched)
        pred_cond, pred_uncond = pred_batched.chunk(2, dim=0)
        return pred_uncond + self.prior_guidance_scale * (pred_cond - pred_uncond)

    def _decoder_guided_pred(
            self,
            curr_imgs: torch.Tensor,
            t: torch.Tensor,
            context: torch.Tensor
    ) -> torch.Tensor:
        """Batched CFG for decoder"""
        curr_imgs_batched = torch.cat([curr_imgs, curr_imgs], dim=0)
        t_batched = torch.cat([t, t], dim=0)
        context_batched = torch.cat([context, torch.zeros_like(context)], dim=0)
        pred_batched = self.decoder_net.diff_net(curr_imgs_batched, t_batched, context_batched, None)
        pred_cond, pred_uncond = pred_batched.chunk(2, dim=0)
        return pred_uncond + self.decoder_guidance_scale * (pred_cond - pred_uncond)

    def _save_images(self, f_256, f_1024, save_path):
        """Helper method for saving images."""
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "imgs_256"), exist_ok=True)
        if f_1024 is not None:
            os.makedirs(os.path.join(save_path, "imgs_1024"), exist_ok=True)
        for i in range(self.batch_size):
            img_path_256 = os.path.join(save_path, "imgs_256", f"img_{i + 1}.png")
            torchvision.utils.save_image(f_256[i], img_path_256)
            if f_1024 is not None:
                img_path_1024 = os.path.join(save_path, "imgs_1024", f"img_{i + 1}.png")
                torchvision.utils.save_image(f_1024[i], img_path_1024)

###==================================================================================================================###

class UpsamplerUnCLIP(nn.Module):
    """Diffusion-based upsampler for UnCLIP models.

    A U-Net-like model that upsamples low-resolution images to high-resolution images,
    conditioned on noisy high-resolution images and timesteps, using residual blocks,
    downsampling, and upsampling layers.

    Parameters
    ----------
    `fwd_unclip` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise during training.
    `rwd_unclip` : nn.Module
        Reverse diffusion module (e.g., ReverseUnCLIP) for removing noise during sampling.
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
            fwd_unclip: nn.Module,
            rwd_unclip: nn.Module,
            in_channels: int = 3,
            out_channels: int = 3,
            model_channels: int = 192,
            num_res_blocks: int = 2,
            channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
            dropout: float = 0.1,
            time_embed_dim: int = 768,
            low_res_size: int = 64,
            high_res_size: int = 256,
    ) -> None:
        super().__init__()

        self.fwd_unclip = fwd_unclip # this will be used on training time inside 'TrainUpsamplerUnCLIP'
        self.rwd_unclip = rwd_unclip # this module will be used in inference time
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
        # input projection
        # concatenate noisy high-res and upsampled low-res
        self.input_proj = nn.Conv2d(in_channels * 2, model_channels, 3, padding=1)

        # encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock(ch, model_channels * mult, time_embed_dim, dropout)
                )
                ch = model_channels * mult

            if level != len(channel_mult) - 1:
                self.down_blocks.append(DownsampleBlock(ch, ch))
        # middle blocks
        self.mid_blocks = nn.ModuleList([
            ResBlock(ch, ch, time_embed_dim, dropout),
            ResBlock(ch, ch, time_embed_dim, dropout),
        ])
        # decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                # skip connections double the input channels
                in_ch = ch + (model_channels * mult if i == 0 else 0)
                out_ch = model_channels * mult
                self.decoder_blocks.append(
                    ResBlock(in_ch, out_ch, time_embed_dim, dropout)
                )
                ch = out_ch
            if level != 0:
                self.up_blocks.append(UpsampleBlock(ch, ch))
        # output projection
        self.out_proj = nn.Sequential(
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
        x_low_up = F.interpolate(
            x_low,
            size=(x_high.shape[-2], x_high.shape[-1]),
            mode='bicubic',
            align_corners=False
        )
        # concatenate noisy high-res and upsampled low-res
        x = torch.cat([x_high, x_low_up], dim=1)
        # time embedding
        time_emb = self.time_embed(t.float())
        # input projection
        h = self.input_proj(x)
        # store skip connections
        skip_cons = []
        # encoder
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, time_emb)
            if (i + 1) % self.num_res_blocks == 0:
                skip_cons.append(h)
                down_idx = (i + 1) // self.num_res_blocks - 1
                if down_idx < len(self.down_blocks):
                    h = self.down_blocks[down_idx](h)
        # middle
        for i, block in enumerate(self.mid_blocks):
            h = block(h, time_emb)
        # decoder
        up_idx = 0
        for i, block in enumerate(self.decoder_blocks):
            # add skip connection
            if i % (self.num_res_blocks + 1) == 0 and skip_cons:
                skip = skip_cons.pop()
                h = torch.cat([h, skip], dim=1)
            h = block(h, time_emb)
            # upsample at the end of each resolution level
            if ((i + 1) % (self.num_res_blocks + 1) == 0 and
                    up_idx < len(self.up_blocks)):
                h = self.up_blocks[up_idx](h)
                up_idx += 1
        # output projection
        out = self.out_proj(h)
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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
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
        device = t.device
        half_dim = self.dim // 2
        embeds = math.log(10000) / (half_dim - 1)
        embeds = torch.exp(torch.arange(half_dim, device=device) * -embeds)
        embeds = t[:, None] * embeds[None, :]
        embeds = torch.cat([torch.sin(embeds), torch.cos(embeds)], dim=-1)
        return embeds

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
            self.skip_con = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_con = nn.Identity()

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
        return h + self.skip_con(x)


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
    `up_net` : nn.Module
        The UnCLIP upsampler model (e.g., UpsamplerUnCLIP) to be trained.
    `train_loader` : torch.utils.data.DataLoader
        DataLoader for training data, providing low- and high-resolution image pairs.
    `optim` : torch.optim.Optimizer
        Optimizer for training the upsampler model.
    `loss_fn` : Callable
        Loss function to compute the difference between predicted and target noise.
    `val_loader` : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    `max_epochs` : int, optional
        Maximum number of training epochs (default: 100).
    `device` : str, optional
        Device for computation (default: CUDA).
    `store_path` : str, optional
        Directory to save model checkpoints (default: "unclip_upsampler").
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
        Frequency (in epochs) for printing progress (default: 1).
    `use_comp` : bool, optional
        Whether to compile the model using torch.compile (default: False).
    `norm_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `norm_out` : bool, optional
        Whether to normalize inputs/outputs (default: True).
    `use_autocast` : bool, optional
        Whether to use automatic mixed precision training (default: True).
    """
    def __init__(
            self,
            up_net: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 1000,
            device: str = 'cuda',
            store_path: str = "unclip_upsampler",
            patience: int = 100,
            warmup_steps: int = 10000,
            val_freq: int = 10,
            use_ddp: bool = False,
            grad_acc: int = 1,
            log_freq: int = 1,
            use_comp: bool = False,
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            norm_out: bool = True,
            use_autocast: bool = True
    ) -> None:
        super().__init__()
        # training configuration
        self.use_ddp = use_ddp
        self.grad_acc = grad_acc
        self.use_comp = use_comp
        self.use_autocast = use_autocast
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()
        self._compile_models()
        self._wrap_models_for_ddp()

        self.up_net = up_net.to(self.device)
        self.optim = optim
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_freq = val_freq
        self.log_freq = log_freq
        self.norm_range = norm_range
        self.norm_out = norm_out
        self.store_path = store_path
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.best_loss = float('inf')
        self.losses = {'train_losses': [], 'val_losses': []}
        # learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optim,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optim, warmup_steps)

    def forward(self) -> Dict:
        """Trains the UnCLIP upsampler model to predict noise for denoising.

        Executes the training loop, optimizing the upsampler model using low- and high-resolution
        image pairs, mixed precision, gradient clipping, and learning rate scheduling. Supports
        validation, early stopping, and checkpointing.

        Returns
        -------
        losses: dictionary contaions train and validation losses.
        """
        self.up_net.train()
        scaler = torch.GradScaler() if self.use_autocast else None
        wait = 0
        for epoch in range(self.max_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}", disable=not self.master_process)
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            train_losses_epoch = []
            for step, (low_imgs, high_imgs) in enumerate(pbar):
                low_imgs = low_imgs.to(self.device, non_blocking=True)
                high_imgs = high_imgs.to(self.device, non_blocking=True)
                with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', enabled=self.use_autocast):
                    batch_size = high_imgs.shape[0]
                    timesteps = torch.randint(0, self.up_net.fwd_unclip.vs.train_steps, (batch_size,), device=self.device)
                    noise = torch.randn_like(high_imgs)
                    high_imgs_noisy, target = self.up_net.fwd_unclip(high_imgs, noise, timesteps)
                    corr_type = "gaussian_blur" if self.up_net.low_res_size == 64 else "bsr_degradation"
                    low_imgs_corr = self.corrupt_cond_img(low_imgs, corr_type)
                    pred = self.up_net(high_imgs_noisy, timesteps, low_imgs_corr)
                    loss = self.loss_fn(pred, target) / self.grad_acc
                if self.use_autocast:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % self.grad_acc == 0:
                    if self.use_autocast:
                        scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.up_net.parameters(), max_norm=1.0)
                    if self.use_autocast:
                        scaler.step(self.optim)
                        scaler.update()
                    else:
                        self.optim.step()
                    self.optim.zero_grad()
                    # torch.cuda.empty_cache()  # clear memory after optimizer step
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
                print(f"Epoch {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}")

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
                    self._save_checkpoint(epoch + 1, self.best_loss)
                else:
                    wait += 1
                    if wait >= self.patience:
                        print("Early stopping triggered")
                        self._save_checkpoint(epoch + 1, mean_train_loss)
                        break
                if (epoch + 1) % self.val_freq == 0:
                    self._save_checkpoint(epoch + 1, mean_train_loss)
        if self.use_ddp:
            destroy_process_group()
        return self.losses

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
        """Wraps models with DistributedDataParallel for multi-GPU training.

        Configures the upsampler model for DDP training by wrapping it with DistributedDataParallel.
        """
        if self.use_ddp:
            self.up_net = self.up_net.to(self.ddp_local_rank)
            self.up_net = DDP(
                self.up_net,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

    def corrupt_cond_img(self, x_low: torch.Tensor, corr_type: str = "gaussian_blur") -> torch.Tensor:
        """Corrupts the low-resolution conditioning image for robustness.

        Applies Gaussian blur or BSR degradation to the low-resolution image to simulate
        real-world degradation, as specified in the UnCLIP paper.

        Parameters
        ----------
        `x_low` : torch.Tensor
            Low-resolution input image, shape (batch_size, channels, low_res_size, low_res_size).
        `corr_type` : str, optional
            Type of corruption to apply: "gaussian_blur" or "bsr_degradation" (default: "gaussian_blur").

        Returns
        -------
        x_degraded : torch.Tensor
            Corrupted low-resolution image, same shape as input.
        """
        if corr_type == "gaussian_blur":
            # apply Gaussian blur
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            return self._gaussian_blur(x_low, kernel_size, sigma)
        elif corr_type == "bsr_degradation":
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
        kernel = self._gaussian_kernel(kernel_size, sigma).to(x.device)
        kernel = kernel.expand(x.shape[1], 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

    def _gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
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
        self.up_net.eval()
        val_losses = []
        with torch.no_grad():
            for low_imgs, high_imgs in self.val_loader:
                low_imgs = low_imgs.to(self.device, non_blocking=True)
                high_imgs = high_imgs.to(self.device, non_blocking=True)
                batch_size = high_imgs.shape[0]
                timesteps = torch.randint(0, self.up_net.fwd_unclip.vs.train_steps, (batch_size,), device=self.device)
                noise = torch.randn_like(high_imgs)
                high_imgs_noisy, target = self.up_net.fwd_unclip(high_imgs, noise, timesteps)
                corr_type = "gaussian_blur" if self.up_net.low_res_size == 64 else "bsr_degradation"
                low_imgs_corr = self.corrupt_cond_img(low_imgs, corr_type)
                pred = self.up_net(high_imgs_noisy, timesteps, low_imgs_corr)
                loss = self.loss_fn(pred, target)
                val_losses.append(loss.item())
        val_loss = torch.tensor(val_losses).mean().item()
        if self.use_ddp:
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()
        self.up_net.train()
        return val_loss

    def _save_checkpoint(self, epoch: int, loss: float, pref: str = ""):
        """Saves model checkpoint.

        Saves the state of the upsampler model, its variance scheduler, optimizer, and
        schedulers, with options for best model and epoch-specific checkpoints.

        Parameters
        ----------
        `epoch` : int
            Current epoch number.
        `loss` : float
            Current loss value.
        `prefix` : str, optional
            prefix to add to checkpoint filename, default "".
        """
        if not self.master_process:
            return
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'losses': self.losses,
            'up_net_state_dict': self.up_net.module.state_dict() if self.use_ddp else self.up_net.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'model_channels': self.up_net.model_channels,
            'num_res_blocks': self.up_net.num_res_blocks,
            'normalize': self.norm_out,
            'norm_range': self.norm_range
        }

        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        checkpoint['warmup_scheduler_state_dict'] = self.warmup_lr_scheduler.state_dict()
        try:
            filename = f"{pref}model_epoch_{epoch}.pth"
            filepath = os.path.join(self.store_path, filename)
            os.makedirs(self.store_path, exist_ok=True)
            torch.save(checkpoint, filepath)
            print(f"Model saved at epoch {epoch} with loss: {loss: .4f}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def load_checkpoint(self, check_path: str) -> Tuple[int, float]:
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
            checkpoint = torch.load(check_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found: {check_path}")
        def _load_model_state_dict(model: nn.Module, state_dict: dict, model_name: str) -> None:
            """Helper function to load state dict with DDP compatibility."""
            try:
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
        if 'up_net_state_dict' in checkpoint:
            _load_model_state_dict(self.up_net, checkpoint['up_net_state_dict'],'up_net')
        # load optimizer
        if 'optim_state_dict' in checkpoint:
            try:
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
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
            if checkpoint['model_channels'] != self.up_net.model_channels:
                warnings.warn(
                    f"Model channels mismatch: checkpoint={checkpoint['model_channels']}, current={self.up_net.model_channels}")

        if 'num_res_blocks' in checkpoint:
            if checkpoint['num_res_blocks'] != self.up_net.num_res_blocks:
                warnings.warn(
                    f"Num res blocks mismatch: checkpoint={checkpoint['num_res_blocks']}, current={self.up_net.num_res_blocks}")

        if 'normalize' in checkpoint:
            if checkpoint['normalize'] != self.normalize_image_outputs:
                warnings.warn(
                    f"Normalize setting mismatch: checkpoint={checkpoint['normalize']}, current={self.normalize_image_outputs}")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Successfully loaded checkpoint from {check_path}")
            print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        return epoch, loss