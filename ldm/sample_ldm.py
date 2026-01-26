import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Self
from transformers import BertTokenizer
from torchvision.utils import save_image
from tqdm import tqdm
import os


class SampleLDM(nn.Module):
    """Sampler for generating images using Latent Diffusion Models (LDM).

    Generates images by iteratively denoising random noise in the latent space using a
    reverse diffusion process, decoding the result back to the image space with a
    pre-trained compressor, as described in Rombach et al. (2022). Supports DDPM, DDIM,
    and SDE diffusion models, as well as conditional generation with text prompts.

    Parameters
    ----------
    diff_type : str
        Diffusion model type. Supported: "ddpm", "ddim", "sde".
    rwd_diff : nn.Module
        Reverse diffusion module (e.g., ReverseDDPM, ReverseDDIM, ReverseSDE).
    diff_net : nn.Module
        Model to predict noise added during the forward diffusion process.
    comp_model : nn.Module
        Pre-trained model to encode/decode between image and latent spaces (e.g., AutoencoderLDM).
    img_size : tuple
        Shape of generated images as (height, width).
    cond_model : nn.Module, optional
        Model for conditional generation (e.g., TextEncoder), default None.
    tokenizer : str or BertTokenizer, optional
        Tokenizer for processing text prompts, default "bert-base-uncased".
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for latent representations (default: 3).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    norm_range : tuple, optional
        Range for clamping generated images (min, max), default (-1, 1).
    """
    def __init__(
            self,
            diff_type: str,
            rwd_diff: torch.nn.Module,
            diff_net: torch.nn.Module,
            comp_model: torch.nn.Module,
            num_steps: int,
            img_size: Tuple[float, float],
            cond_model: Optional[torch.nn.Module] = None,
            tokenizer: str = "bert-base-uncased",
            batch_size: int = 1,
            in_channels: int = 3,
            device: str = 'cuda',
            max_token_length: int = 77,
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            time_eps: float = 1e-5,
            *args
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.diff_type = diff_type
        self.num_steps = num_steps
        self.diff_net = diff_net.to(self.device)
        self.rwd_diff = rwd_diff.to(self.device)
        self.comp_model = comp_model.to(self.device)
        self.cond_model = cond_model.to(self.device) if cond_model else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.in_channels = in_channels
        self.img_size = img_size
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.norm_range = norm_range
        self.time_eps = time_eps
        if not isinstance(img_size, (tuple, list)) or len(img_size) != 2 or not all(isinstance(s, int) and s > 0 for s in img_size):
            raise ValueError("img_size must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not isinstance(norm_range, (tuple, list)) or len(norm_range) != 2 or norm_range[0] >= norm_range[1]:
            raise ValueError("norm_range must be a tuple (min, max) with min < max")

    def tokenize(self, prompts: Union[List, str]):
        """Tokenizes text prompts for conditional generation.

        Converts input prompts into tokenized tensors using the specified tokenizer.

        Parameters
        ----------
        prompts : str or list
            Text prompt(s) for conditional generation. Can be a single string or a list of strings.

        Returns
        -------
        input_ids : torch.Tensor
             Tokenized input IDs, shape (batch_size, max_length).
        attention_mask : torch.Tensor
            Attention mask, shape (batch_size, max_length).
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        elif not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise TypeError("prompts must be a string or list of strings")

        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt"
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)


    def forward(
            self,
            conds: Optional[Union[List, str]] = None,
            norm_output: bool = True,
            save_imgs: bool = True,
            save_path: str = "ldm_samples"
    ) -> torch.Tensor:
        """Generates images using the reverse diffusion process in the latent space.

        Iteratively denoises random noise in the latent space using the specified reverse
        diffusion model (DDPM, DDIM, SDE), then decodes the result to the image space
        with the compressor model. Supports conditional generation with text prompts.

        Parameters
        ----------
        conds : str or list, optional
            Text prompt(s) for conditional generation, default None.
        norm_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        save_imgs : bool, optional
            If True, saves generated images to `save_path` (default: True).
        save_path : str, optional
            Directory to save generated images (default: "ldm_generated").

        Returns
        -------
        generated_imgs (torch.Tensor) - Generated images, shape (batch_size, channels, height, width). If `normalize_output` is True, images are normalized to [0, 1]; otherwise, they are clamped to `output_range`.
        """
        if conds is not None and self.cond_model is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conds is None and self.cond_model is not None:
            raise ValueError("Conditions must be provided for conditional model")
        init_samps = torch.randn(self.batch_size, self.in_channels, self.img_size[0], self.img_size[1]).to(self.device)
        self.diff_net.eval()
        self.comp_model.eval()
        if self.cond_model:
            self.cond_model.eval()

        with torch.no_grad():
            xt = init_samps
            xt, _ = self.comp_model.encode(xt)
            if self.cond_model is not None and conds is not None:
                input_ids, attention_masks = self.tokenize(conds)
                key_padding_mask = (attention_masks == 0)
                y = self.cond_model(input_ids, key_padding_mask)
            else:
                y = None
            if self.diff_type == 'ddpm':
                iterator = tqdm(
                    reversed(range(self.rwd_diff.vs.time_steps)),
                    total=self.rwd_diff.vs.time_steps,
                    desc="Sampling",
                    dynamic_ncols=True,
                    leave=True
                )
                for t in iterator:
                    time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                    pred = self.diff_net(xt, time_steps, y, clip_embeddings=None)
                    xt, _ = self.rwd_diff(xt, pred, time_steps)
            elif self.diff_type == 'ddim':
                timesteps = self.rwd_diff.vs.inference_timesteps.flip(0)
                iterator = tqdm(
                    range(len(timesteps) - 1),
                    total=len(timesteps) - 1,
                    desc="Sampling",
                    dynamic_ncols=True,
                    leave=True
                )
                for t in iterator:
                    t_current = timesteps[t].item()
                    t_next = timesteps[t + 1].item()
                    time = torch.full((xt.shape[0],), t_current, device=self.device, dtype=torch.long)
                    prev_time = torch.full((xt.shape[0],), t_next, device=self.device, dtype=torch.long)
                    pred = self.diff_net(xt, time, y, clip_embeddings=None)
                    xt, _ = self.rwd_diff(xt, time, prev_time, pred)
            else:
                iterator = tqdm(
                    range(self.num_steps),
                    total=self.num_steps,
                    desc="Sampling",
                    dynamic_ncols=True,
                    leave=True
                )
                t_schedule = torch.linspace(1.0, self.time_eps, self.num_steps + 1)
                dt = torch.tensor(-(1.0 - self.time_eps) / self.num_steps, device=xt.device, dtype=xt.dtype)
                for t in iterator:
                    t_current = float(t_schedule[t])
                    t_batch = torch.full((xt.shape[0],), t_current, dtype=xt.dtype, device=self.device)
                    pred = self.diff_net(xt, t_batch, y, None)
                    last_step = (t == self.num_steps - 1)
                    xt = self.rwd_diff(xt, pred, t_batch, dt, last_step=last_step)

            x = self.comp_model.decode(xt)
            samps = torch.clamp(x, min=self.norm_range[0], max=self.norm_range[1])
            if norm_output:
                samps = (samps - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])
            if save_imgs:
                os.makedirs(save_path, exist_ok=True)
                for t in range(samps.size(0)):
                    img_path = os.path.join(save_path, f"img_{t + 1}.png")
                    save_image(samps[t], img_path)
        return samps

    def to(self, device: torch.device) -> Self:
        """Moves the module and its components to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device for computation.

        Returns
        -------
        sample (SampleDDIM, SampleDDIM or SampleSDE) - The module moved to the specified device.
        """
        self.device = device
        self.diff_net.to(device)
        self.rwd_diff.to(device)
        self.comp_model.to(device)
        if self.cond_model:
            self.cond_model.to(device)
        return super().to(device)