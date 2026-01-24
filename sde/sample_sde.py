import torch
import torch.nn as nn
from torchvision.utils import save_image
from transformers import BertTokenizer
import os
from typing import Tuple, Optional, List, Union, Self
from tqdm import tqdm




class SampleSDE(nn.Module):
    """Sampler for generating images using SDE-based generative models.

    Generates images by iteratively denoising random noise using the reverse SDE process
    and a trained noise predictor, as described in Song et al. (2021). Supports both
    unconditional and conditional generation with text prompts.

    Parameters
    ----------
    rwd_sde : ReverseSDE
        Reverse SDE diffusion module for denoising.
    score_net : nn.Module
        Model to predict noise added during the forward SDE process.
    img_size : tuple
        Shape of generated images as (height, width).
    cond_model : nn.Module, optional
        Model for conditional generation (e.g., TextEncoder), default None.
    tokenizer : str or BertTokenizer, optional
        Tokenizer for processing text prompts, default "bert-base-uncased".
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for generated images (default: 3).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    norm_range : tuple, optional
        Range for clamping generated images (min, max), default (-1, 1).
    """
    def __init__(
            self,
            rwd_sde: torch.nn.Module,
            score_net: torch.nn.Module,
            img_size: Tuple[int, int],
            pred_noise: bool,
            cond_model: Optional[torch.nn.Module] = None,
            tokenizer: str = "bert-base-uncased",
            max_token_length: int = 77,
            batch_size: int = 1,
            in_channels: int = 3,
            device: str = 'cuda',
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            time_eps: float =  1e-5
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.rwd_sde = rwd_sde.to(self.device)
        self.score_net = score_net.to(self.device)
        self.cond_model = cond_model.to(self.device) if cond_model else None
        self.pred_noise = pred_noise
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_token_length = max_token_length
        self.in_channels = in_channels
        self.img_size = img_size
        self.batch_size = batch_size
        self.norm_range = norm_range
        self.time_eps = time_eps

        if not isinstance(img_size, (tuple, list)) or len(img_size) != 2 or not all(isinstance(s, int) and s > 0 for s in img_size):
            raise ValueError("img_size must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(norm_range, (tuple, list)) or len(norm_range) != 2 or norm_range[0] >= norm_range[1]:
            raise ValueError("norm_range must be a tuple (min, max) with min < max")

    def tokenize(self, prompts: Union[str, List]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes text prompts for conditional generation.

        Converts input prompts into tokenized tensors using the specified tokenizer.

        Parameters
        ----------
        prompts : str or list
            Text prompt(s) for conditional generation. Can be a single string or a list
            of strings.

        Returns
        -------
        input_ids : torch.Tensor
             Tokenized input IDs, shape (batch_size, max_token_length).
        attention_mask : torch.Tensor
            Attention mask, shape (batch_size, max_token_length).
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
            num_steps: int,
            conds: Optional[Union[str, List]] = None,
            norm_output: bool = True,
            save_imgs: bool = True,
            save_path: str = "sde_samples"
    ) -> torch.Tensor:
        """Generates images using the reverse SDE sampling process.

        Iteratively denoises random noise to generate images using the reverse SDE process
        and noise predictor. Supports conditional generation with text prompts.

        Parameters
        ----------
        conds : str or list, optional
            Text prompt(s) for conditional generation, default None.
        norm_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        save_imgs : bool, optional
            If True, saves generated images to `save_path` (default: True).
        save_path : str, optional
            Directory to save generated images (default: "sde_generated").

        Returns
        -------
        generated_imgs (torch.Tensor) - Generated images, shape (batch_size, in_channels, height, width). If `normalize_output` is True, images are normalized to [0, 1]; otherwise, they are clamped to `output_range`.
        """
        if conds is not None and self.cond_model is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conds is None and self.cond_model is not None:
            raise ValueError("Conditions must be provided for conditional model")

        init_samps = torch.randn(self.batch_size, self.in_channels, self.img_size[0], self.img_size[1]).to(self.device)
        self.score_net.eval()
        self.rwd_sde.eval()
        if self.cond_model:
            self.cond_model.eval()

        if self.cond_model is not None and conds is not None:
            input_ids, attention_masks = self.tokenize(conds)
            key_padding_mask = (attention_masks == 0)
            y = self.cond_model(input_ids, key_padding_mask)
        else:
            y = None

        t_schedule = torch.linspace(1.0, self.time_eps, num_steps + 1, device=self.device)
        dt = -(1.0 - self.time_eps) / num_steps
        iterator = tqdm(range(num_steps), desc="Sampling")
        with torch.no_grad():
            xt = init_samps
            for step in iterator:
                t_current = float(t_schedule[step])
                t_batch = torch.full((self.batch_size,), t_current, dtype=xt.dtype, device=self.device)
                pred = self.score_net(xt, t_batch, y, clip_embeddings=None)
                if self.pred_noise:
                    std = self.rwd_sde.vs.std(t_batch)
                    while std.dim() < len(xt.shape):
                        std = std.unsqueeze(-1)
                    score = -pred / (std + self.rwd_sde.eps)
                else:
                    score = pred
                last_step = (step == num_steps - 1)
                xt = self.rwd_sde(xt, score, t_batch, dt, last_step = last_step)

            samps = torch.clamp(xt, min=self.norm_range[0], max=self.norm_range[1])
            if norm_output:
                samps = (samps - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])
            if save_imgs:
                os.makedirs(save_path, exist_ok=True)
                for i in range(samps.size(0)):
                    img_path = os.path.join(save_path, f"img_{i+1}.png")
                    save_image(samps[i], img_path)

        return samps

    def to(self, device: torch.device) -> Self:
        """Moves the module and its components to the specified device.

        Updates the device attribute and moves the reverse diffusion, noise predictor,
        and conditional model (if present) to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device for the module and its components.

        Returns
        -------
        sample_sde (SampleSDE) - moved to the specified device.
        """
        self.device = device
        self.score_net.to(device)
        self.rwd_sde.to(device)
        if self.cond_model:
            self.cond_model.to(device)
        return super().to(device)