import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Self
from tqdm import tqdm
from transformers import BertTokenizer
from torchvision.utils import save_image
import os




class SampleDDPM(nn.Module):
    """mage generation using a trained Denoising Diffusion Probabilistic Model (DDPM).

    Implements the sampling process for DDPM, generating images by iteratively
    denoising random noise using a trained noise predictor and reverse diffusion
    process. Supports conditional generation with text prompts via a conditional
    model, as inspired by Ho et al. (2020).

    Parameters
    ----------
    rwd_ddpm : nn.Module
        Reverse diffusion module (e.g., ReverseDDPM) for the reverse process.
    diff_net : nn.Module
        Trained model to predict noise at each time step.
    img_size : tuple
        Tuple of (height, width) specifying the generated image dimensions.
    cond_model : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    tokenizer : str, optional
        Pretrained tokenizer name from Hugging Face (default: "bert-base-uncased").
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for generated images (default: 3).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    norm_range : tuple, optional
        Tuple of (min, max) for clamping generated images (default: (-1, 1)).
    """
    def __init__(
            self,
            rwd_ddpm: torch.nn.Module,
            diff_net: torch.nn.Module,
            img_size: Tuple[int, int],
            cond_model: Optional[torch.nn.Module] = None,
            tokenizer: str = "bert-base-uncased",
            max_token_length: int = 77,
            batch_size: int = 1,
            in_channels: int = 3,
            device: Optional[str] = None,
            norm_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.rwd_ddpm = rwd_ddpm.to(self.device)
        self.diff_net = diff_net.to(self.device)
        self.cond_model = cond_model.to(self.device) if cond_model else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_token_length = max_token_length
        self.in_channels = in_channels
        self.img_size = img_size
        self.batch_size = batch_size
        self.norm_range = norm_range
        if not isinstance(img_size, (tuple, list)) or len(img_size) != 2 or not all(
                isinstance(s, int) and s > 0 for s in img_size):
            raise ValueError("image_size must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(norm_range, (tuple, list)) or len(norm_range) != 2 or norm_range[0] >= norm_range[1]:
            raise ValueError("norm_range must be a tuple (min, max) with min < max")

    def tokenize(self, prompts: Union[List, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes text prompts for conditional generation.

        Converts input prompts into tokenized input IDs and attention masks using the
        specified tokenizer, suitable for use with the conditional model.

        Parameters
        ----------
        prompts : str or list
            A single text prompt or a list of text prompts.

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
            conds: Optional[Union[str, List]] = None,
            norm_output: bool = True,
            save_imgs: bool = True,
            save_path: str = "ddpm_samples"
    ) -> torch.Tensor:
        """Generates images using the DDPM sampling process.

        Iteratively denoises random noise to generate images using the reverse diffusion
        process and noise predictor. Supports conditional generation with text prompts.
        Optionally saves generated images to a specified directory.

        Parameters
        ----------
        conds : str or list, optional
            Text prompt(s) for conditional generation, default None.
        norm_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        save_imgs : bool, optional
            If True, saves generated images to `save_path` (default: True).
        save_path : str, optional
            Directory to save generated images (default: "ddpm_generated").

        Returns
        -------
        samps (torch.Tensor) - Generated images, shape (batch_size, in_channels, height, width).
        If `norm_output` is True, images are normalized to [0, 1]; otherwise, they are clamped to `norm_range`.
        """
        if conds is not None and self.cond_model is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conds is None and self.cond_model is not None:
            raise ValueError("Conditions must be provided for conditional model")
        init_samps = torch.randn(self.batch_size, self.in_channels, self.img_size[0], self.img_size[1], device=self.device)
        self.diff_net.eval()
        if self.cond_model:
            self.cond_model.eval()
        iterator = tqdm(
            reversed(range(self.rwd_ddpm.vs.time_steps)),
            total=self.rwd_ddpm.vs.time_steps,
            desc="Sampling",
            dynamic_ncols=True,
            leave=True,
        )
        if self.cond_model is not None and conds is not None:
            input_ids, attention_masks = self.tokenize(conds)
            key_padding_mask = (attention_masks == 0)
            y = self.cond_model(input_ids, key_padding_mask)
        else:
            y = None
        with torch.no_grad():
            xt = init_samps
            for step in iterator:
                time_steps = torch.full((self.batch_size,), step, device=self.device, dtype=torch.long)
                pred = self.diff_net(xt, time_steps, y, clip_embeddings=None)
                xt, _ = self.rwd_ddpm(xt, pred, time_steps)
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
        sample_ddpm (SampleDDPM) - moved to the specified device.
        """
        self.device = device
        self.diff_net.to(device)
        self.rwd_ddpm.to(device)
        if self.cond_model:
            self.cond_model.to(device)
        return super().to(device)