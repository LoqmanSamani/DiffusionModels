"""Image generation using a trained Denoising Diffusion Probabilistic Model (DDPM).

This module implements the sampling process for generating images with a trained DDPM
model, as described in Ho et al. (2020, "Denoising Diffusion Probabilistic Models").
It supports both unconditional and conditional generation with text prompts.
"""


import torch
import torch.nn as nn
from transformers import BertTokenizer



class SampleDDPM(nn.Module):
    """Image generation using a trained DDPM model.

    Implements the sampling process for DDPM, generating images by iteratively
    denoising random noise using a trained noise predictor and reverse diffusion
    process. Supports conditional generation with text prompts via a conditional
    model, as inspired by Ho et al. (2020).

    Parameters
    ----------
    reverse_diffusion : nn.Module
        Reverse diffusion module (e.g., ReverseDDPM) for the reverse process.
    noise_predictor : nn.Module
        Trained model to predict noise at each time step.
    image_shape : tuple
        Tuple of (height, width) specifying the generated image dimensions.
    conditional_model : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    tokenizer : str, optional
        Pretrained tokenizer name from Hugging Face (default: "bert-base-uncased").
    max_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    batch_size : int, optional
        Number of images to generate per batch (default: 1).
    in_channels : int, optional
        Number of input channels for generated images (default: 3).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    output_range : tuple, optional
        Tuple of (min, max) for clamping generated images (default: (-1, 1)).

    Attributes
    ----------
    device : torch.device
        Device used for computation.
    reverse : nn.Module
        Reverse diffusion module.
    noise_predictor : nn.Module
        Noise prediction model.
    conditional_model : nn.Module or None
        Conditional model for text-based generation, if provided.
    tokenizer : BertTokenizer
        Tokenizer for processing text prompts.
    max_length : int
        Maximum length for tokenized prompts.
    in_channels : int
        Number of input channels.
    image_shape : tuple
        Shape of generated images (height, width).
    batch_size : int
        Batch size for generation.
    output_range : tuple
        Range for clamping generated images.

    Raises
    ------
    ValueError
        If `image_shape` is not a tuple of two positive integers, `batch_size` is not
        positive, or `output_range` is not a valid (min, max) tuple with min < max.
    """
    def __init__(self, reverse_diffusion, noise_predictor, image_shape, conditional_model=None, tokenizer="bert-base-uncased",
                 max_length=77, batch_size=1, in_channels=3, device=None, output_range=(-1, 1)):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reverse = reverse_diffusion.to(self.device)
        self.noise_predictor = noise_predictor.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.in_channels = in_channels
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.output_range = output_range

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(
                isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(output_range, (tuple, list)) or len(output_range) != 2 or output_range[0] >= output_range[1]:
            raise ValueError("output_range must be a tuple (min, max) with min < max")

    def tokenize(self, prompts):
        """Tokenizes text prompts for conditional generation.

        Converts input prompts into tokenized input IDs and attention masks using the
        specified tokenizer, suitable for use with the conditional model.

        Parameters
        ----------
        prompts : str or list
            A single text prompt or a list of text prompts.

        Returns
        -------
        tuple
            A tuple containing:
            - input_ids: Tokenized input IDs, shape (batch_size, max_length).
            - attention_mask: Attention mask, shape (batch_size, max_length).

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
        """Generates images using the DDPM sampling process.

        Iteratively denoises random noise to generate images using the reverse diffusion
        process and noise predictor. Supports conditional generation with text prompts.

        Parameters
        ----------
        conditions : str or list, optional
            Text prompt(s) for conditional generation, default None.
        normalize_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).

        Returns
        -------
        torch.Tensor
            Generated images, shape (batch_size, in_channels, height, width).
            If `normalize_output` is True, images are normalized to [0, 1]; otherwise,
            they are clamped to `output_range`.

        Raises
        ------
        ValueError
            If `conditions` is provided but no conditional model is specified, or if
            a conditional model is specified but `conditions` is None.
        """

        if conditions is not None and self.conditional_model is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conditions is None and self.conditional_model is not None:
            raise ValueError("Conditions must be provided for conditional model")

        noisy_samples = torch.randn(self.batch_size, self.in_channels, self.image_shape[0], self.image_shape[1]).to(self.device)

        self.noise_predictor.eval()
        self.reverse.eval()
        if self.conditional_model:
            self.conditional_model.eval()

        with torch.no_grad():
            xt = noisy_samples
            for t in reversed(range(self.reverse.hyper_params.num_steps)):
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)
                if self.conditional_model is not None and conditions is not None:
                    input_ids, attention_masks = self.tokenize(conditions)
                    key_padding_mask = (attention_masks == 0)
                    y = self.conditional_model(input_ids, key_padding_mask)
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)
                xt = self.reverse(xt, predicted_noise, time_steps)

            generated_imgs = torch.clamp(xt, min=self.output_range[0], max=self.output_range[1])
            if normalize_output:
                generated_imgs = (generated_imgs - self.output_range[0]) / (self.output_range[1] - self.output_range[0])

        return generated_imgs

    def to(self, device):
        """Moves the module and its components to the specified device.

        Updates the device attribute and moves the reverse diffusion, noise predictor,
        and conditional model (if present) to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device for the module and its components.

        Returns
        -------
        SampleDDPM
            The module itself, moved to the specified device.
        """
        self.device = device
        self.noise_predictor.to(device)
        self.reverse.to(device)
        self.compressor.to(device)
        if self.conditional_model:
            self.conditional_model.to(device)
        return super().to(device)