"""
**Utilities for text encoding, noise prediction, and evaluation in diffusion models**

This module provides core components for building diffusion model pipelines, including
text encoding, used as conditional model, U-Net-based noise prediction, and image quality evaluation. These
utilities support various diffusion model architectures, such as DDPM, DDIM, LDM, and
SDE, and are designed for standalone use in model training and sampling.

**Primary Components**

- **TextEncoder**: Encodes text prompts into embeddings using a pre-trained BERT model or a custom transformer.
- **NoisePredictor**: U-Net-like architecture for predicting noise in diffusion models, supporting time and text conditioning.
- **Metrics**: Computes image quality metrics (MSE, PSNR, SSIM, FID, LPIPS) for evaluating generated images.

**Notes**

- The primary components are intended to be imported directly for use in diffusion model workflows.
- Additional supporting classes and functions in this module provide internal functionality for the primary components.


---------------------------------------------------------------------------------
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from pytorch_fid import fid_score
from torchvision.utils import save_image
from transformers import BertModel
import os
import lpips
import math
import shutil
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from typing import Optional, Tuple, List


###==================================================================================================================###

class TextEncoder(torch.nn.Module):
    """Transformer-based encoder for text prompts in conditional diffusion models.

    Encodes text prompts into embeddings using either a pre-trained BERT model or a
    custom transformer architecture. Used as the `conditional_model` in diffusion models
    (e.g., DDPM, DDIM, SDE, LDM) to provide conditional inputs for noise prediction.

    Parameters
    ----------
    use_pretrained_model : bool, optional
        If True, uses a pre-trained BERT model; otherwise, builds a custom transformer
        (default: True).
    model_name : str, optional
        Name of the pre-trained model to load (default: "bert-base-uncased").
    vocabulary_size : int, optional
        Size of the vocabulary for the custom transformer's embedding layer
        (default: 30522).
    num_layers : int, optional
        Number of transformer encoder layers for the custom transformer (default: 6).
    input_dimension : int, optional
        Input embedding dimension for the custom transformer (default: 768).
    output_dimension : int, optional
        Output embedding dimension for both pre-trained and custom models
        (default: 768).
    num_heads : int, optional
        Number of attention heads in the custom transformer (default: 8).
    context_length : int, optional
        Maximum sequence length for text prompts (default: 77).
    dropout_rate : float, optional
        Dropout rate for attention and feedforward layers (default: 0.1).
    qkv_bias : bool, optional
        If True, includes bias in query, key, and value projections for the custom
        transformer (default: False).
    scaling_value : int, optional
        Scaling factor for the feedforward layer's hidden dimension in the custom
        transformer (default: 4).
    epsilon : float, optional
        Epsilon for layer normalization in the custom transformer (default: 1e-5).
    use_learned_pos : bool, optional
        If True, in the transformer structure uses learnable positional embeddings instead of sinusoidal encodings
        (default: False).

    **Notes**

    - When `use_pretrained_model` is True, the BERT model's parameters are frozen
      (`requires_grad = False`), and a projection layer maps outputs to
      `output_dimension`.
    - The custom transformer uses `EncoderLayer` modules with multi-head attention and
      feedforward networks, supporting variable input/output dimensions.
    - The output shape is (batch_size, context_length, output_dimension).
    """
    def __init__(
            self,
            use_pretrained_model: bool = True,
            model_name: str = "bert-base-uncased",
            vocabulary_size: int = 30522,
            num_layers: int = 6,
            input_dimension: int = 768,
            output_dimension: int = 768,
            num_heads: int = 8,
            context_length: int = 77,
            dropout_rate: float = 0.1,
            qkv_bias: bool = False,
            scaling_value: int = 4,
            epsilon: float = 1e-5,
            use_learned_pos: bool = False
    ) -> None:
        super().__init__()
        self.use_pretrained_model = use_pretrained_model
        if self.use_pretrained_model:
            self.bert = BertModel.from_pretrained(model_name)
            for param in self.bert.parameters():
                param.requires_grad = False
            self.projection = nn.Linear(self.bert.config.hidden_size, output_dimension)
        else:
            self.embedding = Embedding(
                vocabulary_size=vocabulary_size,
                embedding_dimension=input_dimension,
                max_context_length=context_length,
                use_learned_pos=use_learned_pos
            )
            self.layers = torch.nn.ModuleList([
                EncoderLayer(
                    input_dimension=input_dimension,
                    output_dimension=output_dimension,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias,
                    scaling_value=scaling_value,
                    epsilon=epsilon
                )
                for _ in range(num_layers)
            ])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encodes text prompts into embeddings.

        Processes input token IDs and an optional attention mask to produce embeddings
        using either a pre-trained BERT model or a custom transformer.

        Parameters
        ----------
        x : torch.Tensor
            Token IDs, shape (batch_size, seq_len).
        attention_mask : torch.Tensor, optional
            Attention mask, shape (batch_size, seq_len), where 0 indicates padding
            tokens to ignore (default: None).

        Returns
        -------
        x (torch.Tensor) - Encoded embeddings, shape (batch_size, seq_len, output_dimension).

        **Notes**

        - For pre-trained BERT, the `last_hidden_state` is projected to
          `output_dimension` and this layer is the only trainable layer in the model.
        - For the custom transformer, token embeddings are processed through
          `Embedding` and `EncoderLayer` modules.
        - The attention mask should be 0 for padding tokens and 1 for valid tokens when
          using the custom transformer, or follow BERT's convention for pre-trained
          models.
        """
        if self.use_pretrained_model:
            x = self.bert(input_ids=x, attention_mask=attention_mask)
            x = x.last_hidden_state
            x = self.projection(x)
        else:
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x, attention_mask=attention_mask)
        return x
###==================================================================================================================###

class EncoderLayer(torch.nn.Module):
    """Single transformer encoder layer with multi-head attention and feedforward network.

    Used in the custom transformer of `TextEncoder` to process embedded text prompts.

    Parameters
    ----------
    input_dimension : int
        Input embedding dimension.
    output_dimension : int
        Output embedding dimension.
    num_heads : int
        Number of attention heads.
    dropout_rate : float
        Dropout rate for attention and feedforward layers.
    qkv_bias : bool
        If True, includes bias in query, key, and value projections.
    scaling_value : int
        Scaling factor for the feedforward layer’s hidden dimension.
    epsilon : float, optional
        Epsilon for layer normalization (default: 1e-5).

    **Notes**

    - The layer follows the standard transformer encoder architecture: attention,
      residual connection, normalization, feedforward, residual connection,
      normalization.
    - The attention mechanism uses `batch_first=True` for compatibility with
      `TextEncoder`’s input format.
    """
    def __init__(
            self,
            input_dimension: int,
            output_dimension: int,
            num_heads: int,
            dropout_rate: float,
            qkv_bias: bool,
            scaling_value: int,
            epsilon: float = 1e-5
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dimension,
            num_heads=num_heads,
            dropout=dropout_rate,
            bias=qkv_bias,
            batch_first=True
        )
        self.output_projection = nn.Linear(input_dimension, output_dimension) if input_dimension != output_dimension else nn.Identity()
        self.norm1 = self.norm1 = nn.LayerNorm(normalized_shape=input_dimension, eps=epsilon)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.feedforward = FeedForward(
            embedding_dimension=input_dimension,
            scaling_value=scaling_value,
            dropout_rate=dropout_rate
        )
        self.norm2 = nn.LayerNorm(normalized_shape=output_dimension, eps=epsilon)
        self.dropout2 = nn.Dropout(dropout_rate)
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Processes input embeddings through attention and feedforward layers.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings, shape (batch_size, seq_len, input_dimension).
        attention_mask : torch.Tensor, optional
            Attention mask, shape (batch_size, seq_len), where 0 indicates padding
            tokens to ignore (default: None).

        Returns
        -------
        x (torch.Tensor) - Processed embeddings, shape (batch_size, seq_len, output_dimension).

        **Notes**

        - The attention mask is passed as `key_padding_mask` to
          `nn.MultiheadAttention`, where 0 indicates padding tokens.
        - Residual connections and normalization are applied after attention and
          feedforward layers.
        """
        attn_output, _ = self.attention(x, key_padding_mask=attention_mask)
        attn_output = self.output_projection(attn_output)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

###==================================================================================================================###

class FeedForward(torch.nn.Module):
    """Feedforward network for transformer encoder layers.

    Used in `EncoderLayer` to process attention outputs with a two-layer MLP and GELU
    activation.

    Parameters
    ----------
    embedding_dimension : int
        Input and output embedding dimension.
    scaling_value : int
        Scaling factor for the hidden layer’s dimension (hidden_dim =
        embedding_dimension * scaling_value).
    dropout_rate : float, optional
        Dropout rate after the hidden layer (default: 0.1).


    **Notes**

    - The hidden layer dimension is `embedding_dimension * scaling_value`, following
      standard transformer feedforward designs.
    - GELU activation is used for non-linearity.
    """
    def __init__(self, embedding_dimension: int, scaling_value: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * scaling_value,
                bias=True
            ),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(
                in_features=embedding_dimension * scaling_value,
                out_features=embedding_dimension,
                bias=True
            )
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input embeddings through the feedforward network.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings, shape (batch_size, seq_len, embedding_dimension).

        Returns
        -------
        x (torch.Tensor) - Processed embeddings, shape (batch_size, seq_len, embedding_dimension).
        """
        return self.layers(x)

###==================================================================================================================###


class Attention(nn.Module):
    """Attention module for NoisePredictor, supporting text conditioning or self-attention.

    Applies multi-head attention to enhance features, with optional text embeddings for
    conditional generation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (embedding dimension for attention).
    y_embed_dim : int, optional
        Dimensionality of text embeddings (default: 768).
    num_heads : int, optional
        Number of attention heads (default: 4).
    num_groups : int, optional
        Number of groups for group normalization (default: 8).
    dropout_rate : float, optional
        Dropout rate for attention and output (default: 0.1).

    Attributes
    ----------
    in_channels : int
        Input channel dimension.
    y_embed_dim : int
        Text embedding dimension.
    num_heads : int
        Number of attention heads.
    dropout_rate : float
        Dropout rate.
    attention : torch.nn.MultiheadAttention
        Multi-head attention with `batch_first=True`.
    norm : torch.nn.GroupNorm
        Group normalization before attention.
    dropout : torch.nn.Dropout
        Dropout layer for output.
    y_projection : torch.nn.Linear
        Projection for text embeddings to match `in_channels`.

    Raises
    ------
    AssertionError
        If input channels do not match `in_channels`.
    ValueError
        If text embeddings (`y`) have incorrect dimensions after projection.
    """
    def __init__(
            self,
            in_channels: int,
            y_embed_dim: int = 768,
            num_heads: int = 4,
            num_groups: int = 8,
            dropout_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.y_embed_dim = y_embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.y_projection = nn.Linear(y_embed_dim, in_channels)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Applies attention to input features with optional text conditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).
        y : torch.Tensor, optional
            Text embeddings, shape (batch_size, seq_len, y_embed_dim) or
            (batch_size, y_embed_dim) (default: None).

        Returns
        -------
        torch.Tensor
            Output tensor, same shape as input `x`.
        """
        batch_size, channels, h, w = x.shape
        assert channels == self.in_channels, f"Expected {self.in_channels} channels, got {channels}"
        x_reshaped = x.view(batch_size, channels, h * w).permute(0, 2, 1)
        if y is not None:
            y = self.y_projection(y)
            if y.dim() != 3:
                if y.dim() == 2:
                    y = y.unsqueeze(1)
                else:
                    raise ValueError(
                        f"Expected y to be 2D or 3D after projection, got {y.dim()}D with shape {y.shape}"
                    )
            if y.shape[-1] != self.in_channels:
                raise ValueError(
                    f"Expected y's embedding dim to match in_channels ({self.in_channels}), got {y.shape[-1]}"
                )
            out, _ = self.attention(x_reshaped, y, y)
        else:
            out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        out = out.permute(0, 2, 1).view(batch_size, channels, h, w)
        out = self.norm(out)
        out = self.dropout(out)
        return out


###==================================================================================================================###


class Embedding(nn.Module):
    """Token and positional embedding layer for transformer inputs.

    Used in `TextEncoder`’s transformer to embed token IDs and add positional encodings.

    Parameters
    ----------
    vocabulary_size : int
        Size of the vocabulary for token embeddings.
    embedding_dimension : int, optional
        Dimension of token and positional embeddings (default: 768).
    max_context_length : int, optional
        Maximum sequence length for precomputing positional encodings (default: 77).
    use_learned_pos : bool, optional
        If True, uses learnable positional embeddings instead of sinusoidal encodings
        (default: False).

    **Notes**

    - Supports both sinusoidal (fixed) and learned positional embeddings, selectable via
      `use_learned_pos`.
    - Sinusoidal encodings follow the transformer architecture, computed on-the-fly for
      memory efficiency and cached for sequences up to `max_context_length`.
    - Learned positional embeddings are initialized as a learnable parameter for flexibility.
    - Optimized for device-agnostic operation, ensuring seamless CPU/GPU transitions.
    - The output shape is (batch_size, seq_len, embedding_dimension).
    """
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int = 768,
        max_context_length: int = 77,
        use_learned_pos: bool = False
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.max_context_length = max_context_length
        self.use_learned_pos = use_learned_pos

        # Token embedding layer
        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension
        )

        if use_learned_pos:
            # Learnable positional embeddings
            self.positional_embedding = nn.Parameter(
                torch.randn(1, max_context_length, embedding_dimension) / math.sqrt(embedding_dimension)
            )
        else:
            # Register buffer for sinusoidal encodings
            self.register_buffer(
                "positional_encoding_cache",
                torch.empty(1, 0, embedding_dimension, dtype=torch.float32)
            )

    def _generate_positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generates sinusoidal positional encodings for transformer inputs.

        Computes positional encodings using sine and cosine functions.

        Parameters
        ----------
        seq_len : int
            Length of the sequence for which to generate positional encodings.
        device : torch.device
            Device on which to create the positional encodings.

        Returns
        -------
        torch.Tensor
            Positional encodings, shape (1, seq_len, embedding_dimension), where
            even-indexed dimensions use sine and odd-indexed dimensions use cosine.

        **Notes**

        - Uses the formula: for position `pos` and dimension `i`,
          `PE(pos, 2i) = sin(pos / 10000^(2i/d))` and
          `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`, where `d` is `embedding_dimension`.
        - Fully vectorized for efficiency and supports any sequence length.
        """
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dimension, 2, dtype=torch.float32, device=device) *
            (-math.log(10000.0) / self.embedding_dimension)
        )
        pos_enc = torch.zeros((1, seq_len, self.embedding_dimension), dtype=torch.float32, device=device)
        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term[:, :-1] if self.embedding_dimension % 2 else div_term)
        return pos_enc

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embeds token IDs and adds positional encodings.

        Parameters
        ----------
        token_ids : torch.Tensor
            Token IDs, shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Embedded tokens with positional encodings, shape
            (batch_size, seq_len, embedding_dimension).

        **Notes**

        - Automatically handles sequences longer than `max_context_length` by generating
          positional encodings on-the-fly.
        - For learned positional embeddings, sequences longer than `max_context_length`
          will raise an error unless truncated.
        - Ensures device compatibility by generating encodings on the input’s device.
        """
        assert token_ids.dim() == 2, "Input token_ids should be of shape (batch_size, seq_len)"
        batch_size, seq_len = token_ids.size()
        device = token_ids.device

        # Compute token embeddings
        token_embedded = self.token_embedding(token_ids)

        # Handle positional embeddings
        if self.use_learned_pos:
            if seq_len > self.max_context_length:
                raise ValueError(
                    f"Sequence length ({seq_len}) exceeds max_context_length ({self.max_context_length}) "
                    "for learned positional embeddings."
                )
            position_encoded = self.positional_embedding[:, :seq_len, :]
        else:
            # Use cached sinusoidal encodings if available and sufficient
            if (self.positional_encoding_cache.size(1) < seq_len or
                    self.positional_encoding_cache.device != device):
                self.positional_encoding_cache = self._generate_positional_encoding(
                    max(seq_len, self.max_context_length), device
                )
            position_encoded = self.positional_encoding_cache[:, :seq_len, :]

        return token_embedded + position_encoded


###==================================================================================================================###


class NoisePredictor(nn.Module):
    """U-Net-like architecture for noise prediction in Diffusion Models.

    Predicts noise for diffusion models (DDPM, DDIM, SDE), incorporating
    time embeddings and optional text conditioning. used as the `noise_predictor` in
    `Train` and `Sample` from the `ldm`, `sde`, `ddpm`, `ddim` modules.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    down_channels : list of int
        List of output channels for downsampling blocks.
    mid_channels : list of int
        List of channels for middle blocks.
    up_channels : list of int
        List of output channels for upsampling blocks.
    down_sampling : list of bool
        List indicating whether to downsample in each down block.
    time_embed_dim : int
        Dimensionality of time embeddings.
    y_embed_dim : int
        Dimensionality of text embeddings for conditioning.
    num_down_blocks : int
        Number of convolutional layer pairs per down block.
    num_mid_blocks : int
        Number of convolutional layer pairs per middle block.
    num_up_blocks : int
        Number of convolutional layer pairs per up block.
    dropout_rate : float, optional
        Dropout rate for convolutional and attention layers (default: 0.1).
    down_sampling_factor : int, optional
        Factor for spatial downsampling/upsampling (default: 2).
    where_y : bool, optional
        If True, text embeddings are used in attention; if False, concatenated to input
        (default: True).
    y_to_all : bool, optional
        If True, apply text-conditioned attention to all layers; if False, only first layer
        (default: False).

    **Notes**

    - The architecture follows a U-Net structure with downsampling, bottleneck, and
      upsampling blocks, incorporating time embeddings and optional text conditioning via
      attention or concatenation.
    - Skip connections link down and up blocks, with channel adjustments for concatenation.
    - Weights are initialized with Kaiming normal (Leaky ReLU nonlinearity) for stability.
    - Input and output tensors have the same shape.
    """
    def __init__(
            self,
            in_channels: int,
            down_channels: List[int],
            mid_channels: List[int],
            up_channels: List[int],
            down_sampling: List[bool],
            time_embed_dim: int,
            y_embed_dim: int,
            num_down_blocks: int,
            num_mid_blocks: int,
            num_up_blocks: int,
            dropout_rate: float = 0.1,
            down_sampling_factor: int = 2,
            where_y: bool = True,
            y_to_all: bool = False
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.up_channels = up_channels
        self.down_sampling = down_sampling
        self.time_embed_dim = time_embed_dim
        self.y_embed_dim = y_embed_dim
        self.num_down_blocks = num_down_blocks
        self.num_mid_blocks = num_mid_blocks
        self.num_up_blocks = num_up_blocks
        self.dropout_rate = dropout_rate
        self.where_y = where_y
        self.up_sampling = list(reversed(self.down_sampling))
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.down_channels[0],
            kernel_size=3,
            padding=1
        )
        # initial time embedding projection
        self.time_projection = nn.Sequential(
            nn.Linear(in_features=self.time_embed_dim, out_features=self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(in_features=self.time_embed_dim, out_features=self.time_embed_dim)
        )
        # down blocks
        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels=self.down_channels[i],
                out_channels=self.down_channels[i+1],
                time_embed_dim=self.time_embed_dim,
                y_embed_dim=y_embed_dim,
                num_layers=self.num_down_blocks,
                down_sampling_factor=down_sampling_factor,
                down_sample=self.down_sampling[i],
                dropout_rate=self.dropout_rate,
                y_to_all=y_to_all
            ) for i in range(len(self.down_channels)-1)
        ])
        # middle blocks
        self.mid_blocks = nn.ModuleList([
            MiddleBlock(
                in_channels=self.mid_channels[i],
                out_channels=self.mid_channels[i + 1],
                time_embed_dim=self.time_embed_dim,
                y_embed_dim=y_embed_dim,
                num_layers=self.num_mid_blocks,
                dropout_rate=self.dropout_rate,
                y_to_all=y_to_all
            ) for i in range(len(self.mid_channels) - 1)
        ])
        # up blocks
        skip_channels = list(reversed(self.down_channels))
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=self.up_channels[i],
                out_channels=self.up_channels[i+1],
                skip_channels=skip_channels[i],
                time_embed_dim=self.time_embed_dim,
                y_embed_dim=y_embed_dim,
                num_layers=self.num_up_blocks,
                up_sampling_factor=down_sampling_factor,
                up_sampling=self.up_sampling[i],
                dropout_rate=self.dropout_rate,
                y_to_all=y_to_all
            ) for i in range(len(self.up_channels)-1)
        ])
        # final convolution layer
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=self.up_channels[-1]),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(in_channels=self.up_channels[-1], out_channels=self.in_channels, kernel_size=3, padding=1)
        )

    def initialize_weights(self) -> None:
        """Initializes model weights for training stability.

        Applies Kaiming normal initialization to convolutional and linear layers with
        Leaky ReLU nonlinearity (a=0.2), and zeros biases.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None, clip_embeddings: torch.Tensor = None) -> torch.Tensor:
        """Predicts noise given input, time step, and optional text conditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).
        t : torch.Tensor
            Time steps, shape (batch_size,).
        y : torch.Tensor, optional
            Text embeddings for conditioning, shape (batch_size, seq_len, y_embed_dim)
            or (batch_size, y_embed_dim) (default: None).
        clip_embeddings: torch.Tensor, optional
            used in the context of un-clip algorithm

        Returns
        -------
        output (torch.Tensor) - Predicted noise, same shape as input `x`.
        """
        if not self.where_y and y is not None:
            x = torch.cat(tensors=[x, y], dim=1)
        output = self.conv1(x)
        time_embed = GetEmbeddedTime(embed_dim=self.time_embed_dim)(time_steps=t)
        time_embed = self.time_projection(time_embed)
        #print("time embedded size (before concat): ", time_embed.size())
        #print("time embed:", time_embed.size())
        #print("clip embed (before)", clip_embeddings.size())
        if clip_embeddings is not None:
        #    if len(clip_embeddings.shape) == 3:  # [batch_size, seq_len, time_embed_dim]
        #        #clip_embeddings = clip_embeddings.mean(dim=1)
        #        time_embed = time_embed.unsqueeze(1)
        #        #print("please print it", time_embed.size())

                #print("clip imbed (after)", clip_embeddings.size())
            time_embed = time_embed + clip_embeddings
            #print("time embedded size (after concat): ", time_embed.size())

        skip_connections = []
        for i, down in enumerate(self.down_blocks):
            skip_connections.append(output)
            output = down(x=output, embed_time=time_embed, y=y)
        for i, mid in enumerate(self.mid_blocks):
            output = mid(x=output, embed_time=time_embed, y=y)
        for i, up in enumerate(self.up_blocks):
            skip_connection = skip_connections.pop()
            output = up(x=output, skip_connection=skip_connection, embed_time=time_embed, y=y)

        output = self.conv2(output)
        return output

###==================================================================================================================###

class DownBlock(nn.Module):
    """Downsampling block for NoisePredictor’s encoder.

    Applies convolutional layers with residual connections, time embeddings, and optional
    text-conditioned attention, followed by downsampling if enabled.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    time_embed_dim : int
        Dimensionality of time embeddings.
    y_embed_dim : int
        Dimensionality of text embeddings.
    num_layers : int
        Number of convolutional layer pairs (Conv3).
    down_sampling_factor : int
        Factor for spatial downsampling.
    down_sample : bool
        If True, apply downsampling; if False, use identity (no downsampling).
    dropout_rate : float
        Dropout rate for Conv3 and attention layers.
    y_to_all : bool
        If True, apply text-conditioned attention to all layers; if False, only first layer.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int ,
            time_embed_dim: int,
            y_embed_dim: int,
            num_layers: int,
            down_sampling_factor: int,
            down_sample: bool,
            dropout_rate: float,
            y_to_all: bool
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.y_to_all = y_to_all
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i==0 else out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.time_embedding = nn.ModuleList([
            TimeEmbedding(
                output_dim=out_channels,
                embed_dim=time_embed_dim
            ) for _ in range(self.num_layers)
        ])
        self.attention = nn.ModuleList([
            Attention(
                in_channels=out_channels,
                y_embed_dim=y_embed_dim,
                num_groups=8,
                num_heads=4,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.down_sampling = DownSampling(
            in_channels=out_channels,
            out_channels=out_channels,
            down_sampling_factor=down_sampling_factor,
            conv_block=True,
            max_pool=True
        ) if down_sample else nn.Identity()
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers)

        ])

    def forward(self, x: torch.Tensor, embed_time: torch.Tensor, y: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Processes input through convolutions, time embeddings, attention, and downsampling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).
        embed_time : torch.Tensor
            Time embeddings, shape (batch_size, time_embed_dim).
        y : torch.Tensor, optional
            Text embeddings, shape (batch_size, seq_len, y_embed_dim) or
            (batch_size, y_embed_dim) (default: None).
        key_padding_mask : torch.Tensor, optional
            Boolean mask, shape (batch_size, seq_len) if `y` is None, or (batch_size, seq_len_y) if `y` is provided,
            where `True` indicates positions to mask out (default: None).

        Returns
        -------
        output (torch.Tensor) - Output tensor, shape (batch_size, out_channels, height/down_sampling_factor, width/down_sampling_factor) if downsampling; otherwise, same height/width as input.
        """
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = output + self.time_embedding[i](embed_time)[:, :, None, None]
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)

            if not self.y_to_all and i == 0:
                out_attn = self.attention[i](output, y)
                output = output + out_attn
            elif self.y_to_all:
                out_attn = self.attention[i](output, y)
                output = output + out_attn

        output = self.down_sampling(output)
        return output

###==================================================================================================================###

class MiddleBlock(nn.Module):
    """Bottleneck block for NoisePredictor’s middle layers.

    Applies convolutional layers with residual connections, time embeddings, and optional
    text-conditioned attention, preserving spatial dimensions.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    time_embed_dim : int
        Dimensionality of time embeddings.
    y_embed_dim : int
        Dimensionality of text embeddings.
    num_layers : int
        Number of convolutional layer pairs (Conv3).
    dropout_rate : float
        Dropout rate for Conv3 and attention layers.
    y_to_all : bool
        If True, apply text-conditioned attention to all layers; if False, only first layer
        (default: False).
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_embed_dim: int,
            y_embed_dim: int,
            num_layers: int,
            dropout_rate: float,
            y_to_all: bool
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.y_to_all = y_to_all
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers+1)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers+1)
        ])
        self.time_embedding = nn.ModuleList([
            TimeEmbedding(
                output_dim=out_channels,
                embed_dim=time_embed_dim
            ) for _ in range(self.num_layers+1)
        ])
        self.attention = nn.ModuleList([
            Attention(
                in_channels=out_channels,
                y_embed_dim=y_embed_dim,
                num_groups=8,
                num_heads=4,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers + 1)
        ])
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers+1)
        ])

    def forward(self, x: torch.Tensor, embed_time: torch.Tensor, y: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Processes input through convolutions, time embeddings, and attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).
        embed_time : torch.Tensor
            Time embeddings, shape (batch_size, time_embed_dim).
        y : torch.Tensor, optional
            Text embeddings, shape (batch_size, seq_len, y_embed_dim) or
            (batch_size, y_embed_dim) (default: None).
        key_padding_mask : torch.Tensor, optional
            Boolean mask, shape (batch_size, seq_len) if `y` is None, or (batch_size, seq_len_y) if `y` is provided,
            where `True` indicates positions to mask out (default: None).

        Returns
        -------
        output (torch.Tensor) - Output tensor, shape (batch_size, out_channels, height, width).
        """
        output = x
        resnet_input = output
        output = self.conv1[0](output)
        output = output + self.time_embedding[0](embed_time)[:, :, None, None]
        output = self.conv2[0](output)
        output = output + self.resnet[0](resnet_input)

        for i in range(self.num_layers):
            if not self.y_to_all and i == 0:
                out_attn = self.attention[i](output, y)
                output = output + out_attn
            elif self.y_to_all:
                out_attn = self.attention[i](output, y)
                output = output + out_attn
            resnet_input = output
            output = self.conv1[i + 1](output)
            output = output + self.time_embedding[i + 1](embed_time)[:, :, None, None]
            output = self.conv2[i + 1](output)
            output = output + self.resnet[i+1](resnet_input)
        return output

###==================================================================================================================###

class UpBlock(nn.Module):
    """Upsampling block for NoisePredictor’s decoder.

    Applies upsampling (if enabled), concatenates skip connections, and processes through
    convolutional layers with residual connections, time embeddings, and optional
    text-conditioned attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels (before upsampling).
    out_channels : int
        Number of output channels.
    skip_channels : int
        Number of channels from skip connection.
    time_embed_dim : int
        Dimensionality of time embeddings.
    y_embed_dim : int
        Dimensionality of text embeddings.
    num_layers : int
        Number of convolutional layer pairs (Conv3).
    up_sampling_factor : int
        Factor for spatial upsampling.
    up_sampling : bool
        If True, apply upsampling; if False, use identity (no upsampling).
    dropout_rate : float
        Dropout rate for Conv3 and attention layers.
    y_to_all : bool
        If True, apply text-conditioned attention to all layers; if False, only first layer
        (default: False).
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            skip_channels: int,
            time_embed_dim: int,
            y_embed_dim: int,
            num_layers: int,
            up_sampling_factor: int,
            up_sampling: bool,
            dropout_rate: float,
            y_to_all: bool
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.y_to_all = y_to_all
        effective_in_channels = in_channels // 2 + skip_channels
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=effective_in_channels  if i == 0 else out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.time_embedding = nn.ModuleList([
            TimeEmbedding(
                output_dim=out_channels,
                embed_dim=time_embed_dim
            ) for _ in range(self.num_layers)
        ])
        self.attention = nn.ModuleList([
            Attention(
                in_channels=out_channels,
                y_embed_dim=y_embed_dim,
                num_groups=8,
                num_heads=4,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.up_sampling_ = UpSampling(
            in_channels=in_channels,
            out_channels=in_channels,
            up_sampling_factor=up_sampling_factor,
            conv_block=True,
            up_sampling=True
        ) if up_sampling else nn.Identity()
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=effective_in_channels  if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers)

        ])

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor, embed_time: torch.Tensor, y: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Processes input through upsampling, skip connection, convolutions, time embeddings, and attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).
        skip_connection : torch.Tensor
            Skip connection tensor, shape (batch_size, skip_channels,
            height*up_sampling_factor, width*up_sampling_factor).
        embed_time : torch.Tensor
            Time embeddings, shape (batch_size, time_embed_dim).
        y : torch.Tensor, optional
            Text embeddings, shape (batch_size, seq_len, y_embed_dim) or
            (batch_size, y_embed_dim) (default: None).
        key_padding_mask : torch.Tensor, optional
            Boolean mask, shape (batch_size, seq_len) if `y` is None, or (batch_size, seq_len_y) if `y` is provided,
            where `True` indicates positions to mask out (default: None).

        Returns
        -------
        output (torch.Tensor) - Output tensor, shape (batch_size, out_channels, height*up_sampling_factor, width*up_sampling_factor) if upsampling; otherwise, same height/width as input (after skip connection).
        """
        x = self.up_sampling_(x)
        x = torch.cat(tensors=[x, skip_connection], dim=1)
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = output + self.time_embedding[i](embed_time)[:, :, None, None]
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)

            if not self.y_to_all and i == 0:
                out_attn = self.attention[i](output, y)
                output = output + out_attn
            elif self.y_to_all:
                out_attn = self.attention[i](output, y)
                output = output + out_attn

        return output

###==================================================================================================================###

class Conv3(nn.Module):
    """Convolutional layer with optional group normalization, SiLU activation, and dropout.

    Used in DownBlock, MiddleBlock, and UpBlock for feature extraction in NoisePredictor.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_groups : int, optional
        Number of groups for group normalization (default: 8).
    kernel_size : int, optional
        Convolutional kernel size (default: 3).
    norm : bool, optional
        If True, apply group normalization (default: True).
    activation : bool, optional
        If True, apply SiLU activation (default: True).
    dropout_rate : float, optional
        Dropout rate (default: 0.2).
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_groups: int = 8,
            kernel_size: int = 3,
            norm: bool = True,
            activation: bool = True,
            dropout_rate: float = 0.2
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels) if norm else nn.Identity()
        self.activation = nn.SiLU() if activation else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Processes input through convolution, normalization, activation, and dropout.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        batch (torch.Tensor) - Output tensor, shape (batch_size, out_channels, height, width).
        """
        batch = self.conv(batch)
        batch = self.group_norm(batch)
        batch = self.activation(batch)
        batch = self.dropout(batch)
        return batch

###==================================================================================================================###

class TimeEmbedding(nn.Module):
    """Time embedding projection for conditioning NoisePredictor layers.

    Projects time embeddings to match the channel dimension of convolutional outputs.

    Parameters
    ----------
    output_dim : int
        Output channel dimension (matches convolutional channels).
    embed_dim : int
        Input time embedding dimension.
    """
    def __init__(self, output_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=embed_dim, out_features=output_dim)
        )
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Projects time embeddings to output dimension.

        Parameters
        ----------
        batch : torch.Tensor
            Time embeddings, shape (batch_size, embed_dim).

        Returns
        -------
        torch.Tensor
            Projected embeddings, shape (batch_size, output_dim).
        """
        return self.embedding(batch)

###==================================================================================================================###

class GetEmbeddedTime(nn.Module):
    """Generates sinusoidal time embeddings for NoisePredictor.

    Creates positional encodings for time steps using sine and cosine functions, following
    the transformer embedding approach.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the time embeddings (must be even).
    """
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "The embedding dimension must be divisible by two"
        self.embed_dim = embed_dim

    def forward(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Generates sinusoidal embeddings for time steps.

        Parameters
        ----------
        time_steps : torch.Tensor
            Time steps, shape (batch_size,).

        Returns
        -------
        embed_time (torch.Tensor) - Sinusoidal embeddings, shape (batch_size, embed_dim).
        """
        i = torch.arange(start=0, end=self.embed_dim // 2, dtype=torch.float32, device=time_steps.device)
        factor = 10000 ** (2 * i / self.embed_dim)
        embed_time = time_steps[:, None] / factor
        embed_time = torch.cat(tensors=[torch.sin(embed_time), torch.cos(embed_time)], dim=-1)
        return embed_time

###==================================================================================================================###


class DownSampling(nn.Module):
    """Downsampling module for NoisePredictor’s DownBlock.

    Combines convolutional downsampling and max pooling (if enabled), concatenating
    outputs to preserve feature information.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    down_sampling_factor : int
        Factor for spatial downsampling.
    conv_block : bool, optional
        If True, include convolutional path (default: True).
    max_pool : bool, optional
        If True, include max pooling path (default: True).
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            down_sampling_factor: int,
            conv_block: bool = True,
            max_pool: bool = True
    ) -> None:
        super().__init__()
        self.conv_block = conv_block
        self.max_pool = max_pool
        self.down_sampling_factor = down_sampling_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2 if max_pool else out_channels,
                      kernel_size=3, stride=down_sampling_factor, padding=1)
        ) if conv_block else nn.Identity()
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=down_sampling_factor, stride=down_sampling_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2 if conv_block else out_channels,
                      kernel_size=1, stride=1, padding=0)
        ) if max_pool else nn.Identity()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Downsamples input using convolutional and/or pooling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        batch (torch.Tensor) - Downsampled tensor, shape (batch_size, out_channels, height/down_sampling_factor, width/down_sampling_factor).
        """
        if not self.conv_block:
            return self.pool(batch)
        if not self.max_pool:
            return self.conv(batch)
        return torch.cat(tensors=[self.conv(batch), self.pool(batch)], dim=1)

###==================================================================================================================###

class UpSampling(nn.Module):
    """Upsampling module for NoisePredictor’s UpBlock.

    Combines transposed convolution and nearest-neighbor upsampling (if enabled),
    concatenating outputs to preserve feature information, with interpolation to align
    spatial dimensions if needed.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    up_sampling_factor : int
        Factor for spatial upsampling.
    conv_block : bool, optional
        If True, include transposed convolutional path (default: True).
    up_sampling : bool, optional
        If True, include nearest-neighbor upsampling path (default: True).
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            up_sampling_factor: int,
            conv_block: bool = True,
            up_sampling: bool = True
    ) -> None:
        super().__init__()
        self.conv_block = conv_block
        self.up_sampling = up_sampling
        self.up_sampling_factor = up_sampling_factor
        half_out_channels = out_channels // 2
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=half_out_channels if up_sampling else out_channels,
                kernel_size=3,
                stride=up_sampling_factor,
                padding=1,
                output_padding=up_sampling_factor - 1
            ),
            nn.Conv2d(
                in_channels=half_out_channels if up_sampling else out_channels,
                out_channels=half_out_channels if up_sampling else out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ) if conv_block else nn.Identity()

        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=up_sampling_factor, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=half_out_channels if conv_block else out_channels,
                      kernel_size=1, stride=1, padding=0)
        ) if up_sampling else nn.Identity()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Upsamples input using convolutional and/or upsampling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        batch (torch.Tensor) - Upsampled tensor, shape (batch_size, out_channels, height*up_sampling_factor, width*up_sampling_factor).

        **Notes**

        - Interpolation is applied if the spatial dimensions of the convolutional and
          upsampling paths differ, using nearest-neighbor mode.
        """
        if not self.conv_block:
            return self.up_sample(batch)
        if not self.up_sampling:
            return self.conv(batch)
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

class Metrics:
    """Computes image quality metrics for evaluating diffusion models.

    Supports Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), Structural
    Similarity Index (SSIM), Fréchet Inception Distance (FID), and Learned Perceptual
    Image Patch Similarity (LPIPS) for comparing generated and ground truth images.

    Parameters
    ----------
    device : str, optional
        Device for computation (e.g., 'cuda', 'cpu') (default: 'cuda').
    fid : bool, optional
        If True, compute FID score (default: True).
    metrics : bool, optional
        If True, compute MSE, PSNR, and SSIM (default: False).
    lpips : bool, optional
        If True, compute LPIPS using VGG backbone (default: False).
    """

    def __init__(
            self,
            device: str = "cuda",
            fid: bool = True,
            metrics: bool = False,
            lpips_: bool = False
    ) -> None:
        self.device = device
        self.fid = fid
        self.metrics = metrics
        self.lpips = lpips_
        self.lpips_model = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg',
            normalize=True  # This handles [0,1] -> [-1,1] conversion
        ).to(device) if self.lpips else None
        self.temp_dir_real = "temp_real"
        self.temp_dir_fake = "temp_fake"

    def compute_fid(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """Computes the Fréchet Inception Distance (FID) between real and generated images.

        Saves images to temporary directories and uses Inception V3 to compute FID,
        cleaning up directories afterward.

        Parameters
        ----------
        real_images : torch.Tensor
            Real images, shape (batch_size, channels, height, width), in [-1, 1].
        fake_images : torch.Tensor
            Generated images, same shape, in [-1, 1].

        Returns
        -------
        fid (float) - FID score, or `float('inf')` if computation fails.

        **Notes**

        - Images are normalized to [0, 1] and saved as PNG files for FID computation.
        - Uses Inception V3 with 2048-dimensional features (`dims=2048`).
        """
        if real_images.shape != fake_images.shape:
            raise ValueError(f"Shape mismatch: real_images {real_images.shape}, fake_images {fake_images.shape}")

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

    def compute_metrics(self, x: torch.Tensor, x_hat: torch.Tensor) -> Tuple[float, float, float]:
        """Computes MSE, PSNR, and SSIM for evaluating image quality.

        Parameters
        ----------
        x : torch.Tensor
            Ground truth images, shape (batch_size, channels, height, width).
        x_hat : torch.Tensor
            Generated images, same shape as `x`.

        Returns
        -------
        mse : float
            Mean squared error.
        psnr : float
            Peak signal-to-noise ratio.
        ssim : float
            Structural similarity index (mean over batch).
        """
        if x.shape != x_hat.shape:
            raise ValueError(f"Shape mismatch: x {x.shape}, x_hat {x_hat.shape}")

        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse)
        c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2  # Adjusted for [-1, 1] range
        eps = 1e-8
        mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(x_hat, kernel_size=3, stride=1, padding=1)
        mu_xy = mu_x * mu_y
        sigma_x_sq = F.avg_pool2d(x.pow(2), kernel_size=3, stride=1, padding=1) - mu_x.pow(2)
        sigma_y_sq = F.avg_pool2d(x_hat.pow(2), kernel_size=3, stride=1, padding=1) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(x * x_hat, kernel_size=3, stride=1, padding=1) - mu_xy
        ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x_sq + sigma_y_sq + c2) + eps
        )

        return mse.item(), psnr.item(), ssim.mean().item()

    def compute_lpips(self, x: torch.Tensor, x_hat: torch.Tensor) -> float:
        """Computes LPIPS using a pre-trained VGG network.

        Parameters
        ----------
        x : torch.Tensor
            Ground truth images, shape (batch_size, channels, height, width), in [-1, 1].
        x_hat : torch.Tensor
            Generated images, same shape as `x`.

        Returns
        -------
        lpips (float) - Mean LPIPS score over the batch.
        """
        if self.lpips_model is None:
            raise RuntimeError("LPIPS model not initialized; set lpips=True in __init__")
        if x.shape != x_hat.shape:
            raise ValueError(f"Shape mismatch: x {x.shape}, x_hat {x_hat.shape}")

        # Normalize inputs to [0, 1] range
        x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        x_hat = (x_hat + 1) / 2
        x = x.clamp(0, 1)  # Ensure values are in [0, 1]
        x_hat = x_hat.clamp(0, 1)

        x = x.to(self.device)
        x_hat = x_hat.to(self.device)

        # Convert grayscale to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # Repeat grayscale channel 3 times
        if x_hat.shape[1] == 1:
            x_hat = x_hat.repeat(1, 3, 1, 1)

        return self.lpips_model(x, x_hat).mean().item()

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> Tuple[float, float, float, float, float]:
        """Computes specified metrics for ground truth and generated images.

        Parameters
        ----------
        x : torch.Tensor
            Ground truth images, shape (batch_size, channels, height, width), in [-1, 1].
        x_hat : torch.Tensor
            Generated images, same shape as `x`.

        Returns
        -------
        fid : float, or `float('inf')` if not computed
            Mean FID score.
        mse : float, or None if not computed
            Mean MSE
        psnr : float, or None if not computed
             Mean PSNR
        ssim : float, or None if not computed
            Mean SSIM
        lpips_score :  float, or None if not computed
            Mean LPIPS score
        """
        fid = float('inf')
        mse, psnr, ssim = None, None, None
        lpips_score = None

        if self.metrics:
            mse, psnr, ssim = self.compute_metrics(x, x_hat)
        if self.fid:
            fid = self.compute_fid(x, x_hat)
        if self.lpips:
            lpips_score = self.compute_lpips(x, x_hat)

        return fid, mse, psnr, ssim, lpips_score



"""
# Ensure GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(5, 3, 50, 50).to(device)
t = torch.rand(5).to(device)

np_ = NoisePredictor(
    in_channels=3,
    down_channels=[16, 32],
    mid_channels=[32, 32],
    up_channels=[32, 16],
    down_sampling=[True, True],
    time_embed_dim=20,
    y_embed_dim=20,
    num_down_blocks=2,
    num_mid_blocks=2,
    num_up_blocks=2,
    down_sampling_factor=2
).to(device)

# Uncompiled
times = []
for i in range(50):
    st = time.time()
    np_(x, t)
    torch.cuda.synchronize()  # Ensure GPU operations complete
    ft = time.time()
    times.append(ft - st)
    print(f"Uncompiled iter: {i+1} finished in {ft - st:.4f} s")
print(f"Uncompiled average: {sum(times)/len(times):.4f} s")

# Compiled with warm-up
cm = torch.compile(np_, mode="reduce-overhead")
for _ in range(5):  # Warm-up
    cm(x, t)
torch.cuda.synchronize()

times = []
for i in range(50):
    st = time.time()
    cm(x, t)
    torch.cuda.synchronize()
    ft = time.time()
    times.append(ft - st)
    print(f"Compiled iter: {i+1} finished in {ft - st:.4f} s")
print(f"Compiled average: {sum(times)/len(times):.4f} s")


import transformers
import time

# Set up device and inputs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(5, 3, 50, 50).to(device)  # Image input (not used in TextEncoder)
t = torch.rand(5).to(device)  # Time embedding (not used)
y = ["do", "not to do", "correctly", "wrong", "it is true"]  # Text inputs

# Initialize tokenizer and TextEncoder
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
te = TextEncoder(
    use_pretrained_model=True,
    model_name="bert-base-uncased",
    num_layers=2,
    vocabulary_size=30522,
    input_dimension=768,
    output_dimension=768,
    dropout_rate=0.2,
    num_heads=6,
    context_length=77
).to(device)  # Use FP16 for better performance

# Tokenize inputs
y_list = y  # Already a list of strings
y_encoded = tokenizer(
    y_list,
    padding="max_length",
    truncation=True,
    max_length=77,
    return_tensors="pt"
).to(device)  # Convert to FP16
input_ids = y_encoded["input_ids"]  # Shape: [5, 77]
attention_mask = y_encoded["attention_mask"]  # Shape: [5, 77]

# Uncompiled TextEncoder
times = []
for i in range(50):
    st = time.time()
    with torch.no_grad():  # Disable gradients
        y_encoded = te(input_ids, attention_mask)
    torch.cuda.synchronize()
    ft = time.time()
    times.append(ft - st)
    print(f"Uncompiled iter: {i+1} finished in {ft - st:.4f} s")
print(f"Uncompiled average: {sum(times)/len(times):.4f} s")

# Compiled TextEncoder with warm-up
cm = torch.compile(te, mode="reduce-overhead")
for _ in range(5):  # Warm-up runs
    with torch.no_grad():
        cm(input_ids, attention_mask)
torch.cuda.synchronize()

times = []
for i in range(50):
    st = time.time()
    with torch.no_grad():  # Disable gradients
        y_encoded = cm(input_ids, attention_mask)
    torch.cuda.synchronize()
    ft = time.time()
    times.append(ft - st)
    print(f"Compiled iter: {i+1} finished in {ft - st:.4f} s")
print(f"Compiled average: {sum(times)/len(times):.4f} s")

# Verify output shape
print(f"Output shape: {y_encoded.shape}")
"""









