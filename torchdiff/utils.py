"""
**Utilities for text encoding, score prediction, and evaluation in diffusion models**

This module provides core components for building diffusion model pipelines, including
text encoding (used as a conditional model), U-Net-based score prediction (ScoreNet),
custom loss functions for training, and image quality evaluation. These utilities support
various diffusion model architectures, such as DDPM, DDIM, LDM, and SDE, and are designed
for standalone use in model training and sampling.

**Primary Components**

- **TextEncoder**: Encodes text prompts into embeddings using a pre-trained BERT model or a custom transformer.
- **ScoreNet**: Memory-efficient U-Net-like architecture for predicting noise or scores in diffusion models, supporting time and text conditioning.
- **Loss Functions**:
    - `mse_loss`: Standard mean squared error loss.
    - `snr_capped_loss`: SNR-weighted noise prediction loss with capped weighting, useful for VP/VE training.
    - `ve_sigma_weighted_score_loss`: Sigma-weighted score matching loss for VE-SDEs.
- **Metrics**: Computes image quality metrics (MSE, PSNR, SSIM, FID, LPIPS) for evaluating generated images.

**Notes**

- The primary components are intended to be imported directly for use in diffusion model workflows.
- Additional supporting classes and functions in this module provide internal functionality for the primary components.

------------------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_fid import fid_score
from transformers import BertModel
import os
import math
import shutil
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
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
        Size of the vocabulary for the custom transformer’s embedding layer
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
        Scaling factor for the feedforward layer’s hidden dimension in the custom
        transformer (default: 4).
    epsilon : float, optional
        Epsilon for layer normalization in the custom transformer (default: 1e-5).
    use_learned_pos : bool, optional
        If True, in the transformer structure uses learnable positional embeddings instead of sinusoidal encodings
        (default: False).

    **Notes**

    - When `use_pretrained_model` is True, the BERT model’s parameters are frozen
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
          using the custom transformer, or follow BERT’s convention for pre-trained
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

class DiffusionNetwork(nn.Module):
    """Memory-efficient U-Net architecture for diffusion models supporting time and conditional embeddings"""
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
            y_to_all: bool = False,
            cont_time: bool = True,
            use_flash_attention: bool = True,
            grad_check: bool = False
    ) -> None:
        """Initialize the ScoreNet U-Net with configurable down, middle, and up blocks, time embeddings, and optional attention.

        Args:
            in_channels: Number of input channels.
            down_channels: List of channels for downsampling stages.
            mid_channels: List of channels for middle blocks.
            up_channels: List of channels for upsampling stages.
            down_sampling: Boolean flags indicating whether to downsample at each down block.
            time_embed_dim: Dimensionality of the time embedding.
            y_embed_dim: Dimensionality of the conditional embedding.
            num_down_blocks: Number of residual layers per down block.
            num_mid_blocks: Number of residual layers per middle block.
            num_up_blocks: Number of residual layers per up block.
            dropout_rate: Dropout probability.
            down_sampling_factor: Stride factor for downsampling/upsampling.
            y_to_all: If True, applies conditional embeddings to all attention layers.
            cont_time: Whether to use continuous time embeddings.
            use_flash_attention: Whether to use flash attention for cross-attention layers.
            grad_check: Whether to use gradient checkpointing.
        """
        super().__init__()
        self.cont_time = cont_time
        self.grad_check = grad_check
        assert len(down_channels) - 1 == len(down_sampling), \
            f"down_sampling length must be len(down_channels)-1, got {len(down_sampling)} vs {len(down_channels) - 1}"
        assert len(up_channels) - 1 <= len(down_channels) - 1, \
            f"Cannot have more up blocks than down blocks"

        self.conv_in = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.encoder = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            self.encoder.append(nn.ModuleDict({
                'block': ResBlock(
                    in_channels=down_channels[i],
                    out_channels=down_channels[i + 1],
                    time_channels=time_embed_dim,
                    context_channels=y_embed_dim,
                    num_layers=num_down_blocks,
                    dropout=dropout_rate,
                    use_attention=(i == 0 or y_to_all),
                    use_flash=use_flash_attention
                ),
                'downsample': nn.Conv2d(down_channels[i + 1], down_channels[i + 1], 3,
                                        stride=down_sampling_factor, padding=1) if down_sampling[i] else nn.Identity()
            }))

        self.middle = nn.ModuleList()
        for i in range(len(mid_channels) - 1):
            self.middle.append(
                ResBlock(
                    in_channels=mid_channels[i],
                    out_channels=mid_channels[i + 1],
                    time_channels=time_embed_dim,
                    context_channels=y_embed_dim,
                    num_layers=num_mid_blocks,
                    dropout=dropout_rate,
                    use_attention=True,
                    use_flash=use_flash_attention
                )
            )
        num_decoder_stages = len(up_channels) - 1
        up_sampling_ops = list(reversed(down_sampling[-num_decoder_stages:]))
        encoder_output_channels = list(reversed(down_channels[1:]))[:num_decoder_stages]
        self.decoder = nn.ModuleList()
        for i in range(num_decoder_stages):
            self.decoder.append(nn.ModuleDict({
                'upsample': nn.ConvTranspose2d(up_channels[i], up_channels[i],
                                               down_sampling_factor, stride=down_sampling_factor) if up_sampling_ops[
                    i] else nn.Identity(),
                'block': ResBlock(
                    in_channels=up_channels[i] + encoder_output_channels[i],
                    out_channels=up_channels[i + 1],
                    time_channels=time_embed_dim,
                    context_channels=y_embed_dim,
                    num_layers=num_up_blocks,
                    dropout=dropout_rate,
                    use_attention=(i == 0 or y_to_all),
                    use_flash=use_flash_attention
                )
            }))
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, up_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(up_channels[-1], in_channels, 3, padding=1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights of Conv2d and Linear layers with Kaiming initialization and zero biases.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            clip_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the U-Net with time and optional conditional embeddings.

        Args:
            x: Input tensor of shape [B, C, H, W].
            t: Tensor of timesteps [B] or [B, 1].
            y: Optional context embeddings [B, D] or [B, L, D].
            clip_embeddings: Optional CLIP embeddings [B, D].

        Returns:
            Output tensor of shape [B, in_channels, H, W].
        """
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features, self.cont_time)
        t_emb = self.time_mlp(t_emb)
        if clip_embeddings is not None:
            t_emb = t_emb + clip_embeddings
        h = self.conv_in(x)
        encoder_features = []
        for stage in self.encoder:
            h = self._apply_block(stage['block'], h, t_emb, y)
            encoder_features.append(h)
            h = stage['downsample'](h)
        for block in self.middle:
            h = self._apply_block(block, h, t_emb, y)
        num_decoder_stages = len(self.decoder)
        skips_for_decoder = list(reversed(encoder_features[-num_decoder_stages:]))
        for stage, skip in zip(self.decoder, skips_for_decoder):
            h = stage['upsample'](h)
            h = torch.cat([h, skip], dim=1)
            h = self._apply_block(stage['block'], h, t_emb, y)
        return self.conv_out(h)

    def _apply_block(self, block, x, t_emb, y):
        """Apply a residual block with optional gradient checkpointing.

        Args:
            block: The ResBlock module to apply.
            x: Input tensor.
            t_emb: Time embedding tensor.
            y: Optional conditional embedding.

        Returns:
            Output tensor after applying the block.
        """
        if self.grad_check and self.training:
            return torch.utils.checkpoint.checkpoint(
                block, x, t_emb, y, use_reentrant=False
            )
        return block(x, t_emb, y)


class ResBlock(nn.Module):
    """Efficient residual block with optional cross-attention for U-Net."""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_channels: int,
            context_channels: int,
            num_layers: int = 2,
            dropout: float = 0.1,
            use_attention: bool = False,
            use_flash: bool = True
    ):
        """Initialize a ResBlock with optional attention and multiple residual layers.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_channels: Dimensionality of time embedding.
            context_channels: Dimensionality of conditional embedding.
            num_layers: Number of residual layers in the block.
            dropout: Dropout probability.
            use_attention: Whether to include a cross-attention layer.
            use_flash: Whether to use flash attention if available.
        """
        super().__init__()

        self.num_layers = num_layers
        self.use_attention = use_attention and context_channels > 0
        self.res_layers = nn.ModuleList()
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            self.res_layers.append(
                nn.ModuleDict({
                    'norm1': nn.GroupNorm(8, ch_in),
                    'conv1': nn.Conv2d(ch_in, out_channels, 3, padding=1),
                    'time_emb': nn.Linear(time_channels, out_channels),
                    'norm2': nn.GroupNorm(8, out_channels),
                    'conv2': nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    'dropout': nn.Dropout(dropout),
                    'skip': nn.Conv2d(ch_in, out_channels, 1) if ch_in != out_channels else nn.Identity()
                })
            )
        if self.use_attention:
            self.attention = CrossAttention(
                out_channels, context_channels,
                num_heads=4, dropout=dropout, use_flash=use_flash
            )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: Optional[torch.Tensor] = None):
        """Forward pass through the residual block.

        Args:
            x: Input tensor of shape [B, C, H, W].
            t_emb: Time embedding tensor of shape [B, time_channels].
            context: Optional conditional embeddings for cross-attention.

        Returns:
            Output tensor after residual layers (and optional attention).
        """
        h = x
        for i, layer in enumerate(self.res_layers):
            res = h
            h = layer['norm1'](h)
            h = F.silu(h)
            h = layer['conv1'](h)
            h = h + layer['time_emb'](F.silu(t_emb))[:, :, None, None]
            h = layer['norm2'](h)
            h = F.silu(h)
            h = layer['dropout'](h)
            h = layer['conv2'](h)
            h = h + layer['skip'](res)
            if i == 0 and self.use_attention and context is not None:
                h = h + self.attention(h, context)
        return h


class CrossAttention(nn.Module):
    """Cross-attention module with optional flash attention."""
    def __init__(
            self,
            channels: int,
            context_dim: int,
            num_heads: int = 4,
            dropout: float = 0.0,
            use_flash: bool = True
    ):
        """Initialize cross-attention with query, key, value projections and optional flash attention.

        Args:
            channels: Number of input channels.
            context_dim: Dimensionality of the context embeddings.
            num_heads: Number of attention heads.
            dropout: Dropout probability for attention output.
            use_flash: Whether to use flash attention if available.
        """
        super().__init__()

        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_kv = nn.Linear(context_dim, channels * 2, bias=False)
        self.proj_out = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute cross-attention output.

        Args:
            x: Input feature map tensor [B, C, H, W].
            context: Context embeddings [B, D] or [B, L, D].

        Returns:
            Tensor of shape [B, C, H, W] after applying attention.
        """
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        if context.dim() == 2:
            context = context.unsqueeze(1)  # [B, 1, D]
        q = self.to_q(x_flat)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, H * W, C)
        out = self.proj_out(out)
        out = self.dropout(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


def get_timestep_embedding(timesteps: torch.Tensor, dim: int, continuous: bool = True, scale = 1000.0) -> torch.Tensor:
    """Compute sinusoidal timestep embeddings for continuous or discrete timesteps.

    Args:
        timesteps: Tensor of timesteps [B] or scalar.
        dim: Dimensionality of the embedding vector.
        continuous: If True, scales timesteps by 1000 to emulate discrete DDPM timesteps.

    Returns:
        Tensor of shape [B, dim] containing sinusoidal embeddings.
    """
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    elif timesteps.dim() == 2:
        timesteps = timesteps.squeeze(-1)
    if continuous:
        timesteps = timesteps * scale
    half_dim = dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


###==================================================================================================================###


def mse_loss(pred: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
    """
    Standard mean squared error (MSE) loss.

    Computes the element-wise squared difference between `pred` and `target`
    and returns the mean across all elements.

    Args:
        pred: Predicted tensor, shape [B, ...].
        target: Target tensor, same shape as `pred`.
        *args: Placeholder for optional unused arguments for API compatibility.

    Returns:
        Scalar tensor representing mean squared error.
    """
    return ((pred - target) ** 2).mean()


def snr_capped_loss(pred_noise: torch.Tensor, target_noise: torch.Tensor, variance: torch.Tensor, gamma: float = 5.0, *args) -> torch.Tensor:
    """
    Signal-to-noise-ratio (SNR) capped noise prediction loss for diffusion models.

    This implements a weighted MSE where the weight is the SNR of the timestep,
    capped at a maximum value `gamma`. Typically used in VP/VE noise prediction.

    Args:
        pred_noise: Predicted noise tensor, same shape as target_noise.
        target_noise: True noise tensor.
        variance: Variance (sigma^2) corresponding to the timestep t, shape broadcastable to pred_noise.
        gamma: Maximum SNR weight (default 5.0).
        *args: Placeholder for optional unused arguments for API compatibility.

    Returns:
        Scalar tensor representing the SNR-weighted mean squared error.
    """
    snr = (1 - variance) / variance.clamp(min=1e-8)
    weight = torch.minimum(snr, torch.tensor(gamma, device=snr.device))
    while weight.dim() < target_noise.dim():
        weight = weight.unsqueeze(-1)
    return ((pred_noise - target_noise) ** 2 * weight).mean()


def ve_sigma_weighted_score_loss(pred_score: torch.Tensor, target_score: torch.Tensor, sigma: torch.Tensor, *args) -> torch.Tensor:
    """
    VE-SDE sigma-weighted score matching loss.

    Implements the recommended loss for Variance Exploding SDEs:
        E[ || sigma(t) * s_theta(x_t, t) + epsilon ||^2 ]
    where epsilon is the true noise used to perturb x_0.

    Args:
        pred_score: Model-predicted score tensor (∇_x log p(x_t)), shape [B, ...].
        target_score: Target score, typically -epsilon / sigma(t).
        sigma: Standard deviation (σ(t)) at the corresponding timesteps, shape broadcastable to pred_score.
        *args: Placeholder for optional unused arguments for API compatibility.

    Returns:
        Scalar tensor representing the sigma-weighted score matching loss.
    """
    while sigma.dim() < pred_score.dim():
        sigma = sigma.unsqueeze(-1)
    eps = -target_score * sigma
    return ((sigma * pred_score + eps) ** 2).mean()


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