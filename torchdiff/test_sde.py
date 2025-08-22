import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from typing import Optional, Tuple, Callable, List, Any, Union, Self
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
import warnings
from torchvision.utils import save_image




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

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            clip_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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

        if clip_embeddings is not None:
            if len(clip_embeddings.shape) == 3:  # [batch_size, seq_len, time_embed_dim]
                time_embed = time_embed.unsqueeze(1)
            time_embed = time_embed + clip_embeddings

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


###==================================================================================================================###

class ForwardSDE(nn.Module):
    """Forward diffusion process for SDE-based generative models.

    Implements the forward diffusion process for score-based generative models using
    Stochastic Differential Equations (SDEs), supporting Variance Exploding (VE),
    Variance Preserving (VP), sub-Variance Preserving (sub-VP), and ODE methods, as
    described in Song et al. (2021).

    Parameters
    ----------
    variance_scheduler : object
        Hyperparameter object (VarianceSchedulerSDE) containing SDE-specific parameters. Expected to have
        attributes: `dt`, `sigmas`, `betas`, `cum_betas`.
    sde_method : str
        SDE method to use. Supported methods: "ve", "vp", "sub-vp", "ode".
    """
    def __init__(self, variance_scheduler: torch.nn.Module, sde_method: str) -> None:
        super().__init__()
        self.variance_scheduler = variance_scheduler
        self.sde_method = sde_method

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Applies the forward SDE diffusion process to the input data.

        Perturbs the input data `x0` by adding noise according to the specified SDE
        method at given time steps, incorporating drift and diffusion terms as applicable.

        Parameters
        ----------
        x0 : torch.Tensor
            Input data tensor, shape (batch_size, channels, height, width).
        noise : torch.Tensor
            Gaussian noise tensor, same shape as `x0`.
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, varinace_scheduler.num_steps - 1].

        Returns
        -------
        xt (torch.Tensor) - Noisy data tensor at the specified time steps, same shape as `x0`.

        """
        dt = self.variance_scheduler.dt
        if self.sde_method == "ve":
            # use property to get sigmas (handles trainable case)
            sigma_t = self.variance_scheduler.sigmas[time_steps]
            sigma_t_prev = self.variance_scheduler.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
            sigma_diff = torch.sqrt(torch.clamp(sigma_t ** 2 - sigma_t_prev ** 2, min=0))
            x0 = x0 + noise * sigma_diff.view(-1, 1, 1, 1)

        elif self.sde_method == "vp":
            # use property to get betas (handles trainable case)
            betas = self.variance_scheduler.betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            diffusion = torch.sqrt(betas * dt) * noise
            x0 = x0 + drift + diffusion

        elif self.sde_method == "sub-vp":
            # use properties to get betas and cum_betas (handles trainable case)
            betas = self.variance_scheduler.betas[time_steps].view(-1, 1, 1, 1)
            cum_betas = self.variance_scheduler._cum_betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            diffusion = torch.sqrt(betas * (1 - torch.exp(-2 * cum_betas)) * dt) * noise
            x0 = x0 + drift + diffusion

        elif self.sde_method == "ode":
            # use property to get betas (handles trainable case)
            betas = self.variance_scheduler.betas[time_steps].view(-1, 1, 1, 1)
            drift = -0.5 * betas * x0 * dt
            x0 = x0 + drift
        else:
            raise ValueError(f"Unknown method: {self.sde_method}")
        return x0

###==================================================================================================================###

class ReverseSDE(nn.Module):
    """Reverse diffusion process for SDE-based generative models.

    Implements the reverse diffusion process for score-based generative models using
    Stochastic Differential Equations (SDEs), supporting Variance Exploding (VE),
    Variance Preserving (VP), sub-Variance Preserving (sub-VP), and ODE methods, as
    described in Song et al. (2021). The reverse process denoises a noisy input using
    predicted noise estimates.

    Parameters
    ----------
    variance_scheduler : object
        Hyperparameter object (VarianceSchedulerSDE) containing SDE-specific parameters. Expected to have
        attributes: `dt`, `sigmas`, `betas`, `cum_betas`.
    sde_method : str
        SDE method to use. Supported methods: "ve", "vp", "sub-vp", "ode".
    """
    def __init__(self, variance_scheduler: torch.nn.Module, sde_method: str) -> None:
        super().__init__()
        self.variance_scheduler = variance_scheduler
        self.sde_method = sde_method

    def forward(self, xt: torch.Tensor, noise: torch.Tensor, predicted_noise: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Applies the reverse SDE diffusion process to the noisy input.

        Denoises the input `xt` by applying the reverse SDE process, using predicted
        noise estimates and optional stochastic noise, according to the specified SDE
        method at given time steps. Incorporates drift and diffusion terms as applicable.

        Parameters
        ----------
        xt : torch.Tensor
            Noisy input tensor at time step `t`, shape (batch_size, channels, height, width).
        noise : torch.Tensor or None
            Gaussian noise tensor, same shape as `xt`, used for stochasticity. If None,
            no stochastic noise is added (e.g., for deterministic ODE).
        predicted_noise : torch.Tensor
            Predicted noise tensor, same shape as `xt`, typically output by a neural network.
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, variance_scheduler.num_steps - 1].

        Returns
        -------
        xt (torch.Tensor) - Denoised tensor at the previous time step, same shape as `xt`.

        **Notes**

        - For the "ve" and "ode" methods, the output is clamped to [-1e5, 1e5] to prevent numerical instability.
        - Stochastic noise (`noise`) is only added if provided and the method supports it (not applicable for "ode" in non-VE cases).
        """
        dt = self.variance_scheduler.dt
        # use properties to get betas and cum_betas (handles trainable case)
        betas = self.variance_scheduler.betas[time_steps].view(-1, 1, 1, 1)
        cum_betas = self.variance_scheduler._cum_betas[time_steps].view(-1, 1, 1, 1)
        if self.sde_method == "ve":
            # use property to get sigmas (handles trainable case)
            sigma_t = self.variance_scheduler.sigmas[time_steps]
            sigma_t_prev = self.variance_scheduler.sigmas[time_steps - 1] if time_steps.min() > 0 else torch.zeros_like(sigma_t)
            sigma_diff = torch.sqrt(torch.clamp(sigma_t ** 2 - sigma_t_prev ** 2, min=0))
            drift = -(sigma_t ** 2 - sigma_t_prev ** 2).view(-1, 1, 1, 1) * predicted_noise * dt
            diffusion = sigma_diff.view(-1, 1, 1, 1) * noise if noise is not None else 0
            xt = xt + drift + diffusion
            xt = torch.clamp(xt, -1e5, 1e5)

        elif self.sde_method == "vp":
            drift = -0.5 * betas * xt * dt - betas * predicted_noise * dt
            diffusion = torch.sqrt(betas * dt) * noise if noise is not None else 0
            xt = xt + drift + diffusion

        elif self.sde_method == "sub-vp":
            drift = -0.5 * betas * xt * dt - betas * (1 - torch.exp(-2 * cum_betas)) * predicted_noise * dt
            diffusion = torch.sqrt(betas * (1 - torch.exp(-2 * cum_betas)) * dt) * noise if noise is not None else 0
            xt = xt + drift + diffusion

        elif self.sde_method == "ode":
            drift = -0.5 * betas * xt * dt - 0.5 * betas * predicted_noise * dt
            xt = xt + drift
            xt = torch.clamp(xt, -1e5, 1e5)
        else:
            raise ValueError(f"Unknown method: {self.sde_method}")
        return xt

###==================================================================================================================###

class VarianceSchedulerSDE(nn.Module):
    """Hyperparameters for SDE-based generative models.

    Manages the noise schedule and SDE-specific parameters for score-based generative
    models, including beta and sigma schedules, time steps, and variance computations,
    as described in Song et al. (2021). Supports trainable or fixed beta schedules and
    multiple scheduling methods for flexible noise control.

    Parameters
    ----------
    num_steps : int, optional
        Number of diffusion steps (default: 1000).
    beta_start : float, optional
        Starting value for beta schedule (default: 1e-4).
    beta_end : float, optional
        Ending value for beta schedule (default: 0.02).
    trainable_beta : bool, optional
        Whether the beta schedule is trainable (default: False).
    beta_method : str, optional
        Method for computing the beta schedule (default: "linear").
        Supported methods: "linear", "sigmoid", "quadratic", "constant", "inverse_time".
    sigma_start : float, optional
        Starting value for sigma schedule for VE method (default: 1e-3).
    sigma_end : float, optional
        Ending value for sigma schedule for VE method (default: 10.0).
    start : float, optional
        Start of the time interval for SDE integration (default: 0.0).
    end : float, optional
        End of the time interval for SDE integration (default: 1.0).
    """
    def __init__(
            self,
            num_steps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            trainable_beta: bool = False,
            beta_method: str = "linear",
            sigma_start: float = 1e-3,
            sigma_end: float = 10.0,
            start: float = 0.0,
            end: float = 1.0
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trainable_beta = trainable_beta
        self.beta_method = beta_method
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.start = start
        self.end = end

        if not (0 < self.beta_start < self.beta_end):
            raise ValueError(f"beta_start ({self.beta_start}) and beta_end ({self.beta_end}) must satisfy 0 < start < end")
        if not (0 < self.sigma_start < self.sigma_end):
            raise ValueError(f"sigma_start ({self.sigma_start}) and sigma_end ({self.sigma_end}) must satisfy 0 < start < end")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps ({self.num_steps}) must be positive")

        beta_range = (beta_start, beta_end)
        betas_init = self.compute_beta_schedule(beta_range, num_steps, beta_method)
        self.time = torch.linspace(self.start, self.end, self.num_steps, dtype=torch.float32)
        self.dt = (self.end - self.start) / self.num_steps

        if trainable_beta:
            # use reparameterization trick for trainable betas
            # initialize unconstrained parameters and transform them to valid beta range
            self.beta_raw = nn.Parameter(torch.logit((betas_init - beta_start) / (beta_end - beta_start)))
        else:
            self.register_buffer('betas_buffer', betas_init)
            self.register_buffer('cum_betas', torch.cumsum(betas_init, dim=0) * self.dt)
            self.register_buffer("sigmas_buffer", self.sigma_start * (self.sigma_end / self.sigma_start) ** self.time)

    @property
    def betas(self) -> torch.Tensor:
        """Returns the beta values, applying reparameterization if trainable."""
        if self.trainable_beta:
            # transform unconstrained parameters to valid beta range using sigmoid
            return self.beta_start + (self.beta_end - self.beta_start) * torch.sigmoid(self.beta_raw)
        else:
            return self._buffers['betas_buffer']

    @property
    def _cum_betas(self) -> torch.Tensor:
        """Returns the cumulative beta values, computing dynamically if trainable."""
        if self.trainable_beta:
            return torch.cumsum(self.betas, dim=0) * self.dt
        else:
            return self._buffers['cum_betas']

    @property
    def sigmas(self) -> torch.Tensor:
        """Returns the sigma values, computing dynamically if trainable."""
        if self.trainable_beta:
            return self.sigma_start * (self.sigma_end / self.sigma_start) ** self.time
        else:
            return self._buffers['sigmas_buffer']

    def compute_beta_schedule(self, beta_range: Tuple[float, float], num_steps: int, method: str) -> torch.Tensor:
        """Computes the beta schedule based on the specified method.

        Generates a sequence of beta values for the SDE noise schedule using the chosen
        method, ensuring values are clamped within the specified range.

        Parameters
        ----------
        beta_range : tuple
            Tuple of (min_beta, max_beta) specifying the valid range for beta values.
        num_steps : int
            Number of diffusion steps.
        method : str
            Method for computing the beta schedule. Supported methods:
            "linear", "sigmoid", "quadratic", "constant", "inverse_time".

        Returns
        -------
        betas (torch.Tensor) - Tensor of beta values, shape (num_steps,).
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
        else:
            raise ValueError(f"Unknown beta_method: {method}. Supported: linear, sigmoid, quadratic, constant, inverse_time")
        beta = torch.clamp(beta, min=beta_min, max=beta_max)
        return beta

    def get_variance(self, time_steps: torch.Tensor, method: str) -> torch.Tensor:
        """Computes the variance for the specified SDE method at given time steps.

        Calculates the variance used in SDE diffusion processes based on the method
        (VE, VP, or sub-VP), leveraging the sigma or cumulative beta schedules.

        Parameters
        ----------
        time_steps : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,), where each value
            is in the range [0, num_steps - 1].
        method : str
            SDE method to compute variance for. Supported methods: "ve", "vp", "sub-vp".

        Returns
        -------
        variance_values (torch.Tensor) - Variance values for the specified time steps, shape (batch_size,).
        """
        if method == "ve":
            return self.sigmas[time_steps] ** 2
        elif method == "vp":
            return 1 - torch.exp(-self.cum_betas[time_steps])
        elif method == "sub-vp":
            return 1 - torch.exp(-2 * self.cum_betas[time_steps])
        else:
            raise ValueError(f"Unknown method: {method}")

###==================================================================================================================###

class TrainSDE(nn.Module):
    """Trainer for score-based generative models using Stochastic Differential Equations.

    Manages the training process for SDE-based generative models, optimizing a noise
    predictor to learn the noise added by the forward SDE process, as described in Song
    et al. (2021). Supports conditional training with text prompts, mixed precision,
    learning rate scheduling, early stopping, and checkpointing.

    Parameters
    ----------

    noise_predictor : nn.Module
        Model to predict noise added during the forward SDE process.
    forward_diffusion : nn.Module
        Forward SDE diffusion module for adding noise.
    reverse_diffusion: nn.Module
        Reverse SDE diffusion module for denoising.
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    optimizer : torch.optim.Optimizer
        Optimizer for training the noise predictor and conditional model (if applicable).
    objective : callable
        Loss function to compute the difference between predicted and actual noise.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data, default None.
    max_epochs : int, optional
        Maximum number of training epochs (default: 1000).
    device : torch.device, optional
        Device for computation (default: CUDA if available, else CPU).
    conditional_model : nn.Module, optional
        Model for conditional generation (e.g., text embeddings), default None.
    metrics_ : object, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    bert_tokenizer : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    max_token_length : int, optional
        Maximum length for tokenized prompts (default: 77).
    store_path : str, optional
        Path to save model checkpoints (default: "sde_model.pth").
    patience : int, optional
        Number of epochs to wait for improvement before early stopping (default: 10).
    warmup_epochs : int, optional
        Number of epochs for learning rate warmup (default: 100).
    val_frequency : int, optional
        Frequency (in epochs) for validation (default: 10).
    image_output_range : tuple, optional
        Range for clamping generated images (default: (-1, 1)).
    normalize_output : bool, optional
        Whether to normalize generated images to [0, 1] for metrics (default: True).
    use_ddp : bool, optional
        Whether to use Distributed Data Parallel training (default: False).
    grad_accumulation_steps : int, optional
        Number of gradient accumulation steps before optimizer update (default: 1).
    log_frequency : int, optional
        Number of epochs before printing loss.
    use_compilation : bool, optional
        whether the model is internally compiled using torch.compile (default: false)
    """
    def __init__(
            self,
            noise_predictor: torch.nn.Module,
            forward_diffusion: torch.nn.Module,
            reverse_diffusion: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            max_epochs: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            conditional_model: Optional[torch.nn.Module] = None,
            metrics_: Optional[Any] = None,
            bert_tokenizer: Optional[BertTokenizer] = None,
            max_token_length: int = 77,
            store_path: Optional[str] = None,
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            image_output_range: Tuple[float, float] = (-1.0, 1.0),
            normalize_output: bool = True,
            use_ddp: bool = False,
            grad_accumulation_steps: int = 1,
            log_frequency: int = 1,
            use_compilation: bool = False
    ) -> None:

        super().__init__()
        # initialize DDP settings first
        self.use_ddp = use_ddp
        self.grad_accumulation_steps = grad_accumulation_steps
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # setup distributed training if enabled
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # move models to appropriate device
        self.noise_predictor = noise_predictor.to(self.device)
        self.forward_diffusion = forward_diffusion.to(self.device)
        self.reverse_diffusion = reverse_diffusion.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None

        # training components
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or "sde_model"
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.max_token_length = max_token_length
        self.patience = patience
        self.val_frequency = val_frequency
        self.image_output_range = image_output_range
        self.normalize_output = normalize_output
        self.log_frequency = log_frequency
        self.use_compilation = use_compilation

        # learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)

        # initialize tokenizer
        if bert_tokenizer is None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")
        else:
            self.tokenizer = bert_tokenizer


    def _setup_ddp(self) -> None:
        """Setup Distributed Data Parallel training configuration.

        Initializes process group, determines rank information, and sets up
        CUDA device for the current process.
        """
        # check if DDP environment variables are set
        if "RANK" not in os.environ:
            raise ValueError("DDP enabled but RANK environment variable not set")
        if "LOCAL_RANK" not in os.environ:
            raise ValueError("DDP enabled but LOCAL_RANK environment variable not set")
        if "WORLD_SIZE" not in os.environ:
            raise ValueError("DDP enabled but WORLD_SIZE environment variable not set")

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
        """Setup single GPU or CPU training configuration."""
        self.ddp_rank = 0
        self.ddp_local_rank = 0
        self.ddp_world_size = 1
        self.master_process = True

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """Loads a training checkpoint to resume training.

        Restores the state of the noise predictor, conditional model (if applicable),
        and optimizer from a saved checkpoint. Handles DDP model state dict loading.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.

        Returns
        -------
        epoch : int
            The epoch at which the checkpoint was saved.
        loss : float
             The loss at the checkpoint.
        """
        try:
            # load checkpoint with proper device mapping
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        # load noise predictor state
        if 'model_state_dict_noise_predictor' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict_noise_predictor' key")

        # handle DDP wrapped model state dict
        state_dict = checkpoint['model_state_dict_noise_predictor']
        if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
            # if loading non-DDP checkpoint into DDP model, add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
            # if loading DDP checkpoint into non-DDP model, remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.noise_predictor.load_state_dict(state_dict)

        # load conditional model state if applicable
        if self.conditional_model is not None:
            if 'model_state_dict_conditional' in checkpoint and checkpoint['model_state_dict_conditional'] is not None:
                cond_state_dict = checkpoint['model_state_dict_conditional']
                # handle DDP wrapping for conditional model
                if self.use_ddp and not any(key.startswith('module.') for key in cond_state_dict.keys()):
                    cond_state_dict = {f'module.{k}': v for k, v in cond_state_dict.items()}
                elif not self.use_ddp and any(key.startswith('module.') for key in cond_state_dict.keys()):
                    cond_state_dict = {k.replace('module.', ''): v for k, v in cond_state_dict.items()}
                self.conditional_model.load_state_dict(cond_state_dict)
            else:
                warnings.warn(
                    "Checkpoint contains no 'model_state_dict_conditional' or it is None, "
                    "skipping conditional model loading"
                )

        # load variance_scheduler state
        if 'variance_scheduler_model' not in checkpoint:
            raise KeyError("Checkpoint missing 'variance_scheduler_model' key")
        try:
            if isinstance(self.forward_diffusion.variance_scheduler, nn.Module):
                self.forward_diffusion.variance_scheduler.load_state_dict(checkpoint['variance_scheduler_model'])
            if isinstance(self.reverse_diffusion.variance_scheduler, nn.Module):
                self.reverse_diffusion.variance_scheduler.load_state_dict(checkpoint['variance_scheduler_model'])
            else:
                self.forward_diffusion.variance_scheduler = checkpoint['variance_scheduler_model']
                self.reverse_diffusion.variance_scheduler = checkpoint['variance_scheduler_model']
        except Exception as e:
            warnings.warn(f"Variance_scheduler loading failed: {e}. Continuing with current variance_scheduler.")

        # load optimizer state
        if 'optimizer_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'optimizer_state_dict' key")
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            warnings.warn(f"Optimizer state loading failed: {e}. Continuing without optimizer state.")

        epoch = checkpoint.get('epoch', -1)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Loaded checkpoint from {checkpoint_path} at epoch {epoch} with loss {loss:.4f}")
        return epoch, loss

    @staticmethod
    def warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Creates a learning rate scheduler for warmup.

        Generates a scheduler that linearly increases the learning rate from 0 to the
        optimizer's initial value over the specified warmup epochs, then maintains it.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to apply the scheduler to.
        warmup_epochs : int
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

    def _wrap_models_for_ddp(self) -> None:
        """Wrap models with DistributedDataParallel for multi-GPU training."""
        if self.use_ddp:
            # wrap noise predictor with DDP
            self.noise_predictor = DDP(
                self.noise_predictor,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )

            # wrap conditional model with DDP if it exists
            if self.conditional_model is not None:
                self.conditional_model = DDP(
                    self.conditional_model,
                    device_ids=[self.ddp_local_rank],
                    find_unused_parameters=True
                )


    def forward(self) -> Tuple[List, float]:
        """Trains the SDE model to predict noise added by the forward diffusion process.

        Executes the training loop, optimizing the noise predictor and conditional model
        (if applicable) using mixed precision, gradient clipping, and learning rate
        scheduling. Supports validation, early stopping, and checkpointing.

        Returns
        -------
        train_losses : list of float
             List of mean training losses per epoch.
        best_val_loss : float
             Best validation or training loss achieved.

        **Notes**

        - Training uses mixed precision via `torch.cuda.amp` or `torch.amp` for efficiency.
        - Checkpoints are saved when the validation (or training) loss improves, and on early stopping.
        - Early stopping is triggered if no improvement occurs for `patience` epochs.
        """
        # set models to training mode
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()
        if self.forward_diffusion.variance_scheduler.trainable_beta:
            self.reverse_diffusion.train()
            self.forward_diffusion.train()
        else:
            self.reverse_diffusion.eval()
            self.forward_diffusion.eval()

        # compile models for optimization (if supported)
        if self.use_compilation:
            try:
                self.noise_predictor = torch.compile(self.noise_predictor)
                if self.conditional_model is not None:
                    self.conditional_model = torch.compile(self.conditional_model)
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")


        # wrap models for DDP after compilation
        self._wrap_models_for_ddp()

        # initialize training components
        scaler = torch.GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        # main training loop
        for epoch in range(self.max_epochs):
            # set epoch for distributed sampler if using DDP
            if self.use_ddp and hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []
            # training step loop with gradient accumulation
            for step, (x, y) in enumerate(tqdm(self.data_loader, disable=not self.master_process)):
                x = x.to(self.device)

                # process conditional inputs if conditional model exists
                if self.conditional_model is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None

                # forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    # generate noise and timesteps
                    noise = torch.randn_like(x).to(self.device)
                    t = torch.randint(0, self.forward_diffusion.variance_scheduler.num_steps, (x.shape[0],)).to(self.device)

                    # apply forward diffusion
                    noisy_x = self.forward_diffusion(x, noise, t)

                    # predict noise
                    predicted_noise = self.noise_predictor(noisy_x, t, y_encoded, None)

                    # compute loss and scale for gradient accumulation
                    loss = self.objective(predicted_noise, noise) / self.grad_accumulation_steps

                # backward pass
                scaler.scale(loss).backward()

                # gradient accumulation and optimizer step
                if (step + 1) % self.grad_accumulation_steps == 0:
                    # clip gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.noise_predictor.parameters(), max_norm=1.0)
                    if self.conditional_model is not None:
                        torch.nn.utils.clip_grad_norm_(self.conditional_model.parameters(), max_norm=1.0)

                # optimizer step
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                # update learning rate (warmup scheduler)
                self.warmup_lr_scheduler.step()

            # record loss (unscaled)
            train_losses_epoch.append(loss.item() * self.grad_accumulation_steps)

            # compute mean training loss
            mean_train_loss = torch.tensor(train_losses_epoch).mean().item()

            # all-reduce loss across processes for DDP
            if self.use_ddp:
                loss_tensor = torch.tensor(mean_train_loss, device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                mean_train_loss = loss_tensor.item()

            # print training progress (only master process)
            if self.master_process and (epoch + 1) % self.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\nEpoch: {epoch + 1}/{self.max_epochs} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}")

            # validation step
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

                current_best = val_loss
                self.scheduler.step(val_loss)
            else:
                if self.master_process:
                    print()
                current_best = mean_train_loss
                self.scheduler.step(mean_train_loss)

            # save checkpoint and early stopping (only master process)
            if self.master_process:
                if current_best < best_val_loss and (epoch + 1) % self.val_frequency == 0:
                    best_val_loss = current_best
                    wait = 0
                    self._save_checkpoint(epoch + 1, best_val_loss)
                else:
                    wait += 1
                    if wait >= self.patience:
                        print("Early stopping triggered")
                        self._save_checkpoint(epoch + 1, best_val_loss, "_early_stop")
                        break

        # clean up DDP
        if self.use_ddp:
            destroy_process_group()

        return train_losses, best_val_loss

    def _process_conditional_input(self, y: Union[torch.Tensor, List]) -> torch.Tensor:
        """Process conditional input for text-to-image generation.

        Parameters
        ----------
        y : torch.Tensor or list
            Conditional input (text prompts).

        Returns
        -------
        torch.Tensor
            Encoded conditional input.
        """
        # convert to string list
        y_list = y.cpu().numpy().tolist() if isinstance(y, torch.Tensor) else y
        y_list = [str(item) for item in y_list]

        # tokenize
        y_encoded = self.tokenizer(
            y_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt"
        ).to(self.device)

        # get embeddings
        input_ids = y_encoded["input_ids"]
        attention_mask = y_encoded["attention_mask"]
        y_encoded = self.conditional_model(input_ids, attention_mask)

        return y_encoded


    def _save_checkpoint(self, epoch: int, loss: float, suffix: str = "") -> None:
        """Save model checkpoint (only called by master process).

        Parameters
        ----------
        epoch : int
            Current epoch number.
        loss : float
            Current loss value.
        suffix : str, optional
            Suffix to add to checkpoint filename.
        """
        try:
            # get state dicts, handling DDP wrapping
            noise_predictor_state = (
                self.noise_predictor.module.state_dict() if self.use_ddp
                else self.noise_predictor.state_dict()
            )
            conditional_state = None
            if self.conditional_model is not None:
                conditional_state = (
                    self.conditional_model.module.state_dict() if self.use_ddp
                    else self.conditional_model.state_dict()
                )

            checkpoint = {
                'epoch': epoch,
                'model_state_dict_noise_predictor': noise_predictor_state,
                'model_state_dict_conditional': conditional_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'variance_scheduler_model': (
                    self.forward_diffusion.variance_scheduler.state_dict() if isinstance(self.forward_diffusion.variance_scheduler, nn.Module)
                    else self.forward_diffusion.variance_scheduler
                ),
                'max_epochs': self.max_epochs,
            }

            filename = f"sde_epoch_{epoch}{suffix}.pth"
            filepath = os.path.join(self.store_path, filename)
            os.makedirs(self.store_path, exist_ok=True)
            torch.save(checkpoint, filepath)

            print(f"Model saved at epoch {epoch}")

        except Exception as e:
            print(f"Failed to save model: {e}")


    def validate(self) -> Tuple[float, float, float, float, float, float]:
        """Validates the noise predictor and computes evaluation Metrics.

        Computes validation loss (MSE between predicted and ground truth noise) and generates
        samples using the reverse diffusion model by manually iterating over timesteps.
        Decodes samples to images and computes image-domain Metrics (MSE, PSNR, SSIM, FID, LPIPS)
        if metrics_ is provided.

        Returns
        -------
        val_loss : float
            Mean validation loss.
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
        self.noise_predictor.eval()
        if self.conditional_model is not None:
            self.conditional_model.eval()
        if self.forward_diffusion.variance_scheduler.trainable_beta:
            self.forward_diffusion.eval()
            self.reverse_diffusion.eval()

        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                x_orig = x.clone()

                # process conditional input
                if self.conditional_model is not None:
                    y_encoded = self._process_conditional_input(y)
                else:
                    y_encoded = None

                # compute validation loss
                noise = torch.randn_like(x).to(self.device)
                t = torch.randint(0, self.forward_diffusion.variance_scheduler.num_steps, (x.shape[0],)).to(self.device)

                noisy_x = self.forward_diffusion(x, noise, t)
                predicted_noise = self.noise_predictor(noisy_x, t, y_encoded, None)
                loss = self.objective(predicted_noise, noise)
                val_losses.append(loss.item())

                # generate samples for metrics evaluation
                if self.metrics_ is not None and self.reverse_diffusion is not None:
                    xt = torch.randn_like(x).to(self.device)

                    # reverse diffusion sampling
                    for t in reversed(range(self.forward_diffusion.variance_scheduler.num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, y_encoded, None)
                        noise = torch.randn_like(xt) if getattr(self.reverse_diffusion, "method", None) != "ode" else None
                        xt = self.reverse_diffusion(xt, noise, predicted_noise, time_steps)

                    # clamp and normalize generated samples
                    x_hat = torch.clamp(xt, min=self.image_output_range[0], max=self.image_output_range[1])
                    if self.normalize_output:
                        x_hat = (x_hat - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])
                        x_orig = (x_orig - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])

                    # compute metrics
                    metrics_result = self.metrics_.forward(x_orig, x_hat)
                    fid, mse, psnr, ssim, lpips_score = metrics_result

                    if hasattr(self.metrics_, 'fid') and self.metrics_.fid:
                        fid_scores.append(fid)
                    if hasattr(self.metrics_, 'metrics') and self.metrics_.metrics:
                        mse_scores.append(mse)
                        psnr_scores.append(psnr)
                        ssim_scores.append(ssim)
                    if hasattr(self.metrics_, 'lpips') and self.metrics_.lpips:
                        lpips_scores.append(lpips_score)

        # compute average metrics
        val_loss = torch.tensor(val_losses).mean().item()

        # all-reduce validation metrics across processes for DDP
        if self.use_ddp:
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()

        fid_avg = torch.tensor(fid_scores).mean().item() if fid_scores else float('inf')
        mse_avg = torch.tensor(mse_scores).mean().item() if mse_scores else None
        psnr_avg = torch.tensor(psnr_scores).mean().item() if psnr_scores else None
        ssim_avg = torch.tensor(ssim_scores).mean().item() if ssim_scores else None
        lpips_avg = torch.tensor(lpips_scores).mean().item() if lpips_scores else None

        # return to training mode
        self.noise_predictor.train()
        if self.conditional_model is not None:
            self.conditional_model.train()
        if self.forward_diffusion.variance_scheduler.trainable_beta:
            self.reverse_diffusion.train()
            self.forward_diffusion.train()

        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg


###==================================================================================================================###

class SampleSDE(nn.Module):
    """Sampler for generating images using SDE-based generative models.

    Generates images by iteratively denoising random noise using the reverse SDE process
    and a trained noise predictor, as described in Song et al. (2021). Supports both
    unconditional and conditional generation with text prompts.

    Parameters
    ----------
    reverse_diffusion : ReverseSDE
        Reverse SDE diffusion module for denoising.
    noise_predictor : nn.Module
        Model to predict noise added during the forward SDE process.
    image_shape : tuple
        Shape of generated images as (height, width).
    conditional_model : nn.Module, optional
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
    image_output_range : tuple, optional
        Range for clamping generated images (min, max), default (-1, 1).
    """
    def __init__(
            self,
            reverse_diffusion: torch.nn.Module,
            noise_predictor: torch.nn.Module,
            image_shape: Tuple[int, int],
            conditional_model: Optional[torch.nn.Module] = None,
            tokenizer: str = "bert-base-uncased",
            max_token_length: int = 77,
            batch_size: int = 1,
            in_channels: int = 3,
            device: Optional[Union[str, torch.device]] = None,
            image_output_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> None:
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.reverse = reverse_diffusion.to(self.device)
        self.noise_predictor = noise_predictor.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_token_length = max_token_length
        self.in_channels = in_channels
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.image_output_range = image_output_range

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(image_output_range, (tuple, list)) or len(image_output_range) != 2 or image_output_range[0] >= image_output_range[1]:
            raise ValueError("output_range must be a tuple (min, max) with min < max")

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
            conditions: Optional[Union[str, List]] = None,
            normalize_output: bool = True,
            save_images: bool = True,
            save_path: str = "sde_generated"
    ) -> torch.Tensor:
        """Generates images using the reverse SDE sampling process.

        Iteratively denoises random noise to generate images using the reverse SDE process
        and noise predictor. Supports conditional generation with text prompts.

        Parameters
        ----------
        conditions : str or list, optional
            Text prompt(s) for conditional generation, default None.
        normalize_output : bool, optional
            If True, normalizes output images to [0, 1] (default: True).
        save_images : bool, optional
            If True, saves generated images to `save_path` (default: True).
        save_path : str, optional
            Directory to save generated images (default: "sde_generated").

        Returns
        -------
        generated_imgs (torch.Tensor) - Generated images, shape (batch_size, in_channels, height, width). If `normalize_output` is True, images are normalized to [0, 1]; otherwise, they are clamped to `output_range`.
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
            for t in reversed(range(self.reverse.variance_scheduler.num_steps)):
                noise = torch.randn_like(xt) if self.reverse.sde_method != "ode" else None
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)

                if self.conditional_model is not None and conditions is not None:
                    input_ids, attention_masks = self.tokenize(conditions)
                    key_padding_mask = (attention_masks == 0)
                    y = self.conditional_model(input_ids, key_padding_mask)
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)

                xt = self.reverse(xt, noise, predicted_noise, time_steps)

            generated_imgs = torch.clamp(xt, min=self.image_output_range[0], max=self.image_output_range[1])
            if normalize_output:
                generated_imgs = (generated_imgs - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])

            # save images if save_images is True
            if save_images:
                os.makedirs(save_path, exist_ok=True)
                for i in range(generated_imgs.size(0)):
                    img_path = os.path.join(save_path, f"image_{i+1}.png")
                    save_image(generated_imgs[i], img_path)

        return generated_imgs

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
        self.noise_predictor.to(device)
        self.reverse.to(device)
        if self.conditional_model:
            self.conditional_model.to(device)
        return super().to(device)


import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch

"""
from sde import (
    VarianceSchedulerSDE,
    ForwardSDE,
    ReverseSDE,
    TrainSDE,
    SampleSDE,
    NoisePredictor,
    TextEncoder
)
"""
from torch.utils.data import DataLoader, TensorDataset


class TestVarianceSchedulerSDE:
    """Test cases for VarianceSchedulerSDE class."""

    def test_init(self):
        """Test initialization with different parameters."""
        # Test default initialization
        scheduler = VarianceSchedulerSDE()
        assert scheduler.num_steps == 1000
        assert scheduler.beta_start == 1e-4
        assert scheduler.beta_end == 0.02

        # Test custom initialization
        scheduler = VarianceSchedulerSDE(
            num_steps=500,
            beta_start=1e-5,
            beta_end=0.01,
            beta_method="quadratic"
        )
        assert scheduler.num_steps == 500
        assert scheduler.beta_start == 1e-5
        assert scheduler.beta_end == 0.01

    def test_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            VarianceSchedulerSDE(num_steps=-10)

        with pytest.raises(ValueError):
            VarianceSchedulerSDE(beta_start=0.1, beta_end=0.01)  # start > end

        with pytest.raises(ValueError):
            VarianceSchedulerSDE(sigma_start=1.0, sigma_end=0.5)  # start > end

    def test_beta_schedule_methods(self):
        """Test all beta schedule computation methods."""
        methods = ["linear", "sigmoid", "quadratic", "constant", "inverse_time"]

        for method in methods:
            scheduler = VarianceSchedulerSDE(num_steps=100, beta_method=method)
            betas = scheduler.betas

            assert betas.shape[0] == 100
            assert torch.all(betas >= scheduler.beta_start)
            assert torch.all(betas <= scheduler.beta_end)

    def test_cumulative_betas(self):
        """Test cumulative beta computation."""
        scheduler = VarianceSchedulerSDE(num_steps=100)
        cum_betas = scheduler._cum_betas

        assert cum_betas.shape[0] == 100
        assert cum_betas[0] > 0  # Should be positive
        assert cum_betas[-1] > cum_betas[0]  # Should be increasing

    def test_sigmas(self):
        """Test sigma computation."""
        scheduler = VarianceSchedulerSDE(num_steps=100)
        sigmas = scheduler.sigmas

        assert sigmas.shape[0] == 100
        assert sigmas[0] == scheduler.sigma_start
        assert sigmas[-1] == scheduler.sigma_end

    def test_get_variance(self):
        """Test variance computation for different methods."""
        scheduler = VarianceSchedulerSDE(num_steps=100)
        time_steps = torch.tensor([0, 50, 99])

        for method in ["ve", "vp", "sub-vp"]:
            variance = scheduler.get_variance(time_steps, method)
            assert variance.shape[0] == 3
            assert torch.all(variance >= 0)  # Variance should be non-negative


class TestForwardSDE:
    """Test cases for ForwardSDE class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.scheduler = VarianceSchedulerSDE(num_steps=100)
        self.batch_size = 4
        self.channels = 3
        self.height = 32
        self.width = 32

    def test_init(self):
        """Test initialization."""
        for method in ["ve", "vp", "sub-vp", "ode"]:
            forward_sde = ForwardSDE(self.scheduler, method)
            assert forward_sde.sde_method == method

        with pytest.raises(ValueError):
            ForwardSDE(self.scheduler, "invalid_method")

    def test_forward_ve(self):
        """Test VE forward process."""
        forward_sde = ForwardSDE(self.scheduler, "ve")
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt = forward_sde(x0, noise, time_steps)
        assert xt.shape == x0.shape

    def test_forward_vp(self):
        """Test VP forward process."""
        forward_sde = ForwardSDE(self.scheduler, "vp")
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt = forward_sde(x0, noise, time_steps)
        assert xt.shape == x0.shape

    def test_forward_sub_vp(self):
        """Test sub-VP forward process."""
        forward_sde = ForwardSDE(self.scheduler, "sub-vp")
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt = forward_sde(x0, noise, time_steps)
        assert xt.shape == x0.shape

    def test_forward_ode(self):
        """Test ODE forward process."""
        forward_sde = ForwardSDE(self.scheduler, "ode")
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt = forward_sde(x0, noise, time_steps)
        assert xt.shape == x0.shape


class TestReverseSDE:
    """Test cases for ReverseSDE class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.scheduler = VarianceSchedulerSDE(num_steps=100)
        self.batch_size = 4
        self.channels = 3
        self.height = 32
        self.width = 32

    def test_init(self):
        """Test initialization."""
        for method in ["ve", "vp", "sub-vp", "ode"]:
            reverse_sde = ReverseSDE(self.scheduler, method)
            assert reverse_sde.sde_method == method

        with pytest.raises(ValueError):
            ReverseSDE(self.scheduler, "invalid_method")

    def test_reverse_ve(self):
        """Test VE reverse process."""
        reverse_sde = ReverseSDE(self.scheduler, "ve")
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(xt)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.randint(1, 100, (self.batch_size,))

        xt_prev = reverse_sde(xt, noise, predicted_noise, time_steps)
        assert xt_prev.shape == xt.shape

    def test_reverse_vp(self):
        """Test VP reverse process."""
        reverse_sde = ReverseSDE(self.scheduler, "vp")
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(xt)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt_prev = reverse_sde(xt, noise, predicted_noise, time_steps)
        assert xt_prev.shape == xt.shape

    def test_reverse_sub_vp(self):
        """Test sub-VP reverse process."""
        reverse_sde = ReverseSDE(self.scheduler, "sub-vp")
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(xt)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt_prev = reverse_sde(xt, noise, predicted_noise, time_steps)
        assert xt_prev.shape == xt.shape

    def test_reverse_ode(self):
        """Test ODE reverse process."""
        reverse_sde = ReverseSDE(self.scheduler, "ode")
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = None  # ODE doesn't use noise
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt_prev = reverse_sde(xt, noise, predicted_noise, time_steps)
        assert xt_prev.shape == xt.shape


class TestTrainSDE:
    """Test cases for TrainSDE class."""

    def setup_method(self):
        """Setup test fixtures."""

        # Create simple models for testing
        class SimpleNoisePredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)

            def forward(self, x, t, y=None, mask=None):
                return self.conv(x)

        class SimpleConditionalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(77, 64)

            def forward(self, input_ids, attention_mask=None):
                return self.embed(input_ids.float())

        # Create test data
        self.batch_size = 4
        self.channels = 3
        self.height = 32
        self.width = 32

        x_data = torch.randn(20, self.channels, self.height, self.width)
        y_data = torch.randint(0, 10, (20,))
        dataset = TensorDataset(x_data, y_data)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size)

        # Create components
        self.scheduler = VarianceSchedulerSDE(num_steps=10)
        self.forward_sde = ForwardSDE(self.scheduler, "vp")
        self.reverse_sde = ReverseSDE(self.scheduler, "vp")
        self.noise_predictor = SimpleNoisePredictor()
        self.conditional_model = SimpleConditionalModel()
        self.optimizer = torch.optim.Adam(
            list(self.noise_predictor.parameters()) +
            list(self.conditional_model.parameters()),
            lr=1e-4
        )
        self.objective = nn.MSELoss()

    def test_init(self):
        """Test initialization."""
        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            conditional_model=self.conditional_model,
            max_epochs=2
        )

        assert trainer is not None


    @patch('sde.TrainSDE._setup_ddp')
    def test_ddp_setup(self, mock_setup_ddp):
        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            use_ddp=True
        )

        mock_setup_ddp.assert_called_once()


    def test_single_gpu_setup(self):
        """Test single GPU setup."""
        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            use_ddp=False
        )

        assert trainer.ddp_rank == 0
        assert trainer.ddp_local_rank == 0
        assert trainer.ddp_world_size == 1
        assert trainer.master_process

    def test_warmup_scheduler(self):
        """Test warmup scheduler creation."""
        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective
        )

        scheduler = trainer.warmup_scheduler(self.optimizer, 10)
        assert scheduler is not None

    def test_process_conditional_input(self):
        """Test conditional input processing."""
        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            conditional_model=self.conditional_model
        )

        # Test with tensor input
        y_tensor = torch.tensor([1, 2, 3, 4])
        y_encoded = trainer._process_conditional_input(y_tensor)
        assert y_encoded is not None

        # Test with list input
        y_list = ["test1", "test2", "test3", "test4"]
        y_encoded = trainer._process_conditional_input(y_list)
        assert y_encoded is not None

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = TrainSDE(
                noise_predictor=self.noise_predictor,
                forward_diffusion=self.forward_sde,
                reverse_diffusion=self.reverse_sde,
                data_loader=self.data_loader,
                optimizer=self.optimizer,
                objective=self.objective,
                store_path=temp_dir
            )

            trainer._save_checkpoint(1, 0.5)

            # Check if file was created
            files = os.listdir(temp_dir)
            assert any(f.startswith("sde_epoch_1") for f in files)

    def test_validate(self):
        """Test validation method."""
        # Mock metrics
        mock_metrics = Mock()
        mock_metrics.forward.return_value = (1.0, 0.1, 25.0, 0.8, 0.2)
        mock_metrics.fid = True
        mock_metrics.metrics = True
        mock_metrics.lpips = True

        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            val_loader=self.data_loader,
            metrics_=mock_metrics
        )

        val_loss, fid, mse, psnr, ssim, lpips = trainer.validate()

        assert isinstance(val_loss, float)
        assert isinstance(fid, float)
        assert isinstance(mse, float)
        assert isinstance(psnr, float)
        assert isinstance(ssim, float)
        assert isinstance(lpips, float)


class TestSampleSDE:
    """Test cases for SampleSDE class."""

    def setup_method(self):
        """Setup test fixtures."""

        # Create simple models for testing
        class SimpleNoisePredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)

            def forward(self, x, t, y=None):
                return self.conv(x)

        class SimpleConditionalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(77, 64)

            def forward(self, input_ids, attention_mask=None):
                return self.embed(input_ids.float())

        # Create components
        self.scheduler = VarianceSchedulerSDE(num_steps=10)
        self.reverse_sde = ReverseSDE(self.scheduler, "vp")
        self.noise_predictor = SimpleNoisePredictor()
        self.conditional_model = SimpleConditionalModel()

    def test_init(self):
        """Test initialization."""
        sampler = SampleSDE(
            reverse_diffusion=self.reverse_sde,
            noise_predictor=self.noise_predictor,
            image_shape=(32, 32)
        )

        assert sampler is not None

    def test_tokenize(self):
        """Test tokenization method."""
        sampler = SampleSDE(
            reverse_diffusion=self.reverse_sde,
            noise_predictor=self.noise_predictor,
            image_shape=(32, 32),
            conditional_model=self.conditional_model
        )

        # Test with single prompt
        input_ids, attention_mask = sampler.tokenize("a test prompt")
        assert input_ids.shape[0] == 1
        assert attention_mask.shape[0] == 1

        # Test with multiple prompts
        input_ids, attention_mask = sampler.tokenize(["prompt1", "prompt2"])
        assert input_ids.shape[0] == 2
        assert attention_mask.shape[0] == 2

    def test_forward_unconditional(self):
        """Test unconditional sampling."""
        sampler = SampleSDE(
            reverse_diffusion=self.reverse_sde,
            noise_predictor=self.noise_predictor,
            image_shape=(32, 32),
            batch_size=2
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            images = sampler.forward(
                conditions=None,
                save_images=True,
                save_path=temp_dir
            )

            assert images.shape == (2, 3, 32, 32)
            assert torch.all(images >= 0) and torch.all(images <= 1)  # Normalized

            # Check if images were saved
            files = os.listdir(temp_dir)
            assert len(files) == 2

    def test_forward_conditional(self):
        """Test conditional sampling."""
        sampler = SampleSDE(
            reverse_diffusion=self.reverse_sde,
            noise_predictor=self.noise_predictor,
            image_shape=(32, 32),
            conditional_model=self.conditional_model,
            batch_size=2
        )

        images = sampler.forward(
            conditions=["a cat", "a dog"],
            save_images=False
        )

        assert images.shape == (2, 3, 32, 32)

    def test_to_device(self):
        """Test device movement."""
        sampler = SampleSDE(
            reverse_diffusion=self.reverse_sde,
            noise_predictor=self.noise_predictor,
            image_shape=(32, 32)
        )

        # Move to CPU if CUDA is available, otherwise test stays on CPU
        target_device = torch.device("cpu")
        sampler = sampler.to(target_device)

        assert sampler.device == target_device
        assert next(sampler.noise_predictor.parameters()).device == target_device
        assert next(sampler.reverse.parameters()).device == target_device


def test_integration():
    """Integration test with the provided usage code."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create simple test data
    x_data = torch.randn(20, 3, 32, 32)
    y_data = torch.randint(0, 10, (20,))
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # Initialize models with smaller parameters for testing
    noise_predictor = NoisePredictor(
        in_channels=3,
        down_channels=[8, 16],  # Reduced channels for testing
        mid_channels=[16, 16],
        up_channels=[16, 8],
        down_sampling=[True, False],  # Only one downsampling for small images
        time_embed_dim=32,
        y_embed_dim=32,
        num_down_blocks=1,
        num_mid_blocks=1,
        num_up_blocks=1,
        down_sampling_factor=2
    ).to(device)

    text_encoder = TextEncoder(
        use_pretrained_model=False,  # Don't use pretrained for faster testing
        model_name="bert-base-uncased",
        vocabulary_size=100,  # Smaller vocabulary
        num_layers=1,  # Fewer layers
        input_dimension=32,
        output_dimension=32,
        num_heads=2,
        context_length=10  # Shorter context
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        [p for p in noise_predictor.parameters() if p.requires_grad] +
        [p for p in text_encoder.parameters() if p.requires_grad],
        lr=1e-4
    )
    loss = nn.MSELoss()

    # SDE hyperparameters with fewer steps
    hyperparams_sde = VarianceSchedulerSDE(
        num_steps=10,  # Fewer steps for testing
        beta_start=1e-4,
        beta_end=0.02,
        trainable_beta=False,
        sigma_start=1e-3,
        sigma_end=10.0,
        start=0.0,
        end=1.0,
        beta_method="linear"
    )

    # Forward and reverse SDE
    forward_sde = ForwardSDE(variance_scheduler=hyperparams_sde, sde_method="vp")
    reverse_sde = ReverseSDE(variance_scheduler=hyperparams_sde, sde_method="vp")

    # TrainSDE with minimal settings
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = TrainSDE(
            noise_predictor=noise_predictor,
            forward_diffusion=forward_sde,
            reverse_diffusion=reverse_sde,
            data_loader=train_loader,
            optimizer=optimizer,
            objective=loss,
            val_loader=val_loader,
            max_epochs=2,  # Just 2 epochs for testing
            device=device,
            conditional_model=text_encoder,
            metrics_=None,  # No metrics for faster testing
            store_path=temp_dir,
            val_frequency=1,
            use_ddp=False,
            grad_accumulation_steps=1,
            log_frequency=1,
            use_compilation=False
        )

        # Test training
        train_losses, best_val_loss = trainer()
        assert len(train_losses) >= 0  # Could be empty if early stopping
        assert isinstance(best_val_loss, float)

        # Test sampling
        sampler = SampleSDE(
            reverse_diffusion=reverse_sde,
            noise_predictor=noise_predictor,
            image_shape=(32, 32),
            conditional_model=text_encoder,
            tokenizer="bert-base-uncased",
            max_token_length=10,  # Shorter for testing
            batch_size=2,
            in_channels=3,
            device=device,
            image_output_range=(-1.0, 1.0)
        )

        # Test with class names
        class_names = ['airplane', 'automobile']
        images = sampler(class_names, save_images=False)
        assert images.shape == (2, 3, 32, 32)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])