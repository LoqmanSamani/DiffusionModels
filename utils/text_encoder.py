import torch
import torch.nn as nn
from transformers import BertModel
import math
from typing import Optional



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