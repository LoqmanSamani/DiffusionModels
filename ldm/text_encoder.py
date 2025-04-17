import torch
import torch.nn as nn
import math
from transformers import BertModel, DistilBertModel




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

    Attributes
    ----------
    use_pretrained_model : bool
        Whether a pre-trained model is used.
    bert : transformers.BertModel or None
        Pre-trained BERT model, if `use_pretrained_model` is True.
    projection : torch.nn.Linear or None
        Linear layer to project BERT outputs to `output_dimension`, if
        `use_pretrained_model` is True.
    embedding : Embedding or None
        Token and positional embedding layer for the custom transformer, if
        `use_pretrained_model` is False.
    layers : torch.nn.ModuleList or None
        List of EncoderLayer modules for the custom transformer, if
        `use_pretrained_model` is False.

    Notes
    -----
    - When `use_pretrained_model` is True, the BERT model’s parameters are frozen
      (`requires_grad = False`), and a projection layer maps outputs to
      `output_dimension`.
    - The custom transformer uses `EncoderLayer` modules with multi-head attention and
      feedforward networks, supporting variable input/output dimensions.
    - The output shape is (batch_size, context_length, output_dimension).
    """
    def __init__(
            self,
            use_pretrained_model=True,
            model_name="bert-base-uncased",
            vocabulary_size=30522,
            num_layers=6,
            input_dimension=768,
            output_dimension=768,
            num_heads=8,
            context_length=77,
            dropout_rate=0.1,
            qkv_bias=False,
            scaling_value=4,
            epsilon=1e-5
    ):
        super().__init__()
        self.use_pretrained_model = use_pretrained_model
        if self.use_pretrained_model:
            # self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.bert = BertModel.from_pretrained(model_name)
            for param in self.bert.parameters():
                param.requires_grad = False
            self.projection = nn.Linear(self.bert.config.hidden_size, output_dimension)
        else:
            self.embedding = Embedding(
                vocabulary_size=vocabulary_size,
                embedding_dimension=input_dimension,
                context_length=context_length
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
    def forward(self, x, attention_mask=None):
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
        torch.Tensor
            Encoded embeddings, shape (batch_size, seq_len, output_dimension).

        Notes
        -----
        - For pre-trained BERT, the `last_hidden_state` is projected to
          `output_dimension`.
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
#-----------------------------------------------------------------------------------------------------------------------
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

    Attributes
    ----------
    attention : torch.nn.MultiheadAttention
        Multi-head self-attention mechanism.
    output_projection : torch.nn.Linear or torch.nn.Identity
        Linear layer to project attention outputs to `output_dimension`, or identity
        if `input_dimension` equals `output_dimension`.
    norm1 : torch.nn.LayerNorm
        Layer normalization after attention.
    dropout1 : torch.nn.Dropout
        Dropout after attention.
    feedforward : FeedForward
        Feedforward network.
    norm2 : torch.nn.LayerNorm
        Layer normalization after feedforward.
    dropout2 : torch.nn.Dropout
        Dropout after feedforward.

    Notes
    -----
    - The layer follows the standard transformer encoder architecture: attention,
      residual connection, normalization, feedforward, residual connection,
      normalization.
    - The attention mechanism uses `batch_first=True` for compatibility with
      `TextEncoder`’s input format.
    """
    def __init__(
            self,
            input_dimension,
            output_dimension,
            num_heads,
            dropout_rate,
            qkv_bias,
            scaling_value,
            epsilon=1e-5
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dimension,
            num_heads=num_heads,
            dropout=dropout_rate,
            bias=qkv_bias,
            batch_first=True
        )
        self.output_projection = nn.Linear(input_dimension, output_dimension) if input_dimension != output_dimension else nn.Identity()
        self.norm1 = nn.LayerNorm(normalized_shape=input_dimension, eps=epsilon)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.feedforward = FeedForward(
            embedding_dimension=input_dimension,
            scaling_value=scaling_value,
            dropout_rate=dropout_rate
        )
        self.norm2 = nn.LayerNorm(normalized_shape=output_dimension, eps=epsilon)
        self.dropout2 = nn.Dropout(dropout_rate)
    def forward(self, x, attention_mask=None):
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
        torch.Tensor
            Processed embeddings, shape (batch_size, seq_len, output_dimension).

        Notes
        -----
        - The attention mask is passed as `key_padding_mask` to
          `nn.MultiheadAttention`, where 0 indicates padding tokens.
        - Residual connections and normalization are applied after attention and
          feedforward layers.
        """
        attn_output, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        attn_output = self.output_projection(attn_output)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x
#-----------------------------------------------------------------------------------------------------------------------
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

    Attributes
    ----------
    layers : torch.nn.Sequential
        Sequential container with linear, GELU, dropout, and linear layers.

    Notes
    -----
    - The hidden layer dimension is `embedding_dimension * scaling_value`, following
      standard transformer feedforward designs.
    - GELU activation is used for non-linearity.
    """
    def __init__(self, embedding_dimension, scaling_value, dropout_rate=0.1):
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
    def forward(self, x):
        """Processes input embeddings through the feedforward network.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings, shape (batch_size, seq_len, embedding_dimension).

        Returns
        -------
        torch.Tensor
            Processed embeddings, shape (batch_size, seq_len, embedding_dimension).
        """
        return self.layers(x)
#-----------------------------------------------------------------------------------------------------------------------
class Embedding(torch.nn.Module):
    """Token and positional embedding layer for transformer inputs.

    Used in `TextEncoder`’s custom transformer to embed token IDs and add positional
    encodings.

    Parameters
    ----------
    vocabulary_size : int
        Size of the vocabulary for token embeddings.
    embedding_dimension : int, optional
        Dimension of token and positional embeddings (default: 768).
    context_length : int, optional
        Maximum sequence length for positional encodings (default: 77).

    Attributes
    ----------
    token_embedding : torch.nn.Embedding
        Token embedding layer.
    embedding_dimension : int
        Dimension of embeddings.
    context_length : int
        Maximum sequence length.
    positional_encoding : torch.Tensor
        Pre-computed positional encodings, shape (1, context_length,
        embedding_dimension).

    Notes
    -----
    - Positional encodings are computed using sinusoidal functions, following the
      transformer architecture.
    - For sequences longer than `context_length`, positional encodings are dynamically
      generated.
    - The output shape is (batch_size, seq_len, embedding_dimension).
    """
    def __init__(
        self,
        vocabulary_size,
        embedding_dimension=768,
        context_length=77
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension
        )
        self.embedding_dimension = embedding_dimension
        self.context_length = context_length
        self.register_buffer("positional_encoding", self._generate_positional_encoding(context_length))

    def _generate_positional_encoding(self, seq_len):
        """Generates sinusoidal positional encodings for transformer inputs.

        Computes positional encodings using sine and cosine functions, following the
        transformer architecture, to represent token positions in a sequence.

        Parameters
        ----------
        seq_len : int
            Length of the sequence for which to generate positional encodings.

        Returns
        -------
        torch.Tensor
            Positional encodings, shape (1, seq_len, embedding_dimension), where
            even-indexed dimensions use sine and odd-indexed dimensions use cosine.

        Notes
        -----
        - The encoding follows the formula: for position `pos` and dimension `i`,
          `PE(pos, 2i) = sin(pos / 10000^(2i/d))` and
          `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`, where `d` is
          `embedding_dimension`.
        - The output is unsqueezed to include a batch dimension for compatibility with
          token embeddings.
        - The tensor is created on the same device as the input positions for
          compatibility with the model’s device.
        """
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dimension, 2, dtype=torch.float) *
                             -(math.log(10000.0) / self.embedding_dimension))
        pos_enc = torch.zeros((seq_len, self.embedding_dimension), device=position.device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, token_ids):
        """Embeds token IDs and adds positional encodings.

        Parameters
        ----------
        token_ids : torch.Tensor
            Token IDs, shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Embedded tokens with positional encodings, shape (batch_size, seq_len,
            embedding_dimension).

        Raises
        ------
        AssertionError
            If `token_ids` is not a 2D tensor (batch_size, seq_len).
        """
        assert token_ids.dim() == 2, "Input token_ids should be of shape (batch_size, seq_len)"
        token_embedded = self.token_embedding(token_ids)
        seq_len = token_ids.size(1)
        if seq_len > self.context_length:
            position_encoded = self._generate_positional_encoding(seq_len).to(token_embedded.device)
        else:
            position_encoded = self.positional_encoding[:, :seq_len, :].to(token_embedded.device)
        return token_embedded + position_encoded