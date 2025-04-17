import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel
from tqdm import tqdm
import lpips
from pytorch_fid import fid_score
import os
import shutil
from torchvision.utils import save_image

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

###==================================================================================================================###

class AutoencoderLDM(nn.Module):
    """Variational autoencoder for latent space compression in Latent Diffusion Models.

    Encodes images into a latent space and decodes them back to the image space, used as
    the `compressor_model` in LDM’s `TrainLDM` and `SampleLDM`. Supports KL-divergence
    or vector quantization (VQ) regularization for the latent representation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB images).
    down_channels : list
        List of channel sizes for encoder downsampling blocks (e.g., [32, 64, 128, 256]).
    up_channels : list
        List of channel sizes for decoder upsampling blocks (e.g., [256, 128, 64, 16]).
    out_channels : int
        Number of output channels, typically equal to `in_channels`.
    dropout_rate : float
        Dropout rate for regularization in convolutional and attention layers.
    num_heads : int
        Number of attention heads in self-attention layers.
    num_groups : int
        Number of groups for group normalization in attention layers.
    num_layers_per_block : int
        Number of convolutional layers in each downsampling and upsampling block.
    total_down_sampling_factor : int
        Total downsampling factor across the encoder (e.g., 8 for 8x reduction).
    latent_channels : int
        Number of channels in the latent representation for diffusion models.
    num_embeddings : int
        Number of discrete embeddings in the VQ codebook (if `use_vq=True`).
    use_vq : bool, optional
        If True, uses vector quantization (VQ) regularization; otherwise, uses
        KL-divergence (default: False).
    beta : float, optional
        Weight for KL-divergence loss (if `use_vq=False`) (default: 1.0).

    Attributes
    ----------
    use_vq : bool
        Whether VQ regularization is used.
    beta : float
        Fixed weight for KL-divergence loss.
    current_beta : float
        Current weight for KL-divergence loss (modifiable during training).
    down_sampling_factor : int
        Downsampling factor per block, derived from `total_down_sampling_factor`.
    conv1 : torch.nn.Conv2d
        Initial convolutional layer for encoding.
    down_blocks : torch.nn.ModuleList
        List of DownBlock modules for encoder downsampling.
    attention1 : Attention
        Self-attention layer after encoder downsampling.
    vq_layer : VectorQuantizer or None
        Vector quantization layer (if `use_vq=True`).
    conv_mu : torch.nn.Conv2d or None
        Convolutional layer for mean of latent distribution (if `use_vq=False`).
    conv_logvar : torch.nn.Conv2d or None
        Convolutional layer for log-variance of latent distribution (if `use_vq=False`).
    quant_conv : torch.nn.Conv2d
        Convolutional layer to project latent representation to `latent_channels`.
    conv2 : torch.nn.Conv2d
        Initial convolutional layer for decoding.
    attention2 : Attention
        Self-attention layer after decoder’s initial convolution.
    up_blocks : torch.nn.ModuleList
        List of UpBlock modules for decoder upsampling.
    conv3 : Conv3
        Final convolutional layer for output reconstruction.

    Raises
    ------
    AssertionError
        If `in_channels` does not equal `out_channels`.

    Notes
    -----
    - The encoder downsamples images using `DownBlock` modules, followed by self-attention
      and latent projection (VQ or KL-based).
    - The decoder upsamples the latent representation using `UpBlock` modules, with
      self-attention and final convolution.
    - The `down_sampling_factor` is computed as `total_down_sampling_factor` raised to
      the power of `1 / (len(down_channels) - 1)`, applied per downsampling block.
    - The latent representation has `latent_channels` channels, suitable for LDM’s
      diffusion process.
    """
    def __init__(
            self,
            in_channels,
            down_channels,
            up_channels,
            out_channels,
            dropout_rate,
            num_heads,
            num_groups,
            num_layers_per_block,
            total_down_sampling_factor,
            latent_channels,
            num_embeddings,
            use_vq=False,
            beta=1.0

    ):
        super().__init__()
        assert in_channels == out_channels, "Input and output channels must match for auto-encoding"
        self.use_vq = use_vq
        self.beta = beta
        self.current_beta = beta
        num_down_blocks = len(down_channels) - 1
        self.down_sampling_factor = int(total_down_sampling_factor ** (1 / num_down_blocks))

        # encoder
        self.conv1 = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([
            DownBlock_(
                in_channels=down_channels[i],
                out_channels=down_channels[i + 1],
                num_layers=num_layers_per_block,
                down_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate
            ) for i in range(num_down_blocks)
        ])
        self.attention1 = Attention_(down_channels[-1], num_heads, num_groups, dropout_rate)

        # latent projection
        if use_vq:
            self.vq_layer = VectorQuantizer(num_embeddings, down_channels[-1])
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)
        else:
            self.conv_mu = nn.Conv2d(down_channels[-1], down_channels[-1], kernel_size=3, padding=1)
            self.conv_logvar = nn.Conv2d(down_channels[-1], down_channels[-1], kernel_size=3, padding=1)
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)

        # decoder
        self.conv2 = nn.Conv2d(latent_channels, up_channels[0], kernel_size=3, padding=1)
        self.attention2 = Attention_(up_channels[0], num_heads, num_groups, dropout_rate)
        self.up_blocks = nn.ModuleList([
            UpBlock_(
                in_channels=up_channels[i],
                out_channels=up_channels[i + 1],
                num_layers=num_layers_per_block,
                up_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate
            ) for i in range(len(up_channels) - 1)
        ])
        self.conv3 = Conv3_(up_channels[-1], out_channels, dropout_rate)

    def reparameterize(self, mu, logvar):
        """Applies reparameterization trick for variational autoencoding.

        Samples from a Gaussian distribution using the mean and log-variance to enable
        differentiable training.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution, shape (batch_size, channels, height, width).
        logvar : torch.Tensor
            Log-variance of the latent distribution, same shape as `mu`.

        Returns
        -------
        torch.Tensor
            Sampled latent representation, same shape as `mu`.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encodes images into a latent representation.

        Processes input images through the encoder, applying convolutions, downsampling,
        self-attention, and latent projection (VQ or KL-based).

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (batch_size, in_channels, height, width).

        Returns
        -------
        tuple
            A tuple containing:
            - z: Latent representation, shape (batch_size, latent_channels,
              height/down_sampling_factor, width/down_sampling_factor).
            - reg_loss: Regularization loss (VQ loss if `use_vq=True`, KL-divergence
              loss if `use_vq=False`).

        Notes
        -----
        - The VQ loss is computed by `VectorQuantizer` if `use_vq=True`.
        - The KL-divergence loss is normalized by batch size and latent size, weighted
          by `current_beta`.
        """
        x = self.conv1(x)
        for block in self.down_blocks:
            x = block(x)
        res_x = x
        x = self.attention1(x)
        x = x + res_x
        if self.use_vq:
            z, vq_loss = self.vq_layer(x)
            z = self.quant_conv(z)
            return z, vq_loss
        else:
            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            z = self.reparameterize(mu, logvar)
            z = self.quant_conv(z)
            kl_unnormalized = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            batch_size = x.size(0)
            latent_size = torch.prod(torch.tensor(mu.shape[1:])).item()
            kl_loss = kl_unnormalized / (batch_size * latent_size) * self.current_beta
            return z, kl_loss

    def decode(self, z):
        """Decodes latent representations back to images.

        Processes latent representations through the decoder, applying convolutions,
        self-attention, upsampling, and final reconstruction.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation, shape (batch_size, latent_channels,
            height/down_sampling_factor, width/down_sampling_factor).

        Returns
        -------
        torch.Tensor
            Reconstructed images, shape (batch_size, out_channels, height, width).
        """
        x = self.conv2(z)
        res_x = x
        x = self.attention2(x)
        x = x + res_x
        for block in self.up_blocks:
            x = block(x)
        x = self.conv3(x)
        return x

    def forward(self, x):
        """Encodes images to latent space and decodes them, computing reconstruction and regularization losses.

        Performs a full autoencoding pass, encoding images to the latent space, decoding
        them back, and calculating MSE reconstruction loss and regularization loss (VQ
        or KL-based).

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (batch_size, in_channels, height, width).

        Returns
        -------
        tuple
            A tuple containing:
            - x_hat: Reconstructed images, shape (batch_size, out_channels, height,
              width).
            - total_loss: Sum of reconstruction (MSE) and regularization losses.
            - reg_loss: Regularization loss (VQ or KL-divergence).
            - z: Latent representation, shape (batch_size, latent_channels,
              height/down_sampling_factor, width/down_sampling_factor).

        Notes
        -----
        - The reconstruction loss is computed as the mean squared error between `x_hat`
          and `x`.
        - The regularization loss depends on `use_vq` (VQ loss or KL-divergence).
        """
        z, reg_loss = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_loss + reg_loss
        return x_hat, total_loss, reg_loss, z
#------------------------------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    """Vector quantization layer for discretizing latent representations.

    Quantizes input latent vectors to the nearest embedding in a learned codebook,
    used in `AutoencoderLDM` when `use_vq=True` to enable discrete latent spaces for
    Latent Diffusion Models. Computes commitment and codebook losses to train the
    codebook embeddings.

    Parameters
    ----------
    num_embeddings : int
        Number of discrete embeddings in the codebook.
    embedding_dim : int
        Dimensionality of each embedding vector (matches input channel dimension).
    commitment_cost : float, optional
        Weight for the commitment loss, encouraging inputs to be close to quantized
        values (default: 0.25).

    Attributes
    ----------
    embedding_dim : int
        Dimensionality of embedding vectors.
    num_embeddings : int
        Number of embeddings in the codebook.
    commitment_cost : float
        Weight for commitment loss.
    embedding : torch.nn.Embedding
        Embedding layer containing the codebook, shape (num_embeddings,
        embedding_dim).

    Notes
    -----
    - The codebook embeddings are initialized uniformly in the range
      [-1/num_embeddings, 1/num_embeddings].
    - The forward pass flattens input latents, computes Euclidean distances to
      codebook embeddings, and selects the nearest embedding for quantization.
    - The commitment loss encourages input latents to be close to their quantized
      versions, while the codebook loss updates embeddings to match inputs.
    - A straight-through estimator is used to pass gradients from the quantized output
      to the input.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        """Quantizes latent representations to the nearest codebook embedding.

        Computes the closest embedding for each input vector, applies quantization,
        and calculates commitment and codebook losses for training.

        Parameters
        ----------
        z : torch.Tensor
            Input latent representation, shape (batch_size, embedding_dim, height,
            width).

        Returns
        -------
        tuple
            A tuple containing:
            - quantized: Quantized latent representation, same shape as `z`.
            - vq_loss: Sum of commitment and codebook losses.

        Raises
        ------
        AssertionError
            If the channel dimension of `z` does not match `embedding_dim`.

        Notes
        -----
        - The input is flattened to (batch_size * height * width, embedding_dim) for
          distance computation.
        - Euclidean distances are computed efficiently using vectorized operations.
        - The commitment loss is scaled by `commitment_cost`, and the total VQ loss
          combines commitment and codebook losses.
        """
        z = z.contiguous()
        assert z.size(1) == self.embedding_dim, f"Expected channel dim {self.embedding_dim}, got {z.size(1)}"
        z_flattened = z.reshape(-1, self.embedding_dim)
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float().squeeze(1)
        quantized = torch.matmul(encodings, self.embedding.weight).view_as(z)
        commitment_loss = self.commitment_cost * torch.mean((z.detach() - quantized) ** 2)
        codebook_loss = torch.mean((z - quantized.detach()) ** 2)
        quantized = z + (quantized - z).detach()
        return quantized, commitment_loss + codebook_loss
#------------------------------------------------------------------------------------------------
class DownBlock_(nn.Module):
    """Downsampling block for the encoder in AutoencoderLDM.

    Applies multiple convolutional layers with residual connections followed by
    downsampling to reduce spatial dimensions in the encoder of the variational
    autoencoder used in Latent Diffusion Models.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for convolutional layers.
    num_layers : int
        Number of convolutional layer pairs (Conv3) per block.
    down_sampling_factor : int
        Factor by which to downsample spatial dimensions.
    dropout_rate : float
        Dropout rate for Conv3 layers.

    Attributes
    ----------
    num_layers : int
        Number of convolutional layer pairs.
    conv1 : torch.nn.ModuleList
        List of Conv3 layers for the first convolution in each pair.
    conv2 : torch.nn.ModuleList
        List of Conv3 layers for the second convolution in each pair.
    down_sampling : DownSampling
        Downsampling module to reduce spatial dimensions.
    resnet : torch.nn.ModuleList
        List of 1x1 convolutional layers for residual connections.

    Notes
    -----
    - Each layer pair consists of two Conv3 modules with a residual connection using a
      1x1 convolution to match dimensions.
    - The downsampling is applied after all convolutional layers, reducing spatial
      dimensions by `down_sampling_factor`.
    """
    def __init__(self, in_channels, out_channels, num_layers, down_sampling_factor, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3_(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3_(
                in_channels=out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])

        self.down_sampling = DownSampling_(
            in_channels=out_channels,
            out_channels=out_channels,
            down_sampling_factor=down_sampling_factor
        )
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers)

        ])

    def forward(self, x):
        """Processes input through convolutional layers and downsampling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch_size, out_channels,
            height/down_sampling_factor, width/down_sampling_factor).
        """
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
        output = self.down_sampling(output)
        return output
# ------------------------------------------------------------------------------------------------
class Conv3_(nn.Module):
    """Convolutional layer with group normalization, SiLU activation, and dropout.

    Used in DownBlock and UpBlock of AutoencoderLDM for feature extraction and
    transformation in the encoder and decoder.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : float
        Dropout rate for regularization.

    Attributes
    ----------
    group_norm : torch.nn.GroupNorm
        Group normalization with 8 groups.
    activation : torch.nn.SiLU
        SiLU (Swish) activation function.
    conv : torch.nn.Conv2d
        3x3 convolutional layer with padding to maintain spatial dimensions.
    dropout : torch.nn.Dropout
        Dropout layer for regularization.

    Notes
    -----
    - The layer applies group normalization, SiLU activation, dropout, and a 3x3
      convolution in sequence.
    - Spatial dimensions are preserved due to padding=1 in the convolution.
    """
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.activation = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """Processes input through group normalization, activation, dropout, and convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch_size, out_channels, height, width).
        """
        x = self.group_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x
#------------------------------------------------------------------------------------------------
class DownSampling_(nn.Module):
    """Downsampling module for reducing spatial dimensions in AutoencoderLDM’s encoder.

    Combines convolutional downsampling and max pooling, concatenating their outputs
    to preserve feature information during downsampling in DownBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (sum of conv and pool paths).
    down_sampling_factor : int
        Factor by which to downsample spatial dimensions.

    Attributes
    ----------
    down_sampling_factor : int
        Downsampling factor.
    conv : torch.nn.Sequential
        Convolutional path with 1x1 and 3x3 convolutions, outputting out_channels/2.
    pool : torch.nn.Sequential
        Max pooling path with 1x1 convolution, outputting out_channels/2.

    Notes
    -----
    - The module splits the output channels evenly between convolutional and pooling
      paths, concatenating them along the channel dimension.
    - The convolutional path uses a stride equal to `down_sampling_factor`, while the
      pooling path uses max pooling with the same factor.
    """
    def __init__(self, in_channels, out_channels, down_sampling_factor):
        super().__init__()
        self.down_sampling_factor = down_sampling_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                      kernel_size=3, stride=down_sampling_factor, padding=1)
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=down_sampling_factor, stride=down_sampling_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                      kernel_size=1, stride=1, padding=0)
        )

    def forward(self, batch):
        """Downsamples input by combining convolutional and pooling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Downsampled tensor, shape (batch_size, out_channels,
            height/down_sampling_factor, width/down_sampling_factor).
        """
        return torch.cat(tensors=[self.conv(batch), self.pool(batch)], dim=1)
#------------------------------------------------------------------------------------------------
class Attention_(nn.Module):
    """Self-attention module for feature enhancement in AutoencoderLDM.

    Applies multi-head self-attention to enhance features in the encoder and decoder,
    used after downsampling (in DownBlock) and before upsampling (in UpBlock).

    Parameters
    ----------
    num_channels : int
        Number of input and output channels (embedding dimension for attention).
    num_heads : int
        Number of attention heads.
    num_groups : int
        Number of groups for group normalization.
    dropout_rate : float
        Dropout rate for attention outputs.

    Attributes
    ----------
    group_norm : torch.nn.GroupNorm
        Group normalization before attention.
    attention : torch.nn.MultiheadAttention
        Multi-head self-attention with `batch_first=True`.
    dropout : torch.nn.Dropout
        Dropout layer for regularization.

    Notes
    -----
    - The input is reshaped to (batch_size, height * width, num_channels) for
      attention processing, then restored to (batch_size, num_channels, height, width).
    - Group normalization is applied before attention to stabilize training.
    """
    def __init__(self, num_channels, num_heads, num_groups, dropout_rate):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """Applies self-attention to input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, num_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor, same shape as input.
        """
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h * w)
        x = self.group_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x
#------------------------------------------------------------------------------------------------
class UpBlock_(nn.Module):
    """Upsampling block for the decoder in AutoencoderLDM.

    Applies upsampling followed by multiple convolutional layers with residual
    connections to increase spatial dimensions in the decoder of the variational
    autoencoder used in Latent Diffusion Models.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for convolutional layers.
    num_layers : int
        Number of convolutional layer pairs (Conv3) per block.
    up_sampling_factor : int
        Factor by which to upsample spatial dimensions.
    dropout_rate : float
        Dropout rate for Conv3 layers.

    Attributes
    ----------
    num_layers : int
        Number of convolutional layer pairs.
    up_sampling : UpSampling
        Upsampling module to increase spatial dimensions.
    conv1 : torch.nn.ModuleList
        List of Conv3 layers for the first convolution in each pair.
    conv2 : torch.nn.ModuleList
        List of Conv3 layers for the second convolution in each pair.
    resnet : torch.nn.ModuleList
        List of 1x1 convolutional layers for residual connections.

    Notes
    -----
    - Upsampling is applied first, followed by convolutional layer pairs with residual
      connections using 1x1 convolutions.
    - Each layer pair consists of two Conv3 modules.
    """
    def __init__(self, in_channels, out_channels, num_layers, up_sampling_factor, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        effective_in_channels = in_channels

        self.up_sampling = UpSampling_(
            in_channels=in_channels,
            out_channels=in_channels,
            up_sampling_factor=up_sampling_factor
        )

        self.conv1 = nn.ModuleList([
            Conv3_(
                in_channels=effective_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3_(
                in_channels=out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=effective_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(self.num_layers)
        ])

    def forward(self, x):
        """Processes input through upsampling and convolutional layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch_size, out_channels,
            height * up_sampling_factor, width * up_sampling_factor).
        """
        x = self.up_sampling(x)
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
        return output
#------------------------------------------------------------------------------------------------
class UpSampling_(nn.Module):
    """Upsampling module for increasing spatial dimensions in AutoencoderLDM’s decoder.

    Combines transposed convolution and nearest-neighbor upsampling, concatenating
    their outputs to preserve feature information during upsampling in UpBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (sum of conv and upsample paths).
    up_sampling_factor : int
        Factor by which to upsample spatial dimensions.

    Attributes
    ----------
    up_sampling_factor : int
        Upsampling factor.
    conv : torch.nn.Sequential
        Transposed convolutional path, outputting out_channels/2.
    up_sample : torch.nn.Sequential
        Nearest-neighbor upsampling path with 1x1 convolution, outputting
        out_channels/2.

    Notes
    -----
    - The module splits the output channels evenly between transposed convolution and
      upsampling paths, concatenating them along the channel dimension.
    - If the spatial dimensions of the two paths differ, the upsampling path is
      interpolated to match the convolutional path’s size.
    """
    def __init__(self, in_channels, out_channels, up_sampling_factor):
        super().__init__()
        half_out_channels = out_channels // 2
        self.up_sampling_factor = up_sampling_factor
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=half_out_channels,
                kernel_size=3,
                stride=up_sampling_factor,
                padding=1,
                output_padding=up_sampling_factor - 1
            ),
            nn.Conv2d(
                in_channels=half_out_channels,
                out_channels=half_out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=up_sampling_factor, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=half_out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, batch):
        """Upsamples input by combining transposed convolution and upsampling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Upsampled tensor, shape (batch_size, out_channels,
            height * up_sampling_factor, width * up_sampling_factor).

        Notes
        -----
        - Interpolation is applied if the spatial dimensions of the convolutional and
          upsampling paths differ, using nearest-neighbor mode.
        """
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
#------------------------------------------------------------------------------------------------
class TrainAE:
    """Trainer for the variational autoencoder in Latent Diffusion Models.

    Manages training of the `AutoencoderLDM` compressor model, optimizing for
    reconstruction and regularization losses (KL-divergence or VQ), with optional
    perceptual loss, metrics (MSE, PSNR, SSIM, FID), KL warmup, early stopping, and
    learning rate scheduling.

    Parameters
    ----------
    model : AutoencoderLDM
        The variational autoencoder model to train (compressor model for LDM).
    optimizer : torch.optim.Optimizer
        Optimizer for training the model.
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data, yielding (images, labels) batches.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data (default: None).
    max_epoch : int, optional
        Maximum number of training epochs (default: 100).
    device : str, optional
        Device for training (e.g., 'cuda', 'cpu') (default: 'cuda').
    save_path : str, optional
        File path to save the best model checkpoint (default: 'vlc_model.pth').
    checkpoint : int, optional
        Frequency (in epochs) to save model checkpoints (default: 10).
    kl_warmup_epochs : int, optional
        Number of epochs for KL-divergence loss warmup (default: 10).
    patience : int, optional
        Number of epochs to wait for early stopping if validation loss does not
        improve (default: 10).
    per_loss : bool, optional
        Whether to include perceptual loss using LPIPS (default: True).
    metrics : bool, optional
        Whether to compute MSE, PSNR, and SSIM metrics (default: True).
    fid : bool, optional
        Whether to compute FID score (default: False).
    perceptual_weight : float, optional
        Weight for the perceptual loss term (default: 0.1).

    Attributes
    ----------
    model : AutoencoderLDM
        The autoencoder model being trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    data_loader : torch.utils.data.DataLoader
        Training DataLoader.
    val_loader : torch.utils.data.DataLoader or None
        Validation DataLoader, if provided.
    max_epoch : int
        Maximum training epochs.
    device : str
        Training device.
    save_path : str
        Path for saving model checkpoints.
    checkpoint : int
        Epoch frequency for saving checkpoints.
    kl_warmup_epochs : int
        Epochs for KL loss warmup.
    patience : int
        Epochs for early stopping patience.
    per_loss : bool
        Flag for perceptual loss computation.
    metrics : bool
        Flag for MSE, PSNR, SSIM computation.
    fid : bool
        Flag for FID computation.
    perceptual_loss : lpips.LPIPS
        LPIPS model for perceptual loss (VGG backbone).
    perceptual_weight : float
        Weight for perceptual loss.
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler based on validation loss.
    temp_dir_real : str
        Temporary directory for real images during FID computation.
    temp_dir_fake : str
        Temporary directory for fake (reconstructed) images during FID computation.

    Notes
    -----
    - The total loss includes reconstruction (MSE), regularization (KL or VQ), and
      optional perceptual (LPIPS) losses.
    - KL warmup linearly increases the KL loss weight (`model.current_beta`) from 0 to
      `model.beta` over `kl_warmup_epochs` when `model.use_vq=False`.
    - Metrics (MSE, PSNR, SSIM) and FID are computed if enabled, with FID requiring
      temporary disk storage for images.
    - Early stopping is based on validation loss (or training loss if `val_loader` is
      None), and the learning rate is reduced if validation loss plateaus.
    - The model is saved when the best validation (or training) loss is achieved.
    """
    def __init__(self, model, optimizer, data_loader, val_loader=None, max_epoch=100,
                 device="cuda", save_path="vlc_model.pth", checkpoint=10, kl_warmup_epochs=10,
                 patience=10, per_loss=True, metrics=True, fid=False, perceptual_weight=0.1):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.device = device
        self.save_path = save_path
        self.checkpoint = checkpoint
        self.kl_warmup_epochs = kl_warmup_epochs
        self.patience = patience
        self.per_loss = per_loss
        self.metrics = metrics
        self.fid = fid
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(device)
        self.perceptual_weight = perceptual_weight
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.temp_dir_real = "temp_real"
        self.temp_dir_fake = "temp_fake"

    def compute_metrics(self, x, x_hat):
        """Computes image quality metrics (MSE, PSNR, SSIM) for reconstructed images.

        Parameters
        ----------
        x : torch.Tensor
            Ground truth images, shape (batch_size, channels, height, width).
        x_hat : torch.Tensor
            Reconstructed images, same shape as `x`.

        Returns
        -------
        dict
            Dictionary containing:
            - mse: Mean squared error (float).
            - psnr: Peak signal-to-noise ratio (float).
            - ssim: Structural similarity index (float, mean over batch).
        """
        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse)
        c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
        mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(x_hat, kernel_size=3, stride=1, padding=1)
        mu_xy = mu_x * mu_y
        sigma_x_sq = F.avg_pool2d(x.pow(2), kernel_size=3, stride=1, padding=1) - mu_x.pow(2)
        sigma_y_sq = F.avg_pool2d(x_hat.pow(2), kernel_size=3, stride=1, padding=1) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(x * x_hat, kernel_size=3, stride=1, padding=1) - mu_xy
        ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x_sq + sigma_y_sq + c2))
        return {"mse": mse.item(), "psnr": psnr.item(), "ssim": ssim.mean().item()}


    def compute_fid(self, real_images, fake_images):
        """Computes the Fréchet Inception Distance (FID) between real and reconstructed images.

        Saves images to temporary directories and uses the Inception V3 model to compute
        FID, cleaning up directories afterward.

        Parameters
        ----------
        real_images : torch.Tensor
            Real images, shape (batch_size, channels, height, width), in [-1, 1] range.
        fake_images : torch.Tensor
            Reconstructed images, same shape, in [-1, 1] range.

        Returns
        -------
        float
            FID score, or `float('inf')` if computation fails.

        Notes
        -----
        - Images are normalized to [0, 1] and saved as PNG files for FID computation.
        - The Inception V3 model uses 2048-dimensional features (`dims=2048`).
        - Temporary directories (`temp_dir_real`, `temp_dir_fake`) are created and
          removed automatically.
        """
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

    def train(self):
        """Trains the autoencoder model for the specified number of epochs.

        Optimizes the model using training data, with optional validation, metrics
        computation, and FID scoring. Saves the best model based on validation (or
        training) loss.

        Returns
        -------
        tuple
            A tuple containing:
            - train_losses: List of mean training losses per epoch.
            - best_val_loss: Best validation (or training) loss achieved.

        Notes
        -----
        - The training loss includes reconstruction, regularization, and optional
          perceptual losses.
        - KL warmup adjusts `model.current_beta` for KL-divergence loss if
          `model.use_vq=False`.
        - Early stopping halts training if the best loss does not improve for
          `patience` epochs.
        - The learning rate is adjusted via `scheduler` based on validation loss.
        - Metrics and FID are computed if enabled via `metrics` and `fid` flags.
        """
        self.model.train()
        self.model.to(self.device)
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(self.max_epoch):
            if self.model.use_vq:
                beta = 1.0
            else:
                beta = min(1.0, epoch / self.kl_warmup_epochs) * self.model.beta
                self.model.current_beta = beta

            train_losses_ = []
            metrics_epoch = {"mse": [], "psnr": [], "ssim": []}
            all_real, all_fake = [], []

            for x, _ in tqdm(self.data_loader):
                x = x.to(self.device)
                x_hat, total_loss, reg_loss, z = self.model(x)
                if self.per_loss:
                    percep_loss = self.perceptual_loss(x_hat, x).mean()
                    loss = total_loss + self.perceptual_weight * percep_loss
                else:
                    loss = total_loss
                train_losses_.append(loss.item())

                if self.metrics or self.fid:
                    with torch.no_grad():
                        if self.metrics:
                            batch_metrics = self.compute_metrics(x, x_hat)
                            for k, v in batch_metrics.items():
                                metrics_epoch[k].append(v)
                        if self.fid:
                            all_real.append(x)
                            all_fake.append(x_hat)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            mean_train_loss = torch.mean(torch.tensor(train_losses_)).item()
            train_losses.append(mean_train_loss)
            metrics_summary = {k: sum(v) / len(v) for k, v in metrics_epoch.items()} if self.metrics else {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
            fid = self.compute_fid(torch.cat(all_real), torch.cat(all_fake)) if self.fid and all_real else float('inf')

            print(f"\nEpoch: {epoch + 1} | Loss: {mean_train_loss:.4f} | Reg Weight: {beta:.4f}", end="")
            if self.metrics:
                print(f" | PSNR: {metrics_summary['psnr']:.2f} | SSIM: {metrics_summary['ssim']:.4f}", end="")
            if self.fid:
                print(f" | FID: {fid:.2f}", end="")
            print()
            if self.val_loader is not None:
                val_loss, val_metrics, val_fid = self.validate()
                print(f"Val Loss: {val_loss:.4f}", end="")
                if self.metrics:
                    print(f" | Val PSNR: {val_metrics['psnr']:.2f} | Val SSIM: {val_metrics['ssim']:.4f}", end="")
                if self.fid:
                    print(f" | Val FID: {val_fid:.2f}", end="")
                print()
                current_best = val_loss
                self.scheduler.step(val_loss)
            else:
                current_best = mean_train_loss

            if current_best < best_val_loss:
                best_val_loss = current_best
                wait = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                }, self.save_path)
                print(f"Model saved at epoch {epoch + 1}")
            else:
                wait += 1
                if wait >= self.patience:
                    print("Early stopping triggered")
                    break

        return train_losses, best_val_loss

    def validate(self):
        """Evaluates the model on the validation dataset.

        Computes validation loss, metrics (MSE, PSNR, SSIM), and FID score without
        updating model parameters.

        Returns
        -------
        tuple
            A tuple containing:
            - mean_val_loss: Mean validation loss (float).
            - metrics_summary: Dictionary of mean MSE, PSNR, SSIM (or zeros if
              `metrics=False`).
            - fid: FID score (or `float('inf')` if `fid=False` or computation fails).
        """
        self.model.eval()
        val_losses = []
        metrics_val = {"mse": [], "psnr": [], "ssim": []}
        all_real, all_fake = [], []

        with torch.no_grad():
            for x, _ in self.val_loader:
                x = x.to(self.device)
                x_hat, total_loss, reg_loss, z = self.model(x)
                if self.per_loss:
                    percep_loss = self.perceptual_loss(x_hat, x).mean()
                    loss = total_loss + self.perceptual_weight * percep_loss
                else:
                    loss = total_loss
                val_losses.append(loss.item())
                if self.metrics or self.fid:
                    if self.metrics:
                        batch_metrics = self.compute_metrics(x, x_hat)
                        for k, v in batch_metrics.items():
                            metrics_val[k].append(v)
                    if self.fid:
                        all_real.append(x)
                        all_fake.append(x_hat)

        mean_val_loss = torch.mean(torch.tensor(val_losses)).item()
        metrics_summary = {k: sum(v) / len(v) for k, v in metrics_val.items()} if self.metrics else {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
        fid = self.compute_fid(torch.cat(all_real), torch.cat(all_fake)) if self.fid and all_real else float('inf')

        self.model.train()
        return mean_val_loss, metrics_summary, fid

###==================================================================================================================###

class NoisePredictor(nn.Module):
    def __init__(
            self,
            in_channels,
            down_channels,
            mid_channels,
            up_channels,
            down_sampling,
            time_embed_dim,
            y_embed_dim, # output embedding dimension in text conditional net
            num_down_blocks,
            num_mid_blocks,
            num_up_blocks,
            dropout_rate=0.1,
            down_sampling_factor=2,
            where_y=True,
            y_to_all=False
    ):
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

    def initialize_weights(self):
        """Initialize model weights for better training stability"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, t, y=None):

        if not self.where_y and y is not None:
            x = torch.cat(tensors=[x, y], dim=1)
        output = self.conv1(x)
        time_embed = GetEmbeddedTime(embed_dim=self.time_embed_dim)(time_steps=t)
        time_embed = self.time_projection(time_embed)
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
#-----------------------------------------------------------------------------
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, y_embed_dim,num_layers, down_sampling_factor,  down_sample, dropout_rate, y_to_all):
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
                y_embed_dim= y_embed_dim,
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

    def forward(self, x, embed_time, y):
        #print("down-block input shape:", x.size())
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = output + self.time_embedding[i](embed_time)[:, :, None, None]
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
            if y is not None and not self.y_to_all and i == 0:
                out_attn = self.attention[i](output, y)
                output = output + out_attn
            elif y is not None and self.y_to_all:
                out_attn = self.attention[i](output, y)
                output = output + out_attn
            elif y is None and self.y_to_all:
                out_attn = self.attention[i](output)
                output = output + out_attn
            elif y is None and not self.y_to_all and i == 0:
                out_attn = self.attention[i](output)
                output = output + out_attn

        output = self.down_sampling(output)
        #print("down-block output shape:", output.size())
        return output
#------------------------------------------------------------------------------
class MiddleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim,  y_embed_dim, num_layers, dropout_rate, y_to_all=False):
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

    def forward(self, x, embed_time, y=None):
        #print("mid-input shape:", x.size())
        output = x
        resnet_input = output
        output = self.conv1[0](output)
        output = output + self.time_embedding[0](embed_time)[:, :, None, None]
        output = self.conv2[0](output)
        output = output + self.resnet[0](resnet_input)
        for i in range(self.num_layers):
            if y is not None and not self.y_to_all and i == 0:
                out_attn = self.attention[i](output, y)
                output = output + out_attn
            elif y is not None and self.y_to_all:
                out_attn = self.attention[i](output, y)
                output = output + out_attn
            elif y is None and self.y_to_all:
                out_attn = self.attention[i](output)
                output = output + out_attn
            elif y is None and not self.y_to_all and i == 0:
                out_attn = self.attention[i](output)
                output = output + out_attn
            resnet_input = output
            output = self.conv1[i + 1](output)
            output = output + self.time_embedding[i + 1](embed_time)[:, :, None, None]
            output = self.conv2[i + 1](output)
            output = output + self.resnet[i+1](resnet_input)
        #print("mid-block output shape:", output.size())

        return output
#------------------------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, time_embed_dim,  y_embed_dim, num_layers, up_sampling_factor, up_sampling=True, dropout_rate=0.2, y_to_all=False):
        super().__init__()
        self.num_layers = num_layers
        self.y_to_all = y_to_all
        effective_in_channels = in_channels//2 + skip_channels
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
        self.up_sampling = UpSampling(
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

    def forward(self, x, skip_connection, embed_time, y=None):
        #print("up-block input shape:", x.size())
        x = self.up_sampling(x)
        x = torch.cat(tensors=[x, skip_connection], dim=1)
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = output + self.time_embedding[i](embed_time)[:, :, None, None]
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
            if y is not None and not self.y_to_all and i == 0:
                out_attn = self.attention[i](output, y)
                output = output + out_attn
            elif y is not None and self.y_to_all:
                out_attn = self.attention[i](output, y)
                output = output + out_attn
            elif y is None and self.y_to_all:
                out_attn = self.attention[i](output)
                output = output + out_attn
            elif y is None and not self.y_to_all and i == 0:
                out_attn = self.attention[i](output)
                output = output + out_attn
        #print("up-block output shape:", output.size())
        return output
#------------------------------------------------------------------------
class Conv3(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8, kernel_size=3, norm=True, activation=True, dropout_rate=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels) if norm else nn.Identity()
        self.activation = nn.SiLU() if activation else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, batch):
        batch = self.conv(batch)
        batch = self.group_norm(batch)
        batch = self.activation(batch)
        batch = self.dropout(batch)
        return batch
#----------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, output_dim, embed_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=embed_dim, out_features=output_dim)
        )
    def forward(self, batch):
        return self.embedding(batch)
#----------------------------------------------------------------
class GetEmbeddedTime(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        assert embed_dim % 2 == 0, "The embedding dimension must be divisible by two"
        self.embed_dim = embed_dim

    def forward(self, time_steps):
        i = torch.arange(start=0, end=self.embed_dim // 2, dtype=torch.float32, device=time_steps.device)
        factor = 10000 ** (2 * i / self.embed_dim)
        embed_time = time_steps[:, None] / factor
        embed_time = torch.cat(tensors=[torch.sin(embed_time), torch.cos(embed_time)], dim=-1)
        return embed_time
#----------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, in_channels, y_embed_dim=768, num_heads=4, num_groups=8, dropout_rate=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.y_embed_dim = y_embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.y_projection = nn.Linear(y_embed_dim, in_channels)

    def forward(self, x, y=None):
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
#-----------------------------------------------------------------
class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling_factor, conv_block=True, max_pool=True):
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

    def forward(self, batch):
        if not self.conv_block:
            return self.pool(batch)
        if not self.max_pool:
            return self.conv(batch)
        return torch.cat(tensors=[self.conv(batch), self.pool(batch)], dim=1)
#--------------------------------------------------------------------------
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, up_sampling_factor, conv_block=True, up_sampling=True):
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

    def forward(self, batch):
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