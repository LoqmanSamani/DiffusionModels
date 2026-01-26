import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch.utils.checkpoint as checkpoint



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
    """
    def __init__(
            self,
            in_channels: int,
            down_channels: List[int],
            up_channels: List[int],
            out_channels: int,
            dropout_rate: float,
            num_heads: int,
            num_groups: int,
            num_layers_per_block: int,
            total_down_sampling_factor: int,
            latent_channels: int,
            num_embeddings: int,
            use_vq: bool = False,
            beta: float = 1.0,
            use_flash: bool = True,
            use_grad_check: bool = False,
            *args
    ) -> None:
        super().__init__()
        assert in_channels == out_channels, "Input and output channels must match for auto-encoding"
        self.use_vq = use_vq
        self.beta = beta
        self.current_beta = beta
        self.use_flash = use_flash
        self.use_grad_check = use_grad_check
        num_down_blocks = len(down_channels) - 1
        self.down_sampling_factor = int(total_down_sampling_factor ** (1 / num_down_blocks))

        # encoder
        self.conv1 = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels=down_channels[i],
                out_channels=down_channels[i + 1],
                num_layers=num_layers_per_block,
                down_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate,
                use_grad_check=self.use_grad_check
            ) for i in range(num_down_blocks)
        ])
        self.attention1 = Attention(down_channels[-1], num_heads, num_groups, dropout_rate, use_flash)

        # latent projection
        if use_vq:
            self.vq_layer = VectorQuantizer(num_embeddings, down_channels[-1])
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)
        else:
            self.conv_mu_logvar = nn.Conv2d(down_channels[-1], down_channels[-1] * 2, kernel_size=3, padding=1)
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)

        # decoder
        self.conv2 = nn.Conv2d(latent_channels, up_channels[0], kernel_size=3, padding=1)
        self.attention2 = Attention(up_channels[0], num_heads, num_groups, dropout_rate, use_flash)
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=up_channels[i],
                out_channels=up_channels[i + 1],
                num_layers=num_layers_per_block,
                up_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate,
                use_grad_check=use_grad_check
            ) for i in range(len(up_channels) - 1)
        ])
        self.conv3 = Conv3(up_channels[-1], out_channels, dropout_rate)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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
        reparam (torch.Tensor) - Sampled latent representation, same shape as `mu`.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Encodes images into a latent representation.

        Processes input images through the encoder, applying convolutions, downsampling,
        self-attention, and latent projection (VQ or KL-based).

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (batch_size, in_channels, height, width).

        Returns
        -------
        z : (torch.Tensor)
            Latent representation, shape (batch_size, latent_channels, height/down_sampling_factor, width/down_sampling_factor).
        reg_loss : float
            Regularization loss (VQ loss if `use_vq=True`, KL-divergence loss if `use_vq=False`).

        **Notes**

        - The VQ loss is computed by `VectorQuantizer` if `use_vq=True`.
        - The KL-divergence loss is normalized by batch size and latent size, weighted
          by `current_beta`.
        """
        x = self.conv1(x)
        for block in self.down_blocks:
            x = block(x)

        if self.use_grad_check and self.training:
            x = x + checkpoint.checkpoint(self.attention1, x, use_reentrant=False)
        else:
            x = x + self.attention1(x)
        if self.use_vq:
            z, vq_loss = self.vq_layer(x)
            z = self.quant_conv(z)
            return z, vq_loss
        else:
            mu_logvar = self.conv_mu_logvar(x)
            mu, logvar = mu_logvar.chunk(2, dim=1)
            z = self.reparameterize(mu, logvar)
            z = self.quant_conv(z)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * self.current_beta
            return z, kl_loss

    def decode(self, z: torch.Tensor) -> torch.Tensor:
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
        x (torch.Tensor) - Reconstructed images, shape (batch_size, out_channels, height, width).
        """
        x = self.conv2(z)
        if self.use_grad_check and self.training:
            x = x + checkpoint.checkpoint(self.attention2, x, use_reentrant=False)
        else:
            x = x + self.attention2(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv3(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float, torch.Tensor]:
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
        x_hat : torch.Tensor
            Reconstructed images, shape (batch_size, out_channels, height, width).
        total_loss : float
            Sum of reconstruction (MSE) and regularization losses.
        reg_loss : float
            Regularization loss (VQ or KL-divergence).
        z : torch.Tensor
            Latent representation, shape (batch_size, latent_channels, height/down_sampling_factor, width/down_sampling_factor).

        **Notes**

        - The reconstruction loss is computed as the mean squared error between `x_hat` and `x`.
        - The regularization loss depends on `use_vq` (VQ loss or KL-divergence).
        """
        z, reg_loss = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_loss.item() + reg_loss
        return x_hat, total_loss, reg_loss, z



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
        Weight for the commitment loss, encouraging inputs to be close to quantized values (default: 0.25).

    **Notes**

    - The codebook embeddings are initialized uniformly in the range [-1/num_embeddings, 1/num_embeddings].
    - The forward pass flattens input latents, computes Euclidean distances to codebook embeddings, and selects the nearest embedding for quantization.
    - The commitment loss encourages input latents to be close to their quantized versions, while the codebook loss updates embeddings to match inputs.
    - A straight-through estimator is used to pass gradients from the quantized output to the input.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        quantized : torch.Tensor
            Quantized latent representation, same shape as `z`.
        vq_loss : torch.Tensor
            Sum of commitment and codebook losses.

        **Notes**

        - The input is flattened to (batch_size * height * width, embedding_dim) for distance computation.
        - Euclidean distances are computed efficiently using vectorized operations.
        - The commitment loss is scaled by `commitment_cost`, and the total VQ loss combines commitment and codebook losses.
        """
        batch_size, channels, height, width = z.shape
        assert channels == self.embedding_dim, f"Expected channel dim {self.embedding_dim}, got {channels}"
        z_flattened = z.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        z_sq = torch.sum(z_flattened ** 2, dim=1, keepdim=True)
        e_sq = torch.sum(self.embedding.weight ** 2, dim=1)
        distances = z_sq + e_sq - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        commitment_loss = self.commitment_cost * F.mse_loss(z.detach(), quantized)
        codebook_loss = F.mse_loss(z, quantized.detach())
        quantized = z + (quantized - z).detach()
        return quantized, commitment_loss + codebook_loss

class DownBlock(nn.Module):
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

    **Notes**

    - Each layer pair consists of two Conv3 modules with a residual connection using a 1x1 convolution to match dimensions.
    - The downsampling is applied after all convolutional layers, reducing spatial dimensions by `down_sampling_factor`.
    """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
                 down_sampling_factor: int, dropout_rate: float, use_grad_check: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.use_grad_check = use_grad_check
        self.res_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResidualBlock(in_ch, out_channels, dropout_rate)
            )
        self.down_sampling = DownSampling(out_channels, out_channels, down_sampling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through convolutional layers and downsampling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        output (torch.Tensor) - Output tensor, shape (batch_size, out_channels, height/down_sampling_factor, width/down_sampling_factor).
        """
        for block in self.res_blocks:
            if self.use_grad_check and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.down_sampling(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block"""
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = Conv3(in_channels, out_channels, dropout_rate)
        self.conv2 = Conv3(out_channels, out_channels, dropout_rate)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through residual connection"""
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class Conv3(nn.Module):
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

    **Notes**

    - The layer applies group normalization, SiLU activation, dropout, and a 3x3 convolution in sequence.
    - Spatial dimensions are preserved due to padding=1 in the convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through group normalization, activation, dropout, and convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        x (torch.Tensor) - Output tensor, shape (batch_size, out_channels, height, width).
        """
        x = self.group_norm(x)
        x = F.silu(x)  # In-place SiLU
        x = self.dropout(x)
        x = self.conv(x)
        return x


class DownSampling(nn.Module):
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

    **Notes**

    - The module splits the output channels evenly between convolutional and pooling paths, concatenating them along the channel dimension.
    - The convolutional path uses a stride equal to `down_sampling_factor`, while the pooling path uses max pooling with the same factor.
    """
    def __init__(self, in_channels: int, out_channels: int, down_sampling_factor: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=down_sampling_factor,
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsamples input by combining convolutional and pooling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        x (torch.Tensor) - Downsampled tensor, shape (batch_size, out_channels, height/down_sampling_factor, width/down_sampling_factor).
        """
        return self.conv(x)


class Attention(nn.Module):
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

    **Notes**

    - The input is reshaped to (batch_size, height * width, num_channels) for attention processing, then restored to (batch_size, num_channels, height, width).
    - Group normalization is applied before attention to stabilize training.
    """
    def __init__(self, num_channels: int, num_heads: int, num_groups: int,
                 dropout_rate: float, use_flash: bool = True) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.use_flash = use_flash

        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

        if use_flash and hasattr(F, 'scaled_dot_product_attention'):
            self.qkv = nn.Linear(num_channels, num_channels * 3)
            self.proj = nn.Linear(num_channels, num_channels)
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=num_channels,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout_rate
            )
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies self-attention to input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, num_channels, height, width).

        Returns
        -------
        x (torch.Tensor) - Output tensor, same shape as input.
        """
        batch_size, channels, h, w = x.shape
        x_norm = self.group_norm(x)
        x_norm = x_norm.reshape(batch_size, channels, h * w).transpose(1, 2)
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            qkv = self.qkv(x_norm).reshape(batch_size, h * w, 3, self.num_heads, channels // self.num_heads)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, h * w, channels)
            x_attn = self.proj(attn_output)
        else:
            x_attn, _ = self.attention(x_norm, x_norm, x_norm)
        x_attn = self.dropout(x_attn)
        x_attn = x_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x_attn


class UpBlock(nn.Module):
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

    **Notes**

    - Upsampling is applied first, followed by convolutional layer pairs with residual connections using 1x1 convolutions.
    - Each layer pair consists of two Conv3 modules.
    """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
                 up_sampling_factor: int, dropout_rate: float, use_grad_check: bool = False) -> None:
        super().__init__()
        self.up_sampling = UpSampling(in_channels, in_channels, up_sampling_factor)
        self.use_grad_check = use_grad_check
        self.res_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResidualBlock(in_ch, out_channels, dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through upsampling and convolutional layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        output (torch.Tensor) - Output tensor, shape (batch_size, out_channels, height * up_sampling_factor, width * up_sampling_factor).
        """
        x = self.up_sampling(x)
        for block in self.res_blocks:
            if self.use_grad_check and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class UpSampling(nn.Module):
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

    **Notes**

    - The module splits the output channels evenly between transposed convolution and upsampling paths, concatenating them along the channel dimension.
    - If the spatial dimensions of the two paths differ, the upsampling path is interpolated to match the convolutional path’s size.
    """
    def __init__(self, in_channels: int, out_channels: int, up_sampling_factor: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=up_sampling_factor,
            padding=1,
            output_padding=up_sampling_factor - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsamples input by combining transposed convolution and upsampling paths.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        x (torch.Tensor) - Upsampled tensor, shape
        (batch_size, out_channels, height * up_sampling_factor, width * up_sampling_factor).

        **Notes**

        - Interpolation is applied if the spatial dimensions of the
          convolutional and upsampling paths differ, using nearest-neighbor mode.
        """
        return self.conv(x)