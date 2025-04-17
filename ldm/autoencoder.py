import torch
import torch.nn as nn
import torch.nn.functional as F




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
            in_channels,  # number of channels of the original image. e.g., 3 for RBG.
            down_channels,  # a list of channels used in encoder. e.g., [32, 64, 128, 256].
            up_channels,  # a list of channels used in decoder. e.g., [256, 128, 64, 16].
            out_channels,  # probably the same as in_channels. used to construct the image.
            dropout_rate,  # dropout rate, prevents overfitting.
            num_heads,  # number of attention heads in self-attention layers.
            num_groups,  # number of groups in group normalization. used in self-attention.
            num_layers_per_block, # number of convolutional layers within each down/up block.
            total_down_sampling_factor,  # total down-sampling factor, used to calculate down sampling factor: an integer used to down/up sample the input batch of images.
            latent_channels,  # final z channels for DM.
            num_embeddings, # number of discrete embeddings in the codebook/dimensionality of each embedding vector. in case of using VectorQuantizer
            use_vq=False, # flag to toggle between vq regularization and kl regularization; if false, uses kl.
            beta=1.0 # weight for KL loss.

    ):
        super().__init__()
        assert in_channels == out_channels, "Input and output channels must match for auto-encoding"
        self.use_vq = use_vq
        self.beta = beta
        self.current_beta = beta
        num_down_blocks = len(down_channels) - 1
        self.down_sampling_factor = int(total_down_sampling_factor ** (1 / num_down_blocks))

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels=down_channels[i],
                out_channels=down_channels[i + 1],
                num_layers=num_layers_per_block,
                down_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate
            ) for i in range(num_down_blocks)
        ])
        self.attention1 = Attention(down_channels[-1], num_heads, num_groups, dropout_rate)

        # Latent projection
        if use_vq:
            self.vq_layer = VectorQuantizer(num_embeddings, down_channels[-1])
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)
        else:
            self.conv_mu = nn.Conv2d(down_channels[-1], down_channels[-1], kernel_size=3, padding=1)
            self.conv_logvar = nn.Conv2d(down_channels[-1], down_channels[-1], kernel_size=3, padding=1)
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)

        # Decoder
        self.conv2 = nn.Conv2d(latent_channels, up_channels[0], kernel_size=3, padding=1)
        self.attention2 = Attention(up_channels[0], num_heads, num_groups, dropout_rate)
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=up_channels[i],
                out_channels=up_channels[i + 1],
                num_layers=num_layers_per_block,
                up_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate
            ) for i in range(len(up_channels) - 1)
        ])
        self.conv3 = Conv3(up_channels[-1], out_channels, dropout_rate)

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
        return x_hat, total_loss, reg_loss, z  # return z for DM
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
        # dimensionality of each embedding vector
        self.embedding_dim = embedding_dim
        # number of discrete embeddings in the codebook
        self.num_embeddings = num_embeddings
        # commitment cost for the loss term to encourage z to be close to quantized values
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # initialize embedding weights uniformly
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
        z = z.contiguous() # ensure contingency in memory
        # flatten z to (batch_size * height * width, embedding_dim) for distance computation
        assert z.size(1) == self.embedding_dim, f"Expected channel dim {self.embedding_dim}, got {z.size(1)}"
        z_flattened = z.reshape(-1, self.embedding_dim)
        # compute squared euclidean distances between z_flattened and all embeddings
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))
        # find the index of the closest embedding for each z_flattened vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # convert indices to one-hot encodings
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float().squeeze(1)
        # map one-hot encodings to quantized values using the embedding weights
        quantized = torch.matmul(encodings, self.embedding.weight).view_as(z)
        # commitment loss to encourage z to be close to its quantized version
        commitment_loss = self.commitment_cost * torch.mean((z.detach() - quantized) ** 2)
        # codebook loss to encourage embeddings to move closer to z
        codebook_loss = torch.mean((z - quantized.detach()) ** 2)
        # straight-through estimator: copy gradients from quantized to z
        quantized = z + (quantized - z).detach()
        # return the quantized tensor and the combined vq loss
        return quantized, commitment_loss + codebook_loss
#------------------------------------------------------------------------------------------------
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
            Conv3(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])

        self.down_sampling = DownSampling(
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

        self.up_sampling = UpSampling(
            in_channels=in_channels,
            out_channels=in_channels,
            up_sampling_factor=up_sampling_factor
        )

        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=effective_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
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