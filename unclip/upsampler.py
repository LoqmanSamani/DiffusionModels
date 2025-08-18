import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple




class UpsamplerUnCLIP(nn.Module):
    """Diffusion-based upsampler for UnCLIP models.

    A U-Net-like model that upsamples low-resolution images to high-resolution images,
    conditioned on noisy high-resolution images and timesteps, using residual blocks,
    downsampling, and upsampling layers.

    Parameters
    ----------
    `forward_diffusion` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise during training.
    `in_channels` : int, optional
        Number of input channels (default: 3, for RGB images).
    `out_channels` : int, optional
        Number of output channels (default: 3, for RGB noise prediction).
    `model_channels` : int, optional
        Base number of channels in the model (default: 192).
    `num_res_blocks` : int, optional
        Number of residual blocks per resolution level (default: 2).
    `channel_mult` : Tuple[int, ...], optional
        Channel multiplier for each resolution level (default: (1, 2, 4, 8)).
    `dropout` : float, optional
        Dropout probability for regularization (default: 0.1).
    `time_embed_dim` : int, optional
        Dimensionality of time embeddings (default: 768).
    `low_res_size` : int, optional
        Spatial size of low-resolution input (default: 64).
    `high_res_size` : int, optional
        Spatial size of high-resolution output (default: 256).
    """

    def __init__(
            self,
            forward_diffusion: nn.Module,
            reverse_diffusion: nn.Module,
            in_channels: int = 3,
            out_channels: int = 3,
            model_channels: int = 192,
            num_res_blocks: int = 2,
            channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
            dropout_rate: float = 0.1,
            time_embed_dim: int = 768,
            low_res_size: int = 64,
            high_res_size: int = 256,
    ) -> None:
        super().__init__()

        self.forward_diffusion = forward_diffusion # this will be used on training time inside 'TrainUpsamplerUnCLIP'
        self.reverse_diffusion = reverse_diffusion # this module will be used in inference time
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input projection
        # Concatenate noisy high-res and upsampled low-res
        self.input_proj = nn.Conv2d(in_channels * 2, model_channels, 3, padding=1)

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock(ch, model_channels * mult, time_embed_dim, dropout_rate)
                )
                ch = model_channels * mult

            if level != len(channel_mult) - 1:
                self.downsample_blocks.append(DownsampleBlock(ch, ch))

        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResBlock(ch, ch, time_embed_dim, dropout_rate),
            ResBlock(ch, ch, time_embed_dim, dropout_rate),
        ])

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                # Skip connections double the input channels
                in_ch = ch + (model_channels * mult if i == 0 else 0)
                out_ch = model_channels * mult

                self.decoder_blocks.append(
                    ResBlock(in_ch, out_ch, time_embed_dim, dropout_rate)
                )
                ch = out_ch

            if level != 0:
                self.upsample_blocks.append(UpsampleBlock(ch, ch))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x_high: torch.Tensor, t: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        """Predicts noise for the upsampling process.

        Processes a noisy high-resolution image and a low-resolution conditioning image,
        conditioned on timesteps, to predict the noise component for denoising.

        Parameters
        ----------
        `x_high` : torch.Tensor
            Noisy high-resolution image, shape (batch_size, in_channels, high_res_size, high_res_size).
        `t` : torch.Tensor
            Timestep indices, shape (batch_size,).
        `x_low` : torch.Tensor
            Low-resolution conditioning image, shape (batch_size, in_channels, low_res_size, low_res_size).

        Returns
        -------
        out : torch.Tensor
            Predicted noise, shape (batch_size, out_channels, high_res_size, high_res_size).
        """
        # Upsample low-resolution image to match high-resolution
        x_low_upsampled = F.interpolate(
            x_low,
            size=(x_high.shape[-2], x_high.shape[-1]),
            mode='bicubic',
            align_corners=False
        )
        # print(f"After upsampling x_low: shape={x_low_upsampled.shape}, dtype={x_low_upsampled.dtype}")

        # Concatenate noisy high-res and upsampled low-res
        x = torch.cat([x_high, x_low_upsampled], dim=1)
        # print(f"After concatenating x_high and x_low_upsampled: shape={x.shape}, dtype={x.dtype}")

        # Time embedding
        time_emb = self.time_embed(t.float())  # Ensure float for embedding
        # print(f"After time embedding: shape={time_emb.shape}, dtype={time_emb.dtype}")

        # Input projection
        h = self.input_proj(x)
        # print(f"After input projection: shape={h.shape}, dtype={h.dtype}")

        # Store skip connections
        skip_connections = []

        # Encoder
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, time_emb)
            # print(f"After encoder block {i + 1}: shape={h.shape}, dtype={h.dtype}")
            if (i + 1) % self.num_res_blocks == 0:
                skip_connections.append(h)
                # print(f"Saved skip connection {len(skip_connections)}: shape={h.shape}, dtype={h.dtype}")
                downsample_idx = (i + 1) // self.num_res_blocks - 1
                if downsample_idx < len(self.downsample_blocks):
                    h = self.downsample_blocks[downsample_idx](h)
                    # print(f"After downsample {downsample_idx + 1}: shape={h.shape}, dtype={h.dtype}")

        # Middle
        for i, block in enumerate(self.middle_blocks):
            h = block(h, time_emb)
            # print(f"After middle block {i + 1}: shape={h.shape}, dtype={h.dtype}")

        # Decoder
        upsample_idx = 0
        for i, block in enumerate(self.decoder_blocks):
            # Add skip connection
            if i % (self.num_res_blocks + 1) == 0 and skip_connections:
                skip = skip_connections.pop()
                # print(f"Using skip connection {len(skip_connections) + 1}: shape={skip.shape}, dtype={skip.dtype}")
                h = torch.cat([h, skip], dim=1)
                # print(f"After concatenating skip connection: shape={h.shape}, dtype={h.dtype}")

            h = block(h, time_emb)
            # print(f"After decoder block {i + 1}: shape={h.shape}, dtype={h.dtype}")

            # Upsample at the end of each resolution level
            if ((i + 1) % (self.num_res_blocks + 1) == 0 and
                    upsample_idx < len(self.upsample_blocks)):
                h = self.upsample_blocks[upsample_idx](h)
                # print(f"After upsample {upsample_idx + 1}: shape={h.shape}, dtype={h.dtype}")
                upsample_idx += 1

        # Output projection
        out = self.output_proj(h)
        # print(f"After output projection: shape={out.shape}, dtype={out.dtype}")

        return out



class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps.

    Generates sinusoidal embeddings for timesteps to condition the upsampler on the
    diffusion process stage.

    Parameters
    ----------
    `dim` : int
        Dimensionality of the embedding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generates sinusoidal embeddings for timesteps.

        Parameters
        ----------
        `timesteps` : torch.Tensor
            Timestep indices, shape (batch_size,).

        Returns
        -------
        embeddings : torch.Tensor
            Sinusoidal embeddings, shape (batch_size, dim).
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time embedding and conditioning.

    A convolutional residual block with group normalization, time embedding conditioning,
    and optional scale-shift normalization, used in the UnCLIP upsampler.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels.
    `out_channels` : int
        Number of output channels.
    `time_embed_dim` : int
        Dimensionality of time embeddings.
    `dropout` : float, optional
        Dropout probability (default: 0.1).
    `use_scale_shift_norm` : bool, optional
        Whether to use scale-shift normalization for time embeddings (default: True).
    """
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int,
                 dropout: float = 0.1, use_scale_shift_norm: bool = True):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels * 2 if use_scale_shift_norm else out_channels)
        )

        # Changed: Separated the out_norm from the rest of out_layers to avoid slicing issues with nn.Sequential.
        # Original would raise TypeError because nn.Sequential[1:] does not return a callable Sequential and cannot be directly invoked.
        self.out_norm = nn.GroupNorm(8, out_channels)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Processes input through the residual block with time conditioning.

        Parameters
        ----------
        `x` : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).
        `time_emb` : torch.Tensor
            Time embeddings, shape (batch_size, time_embed_dim).

        Returns
        -------
        out : torch.Tensor
            Output tensor, shape (batch_size, out_channels, height, width).
        """
        h = self.in_layers(x)

        # Apply time embedding
        emb_out = self.time_emb_proj(time_emb)[:, :, None, None]

        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            # Changed: Use self.out_norm instead of self.out_layers[0].
            h = self.out_norm(h) * (1 + scale) + shift
            # Changed: Use self.out_rest instead of self.out_layers[1:].
            h = self.out_rest(h)
        else:
            h = h + emb_out
            # Changed: Apply out_norm and out_rest consistently.
            h = self.out_norm(h)
            h = self.out_rest(h)

        return h + self.skip_connection(x)


class UpsampleBlock(nn.Module):
    """Upsampling block using transposed convolution.

    Increases the spatial resolution of the input tensor using a transposed convolution.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels.
    `out_channels` : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsamples the input tensor.

        Parameters
        ----------
        `x` : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        out : torch.Tensor
            Upsampled tensor, shape (batch_size, out_channels, height*2, width*2).
        """
        return self.conv(x)


class DownsampleBlock(nn.Module):
    """Downsampling block using strided convolution.

    Reduces the spatial resolution of the input tensor using a strided convolution.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels.
    `out_channels` : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsamples the input tensor.

        Parameters
        ----------
        `x` : torch.Tensor
            Input tensor, shape (batch_size, in_channels, height, width).

        Returns
        -------
        out : torch.Tensor
            Downsampled tensor, shape (batch_size, out_channels, height//2, width//2).
        """
        return self.conv(x)


"""
hyp = VarianceSchedulerUnCLIP(
    num_steps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=False,
    beta_method="cosine"
)

forward = ForwardUnCLIP(hyp)

model = UpsamplerUnCLIP(
    forward_diffusion=forward,
    in_channels= 3,
    out_channels= 3,
    model_channels= 32,
    num_res_blocks = 2,
    channel_mult = (1, 2, 4, 8),
    dropout = 0.1,
    time_embed_dim  = 756,
    low_res_size = 256,
    high_res_size = 1024
)
xl = torch.randn((2, 3, 256, 256))
xh = torch.randn((2, 3, 1024, 1024))
t = torch.tensor([3, 5])


result = model(xh, t, xl)
print(result.size())
print(result.dtype)
"""