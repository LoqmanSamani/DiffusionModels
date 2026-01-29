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
    `fwd_unclip` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise during training.
    `rwd_unclip` : nn.Module
        Reverse diffusion module (e.g., ReverseUnCLIP) for removing noise during sampling.
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
            fwd_unclip: nn.Module,
            rwd_unclip: nn.Module,
            in_channels: int = 3,
            out_channels: int = 3,
            model_channels: int = 192,
            num_res_blocks: int = 2,
            channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
            dropout: float = 0.1,
            time_embed_dim: int = 768,
            low_res_size: int = 64,
            high_res_size: int = 256,
    ) -> None:
        super().__init__()

        self.fwd_unclip = fwd_unclip # this will be used on training time inside 'TrainUpsamplerUnCLIP'
        self.rwd_unclip = rwd_unclip # this module will be used in inference time
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size

        # time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # input projection
        # concatenate noisy high-res and upsampled low-res
        self.input_proj = nn.Conv2d(in_channels * 2, model_channels, 3, padding=1)

        # encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock(ch, model_channels * mult, time_embed_dim, dropout)
                )
                ch = model_channels * mult

            if level != len(channel_mult) - 1:
                self.down_blocks.append(DownsampleBlock(ch, ch))
        # middle blocks
        self.mid_blocks = nn.ModuleList([
            ResBlock(ch, ch, time_embed_dim, dropout),
            ResBlock(ch, ch, time_embed_dim, dropout),
        ])
        # decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                # skip connections double the input channels
                in_ch = ch + (model_channels * mult if i == 0 else 0)
                out_ch = model_channels * mult
                self.decoder_blocks.append(
                    ResBlock(in_ch, out_ch, time_embed_dim, dropout)
                )
                ch = out_ch
            if level != 0:
                self.up_blocks.append(UpsampleBlock(ch, ch))
        # output projection
        self.out_proj = nn.Sequential(
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
        # upsample low-resolution image to match high-resolution
        x_low_up = F.interpolate(
            x_low,
            size=(x_high.shape[-2], x_high.shape[-1]),
            mode='bicubic',
            align_corners=False
        )
        # concatenate noisy high-res and upsampled low-res
        x = torch.cat([x_high, x_low_up], dim=1)
        # time embedding
        time_emb = self.time_embed(t.float())
        # input projection
        h = self.input_proj(x)
        # store skip connections
        skip_cons = []
        # encoder
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, time_emb)
            if (i + 1) % self.num_res_blocks == 0:
                skip_cons.append(h)
                down_idx = (i + 1) // self.num_res_blocks - 1
                if down_idx < len(self.down_blocks):
                    h = self.down_blocks[down_idx](h)
        # middle
        for i, block in enumerate(self.mid_blocks):
            h = block(h, time_emb)
        # decoder
        up_idx = 0
        for i, block in enumerate(self.decoder_blocks):
            # add skip connection
            if i % (self.num_res_blocks + 1) == 0 and skip_cons:
                skip = skip_cons.pop()
                h = torch.cat([h, skip], dim=1)
            h = block(h, time_emb)
            # upsample at the end of each resolution level
            if ((i + 1) % (self.num_res_blocks + 1) == 0 and
                    up_idx < len(self.up_blocks)):
                h = self.up_blocks[up_idx](h)
                up_idx += 1
        # output projection
        out = self.out_proj(h)
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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
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
        device = t.device
        half_dim = self.dim // 2
        embeds = math.log(10000) / (half_dim - 1)
        embeds = torch.exp(torch.arange(half_dim, device=device) * -embeds)
        embeds = t[:, None] * embeds[None, :]
        embeds = torch.cat([torch.sin(embeds), torch.cos(embeds)], dim=-1)
        return embeds

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

        self.out_norm = nn.GroupNorm(8, out_channels)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if in_channels != out_channels:
            self.skip_con = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_con = nn.Identity()

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
        # apply time embedding
        emb_out = self.time_emb_proj(time_emb)[:, :, None, None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.out_norm(h) * (1 + scale) + shift
            h = self.out_rest(h)
        else:
            h = h + emb_out
            h = self.out_norm(h)
            h = self.out_rest(h)
        return h + self.skip_con(x)


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