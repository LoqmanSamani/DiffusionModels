import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


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