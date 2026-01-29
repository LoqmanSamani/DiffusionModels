import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Union
import math





class UnCLIPTransformerPrior(nn.Module):
    """Transformer-based prior model for UnCLIP diffusion.

    Predicts clean image embeddings from noisy image embeddings and text embeddings using
    a Transformer architecture, incorporating time embeddings and optional projection
    layers for text and image inputs.

    Parameters
    ----------
    `fwd_unclip` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise during training.
    `rwd_unclip` : nn.Module
        Reverse diffusion module (e.g., ReverseUnCLIP) for denoising during training.
    `clip_text_proj` : nn.Module, optional
        Projection module for text embeddings, default None.
    `clip_img_proj` : nn.Module, optional
        Projection module for image embeddings, default None.
    `trans_embed_dim` : int, optional
        Dimensionality of embeddings (default: 320).
    `num_layers` : int, optional
        Number of Transformer layers (default: 12).
    `num_att_heads` : int, optional
        Number of attention heads in each Transformer layer (default: 8).
    `ff_dim` : int, optional
        Dimensionality of the feedforward network in Transformer layers (default: 768).
    `max_sequence_length` : int, optional
        Maximum sequence length for input embeddings (default: 2).
    `dropoute` : float, optional
        Dropout probability for regularization (default: 0.2).
    `use_flash`: bool, optional
        Enable flash attention if available (default: True).
    `grad_check`: bool, optional
        Apply gradinet checkpointing (default: False).
    `check_every_n_layers`: int, optional
        Frequency of applying gradient checkpoint (default: 2 layers)
    """
    def __init__(
            self,
            fwd_unclip: nn.Module, # will be used during training
            rwd_unclip: nn.Module, # will be used during training
            clip_text_proj: Optional[nn.Module] = None,  # used during training instead of PCA in the main paper
            clip_img_proj: Optional[nn.Module] = None,  # used during training instead of PCA in the main paper
            trans_embed_dim: int = 320,
            num_layers: int = 12,
            num_att_heads: int = 8,
            ff_dim: int = 768,
            max_sequence_length: int = 2,
            dropout: float = 0.2,
            use_flash: bool = True,
            grad_check: bool = False,
            check_every_n_layers: int = 2
    ) -> None:
        super().__init__()

        self.fwd_unclip = fwd_unclip
        self.rwd_unclip = rwd_unclip
        self.clip_text_proj = clip_text_proj
        self.clip_img_proj = clip_img_proj
        self.trans_embed_dim = trans_embed_dim
        self.max_sequence_length = max_sequence_length
        self.grad_check = grad_check
        self.check_every_n_layers = check_every_n_layers
        self.use_flash = use_flash and self._check_flash_attention()
        # time embedding network
        self.time_embed_net = nn.Sequential(
            nn.Linear(trans_embed_dim, trans_embed_dim),
            nn.GELU(),
            nn.Linear(trans_embed_dim, trans_embed_dim)
        )
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(max_sequence_length, trans_embed_dim))
        # transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(trans_embed_dim, num_att_heads, ff_dim, dropout, self.use_flash)
            for _ in range(num_layers)
        ])
        # final output projection
        self.out_proj = nn.Linear(trans_embed_dim, trans_embed_dim)
        # cache for sinusoidal embeddings (reuse across batches)
        self._cached_sinusoidal_embeds = {}

    def forward(
            self,
            text_embed: torch.Tensor,
            noisy_img_embed: torch.Tensor,
            timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Predicts clean image embeddings from noisy inputs and text embeddings.

        Processes text and noisy image embeddings through a Transformer architecture,
        conditioned on time embeddings, to predict the clean image embeddings.

        Parameters
        ----------
        `text_embed` : torch.Tensor
            Text embeddings, shape (batch_size, embed_dim).
        `noisy_img_embed` : torch.Tensor
            Noisy image embeddings, shape (batch_size, embed_dim).
        `timesteps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,).

        Returns
        -------
        pred_clean_embed : torch.Tensor
            Predicted clean image embeddings, shape (batch_size, embed_dim).
        """
        device = text_embed.device
        # create sinusoidal time embeddings
        time_embed = self._sinusoidal_embed_cached(timesteps, self.trans_embed_dim, device)
        time_embed = self.time_embed_net(time_embed)
        # add time information to image embeddings
        cond_img_embed = noisy_img_embed + time_embed
        # create sequence: [text_embed, cond_img_embed]
        seq = torch.stack([text_embed, cond_img_embed], dim=1)  # [B, 2, D]
        # add positional embeddings
        seq = seq + self.pos_embed.unsqueeze(0)
        # pass through transformer blocks
        if self.grad_check and self.training:
            seq = self._forward_with_check(seq)
        else:
            for transformer_block in self.transformer_blocks:
                seq = transformer_block(seq)
        # extract predicted clean image embedding (second position in sequence)
        pred_clean_embed = seq[:, 1, :]  # [B, D]
        # apply final projection
        pred_clean_embed = self.out_proj(pred_clean_embed)

        return pred_clean_embed

    def _forward_with_check(self, seq: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing every N layers"""
        for i in range(0, len(self.transformer_blocks), self.check_every_n_layers):
            end_idx = min(i + self.check_every_n_layers, len(self.transformer_blocks))
            layers_to_checkpoint = self.transformer_blocks[i:end_idx]
            def create_forward_func(layers):
                def forward_func(x):
                    for layer in layers:
                        x = layer(x)
                    return x
                return forward_func
            # apply checkpointing
            seq = checkpoint(
                create_forward_func(layers_to_checkpoint),
                seq,
                use_reentrant=False
            )
        return seq

    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            if hasattr(nn.functional, 'scaled_dot_product_attention'):
                return True
        except:
            pass
        return False
    def enable_grad_check(self):
        """Enable gradient checkpointing for memory savings"""
        self.use_grad_check = True

    def disable_grad_check(self):
        """Disable gradient checkpointing"""
        self.use_grad_check = False

    def _sinusoidal_embed_cached(
            self,
            timesteps: torch.Tensor,
            embed_dim: int,
            device: Union[torch.device, str]
    ) -> torch.Tensor:
        """Generates sinusoidal positional embeddings with caching.

        Caches the sinusoidal embedding computation to avoid recomputation
        for the same timesteps across different batches.
        """
        max_timestep = timesteps.max().item()
        cache_key = (embed_dim, device, max_timestep)
        if cache_key not in self._cached_sinusoidal_embeds:
            half_dim = embed_dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            all_timesteps = torch.arange(max_timestep + 1, device=device).float()
            emb = all_timesteps[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            if embed_dim % 2 == 1:
                emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
            self._cached_sinusoidal_embeds[cache_key] = emb
        cached_emb = self._cached_sinusoidal_embeds[cache_key]
        return cached_emb[timesteps]

    def _sinusoidal_embed(
            self,
            timesteps: torch.Tensor,
            embed_dim: int,
            device: Union[torch.device, str]
    ) -> torch.Tensor:
        """Generates sinusoidal positional embeddings for timesteps.

        Creates sinusoidal embeddings for the given timesteps to condition the Transformer
        on the diffusion process time steps.

        Parameters
        ----------
        `timesteps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,).
        `embed_dim` : int
            Dimensionality of the embeddings.
        `device` : Union[torch.device, str]
            Device to place the embeddings on.

        Returns
        -------
        emb : torch.Tensor
            Sinusoidal time embeddings, shape (batch_size, embed_dim).
        """
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # handle odd embedding dimensions
        if embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

        return emb


class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head attention and feedforward layers.

    Implements a Transformer block with multi-head self-attention, layer normalization,
    and a feedforward network with residual connections for processing sequences in
    the UnCLIPTransformerPrior model.

    Parameters
    ----------
    `embed_dim` : int
        Dimensionality of input and output embeddings.
    `num_heads` : int
        Number of attention heads in the multi-head attention layer.
    `ff_dim` : int
        Dimensionality of the feedforward network.
    `dropout` : float
        Dropout probability for regularization.
    `use_falsh`: bool
        Whethere use flash attention (default: True)
    """
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout: float,
            use_flash: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_flash = use_flash

        self.att_norm = nn.LayerNorm(embed_dim)
        self.ff_norm = nn.LayerNorm(embed_dim)

        # multi-head attention
        if use_flash and hasattr(nn.functional, 'scaled_dot_product_attention'):
            # use manual qkv projection for flash attention
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.dropout_p = dropout
        else:
            # fall back to standard MultiheadAttention
            self.self_att = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                batch_first=True
            )
        # feed forward net
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input sequence through the Transformer block.

        Applies multi-head self-attention followed by a feedforward network, with residual
        connections and layer normalization.

        Parameters
        ----------
        `x` : torch.Tensor
            Input sequence tensor, shape (batch_size, sequence_length, embed_dim).

        Returns
        -------
        `x` : torch.Tensor
            Processed sequence tensor, shape (batch_size, sequence_length, embed_dim).
        """
        n_x = self.att_norm(x)
        if self.use_flash and hasattr(nn.functional, 'scaled_dot_product_attention'):
            att_out = self._flash_attention(n_x)
        else:
            att_out, _ = self.self_att(n_x, n_x, n_x)
        x = x + att_out
        n_x = self.ff_norm(x)
        ff_out = self.ff(n_x)
        x = x + ff_out
        return x

    def _flash_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Flash Attention

        Parameters
        ----------
        `x` : torch.Tensor
            Input tensor, shape (batch_size, seq_len, embed_dim).

        Returns
        -------
        att_out : torch.Tensor
            Attention output, shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape
        # project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, S, 3*D]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.embed_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D//H]
        q, k, v = qkv[0], qkv[1], qkv[2]
        att_out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        att_out = att_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        att_out = self.out_proj(att_out)
        return att_out

class FusedGELU(nn.Module):
    """Fused GELU activation for better efficiency on some hardware"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use approximate GELU for speed
        return nn.functional.gelu(x, approximate='tanh')