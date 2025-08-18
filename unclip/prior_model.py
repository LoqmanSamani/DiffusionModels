import torch
import torch.nn as nn
import math
from typing import Union, Optional


class UnCLIPTransformerPrior(nn.Module):
    """Transformer-based prior model for UnCLIP diffusion.

    Predicts clean image embeddings from noisy image embeddings and text embeddings using
    a Transformer architecture, incorporating time embeddings and optional projection
    layers for text and image inputs.

    Parameters
    ----------
    `forward_diffusion` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise during training.
    `reverse_diffusion` : nn.Module
        Reverse diffusion module (e.g., ReverseUnCLIP) for denoising during training.
    `clip_text_projection` : nn.Module, optional
        Projection module for text embeddings, default None.
    `clip_image_projection` : nn.Module, optional
        Projection module for image embeddings, default None.
    `transformer_embedding_dim` : int, optional
        Dimensionality of embeddings (default: 320).
    `num_layers` : int, optional
        Number of Transformer layers (default: 12).
    `num_attention_heads` : int, optional
        Number of attention heads in each Transformer layer (default: 8).
    `feedforward_dim` : int, optional
        Dimensionality of the feedforward network in Transformer layers (default: 768).
    `max_sequence_length` : int, optional
        Maximum sequence length for input embeddings (default: 2).
    `dropout_rate` : float, optional
        Dropout probability for regularization (default: 0.2).
    """
    def __init__(
        self,
        forward_diffusion: nn.Module, # will be used during training
        reverse_diffusion: nn.Module, # will be used during training
        clip_text_projection: Optional[nn.Module] = None,  # used during training instead of PCA in the main paper
        clip_image_projection: Optional[nn.Module] = None,  # used during training instead of PCA in the main paper
        transformer_embedding_dim: int = 320,
        num_layers: int = 12,
        num_attention_heads: int = 8,
        feedforward_dim: int = 768,
        max_sequence_length: int = 2,
        dropout_rate: float = 0.2
    ) -> None:
        super().__init__()

        self.forward_diffusion = forward_diffusion
        self.reverse_diffusion = reverse_diffusion
        self.clip_text_projection = clip_text_projection
        self.clip_image_projection = clip_image_projection

        self.transformer_embedding_dim = transformer_embedding_dim
        self.max_sequence_length = max_sequence_length

        # Time embedding network
        self.time_embedding_net = nn.Sequential(
            nn.Linear(transformer_embedding_dim, transformer_embedding_dim),
            nn.GELU(),
            nn.Linear(transformer_embedding_dim, transformer_embedding_dim)
        )

        # Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(max_sequence_length, transformer_embedding_dim))

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_embedding_dim, num_attention_heads, feedforward_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        # Final output projection
        self.output_projection = nn.Linear(transformer_embedding_dim, transformer_embedding_dim)

    def forward(
            self,
            text_embeddings: torch.Tensor,
            noisy_image_embeddings: torch.Tensor,
            timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Predicts clean image embeddings from noisy inputs and text embeddings.

        Processes text and noisy image embeddings through a Transformer architecture,
        conditioned on time embeddings, to predict the clean image embeddings.

        Parameters
        ----------
        `text_embeddings` : torch.Tensor
            Text embeddings, shape (batch_size, embedding_dim).
        `noisy_image_embeddings` : torch.Tensor
            Noisy image embeddings, shape (batch_size, embedding_dim).
        `timesteps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,).

        Returns
        -------
        predicted_clean_embeddings : torch.Tensor
            Predicted clean image embeddings, shape (batch_size, embedding_dim).
        """

        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device
        #print("text", text_embeddings.size())
        #print("noisy ", noisy_image_embeddings.size())
        #print("time ", timesteps.size())

        # Create sinusoidal time embeddings
        time_embeddings = self._get_sinusoidal_embeddings(timesteps, self.embedding_dim, device)
        time_embeddings = self.time_embedding_net(time_embeddings)

        # Add time information to image embeddings
        conditioned_image_embeddings = noisy_image_embeddings + time_embeddings

        # Create sequence: [text_embeddings, conditioned_image_embeddings]
        sequence = torch.stack([text_embeddings, conditioned_image_embeddings], dim=1)  # [B, 2, D]

        # Add positional embeddings
        sequence = sequence + self.positional_embeddings.unsqueeze(0)

        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            sequence = transformer_block(sequence)

        # Extract predicted clean image embedding (second position in sequence)
        predicted_clean_embeddings = sequence[:, 1, :]  # [B, D]

        # Apply final projection
        predicted_clean_embeddings = self.output_projection(predicted_clean_embeddings)

        return predicted_clean_embeddings

    def _get_sinusoidal_embeddings(
            self,
            timesteps: torch.Tensor,
            embedding_dim: int,
            device: Union[torch.device, str]
    ) -> torch.Tensor:
        """Generates sinusoidal positional embeddings for timesteps.

        Creates sinusoidal embeddings for the given timesteps to condition the Transformer
        on the diffusion process time steps.

        Parameters
        ----------
        `timesteps` : torch.Tensor
            Tensor of time step indices (long), shape (batch_size,).
        `embedding_dim` : int
            Dimensionality of the embeddings.
        `device` : Union[torch.device, str]
            Device to place the embeddings on.

        Returns
        -------
        embeddings : torch.Tensor
            Sinusoidal time embeddings, shape (batch_size, embedding_dim).
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # Handle odd embedding dimensions
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

        return emb


class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head attention and feedforward layers.

    Implements a Transformer block with multi-head self-attention, layer normalization,
    and a feedforward network with residual connections for processing sequences in
    the UnCLIPTransformerPrior model.

    Parameters
    ----------
    `embedding_dim` : int
        Dimensionality of input and output embeddings.
    `num_heads` : int
        Number of attention heads in the multi-head attention layer.
    `feedforward_dim` : int
        Dimensionality of the feedforward network.
    `dropout` : float
        Dropout probability for regularization.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            feedforward_dim: int,
            dropout: float
    ) -> None:
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feedforward_norm = nn.LayerNorm(embedding_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input sequence through the Transformer block.

        Applies multi-head self-attention followed by a feedforward network, with residual
        connections and layer normalization.

        Parameters
        ----------
        `x` : torch.Tensor
            Input sequence tensor, shape (batch_size, sequence_length, embedding_dim).

        Returns
        -------
        output : torch.Tensor
            Processed sequence tensor, shape (batch_size, sequence_length, embedding_dim).
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x)
        x = self.attention_norm(x + attn_output)

        # Feedforward with residual connection
        ff_output = self.feedforward(x)
        x = self.feedforward_norm(x + ff_output)

        return x


"""
model = UnCLIPTransformerPrior(
    embedding_dim=320,
    num_layers=12,
    num_attention_heads=8,
    feedforward_dim=768,
    max_sequence_length=2,
    dropout_rate=0.3
)

x = torch.randn((10, 320))
t = torch.randint(0, 1000, (10,))
print(t.size())
tm = model._get_sinusoidal_embeddings(t, 320, "cpu")
print(tm.size())
y = torch.randn((10, 320))
p = model(y, x, t)
print(p.size())
"""
