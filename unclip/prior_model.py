import torch
import torch.nn as nn
import math
from typing import Union


class UnCLIPTransformerPrior(nn.Module):
    """UnCLIP prior model using Transformer"""
    def __init__(
        self,
        embedding_dim: int = 319,
        num_layers: int = 12,
        num_attention_heads: int = 8,
        feedforward_dim: int = 768,
        max_sequence_length: int = 2,
        dropout_rate: float = 0.1
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

        # Time embedding network
        self.time_embedding_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(max_sequence_length, embedding_dim))

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_attention_heads, feedforward_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        # Final output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(
            self,
            text_embeddings: torch.Tensor,
            noisy_image_embeddings: torch.Tensor,
            timesteps: torch.Tensor
    ) -> torch.Tensor:

        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device

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
        """Generate sinusoidal positional embeddings for timesteps."""
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
    """Single transformer block with multi-head attention and feedforward layers."""

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
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x)
        x = self.attention_norm(x + attn_output)

        # Feedforward with residual connection
        ff_output = self.feedforward(x)
        x = self.feedforward_norm(x + ff_output)

        return x