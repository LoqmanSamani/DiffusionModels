import torch
import torch.nn as nn
import math
from typing import Union


class UnclipPrior(nn.Module):
    """UnCLIP prior model using Transformer"""
    def __init__(
            self,
            embedding_dim: int = 319,
            num_layers: int = 12,
            num_heads: int = 8,
            feed_forward_dim: int = 768,
            max_seq_len: int = 2,
            dropout_rate: float = 0.2
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.randn(max_seq_len, embedding_dim))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, feed_forward_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)


    def forward(
            self,
            text_embed: torch.Tensor,
            noisy_image_embed: torch.Tensor,
            timestep: torch.Tensor
    ) -> torch.Tensor:

        batch_size = text_embed.shape[0]
        device = text_embed.device

        # Time embedding
        time_embed = self.get_time_embedding(timestep, self.embedding_dim, device)
        time_embed = self.time_embed(time_embed)

        # Add time embedding to image embedding
        image_proj = noisy_image_embed + time_embed

        # Create sequence: [text, noisy_image]
        sequence = torch.stack([text_embed, image_proj], dim=1)  # (B, 2, reduced_dim)

        # Add positional embeddings
        sequence = sequence + self.positional_embedding.unsqueeze(0)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            sequence = layer(sequence)

        # Extract image prediction (second token)
        predicted_clean_image = sequence[:, 1, :]  # (B, reduced_dim)

        # Final output projection
        predicted_clean_image = self.output_projection(predicted_clean_image)

        return predicted_clean_image

    def get_time_embedding(self, timesteps: torch.Tensor, embedding_dim: torch.Tensor, device: Union[torch.device, str]) -> torch.Tensor:
        """Sinusoidal time embeddings"""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb



class TransformerBlock(nn.Module):
    """Transformer block for UnCLIP prior"""

    def __init__(self, embedding_dim: int, num_heads: int, feed_forward_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attention_out, _ = self.attention(sequence, sequence, sequence)
        sequence = self.norm1(sequence + attention_out)

        # Feed-forward
        feed_forward_out = self.feed_forward(sequence)
        sequence = self.norm2(sequence + feed_forward_out)

        return sequence