import torch
import torch.nn as nn


class CLIPContextProjection(nn.Module):
    """Projects CLIP image embeddings into multiple context tokens.

    Transforms a single CLIP image embedding into a specified number of context tokens
    using a linear projection followed by layer normalization.

    Parameters
    ----------
    `clip_embedding_dim` : int
        Dimensionality of the input CLIP embedding (e.g., 319 or 512).
    `num_tokens` : int, optional
        Number of context tokens to generate (default: 4).
    """
    def __init__(self, clip_embedding_dim, num_tokens=4):
        super().__init__()
        self.clip_embedding_dim = clip_embedding_dim
        self.num_tokens = num_tokens
        self.clip_projection = nn.Linear(clip_embedding_dim, clip_embedding_dim * num_tokens)
        self.clip_embedding_norm = nn.LayerNorm(clip_embedding_dim)

    def forward(self, z_i):
        """Projects CLIP image embedding into context tokens.

        Applies a linear projection to transform the input embedding into multiple tokens,
        reshapes the output, and applies layer normalization.

        Parameters
        ----------
        `z_i` : torch.Tensor
            Input CLIP image embedding, shape (batch_size, input_dim).

        Returns
        -------
        c : torch.Tensor
            Context tokens, shape (batch_size, num_tokens, input_dim).
        """
        batch_size = z_i.shape[0]
        projected = self.clip_projection(z_i)
        c = projected.view(batch_size, self.num_tokens, self.clip_embedding_dim)
        c = self.clip_embedding_norm(c)
        return c


"""
# Example usage
batch_size = 32
embed_dim = 319  # Example CLIP embedding dim after PCA

projector = Project(input_dim=embed_dim)
z_i = torch.randn(batch_size, embed_dim)
c = projector(z_i)  # Shape: [batch_size, 4, token_dim]
print(f"Shape of c: {c.shape}")  # Expected: [32, 4, 768]
"""