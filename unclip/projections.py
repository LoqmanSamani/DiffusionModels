import torch
import torch.nn as nn
import torch.nn.functional as F




class CLIPContextProjection(nn.Module):
    """Projects CLIP image embeddings into multiple context tokens.

    Transforms a single CLIP image embedding into a specified number of context tokens
    using a linear projection followed by layer normalization.

    Parameters
    ----------
    `clip_embed_dim` : int
        Dimensionality of the input CLIP embedding (e.g., 319 or 512).
    `num_tokens` : int, optional
        Number of context tokens to generate (default: 4).
    """
    def __init__(self, clip_embed_dim, num_tokens=4):
        super().__init__()
        self.clip_embed_dim = clip_embed_dim
        self.num_tokens = num_tokens
        self.clip_proj = nn.Linear(clip_embed_dim, clip_embed_dim * num_tokens)
        self.clip_embed_norm = nn.LayerNorm(clip_embed_dim)

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
        proj = self.clip_proj(z_i)
        c = proj.view(batch_size, self.num_tokens, self.clip_embed_dim)
        c = self.clip_embed_norm(c)
        return c




class CLIPEmbeddingProjection(nn.Module):
    """Projection module for dimensionality reduction and reconstruction.

    Implements a neural network with forward and inverse projections to reduce and
    restore input dimensionality, supporting customizable hidden layers, dropout, and
    layer normalization.

    Parameters
    ----------
    `clip_embed_dim` : int, optional
        Input dimensionality (default: 1024).
    `trans_embed_dim` : int, optional
        Output dimensionality for forward projection (default: 320).
    `hidden_dim` : int, optional
        Inner dimension of projection (default: 512).
    `dropout`: float
        Dropout rate (default: 0.2)
    `use_layer_norm`: bool
        If normalize output (default: True)
    """
    def __init__(
        self,
        clip_embed_dim: int = 1024,
        trans_embed_dim: int = 320,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_layer_norm: bool = True
    ) -> None:
        super().__init__()
        self.clip_embed_dim = clip_embed_dim
        self.trans_embed_dim = trans_embed_dim
        # input_dim -> output_dim
        self.fwd_proj = self._build_proj_net(
            clip_embed_dim, trans_embed_dim, hidden_dim, num_layers, dropout, use_layer_norm
        )
        # output_dim -> input_dim
        self.inv_proj = self._build_proj_net(
            trans_embed_dim, clip_embed_dim, hidden_dim, num_layers, dropout, use_layer_norm
        )
    def _build_proj_net(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            use_layer_norm: bool
    ) -> nn.Sequential:
        """Builds a projection network with customizable layers.

        Constructs a neural network with linear layers, optional layer normalization,
        GELU activation, and dropout for either forward or inverse projection.

        Parameters
        ----------
        `input_dim` : int
            Input dimensionality for the network.
        `output_dim` : int
            Output dimensionality for the network.
        `hidden_dim` : int
            Hidden layer dimensionality.
        `num_layers` : int
            Number of layers in the network.
        `dropout` : float
            Dropout probability for regularization.
        `use_layer_norm` : bool
            Whether to apply layer normalization after hidden layers.

        Returns
        -------
        network : nn.Sequential
            Sequential container of the projection network layers.
        """
        layers = []
        current_dim = input_dim

        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input to a lower-dimensional space.

        Applies the forward projection network to reduce the dimensionality of the input tensor.

        Parameters
        ----------
        `x` : torch.Tensor
            Input tensor to be projected, shape (batch_size, input_dim).

        Returns
        -------
        x_reduced : torch.Tensor
            Projected tensor, shape (batch_size, output_dim).
        """
        return self.fwd_proj(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstructs input from lower-dimensional space.

        Applies the inverse projection network to restore the original dimensionality
        of the input tensor.

        Parameters
        ----------
        `x` : torch.Tensor
            Reduced-dimensionality tensor, shape (batch_size, output_dim).

        Returns
        -------
        x : torch.Tensor
            Reconstructed tensor, shape (batch_size, input_dim).
        """
        return self.inv_proj(x)

    def rec_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the reconstruction loss for the projection.

        Calculates the mean squared error between the original input and its reconstruction
        after forward and inverse projections.

        Parameters
        ----------
        `x` : torch.Tensor
            Original input tensor, shape (batch_size, input_dim).

        Returns
        -------
        loss : torch.Tensor
            Mean squared error loss between the original and reconstructed tensors.
        """
        x = self.forward(x)
        x_rec = self.inverse_transform(x)
        return F.mse_loss(x_rec, x)