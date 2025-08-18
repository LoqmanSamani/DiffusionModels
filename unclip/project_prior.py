import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPEmbeddingProjection(nn.Module):
    """Projection module for dimensionality reduction and reconstruction.

    Implements a neural network with forward and inverse projections to reduce and
    restore input dimensionality, supporting customizable hidden layers, dropout, and
    layer normalization.

    Parameters
    ----------
    `clip_embedding_dim` : int, optional
        Input dimensionality (default: 1024).
    `transformer_embedding_dim` : int, optional
        Output dimensionality for forward projection (default: 320).
    `hidden_dim` : int, optional
        Hidden layer dimensionality (default: 512).
    `num_layers` : int, optional
        Number of layers in the projection network (default: 2).
    `dropout_rate` : float, optional
        Dropout probability for regularization (default: 0.2).
    `use_layer_norm` : bool, optional
        Whether to apply layer normalization after hidden layers (default: True).
    """
    def __init__(
        self,
        clip_embedding_dim: int = 1024,
        transformer_embedding_dim: int = 320,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        use_layer_norm: bool = True
    ) -> None:
        super().__init__()

        self.clip_embedding_dim = clip_embedding_dim
        self.transformer_embedding_dim = transformer_embedding_dim

        # Forward projection: input_dim -> output_dim
        self.forward_projection = self._build_projection_network(
            clip_embedding_dim, transformer_embedding_dim, hidden_dim, num_layers, dropout_rate, use_layer_norm
        )

        # Inverse projection: output_dim -> input_dim
        self.inverse_projection = self._build_projection_network(
            transformer_embedding_dim, clip_embedding_dim, hidden_dim, num_layers, dropout_rate, use_layer_norm
        )
    def _build_projection_network(
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
        return self.forward_projection(x)

    def inverse_transform(self, x_reduced: torch.Tensor) -> torch.Tensor:
        """Reconstructs input from lower-dimensional space.

        Applies the inverse projection network to restore the original dimensionality
        of the input tensor.

        Parameters
        ----------
        `x_reduced` : torch.Tensor
            Reduced-dimensionality tensor, shape (batch_size, output_dim).

        Returns
        -------
        x_reconstructed : torch.Tensor
            Reconstructed tensor, shape (batch_size, input_dim).
        """
        return self.inverse_projection(x_reduced)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
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
        x_reduced = self.forward(x)
        x_reconstructed = self.inverse_transform(x_reduced)
        return F.mse_loss(x_reconstructed, x)

"""
p = Projection(
    input_dim=1024,
    output_dim=512,
    hidden_dim=768,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
)

x = torch.randn((100, 1024))
o = p(x)
print(o.size())
"""

