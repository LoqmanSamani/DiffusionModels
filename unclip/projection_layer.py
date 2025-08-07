import torch
import torch.nn as nn
import torch.nn.functional as F

class Projection(nn.Module):

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 310,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_layer_norm: bool = True
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Forward projection: input_dim -> output_dim
        self.forward_projection = self._build_projection_network(
            input_dim, output_dim, hidden_dim, num_layers, dropout, use_layer_norm
        )

        # Inverse projection: output_dim -> input_dim
        self.inverse_projection = self._build_projection_network(
            output_dim, input_dim, hidden_dim, num_layers, dropout, use_layer_norm
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
        """Build a projection network."""
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
        """Forward projection: reduce dimensionality."""
        return self.forward_projection(x)

    def inverse_transform(self, x_reduced: torch.Tensor) -> torch.Tensor:
        """Inverse projection: restore original dimensionality."""
        return self.inverse_projection(x_reduced)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss for the projection."""
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

