import torch
import torch.nn as nn

class Projection(nn.Module):
    def __init__(
            self,
            input_dim: int = 512,
            output_dim: int = 319,
            hidden_dim: int = 384,
            num_layers: int = 2,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        # Final projection layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.proj = nn.Sequential(*layers)

        # Inverse projection
        inverse_layers = []
        current_dim = output_dim

        for i in range(num_layers - 1):
            inverse_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        inverse_layers.append(nn.Linear(current_dim, input_dim))
        self.inverse_proj = nn.Sequential(*inverse_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def inverse_transform(self, x_reduced: torch.Tensor) -> torch.Tensor:
        return self.inverse_proj(x_reduced)