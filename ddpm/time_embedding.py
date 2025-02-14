import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    positional time embedding module.
    implements sinusoidal embeddings based on the paper "attention is all you need".
    """

    def __init__(self, embedd_dim):
        super().__init__()
        assert embedd_dim % 2 == 0, "The embedding dimension must be even"
        self.embedd_dim = embedd_dim

    def forward(self, time_steps):

        time_steps = time_steps[:, None]  # expand to (batch_size, 1)
        log_base = torch.log(torch.tensor(10000.0, device=time_steps.device))
        factor = torch.exp(- (2 * torch.arange(self.embedd_dim // 2, device=time_steps.device, dtype=torch.float32) / self.embedd_dim) * log_base)
        embedded_time = time_steps * factor
        out = torch.cat(tensors=[torch.sin(embedded_time), torch.cos(embedded_time)], dim=-1)  # shape: (batch_size, embedd_dim)

        return out



