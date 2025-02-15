import torch
import torch.nn as nn



class TimeEmbedding(nn.Module):
    """
    positional time embedding
    """
    def __init__(self, embed_dim):
        super().__init__()
        assert embed_dim % 2 == 0, "The embedding dimension must be divisible by two"
        self.embed_dim = embed_dim

    def forward(self,  time_steps):

        factor = (2 * torch.arange(start=0, end=self.embed_dim//2, dtype=torch.float32, device=time_steps.device)) / self.embed_dim
        embed_time = time_steps[:, None]
        embed_time = embed_time / factor
        embed_time = torch.cat(tensors=[torch.sin(embed_time), torch.cos(embed_time)], dim=1)

        return embed_time



