import torch
import torch.nn as nn



class Conv3(nn.Module):
    """conv 3 block"""
    def __init__(self, in_channels, out_channels, num_groups=8, kernel_size=3, norm=True, activation=True):
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels) if norm else nn.Identity()
        self.activation = nn.SiLU() if activation else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, batch):

        batch = self.group_norm(batch)
        batch = self.activation(batch)
        batch = self.conv(batch)

        return batch

#----------------------------------------------------------------
class TimeEmbedding(nn.Module):
    """time embedding"""
    def __init__(self, output_dim, embed_dim=128):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=embed_dim, out_features=output_dim)
        )

    def forward(self, batch):
        return self.embedding(batch)

#----------------------------------------------------------------
class Attention(nn.Module):
    """group norm and multi-head attention"""
    def __init__(self, num_channels, num_groups=8, num_heads=4, norm=True):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels) if norm else nn.Identity()
        self.attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)

    def forward(self, batch):

        batch_size, channels, h, w = batch.shape
        batch = batch.reshape(batch_size, channels, h * w)
        batch = self.group_norm(batch)
        batch = batch.transpose(1, 2)
        batch, _ = self.attention(batch, batch, batch)
        batch = batch.transpose(1, 2).reshape(batch_size, channels, h, w)

        return batch

#-----------------------------------------------------------------
class DownSampling(nn.Module):
    """down-sampling by the factor of down_sampling_factor"""

    def __init__(self, in_channels, out_channels, down_sampling_factor=2, conv_block=True, max_pool=True):
        super().__init__()
        self.conv_block = conv_block # if conv block should be applied
        self.max_pool = max_pool # if max pool should be applied

        # down sampling using convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2 if max_pool else out_channels, kernel_size=4, stride=down_sampling_factor, padding=1)
        ) if conv_block else nn.Identity()

        # down sampling using max pool
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=down_sampling_factor, stride=down_sampling_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2 if conv_block else out_channels, kernel_size=1, stride=1, padding=0)
        ) if max_pool else nn.Identity()

    def forward(self, batch):

        if not self.conv_block:
            return self.pool(batch)

        if not self.max_pool:
            return self.conv(batch)

        return torch.cat(tensors=[self.conv(batch), self.pool(batch)], dim=1)


#--------------------------------------------------------------------------
class UpSampling(nn.Module):
    """up sampling by the factor of up_sampling_factor"""

    def __init__(self, in_channels, out_channels, up_sampling_factor=2, conv_block=True, up_sampling=True):
        super().__init__()

        self.conv_block = conv_block
        self.up_sampling = up_sampling

        # up sampling using convolution
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels//2 if conv_block else out_channels, kernel_size=4, stride=up_sampling_factor, padding=1),
            nn.Conv2d(in_channels=out_channels//2 if up_sampling else out_channels, out_channels=out_channels//2 if up_sampling else out_channels, kernel_size=1, stride=1, padding=0)
        ) if conv_block else nn.Identity()

        # up sampling using nn.Upsample
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=up_sampling_factor, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2 if conv_block else out_channels, kernel_size=1, stride=1, padding=0)
        ) if up_sampling else nn.Identity()

    def forward(self, batch):

        if not self.conv_block:
            return self.up_sample(batch)

        if not self.up_sampling:
            return self.conv(batch)

        return torch.cat(tensors=[self.conv(batch), self.up_sample(batch)], dim=1)

#-----------------------------------------------------------------------------




