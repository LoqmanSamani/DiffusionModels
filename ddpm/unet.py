import torch
import torch.nn as nn




class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


#-----------------------------------------------------------------------------
class DownBlock(nn.Module):
    """
    down block/s of the u-net used in ddpm models
    steps:
        1. Conv3 followed by TimeEmbedding
        2. Conv3 layer
        3. add a skip-connection from the input to the output of step 2
        4. self-attention on the output
        5. add a skip-connection from step 3 to the output of step 4
        6. down-sampling (if enabled)

    """
    def __init__(self, in_channels, out_channels, time_embed_dim=128, num_layers=2, down_sample=True):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i==0 else out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True
            ) for _ in range(self.num_layers)
        ])
        self.time_embedding = nn.ModuleList([
            TimeEmbedding(
                output_dim=out_channels,
                embed_dim=time_embed_dim
            ) for _ in range(self.num_layers)
        ])
        self.attention = nn.ModuleList([
            Attention(
                num_channels=out_channels,
                num_groups=8,
                num_heads=4,
                norm=True
            ) for _ in range(self.num_layers)
        ])
        self.down_sampling = DownSampling(
            in_channels=out_channels,
            out_channels=out_channels,
            down_sampling_factor=2,
            conv_block=True,
            max_pool=True
        ) if down_sample else nn.Identity()
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers)

        ])

    def forward(self, batch, embed_time):

        output = batch
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = output + self.time_embedding[i](embed_time)[:, :, None, None]
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
            out_attn = self.attention[i](output)
            output = output + out_attn

        output = self.down_sampling(output)

        return output

#------------------------------------------------------------------------------
class MiddleBlock(nn.Module):
    """
    middle block/s of the u-net used in ddpm models
    steps:
        1. resnet with time embedding
        2. n  self-attention + resnet with time embedding
    """
    def __init__(self, in_channels, out_channels, time_embed_dim=128, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True
            ) for i in range(self.num_layers+1)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True
            ) for _ in range(self.num_layers+1)
        ])
        self.time_embedding = nn.ModuleList([
            TimeEmbedding(
                output_dim=out_channels,
                embed_dim=time_embed_dim
            ) for _ in range(self.num_layers+1)
        ])
        self.attention = nn.ModuleList([
            Attention(
                num_channels=out_channels,
                num_groups=8,
                num_heads=4,
                norm=True
            ) for _ in range(self.num_layers)
        ])
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers+1)

        ])

    def forward(self, batch, embed_time):
        output = batch

        resnet_input = output
        output = self.conv1[0](output)
        output = output + self.time_embedding[0](embed_time)[:, :, None, None]
        output = self.conv2[0](output)
        output = output + self.resnet[0](resnet_input)

        for i in range(self.num_layers):
            out_attn = self.attention[i](output)
            output = output + out_attn
            resnet_input = output
            output = self.conv1[i + 1](output)
            output = output + self.time_embedding[i + 1](embed_time)[:, :, None, None]
            output = self.conv2[i + 1](output)
            output = output + self.resnet[i + 1](resnet_input)

        return output

#------------------------------------------------------------------------------
class UpBlock(nn.Module):
    """
    up-sampling of the u-net used in ddpm models
    steps:
        1. up-sampling
        2. conv3 + time embedding
        3. conv3
        4. skip-connection from 1.
        5. self-attention
        6. skip-connection from 3.

    """
    def __init__(self, in_channels, out_channels, time_embed_dim=128, num_layers=2, up_sampling=True):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True
            ) for _ in range(self.num_layers)
        ])
        self.time_embedding = nn.ModuleList([
            TimeEmbedding(
                output_dim=out_channels,
                embed_dim=time_embed_dim
            ) for _ in range(self.num_layers)
        ])
        self.attention = nn.ModuleList([
            Attention(
                num_channels=out_channels,
                num_groups=8,
                num_heads=4,
                norm=True
            ) for _ in range(self.num_layers)
        ])
        self.up_sampling = UpSampling(
            in_channels=in_channels,
            out_channels=in_channels//2,
            up_sampling_factor=2,
            conv_block=True,
            up_sampling=True
        ) if up_sampling else nn.Identity()
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers)

        ])

    def forward(self, batch, skip_connection, embed_time):

        batch = self.up_sampling(batch)
        batch = torch.cat(tensors=[batch, skip_connection], dim=1)

        output = batch
        for i in range(self.num_layers):
            resnet_input = output

            output = self.conv1[i](output)
            output = output + self.time_embedding[i](embed_time)[:, :, None, None]
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)

            out_attn = self.attention[i](output)
            output = output + out_attn

        return output

#------------------------------------------------------------------------
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

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels//2 if conv_block else out_channels, kernel_size=4, stride=up_sampling_factor, padding=1),
            nn.Conv2d(in_channels=out_channels//2 if up_sampling else out_channels, out_channels=out_channels//2 if up_sampling else out_channels, kernel_size=1, stride=1, padding=0)
        ) if conv_block else nn.Identity()

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











