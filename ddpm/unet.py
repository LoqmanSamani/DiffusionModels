import torch
import torch.nn as nn



class UNet(nn.Module):
    """
    U-net architecture which is used to predict noise
    in the paper "Denoising Diffusion Probabilistic Model"
    """
    def __init__(
            self,
            in_channels,
            down_channels,
            mid_channels,
            up_channels,
            down_sampling,
            time_embed_dim,
            num_down_blocks,
            num_mid_blocks,
            num_up_blocks
    ):
        super().__init__()
        self.in_channels = in_channels or 3 # RGB
        self.down_channels = down_channels or [32, 64, 128, 256]
        self.mid_channels = mid_channels or [256, 256, 128]
        self.up_channels = up_channels or [256, 128, 64, 16]
        self.down_sampling = down_sampling or [True, True, False]
        self.time_embed_dim = time_embed_dim or 128
        self.num_down_blocks = num_down_blocks or 2
        self.num_mid_blocks = num_mid_blocks or 2
        self.num_up_blocks = num_up_blocks or 2

        self.up_sampling = list(reversed(self.down_sampling))
        # initial convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.down_channels[0],
            kernel_size=3,
            padding=1
        )
        # initial time embedding projection
        self.time_projection = nn.Sequential(
            nn.Linear(in_features=self.time_embed_dim, out_features=self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(in_features=self.time_embed_dim, out_features=self.time_embed_dim)
        )
        # down blocks
        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels=self.down_channels[i],
                out_channels=self.down_channels[i+1],
                time_embed_dim=self.time_embed_dim,
                num_layers=self.num_down_blocks,
                down_sample=self.down_sampling[i]
            ) for i in range(len(self.down_channels)-1)
        ])
        # middle blocks
        self.mid_blocks = nn.ModuleList([
            MiddleBlock(
                in_channels=self.mid_channels[i],
                out_channels=self.mid_channels[i+1],
                time_embed_dim=self.time_embed_dim,
                num_layers=self.num_mid_blocks
            ) for i in range(len(self.mid_channels)-1)
        ])
        # up blocks
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=self.up_channels[i],
                out_channels=self.up_channels[i+1],
                time_embed_dim=self.time_embed_dim,
                num_layers=self.num_up_blocks,
                up_sampling=self.up_sampling[i]
            ) for i in range(len(self.up_channels)-1)
        ])
        # final convolution layer
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=self.up_channels[-1]),
            nn.Conv2d(in_channels=self.up_channels[-1], out_channels=self.in_channels, kernel_size=3, padding=1)
        )

    def forward(self, batch, t):

        output = self.conv1(batch)
        # time projection
        time_embed = GetEmbeddedTime(embed_dim=self.time_embed_dim)(time_steps=t)
        time_embed = self.time_projection(time_embed)

        # store the skip connections
        skip_connections = []
        # down blocks
        for down in self.down_blocks:
            skip_connections.append(output)
            output = down(batch=output, embed_time=time_embed)

        # middle blocks
        for mid in self.mid_blocks:
            output = mid(batch=output, embed_time=time_embed)

        # up blocks
        for up in self.up_blocks:
            skip_connection = skip_connections.pop()
            output = up(batch=output, skip_connection=skip_connection, embed_time=time_embed)

        # final convolution layer
        output = self.conv2(output)

        return output

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
class GetEmbeddedTime(nn.Module):
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
