import torch
import torch.nn as nn
from config import Config


class DDPMpp(nn.Module):
    """neural network for the VP SDE and sub-VP SDE (DDPM++)"""

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.down_channels = config.down_channels
        self.mid_channels = config.mid_channels
        self.up_channels = config.up_channels
        self.down_sampling = config.down_sampling
        self.embed_dim = config.embed_dim
        self.num_down_blocks = config.num_down_blocks
        self.num_mid_blocks = config.num_mid_blocks
        self.num_up_blocks = config.num_up_blocks
        self.dropout_rate = config.dropout_rate
        self.up_sampling = list(reversed(self.down_sampling))
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.down_channels[0],
            kernel_size=3,
            padding=1
        )
        self.time_projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim),
            nn.SiLU(),
            nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        )
        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels=self.down_channels[i],
                out_channels=self.down_channels[i + 1],
                embed_dim=self.embed_dim,
                num_layers=self.num_down_blocks,
                down_sample=self.down_sampling[i],
                dropout_rate=self.dropout_rate,
                attention_heads=config.num_attention_heads,
                factor=config.down_sampling_factor,
                conv=config.apply_down_conv,
                pool=config.apply_down_pool,
                num_groups=config.num_groups,
                kernel_size=config.kernel_size,
                norm=config.norm,
                activation=config.activation
            ) for i in range(len(self.down_channels)-1)
        ])
        self.mid_blocks = nn.ModuleList([
            MiddleBlock(
                in_channels=self.mid_channels[i],
                out_channels=self.mid_channels[i + 1],
                embed_dim=self.embed_dim,
                num_layers=self.num_mid_blocks,
                dropout_rate=self.dropout_rate,
                num_groups=config.num_groups,
                kernel_size=config.kernel_size,
                norm=config.norm,
                activation=config.activation,
                attention_heads=config.num_attention_heads
            ) for i in range(len(self.mid_channels)-1)
        ])
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=self.up_channels[i],
                out_channels=self.up_channels[i + 1],
                embed_dim=self.embed_dim,
                num_layers=self.num_up_blocks,
                up=self.up_sampling[i],
                dropout_rate=self.dropout_rate,
                factor=config.upsampling_factor,
                conv=config.apply_up_conv,
                num_groups=config.num_groups,
                kernel_size=config.kernel_size,
                norm=config.norm,
                activation=config.activation,
                attention_heads=config.num_attention_heads
            ) for i in range(len(self.up_channels)-1)
        ])
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=self.up_channels[-1]),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(in_channels=self.up_channels[-1], out_channels=self.in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):

        output = self.conv1(x)
        time_embed = GetEmbeddedTime(embed_dim=self.embed_dim)(time_steps=t)
        time_embed = self.time_projection(time_embed)
        skip_connections = []
        for i, down in enumerate(self.down_blocks):
            skip_connections.append(output)
            output = down(x=output, embed_time=time_embed)
        for i, mid in enumerate(self.mid_blocks):
            output = mid(x=output, embed_time=time_embed)
        for i, up in enumerate(self.up_blocks):
            skip_connection = skip_connections.pop()
            output = up(x=output, skip_connection=skip_connection, embed_time=time_embed)
        output = self.conv2(output)

        return output

#-----------------------------------------------------------------------------
class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, embed_dim, num_layers, down_sample, dropout_rate, attention_heads, factor, conv, pool, num_groups, kernel_size, norm, activation):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i==0 else out_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.time_embedding = nn.ModuleList([
            TimeEmbedding(
                output_dim=out_channels,
                embed_dim=embed_dim
            ) for _ in range(self.num_layers)
        ])
        self.attention = nn.ModuleList([
            Attention(
                in_channels=out_channels,
                num_heads=attention_heads,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.down_sampling = DownSampling(
            in_channels=out_channels,
            out_channels=out_channels,
            factor=factor,
            conv=conv,
            pool=pool
        ) if down_sample else nn.Identity()
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers)

        ])

    def forward(self, x, embed_time):

        output = x
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

    def __init__(self, in_channels, out_channels, embed_dim, num_layers, dropout_rate, num_groups, kernel_size, norm, activation, attention_heads):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers+1)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers+1)
        ])
        self.time_embedding = nn.ModuleList([
            TimeEmbedding(
                output_dim=out_channels,
                embed_dim=embed_dim
            ) for _ in range(self.num_layers+1)
        ])
        self.attention = nn.ModuleList([
            Attention(
                in_channels=out_channels,
                num_heads=attention_heads,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers+1)
        ])

    def forward(self, x, embed_time):

        output = x
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

    def __init__(self, in_channels, out_channels, embed_dim, num_layers, up, dropout_rate, factor, conv,num_groups, kernel_size, norm, activation, attention_heads):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.time_embedding = nn.ModuleList([
            TimeEmbedding(
                output_dim=out_channels,
                embed_dim=embed_dim
            ) for _ in range(self.num_layers)
        ])
        self.attention = nn.ModuleList([
            Attention(
                in_channels=out_channels,
                num_heads=attention_heads,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.up_sampling = UpSampling(
            in_channels=in_channels,
            out_channels=in_channels//2,
            factor=factor,
            conv_=conv,
            up=up
        ) if up else nn.Identity()
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers)

        ])

    def forward(self, x, skip_connection, embed_time):

        x = self.up_sampling(x)
        x = torch.cat(tensors=[x, skip_connection], dim=1)
        output = x
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

    def __init__(self, in_channels, out_channels, num_groups, kernel_size, norm, activation, dropout_rate):
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels) if norm else nn.Identity()
        self.activation = nn.SiLU() if activation else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        x = self.group_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)

        return x

#----------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, output_dim, embed_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=embed_dim, out_features=output_dim)
        )

    def forward(self, x):
        return self.embedding(x)

#----------------------------------------------------------------
class GetEmbeddedTime(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        assert embed_dim % 2 == 0, "The embedding dimension must be divisible by two"
        self.embed_dim = embed_dim

    def forward(self, time_steps):
        i = torch.arange(start=0, end=self.embed_dim // 2, dtype=torch.float32, device=time_steps.device)
        factor = 10000 ** (2 * i / self.embed_dim)

        embed_time = time_steps[:, None] / factor
        embed_time = torch.cat(tensors=[torch.sin(embed_time), torch.cos(embed_time)], dim=-1)

        return embed_time

#----------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, in_channels, num_heads, dropout_rate):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=(min(in_channels//4, 32)), num_channels=in_channels, eps=1e-6)
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        b, c, h, w = x.shape # (batch size, number of channels, height, width)
        x = x.reshape(b, c, h * w)
        x = self.group_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x

#----------------------------------------------------------------------------------
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, factor, conv_, up):
        super().__init__()
        self.conv_ = conv_
        self.up = up

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels // 2 if up else out_channels,
                kernel_size=4,
                stride=factor,
                padding=1
            ),
            nn.Conv2d(
                in_channels=out_channels // 2 if up else out_channels,
                out_channels=out_channels // 2 if up else out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ) if self.conv_ else nn.Identity()

        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2 if self.conv_ else out_channels,
                      kernel_size=1, stride=1, padding=0)
        ) if up else nn.Identity()

    def forward(self, batch):

        if not self.conv_:
            return self.up_sample(batch)

        if not self.up:
            return self.conv(batch)

        return torch.cat(tensors=[self.conv(batch), self.up_sample(batch)], dim=1)

#-----------------------------------------------------------------------------------
class DownSampling(nn.Module):

    def __init__(self, in_channels, out_channels, factor, conv, pool):
        super().__init__()
        self.conv_block = conv
        self.max_pool = pool

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2 if pool else out_channels,
                      kernel_size=4, stride=factor, padding=1)
        ) if conv else nn.Identity()

        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=factor, stride=factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2 if conv else out_channels,
                      kernel_size=1, stride=1, padding=0)
        ) if pool else nn.Identity()

    def forward(self, batch):

        if not self.conv_block:
            return self.pool(batch)

        if not self.max_pool:
            return self.conv(batch)

        return torch.cat(tensors=[self.conv(batch), self.pool(batch)], dim=1)