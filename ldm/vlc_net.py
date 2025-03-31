import torch
import torch.nn as nn




class VariationalLatentCompressor(nn.Module):
    def __init__(
            self,
            in_channels, # number of channels of the original image. e.g., 3 for RBG.
            down_channels, # a list of channels used in encoder. e.g., [32, 64, 128, 256].
            up_channels, # a list of channels used in decoder. e.g., [256, 128, 64, 16].
            out_channels, # probably the same as in_channels. used to construct the image.
            dropout_rate, # dropout rate, prevents overfitting.
            num_heads, # number of attention heads in self-attention layers
            num_groups, # number of groups in group normalization. used in self-attention.
            levels,  # number of down/up sampling layers.
            down_sampling_factor, # down-sampling factor, an integer used to down/up sample the input batch of images.
            beta=1.0 # weight for KL loss
    ):
        super().__init__()
        assert in_channels == out_channels, "Input and output channels must match for auto-encoding"

        self.beta = beta  # original beta value
        self.current_beta = beta  # current beta value (for annealing)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=down_channels[0],
            kernel_size=3,
            padding=1
        )
        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels=down_channels[i],
                out_channels=down_channels[i + 1],
                num_layers=levels,
                down_sampling_factor=down_sampling_factor,
                dropout_rate=dropout_rate
            ) for i in range(len(down_channels) - 1)
        ])
        self.attention1 = Attention(
            num_channels=down_channels[-1],
            num_heads=num_heads,
            num_groups=num_groups,
            dropout_rate=dropout_rate
        )
        self.conv_mu = nn.Conv2d(
            in_channels=down_channels[-1],
            out_channels=down_channels[-1],
            kernel_size=3,
            padding=1
        )
        self.conv_logvar = nn.Conv2d(
            in_channels=down_channels[-1],
            out_channels=down_channels[-1],
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=down_channels[-1],
            out_channels=down_channels[-1],
            kernel_size=3,
            padding=1
        )
        self.attention2 = Attention(
            num_channels=down_channels[-1],
            num_heads=num_heads,
            num_groups=num_groups,
            dropout_rate=dropout_rate
        )
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=up_channels[i],
                out_channels=up_channels[i + 1],
                num_layers=levels,
                up_sampling_factor=down_sampling_factor,
                dropout_rate=dropout_rate
            ) for i in range(len(up_channels) - 1)
        ])
        self.conv3 = Conv3(
            in_channels=up_channels[-1],
            out_channels=out_channels,
            dropout_rate=dropout_rate
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, x):
        x = self.conv1(x)
        for block in self.down_blocks:
            x = block(x)
        res_x = x
        x = self.attention1(x)
        x = x + res_x
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        z_hat = self.reparameterize(mu, logvar)
        return z_hat, mu, logvar

    def decoder(self, zq):
        x = self.conv2(zq)
        res_x = x
        x = self.attention2(x)
        x = x + res_x
        for block in self.up_blocks:
            x = block(x)
        x = self.conv3(x)
        return x

    def forward(self, x):

        z, mu, logvar = self.encoder(x)
        x_hat = self.decoder(z)
        kl_unnormalized = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        batch_size = x.size(0)
        latent_size = torch.prod(torch.tensor(mu.shape[1:])).item()
        kl_loss = kl_unnormalized / (batch_size * latent_size)
        weighted_kl = kl_loss * self.current_beta

        return x_hat, weighted_kl
#------------------------------------------------------------------------------------------------
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, down_sampling_factor, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])

        self.down_sampling = DownSampling(
            in_channels=out_channels,
            out_channels=out_channels,
            down_sampling_factor=down_sampling_factor
        )
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers)

        ])

    def forward(self, x):
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
        output = self.down_sampling(output)
        print(output.shape)
        return output
# ------------------------------------------------------------------------------------------------
class Conv3(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.activation = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.group_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x
#------------------------------------------------------------------------------------------------
class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling_factor):
        super().__init__()
        self.down_sampling_factor = down_sampling_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                      kernel_size=3, stride=down_sampling_factor, padding=1)
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=down_sampling_factor, stride=down_sampling_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                      kernel_size=1, stride=1, padding=0)
        )

    def forward(self, batch):
        return torch.cat(tensors=[self.conv(batch), self.pool(batch)], dim=1)
#------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, num_channels, num_heads, num_groups, dropout_rate):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h * w)
        x = self.group_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x
#------------------------------------------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, up_sampling_factor, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        effective_in_channels = in_channels

        self.up_sampling = UpSampling(
            in_channels=in_channels,
            out_channels=in_channels,
            up_sampling_factor=up_sampling_factor
        )

        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=effective_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=effective_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(self.num_layers)
        ])

    def forward(self, x):
        x = self.up_sampling(x)
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
        return output
#------------------------------------------------------------------------------------------------
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, up_sampling_factor):
        super().__init__()
        half_out_channels = out_channels // 2
        self.up_sampling_factor = up_sampling_factor
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=half_out_channels,
                kernel_size=3,
                stride=up_sampling_factor,
                padding=1,
                output_padding=up_sampling_factor - 1
            ),
            nn.Conv2d(
                in_channels=half_out_channels,
                out_channels=half_out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=up_sampling_factor, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=half_out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, batch):

        conv_output = self.conv(batch)
        up_sample_output = self.up_sample(batch)
        if conv_output.shape[2:] != up_sample_output.shape[2:]:
            _, _, h, w = conv_output.shape
            up_sample_output = torch.nn.functional.interpolate(
                up_sample_output,
                size=(h, w),
                mode='nearest'
            )

        return torch.cat(tensors=[conv_output, up_sample_output], dim=1)