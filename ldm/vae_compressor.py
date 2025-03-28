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
            down_sampling, # a list of bools, if down sampling should be applied.
            beta=1.0 # weight for KL loss
    ):
        super().__init__()
        up_sampling = list(reversed(down_sampling))
        self.beta = beta
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=down_channels[0],
            kernel_size=3,
            padding=1
        )
        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels=down_channels[i],
                out_channels=down_channels[i+1],
                num_layers=levels,
                dropout_rate=dropout_rate,
                down_sample=down_sampling[i],
            ) for i in range(len(down_channels)-1)
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
                out_channels=up_channels[i+1],
                num_layers=levels,
                up_sampling=up_sampling[i],
                dropout_rate=dropout_rate
            ) for i in range(len(up_channels)-1)
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
        print("################### encoder #####################")
        x = self.conv1(x)
        print(f" x : {x.shape}")
        print("------------------ begin down blocks ----------------")
        for block in self.down_blocks:
            x = block(x)
        print("------------------ end down blocks ------------------")
        res_x = x
        print(f" res_x.shape: {res_x.shape}")
        x = self.attention1(x)
        print(f" x attention1 : {x.shape}")
        x = x + res_x
        print(f" x_att + res_x : {x.shape}")
        mu = self.conv_mu(x)
        print(f" mu : {mu.shape}")
        logvar = self.conv_logvar(x)
        print(f" logvar : {logvar.shape}")
        z_hat = self.reparameterize(mu, logvar)
        print(f" z_hat : {z_hat.shape}")
        return z_hat, mu, logvar

    def decoder(self, zq):
        print("################### decoder #####################")
        x = self.conv2(zq)
        print(f" x : {x.shape}")
        res_x = x
        print(f" res_x : {res_x.shape}")
        x = self.attention2(x)
        print(f" x_att : {x.shape}")
        x = x + res_x
        print(f" x + res_x : {x.shape}")

        print("------------------ begin up blocks ----------------")
        for block in self.up_blocks:
            x = block(x)
        print("------------------ end up blocks ------------------")
        x = self.conv3(x)
        print(f" x : {x.shape}")
        return x

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_hat = self.decoder(z)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return x_hat, kl_loss * self.beta
#--------------------------------------------------------------------------------------------------
class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers, dropout_rate, down_sample):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i==0 else out_channels,
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
            down_sampling_factor=2
        ) if down_sample else nn.Identity()
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

        return output
#-----------------------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------------------
class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling_factor):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                      kernel_size=4, stride=down_sampling_factor, padding=1)
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=down_sampling_factor, stride=down_sampling_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2,
                      kernel_size=1, stride=1, padding=0)
        )

    def forward(self, batch):
        return torch.cat(tensors=[self.conv(batch), self.pool(batch)], dim=1)
#-----------------------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------------------

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, up_sampling, dropout_rate):
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
        self.up_sampling = UpSampling(
            in_channels=in_channels,
            out_channels=in_channels//2,
            up_sampling_factor=2
        ) if up_sampling else nn.Identity()
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(self.num_layers)

        ])

    def forward(self, x):
        x = self.up_sampling(x)
        print(f" x up sampling : {x.shape}")
        output = x
        print(f" first output : {output.shape}")
        print("-------------begin layers--------------")
        for i in range(self.num_layers):
            resnet_input = output
            print(f" resnet_input {i}: {resnet_input.shape}")
            output = self.conv1[i](output)
            print(f" conv1: {output.shape}")
            output = self.conv2[i](output)
            print(f" conv2 : {output.shape}")
            output = output + self.resnet[i](resnet_input)
            print(f" output + resnet_input : {output.shape}")
        print("-------------end layers--------------")
        return output
#--------------------------------------------------------------------------------------
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, up_sampling_factor):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels//2,
                kernel_size=4,
                stride=up_sampling_factor,
                padding=1
            ),
            nn.Conv2d(
                in_channels=out_channels//2,
                out_channels=out_channels//2,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=up_sampling_factor, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2,
                      kernel_size=1, stride=1, padding=0)
        )

    def forward(self, batch):
        return torch.cat(tensors=[self.conv(batch), self.up_sample(batch)], dim=1)





def test_variational_latent_compressor():
    # Example parameters
    in_channels = 3
    out_channels = 3
    down_channels = [32, 64, 128, 256]
    up_channels = [256, 128, 64, 32]
    dropout_rate = 0.1
    num_heads = 4
    num_groups = 8
    levels = 1
    down_sampling = [True, True, True]
    beta = 1.0

    # Create a random input tensor (Batch size 2, RGB image, 128x128)
    x = torch.randn(2, in_channels, 128, 128)

    # Initialize model
    model = VariationalLatentCompressor(
        in_channels, down_channels, up_channels, out_channels,
        dropout_rate, num_heads, num_groups, levels, down_sampling, beta
    )

    # Forward pass
    x_hat, kl_loss = model(x)

    # Check output shape
    assert x_hat.shape == x.shape, f"Expected output shape {x.shape}, but got {x_hat.shape}"
    assert kl_loss.shape == torch.Size([]), f"KL loss should be a scalar, got shape {kl_loss.shape}"

    print("Test passed! Model runs successfully and outputs correct shapes.")

# Run the test
test_variational_latent_compressor()
