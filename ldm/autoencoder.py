import torch
import torch.nn as nn
import torch.nn.functional as F




class LDMAutoencoder(nn.Module):
    def __init__(
            self,
            in_channels,  # number of channels of the original image. e.g., 3 for RBG.
            down_channels,  # a list of channels used in encoder. e.g., [32, 64, 128, 256].
            up_channels,  # a list of channels used in decoder. e.g., [256, 128, 64, 16].
            out_channels,  # probably the same as in_channels. used to construct the image.
            dropout_rate,  # dropout rate, prevents overfitting.
            num_heads,  # number of attention heads in self-attention layers.
            num_groups,  # number of groups in group normalization. used in self-attention.
            num_layers_per_block, # number of convolutional layers within each down/up block.
            total_down_sampling_factor,  # total down-sampling factor, used to calculate down sampling factor: an integer used to down/up sample the input batch of images.
            latent_channels,  # final z channels for DM.
            num_embeddings, # number of discrete embeddings in the codebook/dimensionality of each embedding vector. in case of using VectorQuantizer
            use_vq=False, # flag to toggle between vq regularization and kl regularization; if false, uses kl.
            beta=1.0 # weight for KL loss.

    ):
        super().__init__()
        assert in_channels == out_channels, "Input and output channels must match for auto-encoding"
        self.use_vq = use_vq
        self.beta = beta
        self.current_beta = beta
        num_down_blocks = len(down_channels) - 1
        self.down_sampling_factor = int(total_down_sampling_factor ** (1 / num_down_blocks))

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels=down_channels[i],
                out_channels=down_channels[i + 1],
                num_layers=num_layers_per_block,
                down_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate
            ) for i in range(num_down_blocks)
        ])
        self.attention1 = Attention(down_channels[-1], num_heads, num_groups, dropout_rate)

        # Latent projection
        if use_vq:
            self.vq_layer = VectorQuantizer(num_embeddings, down_channels[-1])
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)
        else:
            self.conv_mu = nn.Conv2d(down_channels[-1], down_channels[-1], kernel_size=3, padding=1)
            self.conv_logvar = nn.Conv2d(down_channels[-1], down_channels[-1], kernel_size=3, padding=1)
            self.quant_conv = nn.Conv2d(down_channels[-1], latent_channels, kernel_size=1)

        # Decoder
        self.conv2 = nn.Conv2d(latent_channels, up_channels[0], kernel_size=3, padding=1)
        self.attention2 = Attention(up_channels[0], num_heads, num_groups, dropout_rate)
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=up_channels[i],
                out_channels=up_channels[i + 1],
                num_layers=num_layers_per_block,
                up_sampling_factor=self.down_sampling_factor,
                dropout_rate=dropout_rate
            ) for i in range(len(up_channels) - 1)
        ])
        self.conv3 = Conv3(up_channels[-1], out_channels, dropout_rate)

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
        if self.use_vq:
            z, vq_loss = self.vq_layer(x)
            z = self.quant_conv(z)
            return z, vq_loss
        else:
            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            z = self.reparameterize(mu, logvar)
            z = self.quant_conv(z)
            kl_unnormalized = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            batch_size = x.size(0)
            latent_size = torch.prod(torch.tensor(mu.shape[1:])).item()
            kl_loss = kl_unnormalized / (batch_size * latent_size) * self.current_beta
            return z, kl_loss

    def decoder(self, z):
        x = self.conv2(z)
        res_x = x
        x = self.attention2(x)
        x = x + res_x
        for block in self.up_blocks:
            x = block(x)
        x = self.conv3(x)
        return x

    def forward(self, x):
        z, reg_loss = self.encoder(x)
        x_hat = self.decoder(z)
        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_loss + reg_loss
        return x_hat, total_loss, reg_loss, z  # return z for DM
#------------------------------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        # dimensionality of each embedding vector
        self.embedding_dim = embedding_dim
        # number of discrete embeddings in the codebook
        self.num_embeddings = num_embeddings
        # commitment cost for the loss term to encourage z to be close to quantized values
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # initialize embedding weights uniformly
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        z = z.contiguous() # ensure contingency in memory
        # flatten z to (batch_size * height * width, embedding_dim) for distance computation
        assert z.size(1) == self.embedding_dim, f"Expected channel dim {self.embedding_dim}, got {z.size(1)}"
        z_flattened = z.reshape(-1, self.embedding_dim)
        # compute squared euclidean distances between z_flattened and all embeddings
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))
        # find the index of the closest embedding for each z_flattened vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # convert indices to one-hot encodings
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float().squeeze(1)
        # map one-hot encodings to quantized values using the embedding weights
        quantized = torch.matmul(encodings, self.embedding.weight).view_as(z)
        # commitment loss to encourage z to be close to its quantized version
        commitment_loss = self.commitment_cost * torch.mean((z.detach() - quantized) ** 2)
        # codebook loss to encourage embeddings to move closer to z
        codebook_loss = torch.mean((z - quantized.detach()) ** 2)
        # straight-through estimator: copy gradients from quantized to z
        quantized = z + (quantized - z).detach()
        # return the quantized tensor and the combined vq loss
        return quantized, commitment_loss + codebook_loss
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