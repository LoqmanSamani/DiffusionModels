import torch
import torch.nn as nn
import torch.nn.functional as F



class PerceptualCompression(nn.Module):
    def __init__(self, in_channels, down_channels, up_channels, z_channels, out_channels, dropout_rate, num_heads, levels, down_sampling):
        super().__init__()
        # in_channels: number of channels of the original image. e.g., 3 for RBG.
        # down_channels: a list of channels used in encoder. e.g., [32, 64, 128, 256].
        # up_channels: a list of channels used in decoder. e.g., [256, 128, 64, 16].
        # out_channels: probably the same as in_channels. used to construct the image.
        # z_channels: number of channels used in latent space
        # num_head: number of attention heads in self-attention layers
        # levels: number of down/up sampling layers.
        # down_sampling: a list of bools, if down sampling should be applied.
        up_sampling = list(reversed(down_sampling))
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
            num_channels=z_channels,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        self.conv2 = Conv3(
            in_channels=down_channels[-1],
            out_channels=z_channels,
            dropout_rate=dropout_rate
        )
        self.conv3 = nn.Conv2d(
            in_channels=z_channels,
            out_channels=down_channels[-1],
            kernel_size=3,
            padding=1
        )
        self.attention2 = Attention(
            num_channels=down_channels[-1],
            num_heads=num_heads,
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
        self.conv4 = Conv3(
            in_channels=out_channels[-1],
            out_channels=out_channels,
            dropout_rate=dropout_rate
        )

    def encoder(self, x):
        x = self.conv1(x)
        x = self.down_blocks(x)
        res_x = x
        x = self.attention1(x)
        x = x + res_x
        z_hat = self.conv2(x)
        return z_hat

    def decoder(self, zq):
        x = self.conv3(zq)
        res_x = x
        x = self.attention2(x)
        x = x + res_x
        x = self.up_blocks(x)
        x = self.conv4(x)
        return x

    # TODO: complete quantizer function
    def quantizer(self, z_hat):
        z_q = z_hat
        indices = z_hat
        loss_vq = z_hat
        return z_q, indices, loss_vq

    def forward(self, x):
        z_e = self.encoder(x)  # Encode input image
        z_q, indices, loss_vq = self.quantizer(z_e)  # Vector quantization
        x_hat = self.decoder(z_q)  # Decode quantized representation
        return x_hat, z_q, indices, loss_vq
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
    def __init__(self, num_channels, num_heads, dropout_rate):
        super().__init__()
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

    def forward(self, x):
        x = self.up_sampling(x)
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = self.conv2[i](output)
            output = output + resnet_input
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
#------------------------------------------------------------------------------------------------
# TODO: test this class. it should be modified and renamed as vector quantization.
class PerceptualCompression(nn.Module):
    def __init__(self, encoder, decoder, codebook_size, nz):
        super().__init__()
        self.encoder = encoder  # Already implemented
        self.decoder = decoder  # Already implemented

        # Codebook: learnable tensor (K, nz)
        self.codebook = nn.Parameter(torch.randn(codebook_size, nz))

    def quantize(self, z):
        # Compute L2 distance between z and codebook entries
        z_flattened = z.view(-1, z.shape[-1])  # Shape (h*w, nz)
        codebook_sqr = torch.sum(self.codebook ** 2, dim=1)  # Shape (K,)
        z_sqr = torch.sum(z_flattened ** 2, dim=1, keepdim=True)  # Shape (h*w, 1)
        distances = z_sqr + codebook_sqr - 2 * torch.matmul(z_flattened, self.codebook.T)  # Shape (h*w, K)

        # Get closest codebook indices
        encoding_indices = torch.argmin(distances, dim=1)  # Shape (h*w,)

        # Quantized output using codebook
        z_q = self.codebook[encoding_indices].view(z.shape)  # Reshape to original z shape

        return z_q, encoding_indices

    def forward(self, x):
        # Encode image
        z = self.encoder(x)  # Shape (B, h, w, nz)

        # Quantize latent representation
        z_q, indices = self.quantize(z)

        # Decode reconstructed image
        x_recon = self.decoder(z_q)

        return x_recon, z, z_q, indices

    def loss_function(self, x, x_recon, z, z_q):
        # Reconstruction loss
        L_rec = F.mse_loss(x, x_recon)

        # Codebook loss
        L_codebook = F.mse_loss(z_q.detach(), z)

        # Commitment loss
        L_commit = F.mse_loss(z.detach(), z_q)

        # Total loss
        L_VQ = L_rec + L_codebook + L_commit
        return L_VQ


# TODO: check this correctly
def get_codebook_indices(self, x):
    _, _, indices, _ = self.forward(x)
    return indices  # Output discrete indices for transformer training

# TODO: check this also
class TransformerVQ(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, indices):
        embeddings = self.embedding(indices)  # Convert indices to embeddings
        output = self.transformer(embeddings, embeddings)  # Transformer block
        logits = self.fc_out(output)  # Predict next token
        return logits

# TODO: check this also
def train_transformer(transformer, dataset, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        for indices in dataset:  # Each image is a sequence of codebook indices
            indices = indices.to(device)
            target = indices[:, 1:]  # Shifted target sequence
            input_seq = indices[:, :-1]  # Input sequence

            optimizer.zero_grad()
            logits = transformer(input_seq)
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")