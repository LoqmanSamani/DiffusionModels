import torch
import math
import torch.nn as nn



class UNet(nn.Module):
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
            num_up_blocks,
            dropout_rate,
            down_sampling_factor=2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.up_channels = up_channels
        self.down_sampling = down_sampling
        self.time_embed_dim = time_embed_dim
        self.num_down_blocks = num_down_blocks
        self.num_mid_blocks = num_mid_blocks
        self.num_up_blocks = num_up_blocks
        self.dropout_rate = dropout_rate
        self.up_sampling = list(reversed(self.down_sampling))
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
                down_sampling_factor=down_sampling_factor,
                down_sample=self.down_sampling[i],
                dropout_rate=self.dropout_rate
            ) for i in range(len(self.down_channels)-1)
        ])
        # middle blocks
        self.mid_blocks = nn.ModuleList([
            MiddleBlock(
                in_channels=self.mid_channels[i],
                out_channels=self.mid_channels[i+1],
                time_embed_dim=self.time_embed_dim,
                num_layers=self.num_mid_blocks,
                dropout_rate=self.dropout_rate
            ) for i in range(len(self.mid_channels)-1)
        ])
        # up blocks
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=self.up_channels[i],
                out_channels=self.up_channels[i+1],
                time_embed_dim=self.time_embed_dim,
                num_layers=self.num_up_blocks,
                up_sampling_factor=down_sampling_factor,
                up_sampling=self.up_sampling[i],
                dropout_rate=self.dropout_rate
            ) for i in range(len(self.up_channels)-1)
        ])
        # final convolution layer
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=self.up_channels[-1]),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(in_channels=self.up_channels[-1], out_channels=self.in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, y=None, where_y=False):

        if y is not None and not where_y:
            x = torch.cat(tensors=[x, y], dim=1)

        output = self.conv1(x)
        time_embed = GetEmbeddedTime(embed_dim=self.time_embed_dim)(time_steps=t)
        time_embed = self.time_projection(time_embed)
        skip_connections = []

        if y is not None and where_y:
            for i, down in enumerate(self.down_blocks):
                skip_connections.append(output)
                output = down(x=output, embed_time=time_embed, y=y)
            for i, mid in enumerate(self.mid_blocks):
                output = mid(x=output, embed_time=time_embed, y=y)
            for i, up in enumerate(self.up_blocks):
                skip_connection = skip_connections.pop()
                output = up(x=output, skip_connection=skip_connection, embed_time=time_embed, y=y)
        else:
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
    def __init__(self, in_channels, out_channels, time_embed_dim, num_layers, down_sampling_factor,  down_sample=True, dropout_rate=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i==0 else out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
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
                norm=True,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.down_sampling = DownSampling(
            in_channels=out_channels,
            out_channels=out_channels,
            down_sampling_factor=down_sampling_factor,
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

    def forward(self, x, embed_time, y=None):
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = output + self.time_embedding[i](embed_time)[:, :, None, None]
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
            if y is not None:
                out_attn = self.attention[i](output, y)
            else:
                out_attn = self.attention[i](output)
            output = output + out_attn
        output = self.down_sampling(output)
        return output
#------------------------------------------------------------------------------
class MiddleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers+1)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
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
                norm=True,
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

    def forward(self, x, embed_time, y=None):
        output = x
        resnet_input = output
        output = self.conv1[0](output)
        output = output + self.time_embedding[0](embed_time)[:, :, None, None]
        output = self.conv2[0](output)
        output = output + self.resnet[0](resnet_input)
        for i in range(self.num_layers):
            if y is not None:
                out_attn = self.attention[i](output, y)
            else:
                out_attn = self.attention[i](output)
            output = output + out_attn
            resnet_input = output
            output = self.conv1[i + 1](output)
            output = output + self.time_embedding[i + 1](embed_time)[:, :, None, None]
            output = self.conv2[i + 1](output)
            output = output + self.resnet[i+1](resnet_input)
        return output
#------------------------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, num_layers, up_sampling_factor, up_sampling=True, dropout_rate=0.2):
        super().__init__()
        self.num_layers = num_layers
        effective_in_channels = in_channels
        self.conv1 = nn.ModuleList([
            Conv3(
                in_channels=effective_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
            ) for i in range(self.num_layers)
        ])
        self.conv2 = nn.ModuleList([
            Conv3(
                in_channels=out_channels,
                out_channels=out_channels,
                num_groups=8,
                kernel_size=3,
                norm=True,
                activation=True,
                dropout_rate=dropout_rate
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
                norm=True,
                dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ])
        self.up_sampling = UpSampling(
            in_channels=in_channels,
            out_channels=in_channels,
            up_sampling_factor=up_sampling_factor,
            conv_block=True,
            up_sampling=True
        ) if up_sampling else nn.Identity()
        self.resnet = nn.ModuleList([
            nn.Conv2d(
                in_channels=effective_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1
            ) for i in range(num_layers)

        ])

    def forward(self, x, skip_connection, embed_time, y=None):
        x = self.up_sampling(x)
        x = torch.cat(tensors=[x, skip_connection], dim=1)
        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.conv1[i](output)
            output = output + self.time_embedding[i](embed_time)[:, :, None, None]
            output = self.conv2[i](output)
            output = output + self.resnet[i](resnet_input)
            if y is not None:
                out_attn = self.attention[i](output, y)
            else:
                out_attn = self.attention[i](output)
            output = output + out_attn
        return output
#------------------------------------------------------------------------
class Conv3(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8, kernel_size=3, norm=True, activation=True, dropout_rate=0.2):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels) if norm else nn.Identity()
        self.activation = nn.SiLU() if activation else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, batch):
        batch = self.group_norm(batch)
        batch = self.activation(batch)
        batch = self.dropout(batch)
        batch = self.conv(batch)
        return batch
#----------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, output_dim, embed_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=embed_dim, out_features=output_dim)
        )
    def forward(self, batch):
        return self.embedding(batch)
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
#----------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, num_channels, num_groups=8, num_heads=4, norm=True, dropout_rate=0.2):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels) if norm else nn.Identity()
        self.attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, y=None):

        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h * w)
        x = self.group_norm(x)
        x = x.transpose(1, 2)
        if y is not None:
            assert x.shape[-1] == y.shape[-1], "x and y must have the same number of channels (embed_dim)"
            x, _ = self.attention(x, y, y)
        else:
            x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)

        return x
#-----------------------------------------------------------------
class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling_factor, conv_block=True, max_pool=True):
        super().__init__()
        self.conv_block = conv_block
        self.max_pool = max_pool
        self.down_sampling_factor = down_sampling_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2 if max_pool else out_channels,
                      kernel_size=3, stride=down_sampling_factor, padding=1)
        ) if conv_block else nn.Identity()
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=down_sampling_factor, stride=down_sampling_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2 if conv_block else out_channels,
                      kernel_size=1, stride=1, padding=0)
        ) if max_pool else nn.Identity()

    def forward(self, batch):
        if not self.conv_block:
            return self.pool(batch)
        if not self.max_pool:
            return self.conv(batch)
        return torch.cat(tensors=[self.conv(batch), self.pool(batch)], dim=1)
#--------------------------------------------------------------------------
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, up_sampling_factor, conv_block=True, up_sampling=True):
        super().__init__()
        self.conv_block = conv_block
        self.up_sampling = up_sampling
        self.up_sampling_factor = up_sampling_factor
        half_out_channels = out_channels // 2
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=half_out_channels if up_sampling else out_channels,
                kernel_size=3,
                stride=up_sampling_factor,
                padding=1,
                output_padding=up_sampling_factor - 1
            ),
            nn.Conv2d(
                in_channels=half_out_channels if up_sampling else out_channels,
                out_channels=half_out_channels if up_sampling else out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ) if conv_block else nn.Identity()

        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=up_sampling_factor, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=half_out_channels if conv_block else out_channels,
                      kernel_size=1, stride=1, padding=0)
        ) if up_sampling else nn.Identity()

    def forward(self, batch):
        if not self.conv_block:
            return self.up_sample(batch)
        if not self.up_sampling:
            return self.conv(batch)
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
#-----------------------------------------------------------------------------------------------------------------------
class Encoder(torch.nn.Module):
    def __init__(
            self,
            vocabulary_size,
            num_layers=6,
            input_dimension=512,
            output_dimension=512,
            num_heads=8,
            context_length=512,
            dropout_rate=0.1,
            qkv_bias=False,
            scaling_value=4,
            epsilon=1e-5
    ):
        super().__init__()
        self.embedding = Embedding(
            vocabulary_size=vocabulary_size,
            embedding_dimension=input_dimension,
            context_length=context_length
        )
        self.layers = torch.nn.ModuleList([
            EncoderLayer(
                input_dimension=input_dimension,
                output_dimension=output_dimension,
                num_heads=num_heads,
                context_length=context_length,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias,
                scaling_value=scaling_value,
                epsilon=epsilon
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x
#-----------------------------------------------------------------------------------------------------------------------
class EncoderLayer(torch.nn.Module):
    def __init__(
            self,
            input_dimension,
            output_dimension,
            num_heads,
            context_length,
            dropout_rate,
            qkv_bias,
            scaling_value,
            epsilon
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            num_heads=num_heads,
            context_length=context_length,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias
        )
        self.feedforward = FeedForward(
            embedding_dimension=input_dimension,
            scaling_value=scaling_value
        )
        self.norm1 = LayerNorm(
            embedding_dimension=input_dimension,
            epsilon=epsilon
        )
        self.norm2 = LayerNorm(
            embedding_dimension=input_dimension,
            epsilon=epsilon
        )

    def forward(self, x):

        attention_residual = x
        x = self.attention(x)
        x = self.norm1(x + attention_residual)
        ff_residual = x
        x = self.feedforward(x)
        x = self.norm2(x + ff_residual)

        return x
#---------------------------------------------------------------------------------------------------------------------
class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        input_dimension=512,
        output_dimension=512,
        num_heads=8,
        context_length=512,
        dropout_rate=0.1,
        qkv_bias=False,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.num_heads = num_heads
        self.head_dimension = self.input_dimension // self.num_heads
        assert self.input_dimension % self.num_heads == 0, "Input dimension must be divisible by the number of heads."

        self.Wq = torch.nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.Wk = torch.nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.Wv = torch.nn.Linear(input_dimension, output_dimension, bias=qkv_bias)

        self.out_project = torch.nn.Linear(output_dimension, output_dimension)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        )

    #def create_causal_mask(self, seq_length):
        #return torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1)

    def forward(self, x, y=None, apply_mask=False):
        batch_size, num_tokens, input_dimension = x.shape
        assert input_dimension == self.input_dimension, "Input dimension mismatch."

        if y is not None:
            Q = self.Wq(y)
            K = self.Wk(y)
            V = self.Wv(x)
        else:
            Q = self.Wq(x)
            K = self.Wk(x)
            V = self.Wv(x)

        Q = Q.view(batch_size, num_tokens, self.num_heads, self.head_dimension).transpose(1, 2)
        K = K.view(batch_size, num_tokens, self.num_heads, self.head_dimension).transpose(1, 2)
        V = V.view(batch_size, num_tokens, self.num_heads, self.head_dimension).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dimension ** 0.5)
        if apply_mask:
            causal_mask = self.mask[:num_tokens, :num_tokens]
            attention_scores = attention_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = torch.matmul(attention_weights, V)
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.output_dimension)
        output = self.out_project(context_vector)

        return output
#-----------------------------------------------------------------------------------------------------------------------
class FeedForward(torch.nn.Module):
    def __init__(self, embedding_dimension=512, scaling_value=4):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * scaling_value,
                bias=True
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=embedding_dimension * scaling_value,
                out_features=embedding_dimension,
                bias=True
            )
        )
    def forward(self, x):
        return self.layers(x)
#-----------------------------------------------------------------------------------------------------------------------
class LayerNorm(torch.nn.Module):
    def __init__(self, embedding_dimension=512, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = torch.nn.parameter.Parameter(torch.ones(embedding_dimension))
        self.beta = torch.nn.parameter.Parameter(torch.zeros(embedding_dimension))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        variance = x.var(dim=-1, keepdims=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(variance + self.epsilon)
        return self.gamma * x_norm + self.beta
#-----------------------------------------------------------------------------------------------------------------------
class Embedding(torch.nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dimension=512,
        context_length=512
    ):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension
        )
        self.embedding_dimension = embedding_dimension
        self.context_length = context_length
        self.register_buffer("positional_encoding", self._generate_positional_encoding(context_length))

    def _generate_positional_encoding(self, seq_len):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dimension, 2, dtype=torch.float) *
                             -(math.log(10000.0) / self.embedding_dimension))
        pos_enc = torch.zeros((seq_len, self.embedding_dimension), device=position.device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, token_ids):
        assert token_ids.dim() == 2, "Input token_ids should be of shape (batch_size, seq_len)"
        token_embedded = self.token_embedding(token_ids)
        seq_len = token_ids.size(1)
        if seq_len > self.context_length:
            position_encoded = self._generate_positional_encoding(seq_len).to(token_embedded.device)
        else:
            position_encoded = self.positional_encoding[:, :seq_len, :].to(token_embedded.device)
        return token_embedded + position_encoded




import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Assuming your UNet and Encoder classes are in a file named `net.py`
#from net import UNet, Encoder

class DummyDataset(Dataset):
    def __init__(self, num_samples=32, image_size=64, vocab_size=30522):
        self.num_samples = num_samples
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy image (random noise)
        image = torch.randn(3, self.image_size, self.image_size)
        # Dummy text tokens (random token IDs)
        text = torch.randint(0, self.vocab_size, (20,))  # Sequence length 20
        return image, text
def test_unet_and_encoder():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Hyperparameters
    batch_size = 32
    image_size = 64  # Latent space size (e.g., 256/4 = 64 for f=4)
    latent_channels = 3  # As in LDM-4
    vocab_size = 30522  # BERT tokenizer vocab size
    context_length = 77  # Typical max length for BERT tokenization
    embed_dim = 512  # Match the Encoder's output_dimension

    # Create dummy dataset and dataloader
    dataset = DummyDataset(num_samples=batch_size, image_size=image_size, vocab_size=vocab_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize UNet with channel dimensions matching embed_dim
    unet = UNet(
        in_channels=latent_channels,
        down_channels=[embed_dim, embed_dim, embed_dim],  # Match embed_dim
        mid_channels=[embed_dim, embed_dim, embed_dim],             # Match embed_dim
        up_channels=[embed_dim, embed_dim, embed_dim],   # Match embed_dim
        down_sampling=[True, True, False],
        time_embed_dim=128,
        num_down_blocks=2,
        num_mid_blocks=1,
        num_up_blocks=2,
        dropout_rate=0.1
    )

    # Initialize Transformer Encoder
    encoder = Encoder(
        vocabulary_size=vocab_size,
        num_layers=6,
        input_dimension=embed_dim,
        output_dimension=embed_dim,
        num_heads=8,
        context_length=context_length,
        dropout_rate=0.1
    )

    # Move models to device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    unet = unet.to(device)
    encoder = encoder.to(device)
    unet.train()
    encoder.train()

    # Define optimizer
    optimizer = torch.optim.Adam(list(unet.parameters()) + list(encoder.parameters()), lr=1e-4)

    print("Testing UNet and Encoder for 1 epoch...")

    # Single epoch with one batch
    for images, texts in data_loader:
        images = images.to(device)  # Shape: (32, 3, 64, 64)
        texts = texts.to(device)    # Shape: (32, 20)

        # Timestep (randomly sampled)
        t = torch.randint(1, 1000, (batch_size,), device=device)

        # Encode text prompts
        text_embeddings = encoder(texts)  # Shape: (32, 20, 512)
        print(f"Text embeddings shape: {text_embeddings.shape}")

        # Forward pass through UNet
        noise_pred = unet(images, t, y=text_embeddings, where_y=True)
        print(f"Input shape: {images.shape}")  # Expected: (32, 3, 64, 64)
        print(f"Output (noise_pred) shape: {noise_pred.shape}")  # Expected: (32, 3, 64, 64)

        # Shape checks
        assert noise_pred.shape == images.shape, f"Expected output shape {images.shape}, got {noise_pred.shape}"

        # Dummy loss (MSE between predicted noise and input for simplicity)
        loss = torch.nn.functional.mse_loss(noise_pred, images)
        print(f"Loss: {loss.item():.4f}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        break  # Only one batch for one epoch

    print("Test completed successfully! Models work as expected with correct input/output dimensions.")
if __name__ == "__main__":
    test_unet_and_encoder()