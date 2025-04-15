import torch
import torch.nn as nn
import math
from transformers import BertModel, DistilBertModel




class TextEncoder(torch.nn.Module):
    def __init__(
            self,
            use_pretrained_model=True,
            model_name="bert-base-uncased",
            vocabulary_size=30522,
            num_layers=6,
            input_dimension=768,
            output_dimension=768,
            num_heads=8,
            context_length=77,
            dropout_rate=0.1,
            qkv_bias=False,
            scaling_value=4,
            epsilon=1e-5
    ):
        super().__init__()
        self.use_pretrained_model = use_pretrained_model
        if self.use_pretrained_model:
            # self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.bert = BertModel.from_pretrained(model_name)
            for param in self.bert.parameters():
                param.requires_grad = False
            self.projection = nn.Linear(self.bert.config.hidden_size, output_dimension)
        else:
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
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias,
                    scaling_value=scaling_value,
                    epsilon=epsilon
                )
                for _ in range(num_layers)
            ])
    def forward(self, x, attention_mask=None):
        if self.use_pretrained_model:
            x = self.bert(input_ids=x, attention_mask=attention_mask)
            x = x.last_hidden_state
            x = self.projection(x)
        else:
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x, attention_mask=attention_mask)
        return x
#-----------------------------------------------------------------------------------------------------------------------
class EncoderLayer(torch.nn.Module):
    def __init__(
            self,
            input_dimension,
            output_dimension,
            num_heads,
            dropout_rate,
            qkv_bias,
            scaling_value,
            epsilon=1e-5
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dimension,
            num_heads=num_heads,
            dropout=dropout_rate,
            bias=qkv_bias,
            batch_first=True
        )
        self.output_projection = nn.Linear(input_dimension, output_dimension) if input_dimension != output_dimension else nn.Identity()
        self.norm1 = nn.LayerNorm(normalized_shape=input_dimension, eps=epsilon)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.feedforward = FeedForward(
            embedding_dimension=input_dimension,
            scaling_value=scaling_value,
            dropout_rate=dropout_rate
        )
        self.norm2 = nn.LayerNorm(normalized_shape=output_dimension, eps=epsilon)
        self.dropout2 = nn.Dropout(dropout_rate)
    def forward(self, x, attention_mask=None):
        attn_output, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        attn_output = self.output_projection(attn_output)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x
#-----------------------------------------------------------------------------------------------------------------------
class FeedForward(torch.nn.Module):
    def __init__(self, embedding_dimension, scaling_value, dropout_rate=0.1):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * scaling_value,
                bias=True
            ),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(
                in_features=embedding_dimension * scaling_value,
                out_features=embedding_dimension,
                bias=True
            )
        )
    def forward(self, x):
        return self.layers(x)
#-----------------------------------------------------------------------------------------------------------------------
class Embedding(torch.nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dimension=768,
        context_length=77
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
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