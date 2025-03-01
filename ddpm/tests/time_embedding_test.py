import torch
from network import TimeEmbedding



def test_TimeEmbedding():

    batch_size = 4
    embed_dim = 128
    output_dim = 256

    x = torch.randn(batch_size, embed_dim)
    model = TimeEmbedding(output_dim=output_dim, embed_dim=embed_dim)
    output = model(x)
    assert output.shape == (batch_size, output_dim), \
        f"Expected shape {(batch_size, output_dim)}, but got {output.shape}"

    print("Test passed!  Output shape:", output.shape)


test_TimeEmbedding()
