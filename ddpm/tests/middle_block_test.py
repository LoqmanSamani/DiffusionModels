import torch
from network import MiddleBlock



def test_middleblock():

    batch_size = 4
    in_channels = 64
    out_channels = 128
    height, width = 32, 32
    time_embed_dim = 128
    num_layers = 2


    batch = torch.randn(batch_size, in_channels, height, width)
    embed_time = torch.randn(batch_size, time_embed_dim)

    middle_block = MiddleBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        time_embed_dim=time_embed_dim,
        num_layers=num_layers
    )

    output = middle_block(batch, embed_time)

    assert output.shape == (batch_size, out_channels, height, width), \
        f"Expected output shape {(batch_size, out_channels, height, width)}, but got {output.shape}"

    print("MiddleBlock tests passed!")

test_middleblock()