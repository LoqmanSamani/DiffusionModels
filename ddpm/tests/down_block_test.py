import torch
from network import DownBlock



def test_downblock():

    batch_size = 2
    in_channels = 8
    out_channels = 64
    height, width = 32, 32
    time_embed_dim = 128
    num_layers = 2
    down_sample = True

    batch = torch.randn(batch_size, in_channels, height, width)
    embed_time = torch.randn(batch_size, time_embed_dim)

    down_block = DownBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        time_embed_dim=time_embed_dim,
        num_layers=num_layers,
        down_sample=down_sample
    )

    output = down_block(batch, embed_time)
    print("Output shape:", output.shape)

    if down_sample:
        expected_height, expected_width = height // 2, width // 2
    else:
        expected_height, expected_width = height, width

    assert output.shape == (batch_size, out_channels, expected_height, expected_width), \
        f"Unexpected output shape: {output.shape}"

    print("Test passed!")


test_downblock()