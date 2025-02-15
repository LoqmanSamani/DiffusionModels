import torch
from unet import UpBlock


def test_upblock():


    batch_size = 4
    in_channels = 128
    out_channels = 64
    height, width = 16, 16
    up_height, up_width = height * 2, width * 2
    time_embed_dim = 128
    num_layers = 2

    batch = torch.randn(batch_size, in_channels, height, width)
    skip_connection = torch.randn(batch_size, in_channels // 2, up_height, up_width)
    embed_time = torch.randn(batch_size, time_embed_dim)

    up_block = UpBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        time_embed_dim=time_embed_dim,
        num_layers=num_layers,
        up_sampling=True
    )

    output = up_block(batch, skip_connection, embed_time)
    expected_shape = (batch_size, out_channels, up_height, up_width)
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"

    print("UpBlock test passed!")


test_upblock()