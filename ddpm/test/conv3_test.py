import torch
from unet import Conv3

def test_Conv3():

    in_channels = 8
    out_channels = 16
    height, width = 32, 32
    batch_size = 4

    x = torch.randn(batch_size, in_channels, height, width)
    model = Conv3(in_channels, out_channels)

    output = model(x)

    assert output.shape == (batch_size, out_channels, height, width), \
        f"Expected shape {(batch_size, out_channels, height, width)}, but got {output.shape}"

    print("Test passed! Output shape:", output.shape)


test_Conv3()
