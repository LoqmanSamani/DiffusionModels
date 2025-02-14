import torch
from unet import DownSampling

def test_DownSampling():

    batch_size = 2
    in_channels = 16
    out_channels = 32
    height, width = 8, 8  # Feature map size
    down_sampling_factor = 2

    x = torch.randn(batch_size, in_channels, height, width)
    model = DownSampling(in_channels, out_channels, down_sampling_factor, conv_block=True, max_pool=True)
    output = model(x)

    expected_height, expected_width = height // down_sampling_factor, width // down_sampling_factor
    expected_out_channels = out_channels
    assert output.shape == (batch_size, expected_out_channels, expected_height, expected_width), \
        f"Expected shape {(batch_size, expected_out_channels, expected_height, expected_width)}, but got {output.shape}"

    print("Test passed!  Output shape:", output.shape)


test_DownSampling()
