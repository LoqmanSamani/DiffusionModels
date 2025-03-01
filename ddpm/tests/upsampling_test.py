import torch
from network import UpSampling


def test_UpSampling():

    batch_size = 2
    in_channels = 16
    out_channels = 32
    height, width = 8, 8
    up_sampling_factor = 2

    x = torch.randn(batch_size, in_channels, height, width)
    model = UpSampling(in_channels, out_channels, up_sampling_factor, conv_block=True, up_sampling=True)
    output = model(x)
    expected_height = height * up_sampling_factor
    expected_width = width * up_sampling_factor
    expected_out_channels = out_channels
    assert output.shape == (batch_size, expected_out_channels, expected_height, expected_width), \
        f"Expected shape {(batch_size, expected_out_channels, expected_height, expected_width)}, but got {output.shape}"

    print("Test passed!  Output shape:", output.shape)


test_UpSampling()