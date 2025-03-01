import torch
from conv3 import Attention


def test_Attention():

    batch_size = 2
    num_channels = 16
    height, width = 8, 8
    num_groups = 4
    num_heads = 4

    x = torch.randn(batch_size, num_channels, height, width)
    model = Attention(num_channels=num_channels, num_groups=num_groups, num_heads=num_heads)
    output = model(x)
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, but got {output.shape}"

    print("Test passed!  Output shape:", output.shape)


test_Attention()