import torch
from unet import UNet


def test_UNet():

    batch_size = 2
    in_channels = 3
    img_size = 64


    x = torch.randn(batch_size, in_channels, img_size, img_size)
    t = torch.randint(0, 1000, (batch_size,))

    model = UNet(
        in_channels=in_channels,
        down_channels=[32, 64, 128, 256],
        mid_channels=[256, 256, 128],
        up_channels=[256, 128, 64, 16],
        down_sampling=[True, True, False],
        time_embed_dim=128,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2
    )

    output = model(x, t)


    assert output.shape == x.shape, f"Expected output shape {x.shape}, but got {output.shape}"

    print("UNet test passed! Output shape:", output.shape)

test_UNet()