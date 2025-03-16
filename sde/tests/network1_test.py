import torch
from config import Config
from network1 import DDPMpp




def test_ddpmpp():
    # create a sample config
    config = Config(
        in_channels=3,
        down_channels=[32, 64, 128, 256],
        mid_channels=[256, 256, 128],
        up_channels=[256, 128, 64, 16],
        down_sampling=[True, True, False],
        embed_dim=256,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        dropout_rate=0.2,
        num_attention_heads=4,
        down_sampling_factor=2,
        upsampling_factor=2,
        apply_down_conv=True,
        apply_down_pool=True,
        apply_up_conv=True,
        num_groups=8,
        kernel_size=3,
        norm=True,
        activation=True
    )

    # initialize the model
    model = DDPMpp(config)

    # create a dummy input tensor
    batch_size = 2
    image_size = 64
    x = torch.randn(batch_size, config.in_channels, image_size, image_size)
    t = torch.randint(0, 1000, (batch_size,))  # dummy time steps

    # forward pass
    output = model(x, t)

    # check output shape
    assert output.shape == x.shape, f"Expected output shape {x.shape}, but got {output.shape}"

    print("Test passed: DDPMpp produces the expected output shape!")


test_ddpmpp()