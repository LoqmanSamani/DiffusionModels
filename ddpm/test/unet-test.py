import torch
from torchsummary import summary
from unet import UNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_unet():

    batch_size = 1
    in_channels = 1
    out_channels = 2
    height, width = 572, 572

    model = UNet(in_channels=in_channels, out_channels=out_channels)

    print("\nModel Summary:")
    summary(model, input_size=(in_channels, height, width))

    num_parameters = count_parameters(model)
    print(f"\nTotal number of trainable parameters: {num_parameters}")

    input_tensor = torch.randn(batch_size, in_channels, height, width)

    output = model(input_tensor)

    expected_height = height - 184
    expected_width = width - 184
    expected_shape = (batch_size, out_channels, expected_height, expected_width)

    print(f"\nInput Shape: {input_tensor.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Expected Output Shape: {expected_shape}")

    assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} != {expected_shape}"

    print("U-Net test passed!")


test_unet()