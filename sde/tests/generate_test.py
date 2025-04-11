import torch
from generate import Generate
from config import Config



def test_generate():

    config = Config(
        in_channels=3,
        down_channels=[32, 64, 128],
        mid_channels=[128, 128],
        up_channels=[128, 64, 32, 3],
        down_sampling=[True, True, False],
        num_groups=8,
        image_shape=(3, 32, 32),
        max_steps=10,
        device="cpu"
    )

    class MockModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    model = MockModel()
    for method in ["ve", "vp", "sub-vp"]:
        config.method = method
        generator = Generate(config, model, device="cpu")
        output = generator.forward(,,,,,,,,,,

        assert isinstance(output, torch.Tensor), "Output should be a torch Tensor"
        assert output.shape == (config.max_steps, *config.image_shape), "Output shape mismatch"
        assert output.min() >= 0 and output.max() <= 1, "Output should be normalized between 0 and 1"

    print("Test passed!")


test_generate()