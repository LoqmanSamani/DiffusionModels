import torch
from generate import Generate
from config import Config

def test_generate():

    device = "cpu"
    config = Config(
        model_path="test_model.pth",
        output_shape=(3, 32, 32),
        train_epochs=2,
        in_channels=3,
        beta_range=(1e-4, 0.02),
        beta_method="linear",
        num_steps=1000,
        num_tau_steps=100,
        device=device,
        eta=0,
        image_shape=32
    )

    generate = Generate(config)
    output = generate.forward()
    expected_shape = (config.num_tau_steps, config.in_channels, config.image_shape, config.image_shape)
    assert output.shape == expected_shape, \
        f"unexpected output shape: {output.shape}, expected: {expected_shape}"

    assert torch.all((output >= 0) & (output <= 1)), "generated images contain values outside [0,1]"

    print(f"test passed! generated output shape: {output.shape}")

test_generate()