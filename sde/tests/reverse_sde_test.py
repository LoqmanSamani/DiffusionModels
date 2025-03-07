import torch
from config import Config
from reverse_diffusion import ReverseSDE




def test_reverse_sde():
    """test the ReverseSDE class to ensure it denoises correctly."""

    class DummyModel:
        def __call__(self, x, t):
            return torch.zeros_like(x)  # dummy model predicts zero noise

    config = Config(max_steps=100)
    reverse_sde = ReverseSDE(config, model=DummyModel())

    x_noisy = torch.randn(5, 3)  # start with a noisy input

    for method in ["smld", "ddim", "subvp"]:
        config.method = method
        x_denoised = reverse_sde.forward(x_noisy.clone())

        # ensure x is changing in the reverse process
        assert not torch.equal(x_noisy, x_denoised), f"Reverse {method}: x should change"

        # check that the variance decreases (denoising effect)
        assert x_denoised.std() < x_noisy.std(), f"Reverse {method}: Noise should decrease"

    print("✅ ReverseSDE tests passed.")


test_reverse_sde()