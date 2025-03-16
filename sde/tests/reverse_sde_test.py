import torch
from config import Config
from reverse_diffusion import ReverseSDE




def test_reverse_sde():
    """test the ReverseSDE class to ensure correct denoising behavior."""

    class DummyModel:
        """a simple model that predicts zero noise for testing purposes."""

        def __call__(self, x, t):
            return torch.zeros_like(x)

    config = Config(max_steps=100)
    reverse_sde = ReverseSDE(config, model=DummyModel())

    x_noisy = torch.randn(5, 3, 100, 100)
    t = torch.randint(1, config.max_steps, (x_noisy.shape[0],))

    for method in ["smld", "ddim", "subvp"]:
        config.method = method
        x_denoised = reverse_sde.forward(x_noisy.clone())

        assert x_denoised.shape == x_noisy.shape, f"Reverse {method}: Shape mismatch"
        assert not torch.equal(x_noisy, x_denoised), f"Reverse {method}: x should change after denoising"

    print("ReverseSDE tests passed.")


test_reverse_sde()