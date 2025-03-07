import torch
from forward_diffusion import ForwardSDE
from config import Config



def test_forward_sde():
    """Test the ForwardSDE class for all three methods."""
    config = Config(max_steps=100)
    forward_sde = ForwardSDE(config)

    x = torch.randn(5, 3)  # simulated batch of 5 samples, 3-dimensional

    for method in ["smld", "ddim", "subvp"]:
        config.method = method
        x_noisy = forward_sde.forward(x.clone())

        # ensure x has changed after diffusion
        assert not torch.equal(x, x_noisy), f"Forward {method}: x should change over time"

        # check if noise increases over time
        assert x_noisy.std() > x.std(), f"Forward {method}: Noise should increase"

    print("✅ ForwardSDE tests passed.")



test_forward_sde()