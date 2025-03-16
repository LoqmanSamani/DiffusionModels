import torch
from forward_diffusion import ForwardSDE
from config import Config



def test_forward_sde():
    """test the ForwardSDE class to ensure correct noise addition."""
    config = Config(max_steps=100)
    forward_sde = ForwardSDE(config)

    x = torch.randn(5, 3, 100, 100)
    t = torch.randint(1, config.max_steps, (x.shape[0],))

    for method in ["smld", "ddim", "subvp"]:
        config.method = method
        x_noisy = forward_sde.forward(x.clone())

        assert x_noisy.shape == x.shape, f"Forward {method}: Shape mismatch"
        assert not torch.equal(x, x_noisy), f"Forward {method}: x should change after diffusion"

    print("ForwardSDE tests passed.")


test_forward_sde()