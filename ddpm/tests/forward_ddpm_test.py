import torch
from hyper_param import HyperParamsDDPM
from forward_ddpm import ForwardDDPM





def test_forward_ddpm():
    hyper_params = HyperParamsDDPM(num_steps=1000, beta_method="sigmoid")
    forward = ForwardDDPM(hyper_params)
    x0 = torch.randn(4, 3, 32, 32)
    noise = torch.randn_like(x0)
    time_steps = torch.tensor([0, 250, 500, 750])
    xt = forward(x0, noise, time_steps)
    assert xt.shape == x0.shape, f"Expected shape {x0.shape}, got {xt.shape}"
    print("ForwardDDPM test passed!")


if __name__ == "__main__":
    test_forward_ddpm()