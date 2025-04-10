import torch
from hyper_params import HyperParams
from reverse_ddpm import ReverseDDPM




def test_reverse_ddpm():
    hyper_params = HyperParams(num_steps=1000, beta_method="quadratic")
    reverse = ReverseDDPM(hyper_params)
    xt = torch.randn(4, 3, 32, 32)
    predicted_noise = torch.randn_like(xt)
    time_steps = torch.tensor([0, 250, 500, 750])
    xt_minus_1 = reverse(xt, predicted_noise, time_steps)
    assert xt_minus_1.shape == xt.shape, f"Expected shape {xt.shape}, got {xt_minus_1.shape}"
    print("ReverseDDPM test passed!")


if __name__ == "__main__":
    test_reverse_ddpm()