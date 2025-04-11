import torch
from hyper_params import HyperParamsDDPM



def test_hyper_params():
    """tests different beta schedule methods in HyperParams."""
    methods = ["linear", "sigmoid", "quadratic", "constant", "inverse_time"]
    num_steps = 1000
    beta_start, beta_end = 1e-4, 0.02

    for method in methods:
        hyper_params = HyperParamsDDPM(num_steps=num_steps, beta_start=beta_start, beta_end=beta_end, beta_method=method)
        betas = hyper_params.betas
        assert betas.shape == (num_steps,), f"Expected shape ({num_steps},), got {betas.shape}"
        assert torch.all(betas >= beta_start) and torch.all(betas <= beta_end + 1e-6), \
            f"Method {method}: Betas out of range [{beta_start}, {beta_end}]"
        print(f"HyperParams beta schedule test passed for method: {method}")


if __name__ == "__main__":
    test_hyper_params()