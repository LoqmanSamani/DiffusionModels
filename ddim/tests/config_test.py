import torch
from config import Config


def test_config():
    """
    tests the config class to ensure beta schedules are computed correctly.
    """
    beta_range = (1e-4, 0.02)
    num_steps = 10
    methods = ["linear", "sigmoid", "quadratic", "constant", "inverse_time"]

    config = Config(beta_range=beta_range, num_steps=num_steps)

    for method in methods:
        beta = config.compute_beta_schedule(beta_range, num_steps, method=method)

        assert beta.shape[0] == num_steps, f"{method}: beta should have {num_steps} steps"
        assert not torch.isnan(beta).any(), f"{method}: beta contains NaN values"
        assert (beta >= 0).all(), f"{method}: beta contains negative values"

        if method == "linear" or method == "sigmoid" or method == "quadratic" or method == "inverse_time":
            assert torch.all(beta[:-1] <= beta[1:]), f"{method}: beta should be increasing"
        elif method == "constant":
            assert torch.all(beta == beta[0]), f"{method}: beta should be constant"

    print("all tests passed.")

test_config()