import torch
import numpy as np
from hyper_param import HyperParamsSDE
from forward_sde import ForwardSDE




def test_forward_sde():

    print("Testing ForwardSDE...")
    num_steps = 1000
    batch_size = 4
    channels = 3
    height = 32
    width = 32
    hp = HyperParamsSDE(num_steps=num_steps, sigma_end=10.0)

    x0 = torch.rand(batch_size, channels, height, width) * 2 - 1
    x0 = x0.to(torch.float32)
    input_var = 1.0 / 3.0

    for method in ["ve", "vp", "sub-vp", "ode"]:
        print(f"  SDE method: {method}")
        sde = ForwardSDE(hp, method)

        x_t = x0.clone()
        for t in range(num_steps):
            noise = torch.randn_like(x_t)
            time_steps = torch.tensor([t], dtype=torch.long)
            x_t = sde(x_t, noise, time_steps)

        mean = x_t.mean().item()
        var = x_t.var().item()
        expected_var = None
        if method != "ode":
            marginal_var = hp.get_variance(torch.tensor([num_steps - 1]), method).item()
            if method == "ve":
                expected_var = marginal_var  # Should be sigma_end^2 = 100.0
            else:
                expected_var = marginal_var + (1 - marginal_var) * input_var

        print(
            f"    Mean: {mean:.4f}, Variance: {var:.4f}, Expected Variance: {expected_var if expected_var else 'N/A'}")

        assert x_t.shape == x0.shape, f"Shape mismatch: got {x_t.shape}, expected {x0.shape}"
        assert torch.all(x_t.isfinite()), f"Output contains NaN/Inf for {method}"

        if method == "ve":
            assert abs(mean) < 0.5, f"VE mean too large: {mean}"
            assert abs(
                var - expected_var) / expected_var < 0.1, f"VE variance mismatch: got {var}, expected {expected_var}"
        elif method == "vp":
            assert abs(mean) < 0.5, f"VP mean too large: {mean}"
            assert abs(
                var - expected_var) / expected_var < 0.1, f"VP variance mismatch: got {var}, expected {expected_var}"
        elif method == "sub-vp":
            assert abs(mean) < 0.5, f"Sub-VP mean too large: {mean}"
            assert abs(
                var - expected_var) / expected_var < 0.1, f"Sub-VP variance mismatch: got {var}, expected {expected_var}"
        elif method == "ode":
            if method == "ve":
                assert torch.allclose(x_t, x0, atol=1e-4), "VE ODE should not change x"
            else:
                assert x_t.abs().mean() < x0.abs().mean(), "VP/Sub-VP ODE should reduce magnitude"

    print("ForwardSDE tests passed!")



if __name__ == "__main__":
    torch.manual_seed(42)
    test_forward_sde()
