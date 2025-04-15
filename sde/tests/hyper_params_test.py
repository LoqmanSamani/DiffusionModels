import torch
import numpy as np
from hyper_param import HyperParamsSDE



def test_hyper_params():

    print("Testing HyperParams...")
    num_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    sigma_start = 1e-3
    sigma_end = 10.0
    methods = ["linear", "sigmoid", "quadratic", "constant", "inverse_time"]

    for method in methods:
        print(f"  Beta method: {method}")
        hp = HyperParamsSDE(
            num_steps=num_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            trainable_beta=False,
            beta_method=method,
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )

        assert hp.betas.shape == (num_steps,), f"Beta shape mismatch: {hp.betas.shape}"
        assert torch.all(hp.betas >= beta_start), f"Betas below {beta_start}"
        assert torch.all(hp.betas <= beta_end), f"Betas above {beta_end}"
        assert torch.all(hp.betas.isfinite()), "Betas contain NaN/Inf"

        assert hp.cum_betas.shape == (num_steps,), f"Cum_betas shape mismatch: {hp.cum_betas.shape}"
        assert torch.all(hp.cum_betas >= 0), "Negative cum_betas"
        assert torch.all(hp.cum_betas.isfinite()), "Cum_betas contain NaN/Inf"
        assert torch.all(hp.cum_betas[1:] >= hp.cum_betas[:-1]), "Cum_betas not increasing"

        assert hp.sigmas.shape == (num_steps,), f"Sigma shape mismatch: {hp.sigmas.shape}"
        assert torch.all(hp.sigmas >= sigma_start), f"Sigmas below {sigma_start}"
        assert torch.all(hp.sigmas <= sigma_end), f"Sigmas above {sigma_end}"
        assert torch.all(hp.sigmas.isfinite()), "Sigmas contain NaN/Inf"

        for sde_method in ["ve", "vp", "sub-vp"]:
            var = hp.get_variance(torch.tensor([num_steps - 1]), sde_method)
            assert var.shape == (1,), f"Variance shape mismatch for {sde_method}: {var.shape}"
            assert var >= 0, f"Negative variance for {sde_method}: {var}"
            if sde_method == "ve":
                expected_var = sigma_end ** 2
                assert abs(var.item() - expected_var) < 1e-2, f"VE variance mismatch: got {var.item()}, expected {expected_var}"
            elif sde_method == "vp":
                expected_var = 1 - np.exp(-hp.cum_betas[-1].item())
                assert abs(var.item() - expected_var) < 1e-2, f"VP variance mismatch: got {var.item()}, expected {expected_var}"
            elif sde_method == "sub-vp":
                expected_var = 1 - np.exp(-2 * hp.cum_betas[-1].item())
                assert abs(var.item() - expected_var) < 1e-2, f"Sub-VP variance mismatch: got {var.item()}, expected {expected_var}"

    print("HyperParams tests passed!")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_hyper_params()
