import torch
from hyper_param import HyperParamsSDE
from reverse_sde import ReverseSDE



def test_reverse_sde():
    print("Testing ReverseSDE...")
    num_steps = 1000
    batch_size = 4
    channels = 3
    height = 32
    width = 32
    hp = HyperParamsSDE(num_steps=num_steps, sigma_end=10.0)

    def mock_score(x, t):
        return -x

    x_T = torch.randn(batch_size, channels, height, width)
    x_T = torch.clamp(x_T, -1, 1)

    for method in ["ve", "vp", "sub-vp", "ode"]:
        print(f"  SDE method: {method}")
        sde = ReverseSDE(hp, method)

        x_t = x_T.clone()
        for t in range(num_steps - 1, -1, -1):
            noise = torch.randn_like(x_t) if method != "ode" else None
            time_steps = torch.tensor([t], dtype=torch.long)
            predicted_noise = mock_score(x_t, time_steps)
            x_t = sde(x_t, noise, predicted_noise, time_steps)

        assert x_t.shape == x_T.shape, f"Shape mismatch: got {x_t.shape}, expected {x_T.shape}"
        assert torch.all(x_t.isfinite()), f"Output contains NaN/Inf for {method}"
        assert x_t.abs().max() < 1e5, f"Output values too large for {method}: max {x_t.abs().max().item()}"

        if method == "ode":
            x_t_1 = x_T.clone()
            x_t_2 = x_T.clone()
            for t in range(num_steps - 1, -1, -1):
                time_steps = torch.tensor([t], dtype=torch.long)
                predicted_noise_1 = mock_score(x_t_1, time_steps)
                predicted_noise_2 = mock_score(x_t_2, time_steps)
                x_t_1 = sde(x_t_1, None, predicted_noise_1, time_steps)
                x_t_2 = sde(x_t_2, None, predicted_noise_2, time_steps)
            assert torch.allclose(x_t_1, x_t_2, atol=1e-4), f"ODE not deterministic for {method}"

        print(f"    Final mean: {x_t.mean().item():.4f}, Final variance: {x_t.var().item():.4f}")

    print("ReverseSDE tests passed!")



if __name__ == "__main__":
    torch.manual_seed(42)
    test_reverse_sde()