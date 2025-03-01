import torch
from reverse_diffudion import ReverseDDIM



def test_reverse_ddim():

    class Config:
        def __init__(self):
            self.eta = 0.5
            self.alpha_tau = torch.tensor([0.9, 0.85, 0.8, 0.75, 0.7])

    config = Config()
    reverse_ddim = ReverseDDIM(config)
    x = torch.randn(10, 3, 64, 64)
    p_noise = torch.randn_like(x)
    t = 2
    prev_t = 1

    xt_prev, x0 = reverse_ddim(x, p_noise, t, prev_t)

    assert isinstance(xt_prev, torch.Tensor), "xt_prev should be a tensor"
    assert isinstance(x0, torch.Tensor), "x0 should be a tensor"
    assert xt_prev.shape == x.shape, f"xt_prev shape should match x shape, got {xt_prev.shape} vs {x.shape}"
    assert x0.shape == x.shape, f"x0 shape should match x shape, got {x0.shape} vs {x.shape}"

    print("Test passed! reverse_ddim works as expected.")

test_reverse_ddim()