import torch
from forward_ddim import ForwardDDIM

def test_forward_ddim():

    class DummyConfig:
        def __init__(self, num_steps=1000, batch_size=4, img_size=(3, 32, 32)):
            self.alpha_sqrt = torch.linspace(0.1, 1.0, steps=num_steps)
            self.num_steps = num_steps
            self.batch_size = batch_size
            self.img_size = img_size

    config = DummyConfig()
    forward_ddim = ForwardDDIM(config)
    batch_size, channels, height, width = config.batch_size, *config.img_size
    x = torch.randn(batch_size, channels, height, width)
    noise = torch.randn_like(x)
    t = torch.randint(0, config.num_steps, (batch_size,))

    noisy_x = forward_ddim(x, noise, t)

    print(f"Original Image Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"Noisy Image Mean: {noisy_x.mean():.4f}, Std: {noisy_x.std():.4f}")

test_forward_ddim()