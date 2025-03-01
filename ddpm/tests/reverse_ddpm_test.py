import torch
from reverse_ddpm import ReverseDDPM


def test_reverse_ddpm():

    torch.manual_seed(42)

    ddpm = ReverseDDPM(num_steps=1000, beta_start=1e-4, beta_end=0.02)

    batch_size = 1
    img_channels = 3
    img_size = 32
    batch_t = torch.randn((batch_size, img_channels, img_size, img_size))
    predicted_noise = torch.randn_like(batch_t)
    time_step = torch.randint(0, ddpm.num_steps, (batch_size,))
    predicted  = ddpm.remove_noise(batch_t, predicted_noise, time_step)

    assert predicted.shape == batch_t.shape, "Predicted x_{t-1} shape mismatch!"
    # assert batch0.shape == batch_t.shape, "Estimated x_0 shape mismatch!"
    # assert torch.all(batch0 >= -1.0) and torch.all(batch0 <= 1.0), "x_0 values out of range [-1,1]!"

    print("ReverseDDPM tests passed successfully!")


test_reverse_ddpm()