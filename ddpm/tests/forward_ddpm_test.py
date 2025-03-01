import torch
from forward_ddpm import ForwardDDPM


def test_forward_ddpm():

    torch.manual_seed(42)

    num_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    ddpm = ForwardDDPM(num_steps=num_steps, beta_start=beta_start, beta_end=beta_end)

    batch_size = 8
    channels = 3
    height = 64
    width = 64
    batch = torch.randn(batch_size, channels, height, width)

    # Generate random Gaussian noise
    noise = torch.randn_like(batch)
    # Randomly select time steps for each image in batch
    time_steps = torch.randint(0, num_steps, (batch_size,))
    noisy_images = ddpm.add_noise(batch, noise, time_steps)

    assert noisy_images.shape == batch.shape, "Output shape does not match input shape!"
    assert torch.all(torch.isfinite(noisy_images)), "NaN or Inf detected in output!"
    assert not torch.allclose(noisy_images, batch), "Noise not applied properly!"

    print("Test passed: ForwardDDPM `add_noise` function works correctly!")


test_forward_ddpm()