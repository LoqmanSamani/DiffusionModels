import torch
from config import Config
from reverse_diffusion import ReverseSDE






def test_reverse_sde():
    """test the ReverseSDE class to ensure correct denoising behavior."""

    class DummyModel:
        """a simple model that predicts zero noise for testing purposes."""

        def __call__(self, x, t):
            return torch.zeros_like(x)

    config = Config(
        in_channels=3,
        down_channels=[32, 64, 128, 256],
        mid_channels=[256, 256, 128],
        up_channels=[256, 128, 64, 16],
        down_sampling=[True, True, False],
        num_groups=8,
        embed_dim=128,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        dropout_rate=0.3,
        num_attention_heads=2,
        down_sampling_factor=2,
        upsampling_factor=2,
        apply_down_conv=True,
        apply_down_pool=True,
        apply_up_conv=True,
        kernel_size=3,
        norm=True,
        activation=True,
        method="vp",
        start=None,
        end=None,
        max_steps=400,
        sigma_min=None,
        sigma_max=None,
        beta_range=None,
        beta_schedule_method="linear",
        max_epoch=5,
        device=None,
        optimizer=None,
        objective=None,
        save_path=None,
        checkpoint=None
    )
    reverse_sde = ReverseSDE(config, model=DummyModel())

    x_noisy = torch.randn(5, 3, 100, 100)
    t = torch.randint(1, config.max_steps, (x_noisy.shape[0],))

    for method in ["ve", "vp", "sub-vp"]:
        config.method = method
        x_denoised = reverse_sde.forward(x_noisy.clone(), t, "cpu")

        assert x_denoised.shape == x_noisy.shape, f"Reverse {method}: Shape mismatch"
        assert not torch.equal(x_noisy, x_denoised), f"Reverse {method}: x should change after denoising"

    print("ReverseSDE tests passed.")


test_reverse_sde()