import torch
from vlc import VariationalLatentCompressor



def test_variational_latent_compressor():
    in_channels = 3
    out_channels = 3
    down_channels = [32, 64, 128, 256]
    up_channels = [256, 128, 64, 32]
    dropout_rate = 0.1
    num_heads = 4
    num_groups = 8
    levels = 1
    down_sampling_factor = 4
    beta = 1.0

    x = torch.randn(2, in_channels, 256, 256)
    print(f"Input shape: {x.shape}")

    model = VariationalLatentCompressor(
        in_channels, down_channels, up_channels, out_channels,
        dropout_rate, num_heads, num_groups, levels, down_sampling_factor, beta
    )

    try:
        x_hat, kl_loss = model(x)

        print(kl_loss)
        print(f"Output shape: {x_hat.shape}")
        print(f"KL loss shape: {kl_loss.shape}")

        assert x_hat.shape == x.shape, f"Expected output shape {x.shape}, but got {x_hat.shape}"
        assert kl_loss.shape == torch.Size([]), f"KL loss should be a scalar, got shape {kl_loss.shape}"

        print("Test passed! Model runs successfully and outputs correct shapes.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise


test_variational_latent_compressor()