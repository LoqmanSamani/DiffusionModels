import torch
import pytest
from torchdiff.ldm import AutoencoderLDM


class TestAutoencoderLDM:
    """Test suite for AutoencoderLDM (VAE/VQ-VAE)."""

    @pytest.fixture
    def small_vae(self):
        """Create a small KL-regularized autoencoder."""
        return AutoencoderLDM(
            in_channels=1,
            down_channels=[16, 32],
            up_channels=[32, 16],
            out_channels=1,
            dropout_rate=0.0,
            num_heads=2,
            num_groups=2,
            num_layers_per_block=1,
            total_down_sampling_factor=2,
            latent_channels=4,
            num_embeddings=64,
            use_vq=False,
            beta=1.0,
            use_flash=False,
            use_grad_check=False,
        )

    @pytest.fixture
    def small_vqvae(self):
        """Create a small VQ-regularized autoencoder."""
        return AutoencoderLDM(
            in_channels=1,
            down_channels=[16, 32],
            up_channels=[32, 16],
            out_channels=1,
            dropout_rate=0.0,
            num_heads=2,
            num_groups=2,
            num_layers_per_block=1,
            total_down_sampling_factor=2,
            latent_channels=4,
            num_embeddings=64,
            use_vq=True,
            use_flash=False,
            use_grad_check=False,
        )

    def test_initialization_kl(self, small_vae):
        """Test KL autoencoder initializes correctly."""
        assert small_vae.use_vq is False
        assert small_vae.beta == 1.0
        assert isinstance(small_vae.down_blocks, torch.nn.ModuleList)
        assert isinstance(small_vae.up_blocks, torch.nn.ModuleList)

    def test_initialization_vq(self, small_vqvae):
        """Test VQ autoencoder initializes correctly."""
        assert small_vqvae.use_vq is True
        assert hasattr(small_vqvae, 'vq_layer')

    def test_encode_kl_shapes(self, small_vae):
        """Test encode output shapes for KL autoencoder."""
        x = torch.randn(2, 1, 16, 16)
        z, reg_loss = small_vae.encode(x)
        assert z.dim() == 4
        assert z.shape[0] == 2
        assert z.shape[1] == 4  # latent_channels
        assert isinstance(reg_loss, torch.Tensor)

    def test_encode_vq_shapes(self, small_vqvae):
        """Test encode output shapes for VQ autoencoder."""
        x = torch.randn(2, 1, 16, 16)
        z, vq_loss = small_vqvae.encode(x)
        assert z.dim() == 4
        assert z.shape[0] == 2
        assert z.shape[1] == 4  # latent_channels

    def test_decode_shapes(self, small_vae):
        """Test decode output shapes."""
        # Encode first to get a valid latent
        x = torch.randn(2, 1, 16, 16)
        z, _ = small_vae.encode(x)
        x_hat = small_vae.decode(z)
        assert x_hat.shape == x.shape

    def test_forward_shapes(self, small_vae):
        """Test full forward pass output shapes."""
        x = torch.randn(2, 1, 16, 16)
        x_hat, total_loss, reg_loss, z = small_vae(x)
        assert x_hat.shape == x.shape
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0
        assert z.dim() == 4

    def test_forward_vq_shapes(self, small_vqvae):
        """Test full forward pass for VQ autoencoder."""
        x = torch.randn(2, 1, 16, 16)
        x_hat, total_loss, reg_loss, z = small_vqvae(x)
        assert x_hat.shape == x.shape
        assert total_loss.dim() == 0

    def test_reconstruction_loss_positive(self, small_vae):
        """Test that total loss is positive (recon + reg)."""
        x = torch.randn(2, 1, 16, 16)
        _, total_loss, _, _ = small_vae(x)
        assert total_loss.item() > 0

    def test_output_finite(self, small_vae):
        """Test that outputs are finite."""
        x = torch.randn(2, 1, 16, 16)
        x_hat, total_loss, _, z = small_vae(x)
        assert torch.all(torch.isfinite(x_hat))
        assert torch.isfinite(total_loss)
        assert torch.all(torch.isfinite(z))

    def test_gradient_flow(self, small_vae):
        """Test gradient flow through autoencoder."""
        x = torch.randn(2, 1, 16, 16)
        _, total_loss, _, _ = small_vae(x)
        total_loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in small_vae.parameters() if p.requires_grad)
        assert has_grad

    def test_reparameterize(self, small_vae):
        """Test reparameterization trick produces correct shapes."""
        mu = torch.randn(2, 32, 8, 8)
        logvar = torch.randn(2, 32, 8, 8)
        z = small_vae.reparameterize(mu, logvar)
        assert z.shape == mu.shape
        assert torch.all(torch.isfinite(z))

    def test_different_batch_sizes(self, small_vae):
        """Test with different batch sizes."""
        for bs in [1, 4]:
            x = torch.randn(bs, 1, 16, 16)
            x_hat, _, _, _ = small_vae(x)
            assert x_hat.shape == (bs, 1, 16, 16)

    def test_in_out_channels_match_assertion(self):
        """Test that mismatched in/out channels raises AssertionError."""
        with pytest.raises(AssertionError):
            AutoencoderLDM(
                in_channels=1,
                down_channels=[16, 32],
                up_channels=[32, 16],
                out_channels=3,  # mismatch with in_channels=1
                dropout_rate=0.0,
                num_heads=2,
                num_groups=2,
                num_layers_per_block=1,
                total_down_sampling_factor=2,
                latent_channels=4,
                num_embeddings=64,
                use_flash=False,
            )

    def test_latent_spatial_reduction(self, small_vae):
        """Test that latent spatial dimensions are reduced."""
        x = torch.randn(2, 1, 16, 16)
        z, _ = small_vae.encode(x)
        # With total_down_sampling_factor=2, spatial dims should be halved
        assert z.shape[2] <= x.shape[2]
        assert z.shape[3] <= x.shape[3]

    def test_encode_decode_roundtrip_shapes(self, small_vae):
        """Test encode-decode roundtrip preserves shapes."""
        x = torch.randn(2, 1, 16, 16)
        z, _ = small_vae.encode(x)
        x_hat = small_vae.decode(z)
        assert x_hat.shape == x.shape
