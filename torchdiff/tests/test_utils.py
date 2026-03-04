import torch
import pytest
from torchdiff.utils import (
    DiffusionNetwork, LossAdapter, Metrics,
    mse_loss, snr_capped_loss, ve_sigma_weighted_score_loss
)


class TestDiffusionNetwork:
    """Test suite for DiffusionNetwork (U-Net)."""

    @pytest.fixture
    def small_net(self):
        """Create a small DiffusionNetwork for testing."""
        return DiffusionNetwork(
            in_channels=1,
            down_channels=[16, 32],
            mid_channels=[32, 32],
            up_channels=[32, 16],
            down_sampling=[True],
            time_embed_dim=32,
            y_embed_dim=32,
            num_down_blocks=1,
            num_mid_blocks=1,
            num_up_blocks=1,
            dropout_rate=0.0,
            cont_time=False,
            use_flash_attention=False,
        )

    @pytest.fixture
    def small_net_cont(self):
        """Create a small DiffusionNetwork with continuous time."""
        return DiffusionNetwork(
            in_channels=3,
            down_channels=[16, 32],
            mid_channels=[32, 32],
            up_channels=[32, 16],
            down_sampling=[True],
            time_embed_dim=32,
            y_embed_dim=32,
            num_down_blocks=1,
            num_mid_blocks=1,
            num_up_blocks=1,
            dropout_rate=0.1,
            cont_time=True,
            use_flash_attention=False,
        )

    def test_initialization(self, small_net):
        """Test that DiffusionNetwork initializes correctly."""
        assert small_net.cont_time is False
        assert isinstance(small_net.encoder, torch.nn.ModuleList)
        assert isinstance(small_net.middle, torch.nn.ModuleList)
        assert isinstance(small_net.decoder, torch.nn.ModuleList)

    def test_forward_shape_unconditional(self, small_net):
        """Test output shape without conditioning."""
        x = torch.randn(2, 1, 16, 16)
        t = torch.randint(0, 100, (2,))
        out = small_net(x, t)
        assert out.shape == x.shape

    def test_forward_shape_conditional(self, small_net):
        """Test output shape with conditioning embeddings."""
        x = torch.randn(2, 1, 16, 16)
        t = torch.randint(0, 100, (2,))
        y = torch.randn(2, 4, 32)  # context
        out = small_net(x, t, y=y)
        assert out.shape == x.shape

    def test_forward_shape_continuous_time(self, small_net_cont):
        """Test output shape with continuous time."""
        x = torch.randn(2, 3, 16, 16)
        t = torch.rand(2)  # continuous time in [0, 1]
        out = small_net_cont(x, t)
        assert out.shape == x.shape

    def test_forward_with_clip_embeddings(self, small_net):
        """Test output shape with CLIP embeddings added to time embedding."""
        x = torch.randn(2, 1, 16, 16)
        t = torch.randint(0, 100, (2,))
        clip_emb = torch.randn(2, 32)  # same dim as time_embed_dim
        out = small_net(x, t, clip_embeddings=clip_emb)
        assert out.shape == x.shape

    def test_parameter_count_positive(self, small_net):
        """Test that model has trainable parameters."""
        param_count = sum(p.numel() for p in small_net.parameters())
        assert param_count > 0

    def test_output_finite(self, small_net):
        """Test that output values are finite."""
        x = torch.randn(2, 1, 16, 16)
        t = torch.randint(0, 100, (2,))
        out = small_net(x, t)
        assert torch.all(torch.isfinite(out))

    def test_different_batch_sizes(self, small_net):
        """Test that model handles different batch sizes."""
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 1, 16, 16)
            t = torch.randint(0, 100, (bs,))
            out = small_net(x, t)
            assert out.shape == (bs, 1, 16, 16)

    def test_gradient_flow(self, small_net):
        """Test that gradients flow through the network."""
        x = torch.randn(2, 1, 16, 16)
        t = torch.randint(0, 100, (2,))
        out = small_net(x, t)
        loss = out.mean()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in small_net.parameters() if p.requires_grad)
        assert has_grad

    def test_down_sampling_assertion(self):
        """Test that mismatched down_sampling length raises AssertionError."""
        with pytest.raises(AssertionError):
            DiffusionNetwork(
                in_channels=1,
                down_channels=[16, 32, 64],
                mid_channels=[64, 64],
                up_channels=[64, 32, 16],
                down_sampling=[True],  # should be length 2, not 1
                time_embed_dim=32,
                y_embed_dim=32,
                num_down_blocks=1,
                num_mid_blocks=1,
                num_up_blocks=1,
                use_flash_attention=False,
            )


class TestLossAdapter:
    """Test suite for LossAdapter."""

    def test_adapter_with_mse_loss(self):
        """Test LossAdapter wrapping mse_loss."""
        adapter = LossAdapter(mse_loss)
        pred = torch.randn(4, 3, 16, 16)
        target = torch.randn(4, 3, 16, 16)
        loss = adapter(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_adapter_with_nn_module_loss(self):
        """Test LossAdapter wrapping nn.MSELoss."""
        adapter = LossAdapter(torch.nn.MSELoss())
        pred = torch.randn(4, 3, 16, 16)
        target = torch.randn(4, 3, 16, 16)
        loss = adapter(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_adapter_with_extra_args(self):
        """Test LossAdapter forwards extra args when function supports them."""
        adapter = LossAdapter(snr_capped_loss)
        pred = torch.randn(4, 3, 16, 16)
        target = torch.randn(4, 3, 16, 16)
        variance = torch.rand(4) * 0.5 + 0.01
        loss = adapter(pred, target, variance, 5.0)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_adapter_handles_incompatible_args(self):
        """Test LossAdapter falls back for functions that don't accept extra args."""
        adapter = LossAdapter(torch.nn.MSELoss())
        pred = torch.randn(4, 16)
        target = torch.randn(4, 16)
        # nn.MSELoss ignores extra args via the adapter
        loss = adapter(pred, target, "extra", "args")
        assert loss.shape == ()


class TestMSELoss:
    """Test suite for mse_loss function."""

    def test_zero_loss_identical_inputs(self):
        """Test zero loss when pred equals target."""
        x = torch.randn(4, 3, 16, 16)
        loss = mse_loss(x, x)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_positive_loss_different_inputs(self):
        """Test positive loss when pred differs from target."""
        pred = torch.randn(4, 3, 16, 16)
        target = torch.randn(4, 3, 16, 16)
        loss = mse_loss(pred, target)
        assert loss.item() > 0

    def test_scalar_output(self):
        """Test that output is a scalar."""
        loss = mse_loss(torch.randn(2, 8), torch.randn(2, 8))
        assert loss.dim() == 0

    def test_extra_args_ignored(self):
        """Test that extra args are accepted but ignored."""
        loss = mse_loss(torch.randn(2, 8), torch.randn(2, 8), "extra1", "extra2")
        assert loss.dim() == 0


class TestSNRCappedLoss:
    """Test suite for snr_capped_loss function."""

    def test_basic_computation(self):
        """Test SNR capped loss produces valid output."""
        pred = torch.randn(4, 3, 16, 16)
        target = torch.randn(4, 3, 16, 16)
        variance = torch.rand(4) * 0.5 + 0.01
        loss = snr_capped_loss(pred, target, variance)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_capping_effect(self):
        """Test that gamma caps the SNR weight."""
        pred = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        # Very small variance -> very high SNR -> should be capped
        variance = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6])
        loss_low_gamma = snr_capped_loss(pred, target, variance, gamma=1.0)
        loss_high_gamma = snr_capped_loss(pred, target, variance, gamma=100.0)
        # Higher gamma cap -> higher weighted loss
        assert loss_high_gamma >= loss_low_gamma

    def test_zero_loss_identical_inputs(self):
        """Test zero loss when predictions match targets."""
        x = torch.randn(4, 3, 8, 8)
        variance = torch.rand(4) * 0.5 + 0.01
        loss = snr_capped_loss(x, x, variance)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


class TestVESigmaWeightedScoreLoss:
    """Test suite for ve_sigma_weighted_score_loss function."""

    def test_basic_computation(self):
        """Test VE sigma weighted loss produces valid output."""
        pred_score = torch.randn(4, 3, 8, 8)
        target_score = torch.randn(4, 3, 8, 8)
        sigma = torch.rand(4) + 0.1
        loss = ve_sigma_weighted_score_loss(pred_score, target_score, sigma)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_sigma_broadcasting(self):
        """Test that sigma is broadcast correctly to match score dimensions."""
        pred_score = torch.randn(4, 3, 8, 8)
        target_score = torch.randn(4, 3, 8, 8)
        sigma = torch.rand(4) + 0.1  # shape (4,)
        loss = ve_sigma_weighted_score_loss(pred_score, target_score, sigma)
        assert loss.dim() == 0

    def test_higher_sigma_effect(self):
        """Test that different sigma values affect the loss."""
        pred_score = torch.randn(4, 3, 8, 8)
        target_score = torch.randn(4, 3, 8, 8)
        sigma_low = torch.tensor([0.1, 0.1, 0.1, 0.1])
        sigma_high = torch.tensor([10.0, 10.0, 10.0, 10.0])
        loss_low = ve_sigma_weighted_score_loss(pred_score, target_score, sigma_low)
        loss_high = ve_sigma_weighted_score_loss(pred_score, target_score, sigma_high)
        # Different sigma should give different loss values
        assert not torch.isclose(loss_low, loss_high)


class TestMetrics:
    """Test suite for Metrics class."""

    @pytest.fixture
    def metrics_basic(self):
        """Create a Metrics instance with only MSE/PSNR/SSIM."""
        return Metrics(device="cpu", fid=False, metrics=True, lpips_=False)

    def test_initialization(self, metrics_basic):
        """Test Metrics initialization."""
        assert metrics_basic.device == "cpu"
        assert metrics_basic.fid is False
        assert metrics_basic.metrics is True
        assert metrics_basic.lpips is False

    def test_compute_metrics_identical(self, metrics_basic):
        """Test metrics with identical inputs give perfect scores."""
        x = torch.randn(4, 3, 32, 32)
        mse, psnr, ssim = metrics_basic.compute_metrics(x, x)
        assert mse < 1e-6
        assert psnr > 50  # Very high PSNR for near-identical
        assert ssim > 0.99

    def test_compute_metrics_different(self, metrics_basic):
        """Test metrics with different inputs give reasonable scores."""
        x = torch.randn(4, 3, 32, 32)
        x_hat = torch.randn(4, 3, 32, 32)
        mse, psnr, ssim = metrics_basic.compute_metrics(x, x_hat)
        assert mse > 0
        assert isinstance(psnr, float)
        assert isinstance(ssim, float)

    def test_compute_metrics_shape_mismatch(self, metrics_basic):
        """Test shape mismatch raises ValueError."""
        x = torch.randn(4, 3, 32, 32)
        x_hat = torch.randn(4, 3, 16, 16)
        with pytest.raises(ValueError, match="Shape mismatch"):
            metrics_basic.compute_metrics(x, x_hat)

    def test_forward_with_metrics_only(self, metrics_basic):
        """Test forward call with metrics enabled."""
        x = torch.randn(4, 3, 32, 32)
        x_hat = torch.randn(4, 3, 32, 32)
        fid, mse, psnr, ssim, lpips_score = metrics_basic.forward(x, x_hat)
        assert fid == float('inf')  # FID disabled
        assert mse is not None
        assert psnr is not None
        assert ssim is not None
        assert lpips_score is None  # LPIPS disabled
