import torch
import pytest
from torchdiff.utils import (
    DiffusionNetwork, LossAdapter, Metrics,
    mse_loss, snr_capped_loss, ve_sigma_weighted_score_loss,
    min_snr_loss, get_timestep_embedding,
    TextEncoder, EncoderLayer, FeedForward, Attention,
    Embedding, ResBlock, CrossAttention,
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


###===========================================================================###
# Tests for previously untested classes
###===========================================================================###


class TestFeedForward:
    """Test suite for FeedForward module."""

    @pytest.fixture
    def ff(self):
        return FeedForward(embedding_dimension=64, scaling_value=4, dropout_rate=0.0)

    def test_output_shape(self, ff):
        x = torch.randn(2, 10, 64)
        out = ff(x)
        assert out.shape == (2, 10, 64)

    def test_hidden_dim_scaling(self):
        ff = FeedForward(embedding_dimension=32, scaling_value=2)
        # Hidden dim should be 32*2=64
        assert ff.layers[0].out_features == 64
        assert ff.layers[3].in_features == 64

    def test_gradient_flow(self, ff):
        x = torch.randn(2, 5, 64, requires_grad=True)
        out = ff(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_scaling_values(self):
        for sv in [1, 2, 4, 8]:
            ff = FeedForward(embedding_dimension=32, scaling_value=sv)
            x = torch.randn(1, 5, 32)
            out = ff(x)
            assert out.shape == (1, 5, 32)


class TestEncoderLayer:
    """Test suite for EncoderLayer module."""

    @pytest.fixture
    def layer(self):
        return EncoderLayer(
            input_dimension=64,
            output_dimension=64,
            num_heads=4,
            dropout_rate=0.0,
            qkv_bias=False,
            scaling_value=4,
            epsilon=1e-5,
        )

    def test_output_shape_same_dim(self, layer):
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == (2, 10, 64)

    def test_output_shape_different_dim(self):
        layer = EncoderLayer(
            input_dimension=64, output_dimension=128,
            num_heads=4, dropout_rate=0.0, qkv_bias=False,
            scaling_value=4,
        )
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == (2, 10, 128)

    def test_with_attention_mask(self, layer):
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 8:] = True  # Mask last 2 positions
        out = layer(x, attention_mask=mask)
        assert out.shape == (2, 10, 64)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, layer):
        x = torch.randn(2, 5, 64, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_no_double_norm1_attribute(self):
        """Verify the double assignment bug is fixed."""
        layer = EncoderLayer(
            input_dimension=64, output_dimension=64,
            num_heads=4, dropout_rate=0.0, qkv_bias=False,
            scaling_value=4,
        )
        assert isinstance(layer.norm1, torch.nn.LayerNorm)
        assert layer.norm1.normalized_shape == (64,)


class TestEmbedding:
    """Test suite for Embedding module."""

    @pytest.fixture
    def emb_sinusoidal(self):
        return Embedding(vocabulary_size=100, embedding_dimension=64, max_context_length=20)

    @pytest.fixture
    def emb_learned(self):
        return Embedding(vocabulary_size=100, embedding_dimension=64, max_context_length=20, use_learned_pos=True)

    def test_sinusoidal_output_shape(self, emb_sinusoidal):
        token_ids = torch.randint(0, 100, (2, 10))
        out = emb_sinusoidal(token_ids)
        assert out.shape == (2, 10, 64)

    def test_learned_output_shape(self, emb_learned):
        token_ids = torch.randint(0, 100, (2, 10))
        out = emb_learned(token_ids)
        assert out.shape == (2, 10, 64)

    def test_sinusoidal_position_variance(self, emb_sinusoidal):
        """Verify sinusoidal encodings vary by position (the bug fix)."""
        token_ids = torch.zeros(1, 5, dtype=torch.long)  # Same token at all positions
        out = emb_sinusoidal(token_ids)
        # Token embeddings are the same, but positional encodings should differ
        # So output at different positions should differ
        assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-6)
        assert not torch.allclose(out[0, 0], out[0, 4], atol=1e-6)

    def test_sinusoidal_encoding_correctness(self, emb_sinusoidal):
        """Verify positional encoding is position-dependent, not just constant."""
        enc = emb_sinusoidal._generate_positional_encoding(10, torch.device('cpu'))
        assert enc.shape == (1, 10, 64)
        # Different positions should have different encodings
        assert not torch.allclose(enc[0, 0], enc[0, 1])
        # Position 0 should have sin(0)=0 for even dims
        assert torch.allclose(enc[0, 0, 0::2], torch.zeros(32), atol=1e-6)

    def test_sinusoidal_odd_embedding_dim(self):
        """Test sinusoidal encoding with odd embedding dimension."""
        emb = Embedding(vocabulary_size=100, embedding_dimension=65, max_context_length=10)
        token_ids = torch.randint(0, 100, (2, 5))
        out = emb(token_ids)
        assert out.shape == (2, 5, 65)
        assert torch.isfinite(out).all()

    def test_sinusoidal_cache_extension(self, emb_sinusoidal):
        """Sequences longer than max_context_length should regenerate cache."""
        token_ids = torch.randint(0, 100, (1, 25))  # > max_context_length=20
        out = emb_sinusoidal(token_ids)
        assert out.shape == (1, 25, 64)

    def test_learned_exceeds_max_raises(self, emb_learned):
        """Learned pos emb should raise for sequences > max_context_length."""
        token_ids = torch.randint(0, 100, (1, 25))
        with pytest.raises(ValueError, match="exceeds max_context_length"):
            emb_learned(token_ids)

    def test_input_dim_assertion(self, emb_sinusoidal):
        """1D or 3D inputs should raise."""
        with pytest.raises(AssertionError):
            emb_sinusoidal(torch.tensor([1, 2, 3]))


class TestAttention:
    """Test suite for the Attention module."""

    @pytest.fixture
    def attn(self):
        return Attention(in_channels=32, y_embed_dim=64, num_heads=4, num_groups=8, dropout_rate=0.0)

    def test_self_attention_shape(self, attn):
        x = torch.randn(2, 32, 8, 8)
        out = attn(x)
        assert out.shape == (2, 32, 8, 8)

    def test_cross_attention_3d_y(self, attn):
        x = torch.randn(2, 32, 8, 8)
        y = torch.randn(2, 5, 64)  # seq_len=5
        out = attn(x, y=y)
        assert out.shape == (2, 32, 8, 8)

    def test_cross_attention_2d_y(self, attn):
        x = torch.randn(2, 32, 8, 8)
        y = torch.randn(2, 64)
        out = attn(x, y=y)
        assert out.shape == (2, 32, 8, 8)

    def test_channel_mismatch_assertion(self, attn):
        x = torch.randn(2, 16, 8, 8)  # Wrong channel count
        with pytest.raises(AssertionError):
            attn(x)

    def test_gradient_flow(self, attn):
        x = torch.randn(2, 32, 4, 4, requires_grad=True)
        y = torch.randn(2, 3, 64)
        out = attn(x, y=y)
        out.sum().backward()
        assert x.grad is not None


class TestCrossAttention:
    """Test suite for the CrossAttention module."""

    @pytest.fixture
    def cross_attn(self):
        return CrossAttention(channels=32, context_dim=64, num_heads=4, dropout=0.0, use_flash=False)

    def test_output_shape(self, cross_attn):
        x = torch.randn(2, 32, 8, 8)
        ctx = torch.randn(2, 5, 64)
        out = cross_attn(x, ctx)
        assert out.shape == (2, 32, 8, 8)

    def test_2d_context(self, cross_attn):
        x = torch.randn(2, 32, 8, 8)
        ctx = torch.randn(2, 64)  # Will be unsqueezed to [2, 1, 64]
        out = cross_attn(x, ctx)
        assert out.shape == (2, 32, 8, 8)

    def test_flash_attention_branch(self):
        ca = CrossAttention(channels=32, context_dim=64, num_heads=4, use_flash=True)
        x = torch.randn(2, 32, 4, 4)
        ctx = torch.randn(2, 3, 64)
        out = ca(x, ctx)
        assert out.shape == (2, 32, 4, 4)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, cross_attn):
        x = torch.randn(2, 32, 4, 4, requires_grad=True)
        ctx = torch.randn(2, 3, 64, requires_grad=True)
        out = cross_attn(x, ctx)
        out.sum().backward()
        assert x.grad is not None
        assert ctx.grad is not None


class TestResBlock:
    """Test suite for ResBlock module."""

    @pytest.fixture
    def resblock(self):
        return ResBlock(
            in_channels=16, out_channels=32,
            time_channels=64, context_channels=48,
            num_layers=2, dropout=0.0,
            use_attention=True, use_flash=False,
        )

    @pytest.fixture
    def resblock_no_attn(self):
        return ResBlock(
            in_channels=16, out_channels=16,
            time_channels=32, context_channels=0,
            num_layers=2, dropout=0.0,
            use_attention=False,
        )

    def test_output_shape_with_attention(self, resblock):
        x = torch.randn(2, 16, 8, 8)
        t_emb = torch.randn(2, 64)
        ctx = torch.randn(2, 5, 48)
        out = resblock(x, t_emb, ctx)
        assert out.shape == (2, 32, 8, 8)

    def test_output_shape_no_attention(self, resblock_no_attn):
        x = torch.randn(2, 16, 8, 8)
        t_emb = torch.randn(2, 32)
        out = resblock_no_attn(x, t_emb)
        assert out.shape == (2, 16, 8, 8)

    def test_skip_connection_identity(self, resblock_no_attn):
        """When in_channels == out_channels, skip should be Identity."""
        assert isinstance(resblock_no_attn.res_layers[0]['skip'], torch.nn.Identity)

    def test_skip_connection_conv(self, resblock):
        """When in_channels != out_channels, skip should be Conv2d."""
        assert isinstance(resblock.res_layers[0]['skip'], torch.nn.Conv2d)

    def test_gradient_flow(self, resblock):
        x = torch.randn(2, 16, 4, 4, requires_grad=True)
        t_emb = torch.randn(2, 64)
        ctx = torch.randn(2, 3, 48)
        out = resblock(x, t_emb, ctx)
        out.sum().backward()
        assert x.grad is not None


class TestGetTimestepEmbedding:
    """Test suite for get_timestep_embedding function."""

    def test_output_shape(self):
        t = torch.tensor([0.1, 0.5, 0.9])
        emb = get_timestep_embedding(t, dim=64, continuous=True)
        assert emb.shape == (3, 64)

    def test_scalar_input(self):
        t = torch.tensor(0.5)
        emb = get_timestep_embedding(t, dim=32)
        assert emb.shape == (1, 32)

    def test_2d_input_squeeze(self):
        t = torch.tensor([[0.1], [0.5]])
        emb = get_timestep_embedding(t, dim=32)
        assert emb.shape == (2, 32)

    def test_discrete_mode(self):
        t = torch.tensor([10, 50, 100])
        emb = get_timestep_embedding(t, dim=64, continuous=False)
        assert emb.shape == (3, 64)

    def test_different_timesteps_differ(self):
        t = torch.tensor([0.1, 0.9])
        emb = get_timestep_embedding(t, dim=32)
        assert not torch.allclose(emb[0], emb[1])

    def test_finite_output(self):
        t = torch.tensor([0.0, 0.5, 1.0])
        emb = get_timestep_embedding(t, dim=128)
        assert torch.isfinite(emb).all()


class TestMinSNRLoss:
    """Test suite for min_snr_loss function."""

    def test_basic_computation(self):
        pred = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        snr = torch.tensor([1.0, 2.0, 5.0, 10.0])
        loss = min_snr_loss(pred, target, snr, gamma=5.0)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_zero_loss_identical(self):
        x = torch.randn(4, 3, 8, 8)
        snr = torch.tensor([1.0, 2.0, 5.0, 10.0])
        loss = min_snr_loss(x, x, snr, gamma=5.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_clamp_effect(self):
        """When snr > gamma, weight should be gamma/snr < 1."""
        pred = torch.ones(2, 1, 4, 4)
        target = torch.zeros(2, 1, 4, 4)
        high_snr = torch.tensor([100.0, 100.0])
        low_snr = torch.tensor([1.0, 1.0])
        loss_high = min_snr_loss(pred, target, high_snr, gamma=5.0)
        loss_low = min_snr_loss(pred, target, low_snr, gamma=5.0)
        # High SNR gets downweighted: weight=5/100=0.05 vs weight=5/1=5 (clamped at 1)
        assert loss_high.item() < loss_low.item()


class TestTextEncoderCustom:
    """Test suite for TextEncoder with custom transformer (non-BERT)."""

    @pytest.fixture
    def encoder(self):
        return TextEncoder(
            use_pretrained_model=False,
            vocabulary_size=100,
            num_layers=2,
            input_dimension=64,
            output_dimension=64,
            num_heads=4,
            context_length=20,
            dropout_rate=0.0,
            qkv_bias=False,
            scaling_value=2,
        )

    def test_output_shape(self, encoder):
        token_ids = torch.randint(0, 100, (2, 10))
        out = encoder(token_ids)
        assert out.shape == (2, 10, 64)

    def test_with_attention_mask(self, encoder):
        token_ids = torch.randint(0, 100, (2, 10))
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 8:] = True
        out = encoder(token_ids, attention_mask=mask)
        assert out.shape == (2, 10, 64)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, encoder):
        token_ids = torch.randint(0, 100, (2, 5))
        out = encoder(token_ids)
        out.sum().backward()
        # Embedding and layers should have gradients
        assert encoder.embedding.token_embedding.weight.grad is not None

    def test_different_input_output_dim(self):
        enc = TextEncoder(
            use_pretrained_model=False,
            vocabulary_size=100, num_layers=1,
            input_dimension=64, output_dimension=128,
            num_heads=4, context_length=20,
            dropout_rate=0.0, qkv_bias=False, scaling_value=2,
        )
        token_ids = torch.randint(0, 100, (2, 10))
        out = enc(token_ids)
        assert out.shape == (2, 10, 128)


class TestDiffusionNetworkGradCheckpoint:
    """Test gradient checkpointing in DiffusionNetwork."""

    @pytest.fixture
    def net_gc(self):
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
            grad_check=True,
        )

    def test_grad_checkpoint_forward(self, net_gc):
        net_gc.train()
        x = torch.randn(2, 1, 16, 16, requires_grad=True)
        t = torch.randint(0, 100, (2,))
        out = net_gc(x, t)
        assert out.shape == (2, 1, 16, 16)
        out.sum().backward()
        assert x.grad is not None

    def test_grad_checkpoint_eval_no_checkpoint(self, net_gc):
        """In eval mode, gradient checkpointing should be skipped."""
        net_gc.eval()
        x = torch.randn(2, 1, 16, 16)
        t = torch.randint(0, 100, (2,))
        with torch.no_grad():
            out = net_gc(x, t)
        assert out.shape == (2, 1, 16, 16)
