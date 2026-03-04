import torch
import pytest
from torchdiff.unclip import (
    SchedulerUnCLIP, ForwardUnCLIP, ReverseUnCLIP,
    CLIPContextProjection, CLIPEmbeddingProjection,
    UnCLIPTransformerPrior
)


class TestSchedulerUnCLIP:
    """Test suite for SchedulerUnCLIP."""

    @pytest.fixture
    def linear_scheduler(self):
        """Create a linear schedule scheduler."""
        return SchedulerUnCLIP(schedule_type="linear", train_steps=100)

    @pytest.fixture
    def cosine_scheduler(self):
        """Create a cosine schedule scheduler."""
        return SchedulerUnCLIP(schedule_type="cosine", train_steps=100)

    def test_initialization_linear(self, linear_scheduler):
        """Test linear scheduler initialization."""
        assert linear_scheduler.train_steps == 100
        assert linear_scheduler.schedule_type == "linear"
        assert linear_scheduler.betas.shape == (100,)

    def test_initialization_cosine(self, cosine_scheduler):
        """Test cosine scheduler initialization."""
        assert cosine_scheduler.train_steps == 100
        assert cosine_scheduler.schedule_type == "cosine"

    def test_invalid_schedule_type(self):
        """Test that invalid schedule type raises ValueError."""
        with pytest.raises(ValueError, match="schedule_type must be one of"):
            SchedulerUnCLIP(schedule_type="invalid")

    def test_invalid_beta_range(self):
        """Test that invalid beta range raises ValueError."""
        with pytest.raises(ValueError):
            SchedulerUnCLIP(schedule_type="linear", beta_min=0.5, beta_max=0.1)

    def test_all_schedule_types(self):
        """Test all supported schedule types initialize correctly."""
        for stype in ["linear", "cosine", "quadratic", "sigmoid", "constant", "inverse_time"]:
            sched = SchedulerUnCLIP(schedule_type=stype, train_steps=50)
            assert sched.betas.shape == (50,)
            assert sched.alphas.shape == (50,)

    def test_alphas_computation(self, linear_scheduler):
        """Test that alphas = 1 - betas."""
        expected = 1.0 - linear_scheduler.betas
        assert torch.allclose(linear_scheduler.alphas, expected)

    def test_alphas_cumprod(self, linear_scheduler):
        """Test cumulative product of alphas."""
        expected = torch.cumprod(linear_scheduler.alphas, dim=0)
        assert torch.allclose(linear_scheduler.alphas_cumprod, expected)

    def test_alphas_cumprod_prev(self, linear_scheduler):
        """Test shifted cumulative product of alphas."""
        assert linear_scheduler.alphas_cumprod_prev[0] == 1.0
        assert torch.allclose(
            linear_scheduler.alphas_cumprod_prev[1:],
            linear_scheduler.alphas_cumprod[:-1]
        )

    def test_sqrt_coefficients(self, linear_scheduler):
        """Test square root coefficients."""
        expected_sqrt = torch.sqrt(linear_scheduler.alphas_cumprod)
        expected_sqrt_one_minus = torch.sqrt(1.0 - linear_scheduler.alphas_cumprod)
        assert torch.allclose(linear_scheduler.sqrt_alphas_cumprod, expected_sqrt)
        assert torch.allclose(
            linear_scheduler.sqrt_one_minus_alphas_cumprod, expected_sqrt_one_minus
        )

    def test_inference_timesteps(self, linear_scheduler):
        """Test inference timesteps are created."""
        assert hasattr(linear_scheduler, 'inference_timesteps')
        assert linear_scheduler.inference_timesteps.dtype == torch.long

    def test_set_inf_timesteps(self, linear_scheduler):
        """Test dynamic update of inference timesteps."""
        linear_scheduler.set_inf_timesteps(10)
        assert linear_scheduler.sample_steps == 10
        assert linear_scheduler.inference_timesteps.shape[0] == 10

    def test_get_index_reshaping_2d(self, linear_scheduler):
        """Test get_index for 2D tensors."""
        t = torch.randn(4)
        result = linear_scheduler.get_index(t, torch.Size([4, 128]))
        assert result.shape == (4, 1)

    def test_get_index_reshaping_4d(self, linear_scheduler):
        """Test get_index for 4D tensors."""
        t = torch.randn(4)
        result = linear_scheduler.get_index(t, torch.Size([4, 3, 32, 32]))
        assert result.shape == (4, 1, 1, 1)

    def test_learnable_variance(self):
        """Test that learnable variance creates a parameter."""
        sched = SchedulerUnCLIP(learn_var=True, train_steps=100)
        assert 'log_variance' in dict(sched.named_parameters())

    def test_device_consistency(self, linear_scheduler):
        """Test that all buffers are on the same device."""
        device = linear_scheduler.betas.device
        assert linear_scheduler.alphas.device == device
        assert linear_scheduler.alphas_cumprod.device == device
        assert linear_scheduler.posterior_variance.device == device


class TestForwardUnCLIP:
    """Test suite for ForwardUnCLIP."""

    @pytest.fixture
    def scheduler(self):
        return SchedulerUnCLIP(schedule_type="linear", train_steps=100)

    @pytest.fixture
    def forward_noise(self, scheduler):
        return ForwardUnCLIP(scheduler, pred_type="noise")

    @pytest.fixture
    def forward_x0(self, scheduler):
        return ForwardUnCLIP(scheduler, pred_type="x0")

    def test_initialization(self, forward_noise):
        """Test ForwardUnCLIP initialization."""
        assert forward_noise.pred_type == "noise"

    def test_invalid_pred_type(self, scheduler):
        """Test that invalid pred_type raises ValueError."""
        with pytest.raises(ValueError, match="pred_type must be one of"):
            ForwardUnCLIP(scheduler, pred_type="invalid")

    def test_forward_output_shapes_4d(self, forward_noise):
        """Test forward pass shapes for image tensors."""
        x0 = torch.randn(4, 3, 32, 32)
        t = torch.randint(0, 100, (4,))
        noise = torch.randn_like(x0)
        xt, target = forward_noise(x0, noise, t)
        assert xt.shape == x0.shape
        assert target.shape == x0.shape

    def test_forward_output_shapes_2d(self, forward_noise):
        """Test forward pass shapes for embedding tensors."""
        x0 = torch.randn(4, 512)
        t = torch.randint(0, 100, (4,))
        noise = torch.randn_like(x0)
        xt, target = forward_noise(x0, noise, t)
        assert xt.shape == x0.shape
        assert target.shape == x0.shape

    def test_noise_prediction_target(self, forward_noise):
        """Test that noise prediction returns noise as target."""
        x0 = torch.randn(4, 3, 16, 16)
        t = torch.randint(0, 100, (4,))
        noise = torch.randn_like(x0)
        _, target = forward_noise(x0, noise, t)
        assert torch.allclose(target, noise)

    def test_x0_prediction_target(self, forward_x0):
        """Test that x0 prediction returns x0 as target."""
        x0 = torch.randn(4, 3, 16, 16)
        t = torch.randint(0, 100, (4,))
        noise = torch.randn_like(x0)
        _, target = forward_x0(x0, noise, t)
        assert torch.allclose(target, x0)

    def test_noising_formula(self, forward_noise, scheduler):
        """Test forward noising follows correct formula."""
        x0 = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 100, (2,))
        noise = torch.randn_like(x0)
        xt, _ = forward_noise(x0, noise, t)
        sqrt_alpha = scheduler.get_index(scheduler.sqrt_alphas_cumprod[t], x0.shape)
        sqrt_one_minus = scheduler.get_index(scheduler.sqrt_one_minus_alphas_cumprod[t], x0.shape)
        expected = sqrt_alpha * x0 + sqrt_one_minus * noise
        assert torch.allclose(xt, expected, atol=1e-6)

    def test_deterministic_same_noise(self, forward_noise):
        """Test that same inputs produce same outputs."""
        x0 = torch.randn(2, 512)
        t = torch.tensor([10, 20])
        noise = torch.randn_like(x0)
        xt1, t1 = forward_noise(x0, noise, t)
        xt2, t2 = forward_noise(x0, noise, t)
        assert torch.allclose(xt1, xt2)
        assert torch.allclose(t1, t2)

    def test_invalid_timestep_raises(self, forward_noise):
        """Test that out-of-range timestep raises ValueError."""
        x0 = torch.randn(2, 512)
        t = torch.tensor([100, 200])  # 100 is out of range for train_steps=100
        noise = torch.randn_like(x0)
        with pytest.raises(ValueError, match="t must be in"):
            forward_noise(x0, noise, t)


class TestReverseUnCLIP:
    """Test suite for ReverseUnCLIP."""

    @pytest.fixture
    def scheduler(self):
        return SchedulerUnCLIP(
            schedule_type="linear", train_steps=100, sample_steps=10
        )

    @pytest.fixture
    def reverse_noise(self, scheduler):
        return ReverseUnCLIP(scheduler, pred_type="noise", eta=0.0, clip_=True)

    @pytest.fixture
    def reverse_x0(self, scheduler):
        return ReverseUnCLIP(scheduler, pred_type="x0", eta=0.0, clip_=True)

    def test_initialization(self, reverse_noise):
        """Test ReverseUnCLIP initialization."""
        assert reverse_noise.pred_type == "noise"
        assert reverse_noise.eta == 0.0
        assert reverse_noise.clip_ is True

    def test_invalid_pred_type(self, scheduler):
        """Test that invalid pred_type raises ValueError."""
        with pytest.raises(ValueError, match="pred_type must be one of"):
            ReverseUnCLIP(scheduler, pred_type="invalid")

    def test_predict_x0_from_noise_shape(self, reverse_noise):
        """Test predict_x0 output shapes."""
        xt = torch.randn(2, 3, 16, 16)
        t = torch.tensor([5, 7])  # tau indices
        pred = torch.randn_like(xt)
        x0 = reverse_noise.predict_x0(xt, t, pred)
        assert x0.shape == xt.shape

    def test_predict_x0_clipping(self, reverse_noise):
        """Test that x0 prediction is clipped to [-1, 1]."""
        xt = torch.randn(2, 3, 16, 16) * 5
        t = torch.tensor([5, 7])
        pred = torch.randn_like(xt) * 5
        x0 = reverse_noise.predict_x0(xt, t, pred)
        assert torch.all(x0 >= -1.0)
        assert torch.all(x0 <= 1.0)

    def test_predict_x0_no_clipping(self, scheduler):
        """Test x0 prediction without clipping."""
        reverse = ReverseUnCLIP(scheduler, pred_type="x0", clip_=False)
        xt = torch.randn(2, 512)
        t = torch.tensor([5, 7])
        pred = torch.randn_like(xt) * 5
        x0 = reverse.predict_x0(xt, t, pred)
        assert torch.allclose(x0, pred)

    def test_predict_noise_shape(self, reverse_noise):
        """Test predict_noise output shape."""
        xt = torch.randn(2, 3, 16, 16)
        t = torch.tensor([5, 7])
        x0_pred = torch.randn_like(xt)
        noise = reverse_noise.predict_noise(xt, t, x0_pred)
        assert noise.shape == xt.shape

    def test_forward_output_shapes(self, reverse_noise):
        """Test reverse diffusion step output shapes."""
        xt = torch.randn(2, 3, 16, 16)
        t = torch.tensor([5, 5])
        t_prev = torch.tensor([4, 4])
        pred = torch.randn_like(xt)
        x_prev, pred_x0 = reverse_noise(xt, t, t_prev, pred)
        assert x_prev.shape == xt.shape
        assert pred_x0.shape == xt.shape

    def test_forward_2d_shapes(self, reverse_noise):
        """Test reverse step with 2D embedding tensors."""
        xt = torch.randn(2, 512)
        t = torch.tensor([5, 5])
        t_prev = torch.tensor([4, 4])
        pred = torch.randn_like(xt)
        x_prev, pred_x0 = reverse_noise(xt, t, t_prev, pred)
        assert x_prev.shape == xt.shape
        assert pred_x0.shape == xt.shape

    def test_deterministic_eta_zero(self, reverse_noise):
        """Test deterministic sampling when eta=0."""
        xt = torch.randn(2, 3, 16, 16)
        t = torch.tensor([5, 5])
        t_prev = torch.tensor([4, 4])
        pred = torch.randn_like(xt)
        torch.manual_seed(42)
        x1, _ = reverse_noise(xt, t, t_prev, pred)
        torch.manual_seed(123)
        x2, _ = reverse_noise(xt, t, t_prev, pred)
        assert torch.allclose(x1, x2, atol=1e-6)

    def test_stochastic_eta_nonzero(self, scheduler):
        """Test stochastic sampling when eta>0."""
        reverse = ReverseUnCLIP(scheduler, pred_type="noise", eta=1.0)
        xt = torch.randn(2, 3, 16, 16)
        t = torch.tensor([5, 5])
        t_prev = torch.tensor([4, 4])
        pred = torch.randn_like(xt)
        torch.manual_seed(42)
        x1, _ = reverse(xt, t, t_prev, pred)
        torch.manual_seed(123)
        x2, _ = reverse(xt, t, t_prev, pred)
        assert not torch.allclose(x1, x2)

    def test_set_pred_type(self, reverse_noise):
        """Test changing prediction type."""
        reverse_noise.set_pred_type("x0")
        assert reverse_noise.pred_type == "x0"

    def test_set_pred_type_invalid(self, reverse_noise):
        """Test that invalid pred_type raises ValueError."""
        with pytest.raises(ValueError):
            reverse_noise.set_pred_type("invalid")

    def test_invalid_timestep_raises(self, reverse_noise):
        """Test that out-of-range timestep raises ValueError."""
        xt = torch.randn(2, 512)
        t = torch.tensor([20, 20])  # out of range for sample_steps=10
        t_prev = torch.tensor([19, 19])
        pred = torch.randn_like(xt)
        with pytest.raises(ValueError, match="t must be in"):
            reverse_noise(xt, t, t_prev, pred)

    def test_all_pred_types_forward(self, scheduler):
        """Test all prediction types work."""
        for pred_type in ["noise", "x0"]:
            reverse = ReverseUnCLIP(scheduler, pred_type=pred_type)
            xt = torch.randn(2, 3, 16, 16)
            t = torch.tensor([5, 5])
            t_prev = torch.tensor([4, 4])
            pred = torch.randn_like(xt)
            x_prev, pred_x0 = reverse(xt, t, t_prev, pred)
            assert x_prev.shape == xt.shape
            assert torch.all(torch.isfinite(x_prev))


class TestCLIPContextProjection:
    """Test suite for CLIPContextProjection."""

    @pytest.fixture
    def proj_default(self):
        return CLIPContextProjection(clip_embed_dim=512, num_tokens=4)

    @pytest.fixture
    def proj_custom_output(self):
        return CLIPContextProjection(clip_embed_dim=320, num_tokens=4, output_dim=512)

    def test_initialization(self, proj_default):
        """Test default initialization."""
        assert proj_default.clip_embed_dim == 512
        assert proj_default.num_tokens == 4
        assert proj_default.output_dim == 512

    def test_initialization_custom_output_dim(self, proj_custom_output):
        """Test initialization with custom output_dim."""
        assert proj_custom_output.clip_embed_dim == 320
        assert proj_custom_output.output_dim == 512

    def test_forward_default(self, proj_default):
        """Test forward pass with default dims."""
        z_i = torch.randn(4, 512)
        c = proj_default(z_i)
        assert c.shape == (4, 4, 512)

    def test_forward_custom_output(self, proj_custom_output):
        """Test forward pass with custom output_dim."""
        z_i = torch.randn(4, 320)
        c = proj_custom_output(z_i)
        assert c.shape == (4, 4, 512)

    def test_output_normalized(self, proj_default):
        """Test that output is layer-normalized (finite values)."""
        z_i = torch.randn(4, 512)
        c = proj_default(z_i)
        assert torch.all(torch.isfinite(c))

    def test_gradient_flow(self, proj_default):
        """Test gradient flow."""
        z_i = torch.randn(4, 512)
        c = proj_default(z_i)
        c.mean().backward()
        assert proj_default.clip_proj.weight.grad is not None


class TestCLIPEmbeddingProjection:
    """Test suite for CLIPEmbeddingProjection."""

    @pytest.fixture
    def proj(self):
        return CLIPEmbeddingProjection(
            clip_embed_dim=512, trans_embed_dim=320,
            hidden_dim=256, num_layers=2, dropout=0.0
        )

    def test_initialization(self, proj):
        """Test initialization."""
        assert proj.clip_embed_dim == 512
        assert proj.trans_embed_dim == 320

    def test_forward_projection(self, proj):
        """Test forward projection reduces dimensionality."""
        x = torch.randn(4, 512)
        out = proj(x)
        assert out.shape == (4, 320)

    def test_inverse_projection(self, proj):
        """Test inverse projection restores dimensionality."""
        x = torch.randn(4, 320)
        out = proj.inverse_transform(x)
        assert out.shape == (4, 512)

    def test_roundtrip_shape(self, proj):
        """Test forward then inverse preserves shape."""
        x = torch.randn(4, 512)
        reduced = proj(x)
        restored = proj.inverse_transform(reduced)
        assert restored.shape == x.shape

    def test_rec_loss(self, proj):
        """Test reconstruction loss is computed."""
        x = torch.randn(4, 512)
        loss = proj.rec_loss(x)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flow(self, proj):
        """Test gradient flow through both projections."""
        x = torch.randn(4, 512)
        loss = proj.rec_loss(x)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in proj.parameters())
        assert has_grad


class TestUnCLIPTransformerPrior:
    """Test suite for UnCLIPTransformerPrior."""

    @pytest.fixture
    def prior(self):
        scheduler = SchedulerUnCLIP(
            schedule_type="linear", train_steps=100, sample_steps=10
        )
        fwd = ForwardUnCLIP(scheduler, pred_type="x0")
        rwd = ReverseUnCLIP(scheduler, pred_type="x0")
        return UnCLIPTransformerPrior(
            fwd_unclip=fwd,
            rwd_unclip=rwd,
            trans_embed_dim=64,
            num_layers=2,
            num_att_heads=2,
            ff_dim=128,
            max_sequence_length=2,
            dropout=0.0,
            use_flash=False,
            grad_check=False,
        )

    def test_initialization(self, prior):
        """Test prior initialization."""
        assert prior.trans_embed_dim == 64
        assert len(prior.transformer_blocks) == 2

    def test_forward_shape(self, prior):
        """Test forward pass output shapes."""
        text_embed = torch.randn(2, 64)
        noisy_img_embed = torch.randn(2, 64)
        timesteps = torch.tensor([10, 20])
        pred = prior(text_embed, noisy_img_embed, timesteps)
        assert pred.shape == (2, 64)

    def test_output_finite(self, prior):
        """Test that output is finite."""
        text_embed = torch.randn(2, 64)
        noisy_img_embed = torch.randn(2, 64)
        timesteps = torch.tensor([10, 20])
        pred = prior(text_embed, noisy_img_embed, timesteps)
        assert torch.all(torch.isfinite(pred))

    def test_gradient_flow(self, prior):
        """Test gradient flow through the prior."""
        text_embed = torch.randn(2, 64)
        noisy_img_embed = torch.randn(2, 64)
        timesteps = torch.tensor([10, 20])
        pred = prior(text_embed, noisy_img_embed, timesteps)
        pred.mean().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in prior.parameters() if p.requires_grad)
        assert has_grad

    def test_different_batch_sizes(self, prior):
        """Test with different batch sizes."""
        for bs in [1, 4]:
            text_embed = torch.randn(bs, 64)
            noisy_img_embed = torch.randn(bs, 64)
            timesteps = torch.randint(0, 100, (bs,))
            pred = prior(text_embed, noisy_img_embed, timesteps)
            assert pred.shape == (bs, 64)


class TestIntegrationUnCLIP:
    """Integration tests for the UnCLIP forward-reverse pipeline."""

    @pytest.fixture
    def components(self):
        scheduler = SchedulerUnCLIP(
            schedule_type="linear", train_steps=100, sample_steps=10
        )
        fwd = ForwardUnCLIP(scheduler, pred_type="noise")
        rwd = ReverseUnCLIP(scheduler, pred_type="noise", eta=0.0, clip_=False)
        return scheduler, fwd, rwd

    def test_forward_reverse_consistency_4d(self, components):
        """Test reverse recovers x0 from true noise (4D)."""
        scheduler, fwd, rwd = components
        x0 = torch.randn(1, 3, 16, 16)
        t_train = torch.tensor([50])
        noise = torch.randn_like(x0)
        xt, _ = fwd(x0, noise, t_train)
        # Use tau index that maps to timestep 50
        t_idx = torch.tensor([5])  # 5 * 10 = 50
        x0_pred = rwd.predict_x0(xt, t_idx, noise)
        assert torch.allclose(x0_pred, x0, atol=1e-4)

    def test_forward_reverse_consistency_2d(self, components):
        """Test reverse recovers x0 from true noise (2D embeddings)."""
        scheduler, fwd, rwd = components
        x0 = torch.randn(1, 512)
        t_train = torch.tensor([50])
        noise = torch.randn_like(x0)
        xt, _ = fwd(x0, noise, t_train)
        t_idx = torch.tensor([5])
        x0_pred = rwd.predict_x0(xt, t_idx, noise)
        assert torch.allclose(x0_pred, x0, atol=1e-4)

    def test_different_pred_types_work(self, components):
        """Test all prediction types through forward-reverse."""
        scheduler, _, _ = components
        for pred_type in ["noise", "x0"]:
            fwd = ForwardUnCLIP(scheduler, pred_type=pred_type)
            rwd = ReverseUnCLIP(scheduler, pred_type=pred_type)
            x0 = torch.randn(2, 3, 16, 16)
            t = torch.randint(0, 100, (2,))
            noise = torch.randn_like(x0)
            xt, target = fwd(x0, noise, t)
            assert xt.shape == x0.shape
            assert target.shape == x0.shape
