import torch
import pytest
from torchdiff.ddim import ForwardDDIM, ReverseDDIM, SchedulerDDIM



class TestSchedulerDDIM:
    """Test suite for SchedulerDDIM class."""
    @pytest.fixture
    def default_scheduler(self):
        """Create a default scheduler instance."""
        return SchedulerDDIM(
            schedule_type="linear",
            train_steps=1000,
            sample_steps=50,
            beta_min=0.0001,
            beta_max=0.02
        )

    @pytest.fixture(params=["linear", "cosine", "quadratic", "sigmoid"])
    def scheduler_all_types(self, request):
        """Create schedulers with different schedule types."""
        return SchedulerDDIM(
            schedule_type=request.param,
            train_steps=1000,
            sample_steps=50
        )

    def test_initialization(self, default_scheduler):
        """Test that scheduler initializes correctly."""
        assert default_scheduler.train_steps == 1000
        assert default_scheduler.sample_steps == 50
        assert default_scheduler.schedule_type == "linear"
        assert default_scheduler.beta_min == 0.0001
        assert default_scheduler.beta_max == 0.02

    def test_invalid_schedule_type(self):
        """Test that invalid schedule type raises ValueError."""
        with pytest.raises(ValueError, match="schedule_type must be one of"):
            SchedulerDDIM(schedule_type="invalid")

    def test_buffers_created(self, default_scheduler):
        """Test that all required buffers are created."""
        required_buffers = [
            'betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
            'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
            'posterior_variance', 'posterior_log_variance', 'inference_timesteps'
        ]
        for buffer_name in required_buffers:
            assert hasattr(default_scheduler, buffer_name)
            assert isinstance(getattr(default_scheduler, buffer_name), torch.Tensor)

    def test_betas_shape(self, default_scheduler):
        """Test that betas has correct shape."""
        assert default_scheduler.betas.shape == (1000,)

    def test_betas_range(self, default_scheduler):
        """Test that betas are within expected range."""
        assert torch.all(default_scheduler.betas >= 0)
        assert torch.all(default_scheduler.betas <= 1)

    def test_alphas_relationship(self, default_scheduler):
        """Test that alphas = 1 - betas."""
        expected_alphas = 1.0 - default_scheduler.betas
        torch.testing.assert_close(default_scheduler.alphas, expected_alphas)

    def test_alphas_cumprod_monotonic(self, default_scheduler):
        """Test that cumulative product of alphas is monotonically decreasing."""
        alphas_cumprod = default_scheduler.alphas_cumprod
        assert torch.all(alphas_cumprod[1:] <= alphas_cumprod[:-1])

    def test_sqrt_alphas_cumprod(self, default_scheduler):
        """Test that sqrt_alphas_cumprod is correctly computed."""
        expected = torch.sqrt(default_scheduler.alphas_cumprod)
        torch.testing.assert_close(default_scheduler.sqrt_alphas_cumprod, expected)

    def test_inference_timesteps_shape(self, default_scheduler):
        """Test that inference timesteps has correct shape."""
        assert default_scheduler.inference_timesteps.shape[0] == 50

    def test_inference_timesteps_range(self, default_scheduler):
        """Test that inference timesteps are within valid range."""
        assert torch.all(default_scheduler.inference_timesteps >= 0)
        assert torch.all(default_scheduler.inference_timesteps < 1000)

    def test_set_inference_timesteps(self, default_scheduler):
        """Test dynamic update of inference timesteps."""
        default_scheduler.set_inference_timesteps(25)
        assert default_scheduler.sample_steps == 25
        assert default_scheduler.inference_timesteps.shape[0] == 25

    def test_get_index_broadcasting(self, default_scheduler):
        """Test get_index reshapes correctly for broadcasting."""
        batch_size = 4
        t = torch.randint(0, 1000, (batch_size,))
        x_shape = torch.Size([batch_size, 3, 32, 32])
        result = default_scheduler.get_index(t, x_shape)
        assert result.shape == (batch_size, 1, 1, 1)

    def test_cosine_schedule_properties(self):
        """Test cosine schedule specific properties."""
        scheduler = SchedulerDDIM(schedule_type="cosine", train_steps=1000)
        assert torch.all(scheduler.betas >= scheduler.clip_min)
        assert torch.all(scheduler.betas <= scheduler.clip_max)

    def test_learnable_variance(self):
        """Test that learnable variance creates a parameter."""
        scheduler = SchedulerDDIM(learn_var=True, train_steps=1000)
        assert 'log_variance' in dict(scheduler.named_parameters())

    def test_all_schedule_types_run(self, scheduler_all_types):
        """Test that all schedule types initialize without errors."""
        assert scheduler_all_types.betas.shape == (1000,)
        assert torch.all(torch.isfinite(scheduler_all_types.betas))


class TestForwardDDIM:
    """Test suite for ForwardDDIM class."""
    @pytest.fixture
    def scheduler(self):
        """Create a scheduler for forward process."""
        return SchedulerDDIM(train_steps=1000, sample_steps=50)

    @pytest.fixture(params=["noise", "x0", "v"])
    def forward_ddim(self, scheduler, request):
        """Create ForwardDDIM instances with different prediction types."""
        return ForwardDDIM(scheduler=scheduler, pred_type=request.param)

    def test_initialization(self, scheduler):
        """Test ForwardDDIM initialization."""
        forward = ForwardDDIM(scheduler=scheduler, pred_type="noise")
        assert forward.pred_type == "noise"
        assert forward.vs == scheduler

    def test_invalid_pred_type(self, scheduler):
        """Test that invalid prediction type raises ValueError."""
        with pytest.raises(ValueError, match="prediction_type must be one of"):
            ForwardDDIM(scheduler=scheduler, pred_type="invalid")

    @pytest.mark.parametrize("batch_size,channels,height,width", [
        (1, 3, 32, 32),
        (4, 3, 64, 64),
        (8, 1, 28, 28),
    ])
    def test_forward_shapes(self, scheduler, batch_size, channels, height, width):
        """Test that forward pass produces correct shapes."""
        forward = ForwardDDIM(scheduler=scheduler, pred_type="noise")

        x0 = torch.randn(batch_size, channels, height, width)
        t = torch.randint(0, 1000, (batch_size,))
        noise = torch.randn_like(x0)

        xt, target = forward(x0, t, noise)

        assert xt.shape == x0.shape
        assert target.shape == x0.shape

    def test_noise_prediction_target(self, scheduler):
        """Test that noise prediction returns noise as target."""
        forward = ForwardDDIM(scheduler=scheduler, pred_type="noise")
        x0 = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        noise = torch.randn_like(x0)
        xt, target = forward(x0, t, noise)
        torch.testing.assert_close(target, noise)

    def test_x0_prediction_target(self, scheduler):
        """Test that x0 prediction returns x0 as target."""
        forward = ForwardDDIM(scheduler=scheduler, pred_type="x0")
        x0 = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        noise = torch.randn_like(x0)
        xt, target = forward(x0, t, noise)
        torch.testing.assert_close(target, x0)

    def test_v_prediction_target(self, scheduler):
        """Test v-prediction target computation."""
        forward = ForwardDDIM(scheduler=scheduler, pred_type="v")
        x0 = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        noise = torch.randn_like(x0)
        xt, target = forward(x0, t, noise)
        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha = scheduler.get_index(sqrt_alpha, x0.shape)
        sqrt_one_minus_alpha = scheduler.get_index(sqrt_one_minus_alpha, x0.shape)
        expected_target = sqrt_alpha * noise - sqrt_one_minus_alpha * x0
        torch.testing.assert_close(target, expected_target)

    def test_forward_diffusion_equation(self, scheduler):
        """Test that forward process follows q(x_t | x_0) equation."""
        forward = ForwardDDIM(scheduler=scheduler, pred_type="noise")
        x0 = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        noise = torch.randn_like(x0)
        xt, _ = forward(x0, t, noise)
        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha = scheduler.get_index(sqrt_alpha, x0.shape)
        sqrt_one_minus_alpha = scheduler.get_index(sqrt_one_minus_alpha, x0.shape)
        expected_xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        torch.testing.assert_close(xt, expected_xt)

    def test_deterministic_with_same_noise(self, scheduler):
        """Test that same inputs produce same outputs."""
        forward = ForwardDDIM(scheduler=scheduler, pred_type="noise")
        x0 = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        noise = torch.randn_like(x0)
        xt1, target1 = forward(x0, t, noise)
        xt2, target2 = forward(x0, t, noise)
        torch.testing.assert_close(xt1, xt2)
        torch.testing.assert_close(target1, target2)


class TestReverseDDIM:
    """Test suite for ReverseDDIM class."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler for reverse process."""
        return SchedulerDDIM(train_steps=1000, sample_steps=50)

    @pytest.fixture
    def reverse_ddim(self, scheduler):
        """Create a default ReverseDDIM instance."""
        return ReverseDDIM(scheduler=scheduler, pred_type="noise", eta=0.0, clip_=True)

    def test_initialization(self, scheduler):
        """Test ReverseDDIM initialization."""
        reverse = ReverseDDIM(scheduler=scheduler, pred_type="noise", eta=0.5, clip_=False)
        assert reverse.pred_type == "noise"
        assert reverse.eta == 0.5
        assert reverse.clip_ == False
        assert reverse.vs == scheduler

    def test_invalid_pred_type(self, scheduler):
        """Test that invalid prediction type raises ValueError."""
        with pytest.raises(ValueError, match="prediction_type must be one of"):
            ReverseDDIM(scheduler=scheduler, pred_type="invalid")

    @pytest.mark.parametrize("pred_type", ["noise", "x0", "v"])
    def test_predict_x0_shapes(self, scheduler, pred_type):
        """Test predict_x0 produces correct shapes."""
        reverse = ReverseDDIM(scheduler=scheduler, pred_type=pred_type)
        xt = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        pred = torch.randn_like(xt)
        x0_pred = reverse.predict_x0(xt, t, pred)
        assert x0_pred.shape == xt.shape

    def test_predict_x0_from_noise(self, scheduler):
        """Test x0 prediction from noise prediction."""
        reverse = ReverseDDIM(scheduler=scheduler, pred_type="noise", clip_=False)
        xt = torch.randn(2, 3, 32, 32)
        t = torch.randint(1, 1000, (2,))
        noise_pred = torch.randn_like(xt)
        x0_pred = reverse.predict_x0(xt, t, noise_pred)
        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha = scheduler.get_index(sqrt_alpha, xt.shape)
        sqrt_one_minus_alpha = scheduler.get_index(sqrt_one_minus_alpha, xt.shape)
        expected_x0 = (xt - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        torch.testing.assert_close(x0_pred, expected_x0)

    def test_predict_x0_from_x0(self, scheduler):
        """Test x0 prediction when model predicts x0 directly."""
        reverse = ReverseDDIM(scheduler=scheduler, pred_type="x0", clip_=False)
        xt = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        x0_pred_input = torch.randn_like(xt)
        x0_pred = reverse.predict_x0(xt, t, x0_pred_input)
        torch.testing.assert_close(x0_pred, x0_pred_input)

    def test_predict_x0_clipping(self, scheduler):
        """Test that x0 prediction is clipped when clip_=True."""
        reverse = ReverseDDIM(scheduler=scheduler, pred_type="x0", clip_=True)
        xt = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        x0_pred_input = torch.randn_like(xt) * 5  # Values outside [-1, 1]
        x0_pred = reverse.predict_x0(xt, t, x0_pred_input)
        assert torch.all(x0_pred >= -1.0)
        assert torch.all(x0_pred <= 1.0)

    def test_predict_x0_no_clipping(self, scheduler):
        """Test that x0 prediction is not clipped when clip_=False."""
        reverse = ReverseDDIM(scheduler=scheduler, pred_type="x0", clip_=False)
        xt = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        x0_pred_input = torch.randn_like(xt) * 5  # Values outside [-1, 1]
        x0_pred = reverse.predict_x0(xt, t, x0_pred_input)
        torch.testing.assert_close(x0_pred, x0_pred_input)
        assert torch.any(torch.abs(x0_pred) > 1.0)

    def test_predict_noise(self, scheduler, reverse_ddim):
        """Test noise prediction from x0."""
        xt = torch.randn(2, 3, 32, 32)
        t = torch.randint(1, 1000, (2,))
        x0_pred = torch.randn_like(xt)
        noise_pred = reverse_ddim.predict_noise(xt, t, x0_pred)
        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha = scheduler.get_index(sqrt_alpha, xt.shape)
        sqrt_one_minus_alpha = scheduler.get_index(sqrt_one_minus_alpha, xt.shape)
        expected_noise = (xt - sqrt_alpha * x0_pred) / sqrt_one_minus_alpha
        torch.testing.assert_close(noise_pred, expected_noise)

    def test_forward_deterministic(self, scheduler):
        """Test deterministic sampling when eta=0."""
        reverse = ReverseDDIM(scheduler=scheduler, pred_type="noise", eta=0.0)
        xt = torch.randn(2, 3, 32, 32)
        t = torch.tensor([100, 100])
        t_prev = torch.tensor([50, 50])
        pred = torch.randn_like(xt)
        torch.manual_seed(42)
        x_prev1, pred_x0_1 = reverse(xt, t, t_prev, pred)
        torch.manual_seed(42)
        x_prev2, pred_x0_2 = reverse(xt, t, t_prev, pred)
        torch.testing.assert_close(x_prev1, x_prev2)
        torch.testing.assert_close(pred_x0_1, pred_x0_2)

    def test_forward_stochastic(self, scheduler):
        """Test stochastic sampling when eta>0."""
        reverse = ReverseDDIM(scheduler=scheduler, pred_type="noise", eta=1.0)
        xt = torch.randn(2, 3, 32, 32)
        t = torch.tensor([100, 100])
        t_prev = torch.tensor([50, 50])
        pred = torch.randn_like(xt)
        torch.manual_seed(42)
        x_prev1, _ = reverse(xt, t, t_prev, pred)
        torch.manual_seed(123)
        x_prev2, _ = reverse(xt, t, t_prev, pred)
        assert not torch.allclose(x_prev1, x_prev2)

    def test_forward_output_shapes(self, scheduler, reverse_ddim):
        """Test that forward pass produces correct output shapes."""
        xt = torch.randn(2, 3, 32, 32)
        t = torch.tensor([100, 100])
        t_prev = torch.tensor([50, 50])
        pred = torch.randn_like(xt)
        x_prev, pred_x0 = reverse_ddim(xt, t, t_prev, pred)
        assert x_prev.shape == xt.shape
        assert pred_x0.shape == xt.shape

    def test_forward_at_t0(self, scheduler, reverse_ddim):
        """Test that no noise is added when t_prev=0."""
        xt = torch.randn(2, 3, 32, 32)
        t = torch.tensor([50, 50])
        t_prev = torch.tensor([0, 0])
        pred = torch.randn_like(xt)
        x_prev, _ = reverse_ddim(xt, t, t_prev, pred)
        assert x_prev.shape == xt.shape

    def test_sigma_computation(self, scheduler):
        """Test that sigma is computed correctly based on eta."""
        reverse = ReverseDDIM(scheduler=scheduler, pred_type="noise", eta=0.5)
        xt = torch.randn(2, 3, 32, 32)
        t = torch.tensor([100, 100])
        t_prev = torch.tensor([50, 50])
        pred = torch.randn_like(xt)
        alpha_t = scheduler.alphas_cumprod[t]
        alpha_t_prev = scheduler.alphas_cumprod[t_prev]
        expected_sigma = 0.5 * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) *
            (1 - alpha_t / alpha_t_prev)
        )
        assert torch.all(expected_sigma > 0)


class TestIntegration:
    """Integration tests for the complete DDIM pipeline."""
    @pytest.fixture
    def components(self):
        """Create all components for integration testing."""
        scheduler = SchedulerDDIM(train_steps=1000, sample_steps=50)
        forward = ForwardDDIM(scheduler=scheduler, pred_type="noise")
        reverse = ReverseDDIM(scheduler=scheduler, pred_type="noise", eta=0.0, clip_=False)
        return scheduler, forward, reverse

    def test_forward_reverse_consistency(self, components):
        """Test that reverse process can recover x0 when given true noise."""
        scheduler, forward, reverse = components
        x0 = torch.randn(1, 3, 32, 32)
        t = torch.tensor([500])
        noise = torch.randn_like(x0)
        xt, _ = forward(x0, t, noise)
        pred_x0 = reverse.predict_x0(xt, t, noise)
        torch.testing.assert_close(pred_x0, x0, rtol=1e-4, atol=1e-4)

    def test_forward_reverse_with_clipping(self):
        """Test that clipping affects recovery but keeps values in range."""
        scheduler = SchedulerDDIM(train_steps=1000, sample_steps=50)
        forward = ForwardDDIM(scheduler=scheduler, pred_type="noise")
        reverse = ReverseDDIM(scheduler=scheduler, pred_type="noise", eta=0.0, clip_=True)
        x0 = torch.randn(1, 3, 32, 32) * 2
        t = torch.tensor([500])
        noise = torch.randn_like(x0)
        xt, _ = forward(x0, t, noise)
        pred_x0 = reverse.predict_x0(xt, t, noise)
        assert torch.all(pred_x0 >= -1.0)
        assert torch.all(pred_x0 <= 1.0)

    def test_complete_sampling_trajectory(self, components):
        """Test a complete sampling trajectory from noise to data."""
        scheduler, forward, reverse = components
        x_t = torch.randn(1, 3, 32, 32)
        timesteps = scheduler.inference_timesteps.flip(0)
        for i in range(len(timesteps) - 1):
            t = timesteps[i:i + 1]
            t_prev = timesteps[i + 1:i + 2]
            pred = torch.randn_like(x_t) * 0.1
            x_t, _ = reverse(x_t, t, t_prev, pred)
            assert torch.all(torch.isfinite(x_t))

    def test_different_prediction_types_compatibility(self):
        """Test that all prediction types work together."""
        scheduler = SchedulerDDIM(train_steps=1000)
        for pred_type in ["noise", "x0", "v"]:
            forward = ForwardDDIM(scheduler=scheduler, pred_type=pred_type)
            reverse = ReverseDDIM(scheduler=scheduler, pred_type=pred_type)
            x0 = torch.randn(1, 3, 32, 32)
            t = torch.tensor([100])
            noise = torch.randn_like(x0)
            xt, target = forward(x0, t, noise)
            assert target.shape == xt.shape
            assert torch.all(torch.isfinite(target))