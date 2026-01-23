import torch
import pytest
from torchdiff.ddpm import SchedulerDDPM, ForwardDDPM, ReverseDDPM




class TestSchedulerDDPM:
    """Test suite for SchedulerDDPM class."""

    @pytest.fixture
    def linear_scheduler(self):
        """Create a linear schedule scheduler."""
        return SchedulerDDPM(schedule_type="linear", time_steps=100)

    @pytest.fixture
    def cosine_scheduler(self):
        """Create a cosine schedule scheduler."""
        return SchedulerDDPM(schedule_type="cosine", time_steps=100)

    def test_initialization_linear(self, linear_scheduler):
        """Test linear scheduler initialization."""
        assert linear_scheduler.time_steps == 100
        assert linear_scheduler.schedule_type == "linear"
        assert linear_scheduler.betas.shape == (100,)

    def test_initialization_cosine(self, cosine_scheduler):
        """Test cosine scheduler initialization."""
        assert cosine_scheduler.time_steps == 100
        assert cosine_scheduler.schedule_type == "cosine"
        assert cosine_scheduler.betas.shape == (100,)

    def test_invalid_schedule_type(self):
        """Test that invalid schedule type raises ValueError."""
        with pytest.raises(ValueError, match="schedule_type must be one of"):
            SchedulerDDPM(schedule_type="invalid")

    def test_all_schedule_types(self):
        """Test all supported schedule types initialize correctly."""
        for schedule_type in ["linear", "cosine", "quadratic", "sigmoid"]:
            scheduler = SchedulerDDPM(schedule_type=schedule_type, time_steps=50)
            assert scheduler.betas.shape == (50,)
            assert scheduler.alphas.shape == (50,)

    def test_beta_bounds_linear(self):
        """Test that linear schedule betas are within expected bounds."""
        scheduler = SchedulerDDPM(
            schedule_type="linear",
            time_steps=100,
            beta_min=0.0001,
            beta_max=0.02
        )
        assert torch.all(scheduler.betas >= 0.0001)
        assert torch.all(scheduler.betas <= 0.02)
        assert torch.isclose(scheduler.betas[0], torch.tensor(0.0001), atol=1e-6)
        assert torch.isclose(scheduler.betas[-1], torch.tensor(0.02), atol=1e-6)

    def test_beta_bounds_cosine(self):
        """Test that cosine schedule betas are clipped correctly."""
        scheduler = SchedulerDDPM(
            schedule_type="cosine",
            time_steps=100,
            clip_min=0.0001,
            clip_max=0.9999
        )
        assert torch.all(scheduler.betas >= 0.0001)
        assert torch.all(scheduler.betas <= 0.9999)

    def test_alphas_computation(self, linear_scheduler):
        """Test that alphas are computed correctly from betas."""
        expected_alphas = 1.0 - linear_scheduler.betas
        assert torch.allclose(linear_scheduler.alphas, expected_alphas)

    def test_alphas_cumprod(self, linear_scheduler):
        """Test cumulative product of alphas."""
        expected_cumprod = torch.cumprod(linear_scheduler.alphas, dim=0)
        assert torch.allclose(linear_scheduler.alphas_cumprod, expected_cumprod)

    def test_alphas_cumprod_prev(self, linear_scheduler):
        """Test shifted cumulative product of alphas."""
        assert linear_scheduler.alphas_cumprod_prev[0] == 1.0
        assert torch.allclose(
            linear_scheduler.alphas_cumprod_prev[1:],
            linear_scheduler.alphas_cumprod[:-1]
        )

    def test_sqrt_coefficients(self, linear_scheduler):
        """Test square root coefficients for forward process."""
        expected_sqrt_alpha = torch.sqrt(linear_scheduler.alphas_cumprod)
        expected_sqrt_one_minus = torch.sqrt(1.0 - linear_scheduler.alphas_cumprod)

        assert torch.allclose(linear_scheduler.sqrt_alphas_cumprod, expected_sqrt_alpha)
        assert torch.allclose(
            linear_scheduler.sqrt_one_minus_alphas_cumprod,
            expected_sqrt_one_minus
        )

    def test_posterior_variance(self, linear_scheduler):
        """Test posterior variance computation."""
        betas = linear_scheduler.betas
        alphas_cumprod = linear_scheduler.alphas_cumprod
        alphas_cumprod_prev = linear_scheduler.alphas_cumprod_prev

        expected_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        assert torch.allclose(linear_scheduler.posterior_variance, expected_variance)

    def test_posterior_mean_coefficients(self, linear_scheduler):
        """Test posterior mean coefficient computation."""
        betas = linear_scheduler.betas
        alphas = linear_scheduler.alphas
        alphas_cumprod = linear_scheduler.alphas_cumprod
        alphas_cumprod_prev = linear_scheduler.alphas_cumprod_prev

        expected_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        expected_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        assert torch.allclose(linear_scheduler.posterior_mean_coef1, expected_coef1)
        assert torch.allclose(linear_scheduler.posterior_mean_coef2, expected_coef2)

    def test_get_index_reshaping(self, linear_scheduler):
        """Test get_index method for proper broadcasting."""
        batch_size = 4
        t_values = torch.randn(batch_size)
        x_shape_2d = torch.Size([batch_size, 3])
        result_2d = linear_scheduler.get_index(t_values, x_shape_2d)
        assert result_2d.shape == (batch_size, 1)
        x_shape_4d = torch.Size([batch_size, 3, 32, 32])
        result_4d = linear_scheduler.get_index(t_values, x_shape_4d)
        assert result_4d.shape == (batch_size, 1, 1, 1)

    def test_device_placement(self, linear_scheduler):
        """Test that all buffers are on the same device."""
        device = linear_scheduler.betas.device
        assert linear_scheduler.alphas.device == device
        assert linear_scheduler.alphas_cumprod.device == device
        assert linear_scheduler.posterior_variance.device == device


class TestForwardDDPM:
    """Test suite for ForwardDDPM class."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler for forward diffusion."""
        return SchedulerDDPM(schedule_type="linear", time_steps=100)

    @pytest.fixture
    def forward_v(self, scheduler):
        """Create forward diffusion with v-prediction."""
        return ForwardDDPM(scheduler, pred_type="v")

    @pytest.fixture
    def forward_x0(self, scheduler):
        """Create forward diffusion with x0-prediction."""
        return ForwardDDPM(scheduler, pred_type="x0")

    @pytest.fixture
    def forward_noise(self, scheduler):
        """Create forward diffusion with noise-prediction."""
        return ForwardDDPM(scheduler, pred_type="noise")

    def test_initialization(self, forward_v):
        """Test ForwardDDPM initialization."""
        assert forward_v.pred_type == "v"
        assert forward_v.vs is not None

    def test_invalid_pred_type(self, scheduler):
        """Test that invalid prediction type raises ValueError."""
        with pytest.raises(ValueError, match="prediction_type must be one of"):
            ForwardDDPM(scheduler, pred_type="invalid")

    def test_forward_output_shapes(self, forward_v):
        """Test that forward pass returns correct shapes."""
        batch_size = 4
        x0 = torch.randn(batch_size, 3, 32, 32)
        t = torch.randint(0, 100, (batch_size,))
        noise = torch.randn_like(x0)
        xt, target = forward_v(x0, t, noise)
        assert xt.shape == x0.shape
        assert target.shape == x0.shape

    def test_forward_noising_formula(self, forward_v, scheduler):
        """Test that forward noising follows correct formula."""
        batch_size = 2
        x0 = torch.randn(batch_size, 3, 16, 16)
        t = torch.randint(0, 100, (batch_size,))
        noise = torch.randn_like(x0)
        xt, _ = forward_v(x0, t, noise)
        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha = scheduler.get_index(sqrt_alpha, x0.shape)
        sqrt_one_minus_alpha = scheduler.get_index(sqrt_one_minus_alpha, x0.shape)
        expected_xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        assert torch.allclose(xt, expected_xt, atol=1e-6)

    def test_target_noise_prediction(self, forward_noise):
        """Test that noise prediction returns noise as target."""
        batch_size = 4
        x0 = torch.randn(batch_size, 3, 32, 32)
        t = torch.randint(0, 100, (batch_size,))
        noise = torch.randn_like(x0)
        _, target = forward_noise(x0, t, noise)
        assert torch.allclose(target, noise)

    def test_target_x0_prediction(self, forward_x0):
        """Test that x0 prediction returns x0 as target."""
        batch_size = 4
        x0 = torch.randn(batch_size, 3, 32, 32)
        t = torch.randint(0, 100, (batch_size,))
        noise = torch.randn_like(x0)
        _, target = forward_x0(x0, t, noise)
        assert torch.allclose(target, x0)

    def test_target_v_prediction(self, forward_v, scheduler):
        """Test that v-prediction follows correct formula."""
        batch_size = 2
        x0 = torch.randn(batch_size, 3, 16, 16)
        t = torch.randint(0, 100, (batch_size,))
        noise = torch.randn_like(x0)
        _, target = forward_v(x0, t, noise)
        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha = scheduler.get_index(sqrt_alpha, x0.shape)
        sqrt_one_minus_alpha = scheduler.get_index(sqrt_one_minus_alpha, x0.shape)
        expected_v = sqrt_alpha * noise - sqrt_one_minus_alpha * x0

        assert torch.allclose(target, expected_v, atol=1e-6)

    def test_deterministic_with_same_noise(self, forward_v):
        """Test that same inputs produce same outputs."""
        x0 = torch.randn(2, 3, 16, 16)
        t = torch.tensor([10, 20])
        noise = torch.randn_like(x0)
        xt1, target1 = forward_v(x0, t, noise)
        xt2, target2 = forward_v(x0, t, noise)
        assert torch.allclose(xt1, xt2)
        assert torch.allclose(target1, target2)

    def test_different_timesteps(self, forward_v):
        """Test that different timesteps produce different results."""
        x0 = torch.randn(1, 3, 16, 16)
        noise = torch.randn_like(x0)
        t_early = torch.tensor([10])
        t_late = torch.tensor([90])
        xt_early, _ = forward_v(x0, t_early, noise)
        xt_late, _ = forward_v(x0, t_late, noise)
        assert not torch.allclose(xt_early, xt_late)


class TestReverseDDPM:
    """Test suite for ReverseDDPM class."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler for reverse diffusion."""
        return SchedulerDDPM(schedule_type="linear", time_steps=100)

    @pytest.fixture
    def reverse_v(self, scheduler):
        """Create reverse diffusion with v-prediction."""
        return ReverseDDPM(scheduler, pred_type="v", var_type="fixed_small")

    @pytest.fixture
    def reverse_noise(self, scheduler):
        """Create reverse diffusion with noise-prediction."""
        return ReverseDDPM(scheduler, pred_type="noise", var_type="fixed_small")

    @pytest.fixture
    def reverse_x0(self, scheduler):
        """Create reverse diffusion with x0-prediction."""
        return ReverseDDPM(scheduler, pred_type="x0", var_type="fixed_small")

    def test_initialization(self, reverse_v):
        """Test ReverseDDPM initialization."""
        assert reverse_v.pred_type == "v"
        assert reverse_v.var_type == "fixed_small"
        assert reverse_v.clip_out is True

    def test_invalid_pred_type(self, scheduler):
        """Test that invalid prediction type raises ValueError."""
        with pytest.raises(ValueError, match="prediction_type must be one of"):
            ReverseDDPM(scheduler, pred_type="invalid")

    def test_invalid_var_type(self, scheduler):
        """Test that invalid variance type raises ValueError."""
        with pytest.raises(ValueError, match="var_type must be one of"):
            ReverseDDPM(scheduler, var_type="invalid")

    def test_predict_x0_from_noise(self, reverse_noise, scheduler):
        """Test x0 prediction from noise prediction."""
        batch_size = 2
        xt = torch.randn(batch_size, 3, 16, 16)
        t = torch.tensor([50, 60])
        noise_pred = torch.randn_like(xt)
        x0 = reverse_noise.predict_x0(xt, t, noise_pred)
        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha = scheduler.get_index(sqrt_alpha, xt.shape)
        sqrt_one_minus_alpha = scheduler.get_index(sqrt_one_minus_alpha, xt.shape)
        expected_x0 = (xt - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        expected_x0 = torch.clamp(expected_x0, -1.0, 1.0)

        assert torch.allclose(x0, expected_x0, atol=1e-6)

    def test_predict_x0_from_x0(self, reverse_x0):
        """Test x0 prediction when directly predicting x0."""
        batch_size = 2
        xt = torch.randn(batch_size, 3, 16, 16)
        t = torch.tensor([50, 60])
        x0_pred = torch.randn_like(xt)
        x0 = reverse_x0.predict_x0(xt, t, x0_pred)
        expected_x0 = torch.clamp(x0_pred, -1.0, 1.0)
        assert torch.allclose(x0, expected_x0)

    def test_predict_x0_from_v(self, reverse_v, scheduler):
        """Test x0 prediction from v-prediction."""
        batch_size = 2
        xt = torch.randn(batch_size, 3, 16, 16)
        t = torch.tensor([50, 60])
        v_pred = torch.randn_like(xt)
        x0 = reverse_v.predict_x0(xt, t, v_pred)
        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha = scheduler.get_index(sqrt_alpha, xt.shape)
        sqrt_one_minus_alpha = scheduler.get_index(sqrt_one_minus_alpha, xt.shape)
        expected_x0 = sqrt_alpha * xt - sqrt_one_minus_alpha * v_pred
        expected_x0 = torch.clamp(expected_x0, -1.0, 1.0)
        assert torch.allclose(x0, expected_x0, atol=1e-6)

    def test_clipping_disabled(self, scheduler):
        """Test that clipping can be disabled."""
        reverse = ReverseDDPM(scheduler, pred_type="x0", clip_out=False)
        xt = torch.randn(2, 3, 16, 16)
        t = torch.tensor([50, 60])
        x0_pred = torch.randn_like(xt) * 5  # Values outside [-1, 1]
        x0 = reverse.predict_x0(xt, t, x0_pred)
        assert torch.allclose(x0, x0_pred)

    def test_variance_fixed_small(self, reverse_v, scheduler):
        """Test fixed_small variance type."""
        t = torch.tensor([50, 60])
        var = reverse_v.get_variance(t)
        expected_var = scheduler.posterior_variance[t]
        assert torch.allclose(var, expected_var)

    def test_variance_fixed_large(self, scheduler):
        """Test fixed_large variance type."""
        reverse = ReverseDDPM(scheduler, var_type="fixed_large")
        t = torch.tensor([50, 60])
        var = reverse.get_variance(t)
        expected_var = scheduler.betas[t]
        assert torch.allclose(var, expected_var)

    def test_variance_learned(self, scheduler):
        """Test learned variance type."""
        reverse = ReverseDDPM(scheduler, var_type="learned")
        t = torch.tensor([50, 60])
        pred_var = torch.tensor([0.0, 0.5])
        var = reverse.get_variance(t, pred_var)
        assert torch.all(var > 0)

    def test_variance_learned_requires_pred_var(self, scheduler):
        """Test that learned variance requires pred_var parameter."""
        reverse = ReverseDDPM(scheduler, var_type="learned")
        t = torch.tensor([50, 60])
        with pytest.raises(ValueError, match="predicted_variance must be provided"):
            reverse.get_variance(t)

    def test_forward_output_shapes(self, reverse_v):
        """Test that forward pass returns correct shapes."""
        batch_size = 4
        xt = torch.randn(batch_size, 3, 32, 32)
        t = torch.randint(1, 100, (batch_size,))
        pred = torch.randn_like(xt)
        x_prev, pred_x0 = reverse_v(xt, pred, t)

        assert x_prev.shape == xt.shape
        assert pred_x0.shape == xt.shape

    def test_no_noise_at_t0(self, reverse_v):
        """Test that no noise is added when t=0."""
        torch.manual_seed(42)
        batch_size = 4
        xt = torch.randn(batch_size, 3, 16, 16)
        t = torch.zeros(batch_size, dtype=torch.long)
        pred = torch.randn_like(xt)
        x_prev1, _ = reverse_v(xt, pred, t)
        torch.manual_seed(123)
        x_prev2, _ = reverse_v(xt, pred, t)

        assert torch.allclose(x_prev1, x_prev2)

    def test_noise_added_at_nonzero_t(self, reverse_v):
        """Test that noise is added when t>0."""
        torch.manual_seed(42)
        batch_size = 4
        xt = torch.randn(batch_size, 3, 16, 16)
        t = torch.ones(batch_size, dtype=torch.long) * 50
        pred = torch.randn_like(xt)
        x_prev1, _ = reverse_v(xt, pred, t)
        torch.manual_seed(123)
        x_prev2, _ = reverse_v(xt, pred, t)

        assert not torch.allclose(x_prev1, x_prev2)

    def test_posterior_mean_computation(self, reverse_v, scheduler):
        """Test posterior mean computation in forward pass."""
        batch_size = 2
        xt = torch.randn(batch_size, 3, 16, 16)
        t = torch.tensor([50, 60])
        pred = torch.randn_like(xt)

        torch.manual_seed(42)
        x_prev, pred_x0 = reverse_v(xt, pred, t)

        coef1 = scheduler.posterior_mean_coef1[t]
        coef2 = scheduler.posterior_mean_coef2[t]
        coef1 = scheduler.get_index(coef1, xt.shape)
        coef2 = scheduler.get_index(coef2, xt.shape)

        expected_mean = coef1 * pred_x0 + coef2 * xt
        assert pred_x0.shape == xt.shape

    def test_integration_all_pred_types(self, scheduler):
        """Test that all prediction types work in forward pass."""
        batch_size = 2
        xt = torch.randn(batch_size, 3, 16, 16)
        t = torch.tensor([50, 60])

        for pred_type in ["noise", "x0", "v"]:
            reverse = ReverseDDPM(scheduler, pred_type=pred_type)
            pred = torch.randn_like(xt)

            x_prev, pred_x0 = reverse(xt, pred, t)

            assert x_prev.shape == xt.shape
            assert pred_x0.shape == xt.shape
            assert torch.all(torch.isfinite(x_prev))
            assert torch.all(torch.isfinite(pred_x0))