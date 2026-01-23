import torch
import pytest
from torchdiff.sde import SchedulerSDE, ForwardSDE, ReverseSDE




class TestSchedulerSDE:
    """Tests for SchedulerSDE"""

    @pytest.fixture
    def linear_scheduler(self):
        return SchedulerSDE(schedule_type="linear", beta_min=0.1, beta_max=20.0)

    @pytest.fixture
    def cosine_scheduler(self):
        return SchedulerSDE(schedule_type="cosine", cosine_s=0.008)

    def test_initialization_valid(self):
        """Test valid initialization"""
        scheduler = SchedulerSDE(schedule_type="linear", beta_min=0.1, beta_max=20.0)
        assert scheduler.schedule_type == "linear"
        assert scheduler.beta_min == 0.1
        assert scheduler.beta_max == 20.0

    def test_initialization_invalid_schedule(self):
        """Test invalid schedule type raises error"""
        with pytest.raises(ValueError, match="schedule_type must be one of"):
            SchedulerSDE(schedule_type="invalid")

    def test_initialization_invalid_beta_range(self):
        """Test invalid beta range raises error"""
        with pytest.raises(ValueError, match="require 0 < beta_min < beta_max"):
            SchedulerSDE(schedule_type="linear", beta_min=20.0, beta_max=0.1)

    def test_beta_monotonic_linear(self, linear_scheduler):
        """Test β(t) is monotonically increasing for linear schedule"""
        t = torch.linspace(0, 1, 100)
        beta_values = linear_scheduler.beta(t)
        assert torch.all(beta_values[1:] >= beta_values[:-1])
        assert torch.isclose(beta_values[0], torch.tensor(0.1), atol=1e-5)
        assert torch.isclose(beta_values[-1], torch.tensor(20.0), atol=1e-5)

    def test_variance_preserving_property(self, linear_scheduler):
        """Test α²(t) + σ²(t) = 1 (variance preserving)"""
        t = torch.linspace(0, 1, 100)
        alpha_sq = linear_scheduler.alpha_squared(t)
        var = linear_scheduler.variance(t)
        sum_vals = alpha_sq + var
        assert torch.allclose(sum_vals, torch.ones_like(sum_vals), atol=1e-5)

    def test_alpha_consistency(self, linear_scheduler):
        """Test α(t) = √(α²(t))"""
        t = torch.linspace(0.1, 1, 100)  # Avoid t=0 for numerical stability
        alpha = linear_scheduler.alpha(t)
        alpha_sq = linear_scheduler.alpha_squared(t)
        assert torch.allclose(alpha ** 2, alpha_sq, atol=1e-5)

    def test_std_consistency(self, linear_scheduler):
        """Test σ(t) = √(σ²(t))"""
        t = torch.linspace(0, 1, 100)
        std = linear_scheduler.std(t)
        var = linear_scheduler.variance(t)
        assert torch.allclose(std ** 2, var, atol=1e-5)

    def test_snr_formula(self, linear_scheduler):
        """Test SNR(t) = α²(t) / σ²(t)"""
        t = torch.linspace(0.1, 0.9, 100)
        snr = linear_scheduler.snr(t)
        alpha_sq = linear_scheduler.alpha_squared(t)
        var = linear_scheduler.variance(t)
        expected_snr = alpha_sq / var
        assert torch.allclose(snr, expected_snr, atol=1e-5)

    def test_snr_decreasing(self, linear_scheduler):
        """Test SNR decreases monotonically (signal degrades over time)"""
        t = torch.linspace(0.01, 0.99, 100)
        snr = linear_scheduler.snr(t)
        assert torch.all(snr[1:] <= snr[:-1])

    def test_boundary_conditions_t0(self, linear_scheduler):
        """Test boundary conditions at t=0"""
        t = torch.tensor([0.0])
        assert torch.isclose(linear_scheduler.alpha(t), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(linear_scheduler.std(t), torch.tensor(0.0), atol=1e-5)

    def test_boundary_conditions_t1(self, linear_scheduler):
        """Test that at t=1, noise dominates (σ²(1) > α²(1))"""
        t = torch.tensor([1.0])
        alpha_sq = linear_scheduler.alpha_squared(t)
        var = linear_scheduler.variance(t)
        assert var > alpha_sq

    def test_cosine_schedule_properties(self, cosine_scheduler):
        """Test cosine schedule satisfies variance preserving"""
        t = torch.linspace(0, 1, 100)
        alpha_sq = cosine_scheduler.alpha_squared(t)
        var = cosine_scheduler.variance(t)
        sum_vals = alpha_sq + var
        assert torch.allclose(sum_vals, torch.ones_like(sum_vals), atol=1e-5)

    def test_batch_handling(self, linear_scheduler):
        """Test scheduler handles batched time inputs"""
        t = torch.rand(32)  # Batch of 32 time values
        beta = linear_scheduler.beta(t)
        alpha = linear_scheduler.alpha(t)
        assert beta.shape == (32,)
        assert alpha.shape == (32,)


class TestForwardSDE:
    """Tests for ForwardSDE"""
    @pytest.fixture
    def scheduler(self):
        return SchedulerSDE(schedule_type="linear", beta_min=0.1, beta_max=20.0)

    @pytest.fixture
    def forward_vp(self, scheduler):
        return ForwardSDE(scheduler, method="vp")

    @pytest.fixture
    def forward_ve(self, scheduler):
        return ForwardSDE(scheduler, method="ve", sigma_min=0.01, sigma_max=50.0)

    @pytest.fixture
    def forward_subvp(self, scheduler):
        return ForwardSDE(scheduler, method="sub-vp")

    @pytest.fixture
    def forward_ode(self, scheduler):
        return ForwardSDE(scheduler, method="ode")

    def test_initialization_valid(self, scheduler):
        """Test valid initialization"""
        forward = ForwardSDE(scheduler, method="vp")
        assert forward.method == "vp"

    def test_initialization_invalid_method(self, scheduler):
        """Test invalid method raises error"""
        with pytest.raises(ValueError, match="sde_method must be one of"):
            ForwardSDE(scheduler, method="invalid")

    def test_output_shapes(self, forward_vp):
        """Test output shapes match input shapes"""
        batch_size = 16
        dim = 64
        x0 = torch.randn(batch_size, dim)
        noise = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)
        xt, score = forward_vp(x0, noise, t)
        assert xt.shape == (batch_size, dim)
        assert score.shape == (batch_size, dim)

    def test_forward_marginal_mean_vp(self, forward_vp, scheduler):
        """Test VP forward marginal has correct mean"""
        batch_size = 1000
        dim = 64
        x0 = torch.randn(batch_size, dim)
        noise = torch.randn(batch_size, dim)
        t = torch.ones(batch_size) * 0.5
        xt, _ = forward_vp(x0, noise, t)
        mean_coeff, _ = forward_vp.get_forward_params(t[:1])
        expected_mean = mean_coeff.item() * x0.mean(dim=0)
        actual_mean = xt.mean(dim=0)
        assert torch.allclose(actual_mean, expected_mean, atol=0.5)

    def test_forward_marginal_variance_vp(self, forward_vp, scheduler):
        """Test VP forward marginal has correct variance"""
        batch_size = 10000
        dim = 1
        x0 = torch.zeros(batch_size, dim)
        noise = torch.randn(batch_size, dim)
        t = torch.ones(batch_size) * 0.5
        xt, _ = forward_vp(x0, noise, t)
        _, std = forward_vp.get_forward_params(t[:1])
        expected_var = std.item() ** 2
        actual_var = xt.var().item()

        assert abs(actual_var - expected_var) < 0.1  # Within 0.1 tolerance

    def test_score_computation(self, forward_vp):
        """Test score = -ε / σ(t)"""
        batch_size = 16
        dim = 64
        x0 = torch.randn(batch_size, dim)
        noise = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)
        _, score = forward_vp(x0, noise, t)
        _, std = forward_vp.get_forward_params(t)
        std = forward_vp._broadcast_to_shape(std, x0.shape)
        expected_score = -noise / (std + forward_vp.eps)
        assert torch.allclose(score, expected_score, atol=1e-6)

    def test_ve_mean_preserved(self, forward_ve):
        """Test VE-SDE preserves mean (mean_coeff = 1)"""
        batch_size = 16
        dim = 64
        x0 = torch.randn(batch_size, dim)
        noise = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)
        mean_coeff, _ = forward_ve.get_forward_params(t)
        assert torch.allclose(mean_coeff, torch.ones_like(mean_coeff))

    def test_ve_variance_grows(self, forward_ve):
        """Test VE-SDE variance grows with time"""
        t = torch.linspace(0.1, 1, 100)
        _, std_vals = forward_ve.get_forward_params(t)
        assert torch.all(std_vals[1:] >= std_vals[:-1])

    def test_subvp_mean_preserved(self, forward_subvp):
        """Test Sub-VP-SDE preserves mean"""
        batch_size = 16
        dim = 64
        x0 = torch.randn(batch_size, dim)
        noise = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)
        mean_coeff, _ = forward_subvp.get_forward_params(t)
        assert torch.allclose(mean_coeff, torch.ones_like(mean_coeff))

    def test_ode_same_marginals_as_vp(self, forward_ode, forward_vp):
        """Test ODE has same marginals as VP-SDE"""
        t = torch.linspace(0, 1, 100)

        mean_ode, std_ode = forward_ode.get_forward_params(t)
        mean_vp, std_vp = forward_vp.get_forward_params(t)
        assert torch.allclose(mean_ode, mean_vp, atol=1e-6)
        assert torch.allclose(std_ode, std_vp, atol=1e-6)

    def test_t0_no_noise(self, forward_vp):
        """Test at t=0, x_t = x_0 (no noise)"""
        batch_size = 16
        dim = 64
        x0 = torch.randn(batch_size, dim)
        noise = torch.randn(batch_size, dim)
        t = torch.zeros(batch_size)
        xt, _ = forward_vp(x0, noise, t)
        assert torch.allclose(xt, x0, atol=1e-5)


class TestReverseSDE:
    """Tests for ReverseSDE"""

    @pytest.fixture
    def scheduler(self):
        return SchedulerSDE(schedule_type="linear", beta_min=0.1, beta_max=20.0)

    @pytest.fixture
    def reverse_vp(self, scheduler):
        return ReverseSDE(scheduler, method="vp")

    @pytest.fixture
    def reverse_ode(self, scheduler):
        return ReverseSDE(scheduler, method="ode")

    def test_initialization_valid(self, scheduler):
        """Test valid initialization"""
        reverse = ReverseSDE(scheduler, method="vp")
        assert reverse.method == "vp"

    def test_negative_dt_required(self, reverse_vp):
        """Test that positive dt raises assertion error"""
        xt = torch.randn(16, 64)
        score = torch.randn(16, 64)
        t = torch.rand(16)
        dt = 0.01

        with pytest.raises(AssertionError, match="dt must be negative"):
            reverse_vp(xt, score, t, dt)

    def test_negative_dt_accepted(self, reverse_vp):
        """Test that negative dt works"""
        xt = torch.randn(16, 64)
        score = torch.randn(16, 64)
        t = torch.rand(16)
        dt = -0.01
        x_prev = reverse_vp(xt, score, t, dt)
        assert x_prev.shape == xt.shape

    def test_output_shape(self, reverse_vp):
        """Test output shape matches input"""
        batch_size = 16
        dim = 64
        xt = torch.randn(batch_size, dim)
        score = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)
        dt = -0.01
        x_prev = reverse_vp(xt, score, t, dt)
        assert x_prev.shape == (batch_size, dim)

    def test_ode_deterministic(self, reverse_ode):
        """Test ODE produces deterministic output (no randomness)"""
        torch.manual_seed(42)
        xt = torch.randn(16, 64)
        score = torch.randn(16, 64)
        t = torch.rand(16)
        dt = -0.01
        x_prev_1 = reverse_ode(xt, score, t, dt)
        x_prev_2 = reverse_ode(xt, score, t, dt)
        assert torch.allclose(x_prev_1, x_prev_2, atol=1e-7)

    def test_sde_stochastic(self, reverse_vp):
        """Test SDE produces stochastic output"""
        torch.manual_seed(42)
        xt = torch.randn(16, 64)
        score = torch.randn(16, 64)
        t = torch.rand(16)
        dt = -0.01
        x_prev_1 = reverse_vp(xt, score, t, dt)
        torch.manual_seed(43)
        x_prev_2 = reverse_vp(xt, score, t, dt)
        assert not torch.allclose(x_prev_1, x_prev_2, atol=1e-5)

    def test_last_step_deterministic(self, reverse_vp):
        """Test last_step=True makes output deterministic"""
        torch.manual_seed(42)
        xt = torch.randn(16, 64)
        score = torch.randn(16, 64)
        t = torch.rand(16)
        dt = -0.01
        x_prev_1 = reverse_vp(xt, score, t, dt, last_step=True)
        x_prev_2 = reverse_vp(xt, score, t, dt, last_step=True)
        assert torch.allclose(x_prev_1, x_prev_2, atol=1e-7)

    def test_vp_drift_coefficients(self, reverse_vp, scheduler):
        """Test VP-SDE drift coefficient is -0.5 * β(t)"""
        t = torch.rand(16)
        drift_coeff, g_squared, diffusion_coeff = reverse_vp.get_reverse_coeffs(t)
        expected_drift = -0.5 * scheduler.beta(t)
        expected_g_sq = scheduler.beta(t)
        assert torch.allclose(drift_coeff, expected_drift, atol=1e-6)
        assert torch.allclose(g_squared, expected_g_sq, atol=1e-6)

    def test_diffusion_coeff_sqrt_relationship(self, reverse_vp):
        """Test diffusion_coeff = √(g²)"""
        t = torch.rand(16)
        drift_coeff, g_squared, diffusion_coeff = reverse_vp.get_reverse_coeffs(t)
        assert torch.allclose(diffusion_coeff ** 2, g_squared, atol=1e-6)

    def test_tensor_dt_conversion(self, reverse_vp):
        """Test that scalar dt is converted to tensor"""
        xt = torch.randn(16, 64)
        score = torch.randn(16, 64)
        t = torch.rand(16)
        dt = -0.01
        x_prev = reverse_vp(xt, score, t, dt)
        assert x_prev.shape == xt.shape


class TestIntegration:
    """Integration tests across multiple components"""
    @pytest.fixture
    def scheduler(self):
        return SchedulerSDE(schedule_type="linear", beta_min=0.1, beta_max=20.0)
    @pytest.fixture
    def forward_vp(self, scheduler):
        return ForwardSDE(scheduler, method="vp")
    @pytest.fixture
    def reverse_vp(self, scheduler):
        return ReverseSDE(scheduler, method="vp")

    def test_forward_backward_consistency(self, forward_vp, reverse_vp):
        """Test that forward then backward (with true score) approximately recovers x0"""
        batch_size = 4
        dim = 32
        torch.manual_seed(42)
        x0 = torch.randn(batch_size, dim)
        noise = torch.randn(batch_size, dim)
        t = torch.ones(batch_size) * 0.1
        xt, true_score = forward_vp(x0, noise, t)
        forward_ode = ForwardSDE(forward_vp.vs, method="ode")
        reverse_ode = ReverseSDE(reverse_vp.vs, method="ode")

        dt = -0.01
        num_steps = int(t[0].item() / abs(dt))
        x_curr = xt.clone()
        t_curr = t.clone()

        for i in range(num_steps):
            mean_coeff, std = forward_ode.get_forward_params(t_curr)
            mean_coeff = forward_ode._broadcast_to_shape(mean_coeff, x0.shape)
            std = forward_ode._broadcast_to_shape(std, x0.shape)
            epsilon = (x_curr - mean_coeff * x0) / std
            score_curr = -epsilon / (std + 1e-8)
            x_curr = reverse_ode(x_curr, score_curr, t_curr, dt, last_step=(i == num_steps - 1))
            t_curr = torch.clamp(t_curr + dt, min=0.0)
        recovery_error = torch.norm(x_curr - x0).item() / torch.norm(x0).item()
        assert recovery_error < 0.3

    def test_ode_deterministic_sampling(self, scheduler):
        """Test ODE sampling is deterministic and can reverse reasonably well"""
        forward_ode = ForwardSDE(scheduler, method="ode")
        reverse_ode = ReverseSDE(scheduler, method="ode")

        batch_size = 8
        dim = 16
        torch.manual_seed(42)
        x0 = torch.randn(batch_size, dim)
        t_forward = torch.ones(batch_size) * 0.2
        noise = torch.randn(batch_size, dim)
        xt, true_score = forward_ode(x0, noise, t_forward)

        dt = -0.01
        num_steps = int(t_forward[0].item() / abs(dt))
        x_curr = xt.clone()
        t_curr = t_forward.clone()

        for i in range(num_steps):
            mean_coeff, std = forward_ode.get_forward_params(t_curr)
            mean_coeff = forward_ode._broadcast_to_shape(mean_coeff, x0.shape)
            std = forward_ode._broadcast_to_shape(std, x0.shape)

            epsilon = (x_curr - mean_coeff * x0) / std
            score_curr = -epsilon / (std + 1e-8)

            x_curr = reverse_ode(x_curr, score_curr, t_curr, dt)
            t_curr = torch.clamp(t_curr + dt, min=0.0)

        recovery_error = torch.norm(x_curr - x0).item() / torch.norm(x0).item()
        assert recovery_error < 1.0

        torch.manual_seed(42)
        x0_2 = torch.randn(batch_size, dim)
        noise_2 = torch.randn(batch_size, dim)
        xt_2, _ = forward_ode(x0_2, noise_2, t_forward)
        x_out_1 = reverse_ode(xt_2, true_score, t_forward, -0.01)
        x_out_2 = reverse_ode(xt_2, true_score, t_forward, -0.01)
        assert torch.allclose(x_out_1, x_out_2, atol=1e-7)

    def test_score_matching_objective(self, forward_vp):
        """Test that the true score satisfies score matching"""
        batch_size = 16
        dim = 64
        x0 = torch.randn(batch_size, dim)
        noise = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)

        xt, true_score = forward_vp(x0, noise, t)
        mean_coeff, std = forward_vp.get_forward_params(t)
        mean_coeff = forward_vp._broadcast_to_shape(mean_coeff, x0.shape)
        std = forward_vp._broadcast_to_shape(std, x0.shape)

        expected_score = -noise / std
        assert torch.allclose(true_score, expected_score, atol=1e-5)

    def test_variance_explosion_growth(self):
        
        scheduler = SchedulerSDE(schedule_type="linear")
        forward_ve = ForwardSDE(scheduler, method="ve", sigma_min=0.01, sigma_max=50.0)

        batch_size = 10000
        dim = 1
        x0 = torch.ones(batch_size, dim) * 5.0  # Non-zero mean
        variances = []
        means = []

        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = torch.ones(batch_size) * t_val
            noise = torch.randn(batch_size, dim)
            noise -= noise.mean(dim=0, keepdim=True)
            xt, _ = forward_ve(x0, noise, t)

            means.append(xt.mean().item())
            variances.append(xt.var().item())

        for mean in means:
            assert abs(mean - 5.0) < 0.2

        for i in range(len(variances) - 1):
            assert variances[i + 1] > variances[i]


    def test_sub_vp_properties(self):
        """Test Sub-VP preserves mean and has variance < 1"""
        scheduler = SchedulerSDE(schedule_type="linear")
        forward_subvp = ForwardSDE(scheduler, method="sub-vp")

        batch_size = 1000
        dim = 8
        x0 = torch.randn(batch_size, dim) * 2.0
        noise = torch.randn(batch_size, dim)
        t = torch.ones(batch_size) * 0.5

        xt, score = forward_subvp(x0, noise, t)

        mean_coeff, std = forward_subvp.get_forward_params(t[:1])
        assert torch.isclose(mean_coeff, torch.tensor(1.0), atol=1e-6)
        assert std.item() < 1.0

    def test_all_methods_valid_scores(self, scheduler):
        """Test all SDE methods produce valid scores (no NaN/Inf)"""
        methods = ["vp", "ve", "sub-vp", "ode"]

        for method in methods:
            if method == "ve":
                forward_sde = ForwardSDE(scheduler, method=method, sigma_min=0.01, sigma_max=50.0)
            else:
                forward_sde = ForwardSDE(scheduler, method=method)

            batch_size = 16
            dim = 32
            x0 = torch.randn(batch_size, dim)
            noise = torch.randn(batch_size, dim)
            t = torch.rand(batch_size) * 0.9 + 0.05
            xt, score = forward_sde(x0, noise, t)
            assert not torch.isnan(xt).any(), f"NaN in xt for method {method}"
            assert not torch.isinf(xt).any(), f"Inf in xt for method {method}"
            assert not torch.isnan(score).any(), f"NaN in score for method {method}"
            assert not torch.isinf(score).any(), f"Inf in score for method {method}"

    def test_reverse_time_direction(self, scheduler):
        """Test that reverse process actually moves backward in time"""
        reverse_vp = ReverseSDE(scheduler, method="vp")

        batch_size = 16
        dim = 32
        xt = torch.randn(batch_size, dim)
        score = torch.randn(batch_size, dim)
        t = torch.ones(batch_size) * 0.8
        dt = -0.1
        x_prev = reverse_vp(xt, score, t, dt)
        assert not torch.allclose(x_prev, xt, atol=1e-5)
        t_new = t + dt
        assert torch.allclose(t_new, torch.ones(batch_size) * 0.7)

    def test_numerical_stability_near_boundaries(self, forward_vp, reverse_vp):
        """Test numerical stability near t=0 and t=1"""
        batch_size = 8
        dim = 16
        x0 = torch.randn(batch_size, dim)
        noise = torch.randn(batch_size, dim)

        t_small = torch.ones(batch_size) * 1e-5
        xt_small, score_small = forward_vp(x0, noise, t_small)
        assert not torch.isnan(xt_small).any()
        assert not torch.isnan(score_small).any()

        t_large = torch.ones(batch_size) * (1.0 - 1e-5)
        xt_large, score_large = forward_vp(x0, noise, t_large)
        assert not torch.isnan(xt_large).any()
        assert not torch.isnan(score_large).any()
        x_prev = reverse_vp(xt_large, score_large, t_large, -0.001)
        assert not torch.isnan(x_prev).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])