"""
Benchmark script to measure the performance of DDPM optimizations.

Tests:
1. SchedulerDDPM.get_index() - removed redundant .to(t.device) no-op
2. ReverseDDPM.forward() - precomputed sqrt_posterior_variance instead of torch.sqrt() per step
3. Full reverse sampling loop - end-to-end benefit of precomputed buffers
"""

import time
import torch
from torchdiff.ddpm import SchedulerDDPM, ForwardDDPM, ReverseDDPM


def benchmark_get_index(n_iters=50000):
    """Benchmark get_index: current (no .to() no-op) vs simulated old version."""
    scheduler = SchedulerDDPM(schedule_type="linear", time_steps=1000)
    batch_size = 64
    t_vals = scheduler.sqrt_alphas_cumprod[torch.randint(0, 1000, (batch_size,))]
    x_shape = torch.Size([batch_size, 3, 32, 32])

    # Current optimized version (no .to(t.device))
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = scheduler.get_index(t_vals, x_shape)
    current_time = time.perf_counter() - start

    # Simulate old version with redundant .to()
    def old_get_index(t, x_shape):
        batch_size = t.shape[0]
        out = t.to(t.device)  # no-op copy
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = old_get_index(t_vals, x_shape)
    old_time = time.perf_counter() - start

    print(f"[get_index] old: {old_time:.4f}s | new: {current_time:.4f}s | "
          f"speedup: {old_time / current_time:.2f}x ({n_iters} iters)")


def benchmark_reverse_step(n_iters=5000):
    """Benchmark reverse step: precomputed sqrt vs torch.sqrt() per call."""
    scheduler = SchedulerDDPM(schedule_type="linear", time_steps=1000)
    reverse = ReverseDDPM(scheduler, pred_type="noise", var_type="fixed_small")
    batch_size = 16
    xt = torch.randn(batch_size, 3, 32, 32)
    pred = torch.randn_like(xt)
    t = torch.randint(1, 1000, (batch_size,))

    # Current optimized: uses precomputed sqrt_posterior_variance
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = reverse(xt, pred, t)
    current_time = time.perf_counter() - start

    # Simulate old version: compute sqrt(variance) every call
    def old_reverse_step(reverse, xt, pred, t):
        pred_x0 = reverse.predict_x0(xt, t, pred)
        coef1 = reverse.vs.posterior_mean_coef1[t]
        coef2 = reverse.vs.posterior_mean_coef2[t]
        coef1 = reverse.vs.get_index(coef1, xt.shape)
        coef2 = reverse.vs.get_index(coef2, xt.shape)
        posterior_mean = coef1 * pred_x0 + coef2 * xt
        variance = reverse.vs.posterior_variance[t]
        variance = reverse.vs.get_index(variance, xt.shape)
        noise = torch.randn_like(xt)
        mask = (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        x_prev = posterior_mean + mask * torch.sqrt(variance) * noise  # sqrt each call
        return x_prev, pred_x0

    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = old_reverse_step(reverse, xt, pred, t)
    old_time = time.perf_counter() - start

    print(f"[reverse_step] old: {old_time:.4f}s | new: {current_time:.4f}s | "
          f"speedup: {old_time / current_time:.2f}x ({n_iters} iters)")


def benchmark_full_sampling_loop(time_steps=200, n_runs=3):
    """Benchmark a full reverse sampling loop (end-to-end)."""
    scheduler = SchedulerDDPM(schedule_type="linear", time_steps=time_steps)
    reverse = ReverseDDPM(scheduler, pred_type="noise", var_type="fixed_small")
    batch_size = 4
    channels = 3
    img_size = 32

    # Simple dummy "model" that returns input (identity)
    class DummyModel(torch.nn.Module):
        def forward(self, x, t, *args, **kwargs):
            return x

    model = DummyModel()

    def run_sampling(reverse_module, model):
        xt = torch.randn(batch_size, channels, img_size, img_size)
        with torch.no_grad():
            for step in reversed(range(time_steps)):
                t = torch.full((batch_size,), step, dtype=torch.long)
                pred = model(xt, t)
                xt, _ = reverse_module(xt, pred, t)
        return xt

    # Warmup
    run_sampling(reverse, model)

    # Current optimized version
    times_new = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_sampling(reverse, model)
        times_new.append(time.perf_counter() - start)
    avg_new = sum(times_new) / len(times_new)

    # Old version: simulate without precomputed sqrt
    def old_run_sampling(scheduler, model):
        xt = torch.randn(batch_size, channels, img_size, img_size)
        with torch.no_grad():
            for step in reversed(range(time_steps)):
                t = torch.full((batch_size,), step, dtype=torch.long)
                pred = model(xt, t)
                # Old predict_x0
                sqrt_alpha = scheduler.sqrt_alphas_cumprod[t]
                sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[t]
                sqrt_alpha = scheduler.get_index(sqrt_alpha, xt.shape)
                sqrt_one_minus = scheduler.get_index(sqrt_one_minus, xt.shape)
                pred_x0 = (xt - sqrt_one_minus * pred) / sqrt_alpha
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                coef1 = scheduler.posterior_mean_coef1[t]
                coef2 = scheduler.posterior_mean_coef2[t]
                coef1_r = scheduler.get_index(coef1, xt.shape)
                coef2_r = scheduler.get_index(coef2, xt.shape)
                posterior_mean = coef1_r * pred_x0 + coef2_r * xt
                variance = scheduler.posterior_variance[t]
                variance = scheduler.get_index(variance, xt.shape)
                noise = torch.randn_like(xt)
                mask = (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
                xt = posterior_mean + mask * torch.sqrt(variance) * noise
        return xt

    times_old = []
    for _ in range(n_runs):
        start = time.perf_counter()
        old_run_sampling(scheduler, model)
        times_old.append(time.perf_counter() - start)
    avg_old = sum(times_old) / len(times_old)

    print(f"[full_sampling T={time_steps}] old: {avg_old:.4f}s | new: {avg_new:.4f}s | "
          f"speedup: {avg_old / avg_new:.2f}x (avg of {n_runs} runs)")


def benchmark_precomputed_buffers():
    """Verify precomputed buffers are correct and measure scheduler init cost."""
    scheduler = SchedulerDDPM(schedule_type="linear", time_steps=1000)

    # Verify sqrt_posterior_variance
    expected = torch.sqrt(torch.clamp(scheduler.posterior_variance, min=1e-20))
    assert torch.allclose(scheduler.sqrt_posterior_variance, expected), "sqrt_posterior_variance mismatch!"

    # Verify sqrt_betas
    expected_sqrt_betas = torch.sqrt(scheduler.betas)
    assert torch.allclose(scheduler.sqrt_betas, expected_sqrt_betas), "sqrt_betas mismatch!"

    print("[buffers] sqrt_posterior_variance and sqrt_betas verified correct")


if __name__ == "__main__":
    print("=" * 70)
    print("DDPM Optimization Benchmarks")
    print("=" * 70)
    benchmark_precomputed_buffers()
    benchmark_get_index()
    benchmark_reverse_step()
    benchmark_full_sampling_loop(time_steps=200)
    benchmark_full_sampling_loop(time_steps=500)
    print("=" * 70)
    print("All benchmarks completed.")
