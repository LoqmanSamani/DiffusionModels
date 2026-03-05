"""
Benchmark script to measure the performance of SDE optimizations.

Tests:
1. Full SDE reverse sampling loop (VP, VE, ODE methods)
2. Tensor allocation on device vs .to(device)
3. Forward SDE marginal computation
"""

import time
import torch
from torchdiff.sde import SchedulerSDE, ForwardSDE, ReverseSDE


def benchmark_tensor_allocation(n_iters=5000):
    """Benchmark torch.randn(..., device=dev) vs torch.randn(...).to(dev)."""
    device = torch.device('cpu')
    batch_size, channels, h, w = 4, 3, 32, 32

    # New: direct allocation
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = torch.randn(batch_size, channels, h, w, device=device)
    new_time = time.perf_counter() - start

    # Old: allocate then .to()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = torch.randn(batch_size, channels, h, w).to(device)
    old_time = time.perf_counter() - start

    print(f"[tensor_alloc] old: {old_time:.4f}s | new: {new_time:.4f}s | "
          f"speedup: {old_time / new_time:.2f}x ({n_iters} iters)")


def benchmark_forward_sde(n_iters=5000):
    """Benchmark forward SDE marginal computation across methods."""
    scheduler = SchedulerSDE(schedule_type="linear", beta_min=0.1, beta_max=20.0)
    batch_size = 64
    dim = 32

    for method in ["vp", "ve", "sub-vp", "ode"]:
        kwargs = {}
        if method == "ve":
            kwargs = {"sigma_min": 0.01, "sigma_max": 50.0}
        forward = ForwardSDE(scheduler, method=method, **kwargs)

        x0 = torch.randn(batch_size, dim)
        noise = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)

        # Warmup
        for _ in range(10):
            forward(x0, t, noise)

        start = time.perf_counter()
        for _ in range(n_iters):
            forward(x0, t, noise)
        elapsed = time.perf_counter() - start

        print(f"[forward_{method}] {elapsed:.4f}s ({n_iters} iters, "
              f"{n_iters / elapsed:.0f} it/s)")


def benchmark_reverse_step(n_iters=5000):
    """Benchmark a single reverse SDE step across methods."""
    scheduler = SchedulerSDE(schedule_type="linear", beta_min=0.1, beta_max=20.0)
    batch_size = 64
    dim = 32

    for method in ["vp", "ve", "sub-vp", "ode"]:
        kwargs = {}
        if method == "ve":
            kwargs = {"sigma_min": 0.01, "sigma_max": 50.0}
        reverse = ReverseSDE(scheduler, method=method, **kwargs)

        xt = torch.randn(batch_size, dim)
        pred = torch.randn(batch_size, dim)
        t = torch.ones(batch_size) * 0.5
        dt = -0.01

        # Warmup
        for _ in range(10):
            reverse(xt, pred, t, dt)

        start = time.perf_counter()
        for _ in range(n_iters):
            reverse(xt, pred, t, dt)
        elapsed = time.perf_counter() - start

        print(f"[reverse_{method}] {elapsed:.4f}s ({n_iters} iters, "
              f"{n_iters / elapsed:.0f} it/s)")


def benchmark_full_sampling_loop(num_steps=100, n_runs=3):
    """Benchmark a full SDE reverse sampling loop."""
    scheduler = SchedulerSDE(schedule_type="linear", beta_min=0.1, beta_max=20.0)
    batch_size = 4
    channels = 3
    img_size = 32
    time_eps = 1e-5

    class DummyModel(torch.nn.Module):
        def forward(self, x, t, *args, **kwargs):
            return x

    model = DummyModel()

    for method in ["vp", "ode"]:
        reverse = ReverseSDE(scheduler, method=method)

        def run_sampling():
            xt = torch.randn(batch_size, channels, img_size, img_size)
            t_schedule = torch.linspace(1.0, time_eps, num_steps + 1)
            dt = -(1.0 - time_eps) / num_steps
            with torch.no_grad():
                for step in range(num_steps):
                    t_current = float(t_schedule[step])
                    t_batch = torch.full((batch_size,), t_current, dtype=xt.dtype)
                    pred = model(xt, t_batch)
                    last_step = (step == num_steps - 1)
                    xt = reverse(xt, pred, t_batch, dt, last_step=last_step)
            return xt

        # Warmup
        run_sampling()

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            run_sampling()
            times.append(time.perf_counter() - start)
        avg = sum(times) / len(times)

        print(f"[sde_sampling method={method} steps={num_steps}] "
              f"avg: {avg:.4f}s (over {n_runs} runs)")


def benchmark_schedule_comparison(n_iters=5000):
    """Compare linear vs cosine schedule performance."""
    batch_size = 64
    t = torch.rand(batch_size)

    for sched_type in ["linear", "cosine"]:
        scheduler = SchedulerSDE(schedule_type=sched_type)

        # Warmup
        for _ in range(10):
            scheduler.beta(t)
            scheduler.alpha(t)
            scheduler.std(t)

        start = time.perf_counter()
        for _ in range(n_iters):
            scheduler.beta(t)
            scheduler.alpha(t)
            scheduler.std(t)
        elapsed = time.perf_counter() - start

        print(f"[schedule_{sched_type}] {elapsed:.4f}s ({n_iters} iters, "
              f"{n_iters / elapsed:.0f} it/s)")


if __name__ == "__main__":
    print("=" * 70)
    print("SDE Optimization Benchmarks")
    print("=" * 70)
    benchmark_tensor_allocation()
    benchmark_schedule_comparison()
    benchmark_forward_sde()
    benchmark_reverse_step()
    benchmark_full_sampling_loop(num_steps=100)
    benchmark_full_sampling_loop(num_steps=400)
    print("=" * 70)
    print("All benchmarks completed.")
