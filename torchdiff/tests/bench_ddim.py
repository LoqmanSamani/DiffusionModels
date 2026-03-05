"""
Benchmark script to measure the performance of DDIM optimizations.

Tests:
1. SchedulerDDIM.get_index() - removed redundant .to(t.device) no-op
2. SampleDDIM init_samps allocation - direct device vs .to(device) copy
3. Full DDIM reverse sampling loop - end-to-end benefit
"""

import time
import torch
from torchdiff.ddim import SchedulerDDIM, ForwardDDIM, ReverseDDIM


def benchmark_get_index(n_iters=50000):
    """Benchmark get_index: current (no .to() no-op) vs simulated old version."""
    scheduler = SchedulerDDIM(schedule_type="linear", train_steps=1000, sample_steps=50)
    batch_size = 64
    t_vals = scheduler.sqrt_alphas_cumprod[torch.randint(0, 1000, (batch_size,))]
    x_shape = torch.Size([batch_size, 3, 32, 32])

    # Current optimized version
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = scheduler.get_index(t_vals, x_shape)
    current_time = time.perf_counter() - start

    # Simulate old version with redundant .to()
    def old_get_index(t, x_shape):
        batch_size = t.shape[0]
        out = t.to(t.device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = old_get_index(t_vals, x_shape)
    old_time = time.perf_counter() - start

    print(f"[get_index] old: {old_time:.4f}s | new: {current_time:.4f}s | "
          f"speedup: {old_time / current_time:.2f}x ({n_iters} iters)")


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


def benchmark_full_sampling_loop(sample_steps=50, n_runs=3):
    """Benchmark a full DDIM reverse sampling loop."""
    scheduler = SchedulerDDIM(schedule_type="linear", train_steps=1000, sample_steps=sample_steps)
    reverse = ReverseDDIM(scheduler, pred_type="noise", eta=0.0, clip_=True)
    batch_size = 4
    channels = 3
    img_size = 32

    class DummyModel(torch.nn.Module):
        def forward(self, x, t, *args, **kwargs):
            return x

    model = DummyModel()

    def run_ddim_sampling(reverse_module, model):
        xt = torch.randn(batch_size, channels, img_size, img_size)
        timesteps = reverse_module.vs.inference_timesteps.flip(0)
        with torch.no_grad():
            for i in range(len(timesteps) - 1):
                t_current = timesteps[i].item()
                t_prev = timesteps[i + 1].item()
                t = torch.full((batch_size,), t_current, dtype=torch.long)
                t_p = torch.full((batch_size,), t_prev, dtype=torch.long)
                pred = model(xt, t)
                xt, _ = reverse_module(xt, t, t_p, pred)
        return xt

    # Warmup
    run_ddim_sampling(reverse, model)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_ddim_sampling(reverse, model)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)

    print(f"[ddim_sampling steps={sample_steps}] avg: {avg:.4f}s (over {n_runs} runs)")


def benchmark_eta0_vs_eta1(sample_steps=50, n_runs=3):
    """Compare deterministic (eta=0) vs stochastic (eta=1) DDIM sampling."""
    scheduler = SchedulerDDIM(schedule_type="linear", train_steps=1000, sample_steps=sample_steps)
    batch_size = 4
    channels = 3
    img_size = 32

    class DummyModel(torch.nn.Module):
        def forward(self, x, t, *args, **kwargs):
            return x

    model = DummyModel()

    for eta in [0.0, 1.0]:
        reverse = ReverseDDIM(scheduler, pred_type="noise", eta=eta, clip_=True)

        def run_sampling():
            xt = torch.randn(batch_size, channels, img_size, img_size)
            timesteps = reverse.vs.inference_timesteps.flip(0)
            with torch.no_grad():
                for i in range(len(timesteps) - 1):
                    t = torch.full((batch_size,), timesteps[i].item(), dtype=torch.long)
                    t_p = torch.full((batch_size,), timesteps[i + 1].item(), dtype=torch.long)
                    pred = model(xt, t)
                    xt, _ = reverse(xt, t, t_p, pred)
            return xt

        run_sampling()  # warmup
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            run_sampling()
            times.append(time.perf_counter() - start)
        avg = sum(times) / len(times)
        print(f"[ddim eta={eta}] avg: {avg:.4f}s (over {n_runs} runs, {sample_steps} steps)")


if __name__ == "__main__":
    print("=" * 70)
    print("DDIM Optimization Benchmarks")
    print("=" * 70)
    benchmark_get_index()
    benchmark_tensor_allocation()
    benchmark_full_sampling_loop(sample_steps=50)
    benchmark_full_sampling_loop(sample_steps=100)
    benchmark_eta0_vs_eta1(sample_steps=50)
    print("=" * 70)
    print("All benchmarks completed.")
