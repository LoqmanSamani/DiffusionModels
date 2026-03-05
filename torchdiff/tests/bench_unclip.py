"""
Benchmark script to measure the performance of UnCLIP optimizations.

Tests:
1. SchedulerUnCLIP.get_index: direct return vs .to(t.device) no-op
2. ForwardUnCLIP + ReverseUnCLIP throughput (2D embeddings and 4D images)
3. UnCLIPTransformerPrior forward throughput
4. torch.amp.GradScaler API correctness on CPU
"""

import time
import torch
from torchdiff.unclip import (
    SchedulerUnCLIP, ForwardUnCLIP, ReverseUnCLIP,
    UnCLIPTransformerPrior, CLIPContextProjection, CLIPEmbeddingProjection,
)


def benchmark_get_index(n_iters=50000):
    """Benchmark SchedulerUnCLIP.get_index with optimized direct return."""
    sched = SchedulerUnCLIP(schedule_type="linear", train_steps=1000)
    t = torch.randint(0, 1000, (8,))

    # Warmup
    for _ in range(100):
        _ = sched.get_index(t, torch.Size([8, 512]))

    # Optimized: direct return (no .to(t.device))
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = sched.get_index(t, torch.Size([8, 512]))
    direct_time = time.perf_counter() - start

    # Simulated old: with .to(t.device) no-op
    start = time.perf_counter()
    for _ in range(n_iters):
        out = t.to(t.device)
        _ = out.reshape(8, 1)
    old_time = time.perf_counter() - start

    print(f"[get_index] direct: {direct_time:.4f}s | .to(t.device): {old_time:.4f}s | "
          f"speedup: {old_time / direct_time:.2f}x ({n_iters} iters)")


def benchmark_forward_reverse_2d(n_iters=500):
    """Benchmark ForwardUnCLIP + ReverseUnCLIP on 2D embeddings."""
    sched = SchedulerUnCLIP(schedule_type="cosine", train_steps=100, sample_steps=20)
    fwd = ForwardUnCLIP(sched, pred_type="noise")
    rwd = ReverseUnCLIP(sched, pred_type="noise")

    batch_size = 8
    embed_dim = 512
    x = torch.randn(batch_size, embed_dim)
    noise = torch.randn_like(x)
    t_fwd = torch.randint(0, 100, (batch_size,))

    # Warmup
    for _ in range(10):
        x_noisy, target = fwd(x, noise, t_fwd)

    start = time.perf_counter()
    for _ in range(n_iters):
        x_noisy, target = fwd(x, noise, t_fwd)
    fwd_time = time.perf_counter() - start

    # Reverse step (tau indices: [0, sample_steps-1])
    t_rev = torch.randint(1, 20, (batch_size,))
    t_rev_prev = t_rev - 1
    model_pred = torch.randn(batch_size, embed_dim)
    for _ in range(10):
        _ = rwd(x_noisy, t_rev, t_rev_prev, model_pred)

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = rwd(x_noisy, t_rev, t_rev_prev, model_pred)
    rwd_time = time.perf_counter() - start

    print(f"[fwd+rwd 2D] forward: {fwd_time:.4f}s | reverse: {rwd_time:.4f}s | "
          f"total: {fwd_time + rwd_time:.4f}s ({n_iters} iters, shape={x.shape})")


def benchmark_forward_reverse_4d(n_iters=200):
    """Benchmark ForwardUnCLIP + ReverseUnCLIP on 4D images."""
    sched = SchedulerUnCLIP(schedule_type="linear", train_steps=100, sample_steps=20)
    fwd = ForwardUnCLIP(sched, pred_type="noise")
    rwd = ReverseUnCLIP(sched, pred_type="noise")

    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    noise = torch.randn_like(x)
    t_fwd = torch.randint(0, 100, (batch_size,))

    # Warmup
    for _ in range(10):
        x_noisy, target = fwd(x, noise, t_fwd)

    start = time.perf_counter()
    for _ in range(n_iters):
        x_noisy, target = fwd(x, noise, t_fwd)
    fwd_time = time.perf_counter() - start

    t_rev = torch.randint(1, 20, (batch_size,))
    t_rev_prev = t_rev - 1
    model_pred = torch.randn_like(x)
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = rwd(x_noisy, t_rev, t_rev_prev, model_pred)
    rwd_time = time.perf_counter() - start

    print(f"[fwd+rwd 4D] forward: {fwd_time:.4f}s | reverse: {rwd_time:.4f}s | "
          f"total: {fwd_time + rwd_time:.4f}s ({n_iters} iters, shape={x.shape})")


def benchmark_transformer_prior(n_iters=100):
    """Benchmark UnCLIPTransformerPrior forward pass."""
    fwd = ForwardUnCLIP(
        SchedulerUnCLIP(schedule_type="cosine", train_steps=50),
        pred_type="noise"
    )
    rwd = ReverseUnCLIP(
        SchedulerUnCLIP(schedule_type="cosine", train_steps=50),
        pred_type="noise"
    )
    prior = UnCLIPTransformerPrior(
        fwd_unclip=fwd,
        rwd_unclip=rwd,
        trans_embed_dim=64,
        num_att_heads=4,
        num_layers=2,
        use_flash=False,
        grad_check=False,
    )
    prior.eval()

    batch_size = 4
    text_embed = torch.randn(batch_size, 64)
    noisy_embed = torch.randn(batch_size, 64)
    t = torch.randint(0, 50, (batch_size,))

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = prior(text_embed, noisy_embed, t)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = prior(text_embed, noisy_embed, t)
    elapsed = time.perf_counter() - start

    throughput = n_iters / elapsed
    print(f"[TransformerPrior] {elapsed:.4f}s for {n_iters} iters | "
          f"{throughput:.1f} iter/s (embed_dim=64, layers=2)")


def benchmark_gradscaler_api():
    """Verify torch.amp.GradScaler works correctly with enabled=False on CPU."""
    scaler = torch.amp.GradScaler(enabled=False)

    model = torch.nn.Linear(10, 10)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 10)

    with torch.autocast(device_type='cpu', enabled=False):
        y = model(x)
        loss = y.sum()

    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)

    print("[GradScaler API] torch.amp.GradScaler(enabled=False) works correctly on CPU")


def benchmark_projection_roundtrip(n_iters=1000):
    """Benchmark CLIPEmbeddingProjection forward + inverse."""
    proj = CLIPEmbeddingProjection(clip_embed_dim=512, trans_embed_dim=64)
    proj.eval()
    x = torch.randn(8, 512)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            z = proj(x)
            _ = proj.inverse_transform(z)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            z = proj(x)
            _ = proj.inverse_transform(z)
    elapsed = time.perf_counter() - start

    print(f"[EmbedProjection] {elapsed:.4f}s for {n_iters} roundtrips | "
          f"{n_iters / elapsed:.1f} iter/s (512->64->512)")


if __name__ == "__main__":
    print("=" * 70)
    print("UnCLIP Optimization Benchmarks")
    print("=" * 70)
    benchmark_gradscaler_api()
    benchmark_get_index()
    benchmark_forward_reverse_2d()
    benchmark_forward_reverse_4d()
    benchmark_transformer_prior()
    benchmark_projection_roundtrip()
    print("=" * 70)
    print("All benchmarks completed.")
