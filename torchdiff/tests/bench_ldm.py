"""
Benchmark script to measure the performance of LDM optimizations.

Tests:
1. AutoencoderLDM encode/decode throughput
2. SampleLDM device-direct allocation vs .to(device) for init_samps
3. torch.amp.GradScaler vs deprecated torch.GradScaler (API correctness)
"""

import time
import torch
from torchdiff.ldm import AutoencoderLDM


def make_small_vae(device='cpu'):
    """Create a small KL autoencoder for benchmarking."""
    return AutoencoderLDM(
        in_channels=1,
        down_channels=[16, 32],
        up_channels=[32, 16],
        out_channels=1,
        dropout_rate=0.0,
        num_heads=2,
        num_groups=2,
        num_layers_per_block=1,
        total_down_sampling_factor=2,
        latent_channels=4,
        num_embeddings=64,
        use_vq=False,
        beta=1.0,
        use_flash=False,
        use_grad_check=False,
    ).to(device)


def benchmark_encode_decode(n_iters=200, device='cpu'):
    """Benchmark autoencoder encode + decode throughput."""
    vae = make_small_vae(device)
    vae.eval()
    x = torch.randn(4, 1, 16, 16, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            z, _ = vae.encode(x)
            _ = vae.decode(z)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            z, _ = vae.encode(x)
            _ = vae.decode(z)
    elapsed = time.perf_counter() - start

    throughput = n_iters / elapsed
    print(f"[encode+decode] {elapsed:.4f}s for {n_iters} iters | "
          f"{throughput:.1f} iter/s (device={device})")


def benchmark_randn_allocation(n_iters=10000, device='cpu'):
    """Benchmark torch.randn(..., device=device) vs torch.randn(...).to(device)."""
    shape = (4, 4, 8, 8)

    # Direct allocation (optimized)
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = torch.randn(*shape, device=device)
    direct_time = time.perf_counter() - start

    # .to(device) allocation (old)
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = torch.randn(*shape).to(device)
    to_time = time.perf_counter() - start

    print(f"[randn alloc] .to(): {to_time:.4f}s | direct: {direct_time:.4f}s | "
          f"speedup: {to_time / direct_time:.2f}x ({n_iters} iters, device={device})")


def benchmark_gradscaler_api():
    """Verify torch.amp.GradScaler works correctly with enabled=False on CPU."""
    # This is the new API (non-deprecated)
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


def benchmark_autoencoder_forward(n_iters=200, device='cpu'):
    """Benchmark full autoencoder forward pass (encode + decode + loss)."""
    vae = make_small_vae(device)
    vae.train()
    x = torch.randn(4, 1, 16, 16, device=device)

    # Warmup
    for _ in range(5):
        x_hat, loss, reg_loss, z = vae(x)
        loss.backward()
        vae.zero_grad()

    start = time.perf_counter()
    for _ in range(n_iters):
        x_hat, loss, reg_loss, z = vae(x)
        loss.backward()
        vae.zero_grad()
    elapsed = time.perf_counter() - start

    throughput = n_iters / elapsed
    print(f"[full forward+backward] {elapsed:.4f}s for {n_iters} iters | "
          f"{throughput:.1f} iter/s (device={device})")


if __name__ == "__main__":
    print("=" * 70)
    print("LDM Optimization Benchmarks")
    print("=" * 70)
    benchmark_gradscaler_api()
    benchmark_randn_allocation()
    benchmark_encode_decode()
    benchmark_autoencoder_forward()
    print("=" * 70)
    print("All benchmarks completed.")
