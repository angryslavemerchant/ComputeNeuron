"""Speed benchmark: DendriticLinear vs the FFN blocks it's meant to replace.

Compares, at matched d_model:
  * nn.Linear(d, d)          — single-layer baseline
  * FFN d -> 4d -> d (GELU)  — the 2-layer block DendriticLinear claims to replace
  * FFN d -> d -> d  (GELU)  — smaller 2-layer block
  * DendriticLinear(d, d) at several (fan_in_ratio, coverage) settings

Reports parameter count, forward latency, forward+backward latency, and peak
allocated memory (CUDA only).

Usage:  python bench_dendritic.py [--device cuda|cpu] [--dtype fp32|fp16]
"""

import argparse
import time

import torch
import torch.nn as nn

from dendritic_linear import DendriticLinear


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, warmup=5, min_seconds=0.4, max_iters=2000):
    for _ in range(warmup):
        fn()
    sync(device)

    t0 = time.perf_counter()
    fn()
    sync(device)
    single = time.perf_counter() - t0

    n = max(10, int(min_seconds / max(single, 1e-6)))
    n = min(n, max_iters)

    sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / n * 1e3  # ms


def make_ffn(d, hidden):
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))


class GatherRef(nn.Module):
    """Wraps DendriticLinear to run the old gather-based forward, so the
    einsum speedup is visible as a direct A/B."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner.forward_reference(x)


def bench(model, x, device, label, extra=""):
    try:
        return _bench(model, x, device, label, extra)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  {label:<34} {'':>10} {'OOM':>9} {'OOM':>10} {'':>9}  {extra}")
        return float("nan"), float("nan")


def _bench(model, x, device, label, extra=""):
    model = model.to(device=device, dtype=x.dtype)
    params = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        fwd = timeit(lambda: model(x), device)

    xg = x.clone().requires_grad_(True)

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        model(xg).sum().backward()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    fb = timeit(fwd_bwd, device)
    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else float("nan")

    print(f"  {label:<34} {params/1e3:>9.1f}K {fwd:>9.3f} {fb:>10.3f} {peak:>9.0f}  {extra}")
    return fwd, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--ref", action="store_true",
                    help="also time the old gather forward (slow, memory-hungry)")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    torch.manual_seed(0)
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"torch {torch.__version__} | device={device} ({name}) | dtype={args.dtype}\n")

    configs = [
        (256, 256),
        (256, 512),
        (1024, 512),
        (1024, 1024),
        (4096, 512),
    ]

    for B, d in configs:
        x = torch.randn(B, d, device=device, dtype=dtype)
        print(f"=== tokens={B}, d_model={d} ===")
        print(f"  {'module':<34} {'params':>10} {'fwd(ms)':>9} {'fwd+bwd':>10} {'peakMB':>9}")

        base_fwd, base_fb = bench(nn.Linear(d, d), x, device, "nn.Linear(d, d)")
        bench(make_ffn(d, 4 * d), x, device, "FFN d->4d->d")
        bench(make_ffn(d, d), x, device, "FFN d->d->d")

        # fan_in: float -> fraction of in_features, int -> absolute feature count
        for fan_in, cov in [(0.1, 1.0), (0.25, 1.0), (0.1, 2.0), (0.5, 1.0), (16, 1.0), (64, 1.0)]:
            m = DendriticLinear(d, d, fan_in=fan_in, coverage=cov)
            K, D = m.K, m.D
            f, fb = bench(
                m, x, device, f"Dendritic(fan_in={fan_in}, cov={cov})", extra=f"K={K} D={D}"
            )
            print(
                f"  {'':<34} {'':>10} "
                f"{f/base_fwd:>8.1f}x {fb/base_fb:>9.1f}x            vs nn.Linear"
            )
            if args.ref:
                bench(GatherRef(m), x, device, "  ^ old gather forward", extra=f"K={K} D={D}")
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
