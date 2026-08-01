"""Shape-matched benchmark: dense vs dendritic.

Both models have the identical graph shape --

    N inputs  ->  M hidden units  ->  (nonlinearity)  ->  S outputs

The ONLY difference is connectivity:

    dense       every hidden unit reads all N inputs;
                every output reads all M hidden units.
    dendritic   every dendrite reads K inputs;
                every soma reads its own D = M/S dendrites.

Same layer sizes, same nonlinearity, same everything else — so any difference
in wall clock is attributable to the sparsity and nothing else.

Batch size matters more than anything else here. At small batches every model
is bound by kernel launch latency and the results say nothing; the dendritic
layer's arithmetic advantage only shows up once the GPU is saturated.

Usage:
    python bench.py                    # cuda if available
    python bench.py --no-compile       # skip the compiled rows
    python bench.py --fan-in 32        # inputs per dendrite
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_linear import DendriticLinear

# Dendritic layers are launch-bound in eager mode and recompile on grad-mode
# flips; without this the later configs silently fall back to eager.
torch._dynamo.config.recompile_limit = 64


class DenseMLP(nn.Module):
    """N -> M -> S, fully connected, same nonlinearity as the dendritic model."""

    def __init__(self, n, m, s):
        super().__init__()
        self.fc1 = nn.Linear(n, m)
        self.fc2 = nn.Linear(m, s)

    def forward(self, x):
        return self.fc2(F.leaky_relu(self.fc1(x), 0.1))

    @staticmethod
    def macs(n, m, s):
        return n * m + m * s


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, warmup=8, min_seconds=0.3, max_iters=2000):
    for _ in range(warmup):
        fn()
    sync(device)

    t0 = time.perf_counter()
    fn()
    sync(device)
    single = time.perf_counter() - t0

    n = min(max(10, int(min_seconds / max(single, 1e-6))), max_iters)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / n * 1e3  # ms


def measure(model, x, device, label, macs, base=None):
    params = sum(p.numel() for p in model.parameters())

    try:
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
        peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  {label:<26} {params/1e3:>8.1f}K {macs/1e3:>9.1f}K {'OOM':>8} {'OOM':>9}")
        return None

    ratio = "" if base is None else f"{fwd/base[0]:>7.2f}x {fb/base[1]:>8.2f}x"
    print(f"  {label:<26} {params/1e3:>8.1f}K {macs/1e3:>9.1f}K "
          f"{fwd:>8.3f} {fb:>9.3f} {peak:>8.0f}  {ratio}")
    return fwd, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fan-in", type=int, default=16, help="inputs per dendrite (K)")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"torch {torch.__version__} | {device} ({name}) | K={args.fan_in}\n")

    # (batch, N inputs, M dendrites/hidden, S soma/outputs)
    configs = [
        (1024, 1024, 1024, 256),
        (1024, 1024, 512, 128),      # fewer dendrites -> narrower dense hidden too
        (16384, 1024, 1024, 256),    # big batch: throughput, not launch latency
        (16384, 2048, 2048, 512),
    ]

    for B, N, M, S in configs:
        D = M // S
        x = torch.randn(B, N, device=device)

        print(f"=== batch={B} | {N} inputs -> {M} hidden -> {S} out | K={args.fan_in}, D={D} ===")
        print(f"  {'module':<26} {'params':>9} {'MACs/vec':>10} "
              f"{'fwd(ms)':>8} {'fwd+bwd':>9} {'peakMB':>8}   vs dense")

        base = measure(DenseMLP(N, M, S).to(device), x, device,
                       f"dense {N}->{M}->{S}", DenseMLP.macs(N, M, S))
        if base is None:
            continue

        m = DendriticLinear(N, S, fan_in=args.fan_in,
                            dendrites_per_soma=D).to(device)
        macs = m.sparsity()["macs"]
        measure(m, x, device, "dendritic", macs, base)

        if not args.no_compile:
            torch._dynamo.reset()
            measure(torch.compile(m), x, device, "  ^ compiled", macs, base)
            torch._dynamo.reset()

        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
