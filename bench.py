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

from dendritic_linear import DendriticLinear, DendriticMLP

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


class DenseMLP3(nn.Module):
    """N -> M -> S -> out, fully connected. Shape-matched to DendriticMLP."""

    def __init__(self, n, m, s, out):
        super().__init__()
        self.fc1 = nn.Linear(n, m)
        self.fc2 = nn.Linear(m, s)
        self.fc3 = nn.Linear(s, out)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x), 0.1)
        h = F.leaky_relu(self.fc2(h), 0.1)
        return self.fc3(h)

    @staticmethod
    def macs(n, m, s, out):
        return n * m + m * s + s * out


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


def measure(model, x, device, label, macs, base=None, pad=26):
    params = sum(p.numel() for p in model.parameters())
    tag = f"{label:<{pad}}" if pad else label

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
        print(f"  {tag} {params/1e3:>8.1f}K {macs/1e3:>9.1f}K {'OOM':>8} {'OOM':>9}")
        return None

    ratio = "" if base is None else f"{fwd/base[0]:>7.2f}x {fb/base[1]:>8.2f}x"
    print(f"  {tag} {params/1e3:>8.1f}K {macs/1e3:>9.1f}K "
          f"{fwd:>8.3f} {fb:>9.3f} {peak:>8.0f}  {ratio}")
    return fwd, fb


def crossover(rows):
    """Where does the dendritic/dense ratio cross 1.0?

    rows is [(D, coverage, ratio), ...] in increasing D. Returns an
    interpolated (D, coverage) or None if it never crosses in range.
    """
    for (d0, c0, r0), (d1, c1, r1) in zip(rows, rows[1:]):
        if r0 <= 1.0 <= r1:
            t = (1.0 - r0) / (r1 - r0) if r1 != r0 else 0.0
            return d0 + t * (d1 - d0), c0 + t * (c1 - c0)
    return None


def sweep(args, device):
    """Grow the dendrite count until the layer costs as much as a dense one.

    The dense baseline is FIXED at N -> N -> S. Only the dendritic layer grows
    with D, so there is a real crossing point: the number of dendrites per soma
    you can afford before you may as well have been dense.

    Note that D * K is how many inputs each soma sees, so D = N/K (coverage
    1.0) is the point where every soma reads the entire input.

    With --readout both sides gain the extra fully connected layer, so the
    comparison is DendriticMLP (N -> M -> S -> S) against a dense N -> N -> S
    -> S. The readout is identical in both models, which flattens the curve:
    it is a fixed cost the dendritic side pays no matter how small D is.
    """
    B, N, S, K = args.batch, args.in_features, args.out_features, args.fan_in
    x = torch.randn(B, N, device=device)

    shape = f"{N} -> {N} -> {S}" + (f" -> {S}" if args.readout else "")
    print(f"=== sweep: batch={B} | N={N} -> S={S} | K={K}"
          f"{' | +readout' if args.readout else ''} ===")
    print(f"  dense baseline: {shape}\n")
    print(f"  {'D':>4} {'M':>7} {'coverage':>9} {'seen/soma':>10} {'params':>9} "
          f"{'MACs/vec':>10} {'fwd(ms)':>8} {'fwd+bwd':>9} {'peakMB':>8}   vs dense")

    if args.readout:
        dense_model = DenseMLP3(N, N, S, S)
        dense_macs = DenseMLP3.macs(N, N, S, S)
    else:
        dense_model = DenseMLP(N, N, S)
        dense_macs = DenseMLP.macs(N, N, S)
    base = measure(dense_model.to(device), x, device,
                   f"dense {shape}", dense_macs, pad=33)
    if base is None:
        return
    print()

    d = 1
    fwd_rows, fb_rows = [], []
    while d <= N // K:
        if args.readout:
            m = DendriticMLP(N, S, out_features=S, fan_in=K,
                             dendrites_per_soma=d).to(device)
            info = m.sparsity()
            info["inputs_seen_per_soma"] = min(d * K, N)
        else:
            m = DendriticLinear(N, S, fan_in=K, dendrites_per_soma=d).to(device)
            info = m.sparsity()
        if not args.no_compile:
            torch._dynamo.reset()
            m = torch.compile(m)

        label = (f"{d:>4} {info['dendrites']:>7} {d * K / N:>9.3f} "
                 f"{info['inputs_seen_per_soma']:>10}")
        got = measure(m, x, device, label, info["macs"], base, pad=0)
        if got is not None:
            fwd_rows.append((d, d * K / N, got[0] / base[0]))
            fb_rows.append((d, d * K / N, got[1] / base[1]))

        if not args.no_compile:
            torch._dynamo.reset()
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        d *= 2

    print()
    for name, rows in (("forward", fwd_rows), ("fwd+bwd", fb_rows)):
        hit = crossover(rows)
        if hit:
            print(f"  {name}: matches dense at D ~= {hit[0]:.1f} "
                  f"(coverage ~= {hit[1]:.3f}, {hit[1] * N:.0f} of {N} inputs per soma)")
        elif rows and rows[-1][2] < 1.0:
            print(f"  {name}: still {1/rows[-1][2]:.1f}x FASTER than dense at "
                  f"D={rows[-1][0]} (coverage 1.0) — never crosses")
        else:
            print(f"  {name}: slower than dense across the whole sweep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fan-in", type=int, default=16, help="inputs per dendrite (K)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="grow D until the dendritic layer costs as much as dense")
    ap.add_argument("--readout", action="store_true",
                    help="sweep with the dense readout layer on both models")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--in-features", type=int, default=1024)
    ap.add_argument("--out-features", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"torch {torch.__version__} | {device} ({name}) | K={args.fan_in}\n")

    if args.sweep:
        sweep(args, device)
        return

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

        # --- with a dense readout on the soma: N -> M -> S -> S ---
        # Baseline is the same three layers fully connected, so this stays a
        # like-for-like comparison.
        print(f"  -- plus dense readout: {N} -> {M} -> {S} -> {S} --")
        base3 = measure(DenseMLP3(N, M, S, S).to(device), x, device,
                        f"dense {N}->{M}->{S}->{S}", DenseMLP3.macs(N, M, S, S))
        if base3 is not None:
            mm = DendriticMLP(N, S, out_features=S, fan_in=args.fan_in,
                              dendrites_per_soma=D).to(device)
            info = mm.sparsity()
            measure(mm, x, device, "dendritic + readout", info["macs"], base3)
            if not args.no_compile:
                torch._dynamo.reset()
                measure(torch.compile(mm), x, device, "  ^ compiled",
                        info["macs"], base3)
                torch._dynamo.reset()
            del mm
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
