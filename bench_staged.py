"""Speed/params/FLOPs for the three per-neuron variants against a plain
Linear + leaky_relu.

All are width-matched — one dense matmul, then something cheap per neuron:

    staged     k sequential scale/shift/nonlinearity stages (slopes multiply)
    branched   k parallel branches summed together (slopes add)
    neighbors  a second stage reading k neurons instead of 1 (mixing)

Setting 0 makes each of them a plain Linear + leaky_relu with no extra
parameters, so those rows should land on the baseline and anything above is
the real cost.

Usage:
    python bench_staged.py
    python bench_staged.py --extra-stages 0,1,4,8 --batch 16384
    python bench_staged.py --no-variants --no-compile
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench import sync, timeit
from branched_linear import BranchedLinear
from neighbor_linear import NeighborLinear
from staged_linear import StagedLinear

torch._dynamo.config.recompile_limit = 64


class PlainLinear(nn.Module):
    """nn.Linear + leaky_relu — the baseline StagedLinear must not lose to."""

    def __init__(self, n, m, negative_slope=0.1):
        super().__init__()
        self.linear = nn.Linear(n, m)
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(self.linear(x), self.negative_slope)


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
        print(f"  {label:<24} {params/1e3:>9.1f}K {macs/1e3:>10.1f}K {'OOM':>8}")
        return None

    ratio = "" if base is None else f"{fwd/base[0]:>7.2f}x {fb/base[1]:>8.2f}x"
    print(f"  {label:<24} {params/1e3:>9.1f}K {macs/1e3:>10.1f}K "
          f"{fwd:>8.3f} {fb:>9.3f} {peak:>8.0f}  {ratio}")
    return fwd, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--extra-stages", default="0,1,2,3,7",
                    help="comma-separated per-neuron stage counts; 0 is a plain "
                         "Linear + leaky_relu")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-variants", action="store_true",
                    help="staged only; skip the branched and neighbor rows")
    ap.add_argument("--shape", default=None, help="N,M (default: several)")
    args = ap.parse_args()
    stage_list = [int(s) for s in args.extra_stages.split(",") if s.strip()]

    device = torch.device(args.device)
    torch.manual_seed(0)
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"torch {torch.__version__} | {device} ({name})\n")

    if args.shape:
        n, m = (int(v) for v in args.shape.split(","))
        shapes = [(args.batch, n, m)]
    else:
        shapes = [
            (args.batch, 2048, 512),
            (args.batch, 1024, 1024),
            (args.batch, 2048, 2048),
        ]

    for B, N, M in shapes:
        x = torch.randn(B, N, device=device)
        print(f"=== batch={B} | {N} -> {M} ===")
        print(f"  {'module':<24} {'params':>10} {'MACs/vec':>11} "
              f"{'fwd(ms)':>8} {'fwd+bwd':>9} {'peakMB':>8}   vs plain (like for like)")

        # Two baselines: eager rows are compared against the eager baseline and
        # compiled rows against the compiled one, so no row is ever measured
        # against a differently-optimised reference.
        plain = PlainLinear(N, M).to(device)
        base = measure(plain, x, device, "Linear + leaky_relu", N * M)
        if base is None:
            continue
        base_c = base
        if not args.no_compile:
            torch._dynamo.reset()
            base_c = measure(torch.compile(plain), x, device,
                             "  ^ compiled", N * M, base) or base
            torch._dynamo.reset()

        variants = []
        for s in stage_list:
            variants.append((
                f"staged x{s}" + (" = plain" if s == 0 else ""),
                lambda s=s: StagedLinear(N, M, extra_stages=s)))
        if not args.no_variants:
            for b in stage_list:
                variants.append((
                    f"branched x{b}" + (" = plain" if b == 0 else ""),
                    lambda b=b: BranchedLinear(N, M, extra_branches=b)))
            for k in [n for n in (0, 1, 3, 5, 9) if n <= M]:
                variants.append((
                    f"neighbors x{k}" + (" = plain" if k == 0 else ""),
                    lambda k=k: NeighborLinear(N, M, neighbors=k)))

        for tag, build in variants:
            m_ = build().to(device)
            c = m_.cost()
            measure(m_, x, device, tag, c["macs"], base)

            if not args.no_compile:
                torch._dynamo.reset()
                measure(torch.compile(m_), x, device, "  ^ compiled",
                        c["macs"], base_c)
                torch._dynamo.reset()

            del m_
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
