"""Speed/params/FLOPs for StagedLinear against a plain Linear + leaky_relu.

Everything is width-matched, so the only difference is how many
scale/shift/nonlinearity stages sit on top of the one matmul. The question is
whether extra stages are effectively free.

Usage:
    python bench_staged.py
    python bench_staged.py --stages 1,2,4,8 --batch 16384
    python bench_staged.py --no-compile
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench import sync, timeit
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
    ap.add_argument("--stages", default="1,2,3,4,8",
                    help="comma-separated stage counts to test")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--shape", default=None, help="N,M (default: several)")
    args = ap.parse_args()
    stage_list = [int(s) for s in args.stages.split(",") if s.strip()]

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
              f"{'fwd(ms)':>8} {'fwd+bwd':>9} {'peakMB':>8}   vs plain")

        base = measure(PlainLinear(N, M).to(device), x, device,
                       "Linear + leaky_relu", N * M)
        if base is None:
            continue

        for s in stage_list:
            m_ = StagedLinear(N, M, stages=s).to(device)
            c = m_.cost()
            tag = f"staged s={s} ({s} bend{'s' if s > 1 else ''})"
            measure(m_, x, device, tag, c["macs"], base)

            if not args.no_compile:
                torch._dynamo.reset()
                measure(torch.compile(m_), x, device, "  ^ compiled",
                        c["macs"], base)
                torch._dynamo.reset()

            del m_
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
