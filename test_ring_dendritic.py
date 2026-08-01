"""Check RingDendriticLinear against an explicit masked-dense version, and
confirm the ring gives even input coverage."""

import torch

from ring_dendritic import RingDendriticLinear

CASES = [
    # (N, S, K, D, dilation)
    (64, 16, 16, 4, 1),          # M == N, shift fast path
    (64, 16, 16, 4, "uniform"),  # evenly spread taps
    (64, 8, 8, 4, 1),            # M < N, indexed path
    (64, 32, 16, 4, 1),          # M > N
    (60, 15, 16, 4, 3),          # N not divisible by K, dilation 3
    (32, 8, 32, 4, 1),           # K == N, full fan-in
]


def test_matches_masked_dense():
    torch.manual_seed(0)
    for N, S, K, D, dil in CASES:
        m = RingDendriticLinear(N, S, fan_in=K, dendrites_per_soma=D,
                                dilation=dil).double()
        x = torch.randn(5, N, dtype=torch.float64)

        fast, ref = m(x), m.forward_reference(x)
        assert torch.allclose(fast, ref, atol=1e-12), (
            f"N={N} S={S} K={K} D={D} dil={dil}: "
            f"max diff {(fast - ref).abs().max()}")

        m.zero_grad(set_to_none=True)
        m(x).sum().backward()
        g_fast = [p.grad.clone() for p in m.parameters()]
        m.zero_grad(set_to_none=True)
        m.forward_reference(x).sum().backward()
        for a, b in zip(g_fast, [p.grad for p in m.parameters()]):
            assert torch.allclose(a, b, atol=1e-10), f"grad mismatch {N} {S}"

        x3 = torch.randn(2, 4, N, dtype=torch.float64)
        assert torch.allclose(m(x3), m.forward_reference(x3), atol=1e-12)

        print(f"  ok  N={N} S={S} K={K} D={D} dil={dil} -> M={m.M}")


def test_even_coverage():
    """Every input should be read the same number of times (+-1)."""
    for N, S, K, D, dil in CASES:
        m = RingDendriticLinear(N, S, fan_in=K, dendrites_per_soma=D, dilation=dil)
        counts = torch.bincount(m.taps.reshape(-1), minlength=N)
        spread = int(counts.max() - counts.min())
        assert spread <= 1, f"uneven coverage (spread {spread}) for N={N} M={m.M}"
    print("  ok  every input read equally often (spread <= 1)")


if __name__ == "__main__":
    test_matches_masked_dense()
    test_even_coverage()

    m = RingDendriticLinear(1024, 256, fan_in=16, dendrites_per_soma=4)
    print("\n1024 -> 256, K=16, D=4:", m.sparsity())
    print("all good")
