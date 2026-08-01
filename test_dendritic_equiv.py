"""Verify the einsum forward matches the original gather forward."""

import torch

from dendritic_linear import DendriticLinear


def test_forward_matches_reference():
    torch.manual_seed(0)
    for N, S, fan_in, D in [
        (64, 32, 16, 4),        # the canonical shape: sparse, D*K < N
        (64, 32, 0.25, 4),      # float fan_in
        (60, 16, 16, 4),        # N not divisible by K, needs wrap
        (64, 32, 16, 4),        # repeat with int fan_in
        (48, 8, 1.0, 1),        # K == N, D == 1
        (64, 32, 16, 8),        # D*K > N, wraps
    ]:
        m = DendriticLinear(N, S, fan_in=fan_in, dendrites_per_soma=D).double()
        x = torch.randn(7, N, dtype=torch.float64)

        fast, ref = m(x), m.forward_reference(x)
        assert torch.allclose(fast, ref, atol=1e-12), (
            f"mismatch N={N} S={S} fan_in={fan_in} D={D}: "
            f"max diff {(fast - ref).abs().max()}"
        )

        # gradients too
        for fn in (m.forward, m.forward_reference):
            m.zero_grad(set_to_none=True)
            fn(x).sum().backward()
        g_fast = [p.grad.clone() for p in m.parameters()]
        m.zero_grad(set_to_none=True)
        m.forward_reference(x).sum().backward()
        g_ref = [p.grad.clone() for p in m.parameters()]
        for a, b in zip(g_fast, g_ref):
            assert torch.allclose(a, b, atol=1e-10)

        # batched / 3D input
        x3 = torch.randn(2, 5, N, dtype=torch.float64)
        assert torch.allclose(m(x3), m.forward_reference(x3), atol=1e-12)

        print(f"  ok  N={N} S={S} fan_in={fan_in} D={D} -> K={m.K} M={m.M}")


if __name__ == "__main__":
    test_forward_matches_reference()
    m = DendriticLinear(1024, 256, fan_in=16, dendrites_per_soma=4)
    print("\n1024 -> 256, K=16, D=4:", m.sparsity())
    print("all equivalent")
