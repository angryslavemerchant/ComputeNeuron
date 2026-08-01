"""Checks for DendriticLinear:

  1. the batched-matmul forward matches the plain gather reference
  2. every input actually reaches the output (the coverage bug regression)
"""

import torch

from dendritic_linear import DendriticLinear, DendriticMLP

CASES = [
    # (N, S, fan_in, D)
    (64, 32, 16, 4),      # sparse: window 64 == N, one group
    (256, 32, 16, 4),     # sparse: window 64, needs 4 groups
    (2048, 64, 16, 4),    # the benchmark regime: window 64, 32 groups
    (256, 32, 0.25, 4),   # float fan_in
    (60, 16, 16, 4),      # N not divisible by the window
    (48, 8, 1.0, 1),      # K == N, D == 1
    (256, 32, 16, 16),    # window 256 == N, full coverage per soma
    (256, 6, 16, 4),      # S prime-ish: G does not divide S, soma padded
    (2048, 10, 16, 51),   # the 2048->510->10 case: G=3 vs S=10
    (2048, 10, 16, 4),    # too few soma to cover N at all
    (256, 6, 16, 4),      # S=6, 4 windows needed: a group must not be all padding
    (256, 7, 16, 4),      # likewise with an odd soma count
    (1000, 12, 16, 5),    # nothing divides anything
]


def test_forward_matches_reference():
    torch.manual_seed(0)
    for N, S, fan_in, D in CASES:
        m = DendriticLinear(N, S, fan_in=fan_in, dendrites_per_soma=D).double()
        x = torch.randn(5, N, dtype=torch.float64)

        fast, ref = m(x), m.forward_reference(x)
        assert torch.allclose(fast, ref, atol=1e-12), (
            f"N={N} S={S} fan_in={fan_in} D={D}: "
            f"max diff {(fast - ref).abs().max()}")

        m.zero_grad(set_to_none=True)
        m(x).sum().backward()
        g_fast = [p.grad.clone() for p in m.parameters()]
        m.zero_grad(set_to_none=True)
        m.forward_reference(x).sum().backward()
        for a, b in zip(g_fast, [p.grad for p in m.parameters()]):
            assert torch.allclose(a, b, atol=1e-10), f"grad mismatch N={N} S={S}"

        x3 = torch.randn(2, 4, N, dtype=torch.float64)
        assert torch.allclose(m(x3), m.forward_reference(x3), atol=1e-12)

        info = m.sparsity()
        print(f"  ok  N={N:>5} S={S:>3} K={m.K:>3} D={D:>3} -> M={m.M:>4} "
              f"G={info['groups']:>3} covers {info['inputs_covered']}/{N}")


def test_every_input_is_used():
    """Regression: with one shared tiling the layer only read the first D*K
    inputs and silently ignored the rest. Every input must reach the output."""
    torch.manual_seed(0)
    for N, S, fan_in, D in CASES:
        m = DendriticLinear(N, S, fan_in=fan_in, dendrites_per_soma=D).double()
        covered = m.sparsity()["inputs_covered"]

        x = torch.randn(1, N, dtype=torch.float64, requires_grad=True)
        m(x).sum().backward()
        touched = int((x.grad != 0).sum())

        assert touched == covered, (
            f"N={N} S={S} D={D}: gradient reaches {touched} inputs but "
            f"sparsity() claims {covered}")
        # Full coverage must hold whenever there are enough soma to reach every
        # input; only S*D*K < N is a legitimate excuse for missing some.
        if S * D * m.K >= N:
            assert touched == N, (
                f"N={N} S={S} D={D}: {S}*{D}*{m.K} >= {N} so the groups should "
                f"tile the input, but only {touched}/{N} inputs are used")

        # and each soma individually sees only its own window
        assert len(set(m.soma_indices()[0].flatten().tolist())) <= m.window

        # every group must own at least one real (non-padding) soma, else its
        # window is computed and then thrown away
        assert (m.G - 1) * m.Sg < S, (
            f"N={N} S={S} D={D}: group {m.G - 1} is entirely padding "
            f"(G={m.G}, Sg={m.Sg}, S_pad={m.S_pad})")
        assert m.S_pad - S < m.Sg, f"padding {m.S_pad - S} >= group size {m.Sg}"
    print("  ok  every input reaches the output")


def test_mlp_matches_parts():
    torch.manual_seed(0)
    m = DendriticMLP(256, 32, out_features=64, fan_in=16,
                     dendrites_per_soma=4).double()
    x = torch.randn(3, 256, dtype=torch.float64)
    expected = m.readout(torch.nn.functional.leaky_relu(m.dendritic(x), 0.1))
    assert torch.allclose(m(x), expected, atol=1e-12)
    print("  ok  DendriticMLP composes the layer and readout")


if __name__ == "__main__":
    test_forward_matches_reference()
    test_every_input_is_used()
    test_mlp_matches_parts()

    m = DendriticLinear(2048, 512, fan_in=16, dendrites_per_soma=4)
    print("\n2048 -> 512, K=16, D=4:")
    for k, v in m.sparsity().items():
        print(f"   {k}: {v:,}" if isinstance(v, int) else f"   {k}: {v:.4f}")
    print("\nall good")
