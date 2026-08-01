"""Checks for BranchedLinear (parallel) and NeighborLinear (mixing)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from branched_linear import BranchedLinear
from neighbor_linear import NeighborLinear
from staged_linear import StagedLinear


def test_zero_is_plain():
    """Both modules must reduce to Linear + leaky_relu with no extra params."""
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float64)
    for m in (BranchedLinear(32, 16, extra_branches=0).double(),
              NeighborLinear(32, 16, neighbors=0).double()):
        expected = F.leaky_relu(m.linear(x), m.negative_slope)
        assert torch.allclose(m(x), expected, atol=1e-12), type(m).__name__
        plain = nn.Linear(32, 16)
        assert (sum(p.numel() for p in m.parameters())
                == sum(p.numel() for p in plain.parameters()))
    print("  ok  extra_branches=0 and neighbors=0 are plain layers")


def test_neighbors_one_is_a_sequential_stage():
    """neighbors=1 is the diagonal case, i.e. exactly one StagedLinear stage."""
    torch.manual_seed(0)
    n = NeighborLinear(16, 8, neighbors=1).double()
    s = StagedLinear(16, 8, extra_stages=1).double()
    with torch.no_grad():
        s.linear.weight.copy_(n.linear.weight)
        s.linear.bias.copy_(n.linear.bias)
        s.scale[0].copy_(n.mix[0])
        s.shift[0].copy_(n.shift)
    assert abs(s.slope - n.slope) < 1e-12
    x = torch.randn(5, 16, dtype=torch.float64)
    assert torch.allclose(n(x), s(x), atol=1e-12)
    print("  ok  neighbors=1 == StagedLinear extra_stages=1")


def test_neighbors_actually_mix():
    """A neuron's output must depend on other neurons' pre-activations."""
    torch.manual_seed(0)
    m = NeighborLinear(16, 8, neighbors=3).double()
    x = torch.randn(1, 16, dtype=torch.float64, requires_grad=True)

    # gradient of output neuron 0 w.r.t. the first layer's other rows
    m(x)[0, 0].backward()
    g = m.linear.weight.grad
    touched = (g.abs().sum(1) > 0).nonzero().flatten().tolist()
    assert len(touched) == 3, f"neuron 0 should read 3 rows, reads {touched}"
    assert 0 in touched and 1 in touched and 7 in touched, (
        f"expected ring neighbours 7, 0, 1; got {touched}")
    print("  ok  neighbors=3 reads 3 rows, wrapping around the ring")


def test_branches_are_parallel_not_sequential():
    """Each branch must read the same z: zeroing one branch weight must not
    change what the others compute."""
    torch.manual_seed(0)
    m = BranchedLinear(16, 8, extra_branches=3).double()
    x = torch.randn(4, 16, dtype=torch.float64)

    with torch.no_grad():
        full = m(x).clone()
        w1 = m.weight[1].clone()
        m.weight[1].zero_()
        without = m(x).clone()
        m.weight[1].copy_(w1)

        # the removed branch's own contribution, computed independently
        z = m.linear(x)
        contrib = w1 * F.leaky_relu(m.breakpoint[1] - z, m.negative_slope)
    assert torch.allclose(full - without, contrib, atol=1e-12)
    print("  ok  branches are independent and additive")


def test_branch_bends_are_reachable():
    """A branch weight should move the curve only on one side of its
    breakpoint — that is what 'adds a bend' means."""
    torch.manual_seed(0)
    m = BranchedLinear(1, 1, extra_branches=1, random_init=False).double()
    with torch.no_grad():
        m.linear.weight.fill_(1.0)
        m.linear.bias.zero_()
        m.breakpoint[0].fill_(0.0)
        m.weight[0].fill_(1.0)      # bend at z = 0

    z = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=torch.float64)
    y = m(z).flatten()
    left = (y[1] - y[0]).item()
    right = (y[3] - y[2]).item()
    assert abs(left - right) > 1e-6, f"no bend: slopes {left} vs {right}"
    print(f"  ok  branch creates a bend (slope {left:.3f} -> {right:.3f})")


def test_shapes_and_cost():
    b = BranchedLinear(2048, 512, extra_branches=4)
    n = NeighborLinear(2048, 512, neighbors=5)
    for m in (b, n):
        assert tuple(m(torch.randn(3, 2048)).shape) == (3, 512)
        assert tuple(m(torch.randn(2, 5, 2048)).shape) == (2, 5, 512)

    cb, cn = b.cost(), n.cost()
    assert cb["branch_params"] == 2 * 4 * 512
    assert cn["mix_params"] == 5 * 512 + 512
    print(f"  ok  branched 2048->512 x4: {cb['params']:,} params, "
          f"branches {cb['branch_fraction']:.2%} of MACs")
    print(f"  ok  neighbor 2048->512 x5: {cn['params']:,} params, "
          f"mixing {cn['mix_fraction']:.2%} of MACs")


if __name__ == "__main__":
    test_zero_is_plain()
    test_neighbors_one_is_a_sequential_stage()
    test_neighbors_actually_mix()
    test_branches_are_parallel_not_sequential()
    test_branch_bends_are_reachable()
    test_shapes_and_cost()
    print("\nall good")
