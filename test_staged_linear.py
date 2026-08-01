"""Checks for StagedLinear."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from staged_linear import StagedLinear


def test_zero_stages_is_exactly_a_plain_layer():
    """extra_stages=0 must BE Linear + leaky_relu — no extra parameters, no
    folding required. That is what makes it a clean control."""
    torch.manual_seed(0)
    m = StagedLinear(32, 16, extra_stages=0).double()
    x = torch.randn(4, 32, dtype=torch.float64)

    expected = F.leaky_relu(m.linear(x), m.negative_slope)
    assert torch.allclose(m(x), expected, atol=1e-12)

    assert m.scale is None and m.shift is None
    assert m.cost()["stage_params"] == 0
    plain = nn.Linear(32, 16)
    assert (sum(p.numel() for p in m.parameters())
            == sum(p.numel() for p in plain.parameters()))
    assert abs(m.slope - m.negative_slope) < 1e-12, "no correction needed at 0"
    print("  ok  extra_stages=0 is exactly Linear + leaky_relu, same params")


def test_one_stage_adds_real_structure():
    """extra_stages=1 must NOT be foldable into a plain layer: the inner
    nonlinearity is what stops a and c being absorbed into W and b."""
    torch.manual_seed(0)
    m = StagedLinear(8, 4, extra_stages=1, random_init=True).double()
    x = torch.randn(64, 8, dtype=torch.float64)

    # any plain Linear + leaky_relu is monotone in each pre-activation; with a
    # negative scale the staged version is not, so no rescaling can match it
    with torch.no_grad():
        m.scale[0].fill_(-1.0)
        m.shift[0].fill_(0.5)
    z = m.linear(x)
    y = m(x)
    order = z[:, 0].argsort()
    diffs = y[order, 0].diff()
    assert (diffs > 0).any() and (diffs < 0).any(), (
        "output should be non-monotone in the pre-activation, so it cannot be "
        "a rescaled plain layer")
    print("  ok  extra_stages=1 is not foldable into a plain layer")


def test_slope_correction():
    """The composite negative slope should be the target at any depth."""
    for extra in (0, 1, 2, 7):
        m = StagedLinear(8, 4, extra_stages=extra, correct_slope=True,
                         random_init=False).double()
        composite = m.slope ** m.nonlinearities
        assert abs(composite - m.negative_slope) < 1e-9, (
            f"extra_stages={extra}: composite {composite} != {m.negative_slope}")

        # empirically: far negative, every nonlinearity is in its leaky branch
        with torch.no_grad():
            m.linear.weight.zero_()
            m.linear.weight[:, 0] = 1.0
            m.linear.bias.zero_()
        x = torch.full((1, 8), -1e6, dtype=torch.float64)
        y = m(x)
        assert torch.allclose(y / x[0, 0],
                              torch.full_like(y, m.negative_slope), rtol=1e-6)
    print("  ok  composite negative slope is the target at every depth")


def test_uncorrected_slope_decays():
    """Without correction the slope shrinks geometrically — the reason the
    correction exists."""
    m = StagedLinear(8, 4, extra_stages=2, correct_slope=False, random_init=False)
    assert abs(m.slope - 0.1) < 1e-12
    assert abs(m.slope ** m.nonlinearities - 0.001) < 1e-12
    print("  ok  uncorrected stacking decays 0.1 -> 0.001 as expected")


def test_shapes_and_cost():
    m = StagedLinear(2048, 512, extra_stages=4)
    assert tuple(m(torch.randn(3, 2048)).shape) == (3, 512)
    assert tuple(m(torch.randn(2, 5, 2048)).shape) == (2, 5, 512)

    c = m.cost()
    assert c["stage_params"] == 2 * 4 * 512
    assert c["matmul_macs"] == 2048 * 512
    assert c["stage_macs"] == 4 * 512
    assert c["bends"] == 5
    print(f"  ok  2048->512 extra=4: {c['params']:,} params "
          f"({c['stage_params']:,} from stages), {c['bends']} bends, "
          f"stages are {c['stage_fraction']:.2%} of MACs")


def test_gradients_flow_to_every_stage():
    torch.manual_seed(0)
    m = StagedLinear(16, 8, extra_stages=3).double()
    m(torch.randn(4, 16, dtype=torch.float64)).sum().backward()
    assert (m.scale.grad.abs().sum(1) > 0).all(), "a stage got no scale gradient"
    assert (m.shift.grad.abs().sum(1) > 0).all(), "a stage got no shift gradient"
    print("  ok  every stage receives gradient")


if __name__ == "__main__":
    test_zero_stages_is_exactly_a_plain_layer()
    test_one_stage_adds_real_structure()
    test_slope_correction()
    test_uncorrected_slope_decays()
    test_shapes_and_cost()
    test_gradients_flow_to_every_stage()
    print("\nall good")
