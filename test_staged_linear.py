"""Checks for StagedLinear."""

import torch
import torch.nn.functional as F

from staged_linear import StagedLinear


def test_one_stage_is_a_plain_layer():
    """stages=1 must reduce to Linear + leaky_relu, since a*(Wx+b)+c is still
    affine. This is what makes stages=1 a valid control."""
    torch.manual_seed(0)
    m = StagedLinear(32, 16, stages=1, correct_slope=True).double()
    x = torch.randn(4, 32, dtype=torch.float64)

    # fold the stage's scale/shift back into the matmul
    w = m.linear.weight * m.scale[0].unsqueeze(1)
    b = m.linear.bias * m.scale[0] + m.shift[0]
    expected = F.leaky_relu(F.linear(x, w, b), m.slope)

    assert torch.allclose(m(x), expected, atol=1e-12)
    assert abs(m.slope - m.negative_slope) < 1e-12, "1 stage needs no correction"
    print("  ok  stages=1 folds into a plain Linear + leaky_relu")


def test_slope_correction():
    """The composite negative slope should be the target at any depth."""
    for stages in (1, 2, 3, 8):
        m = StagedLinear(8, 4, stages=stages, correct_slope=True,
                         random_init=False).double()
        # far into the negative region every stage is in its leaky branch,
        # so the composite slope is slope**stages
        composite = m.slope ** stages
        assert abs(composite - m.negative_slope) < 1e-9, (
            f"stages={stages}: composite slope {composite} != {m.negative_slope}")

        # and check it empirically with identity-init scales
        x = torch.full((1, 8), -1e6, dtype=torch.float64)
        with torch.no_grad():
            m.linear.weight.fill_(0.0)
            m.linear.weight[:, 0] = 1.0
            m.linear.bias.zero_()
        y = m(x)
        assert torch.allclose(y / x[0, 0], torch.full_like(y, m.negative_slope),
                              rtol=1e-6)
    print("  ok  composite negative slope is the target at every depth")


def test_uncorrected_slope_decays():
    """Without correction the slope shrinks geometrically — the reason the
    correction exists."""
    m = StagedLinear(8, 4, stages=3, correct_slope=False, random_init=False)
    assert abs(m.slope - 0.1) < 1e-12
    assert abs(m.slope ** m.stages - 0.001) < 1e-12
    print("  ok  uncorrected stacking decays 0.1 -> 0.001 as expected")


def test_shapes_and_cost():
    m = StagedLinear(2048, 512, stages=4)
    assert tuple(m(torch.randn(3, 2048)).shape) == (3, 512)
    assert tuple(m(torch.randn(2, 5, 2048)).shape) == (2, 5, 512)

    c = m.cost()
    assert c["stage_params"] == 2 * 4 * 512
    assert c["matmul_macs"] == 2048 * 512
    assert c["stage_macs"] == 4 * 512
    print(f"  ok  2048->512 s=4: {c['params']:,} params "
          f"({c['stage_params']:,} from stages), "
          f"stages are {c['stage_fraction']:.2%} of MACs")


def test_gradients_flow_to_every_stage():
    torch.manual_seed(0)
    m = StagedLinear(16, 8, stages=3).double()
    m(torch.randn(4, 16, dtype=torch.float64)).sum().backward()
    assert (m.scale.grad.abs().sum(1) > 0).all(), "a stage got no scale gradient"
    assert (m.shift.grad.abs().sum(1) > 0).all(), "a stage got no shift gradient"
    print("  ok  every stage receives gradient")


if __name__ == "__main__":
    test_one_stage_is_a_plain_layer()
    test_slope_correction()
    test_uncorrected_slope_decays()
    test_shapes_and_cost()
    test_gradients_flow_to_every_stage()
    print("\nall good")
