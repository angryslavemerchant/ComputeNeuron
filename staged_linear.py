import torch
import torch.nn as nn
import torch.nn.functional as F


class StagedLinear(nn.Module):
    """A dense layer whose neurons learn their own activation shape.

    One matmul, then `stages` repetitions of a per-neuron scale, shift, and
    nonlinearity:

        z  = W x + b
        z <- phi(a_i * z + c_i)     for i in 1..stages

    a_i and c_i are one scalar per neuron, not matrices — neuron j in stage i
    reads only neuron j from the stage below. All the mixing between neurons
    happens in the single matmul; the stages only reshape each neuron's
    response curve.

    Every stage adds one bend to that curve. With leaky_relu, stages=1 is a
    plain activation (one bend, at zero), stages=2 gives two bends, and so on,
    with the layer learning where each bend sits and how steep the segments
    are. This is the Adaptive-Piecewise-Linear-unit idea, folded into the layer.

    stages=1 is the CONTROL, not a feature: phi(a*(Wx+b)+c) is affine inside
    the nonlinearity, so it is exactly nn.Linear followed by leaky_relu with
    the weights rescaled. It costs 2*out_features parameters that buy nothing,
    which makes it the right baseline to measure the other settings against.

    Cost is dominated entirely by the matmul: each stage adds out_features
    parameters times two and one elementwise pass, against a matmul that is
    in_features * out_features. At 2048 -> 512 a stage is 1024 parameters on
    top of 1.05M, about 0.1%.

    Args:
        in_features:     Input dimension.
        out_features:    Output dimension.
        stages:          How many scale/shift/nonlinearity stages (>= 1).
                         2 is the smallest setting that does anything.
        negative_slope:  Target negative slope of the COMPOSITE activation.
        correct_slope:   Give each stage a slope of negative_slope**(1/stages)
                         so the composition lands on negative_slope regardless
                         of depth. Without this, stacking leaky_relu shrinks
                         the negative slope geometrically (0.1 -> 0.01 -> 0.001)
                         and deeper settings start from a different function,
                         confounding any comparison.
        random_init:     Draw the stage scales from U(0.5, 1.5) and shifts from
                         U(-0.1, 0.1), so neurons start with different curves.
                         False starts every stage at scale 1, shift 0.
        bias:            Bias on the matmul.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stages: int = 2,
        negative_slope: float = 0.1,
        correct_slope: bool = True,
        random_init: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if stages < 1:
            raise ValueError("stages must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.stages = stages
        self.negative_slope = negative_slope
        self.slope = negative_slope ** (1.0 / stages) if correct_slope else negative_slope

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Parameter(torch.empty(stages, out_features))
        self.shift = nn.Parameter(torch.empty(stages, out_features))

        if random_init:
            nn.init.uniform_(self.scale, 0.5, 1.5)
            nn.init.uniform_(self.shift, -0.1, 0.1)
        else:
            nn.init.ones_(self.scale)
            nn.init.zeros_(self.shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)"""
        z = self.linear(x)
        for i in range(self.stages):
            z = F.leaky_relu(self.scale[i] * z + self.shift[i], self.slope)
        return z

    # ------------------------------------------------------------------
    def bends(self) -> int:
        """Bends in each neuron's piecewise-linear activation."""
        return self.stages

    def cost(self) -> dict:
        """Multiply-adds per input vector, split between matmul and stages."""
        matmul = self.in_features * self.out_features
        stages = self.stages * self.out_features
        params = sum(p.numel() for p in self.parameters())
        return {
            "params": params,
            "stage_params": 2 * self.stages * self.out_features,
            "matmul_macs": matmul,
            "stage_macs": stages,
            "macs": matmul + stages,
            "stage_fraction": stages / (matmul + stages),
        }

    def extra_repr(self) -> str:
        c = self.cost()
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"stages={self.stages}, bends={self.bends()}, "
            f"slope_per_stage={self.slope:.4f}, "
            f"stage_params={c['stage_params']} of {c['params']}"
        )
