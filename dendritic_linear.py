import torch
import torch.nn as nn
import torch.nn.functional as F


class DendriticLinear(nn.Module):
    """Drop-in replacement for nn.Linear (or a full FFN) with dendritic structure.

    Each output (soma) owns D dendrites. Each dendrite gathers K features
    from the input via a fixed random index pattern, computes a weighted sum,
    and passes through a nonlinearity. The soma then takes a cable-weighted
    sum of its dendrites to produce one scalar output.

    Because the dendrite stage already provides expand -> nonlinearity -> contract,
    a single DendriticLinear(d, d) can replace an entire 2-layer FFN.

    Parameter count ≈ out_features * coverage * in_features  (for K >> 1).
    At coverage=1.0 this matches nn.Linear(in, out) in param count.

    Args:
        in_features:   Input dimension  (N)
        out_features:  Output dimension (S — number of soma)
        fan_in:        How many input features each dendrite reads.
                       int  -> absolute count (e.g. 64 means each dendrite reads 64 features)
                       float -> fraction of in_features (e.g. 0.1 means 10%)
        coverage:      How many times each soma's dendrites tile the input.
                       D = max(1, round(coverage * N / K))
        bias:          Include bias terms on dendrites and soma.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fan_in: int | float = 0.1,
        coverage: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Derive K (fan-in per dendrite) and D (dendrites per soma)
        if isinstance(fan_in, int):
            self.K = max(1, min(in_features, fan_in))
        else:
            self.K = max(1, min(in_features, round(fan_in * in_features)))
        self.D = max(1, round(coverage * in_features / self.K))

        S, D, K = self.out_features, self.D, self.K

        # --- Fixed gather pattern ---
        # Every soma uses the SAME tiling, so we only store the (D*K,) stream
        # once instead of S redundant copies. When D*K <= N the stream is just
        # arange(D*K) and the gather degenerates to a slice.
        self.register_buffer("tile_index", self._build_tile(D, K, in_features))
        self.slice_only = D * K <= in_features

        # --- Learnable parameters ---
        self.synaptic = nn.Parameter(torch.empty(S * D, K))   # dendrite input weights
        self.cable = nn.Parameter(torch.empty(S, D))           # dendrite -> soma weights

        if bias:
            self.dendrite_bias = nn.Parameter(torch.zeros(S * D))
            self.soma_bias = nn.Parameter(torch.zeros(S))
        else:
            self.register_parameter("dendrite_bias", None)
            self.register_parameter("soma_bias", None)

        self._init_weights()

    # ------------------------------------------------------------------
    @staticmethod
    def _build_tile(D: int, K: int, N: int) -> torch.Tensor:
        """The (D*K,) index stream every soma tiles the input with.

        Dendrite 0 reads [0..K-1], dendrite 1 reads [K..2K-1], etc., wrapping
        around when D*K > N (coverage > 1). Every soma gets the same tiling —
        different soma learn different weights, which is what differentiates
        them.
        """
        return torch.arange(N).repeat((D * K + N - 1) // N)[:D * K]

    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.synaptic, a=0.1, mode="fan_in")
        nn.init.kaiming_uniform_(self.cable, a=0.1, mode="fan_in")

    # ------------------------------------------------------------------
    def _tile(self, x: torch.Tensor) -> torch.Tensor:
        """(..., N) -> (..., D, K): the input as each dendrite sees it, once."""
        xt = x[..., : self.D * self.K] if self.slice_only else x[..., self.tile_index]
        return xt.unflatten(-1, (self.D, self.K))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)"""
        S, D, K = self.out_features, self.D, self.K

        # 1 ── Tile the input ONCE, shared by every soma      (..., D, K)
        xt = self._tile(x)

        # 2 ── Dendrites: D batched matmuls (B,K)x(K,S)       (..., S, D)
        # Contracting K here instead of materializing a (..., S*D, K) gather
        # is what keeps this on cuBLAS. Mathematically identical.
        d = torch.einsum("...dk,sdk->...sd", xt, self.synaptic.view(S, D, K))
        if self.dendrite_bias is not None:
            d = d + self.dendrite_bias.view(S, D)
        d = F.leaky_relu(d, 0.1)

        # 3 ── Soma: cable-weighted sum                       (..., S)
        s = (d * self.cable).sum(-1)
        if self.soma_bias is not None:
            s = s + self.soma_bias

        # No output activation — apply externally, same as nn.Linear
        return s

    # ------------------------------------------------------------------
    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Original gather-based forward. Kept only to verify that `forward`
        is mathematically equivalent — it materializes a (..., S*D, K)
        tensor and is much slower."""
        S, D, K = self.out_features, self.D, self.K
        indices = self.tile_index.view(D, K).repeat(S, 1)      # (S*D, K)

        g = x[..., indices]
        d = (g * self.synaptic).sum(-1)
        if self.dendrite_bias is not None:
            d = d + self.dendrite_bias
        d = F.leaky_relu(d, 0.1)
        d = d.unflatten(-1, (S, D))

        s = (d * self.cable).sum(-1)
        if self.soma_bias is not None:
            s = s + self.soma_bias
        return s

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"K={self.K}, D={self.D}, "
            f"total_params={p}"
        )
