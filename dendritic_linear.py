import torch
import torch.nn as nn
import torch.nn.functional as F


class DendriticLinear(nn.Module):
    """A sparse two-stage layer:

        N inputs  ->  M = S*D dendrites  ->  leaky_relu  ->  S soma

    Each dendrite reads K inputs and applies a nonlinearity; each soma takes a
    weighted sum of its OWN D dendrites. Dendrites are private to their soma.

    The shape-matched dense equivalent is Linear(N, M) -> leaky_relu ->
    Linear(M, S): same units in every layer, same nonlinearity, but every unit
    connected to everything. This layer does M*K + M multiply-adds per input
    vector against the dense N*M + M*S, which at N=1024, K=16, D=4 is ~75x
    less arithmetic and ~70x fewer parameters.

    Sizing: specify `dendrites_per_soma` (D). The total dendrite count is
    M = out_features * D, and that M is what a dense baseline's hidden layer
    should be set to for a fair comparison.

    Note that D and K together decide how much of the input each soma sees:
    a soma reads D*K inputs out of N. With D=4, K=16, N=1024 that is 64 of
    1024 — genuinely sparse. Setting D = N/K makes every soma read the entire
    input, which costs the same as a dense layer while being far slower to
    execute; that is what `coverage=1.0` does, and it is almost never what
    you want.

    Args:
        in_features:        Input dimension (N).
        out_features:       Number of soma (S) — the output dimension.
        fan_in:             Inputs per dendrite (K). int -> absolute count,
                            float -> fraction of in_features.
        dendrites_per_soma: D. Preferred way to size the layer.
        coverage:           Legacy alternative to D: D = round(coverage*N/K),
                            i.e. how many times each soma tiles the input.
                            Ignored when dendrites_per_soma is given.
        bias:               Bias terms on dendrites and soma.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fan_in: int | float = 16,
        dendrites_per_soma: int | None = None,
        coverage: float | None = None,
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

        if dendrites_per_soma is not None:
            self.D = max(1, dendrites_per_soma)
        elif coverage is not None:
            self.D = max(1, round(coverage * in_features / self.K))
        else:
            self.D = 4
        self.M = out_features * self.D

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
    def sparsity(self) -> dict:
        """Multiply-adds per input vector vs the shape-matched dense MLP
        (Linear(N, M) -> leaky_relu -> Linear(M, S))."""
        ours = self.M * self.K + self.M
        dense = self.in_features * self.M + self.M * self.out_features
        return {
            "dendrites": self.M,
            "inputs_seen_per_soma": min(self.D * self.K, self.in_features),
            "macs": ours,
            "dense_macs": dense,
            "fraction_of_dense": ours / dense,
        }

    def extra_repr(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"K={self.K}, D={self.D}, M={self.M}, "
            f"total_params={p}"
        )


class DendriticMLP(nn.Module):
    """DendriticLinear followed by a dense readout.

        N inputs -> M dendrites -> leaky_relu -> S soma -> leaky_relu
                 -> out_features   (fully connected)

    The point of the readout is mixing. In DendriticLinear each soma only ever
    sees D*K of the N inputs (64 of 1024 at the defaults) and soma never see
    each other, so nothing in the layer combines information across soma. One
    dense layer on the soma fixes that: every output reads every soma.

    Cost note: the readout is S * out_features weights, which at S=256 and
    out_features=256 is 65.8K — more than the 18.7K in the dendritic stage
    itself. It is the expensive part of this model, so keep S modest. Sizing
    the soma layer narrow and letting the readout widen is usually the right
    trade, since that is the direction that keeps the dense matmul small.

    Args:
        in_features:        Input dimension (N).
        soma:               Number of soma (S) — the dendritic layer's width.
        out_features:       Final output dimension. Defaults to `soma`.
        fan_in:             Inputs per dendrite (K).
        dendrites_per_soma: D. Total dendrites M = soma * D.
        bias:               Bias terms throughout.
    """

    def __init__(
        self,
        in_features: int,
        soma: int,
        out_features: int | None = None,
        fan_in: int | float = 16,
        dendrites_per_soma: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        out_features = soma if out_features is None else out_features
        self.dendritic = DendriticLinear(
            in_features, soma, fan_in=fan_in,
            dendrites_per_soma=dendrites_per_soma, bias=bias,
        )
        self.readout = nn.Linear(soma, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.leaky_relu(self.dendritic(x), 0.1)
        return self.readout(s)

    def sparsity(self) -> dict:
        """MACs per input vector, split across the two stages.

        The dense comparison is the same three layers fully connected:
        Linear(N, M) -> Linear(M, S) -> Linear(S, out).
        """
        d = self.dendritic
        readout = d.out_features * self.out_features
        ours = d.sparsity()["macs"] + readout
        dense = (d.in_features * d.M + d.M * d.out_features
                 + d.out_features * self.out_features)
        return {
            "dendrites": d.M,
            "dendritic_macs": d.sparsity()["macs"],
            "readout_macs": readout,
            "macs": ours,
            "dense_macs": dense,
            "fraction_of_dense": ours / dense,
        }
