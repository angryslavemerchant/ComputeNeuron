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

        # --- Fixed gather pattern: tile first, then fill remainder randomly ---
        # For each soma, deal out input features evenly across its dendrites
        # so the first pass guarantees full coverage with no blind spots.
        # Any extra dendrites (coverage > 1) get a fresh shuffle.
        indices = self._build_indices(S, D, K, in_features)
        self.register_buffer("indices", indices)  # (S*D, K)

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
    def _build_indices(S: int, D: int, K: int, N: int) -> torch.Tensor:
        """Build gather indices so each soma's dendrites tile the input
        as evenly as possible.

        For each soma: concatenate enough shuffled copies of range(N) to
        fill D*K slots, then reshape to (D, K). Each full pass of N
        guarantees every feature appears once, so imbalance is at most ±1.
        """
        total = D * K  # how many index slots this soma needs
        passes = (total + N - 1) // N  # how many full shuffles to concatenate

        all_indices = []
        for _ in range(S):
            stream = torch.cat([torch.randperm(N) for _ in range(passes)])
            all_indices.append(stream[:total].view(D, K))
        return torch.cat(all_indices, dim=0)  # (S*D, K)

    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.synaptic, a=0.1, mode="fan_in")
        nn.init.kaiming_uniform_(self.cable, a=0.1, mode="fan_in")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)"""

        # 1 ── Gather: each dendrite picks its K inputs   (..., S*D, K)
        g = x[..., self.indices]

        # 2 ── Dendrite: dot-product + nonlinearity        (..., S*D)
        d = (g * self.synaptic).sum(-1)
        if self.dendrite_bias is not None:
            d = d + self.dendrite_bias
        d = F.leaky_relu(d, 0.1)

        # 3 ── Reshape to per-soma groups                  (..., S, D)
        d = d.unflatten(-1, (self.out_features, self.D))

        # 4 ── Soma: cable-weighted sum                    (..., S)
        s = (d * self.cable).sum(-1)
        if self.soma_bias is not None:
            s = s + self.soma_bias

        # No output activation — apply externally, same as nn.Linear
        return s

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"K={self.K}, D={self.D}, "
            f"total_params={p}"
        )
