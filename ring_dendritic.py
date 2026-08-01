import torch
import torch.nn as nn
import torch.nn.functional as F


class RingDendriticLinear(nn.Module):
    """Sparse dendritic layer with ring-window connectivity.

    Three layers of units:

        inputs (N)  ->  dendrites (M = out_features * dendrites_per_soma)
                    ->  soma (out_features)

    Each dendrite reads K inputs, weights them, sums, and applies a
    nonlinearity. Each soma reads its own D dendrites and takes a weighted
    sum. Dendrites are private to their soma; nothing is shared.

    Connectivity is a ring window rather than a random pattern. Dendrite m
    starts at position `start(m) = m * N // M` and reads

        x[(start(m) + j * dilation) mod N]    for j in 0..K-1

    which has two useful properties:

      * Even sampling. Every input is read by exactly M*K/N dendrites, with
        no edge effects — the ring wraps, so no channel is under-served the
        way it would be at the boundary of a non-circular window.
      * No lookups. The taps are a fixed shift of the input, so the whole
        dendrite stage is K shifted multiply-accumulates over contiguous
        memory instead of a gather. This is what makes it cheap.

    Work per input vector is M*K + M multiply-adds, versus N*out_features
    for a dense nn.Linear. At N=1024, K=16, D=4 that is ~1.6% of dense.

    Because the arithmetic is so small, this layer is bound by memory traffic
    and kernel launches, not by compute. Run it under torch.compile: the
    accumulation loop fuses into a single kernel that keeps the running sum
    in registers. Eager mode launches ~2K kernels and moves far more memory
    than it needs to.

    Args:
        in_features:        Input dimension (N).
        out_features:       Number of soma (S) — the output dimension.
        fan_in:             Inputs per dendrite (K).
        dendrites_per_soma: Dendrites owned by each soma (D). Total
                            dendrites M = S * D.
        dilation:           Spacing between a dendrite's taps. 1 gives a
                            contiguous window (neighbouring dendrites overlap
                            heavily). "uniform" gives N // K, so each dendrite
                            samples evenly around the whole ring — the closest
                            thing to random wiring that still costs nothing.
        bias:               Bias terms on dendrites and soma.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fan_in: int = 16,
        dendrites_per_soma: int = 4,
        dilation: int | str = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = max(1, min(in_features, fan_in))
        self.D = max(1, dendrites_per_soma)
        self.M = out_features * self.D

        if dilation == "uniform":
            dilation = max(1, in_features // self.K)
        self.dilation = int(dilation)

        # taps[j, m] = which input dendrite m reads for its j-th weight.
        # Registered as a buffer so .to(device) and state_dict work, but the
        # fast path never actually indexes with it (see forward).
        starts = torch.arange(self.M) * in_features // self.M      # (M,)
        j = torch.arange(self.K).unsqueeze(-1)                     # (K, 1)
        self.register_buffer("taps", (starts + j * self.dilation) % in_features)

        # A pure shift is only available when dendrite m starts exactly at
        # input m, i.e. one dendrite per input position.
        self.shift_only = self.M == in_features

        self.synaptic = nn.Parameter(torch.empty(self.M, self.K))
        self.cable = nn.Parameter(torch.empty(out_features, self.D))

        if bias:
            self.dendrite_bias = nn.Parameter(torch.zeros(self.M))
            self.soma_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("dendrite_bias", None)
            self.register_parameter("soma_bias", None)

        nn.init.kaiming_uniform_(self.synaptic, a=0.1, mode="fan_in")
        nn.init.kaiming_uniform_(self.cable, a=0.1, mode="fan_in")

    # ------------------------------------------------------------------
    def dendrites(self, x: torch.Tensor) -> torch.Tensor:
        """(..., N) -> (..., M), pre-nonlinearity.

        K shifted multiply-accumulates. No (..., M, K) tensor is built, so
        under torch.compile this fuses to one kernel holding the running sum
        in registers.
        """
        acc = None
        for j in range(self.K):
            # roll(-shift) puts input (m + shift) at position m, which is
            # exactly what dendrite m wants for its j-th tap.
            if self.shift_only:
                xj = x.roll(-(j * self.dilation) % self.in_features, dims=-1)
            else:
                xj = x[..., self.taps[j]]
            wj = self.synaptic[:, j]
            acc = wj * xj if acc is None else acc + wj * xj
        return acc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)"""
        d = self.dendrites(x)
        if self.dendrite_bias is not None:
            d = d + self.dendrite_bias
        d = F.leaky_relu(d, 0.1)

        # Each soma owns D consecutive dendrites -> reshape, not gather.
        d = d.unflatten(-1, (self.out_features, self.D))
        s = (d * self.cable).sum(-1)
        if self.soma_bias is not None:
            s = s + self.soma_bias
        return s

    # ------------------------------------------------------------------
    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Same layer written as two explicit masked dense matmuls.

        Only for verifying correctness — it materializes the full (M, N)
        weight matrix that the real forward avoids, and does ~1/sparsity
        times more arithmetic.
        """
        W = x.new_zeros(self.M, self.in_features)
        rows = torch.arange(self.M, device=x.device).expand(self.K, self.M)
        # accumulate=True so a dendrite that taps the same input twice
        # (possible when K * dilation > N) sums rather than overwrites
        W = W.index_put(
            (rows.reshape(-1), self.taps.reshape(-1)),
            self.synaptic.t().reshape(-1),
            accumulate=True,
        )

        d = F.linear(x, W, self.dendrite_bias)
        d = F.leaky_relu(d, 0.1)

        C = x.new_zeros(self.out_features, self.M)
        for s in range(self.out_features):
            C[s, s * self.D:(s + 1) * self.D] = self.cable[s]
        return F.linear(d, C, self.soma_bias)

    # ------------------------------------------------------------------
    def sparsity(self) -> dict:
        """Multiply-adds per input vector, against a dense nn.Linear."""
        ours = self.M * self.K + self.M
        dense = self.in_features * self.out_features
        return {
            "dendrites": self.M,
            "macs": ours,
            "dense_macs": dense,
            "fraction_of_dense": ours / dense,
            "reads_per_input": self.M * self.K / self.in_features,
        }

    def extra_repr(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"K={self.K}, D={self.D}, M={self.M}, dilation={self.dilation}, "
            f"total_params={p}"
        )
