# Multi-Gate Neuron (MGN)

A neuron primitive that computes three parallel reduction operations over shared weighted inputs and learns a per-neuron mixture across them.

---

## Motivation

A standard neuron computes `f(sum(w * x) + b)` — accumulation followed by a nonlinearity. This can only approximate conjunction (AND) and disjunction (OR) behavior indirectly through learned bias and weight magnitudes, wasting capacity.

The MGN gives each neuron native access to three fundamental operations — accumulation, conjunction, and disjunction — over the same weighted inputs, at a cost of 3 extra scalars per neuron.

---

## Forward Pass

Given:
- `x ∈ R^n` — input vector
- `w ∈ R^n` — weight vector (shared across all three paths)
- `b ∈ R` — bias (standard)
- `α, β, γ ∈ R` — learnable mixture scalars (per neuron)

### Step 1: Weighted inputs (shared)

```
z = w * x        # element-wise, shape [n]
```

### Step 2: Three parallel reductions

**SUM path (accumulation):**
```
s_sum = sum(z) + b
```
Standard neuron behavior. "Is there enough total input?"

**AND path (conjunction):**
```
s_and = exp(mean(log(sigmoid(z) + ε)))
```
Geometric mean of sigmoid-squashed weighted inputs. Collapses toward 0 if any input drives its sigmoid low. Computed in log-space for numerical stability.

Equivalent to: `prod(sigmoid(z))^(1/n)` but stable.

Why geometric mean instead of raw product: raw product of n terms in [0,1] vanishes exponentially with n. The geometric mean (nth root of product) normalizes this so the output stays in a usable range regardless of the number of inputs.

**OR path (disjunction):**
```
s_or = (1/τ) * logsumexp(τ * z)
```
Smooth approximation to `max(z)`. Temperature `τ` controls hardness: τ→∞ recovers hard max, τ=1 is standard logsumexp. `τ` can be fixed (e.g. τ=5) or learnable per neuron (+1 param).

### Step 3: Mixture

```
gate = softmax([α, β, γ])          # normalized to sum to 1
output = gate[0] * s_sum + gate[1] * s_and + gate[2] * s_or
```

Softmax over the mixture scalars ensures the output is a convex combination. The neuron learns which mode(s) to rely on.

### Step 4: Activation

```
y = activation(output)              # e.g. ReLU, GELU, identity
```

Standard nonlinearity applied after mixing. Choice is orthogonal to the MGN design.

---

## Parameter Count

Per neuron:
| Component     | Parameters | Notes                        |
|---------------|------------|------------------------------|
| `w`           | n          | same as standard neuron      |
| `b`           | 1          | same as standard neuron      |
| `α, β, γ`    | 3          | mixture logits               |
| `τ` (optional)| 1          | OR path temperature          |

**Overhead vs standard neuron: 3 scalars (or 4 with learnable τ).**

For a layer with `m` neurons and `n` inputs: standard layer has `m * (n + 1)` params, MGN layer has `m * (n + 4)` params. For any reasonable n (e.g. 256+), the overhead is negligible.

---

## Compute Cost

Per neuron, compared to a standard neuron:

| Operation       | Standard | MGN      |
|-----------------|----------|----------|
| Weighted inputs | n muls   | n muls (shared) |
| SUM reduction   | n adds   | n adds   |
| AND reduction   | —        | n sigmoids + n logs + mean + exp |
| OR reduction    | —        | n muls (scale by τ) + logsumexp |
| Mixture         | —        | softmax(3) + 2 muls + 2 adds |

Roughly **3x the reduction compute** of a standard neuron, but the expensive part (the weight-input matmul in a full layer) is shared and computed only once. In a matrix-multiply-bound regime (GPU, large layers), the reduction overhead is likely hidden behind memory bandwidth.

---

## Layer-Level Implementation (PyTorch sketch)

```python
class MGNLinear(nn.Module):
    def __init__(self, in_features, out_features, tau_init=5.0, eps=1e-7):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

        # mixture logits: [out_features, 3]
        self.mix_logits = nn.Parameter(torch.zeros(out_features, 3))

        # OR path temperature
        self.log_tau = nn.Parameter(torch.full((out_features,), math.log(tau_init)))

        self.eps = eps

    def forward(self, x):
        # x: [batch, in_features]
        # Shared weighted inputs: [batch, out_features, in_features]
        # We need per-neuron element-wise products, not just the sum.
        # So we expand and multiply manually:
        w = self.linear.weight                    # [out, in]
        b = self.linear.bias                      # [out]

        z = x.unsqueeze(1) * w.unsqueeze(0)       # [batch, out, in]

        # --- SUM path ---
        s_sum = z.sum(dim=-1) + b                  # [batch, out]

        # --- AND path (geometric mean of sigmoid) ---
        sig_z = torch.sigmoid(z)                   # [batch, out, in]
        log_sig = torch.log(sig_z + self.eps)      # [batch, out, in]
        s_and = torch.exp(log_sig.mean(dim=-1))    # [batch, out]

        # --- OR path (smooth max) ---
        tau = self.log_tau.exp().unsqueeze(0)      # [1, out]
        # logsumexp over input dim, scaled by tau
        s_or = (1.0 / tau) * torch.logsumexp(
            tau.unsqueeze(-1) * z, dim=-1          # [batch, out]
        )

        # --- Mixture ---
        gate = F.softmax(self.mix_logits, dim=-1)  # [out, 3]
        output = (
            gate[:, 0] * s_sum +
            gate[:, 1] * s_and +
            gate[:, 2] * s_or
        )                                          # [batch, out]

        return output
```

**Note on the matmul:** The naive implementation above expands to `[batch, out, in]` which is memory-expensive. For production use, you'd compute `s_sum` via standard `F.linear` and only expand for the AND/OR paths, or find a fused kernel. The SUM path is just `F.linear(x, w, b)` — no expansion needed.

---

## Initialization

- `w, b`: standard init (Kaiming, Xavier, etc.)
- `α, β, γ`: all zeros → softmax gives [1/3, 1/3, 1/3], equal mixture at init. Alternatively, bias toward SUM at init (e.g. `α=1, β=0, γ=0`) so the network starts as a standard network and gradually discovers AND/OR if useful.
- `τ`: init to 5.0 (moderately sharp max approximation). Too low → OR path degenerates to mean. Too high → hard max with sparse gradients.
- `ε`: 1e-7, numerical stability only.

---

## Expected Behavior During Training

- **If the task is purely linear/accumulative:** mixture collapses to α-dominant, MGN reduces to standard neuron. No harm done.
- **If the task needs feature co-occurrence detection:** AND-dominant neurons should emerge (e.g. "this is a cat" requires ears AND whiskers AND fur texture all present).
- **If the task needs any-feature-sufficient detection:** OR-dominant neurons should emerge (e.g. "this is a vehicle" if wheels OR wings OR hull).
- **Mixed neurons:** some neurons may learn non-trivial mixtures, capturing "mostly AND but with an OR fallback."

---

## Key Risks and Mitigations

**AND path gradient vanishing:**
Any input driving sigmoid(z) near 0 suppresses the entire AND output. Gradient through other inputs in that path vanishes.
*Mitigation:* geometric mean (instead of raw product) limits severity. Log-space computation keeps numerics clean. If AND paths still struggle, try `softplus` instead of `sigmoid` as the squashing function.

**OR path gradient sparsity:**
With high τ, logsumexp approximates hard max and only the winning input gets meaningful gradient.
*Mitigation:* moderate τ init (5.0), or learnable τ that starts moderate. Straight-through estimator as fallback.

**Mixture collapse:**
All neurons converge to the same gate pattern (e.g. all SUM-dominant).
*Mitigation:* monitor gate entropy during training. If collapsing, add a small entropy bonus to the mixture logits, or initialize different neurons with different biases.

**Memory cost of expansion:**
The `[batch, out, in]` tensor for AND/OR paths is large.
*Mitigation:* chunk over the output dimension, or only apply MGN to specific layers (e.g. the final classifier, or bottleneck layers where `in` and `out` are small).

---

## Minimal Experiment

Cheapest validation: MNIST or CIFAR-10 with a small MLP.

1. **Baseline:** 2-layer MLP, 256 hidden units, ReLU. Standard training.
2. **MGN:** Same architecture, replace `nn.Linear` with `MGNLinear`. Same training.
3. **Metrics:** test accuracy, parameter count, and — critically — **inspect the learned gate distributions.** Histogram of softmax(α,β,γ) across all neurons. If every neuron learns [1, 0, 0] (pure SUM), the AND/OR paths aren't useful for this task. If there's diversity, something interesting is happening.
4. **Ablation:** MGN but fix gate to pure SUM (should recover baseline), pure AND, pure OR. See which degrades least / most.
