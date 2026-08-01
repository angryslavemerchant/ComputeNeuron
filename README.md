# ComputeNeuron

Experiments in giving individual artificial neurons more internal computation than a
weighted sum, and checking whether the extra structure pays for its cost.

## DendriticLinear

A sparse two-stage layer:

```
N inputs  ->  M = S*D dendrites  ->  leaky_relu  ->  S soma
```

Each dendrite reads `K` inputs (default 16) and applies a nonlinearity. Each soma
takes a weighted sum of its **own** `D` dendrites — dendrites are private to their
soma. The shape-matched dense equivalent is `Linear(N, M) -> leaky_relu -> Linear(M, S)`:
identical layer sizes and nonlinearity, but everything fully connected.

```python
from dendritic_linear import DendriticLinear

layer = DendriticLinear(1024, 256, fan_in=16, dendrites_per_soma=4)
# 1024 inputs -> 1024 dendrites -> 256 soma, 18.7K params
```

Size it with `dendrites_per_soma`. `D` and `K` together decide how much of the input
each soma sees: `D*K` out of `N`. At D=4, K=16, N=1024 that's 64 of 1024 — genuinely
sparse. Setting `D = N/K` makes every soma read the whole input, which costs as much
as a dense layer while executing far slower. That's what the legacy `coverage=1.0`
does and it is almost never what you want.

### Results

On an RTX 5090, fp32, at 2048 → 2048 hidden → 512, batch 16384:

| | params | MACs/vector | fwd | fwd+bwd |
|---|---|---|---|---|
| dense `2048->2048->512` | 5.25M | 5.24M | 3.410 ms | 10.085 ms |
| dendritic | 37.4K | 34.8K | 0.665 ms | 2.962 ms |
| dendritic + `torch.compile` | 37.4K | 34.8K | **0.175 ms** | **2.100 ms** |

**20x faster forward, 4.8x faster training step, 140x fewer parameters**, and less
peak memory than dense (772MB vs 931MB).

Two caveats worth knowing:

- **Batch size decides everything.** At batch 1024 the same comparison is roughly
  break-even, because every model is bound by kernel launch latency rather than
  arithmetic. The advantage only appears once the GPU is saturated.
- **The backward pass is the weak point.** Dense does 150x more arithmetic but the
  training step is only 4.8x faster, because the weight gradient is a batch reduction
  rather than the single cuBLAS GEMM a dense layer gets. There is likely another
  2-3x available there.

No accuracy results yet — the speed case is established, the quality case is not.

## Contents

| File | What it is |
|---|---|
| `dendritic_linear.py` | `DendriticLinear`, plus `forward_reference` (the slow gather formulation) and `sparsity()` |
| `bench.py` | Shape-matched dense vs dendritic benchmark |
| `test_dendritic_equiv.py` | Verifies the fast forward matches the reference on values and gradients |
| `mgn.py` | Multi-gate neuron (MGN) layers, v1–v4: each neuron mixes SUM / AND / OR reductions with a learned per-neuron softmax gate |
| `test_mgn.py` | Tests for the MGN layers |
| `multi-gate-neuron-spec.md` | Design spec for the MGN family |
| `new_neuron_guide.md` | Notes on adding a new neuron type |

## Running

```bash
python bench.py                  # cuda if available
python bench.py --no-compile
python bench.py --fan-in 32
python test_dendritic_equiv.py
```

Requires PyTorch. No other dependencies.
