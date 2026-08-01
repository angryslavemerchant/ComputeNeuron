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

Since one soma only sees `D*K` inputs, soma are split into `G = ceil(N / (D*K))`
groups and each group reads a different window of the input, so collectively they
cover everything. `layer.sparsity()["inputs_covered"]` reports this, and the tests
assert that every input's gradient is non-zero.

### Results

**These numbers predate the input-coverage fix** — they were measured when every soma
read the same first `D*K` inputs, so the layer was doing less work than intended. They
need a rerun.

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

## StagedLinear

A different bet: instead of sparsifying the connections, keep the dense matmul and
give each neuron a learnable activation shape.

```
z  = W x + b
z <- leaky_relu(a_i * z + c_i)     for i in 1..stages
```

`a_i` and `c_i` are one scalar per neuron, so neuron *j* in each stage reads only
neuron *j* below it. All mixing happens in the single matmul; the stages only bend
each neuron's response curve, one extra bend per stage.

```python
from staged_linear import StagedLinear

layer = StagedLinear(2048, 512, stages=2)
```

`stages=1` is the **control**, not a setting: `phi(a*(Wx+b)+c)` is affine inside the
nonlinearity, so it is exactly `nn.Linear` + `leaky_relu` with rescaled weights.
`stages=2` is the smallest setting that adds anything.

Stages are nearly free — 2·`out_features` parameters and one elementwise pass each,
against a matmul of `in_features`·`out_features`. At 2048 → 512 that's 1024 params on
1.05M, ~0.1%.

By default each stage uses a negative slope of `negative_slope**(1/stages)`, so the
composite slope is 0.1 regardless of depth. Without that, stacking leaky_relu decays
the negative slope geometrically (0.1 → 0.01 → 0.001) and depths would start from
different functions, confounding the comparison.

```bash
python bench_staged.py --stages 1,2,4,8
python test_staged_linear.py
```

## Contents

| File | What it is |
|---|---|
| `dendritic_linear.py` | `DendriticLinear` and `DendriticMLP`, plus `forward_reference` and `sparsity()` |
| `bench.py` | Shape-matched dense vs dendritic benchmark (`--sweep`, `--shape`, `--readout`) |
| `test_dendritic_equiv.py` | Forward matches the reference; every input reaches the output |
| `staged_linear.py` | `StagedLinear` — dense matmul, per-neuron learnable activation |
| `bench_staged.py` | `StagedLinear` vs plain `Linear + leaky_relu` |
| `test_staged_linear.py` | stages=1 folds to a plain layer; slope correction holds at depth |
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
