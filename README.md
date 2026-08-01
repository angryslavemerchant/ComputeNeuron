# ComputeNeuron

Experiments in giving individual artificial neurons more internal computation than a
weighted sum, and checking whether the extra structure pays for its cost.

## Contents

| File | What it is |
|---|---|
| `dendritic_linear.py` | `DendriticLinear` — an `nn.Linear`/FFN replacement where each output neuron owns `D` dendrites that each gather `K` inputs, apply a nonlinearity, and are summed by the soma. |
| `bench_dendritic.py` | Speed/memory benchmark: `DendriticLinear` vs `nn.Linear` and 2-layer FFN blocks. |
| `mgn.py` | Multi-gate neuron (MGN) layers, v1–v4. Each neuron mixes SUM / AND / OR reductions with a learned per-neuron softmax gate. |
| `test_mgn.py` | Tests for the MGN layers. |
| `multi-gate-neuron-spec.md` | Design spec for the MGN family. |
| `new_neuron_guide.md` | Notes on adding a new neuron type. |

## Running the benchmark

```bash
python bench_dendritic.py --device cuda          # or --device cpu
python bench_dendritic.py --device cuda --dtype fp16
```

For each `(tokens, d_model)` shape it reports parameter count, forward latency,
forward+backward latency, and peak CUDA memory for:

- `nn.Linear(d, d)` — the single-GEMM speed floor
- `FFN d -> 4d -> d` (GELU) — the transformer block `DendriticLinear` aims to replace
- `FFN d -> d -> d` (GELU) — a cheaper two-layer block
- `DendriticLinear(d, d)` across several `fan_in` / `coverage` settings

`fan_in` accepts a float (fraction of `in_features`) or an int (absolute feature count).

Timing uses warmup plus an auto-scaled iteration count with `torch.cuda.synchronize()`
around the measured region, so GPU numbers reflect real execution rather than async
kernel launches.

## Requirements

PyTorch. No other dependencies.
