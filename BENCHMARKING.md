# Benchmarking notes

Things that silently produced wrong numbers in this repo. All of these looked like
plausible results at the time.

## torch.compile fails silently — verify, don't assume

When compilation doesn't happen, PyTorch runs eager and says nothing. The result is a
benchmark row that looks real but measures the wrong thing.

**The tell: compiled and eager timings are identical, including peak memory.**

```
  neighbors x3    4.730ms   12.763ms   1594MB
    ^ compiled    4.708ms   12.764ms   1594MB      <- not compiled
```

Timings can coincide; peak memory to the megabyte cannot. Check that column first.

Verify directly instead of inferring from timings:

```python
import torch._dynamo as dynamo
explanation = dynamo.explain(model)(example_input)
assert explanation.graph_break_count == 0, explanation.break_reasons
```

### Cause 1: reading a Python value out of a tensor inside `forward`

```python
off = int(self.offsets[t])        # graph break, every iteration
h.roll(-off, dims=-1)
```

Anything that forces a tensor to a Python scalar (`int()`, `.item()`, `if tensor:`,
using a tensor element as a shape or a loop bound) breaks the graph. Resolve such
values in `__init__` and store them as plain Python ints/tuples, not buffers.

### Cause 2: hitting the recompile limit

The default `recompile_limit` is 8, and it is **per code object**, so every instance of
a module shares the budget. A benchmark that constructs the same class a dozen times
will exhaust it and fall back to eager partway through — the earlier rows are real and
the later ones are not, which is worse than all of them being wrong.

Toggling grad mode counts as a recompile, so a harness that times `no_grad` forward and
then forward+backward burns two per model.

```python
torch._dynamo.config.recompile_limit = 64
...
torch._dynamo.reset()             # between modules
model = torch.compile(model)
```

## Compile the baseline too

Comparing a compiled variant against an uncompiled baseline overstates every speedup —
inductor speeds up the baseline as well. This inflated a "20x faster" claim here until
both sides were compiled.

Time the baseline both ways and score each row against its like: eager rows against the
eager baseline, compiled rows against the compiled one.

## Batch size decides the conclusion, not just the magnitude

At small batches everything is bound by kernel launch latency and the numbers say
nothing about the design. In this repo, a layer that looked ~1x at batch 1024 was 12x
at batch 16384, and a sweep over dendrite counts was completely flat until the GPU was
saturated.

Benchmark at a batch that saturates the device, and treat small-batch results as
measuring overhead rather than the idea.

## Include an exact control

Every variant here has a setting (`extra_stages=0`, `extra_branches=0`, `neighbors=0`)
that is *literally* a plain `Linear + leaky_relu` with no extra parameters. That row
should land on the baseline; when it doesn't, the gap is harness noise and every other
row needs reading with that in mind.

Prefer a control that is exactly the baseline over one that is merely equivalent after
folding weights — the folded version carries dead parameters and muddies param counts.

## Fusion favours parallel over sequential

Sequential per-neuron stages form a dependency chain, so the backward is a chain of
serial kernels. Independent branches reading the same input fuse into one pass. Same
parameter count, same arithmetic:

```
staged   x7 compiled   0.98x fwd   1.11x fwd+bwd   1736MB
branched x7 compiled   0.99x fwd   1.01x fwd+bwd    923MB
```

If a design admits a parallel formulation, prefer it.

## Correctness first, and check coverage explicitly

Two bugs here made the model quietly do less work than intended, both of which showed
up as *good* benchmark numbers:

- a shared tiling meant every unit read the same first 64 inputs, so 97% of the input
  had no path to the output
- a padding bug put an entire group of units in a discarded tail

Neither is visible in timings. Assert reachability directly — every input's gradient
should be non-zero — and run the correctness test before believing any speed result.
