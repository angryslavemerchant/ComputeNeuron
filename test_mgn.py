"""Sanity checks for mgn.py, mirroring VectorMLP/tests/sanity_check.py:
shapes, gradients, pure-SUM equivalence to nn.Linear, AND/OR semantics,
stack_module_state + vmap(grad(...)) compatibility, and a tiny overfit run."""

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, stack_module_state, vmap

from mgn import (MGNLinear, MGNNet, MGNv2Linear, MGNv2Net,
                 MGNv3Linear, MGNv3Net, MGNv4Linear, MGNv4Net)

torch.manual_seed(0)
B, N_IN, N_OUT = 4, 12, 8


def check(name, ok, detail=''):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    assert ok, name


# --- shapes + gradients, both affine variants ---
x = torch.randn(B, N_IN)
for affine in (True, False):
    layer = MGNLinear(N_IN, N_OUT, path_affine=affine)
    y = layer(x)
    check(f'layer shape affine={affine}', y.shape == (B, N_OUT))
    y.sum().backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in layer.parameters())
    check(f'layer grads affine={affine}', grads_ok)

net = MGNNet(N_IN, [16, 16], 10)
logits = net(x)
check('net shape', logits.shape == (B, 10))
logits.sum().backward()
check('net grads', all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in net.parameters()))

# --- gate forced to SUM must reproduce nn.Linear exactly ---
layer = MGNLinear(N_IN, N_OUT, path_affine=False)
with torch.no_grad():
    layer.mix_logits[:, 0] = 100.0
    layer.mix_logits[:, 1:] = 0.0
expected = F.linear(x, layer.linear.weight, layer.linear.bias)
err = (layer(x) - expected).abs().max().item()
check('pure-SUM == nn.Linear', err < 1e-5, f'max err={err:.2e}')

# --- AND semantics: high only when ALL weighted inputs high ---
layer = MGNLinear(4, 1, path_affine=False)
with torch.no_grad():
    layer.linear.weight.fill_(1.0)
    layer.mix_logits[:, 1] = 100.0
hi = layer(torch.full((1, 4), 5.0)).item()
one_low = layer(torch.tensor([[5.0, 5.0, 5.0, -5.0]])).item()
lo = layer(torch.full((1, 4), -5.0)).item()
check('AND semantics', hi > 0.9 and one_low < 0.4 and lo < 0.05,
      f'all-hi={hi:.3f} one-low={one_low:.3f} all-lo={lo:.3f}')

# --- OR semantics: smooth max of weighted inputs ---
layer = MGNLinear(4, 1, tau_init=10.0, path_affine=False)
with torch.no_grad():
    layer.linear.weight.fill_(1.0)
    layer.mix_logits[:, 2] = 100.0
y = layer(torch.tensor([[3.0, -1.0, 0.5, -2.0]])).item()
check('OR semantics ~ max', abs(y - 3.0) < 0.2, f'or={y:.3f} vs max=3.0')

# --- numerics at extreme input scales ---
layer = MGNLinear(N_IN, N_OUT)
ok = True
for scale in (1e3, 1e-6):
    out = layer(torch.randn(B, N_IN) * scale)
    ok = ok and torch.isfinite(out).all().item()
check('extreme-scale stability', ok)

# --- harness compatibility: stack_module_state + vmap(grad(...)) ---
models = [MGNNet(N_IN, [16], 10) for _ in range(3)]
params, buffers = stack_module_state(models)
base = models[0].to('meta')

def loss_fn(p, b, xs, ys):
    return F.cross_entropy(functional_call(base, (p, b), (xs,)), ys)

xs = torch.randn(3, B, N_IN)
ys = torch.randint(0, 10, (3, B))
grads = vmap(grad(loss_fn))(params, buffers, xs, ys)
check('vmap(grad) over stacked models',
      all(torch.isfinite(g).all() for g in grads.values()))

# --- param count: everything trainable and countable ---
net = MGNNet(64, [32, 32], 10)
n = sum(p.numel() for p in net.parameters() if p.requires_grad)
expected = sum(  # per layer: linear + 3 mix + tau + 4 affine
    (a * b + b) + 3 * b + b + 4 * b
    for a, b in ((64, 32), (32, 32))) + (32 * 10 + 10)
check('param count', n == expected, f'{n} vs {expected}')

# --- tiny overfit: memorize 64 random samples ---
def overfit(model, steps=400):
    xs = torch.randn(64, 64)
    ys = torch.randint(0, 10, (64,))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(xs), ys)
        loss.backward()
        opt.step()
    return (model(xs).argmax(-1) == ys).float().mean().item()

torch.manual_seed(0)
acc = overfit(MGNNet(64, [32, 32], 10))
check('mgn net overfits', acc > 0.95, f'acc={acc:.2f}')


# ===================== v2 (matmul-native) checks =====================

# --- shapes + gradients ---
x = torch.randn(B, N_IN)
for affine in (True, False):
    layer = MGNv2Linear(N_IN, N_OUT, path_affine=affine)
    y = layer(x)
    check(f'v2 layer shape affine={affine}', y.shape == (B, N_OUT))
    y.sum().backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in layer.parameters())
    check(f'v2 layer grads affine={affine}', grads_ok)

# --- gate forced to SUM must reproduce nn.Linear exactly ---
layer = MGNv2Linear(N_IN, N_OUT, path_affine=False)
with torch.no_grad():
    layer.mix_logits[:, 0] = 100.0
    layer.mix_logits[:, 1:] = 0.0
expected = F.linear(x, layer.linear.weight, layer.linear.bias)
err = (layer(x) - expected).abs().max().item()
check('v2 pure-SUM == nn.Linear', err < 1e-5, f'max err={err:.2e}')

# --- AND semantics (weights=1: same geometric mean as v1) ---
layer = MGNv2Linear(4, 1, path_affine=False)
with torch.no_grad():
    layer.linear.weight.fill_(1.0)
    layer.mix_logits[:, 1] = 100.0
hi = layer(torch.full((1, 4), 5.0)).item()
one_low = layer(torch.tensor([[5.0, 5.0, 5.0, -5.0]])).item()
lo = layer(torch.full((1, 4), -5.0)).item()
check('v2 AND semantics', hi > 0.9 and one_low < 0.4 and lo < 0.05,
      f'all-hi={hi:.3f} one-low={one_low:.3f} all-lo={lo:.3f}')

# --- OR semantics (noisy-OR: any strong input suffices) ---
layer = MGNv2Linear(4, 1, path_affine=False)
with torch.no_grad():
    layer.linear.weight.fill_(1.0)
    layer.mix_logits[:, 2] = 100.0
all_off = layer(torch.full((1, 4), -5.0)).item()
one_on = layer(torch.tensor([[5.0, -5.0, -5.0, -5.0]])).item()
all_on = layer(torch.full((1, 4), 5.0)).item()
check('v2 OR semantics', all_off < 0.05 and one_on > 0.5 and all_on > 0.9,
      f'all-off={all_off:.3f} one-on={one_on:.3f} all-on={all_on:.3f}')

# --- NOT via negative weight: AND path fires when input is off ---
layer = MGNv2Linear(1, 1, path_affine=False)
with torch.no_grad():
    layer.linear.weight.fill_(-1.0)
    layer.mix_logits[:, 1] = 100.0
off = layer(torch.tensor([[-5.0]])).item()   # input off -> 1/p large
on = layer(torch.tensor([[5.0]])).item()     # input on  -> 1/p ~ 1
check('v2 NOT via negative weight', off > 2.0 and 0.9 < on < 1.2,
      f'off={off:.2f} on={on:.2f}')

# --- numerics at extreme input scales (exp clamp) ---
layer = MGNv2Linear(N_IN, N_OUT)
ok = True
for scale in (1e3, 1e-6):
    out = layer(torch.randn(B, N_IN) * scale)
    ok = ok and torch.isfinite(out).all().item()
check('v2 extreme-scale stability', ok)

# --- harness compatibility: stack_module_state + vmap(grad(...)) ---
models = [MGNv2Net(N_IN, [16], 10) for _ in range(3)]
params, buffers = stack_module_state(models)
base2 = models[0].to('meta')

def loss_fn2(p, b, xs, ys):
    return F.cross_entropy(functional_call(base2, (p, b), (xs,)), ys)

xs = torch.randn(3, B, N_IN)
ys = torch.randint(0, 10, (3, B))
grads = vmap(grad(loss_fn2))(params, buffers, xs, ys)
check('v2 vmap(grad) over stacked models',
      all(torch.isfinite(g).all() for g in grads.values()))

# --- param count ---
net = MGNv2Net(64, [32, 32], 10)
n = sum(p.numel() for p in net.parameters() if p.requires_grad)
expected = sum(  # per layer: linear + in-affine + 3 mix + 4 path-affine
    (a * b + b) + 2 * a + 3 * b + 4 * b
    for a, b in ((64, 32), (32, 32))) + (32 * 10 + 10)
check('v2 param count', n == expected, f'{n} vs {expected}')

# --- tiny overfit ---
torch.manual_seed(0)
acc = overfit(MGNv2Net(64, [32, 32], 10))
check('v2 net overfits', acc > 0.95, f'acc={acc:.2f}')


# ============ v3 (soft-max / soft-min, fan-in robust) ============

x = torch.randn(B, N_IN)
for affine in (True, False):
    layer = MGNv3Linear(N_IN, N_OUT, path_affine=affine)
    y = layer(x)
    check(f'v3 layer shape affine={affine}', y.shape == (B, N_OUT))
    y.sum().backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in layer.parameters())
    check(f'v3 layer grads affine={affine}', grads_ok)

# --- gate forced to SUM must reproduce nn.Linear exactly ---
layer = MGNv3Linear(N_IN, N_OUT, path_affine=False)
with torch.no_grad():
    layer.mix_logits[:, 0] = 100.0
    layer.mix_logits[:, 1:] = 0.0
expected = F.linear(x, layer.linear.weight, layer.linear.bias)
err = (layer(x) - expected).abs().max().item()
check('v3 pure-SUM == nn.Linear', err < 1e-5, f'max err={err:.2e}')


def v3_logic(n_in, tau=5.0):
    """Pure logic neuron: unit weights, inputs +-5. Returns AND/OR gaps."""
    layer = MGNv3Linear(n_in, 1, tau_init=tau, path_affine=False)
    with torch.no_grad():
        layer.linear.weight.fill_(1.0)

    def path(idx, x):
        with torch.no_grad():
            layer.mix_logits.zero_()
            layer.mix_logits[:, idx] = 100.0
            return layer(x).item()

    on, off = torch.full((1, n_in), 5.0), torch.full((1, n_in), -5.0)
    one_off = torch.cat([torch.full((1, n_in - 1), 5.0),
                         torch.full((1, 1), -5.0)], -1)
    one_on = torch.cat([torch.full((1, n_in - 1), -5.0),
                        torch.full((1, 1), 5.0)], -1)
    return (path(1, on) - path(1, one_off),      # AND gap
            path(2, one_on) - path(2, off))      # OR gap


# --- the property v2 lacks: logic still works at large fan-in ---
gaps = {n: v3_logic(n) for n in (4, 64, 256, 1024)}
for n, (ga, go) in gaps.items():
    check(f'v3 logic gap n={n}', ga > 1.0 and go > 1.0,
          f'AND gap={ga:.3f} OR gap={go:.3f}')
ratio = gaps[1024][1] / gaps[4][1]
check('v3 OR gap stable across fan-in', ratio > 0.5,
      f'gap(1024)/gap(4)={ratio:.3f}')

# --- OR tracks the max, AND tracks the min ---
layer = MGNv3Linear(4, 1, tau_init=20.0, path_affine=False)
with torch.no_grad():
    layer.linear.weight.fill_(1.0)
    layer.mix_logits[:, 2] = 100.0
xs = torch.tensor([[3.0, -1.0, 0.5, -2.0]])
check('v3 OR ~ max', abs(layer(xs).item() - 3.0) < 0.3,
      f'or={layer(xs).item():.3f} vs max=3.0')
with torch.no_grad():
    layer.mix_logits.zero_()
    layer.mix_logits[:, 1] = 100.0
check('v3 AND ~ min', abs(layer(xs).item() + 2.0) < 0.3,
      f'and={layer(xs).item():.3f} vs min=-2.0')

# --- numerics: exp must not overflow at any input scale ---
layer = MGNv3Linear(N_IN, N_OUT)
ok = True
for scale in (1e-6, 1.0, 1e3, 1e5):
    out = layer(torch.randn(B, N_IN) * scale)
    ok = ok and torch.isfinite(out).all().item()
check('v3 extreme-scale stability', ok)

# --- harness compatibility ---
models = [MGNv3Net(N_IN, [16], 10) for _ in range(3)]
params, buffers = stack_module_state(models)
base3 = models[0].to('meta')

def loss_fn3(p, b, xs, ys):
    return F.cross_entropy(functional_call(base3, (p, b), (xs,)), ys)

xs = torch.randn(3, B, N_IN)
ys = torch.randint(0, 10, (3, B))
grads = vmap(grad(loss_fn3))(params, buffers, xs, ys)
check('v3 vmap(grad) over stacked models',
      all(torch.isfinite(g).all() for g in grads.values()))

# --- param count: linear + in-affine + 3 mix + 4 path-affine + 1 tau ---
net = MGNv3Net(64, [32, 32], 10)
n = sum(p.numel() for p in net.parameters() if p.requires_grad)
expected = sum((a * b + b) + 2 * a + 3 * b + 4 * b + 1
               for a, b in ((64, 32), (32, 32))) + (32 * 10 + 10)
check('v3 param count', n == expected, f'{n} vs {expected}')

torch.manual_seed(0)
acc = overfit(MGNv3Net(64, [32, 32], 10))
check('v3 net overfits', acc > 0.95, f'acc={acc:.2f}')


# ============ v4 (project to k features, then reduce) ============

x = torch.randn(B, N_IN)
for affine in (True, False):
    layer = MGNv4Linear(N_IN, N_OUT, k=4, path_affine=affine)
    y = layer(x)
    check(f'v4 layer shape affine={affine}', y.shape == (B, N_OUT))
    y.sum().backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in layer.parameters())
    check(f'v4 layer grads affine={affine}', grads_ok)

check('v4 k defaults to dim', MGNv4Linear(8, 4, dim=6).k == 6)

# --- gate forced to SUM must reproduce nn.Linear exactly ---
layer = MGNv4Linear(N_IN, N_OUT, k=4, path_affine=False)
with torch.no_grad():
    layer.mix_logits[:, 0] = 100.0
    layer.mix_logits[:, 1:] = 0.0
expected = F.linear(x, layer.linear.weight, layer.linear.bias)
err = (layer(x) - expected).abs().max().item()
check('v4 pure-SUM == nn.Linear', err < 1e-5, f'max err={err:.2e}')


def v4_logic(n_in, k=4, tau=5.0):
    """One neuron whose k features read the first k inputs directly."""
    layer = MGNv4Linear(n_in, 1, k=k, tau_init=tau, path_affine=False)
    with torch.no_grad():
        layer.proj.weight.zero_()
        layer.proj.bias.zero_()
        for j in range(k):                     # feature j = input j
            layer.proj.weight[j, j] = 1.0

    def path(idx, x):
        with torch.no_grad():
            layer.mix_logits.zero_()
            layer.mix_logits[:, idx] = 100.0
            return layer(x).item()

    def vec(vals):
        x = torch.zeros(1, n_in)
        x[0, :k] = torch.tensor(vals)
        return x

    on, off = vec([5.0] * k), vec([-5.0] * k)
    one_off = vec([5.0] * (k - 1) + [-5.0])
    one_on = vec([-5.0] * (k - 1) + [5.0])
    return (path(1, on) - path(1, one_off),    # AND gap
            path(2, one_on) - path(2, off))    # OR gap


# --- the point of v4: logic quality is set by k, NOT by layer width ---
gaps = {n: v4_logic(n) for n in (8, 128, 1024)}
for n, (ga, go) in gaps.items():
    check(f'v4 logic gap n_in={n}', ga > 0.5 and go > 1.0,
          f'AND gap={ga:.3f} OR gap={go:.3f}')
spread = max(abs(gaps[1024][i] - gaps[8][i]) for i in (0, 1))
check('v4 logic independent of fan-in', spread < 1e-4, f'spread={spread:.2e}')

# --- numerics ---
layer = MGNv4Linear(N_IN, N_OUT, k=4)
ok = True
for scale in (1e-6, 1.0, 1e3, 1e5):
    ok = ok and torch.isfinite(layer(torch.randn(B, N_IN) * scale)).all().item()
check('v4 extreme-scale stability', ok)

# --- harness compatibility ---
models = [MGNv4Net(N_IN, [16], 10, 4) for _ in range(3)]
params, buffers = stack_module_state(models)
base4 = models[0].to('meta')

def loss_fn4(p, b, xs, ys):
    return F.cross_entropy(functional_call(base4, (p, b), (xs,)), ys)

xs = torch.randn(3, B, N_IN)
ys = torch.randint(0, 10, (3, B))
grads = vmap(grad(loss_fn4))(params, buffers, xs, ys)
check('v4 vmap(grad) over stacked models',
      all(torch.isfinite(g).all() for g in grads.values()))

# --- param count: sum linear + k-way proj + tau + 3 mix + 4 affine ---
K = 4
net = MGNv4Net(64, [32, 32], 10, K)
n = sum(p.numel() for p in net.parameters() if p.requires_grad)
expected = sum((a * b + b) + (a * b * K + b * K) + b + 3 * b + 4 * b
               for a, b in ((64, 32), (32, 32))) + (32 * 10 + 10)
check('v4 param count', n == expected, f'{n} vs {expected}')

torch.manual_seed(0)
acc = overfit(MGNv4Net(64, [32, 32], 10, 4))
check('v4 net overfits', acc > 0.95, f'acc={acc:.2f}')

print('\nall checks passed')
