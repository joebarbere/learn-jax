"""Module 31b helpers: the models the lesson trains, built from YOUR functions.

The MDRNNs call your `mdrnn_sweep` and `four_direction_sweeps`, the linear scan your `cross_scan` and
`cross_merge` (pass them in `fns`), so the lesson trains the code you wrote in exercises.py (or the reference
solutions when LEARNJAX_SOLUTIONS=1). Input everywhere: Fashion-MNIST as 14x14 grids of 2x2 pixel patches.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

H = 32                    # hidden units per sweep
KINDS = {                 # name -> (description, hidden size per sweep)
    "mlp": ("MLP, 784 -> 16 -> 10", None),
    "cnn": ("CNN, two 3x3 convs + average pool", None),
    "mdrnn": ("MDRNN, 4 corners, separate weights per corner (Graves 2007)", H),
    "mdrnn_tied": ("MDRNN, 4 corners, one set of weights", H),
    "mdrnn_tied_wide": ("MDRNN, one set of weights, wider (same parameter count as separate)", 54),
    "linscan": ("Gated linear recurrence over VMamba's 4 cross-scan orders (parallel scan)", H),
}


def patches(images):
    """(N, 28, 28) uint8 -> (N, 14, 14, 4) float32: 2x2 pixel patches as the 4 features of each cell."""
    x = np.asarray(images, np.float32) / 255.0
    return x.reshape(-1, 14, 2, 14, 2).transpose(0, 1, 3, 2, 4).reshape(-1, 14, 14, 4)


def load_data():
    from learnjax.data import fashion_mnist

    d = fashion_mnist()
    return (patches(d["train_images"]), d["train_labels"].astype(np.int32),
            patches(d["test_images"]), d["test_labels"].astype(np.int32))


def _dense(key, i, o):
    return {"w": jax.random.normal(key, (i, o)) / np.sqrt(i), "b": jnp.zeros(o)}


def init_sweep(key, F, h):
    k = jax.random.split(key, 3)
    return {"wx": jax.random.normal(k[0], (F, h)) / np.sqrt(F),
            "wu": jax.random.normal(k[1], (h, h)) * 0.5 / np.sqrt(h),
            "wl": jax.random.normal(k[2], (h, h)) * 0.5 / np.sqrt(h), "b": jnp.zeros(h)}


def sweep_cellwise(p, x):
    """The top-left sweep strictly one cell at a time, as a scan over rows of scans over columns: R * C
    sequential steps, the way a CPU implementation (RNNLIB, 2007) visits the grid. For timing against the
    wavefront; the exercise's mdrnn_sweep_naive does the same with Python loops, which jit unrolls."""
    B, _, C, _ = x.shape
    h = p["wu"].shape[0]
    z = jnp.moveaxis(x @ p["wx"] + p["b"], (1, 2), (0, 1))          # (R, C, B, h)

    def row(prev_row, zr):                                         # prev_row: (C, B, h), the row above
        def cell(left, inp):
            zc, up = inp
            hc = jnp.tanh(zc + up @ p["wu"] + left @ p["wl"])
            return hc, hc
        hr = jax.lax.scan(cell, jnp.zeros((B, h), x.dtype), (zr, prev_row))[1]
        return hr, hr

    hs = jax.lax.scan(row, jnp.zeros((C, B, h), x.dtype), z)[1]     # (R, C, B, h)
    return jnp.moveaxis(hs, (0, 1), (1, 2))


def init(kind: str, key):
    k = jax.random.split(key, 8)
    if kind == "mlp":
        return {"l1": _dense(k[0], 784, 16), "l2": _dense(k[1], 16, 10)}
    if kind == "cnn":
        return {"c1": jax.random.normal(k[0], (3, 3, 4, 32)) / 6, "b1": jnp.zeros(32),
                "c2": jax.random.normal(k[1], (3, 3, 32, 48)) / 17, "b2": jnp.zeros(48), "out": _dense(k[2], 48, 10)}
    h = KINDS[kind][1]
    if kind.startswith("mdrnn"):
        n = 4 if kind == "mdrnn" else 1
        return {"dirs": [init_sweep(k[i], 4, h) for i in range(n)], "mix": _dense(k[5], 4 * h, 32),
                "out": _dense(k[6], 32, 10)}
    if kind == "linscan":
        ka = jax.random.split(k[1], 2)
        return {"inp": _dense(k[0], 4, 16), "wa": jax.random.normal(ka[0], (4, 16, h)) / 4.0,
                "ba": jnp.full((4, 1, 1, h), 2.0), "wb": jax.random.normal(ka[1], (4, 16, h)) / 4.0,
                "mix": _dense(k[5], h, 32), "out": _dense(k[6], 32, 10)}
    raise ValueError(kind)


def n_params(p) -> int:
    return int(sum(x.size for x in jax.tree.leaves(p)))


def linear_scan(a, b, parallel=True):
    """h_t = a_t * h_{t-1} + b_t along axis -2, h_0 = 0. a, b: (..., T, H). Module 36's parallel scan."""
    if parallel:
        def comb(l, r):
            return l[0] * r[0], r[0] * l[1] + r[1]
        return jax.lax.associative_scan(comb, (a, b), axis=a.ndim - 2)[1]

    def step(h, ab):
        h = ab[0] * h + ab[1]
        return h, h
    a_t, b_t = jnp.moveaxis(a, -2, 0), jnp.moveaxis(b, -2, 0)
    return jnp.moveaxis(jax.lax.scan(step, jnp.zeros_like(a_t[0]), (a_t, b_t))[1], 0, -2)


def forward(kind: str, p, x, fns: dict, parallel=True):
    """Logits (B, 10). fns: {"sweep", "four_direction_sweeps", "cross_scan", "cross_merge"} (yours)."""
    if kind == "mlp":
        return jax.nn.relu(x.reshape(x.shape[0], -1) @ p["l1"]["w"] + p["l1"]["b"]) @ p["l2"]["w"] + p["l2"]["b"]
    if kind == "cnn":
        dn = ("NHWC", "HWIO", "NHWC")
        h = jax.nn.relu(jax.lax.conv_general_dilated(x, p["c1"], (1, 1), "SAME", dimension_numbers=dn) + p["b1"])
        h = jax.nn.relu(jax.lax.conv_general_dilated(h, p["c2"], (2, 2), "SAME", dimension_numbers=dn) + p["b2"])
        return h.mean((1, 2)) @ p["out"]["w"] + p["out"]["b"]
    if kind.startswith("mdrnn"):
        z = fns["four_direction_sweeps"](p["dirs"], x, fns["sweep"])      # (B, R, C, 4h): each cell sees it all
    elif kind == "linscan":
        _, R, C, _ = x.shape
        s = fns["cross_scan"](jax.nn.gelu(x @ p["inp"]["w"] + p["inp"]["b"]))   # (4, B, T, 16)
        a = jax.nn.sigmoid(jnp.einsum("kbtf,kfh->kbth", s, p["wa"]) + p["ba"])    # per-order decay gates
        hs = linear_scan(a, (1 - a) * jnp.einsum("kbtf,kfh->kbth", s, p["wb"]), parallel)
        z = fns["cross_merge"](hs, R, C)
    else:
        raise ValueError(kind)
    z = jax.nn.relu(z @ p["mix"]["w"] + p["mix"]["b"])
    return z.mean((1, 2)) @ p["out"]["w"] + p["out"]["b"]


def occlude(x, size: int, seed: int = 0):
    """Black out one random size x size square of cells per image (a 2*size-pixel square)."""
    rng = np.random.default_rng(seed)
    x = np.array(x)
    R, C = x.shape[1:3]
    for n in range(len(x)):
        i, j = rng.integers(0, R - size + 1), rng.integers(0, C - size + 1)
        x[n, i:i + size, j:j + size] = 0.0
    return x


def train(kind, fns, data, epochs=3, n_train=None, seed=0, bs=256, lr=3e-3, parallel=True, log=None,
          occlusion=(4,)):
    """Train one model; return params, test accuracy (clean and occluded), timing."""
    Xtr, Ytr, Xte, Yte = data
    if n_train:
        idx = np.random.default_rng(1000 + seed).permutation(len(Xtr))[:n_train]
        Xtr, Ytr = Xtr[idx], Ytr[idx]
    bs = min(bs, len(Xtr))
    p = init(kind, jax.random.key(seed))
    opt = optax.adam(lr)
    s = opt.init(p)

    def loss_fn(p, x, y):
        return optax.softmax_cross_entropy_with_integer_labels(forward(kind, p, x, fns, parallel), y).mean()

    @jax.jit
    def step(p, s, x, y):
        loss, g = jax.value_and_grad(loss_fn)(p, x, y)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, loss

    @jax.jit
    def correct(p, x, y):
        return (forward(kind, p, x, fns, parallel).argmax(-1) == y).sum()

    def acc(p, X, Y):
        return float(sum(int(correct(p, jnp.asarray(X[i:i + 1000]), jnp.asarray(Y[i:i + 1000])))
                         for i in range(0, len(X), 1000)) / len(X))

    t0 = time.time()
    jax.block_until_ready(step(p, s, jnp.asarray(Xtr[:bs]), jnp.asarray(Ytr[:bs])))
    compile_s = time.time() - t0
    rng = np.random.default_rng(seed)
    n = len(Xtr) // bs * bs
    times, hist = [], []
    for ep in range(epochs):
        order = rng.permutation(len(Xtr))[:n].reshape(-1, bs)
        t0 = time.time()
        for b in order:
            p, s, _ = step(p, s, jnp.asarray(Xtr[b]), jnp.asarray(Ytr[b]))
        jax.block_until_ready(p)
        times.append(time.time() - t0)
        hist.append(acc(p, Xte, Yte))
        if log:
            log(f"{kind:16s} epoch {ep + 1}/{epochs}  test acc {hist[-1]:.4f}  {times[-1]:.1f}s")
    res = {"kind": kind, "params": n_params(p), "acc": hist[-1], "acc_by_epoch": hist,
           "s_per_epoch": float(np.median(times)), "compile_s": compile_s, "n_train": len(Xtr), "epochs": epochs,
           "seed": seed}
    for sz in occlusion:
        res[f"acc_occluded_{sz}"] = acc(p, occlude(Xte, sz, seed), Yte)
    return p, res


def time_forward_backward(fn, args, reps=5):
    """Median seconds for jit(grad) of fn(*args).sum(), after compiling."""
    g = jax.jit(jax.grad(lambda *a: fn(*a).sum()))
    jax.block_until_ready(g(*args))
    ts = []
    for _ in range(reps):
        t0 = time.time()
        jax.block_until_ready(g(*args))
        ts.append(time.time() - t0)
    return float(np.median(ts))
