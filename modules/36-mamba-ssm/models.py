"""Module 36 helpers: the benchmark, a small Mamba-style model, the selective-copying task, recurrent decoding.

Everything that matters for the lesson is built from the learner's exercise functions, which are passed in
(`scan_fn` = their linear_recurrence_assoc). Loaded by the lesson with importlib.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ---------------------------------------------------------------- benchmark: four ways over length L


def bench_lengths(ex, lengths=(256, 1024, 4096, 16384), D=256, N=16, cap=4096, d_head=64, reps=3):
    """Time (ms) and compiled scratch memory (MiB) for one forward pass over a length-L sequence."""
    key = jax.random.key(0)
    A = -jnp.exp(jax.random.normal(key, (D, N)) * 0.5)
    Bv = jax.random.normal(jax.random.key(1), (D, N))
    Cv = jax.random.normal(jax.random.key(2), (D, N))
    A_bar, B_bar = ex.discretize_zoh(A, Bv, 0.05)

    # The SSM parameters go in as jit *arguments*, not closed-over constants: closed over, XLA constant-folds the
    # whole (L, D, N) FFT kernel at compile time (minutes on the CPU). Module 06: closed-over values are constants.
    ssm = (A_bar, B_bar, Cv)

    def via_scan(xs, ssm):                   # xs (L, D): per-channel recurrence, sequential
        A_bar, B_bar, Cv = ssm

        def step(h, x):
            h = A_bar * h + B_bar * x[:, None]
            return h, jnp.sum(Cv * h, -1)
        return jax.lax.scan(step, jnp.zeros((D, N)), xs)[1]

    def via_assoc(xs, ssm):                  # parallel prefix scan
        A_bar, B_bar, Cv = ssm
        a = jnp.broadcast_to(A_bar, (xs.shape[0], D, N))
        h = ex.linear_recurrence_assoc(a, B_bar[None] * xs[:, :, None])
        return jnp.sum(h * Cv[None], -1)

    def via_fft(xs, ssm):                    # LTI only: kernel + FFT convolution, per channel
        A_bar, B_bar, Cv = ssm
        L = xs.shape[0]
        # A_bar**k written as exp(k·log A_bar) (A_bar is in (0, 1)): same numbers, but `**` with a broadcast exponent
        # inside a reduction compiles for minutes on ROCm, exp-then-sum in about a second.
        K = jnp.exp(jnp.arange(L, dtype=jnp.float32)[:, None, None] * jnp.log(A_bar)[None]) * (Cv * B_bar)[None]
        K = K.sum(-1)                                                             # (L, D)
        n = 2 * L
        return jnp.fft.irfft(jnp.fft.rfft(K, n, axis=0) * jnp.fft.rfft(xs, n, axis=0), n, axis=0)[:L]

    def via_attn(q, ssm):                    # one causal head, for comparison (ignores ssm)
        L = q.shape[0]
        s = (q @ q.T) / jnp.sqrt(d_head)
        s = jnp.where(jnp.tril(jnp.ones((L, L), bool)), s, -jnp.inf)
        return jax.nn.softmax(s, -1) @ q

    rows = []
    for L in lengths:
        xs = jax.random.normal(jax.random.key(L), (L, D))
        q = jax.random.normal(jax.random.key(L + 1), (L, d_head))
        row = {"L": L}
        for name, fn, arg, ok in [("scan", via_scan, xs, L <= cap), ("assoc", via_assoc, xs, True),
                                  ("fft", via_fft, xs, L <= cap), ("attention", via_attn, q, L <= cap)]:
            if not ok:
                row[name] = (None, None)
                continue
            f = jax.jit(fn)
            try:
                mem = f.lower(arg, ssm).compile().memory_analysis()
                mib = getattr(mem, "temp_size_in_bytes", 0) / 2**20 if mem is not None else None
            except Exception:  # noqa: BLE001
                mib = None
            jax.block_until_ready(f(arg, ssm))
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                jax.block_until_ready(f(arg, ssm))
                ts.append(time.perf_counter() - t0)
            row[name] = (1e3 * float(np.median(ts)), mib)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- a small Mamba-style model

def init_mamba(key, V, D=96, E=192, N=16, R=8, K=4, layers=3, selective=True):
    """Embedding -> [RMSNorm -> Mamba block] x layers (residual) -> RMSNorm -> tied head."""
    keys = jax.random.split(key, 1 + layers)
    params = {"embed": 0.1 * jax.random.normal(keys[0], (V, D)), "norm_f": jnp.ones(D), "layers": []}
    for i in range(layers):
        k = jax.random.split(keys[1 + i], 8)
        dt = jnp.exp(jax.random.uniform(k[0], (E,), minval=np.log(1e-3), maxval=np.log(1e-1)))
        lp = {
            "norm": jnp.ones(D),
            "in_proj": jax.random.normal(k[1], (D, 2 * E)) / np.sqrt(D),
            "conv_w": jax.random.normal(k[2], (K, E)) / np.sqrt(K),
            "conv_b": jnp.zeros(E),
            "x_proj": jax.random.normal(k[3], (E, R + 2 * N)) / np.sqrt(E),
            "dt_proj": jax.random.normal(k[4], (R, E)) / np.sqrt(R),
            "dt_bias": dt + jnp.log(-jnp.expm1(-dt)),          # softplus^-1(dt): Δ starts in [1e-3, 1e-1]
            "A_log": jnp.log(jnp.broadcast_to(jnp.arange(1, N + 1, dtype=jnp.float32), (E, N))),
            "Dskip": jnp.ones(E),
            "out_proj": jax.random.normal(k[5], (E, D)) / np.sqrt(E) / np.sqrt(2 * layers),
        }
        if not selective:                                        # LTI: B and C are learned constants
            lp["B_const"] = jax.random.normal(k[6], (N,)) / np.sqrt(N)
            lp["C_const"] = jax.random.normal(k[7], (N,)) / np.sqrt(N)
        params["layers"].append(lp)
    return params


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree.leaves(params)))


def _rms(x, w):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-5) * w


def _ssm_inputs(lp, u, selective):
    """Δ, B, C for a block. u: (..., L, E). Selective: functions of u. LTI: constants."""
    E, N = lp["A_log"].shape
    R = lp["dt_proj"].shape[0]
    if selective:
        proj = u @ lp["x_proj"]
        dt_r, Bm, Cm = proj[..., :R], proj[..., R:R + N], proj[..., R + N:]
        delta = jax.nn.softplus(dt_r @ lp["dt_proj"] + lp["dt_bias"])
    else:
        shape = u.shape[:-1]
        delta = jnp.broadcast_to(jax.nn.softplus(lp["dt_bias"]), shape + (E,))
        Bm = jnp.broadcast_to(lp["B_const"], shape + (N,))
        Cm = jnp.broadcast_to(lp["C_const"], shape + (N,))
    return delta, Bm, Cm


def block(lp, x, scan_fn, selective=True):
    """One Mamba-style block on x (B, L, D). scan_fn solves h_t = a_t h_{t-1} + b_t along axis 0."""
    _, L, _ = x.shape
    Kc = lp["conv_w"].shape[0]
    h = _rms(x, lp["norm"])
    u, z = jnp.split(h @ lp["in_proj"], 2, axis=-1)                           # (B, L, E) each
    up = jnp.pad(u, ((0, 0), (Kc - 1, 0), (0, 0)))                            # causal: pad the past only
    u = sum(lp["conv_w"][k] * up[:, k:k + L] for k in range(Kc)) + lp["conv_b"]
    u = jax.nn.silu(u)
    delta, Bm, Cm = _ssm_inputs(lp, u, selective)
    A = -jnp.exp(lp["A_log"])                                                 # (E, N), negative
    a = jnp.exp(delta[..., None] * A)                                         # (B, L, E, N)
    b = delta[..., None] * Bm[:, :, None, :] * u[..., None]
    hs = jnp.swapaxes(scan_fn(jnp.swapaxes(a, 0, 1), jnp.swapaxes(b, 0, 1)), 0, 1)
    y = jnp.einsum("blen,bln->ble", hs, Cm) + lp["Dskip"] * u
    return x + (y * jax.nn.silu(z)) @ lp["out_proj"]


def attn_block(lp, x):
    """One causal multi-head attention block on x (B, L, D), pre-RMSNorm and residual, like Module 16's.
    No positional encoding: in a hybrid the Mamba layers already carry position (as in Jamba)."""
    B, L, D = x.shape
    H = lp["heads"].shape[0]
    q, k, v = jnp.split(_rms(x, lp["norm"]) @ lp["wqkv"], 3, axis=-1)
    q, k, v = (t.reshape(B, L, H, D // H) for t in (q, k, v))
    s = jnp.einsum("bthd,bshd->bhts", q, k) / np.sqrt(D // H)
    s = jnp.where(jnp.tril(jnp.ones((L, L), bool)), s, -jnp.inf)
    o = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(s, -1), v).reshape(B, L, D)
    return x + o @ lp["wo"]


def forward(params, tokens, scan_fn, selective=True):
    x = params["embed"][tokens]
    for lp in params["layers"]:
        x = attn_block(lp, x) if "wqkv" in lp else block(lp, x, scan_fn, selective)
    return _rms(x, params["norm_f"]) @ params["embed"].T


def init_hybrid(key, V, attn_layers=(), heads=4, **kw):
    """A Mamba stack with the layers listed in `attn_layers` replaced by attention blocks (Jamba-style).
    attn_layers=() is plain Mamba; every index listed is plain attention there."""
    params = init_mamba(key, V, **kw)
    D, n = params["embed"].shape[1], len(params["layers"])
    for i in attn_layers:
        k1, k2 = jax.random.split(jax.random.fold_in(key, 1000 + i))
        params["layers"][i] = {"norm": jnp.ones(D), "heads": jnp.zeros(heads),   # `heads` only carries H
                               "wqkv": jax.random.normal(k1, (D, 3 * D)) / np.sqrt(D),
                               "wo": jax.random.normal(k2, (D, D)) / np.sqrt(D) / np.sqrt(2 * n)}
    return params


# ---------------------------------------------------------------- recurrent (constant-memory) decoding

def init_state(params, batch=1):
    return [{"conv": jnp.zeros((batch, lp["conv_w"].shape[0] - 1, lp["conv_w"].shape[1])),
             "h": jnp.zeros((batch,) + lp["A_log"].shape)} for lp in params["layers"]]


def step(params, state, token, selective=True):
    """One token through the model in recurrent mode. token: (B,). Returns (logits (B, V), new state)."""
    x = params["embed"][token]                                                # (B, D)
    new = []
    for lp, st in zip(params["layers"], state):
        h = _rms(x, lp["norm"])
        u, z = jnp.split(h @ lp["in_proj"], 2, axis=-1)                       # (B, E)
        window = jnp.concatenate([st["conv"], u[:, None]], axis=1)            # (B, K, E)
        uc = jax.nn.silu(jnp.einsum("bke,ke->be", window, lp["conv_w"]) + lp["conv_b"])
        delta, Bm, Cm = _ssm_inputs(lp, uc, selective)
        A = -jnp.exp(lp["A_log"])
        hs = jnp.exp(delta[..., None] * A) * st["h"] + delta[..., None] * Bm[:, None, :] * uc[..., None]
        y = jnp.einsum("ben,bn->be", hs, Cm) + lp["Dskip"] * uc
        x = x + (y * jax.nn.silu(z)) @ lp["out_proj"]
        new.append({"conv": window[:, 1:], "h": hs})
    return _rms(x, params["norm_f"]) @ params["embed"].T, new


def state_floats(params) -> int:
    return int(sum(lp["A_log"].size + (lp["conv_w"].shape[0] - 1) * lp["conv_w"].shape[1] for lp in params["layers"]))


def generate(params, prompt_ids, n, key, temperature=0.8, selective=True):
    stepj = jax.jit(lambda p, s, t: step(p, s, t, selective))
    st = init_state(params)
    logits = None
    for t in prompt_ids:
        logits, st = stepj(params, st, jnp.array([t]))
    out = []
    for _ in range(n):
        key, k = jax.random.split(key)
        nxt = jax.random.categorical(k, logits[0] / temperature)
        out.append(int(nxt))
        logits, st = stepj(params, st, nxt[None])
    return out


# ---------------------------------------------------------------- training

def _windows(key, data, batch, T):
    starts = jax.random.randint(key, (batch,), 0, data.shape[0] - T - 1)
    idx = starts[:, None] + jnp.arange(T + 1)[None]
    w = data[idx]
    return w[:, :-1], w[:, 1:]


def train_lm(scan_fn, V, train_ids, val_ids, steps=1500, batch=32, T=128, lr=3e-3, seed=0, **kw):
    params = init_mamba(jax.random.key(seed), V, **kw)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, 100, steps, lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))

    def loss_fn(p, x, y):
        return optax.softmax_cross_entropy_with_integer_labels(forward(p, x, scan_fn), y).mean()

    @jax.jit
    def chunk(p, s, key, data):
        def body(carry, k):
            p, s = carry
            x, y = _windows(k, data, batch, T)
            loss, g = jax.value_and_grad(loss_fn)(p, x, y)
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), loss
        (p, s), ls = jax.lax.scan(body, (p, s), jax.random.split(key, 100))
        return p, s, ls

    evaluate = jax.jit(lambda p, k, data: loss_fn(p, *_windows(k, data, 64, T)))
    s = opt.init(params)
    train_d, val_d = jnp.asarray(train_ids), jnp.asarray(val_ids)
    key = jax.random.key(seed + 1)
    curve, t0 = [], time.perf_counter()
    for _ in range(steps // 100):
        key, k = jax.random.split(key)
        params, s, ls = chunk(params, s, k, train_d)
        curve.append(float(ls.mean()))
    val = float(np.mean([float(evaluate(params, jax.random.key(500 + j), val_d)) for j in range(8)]))
    return {"params": params, "curve": curve, "val": val, "seconds": time.perf_counter() - t0,
            "n_params": count_params(params)}


# ---------------------------------------------------------------- selective copying

NOISE, MARK, COLORS = 0, 9, 8


def copy_batch(key, batch, ctx=32, M=4):
    """Selective copying (Gu & Dao 2023, simplified): M colour tokens at random positions among noise,
    then M markers; at each marker the model must emit the colours in order of appearance."""
    k1, k2 = jax.random.split(key)
    pos = jnp.sort(jax.vmap(lambda k: jax.random.choice(k, ctx, (M,), replace=False))(jax.random.split(k1, batch)), -1)
    cols = jax.random.randint(k2, (batch, M), 1, COLORS + 1)
    seq = jnp.zeros((batch, ctx), jnp.int32).at[jnp.arange(batch)[:, None], pos].set(cols)
    x = jnp.concatenate([seq, jnp.full((batch, M), MARK, jnp.int32)], 1)
    return x, cols


def train_copy(scan_fn, selective, steps=800, batch=128, lr=2e-3, seed=0, ctx=128, M=8):
    params = init_mamba(jax.random.key(seed), 10, D=64, E=128, N=16, layers=2, selective=selective)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=0.0))

    def loss_acc(p, x, cols):
        logits = forward(p, x, scan_fn, selective)[:, -M:]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, cols).mean()
        return loss, jnp.mean(jnp.argmax(logits, -1) == cols)

    @jax.jit
    def chunk(p, s, key):
        def body(carry, k):
            p, s = carry
            (_loss, acc), g = jax.value_and_grad(loss_acc, has_aux=True)(p, *copy_batch(k, batch, ctx, M))
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), acc
        (p, s), accs = jax.lax.scan(body, (p, s), jax.random.split(key, 100))
        return p, s, accs

    s = opt.init(params)
    key, curve, t0 = jax.random.key(seed + 7), [], time.perf_counter()
    for _ in range(steps // 100):
        key, k = jax.random.split(key)
        params, s, accs = chunk(params, s, k)
        curve.append(float(accs.mean()))
    test_acc = float(jax.jit(lambda p, k: loss_acc(p, *copy_batch(k, 2048, ctx, M))[1])(params, jax.random.key(999)))
    return {"curve": curve, "test_acc": test_acc, "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------- hybrids: multi-query associative recall

def mqar_batch(key, batch, n_pairs, n_keys=64, n_vals=64):
    """Multi-query associative recall (Arora et al. 2023, "Zoology"): n_pairs key-value pairs, then every key
    again in a new order; at each repeated key the model must output its value. Keys are tokens
    1..n_keys, values n_keys+1..n_keys+n_vals. Returns (tokens (B, 3·n_pairs), targets (B, n_pairs))."""
    k1, k2, k3 = jax.random.split(key, 3)
    rows = jax.random.split(k1, batch)
    keys = jax.vmap(lambda k: jax.random.choice(k, n_keys, (n_pairs,), replace=False))(rows) + 1
    vals = jax.random.randint(k2, (batch, n_pairs), 0, n_vals) + n_keys + 1
    order = jax.vmap(lambda k: jax.random.permutation(k, n_pairs))(jax.random.split(k3, batch))
    pairs = jnp.stack([keys, vals], -1).reshape(batch, 2 * n_pairs)
    queries = jnp.take_along_axis(keys, order, 1)
    return jnp.concatenate([pairs, queries], 1), jnp.take_along_axis(vals, order, 1)


def train_mqar(scan_fn, attn_layers, n_pairs=32, steps=1000, batch=64, lr=2e-3, seed=0, layers=4, **kw):
    """Train a (hybrid) model on MQAR; returns test accuracy over 1,024 fresh sequences, curve and time."""
    V = 1 + 64 + 64
    params = init_hybrid(jax.random.key(seed), V, attn_layers, layers=layers, **{"D": 64, "E": 128, **kw})
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=0.0))

    def loss_acc(p, x, y):
        logits = forward(p, x, scan_fn)[:, -n_pairs:]
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean(), jnp.mean(jnp.argmax(logits, -1) == y)

    @jax.jit
    def chunk(p, s, key):
        def body(carry, k):
            p, s = carry
            (_l, acc), g = jax.value_and_grad(loss_acc, has_aux=True)(p, *mqar_batch(k, batch, n_pairs))
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), acc
        (p, s), accs = jax.lax.scan(body, (p, s), jax.random.split(key, 100))
        return p, s, accs

    s, key, curve, t0 = opt.init(params), jax.random.key(seed + 7), [], time.perf_counter()
    for _ in range(steps // 100):
        key, k = jax.random.split(key)
        params, s, accs = chunk(params, s, k)
        curve.append(float(accs.mean()))
    test = float(jax.jit(lambda p, k: loss_acc(p, *mqar_batch(k, 1024, n_pairs))[1])(params, jax.random.key(999)))
    return {"test_acc": test, "curve": curve, "seconds": time.perf_counter() - t0, "n_params": count_params(params)}
