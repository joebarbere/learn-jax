"""Module 41 model code: the linear-attention family and hybrids, built on your exercises.

Provided (not exercises): the chunked *delta-rule* algorithm (the WY / UT transform), the layers around the
kernels (projections, short convolution, gates, norms), a small language model whose layer pattern is a
string ("GGGAGGGA" = three Gated DeltaNet layers, one attention layer, twice), the associative-recall task,
the training loop, and the timing helpers.

Every kernel works on head-major arrays: q, k (B, H, T, dk), v (B, H, T, dv), per-step gates (B, H, T).
"""
from __future__ import annotations

import json
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ---------------------------------------------------------------- configs
BASE = {"d_model": 384, "n_heads": 6, "d_head": 64, "d_ff": 1024, "layout": "AAAAAAAA", "conv": 4,
        "chunk": 64, "rope_theta": 1e4}
LAYOUTS = {                                   # the race: same width, depth, FFNs and parameters (±1%)
    "attention": "AAAAAAAA",                  # a Transformer (gated attention, RoPE)
    "linear": "LLLLLLLL",                     # linear attention: add to the state, never forget
    "mamba2": "MMMMMMMM",                     # + a per-step decay (Mamba-2's SSD, written as attention)
    "gdn": "GGGGGGGG",                        # + the delta rule: overwrite instead of add (Gated DeltaNet)
    "hybrid": "GGGAGGGA",                     # 3 : 1, the Qwen3-Next / Kimi Linear pattern
}
KINDS = {"A": "attention", "L": "linear", "M": "decay (Mamba-2)", "D": "delta rule", "G": "gated delta rule"}


def variant(name: str, **over) -> dict:
    return {**BASE, "layout": LAYOUTS.get(name, name), **over}


# ---------------------------------------------------------------- small pieces
def rms_norm(x, w, eps: float = 1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps)).astype(x.dtype) * w


def l2_norm(x, eps: float = 1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.sum(x32 * x32, -1, keepdims=True) + eps)).astype(x.dtype)


def rope(x, theta: float = 1e4):
    """Rotate-half RoPE. x: (B, H, T, D)."""
    T, half = x.shape[-2], x.shape[-1] // 2
    ang = jnp.arange(T)[:, None] * (1.0 / theta ** (jnp.arange(half) / half))[None]
    cos, sin = jnp.cos(ang).astype(x.dtype), jnp.sin(ang).astype(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def causal_conv(x, w):
    """Depthwise causal convolution over time. x: (B, T, C), w: (K, C)."""
    K, T = w.shape[0], x.shape[1]
    xp = jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0)))
    return sum(w[i] * xp[:, i:i + T] for i in range(K))


def softmax_attention(q, k, v):
    """Causal softmax attention, the reference. q, k (B, H, T, d), v (B, H, T, dv)."""
    T = q.shape[-2]
    s = jnp.einsum("bhtd,bhsd->bhts", q, k).astype(jnp.float32) / jnp.sqrt(q.shape[-1])
    s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -jnp.inf)
    return jnp.einsum("bhts,bhse->bhte", jax.nn.softmax(s, -1).astype(v.dtype), v)


# ---------------------------------------------------------------- the chunked delta rule
def chunked_gated_delta(parts, q, k, v, beta, log_a, chunk: int):
    """The gated delta rule, chunk by chunk (Yang et al. 2024, "Parallelizing Linear Transformers with the
    Delta Rule"; Gated DeltaNet, 2025). Same output as your `gated_delta_scan`, but with matmuls.

    The trick: what the delta rule *writes* at step i is u_i = β_i (v_i − (a_i S_{i−1})ᵀ k_i), the error.
    Once the u's are known, S_i = a_i S_{i−1} + k_i u_iᵀ is plain decayed linear attention with values u,
    which your chunked algorithm already handles. Inside a chunk the u's depend on each other through a
    lower-triangular system, (I + A) U = diag(β) V − diag(β γ) K S_in, so one triangular solve per chunk
    gives T = (I + A)⁻¹, and then U = T diag(β) V − T diag(β γ) K S_in: two matmuls, and only the second
    needs the incoming state (the "WY representation" — W = T diag(β γ) K)."""
    *lead, T_len, dk = q.shape
    dv, n = v.shape[-1], T_len // chunk
    f = lambda x, *s: x.reshape(*lead, n, chunk, *s)
    qc, kc, vc, bc, ac = f(q, dk), f(k, dk), f(v, dv), f(beta), f(log_a)
    cs = jnp.cumsum(ac, -1)                                              # log γ_i inside each chunk
    decay = jnp.exp(parts.segsum(ac))                                    # γ_i / γ_j, zero above the diagonal
    kk = jnp.einsum("...id,...jd->...ij", kc, kc)
    strict = jnp.tril(jnp.ones((chunk, chunk), bool), -1)
    A = jnp.where(strict, bc[..., :, None] * decay * kk, 0.0)
    eye = jnp.eye(chunk, dtype=q.dtype)
    Tm = jax.scipy.linalg.solve_triangular(eye + A, jnp.broadcast_to(eye, A.shape), lower=True,
                                           unit_diagonal=True)
    W = Tm @ ((bc * jnp.exp(cs))[..., None] * kc)                        # (..., n, C, dk)
    U0 = Tm @ (bc[..., None] * vc)                                       # (..., n, C, dv)
    qk = jnp.einsum("...id,...jd->...ij", qc, kc) * decay                # causal, decayed
    to_end = jnp.exp(cs[..., -1:] - cs)

    def step(S, inp):
        qk_c, q_c, k_c, W_c, U0_c, cs_c, te_c = inp
        U = U0_c - W_c @ S                                               # what each step writes
        y = qk_c @ U + jnp.exp(cs_c)[..., None] * (q_c @ S)
        S = jnp.exp(cs_c[..., -1])[..., None, None] * S + jnp.swapaxes(te_c[..., None] * k_c, -1, -2) @ U
        return S, y

    m = lambda x: jnp.moveaxis(x, len(lead), 0)
    S0 = jnp.zeros((*lead, dk, dv), q.dtype)
    _, y = jax.lax.scan(step, S0, tuple(map(m, (qk, qc, kc, W, U0, cs, to_end))))
    return jnp.moveaxis(y, 0, len(lead)).reshape(*lead, T_len, dv)


def run_kernel(parts, kind, q, k, v, beta, log_a, chunk):
    """The sequence-mixing kernel of each linear layer kind, in float32."""
    q, k, v, beta, log_a = (x.astype(jnp.float32) for x in (q, k, v, beta, log_a))
    if kind == "L":
        return parts.chunked_linear_attention(q, k, v, jnp.zeros_like(log_a), chunk)
    if kind == "M":
        return parts.chunked_linear_attention(q, k, v, log_a, chunk)
    if kind == "D":
        return chunked_gated_delta(parts, q, k, v, beta, jnp.zeros_like(log_a), chunk)
    return chunked_gated_delta(parts, q, k, v, beta, log_a, chunk)


# ---------------------------------------------------------------- layers
def _dense(key, fan_in, shape):
    return jax.random.normal(key, shape, jnp.float32) * fan_in ** -0.5


def init_mixer(key, kind, cfg):
    d, H, dh = cfg["d_model"], cfg["n_heads"], cfg["d_head"]
    ks = jax.random.split(key, 8)
    p = {"wqkv": _dense(ks[0], d, (d, 3 * H * dh)), "wg": _dense(ks[1], d, (d, H * dh)),
         "wo": _dense(ks[2], H * dh, (H * dh, d)) / np.sqrt(2 * len(cfg["layout"]))}
    if kind == "A" and not cfg.get("attn_conv"):
        return p
    p["conv"] = jnp.zeros((cfg["conv"], 3 * H * dh)).at[-1].set(1.0)    # starts as the identity
    if kind == "A":
        return p
    p["norm"] = jnp.ones((dh,))
    if kind in "MG":                                                     # Mamba-2's decay parameters
        p["w_dt"] = _dense(ks[3], d, (d, H)) * 0.1
        lo, hi = cfg.get("dt_range", (1e-3, 1e-1))                      # Mamba-2's default step-size range
        dt = jnp.exp(jax.random.uniform(ks[4], (H,), minval=np.log(lo), maxval=np.log(hi)))
        p["dt_bias"] = dt + jnp.log(-jnp.expm1(-dt))                     # softplus⁻¹(dt)
        p["A_log"] = jnp.log(jax.random.uniform(ks[5], (H,), minval=1.0, maxval=16.0))
    if kind in "DG":
        p["w_beta"] = _dense(ks[6], d, (d, H))
    return p


def gates(p, x, kind, H, T):
    """β (B, H, T) and log a (B, H, T) for a linear layer; zeros where the kind has none."""
    B = x.shape[0]
    beta = jax.nn.sigmoid(x @ p["w_beta"]) if "w_beta" in p else jnp.zeros((B, T, H), x.dtype)
    if "A_log" in p:
        log_a = -jnp.exp(p["A_log"]) * jax.nn.softplus((x @ p["w_dt"]).astype(jnp.float32) + p["dt_bias"])
    else:
        log_a = jnp.zeros((B, T, H), jnp.float32)
    return jnp.swapaxes(beta, 1, 2), jnp.swapaxes(log_a, 1, 2)


def mixer(parts, p, x, kind, cfg):
    B, T, _ = x.shape
    H, dh = cfg["n_heads"], cfg["d_head"]
    qkv = x @ p["wqkv"]
    heads = lambda a: a.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    if kind == "A":
        if "conv" in p:                                                  # optional: the same short conv
            qkv = causal_conv(qkv, p["conv"].astype(qkv.dtype))
        q, k, v = (heads(a) for a in jnp.split(qkv, 3, -1))
        o = softmax_attention(rope(q, cfg["rope_theta"]), rope(k, cfg["rope_theta"]), v)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, H * dh) * jax.nn.sigmoid(x @ p["wg"])   # gated attention
        return o @ p["wo"]
    qkv = jax.nn.silu(causal_conv(qkv, p["conv"].astype(qkv.dtype)))
    q, k, v = (heads(a) for a in jnp.split(qkv, 3, -1))
    q, k = l2_norm(q) * dh ** -0.5, l2_norm(k)                           # unit keys keep the delta rule stable
    beta, log_a = gates(p, x, kind, H, T)
    o = run_kernel(parts, kind, q, k, v, beta, log_a, min(cfg["chunk"], T)).astype(x.dtype)
    o = rms_norm(o, p["norm"]).transpose(0, 2, 1, 3).reshape(B, T, H * dh)
    return (o * jax.nn.silu(x @ p["wg"])) @ p["wo"]


def block_pattern(layout: str) -> tuple[str, int]:
    """The shortest repeating block of a layout and how many times it repeats: "GGGAGGGA" -> ("GGGA", 2).
    Model cards describe hybrids this way (Qwen3-Next: 12 × (3 × Gated DeltaNet, 1 × gated attention))."""
    for p in range(1, len(layout) + 1):
        if len(layout) % p == 0 and layout[:p] * (len(layout) // p) == layout:
            return layout[:p], len(layout) // p
    raise AssertionError(layout)


def _init_layer(key, kind, cfg):
    d, L = cfg["d_model"], len(cfg["layout"])
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {"norm1": jnp.ones((d,)), "norm2": jnp.ones((d,)), "mix": init_mixer(k1, kind, cfg),
            "ffn": {"w1": _dense(k2, d, (d, cfg["d_ff"])), "w3": _dense(k3, d, (d, cfg["d_ff"])),
                    "w2": _dense(k4, cfg["d_ff"], (cfg["d_ff"], d)) / np.sqrt(2 * L)}}


def init_model(key, vocab, cfg):
    """params["blocks"][i] holds position i of the repeating block, stacked over the repeats (leading axis),
    so the forward pass can `lax.scan` over repeats: XLA compiles one block, not every layer."""
    pattern, reps = block_pattern(cfg["layout"])
    ke, kb = jax.random.split(key)
    blocks = [jax.vmap(lambda k, kind=kind: _init_layer(k, kind, cfg))(jax.random.split(jax.random.fold_in(kb, i), reps))
              for i, kind in enumerate(pattern)]
    return {"embed": jax.random.normal(ke, (vocab, cfg["d_model"])) * 0.02, "blocks": blocks,
            "norm": jnp.ones((cfg["d_model"],))}


def layer(parts, lp, x, kind, cfg):
    x = x + mixer(parts, lp["mix"], rms_norm(x, lp["norm1"]), kind, cfg)
    h = rms_norm(x, lp["norm2"])
    f = lp["ffn"]
    return x + (jax.nn.silu(h @ f["w1"]) * (h @ f["w3"])) @ f["w2"]


def forward(parts, params, tokens, cfg):
    pattern, _ = block_pattern(cfg["layout"])

    def block(x, bp):
        for kind, lp in zip(pattern, bp, strict=True):
            x = layer(parts, lp, x, kind, cfg)
        return x, None

    x, _ = jax.lax.scan(block, params["embed"][tokens], params["blocks"])
    return rms_norm(x, params["norm"]) @ params["embed"].T              # tied embeddings


def count_params(params) -> int:
    return sum(int(np.prod(a.shape)) for a in jax.tree.leaves(params))


# ---------------------------------------------------------------- training
def get_windows(key, data, batch, seq_len):
    starts = jax.random.randint(key, (batch,), 0, data.shape[0] - seq_len - 1)
    win = jax.vmap(lambda s: jax.lax.dynamic_slice(data, (s,), (seq_len + 1,)))(starts)
    return win[:, :-1], win[:, 1:]


def cross_entropy(logits, targets):
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    return -jnp.take_along_axis(logp, targets[..., None], -1).mean()


def make_optimizer(lr, steps, warmup=200):
    warmup = min(warmup, steps // 2)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(steps, warmup + 1), lr * 0.1)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=0.1))


def _bf16(params):
    """bf16 compute copy, except the decay parameters (they go through exp twice: keep them exact)."""
    keep = ("A_log", "dt_bias")
    return jax.tree_util.tree_map_with_path(
        lambda path, a: a if any(n in jax.tree_util.keystr(path) for n in keep) else a.astype(jnp.bfloat16), params)


def make_trainer(parts, cfg, optimizer, batch=16, seq_len=512):
    """train_chunk(params, opt_state, data, key, n_steps) -> (params, opt_state, losses); eval_loss(params, x, y)."""

    def loss_fn(params, x, y):
        return cross_entropy(forward(parts, _bf16(params), x, cfg), y)

    def step(carry, key):
        params, opt_state, data = carry
        x, y = get_windows(key, data, batch, seq_len)
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, data), loss

    @partial(jax.jit, static_argnames="n_steps", donate_argnums=(0, 1))
    def train_chunk(params, opt_state, data, key, n_steps):
        (params, opt_state, _), losses = jax.lax.scan(step, (params, opt_state, data), jax.random.split(key, n_steps))
        return params, opt_state, losses

    @jax.jit
    def eval_loss(params, x, y):
        return cross_entropy(forward(parts, _bf16(params), x, cfg), y)

    return train_chunk, eval_loss


def val_batches(val, batch, seq_len, n=8, seed=1234):
    rng = np.random.default_rng(seed)
    out = []
    for row in rng.integers(0, len(val) - seq_len - 1, (n, batch)):
        w = np.stack([val[s:s + seq_len + 1] for s in row]).astype(np.int32)
        out.append((jnp.asarray(w[:, :-1]), jnp.asarray(w[:, 1:])))
    return out


def save(params, cfg, out, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    flat = {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    np.savez(out / "params.tmp.npz", **flat)
    (out / "params.tmp.npz").replace(out / "params.npz")
    (out / "config.json").write_text(json.dumps({"model": cfg, **(extra or {})}, indent=2))


def load(out, vocab=8192):
    meta = json.loads((out / "config.json").read_text())
    cfg = meta["model"]
    arrays = np.load(out / "params.npz")
    template = jax.eval_shape(lambda: init_model(jax.random.key(0), vocab, cfg))
    params = jax.tree_util.tree_map_with_path(lambda p, _: jnp.asarray(arrays[jax.tree_util.keystr(p)]), template)
    return params, cfg, meta


def read_metrics(path):
    try:
        return [json.loads(line) for line in open(path)]
    except OSError:
        return []


# ---------------------------------------------------------------- associative recall
def mqar_batch(key, batch, n_pairs, n_keys=512, n_vals=128):
    """Multi-query associative recall (Arora et al. 2023, "Zoology"), as in Module 36: n_pairs key-value
    pairs, then the keys again in a new order. Returns (tokens (B, 3·n_pairs), targets (B, n_pairs))."""
    k1, k2, k3 = jax.random.split(key, 3)
    keys = jax.vmap(lambda k: jax.random.choice(k, n_keys, (n_pairs,), replace=False))(jax.random.split(k1, batch)) + 1
    vals = jax.random.randint(k2, (batch, n_pairs), 0, n_vals) + n_keys + 1
    order = jax.vmap(lambda k: jax.random.permutation(k, n_pairs))(jax.random.split(k3, batch))
    pairs = jnp.stack([keys, vals], -1).reshape(batch, 2 * n_pairs)
    queries = jnp.take_along_axis(keys, order, 1)
    return jnp.concatenate([pairs, queries], 1), jnp.take_along_axis(vals, order, 1)


MQAR = {"d_model": 128, "n_heads": 2, "d_head": 32, "d_ff": 256, "conv": 4, "chunk": 32, "rope_theta": 1e4}


def train_mqar(parts, layout, n_pairs, steps=1500, batch=64, lr=1e-3, seed=0, **over):
    """Train a 2-layer model on MQAR (keys from 512 tokens, values from 128). Every linear layer's state is
    n_heads × d_head × d_head = 2 × 32 × 32 = 2,048 numbers; an attention layer keeps every key.
    Returns test accuracy on 1,024 fresh sequences, the training-accuracy curve, and the time."""
    cfg = {**MQAR, "layout": layout, **over}
    V = 1 + 512 + 128
    T = 3 * n_pairs
    pad = -T % cfg["chunk"]
    params = init_model(jax.random.key(seed), V, cfg)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=0.0))

    def loss_acc(p, x, y):
        logits = forward(parts, p, jnp.pad(x, ((0, 0), (0, pad))), cfg)[:, T - n_pairs:T]
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean(), jnp.mean(jnp.argmax(logits, -1) == y)

    @jax.jit
    def chunk(p, s, key):
        def body(carry, k):
            p, s = carry
            (_l, acc), g = jax.value_and_grad(loss_acc, has_aux=True)(p, *mqar_batch(k, batch, n_pairs))
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), acc
        (p, s), accs = jax.lax.scan(body, (p, s), jax.random.split(key, 50))
        return p, s, accs

    s, key, curve, t0 = opt.init(params), jax.random.key(seed + 7), [], time.perf_counter()
    for _ in range(steps // 50):
        key, k = jax.random.split(key)
        params, s, accs = chunk(params, s, k)
        curve.append(float(accs.mean()))
    acc_fn = jax.jit(lambda p, k: loss_acc(p, *mqar_batch(k, 128, n_pairs))[1])        # 8 × 128: fits in memory
    test = float(np.mean([acc_fn(params, jax.random.fold_in(jax.random.key(999), i)) for i in range(8)]))
    return {"test_acc": test, "curve": curve, "seconds": time.perf_counter() - t0, "n_params": count_params(params)}


# ---------------------------------------------------------------- timing: one layer, long sequences
def time_fn(fn, *args, reps=5):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench_prefill(parts, lengths=(512, 1024, 2048, 4096, 8192, 16384), H=6, dh=64, chunk=64, cap=16384):
    """Seconds for one layer's sequence mixing over a whole prompt (batch 1, bf16 attention, f32 kernels)."""
    out = {}
    for T in lengths:
        ks = jax.random.split(jax.random.key(T), 4)
        q, k, v = (l2_norm(jax.random.normal(kk, (1, H, T, dh))) for kk in ks[:3])
        beta = jax.nn.sigmoid(jax.random.normal(ks[3], (1, H, T)))
        log_a = jnp.full((1, H, T), -0.05)
        row = {}
        if T <= cap:
            att = jax.jit(softmax_attention)
            b = lambda x: x.astype(jnp.bfloat16)
            row["attention"] = time_fn(att, b(q), b(k), b(v))
        row["mamba2 (chunked)"] = time_fn(jax.jit(lambda *a: parts.chunked_linear_attention(*a, chunk)), q, k, v, log_a)
        row["gdn (chunked)"] = time_fn(jax.jit(lambda *a: chunked_gated_delta(parts, *a, chunk)), q, k, v, beta, log_a)
        if T <= 4096:
            row["gdn (step by step)"] = time_fn(jax.jit(parts.gated_delta_scan), q, k, v, beta, log_a)
        out[T] = row
    return out


def bench_decode(contexts=(1024, 4096, 16384, 65536), H=6, dh=64, n_tokens=64):
    """Per-token decode time and memory for one layer: attention reads a KV cache of `context` entries;
    the gated delta rule updates an H × dh × dh state. Batch 1, bf16 cache, f32 state."""
    out = {}

    @jax.jit
    def attn_steps(cache_k, cache_v, q):
        def body(c, qt):
            s = jnp.einsum("hd,htd->ht", qt, c[0]).astype(jnp.float32) / jnp.sqrt(dh)
            return c, jnp.einsum("ht,htd->hd", jax.nn.softmax(s, -1).astype(c[1].dtype), c[1])
        return jax.lax.scan(body, (cache_k, cache_v), q)[1]

    @jax.jit
    def gdn_steps(S, q, k, v, beta, la):
        def body(S, inp):
            qt, kt, vt, bt, at = inp
            S = jnp.exp(at)[:, None, None] * S
            S = S + bt[:, None, None] * kt[:, :, None] * (vt - jnp.einsum("hd,hde->he", kt, S))[:, None, :]
            return S, jnp.einsum("hd,hde->he", qt, S)
        return jax.lax.scan(body, S, (q, k, v, beta, la))[1]

    ks = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(ks[0], (n_tokens, H, dh))
    for C in contexts:
        ck = jax.random.normal(ks[1], (H, C, dh), jnp.bfloat16)
        cv = jax.random.normal(ks[2], (H, C, dh), jnp.bfloat16)
        ta = time_fn(attn_steps, ck, cv, q.astype(jnp.bfloat16)) / n_tokens
        S = jnp.zeros((H, dh, dh))
        tg = time_fn(gdn_steps, S, q, q, q, jnp.full((n_tokens, H), 0.5), jnp.full((n_tokens, H), -0.05)) / n_tokens
        out[C] = {"attention_ms": ta * 1e3, "gdn_ms": tg * 1e3, "kv_bytes": 2 * H * C * dh * 2,
                  "state_bytes": H * dh * dh * 4}
    return out
