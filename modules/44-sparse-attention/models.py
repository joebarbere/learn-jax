"""Module 44 model code: one small Transformer, five ways of choosing which past tokens each query reads.

Provided (not exercises): the model (Module 41's attention model: 8 layers of width 384, 6 heads of 64,
gated attention with RoPE, SwiGLU, tied 8K embeddings) where every layer gets a mask built from your
exercises; DeepSeek Sparse Attention's indexer and its training signal; the training loop; the attention-sink
experiment on SmolLM2-135M; and the timing helpers.

Sparse attention here is *masked dense* attention: it computes all T² scores and hides most of them, so it
shows what each pattern does to the model's quality, not how fast a real kernel would be. Section 6 of the
lesson times a real O(T·w) kernel (your banded_attention) separately.
"""
from __future__ import annotations

import importlib.util
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent / "runs" / "44-sparse-attention"
VOCAB = 8192
SEQ = 2048                                   # long enough that a 128-token window really is local
BASE = {"d_model": 384, "n_heads": 6, "d_head": 64, "d_ff": 1024, "n_layers": 8, "rope_theta": 1e4,
        "window": SEQ, "n_sink": 0, "local": "", "sparse": None, "block": 64, "top_blocks": 2, "top_k": 128,
        "idx_heads": 4, "idx_dim": 32, "dense_warmup": 500}
VARIANTS = {                                 # the race: same model, same data; only the mask differs
    "full": {},                                                        # every past token
    "swa": {"window": 128},                                            # the last 128 tokens, every layer
    "swa-sink": {"window": 124, "n_sink": 4},                          # 124 recent + the first 4 (StreamingLLM)
    "local-global": {"window": 128, "local": "LLLGLLLG"},              # 3 local : 1 global (Gemma 3 uses 5 : 1)
    "moba": {"sparse": "moba", "block": 64, "top_blocks": 2},          # own block + 2 chosen 64-token blocks
    "dsa": {"sparse": "dsa", "top_k": 128},                            # 128 tokens chosen by the indexer
}


def variant(name: str, **over) -> dict:
    return {**BASE, **VARIANTS[name], **over}


def layer_windows(cfg, T):
    """Per-layer window and sink count, as arrays the layer scan can carry."""
    kinds = cfg["local"] or "L" * cfg["n_layers"]
    w = [cfg["window"] if c == "L" else T for c in kinds]
    return jnp.asarray(w, jnp.int32), jnp.full((cfg["n_layers"],), cfg["n_sink"], jnp.int32)


# ---------------------------------------------------------------- pieces
def rms_norm(x, w, eps: float = 1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps)).astype(x.dtype) * w


def rope(x, theta: float = 1e4):
    T, half = x.shape[-2], x.shape[-1] // 2
    ang = jnp.arange(T)[:, None] * (1.0 / theta ** (jnp.arange(half) / half))[None]
    cos, sin = jnp.cos(ang).astype(x.dtype), jnp.sin(ang).astype(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def _dense(key, fan_in, shape):
    return jax.random.normal(key, shape, jnp.float32) * fan_in ** -0.5


def init_model(key, cfg, vocab=VOCAB):
    d, H, dh, F, L = cfg["d_model"], cfg["n_heads"], cfg["d_head"], cfg["d_ff"], cfg["n_layers"]
    ks = jax.random.split(key, 10)
    out = 1 / np.sqrt(2 * L)
    layers = {"n1": jnp.ones((L, d)), "n2": jnp.ones((L, d)),
              "wqkv": _dense(ks[1], d, (L, d, 3 * H * dh)), "wg": _dense(ks[2], d, (L, d, H * dh)),
              "wo": _dense(ks[3], H * dh, (L, H * dh, d)) * out, "w1": _dense(ks[4], d, (L, d, F)),
              "w3": _dense(ks[5], d, (L, d, F)), "w2": _dense(ks[6], F, (L, F, d)) * out}
    if cfg["sparse"] == "dsa":                                        # the lightning indexer, per layer
        Hi, di = cfg["idx_heads"], cfg["idx_dim"]
        layers |= {"iq": _dense(ks[7], d, (L, d, Hi * di)), "ik": _dense(ks[8], d, (L, d, di)),
                   "iw": _dense(ks[9], d, (L, d, Hi)) * 0.1}
    return {"embed": jax.random.normal(ks[0], (vocab, d)) * 0.02, "layers": layers, "norm": jnp.ones((d,))}


def count_params(params) -> int:
    return sum(int(np.prod(a.shape)) for a in jax.tree.leaves(params))


def _bf16(params):
    return jax.tree.map(lambda a: a.astype(jnp.bfloat16), params)


# ---------------------------------------------------------------- the model
def dsa_indexer(parts, lp, h, cfg):
    """Index scores (B, T, T) from the layer's (detached) input, as DeepSeek V3.2 does."""
    B, T, _ = h.shape
    h = jax.lax.stop_gradient(h)
    q = (h @ lp["iq"]).reshape(B, T, cfg["idx_heads"], cfg["idx_dim"])
    return parts.indexer_scores(q, h @ lp["ik"], h @ lp["iw"]).astype(jnp.float32)


def indexer_loss(probs, scores, mask):
    """DSA's training signal for the indexer: KL from the main attention (summed over heads, renormalized,
    no gradient) to the indexer's softmax, over the keys the query actually read."""
    target = jax.lax.stop_gradient(probs.astype(jnp.float32).sum(1))
    target = target / jnp.maximum(target.sum(-1, keepdims=True), 1e-9)
    logq = jax.nn.log_softmax(jnp.where(mask, scores, -1e9), -1)
    kl = jnp.where(mask, target * (jnp.log(jnp.maximum(target, 1e-9)) - logq), 0.0)
    return kl.sum(-1).mean()


def attention_layer(parts, lp, x, window, n_sink, cfg, sparse_on):
    """One layer. Returns (x, indexer loss, share of causal keys each query read)."""
    B, T, _ = x.shape
    H, dh = cfg["n_heads"], cfg["d_head"]
    h = rms_norm(x, lp["n1"])
    q, k, v = (a.reshape(B, T, H, dh).transpose(0, 2, 1, 3) for a in jnp.split(h @ lp["wqkv"], 3, -1))
    q, k = rope(q, cfg["rope_theta"]), rope(k, cfg["rope_theta"])
    causal = jnp.tril(jnp.ones((T, T), bool))
    aux = jnp.zeros((), jnp.float32)
    if cfg["sparse"] == "moba":
        mask = jnp.where(sparse_on, parts.moba_mask(q, k, cfg["block"], cfg["top_blocks"]), causal)
    elif cfg["sparse"] == "dsa":
        scores = dsa_indexer(parts, lp, h, cfg)
        sel = parts.topk_causal_mask(scores, cfg["top_k"])
        mask = jnp.where(sparse_on, sel, causal)[:, None]
    else:
        mask = parts.window_mask(T, window, n_sink)
    s = jnp.einsum("bhtd,bhsd->bhts", q, k).astype(jnp.float32) / np.sqrt(dh)
    p = jax.nn.softmax(jnp.where(mask, s, -jnp.inf), -1)
    if cfg["sparse"] == "dsa":
        aux = indexer_loss(p, scores, mask[:, 0])
    o = jnp.einsum("bhts,bhsd->bhtd", p.astype(x.dtype), v).transpose(0, 2, 1, 3).reshape(B, T, H * dh)
    x = x + (o * jax.nn.sigmoid(h @ lp["wg"])) @ lp["wo"]
    h = rms_norm(x, lp["n2"])
    x = x + (jax.nn.silu(h @ lp["w1"]) * (h @ lp["w3"])) @ lp["w2"]
    read = jnp.broadcast_to(mask, (B, H, T, T)).sum((-1, -2)).mean() / (T * (T + 1) / 2)
    return x, aux, read


def forward(parts, params, tokens, cfg, sparse_on=True, remat=False):
    """Logits (B, T, V), the summed indexer loss, and the mean share of causal keys read per layer."""
    T = tokens.shape[1]
    windows, sinks = layer_windows(cfg, T)

    def body(x, xs):
        lp, w, s = xs
        x, aux, read = attention_layer(parts, lp, x, w, s, cfg, sparse_on)
        return x, (aux, read)

    if remat:
        body = jax.checkpoint(body)
    x, (aux, read) = jax.lax.scan(body, params["embed"][tokens], (params["layers"], windows, sinks))
    return rms_norm(x, params["norm"]) @ params["embed"].T, aux.sum(), read


# ---------------------------------------------------------------- training
def cross_entropy(logits, targets):
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    return -jnp.take_along_axis(logp, targets[..., None], -1)[..., 0]


def make_optimizer(lr, steps, warmup=200):
    warmup = min(warmup, steps // 2)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(steps, warmup + 1), lr * 0.1)

    def opt():
        return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=0.1))

    # DSA's indexer gets its own clipping and Adam state. With one shared global-norm clip, the indexer's large
    # KL gradients set the norm and shrink every update to the language model (it fell behind even while dense).
    def labels(params):
        return {k: ({n: ("indexer" if n in ("iq", "ik", "iw") else "lm") for n in v} if k == "layers"
                    else "lm") for k, v in params.items()}

    return optax.multi_transform({"lm": opt(), "indexer": opt()}, labels)


def make_trainer(parts, cfg, optimizer, batch, seq_len):
    def loss_fn(params, x, y, sparse_on):
        logits, aux, _ = forward(parts, _bf16(params), x, cfg, sparse_on, remat=True)
        ce = cross_entropy(logits, y).mean()
        return ce + aux, ce

    def step(carry, inp):
        params, opt_state, data = carry
        key, sparse_on = inp
        starts = jax.random.randint(key, (batch,), 0, data.shape[0] - seq_len - 1)
        win = jax.vmap(lambda s: jax.lax.dynamic_slice(data, (s,), (seq_len + 1,)))(starts)
        (_, ce), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, win[:, :-1], win[:, 1:], sparse_on)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, data), ce

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_chunk(params, opt_state, data, keys, sparse_on):
        (params, opt_state, _), losses = jax.lax.scan(step, (params, opt_state, data), (keys, sparse_on))
        return params, opt_state, losses

    @jax.jit
    def eval_pos(params, x, y):
        """Loss at every position, averaged over the batch, and the share of keys read."""
        logits, _, read = forward(parts, _bf16(params), x, cfg)
        return cross_entropy(logits, y).mean(0), read

    return train_chunk, eval_pos


def val_batches(val, batch, seq_len, n=8, seed=1234):
    rng = np.random.default_rng(seed)
    out = []
    for row in rng.integers(0, len(val) - seq_len - 1, (n, batch)):
        w = np.stack([val[s:s + seq_len + 1] for s in row]).astype(np.int32)
        out.append((jnp.asarray(w[:, :-1]), jnp.asarray(w[:, 1:])))
    return out


def save(params, cfg, out: Path, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    flat = {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    np.savez(out / "params.tmp.npz", **flat)
    (out / "params.tmp.npz").replace(out / "params.npz")
    (out / "config.json").write_text(json.dumps({"model": cfg, **(extra or {})}, indent=2))


def load(out: Path):
    cfg = json.loads((out / "config.json").read_text())["model"]
    arrays = np.load(out / "params.npz")
    template = jax.eval_shape(lambda: init_model(jax.random.key(0), cfg))
    params = jax.tree_util.tree_map_with_path(lambda p, _: jnp.asarray(arrays[jax.tree_util.keystr(p)]), template)
    return params, cfg


def read_metrics(path):
    try:
        return [json.loads(line) for line in open(path)]
    except OSError:
        return []


# ---------------------------------------------------------------- attention sinks on SmolLM2
def _m42():
    spec = importlib.util.spec_from_file_location("m42_models", HERE.parent / "42-small-models" / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def smol_masked_forward(params, cfg, tokens, mask):
    """learnjax.llama's forward (no cache) with an arbitrary (T, T) attention mask instead of the causal one."""
    from learnjax import llama

    B, T = tokens.shape
    H, KV, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    positions = jnp.broadcast_to(jnp.arange(T), (B, T))

    def layer(x, lp):
        h = llama.rms_norm(x, lp["attn_norm"], cfg.norm_eps)
        q = llama.rope((h @ lp["wq"]).reshape(B, T, H, hd), positions, cfg.rope_theta)
        k = llama.rope((h @ lp["wk"]).reshape(B, T, KV, hd), positions, cfg.rope_theta)
        v = (h @ lp["wv"]).reshape(B, T, KV, hd)
        qg = q.reshape(B, T, KV, H // KV, hd)
        s = jnp.einsum("btgrd,bsgd->bgrts", qg, k).astype(jnp.float32) / jnp.sqrt(hd)
        w = jax.nn.softmax(jnp.where(mask, s, -jnp.inf), -1).astype(x.dtype)
        x = x + jnp.einsum("bgrts,bsgd->btgrd", w, v).reshape(B, T, H * hd) @ lp["wo"]
        h = llama.rms_norm(x, lp["mlp_norm"], cfg.norm_eps)
        return x + (jax.nn.silu(h @ lp["w_gate"]) * (h @ lp["w_up"])) @ lp["w_down"], None

    x, _ = jax.lax.scan(layer, params["embed"][tokens], params["layers"])
    x = llama.rms_norm(x, params["final_norm"], cfg.norm_eps)
    head = params["embed"].T if cfg.tied else params["head"]
    return (x @ head).astype(jnp.float32)


def sink_experiment(parts, T=2048, window=256, n=8, batch=2):
    """Per-position loss of SmolLM2-135M on detective text with: full causal attention, a sliding window,
    the window plus 4 sink tokens, and a window 4 tokens wider instead (the same number of keys)."""
    from learnjax import llama

    params, cfg, _ = llama.load(llama.SMOLLM_135M, max_seq=T)
    val = _m42().smol_corpus()["val"]
    masks = {"full": parts.window_mask(T, T), f"window {window}": parts.window_mask(T, window),
             f"window {window} + 4 sinks": parts.window_mask(T, window, 4),
             f"window {window + 4}": parts.window_mask(T, window + 4)}
    fn = jax.jit(lambda p, x, y, m: cross_entropy(smol_masked_forward(p, cfg, x, m), y).mean(0))
    out = {}
    batches = val_batches(val, batch, T, n=n // batch, seed=44)
    for name, m in masks.items():
        out[name] = np.mean([np.asarray(fn(params, x, y, m)) for x, y in batches], 0).round(4).tolist()
        print(name, f"{np.mean(out[name]):.3f}", flush=True)
    return out


def attention_to_first(parts, T=1024, n_heads_show=None):
    """How much attention SmolLM2-135M's queries pay to token 0, per layer (mean over heads and queries
    past position 64): the 'sink' StreamingLLM found."""
    from learnjax import llama

    params, cfg, _ = llama.load(llama.SMOLLM_135M, max_seq=T)
    val = _m42().smol_corpus()["val"]
    x = jnp.asarray(val[1000:1000 + T].astype(np.int32))[None]
    B, H, KV, hd = 1, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    positions = jnp.arange(T)[None]

    @jax.jit
    def run(params, x):
        def layer(h0, lp):
            h = llama.rms_norm(h0, lp["attn_norm"], cfg.norm_eps)
            q = llama.rope((h @ lp["wq"]).reshape(B, T, H, hd), positions, cfg.rope_theta)
            k = llama.rope((h @ lp["wk"]).reshape(B, T, KV, hd), positions, cfg.rope_theta)
            v = (h @ lp["wv"]).reshape(B, T, KV, hd)
            s = jnp.einsum("btgrd,bsgd->bgrts", q.reshape(B, T, KV, H // KV, hd), k).astype(jnp.float32)
            w = jax.nn.softmax(jnp.where(jnp.tril(jnp.ones((T, T), bool)), s / jnp.sqrt(hd), -jnp.inf), -1)
            first = w[..., 64:, 0].mean()
            h0 = h0 + jnp.einsum("bgrts,bsgd->btgrd", w.astype(h0.dtype), v).reshape(B, T, H * hd) @ lp["wo"]
            h = llama.rms_norm(h0, lp["mlp_norm"], cfg.norm_eps)
            return h0 + (jax.nn.silu(h @ lp["w_gate"]) * (h @ lp["w_up"])) @ lp["w_down"], first
        return jax.lax.scan(layer, params["embed"][x], params["layers"])[1]

    return np.asarray(run(params, x)).round(4).tolist()


# ---------------------------------------------------------------- timing
def time_fn(fn, *args, reps=5):
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best


def bench_prefill(parts, lengths=(1024, 2048, 4096, 8192, 16384, 32768), window=512, H=6, dh=64):
    """One attention layer's prefill time (batch 1, bf16): dense causal attention vs your banded sliding
    window. Dense stops where its T × T scores no longer fit."""
    dense = jax.jit(lambda q, k, v: jnp.einsum("bhts,bhsd->bhtd", jax.nn.softmax(jnp.where(
        jnp.tril(jnp.ones((q.shape[2],) * 2, bool)),
        jnp.einsum("bhtd,bhsd->bhts", q, k).astype(jnp.float32) / np.sqrt(dh), -jnp.inf), -1).astype(v.dtype), v))
    banded = jax.jit(lambda q, k, v: parts.banded_attention(q, k, v, window))
    out = {}
    for T in lengths:
        ks = jax.random.split(jax.random.key(T), 3)
        q, k, v = (jax.random.normal(kk, (1, H, T, dh), jnp.bfloat16) for kk in ks)
        row = {"banded_ms": time_fn(banded, q, k, v) * 1e3}
        if T <= 8192:                                                  # 16K: 6 heads × 16K² f32 scores = 6 GB
            row["dense_ms"] = time_fn(dense, q, k, v) * 1e3
        out[T] = row
        print(T, {k_: round(v_, 2) for k_, v_ in row.items()}, flush=True)
    return out
