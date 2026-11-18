"""Module 40: a DeepSeek-V3-style model (MLA + DeepSeekMoE) with two optional upgrades, built from *your* parts.

* **LatentMoE** (Elango et al., NVIDIA, 2026): routed experts work in a smaller latent width ℓ = d/α, with α× as
  many experts (and, in the "acc" variant, α× as many picked per token). Uses your `latent_moe`.
* **Engram** (DeepSeek, 2026): a hashed n-gram lookup table read at an early layer, gated by the hidden state and
  smoothed by a short causal convolution. Uses your `compress_vocab` and `ngram_hashes` (on the host, in NumPy)
  and `engram_gate`, `causal_dwconv` (on the GPU).

`parts` is your exercises module (or the reference solutions). Module 17's pieces — RMSNorm, RoPE, MLA, top-k
routing with aux-loss-free bias balancing — are included below as reference code, so this module doesn't depend
on your Module 17 answers. The MLA and MoE layers follow deepseek_v3.py from
github.com/Vaibhaav-Tiwari/deepseek-v3-jax (MIT), as Module 17's do. The hashing helpers follow
engram_demo_v1.py from github.com/deepseek-ai/Engram (Apache-2.0) and match it bit for bit.

Params are plain nested dicts of arrays. Configs are plain dicts.
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

# deepseek-v3-jax's Config (the owner's replication), without multi-token prediction and the sequence-wise
# balance loss: 6 layers of width 384, 1 dense + 5 MoE layers, 1 shared + 16 routed experts, top-4.
BASE = {"d_model": 384, "n_layers": 6, "n_dense": 1, "n_heads": 6, "q_rank": 192, "kv_rank": 128,
        "d_nope": 32, "d_rope": 16, "d_v": 32, "d_ff": 1024, "n_shared": 1, "n_routed": 16, "top_k": 4,
        "d_expert": 256, "rope_theta": 10000.0, "gamma": 1e-3}
# The lesson's live runs: Module 17's tiny width, on the same BPE tokens.
SMALL = {"d_model": 128, "n_layers": 4, "n_dense": 1, "n_heads": 4, "q_rank": 64, "kv_rank": 32,
         "d_nope": 32, "d_rope": 16, "d_v": 32, "d_ff": 352, "n_shared": 1, "n_routed": 8, "top_k": 2,
         "d_expert": 96, "rope_theta": 10000.0, "gamma": 1e-3}
ENGRAM = {"orders": (2, 3), "heads": 4, "slot": 24, "kernel": 4, "dilation": 3, "seed": 0}


def variant(cfg: dict, name: str, alpha: int = 4, rho: float = 1.0, engram_layer: int = 1,
            rows: int | None = None) -> dict:
    """A config for one of the module's variants.

    moe          the baseline (cfg as given)
    latent-acc   routed experts in width d/alpha; alpha× experts, alpha× top-k (LatentMoE's recommended variant)
    latent-eff   the same, but top-k unchanged: fewer FLOPs and bytes per token
    engram       a fraction rho of the routed experts kept, the freed parameters moved into one Engram table
                 read at layer `engram_layer` (0-based), with the same top-k (same active compute).
                 `rows` (per head) overrides the iso-parameter size.
    """
    c = dict(cfg)
    if name == "moe":
        return c
    if name in ("latent-acc", "latent-eff"):
        c["latent"] = cfg["d_model"] // alpha
        c["n_routed"] = cfg["n_routed"] * alpha
        c["top_k"] = cfg["top_k"] * (alpha if name == "latent-acc" else 1)
        return c
    if name == "engram":
        kept = max(round(cfg["n_routed"] * rho), cfg["top_k"])
        n_moe = cfg["n_layers"] - cfg["n_dense"]
        freed = (cfg["n_routed"] - kept) * 3 * cfg["d_model"] * cfg["d_expert"] * n_moe
        e = dict(ENGRAM, layer=engram_layer)
        n_slots = len(e["orders"]) * e["heads"]
        d_mem = n_slots * e["slot"]
        proj = 2 * d_mem * cfg["d_model"] + e["kernel"] * cfg["d_model"] + 3 * cfg["d_model"]
        e["rows"] = rows or max((freed - proj) // (n_slots * e["slot"]), 1009)
        c["n_routed"], c["engram"] = kept, e
        return c
    raise ValueError(name)


def _normal(key, shape, std=0.02):
    return jax.random.normal(key, shape) * std


def _swiglu_params(key, d, h):
    k1, k2, k3 = jax.random.split(key, 3)
    return {"w1": _normal(k1, (d, h)), "w3": _normal(k2, (d, h)), "w2": _normal(k3, (h, d))}


# ------------------------------------------------------------------------------- Module 17's pieces
def rms_norm(x, w, eps: float = 1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps)).astype(x.dtype) * w


def rope(x, positions, theta: float = 10000.0):
    """Rotate-half RoPE on the last axis. x: (B, T, ..., D), positions: (T,)."""
    half = x.shape[-1] // 2
    freqs = 1.0 / theta ** (jnp.arange(half) / half)
    ang = positions[:, None].astype(jnp.float32) * freqs[None]                     # (T, half)
    ang = ang.reshape((1, ang.shape[0]) + (1,) * (x.ndim - 3) + (half,))
    cos, sin = jnp.cos(ang).astype(x.dtype), jnp.sin(ang).astype(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def swiglu(p, x):
    return (jax.nn.silu(x @ p["w1"]) * (x @ p["w3"])) @ p["w2"]


def moe_route(scores, bias, top_k: int):
    """Module 17: pick top_k by score + bias; gate = the raw scores of the picked experts, normalized."""
    _, idx = jax.lax.top_k(scores + jax.lax.stop_gradient(bias), top_k)
    picked = jax.nn.one_hot(idx, scores.shape[-1]).sum(-2) > 0
    gated = jnp.where(picked, scores, 0.0)
    return gated / gated.sum(-1, keepdims=True)


def update_bias(bias, load, gamma: float):
    """Module 17: nudge each expert's bias toward the mean load (aux-loss-free balancing)."""
    return bias + gamma * jnp.sign(load.mean(-1, keepdims=True) - load)


def mla(p, x, positions, cfg):
    """Multi-head Latent Attention, training path (no cache), as in Module 17."""
    B, T, _ = x.shape
    H, dn, dr, dv = cfg["n_heads"], cfg["d_nope"], cfg["d_rope"], cfg["d_v"]
    theta = cfg["rope_theta"]
    q = (rms_norm(x @ p["wq_a"], p["q_norm"]) @ p["wq_b"]).reshape(B, T, H, dn + dr)
    q = jnp.concatenate([q[..., :dn], rope(q[..., dn:], positions, theta)], -1)
    kv_a = x @ p["wkv_a"]
    c = rms_norm(kv_a[..., :cfg["kv_rank"]], p["kv_norm"])
    kr = rope(kv_a[..., cfg["kv_rank"]:], positions, theta)
    kv = (c @ p["wkv_b"]).reshape(B, T, H, dn + dv)
    k = jnp.concatenate([kv[..., :dn], jnp.broadcast_to(kr[:, :, None], (B, T, H, dr))], -1)
    v = kv[..., dn:]
    scores = jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32) / jnp.sqrt(dn + dr)
    causal = jnp.tril(jnp.ones((T, T), bool))
    attn = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), -1).astype(x.dtype)
    return jnp.einsum("bhts,bshd->bthd", attn, v).reshape(B, T, H * dv) @ p["wo"]


def dense_moe(p, xf, gates):
    """Module 17's DeepSeekMoE with dense compute: every expert on every token, unpicked experts gated to 0."""
    w = p["experts"]
    h = jax.nn.silu(jnp.einsum("nd,edh->neh", xf, w["w1"])) * jnp.einsum("nd,edh->neh", xf, w["w3"])
    routed = jnp.einsum("ne,ned->nd", gates, jnp.einsum("neh,ehd->ned", h, w["w2"]))
    return routed + swiglu(p["shared"], xf)


# ------------------------------------------------------------------------------- init
def init_model(key, vocab, cfg):
    d, H = cfg["d_model"], cfg["n_heads"]
    dn, dr, dv, E = cfg["d_nope"], cfg["d_rope"], cfg["d_v"], cfg["n_routed"]
    w = cfg.get("latent", d)                     # the routed experts' width: d, or the latent ℓ
    keys = jax.random.split(key, cfg["n_layers"] + 3)
    layers = []
    for i, k in enumerate(keys[:cfg["n_layers"]]):
        ka, kf = jax.random.split(k)
        k1, k2, k3, k4, k5 = jax.random.split(ka, 5)
        layer = {
            "attn_norm": jnp.ones(d),
            "attn": {"wq_a": _normal(k1, (d, cfg["q_rank"])), "q_norm": jnp.ones(cfg["q_rank"]),
                     "wq_b": _normal(k2, (cfg["q_rank"], H * (dn + dr))),
                     "wkv_a": _normal(k3, (d, cfg["kv_rank"] + dr)), "kv_norm": jnp.ones(cfg["kv_rank"]),
                     "wkv_b": _normal(k4, (cfg["kv_rank"], H * (dn + dv))), "wo": _normal(k5, (H * dv, d))},
            "ffn_norm": jnp.ones(d),
        }
        if i < cfg["n_dense"]:
            layer["ffn"] = _swiglu_params(kf, d, cfg["d_ff"])
        else:
            kr, ks, k1, k2, k3, kd, ku = jax.random.split(kf, 7)
            h = cfg["d_expert"]
            moe = {"router": _normal(kr, (d, E)),
                   "shared": _swiglu_params(ks, d, cfg["n_shared"] * h),
                   "experts": {"w1": _normal(k1, (E, w, h)), "w3": _normal(k2, (E, w, h)),
                               "w2": _normal(k3, (E, h, w))}}
            if "latent" in cfg:
                moe["down"], moe["up"] = _normal(kd, (d, w)), _normal(ku, (w, d))
            layer["moe"] = moe
        layers.append(layer)
    params = {"embed": _normal(keys[-3], (vocab, d)), "layers": layers, "norm_f": jnp.ones(d),
              "head": _normal(keys[-2], (d, vocab))}
    if "engram" in cfg:
        params["engram"] = init_engram(keys[-1], cfg)
    return params


def init_engram(key, cfg):
    e, d = cfg["engram"], cfg["d_model"]
    n_slots = len(e["orders"]) * e["heads"]
    d_mem = n_slots * e["slot"]
    k1, k2, k3 = jax.random.split(key, 3)
    total_rows = sum(sum(p) for p in head_primes(e["rows"], e["orders"], e["heads"]))
    return {"table": _normal(k1, (total_rows, e["slot"])),             # every head's table, stacked
            "w_k": _normal(k2, (d_mem, d)), "w_v": _normal(k3, (d_mem, d)),
            "norm_h": jnp.ones(d), "norm_k": jnp.ones(d), "norm_v": jnp.ones(d),
            "conv": jnp.zeros((e["kernel"], d))}                       # zero-init: the conv starts switched off


# ------------------------------------------------------------------------------- forward
def gather_rows(table, hash_ids, cfg):
    """(B, T, n_slots) per-head row indices -> (B, T, n_slots * slot) memory vectors, from the stacked table.
    Works on a jnp table (on the GPU) or a NumPy one (in host memory) alike."""
    e = cfg["engram"]
    offsets = np.cumsum([0] + [s for p in head_primes(e["rows"], e["orders"], e["heads"]) for s in p])[:-1]
    xp = np if isinstance(table, np.ndarray) else jnp
    rows = xp.take(table, hash_ids + xp.asarray(offsets, dtype=hash_ids.dtype), axis=0)
    return rows.reshape(*hash_ids.shape[:-1], -1)


def engram_block(parts, p, h, hash_ids, cfg, rows=None):
    """Engram (eq. 3–5 of the paper): read the n-gram rows, gate them by h, smooth with a short causal conv.

    hash_ids: (B, T, n_slots) row indices *within* each head's table (from your ngram_hashes); head j's table
    starts at row offsets[j] of the stacked table (each head's table is a different prime size).
    rows: the memory vectors already gathered (e.g. prefetched from host memory) — then hash_ids is unused."""
    e = cfg["engram"]
    emb = (gather_rows(p["table"], hash_ids, cfg) if rows is None else rows).astype(h.dtype)
    v, alpha = parts.engram_gate(h, emb, p)
    y = v + jax.nn.silu(parts.causal_dwconv(rms_norm(v, p["norm_v"]), p["conv"].astype(v.dtype), e["dilation"]))
    return y, alpha


def forward(parts, params, tokens, hash_ids, biases, cfg, engram_scale=1.0, return_gates=False, engram_rows=None):
    """Logits and per-MoE-layer expert loads. hash_ids: (B, T, n_slots) or None (no Engram).
    engram_scale=0 switches Engram off at inference (the paper's suppression test). engram_rows: pre-gathered
    memory vectors (B, T, d_mem), used instead of reading the table."""
    positions = jnp.arange(tokens.shape[1])
    x = params["embed"][tokens]
    loads, gate_vals = [], None
    for i, p in enumerate(params["layers"]):
        if "engram" in cfg and i == cfg["engram"]["layer"]:
            y, gate_vals = engram_block(parts, params["engram"], x, hash_ids, cfg, engram_rows)
            x = x + engram_scale * y                                         # before attention, like the paper
        x = x + mla(p["attn"], rms_norm(x, p["attn_norm"]), positions, cfg)
        h = rms_norm(x, p["ffn_norm"])
        if "moe" in p:
            B, T, d = h.shape
            hf = h.reshape(B * T, d)
            scores = jax.nn.sigmoid(hf.astype(jnp.float32) @ p["moe"]["router"].astype(jnp.float32))
            gates = moe_route(scores, biases[i - cfg["n_dense"]], cfg["top_k"]).astype(h.dtype)
            f = (parts.latent_moe(p["moe"], hf, gates) if "latent" in cfg else dense_moe(p["moe"], hf, gates))
            loads.append((gates > 0).mean(0) / cfg["top_k"])
            f = f.reshape(B, T, d)
        else:
            f = swiglu(p["ffn"], h)
        x = x + f
    logits = rms_norm(x, params["norm_f"]) @ params["head"]
    return (logits, jnp.stack(loads), gate_vals) if return_gates else (logits, jnp.stack(loads))


def init_biases(cfg):
    return jnp.zeros((cfg["n_layers"] - cfg["n_dense"], cfg["n_routed"]))


# ------------------------------------------------------------------------------- counting
def count_params(params) -> int:
    return sum(int(a.size) for a in jax.tree.leaves(params))


def param_report(params, cfg) -> dict:
    """Total, active per token, and where they sit (millions)."""
    total = count_params(params)
    routed = sum(count_params(p["moe"]["experts"]) for p in params["layers"] if "moe" in p)
    engram_table = int(params["engram"]["table"].size) if "engram" in params else 0
    n_slots = len(cfg["engram"]["orders"]) * cfg["engram"]["heads"] if "engram" in cfg else 0
    # active: routed experts scale by top_k / n_routed; the table only by the rows actually read
    table_active = n_slots * cfg["engram"]["slot"] if "engram" in cfg else 0
    active = total - routed * (1 - cfg["top_k"] / cfg["n_routed"]) - engram_table + table_active
    return {"total": total / 1e6, "active": active / 1e6, "routed": routed / 1e6, "engram": engram_table / 1e6}


# ------------------------------------------------------------------------------- Engram, host side
def token_texts(tokenizer) -> list[str]:
    """Each token's text on its own. Byte-level tokens that are only part of a multi-byte UTF-8 character
    decode to the replacement character U+FFFD; compress_vocab never merges those (DeepSeek's demo keys them
    by their raw token names, which are all different)."""
    return [tokenizer.decode([i], skip_special_tokens=False) for i in range(tokenizer.get_vocab_size())]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    return all(n % f for f in range(3, math.isqrt(n) + 1, 2))


def head_primes(rows: int, orders, heads: int) -> list[list[int]]:
    """One distinct prime table size per (order, head), searching upward from `rows` (engram_demo_v1.py's
    find_next_prime). Distinct primes make the heads' collisions independent."""
    seen, out = set(), []
    for _ in orders:
        start, sizes = rows - 1, []
        for _ in range(heads):
            c = start + 1
            while not _is_prime(c) or c in seen:
                c += 1
            seen.add(c)
            sizes.append(c)
            start = c
        out.append(sizes)
    return out


def multipliers(vocab_size: int, max_n: int, seed: int, layer_id: int) -> np.ndarray:
    """Odd random int64 multipliers, one per n-gram position (engram_demo_v1.py). Each is below
    int64_max / vocab_size, so id × multiplier never overflows."""
    half_bound = max(1, int(np.iinfo(np.int64).max // vocab_size) // 2)
    r = np.random.default_rng(int(seed + 10007 * layer_id)).integers(0, half_bound, size=(max_n,), dtype=np.int64)
    return r * 2 + 1


class Hasher:
    """Everything the host needs to turn token ids into Engram row indices for one Engram layer."""

    def __init__(self, parts, texts: list[str], cfg: dict, pad_token: int = 0):
        e = cfg["engram"]
        self.parts, self.e = parts, e
        self.lookup, self.n_new = parts.compress_vocab(texts)
        self.lookup = np.asarray(self.lookup, np.int64)
        self.pad = int(self.lookup[pad_token])
        self.mult = multipliers(self.n_new, max(e["orders"]), e["seed"], e["layer"])
        self.primes = head_primes(e["rows"], e["orders"], e["heads"])

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        """ids (B, T) raw token ids -> (B, T, n_slots) int32 row indices within each head's table."""
        x = self.lookup[np.asarray(ids, np.int64)]
        return np.asarray(self.parts.ngram_hashes(x, self.mult, self.primes, self.pad)).astype(np.int32)

    def corpus(self, ids: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
        """A whole token stream (L,) -> (L, n_slots), in chunks. Each chunk is hashed with the (n-1) tokens
        before it as context, so the result equals hashing the stream in one go (position 0 sees padding)."""
        ctx = max(self.e["orders"]) - 1
        out = []
        for s in range(0, len(ids), chunk):
            lo = max(s - ctx, 0)
            out.append(self(np.asarray(ids[lo:s + chunk])[None])[0, s - lo:])
        return np.concatenate(out)


# ------------------------------------------------------------------------------- training
def get_windows(key, data, hashes, batch, seq_len):
    """B random windows of tokens (and their precomputed hash rows) at the same offsets."""
    starts = jax.random.randint(key, (batch,), 0, data.shape[0] - seq_len - 1)
    win = jax.vmap(lambda s: jax.lax.dynamic_slice(data, (s,), (seq_len + 1,)))(starts)
    hw = None
    if hashes is not None:
        hw = jax.vmap(lambda s: jax.lax.dynamic_slice(hashes, (s, 0), (seq_len, hashes.shape[1])))(starts)
    return win[:, :-1], win[:, 1:], hw


def cross_entropy(logits, targets):
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    return -jnp.take_along_axis(logp, targets[..., None], -1).mean()


def make_optimizer(params, lr, steps, warmup=200, table_lr_mult=5.0):
    """deepseek-v3-jax's AdamW (warmup-cosine, clip 1.0, wd 0.1), except Engram's table: Adam with 5× the
    learning rate and no weight decay (the paper's Table 5)."""
    warmup = min(warmup, steps // 2)
    sched = lambda m: optax.warmup_cosine_decay_schedule(0.0, lr * m, warmup, max(steps, warmup + 1), lr * m * 0.1)
    labels = jax.tree_util.tree_map_with_path(
        lambda path, _: "table" if "table" in jax.tree_util.keystr(path) else "rest", params)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.multi_transform(
        {"rest": optax.adamw(sched(1.0), b1=0.9, b2=0.95, weight_decay=0.1),
         "table": optax.adam(sched(table_lr_mult), b1=0.9, b2=0.95)}, labels))


def _bf16(params):
    """bf16 compute copy of the f32 master weights — except Engram's table, which stays f32: rows are gathered
    in f32 and cast afterwards (Module 38 hit a ROCm codegen bug gathering from a big bf16 table)."""
    return jax.tree_util.tree_map_with_path(
        lambda path, a: a if "table" in jax.tree_util.keystr(path) else a.astype(jnp.bfloat16), params)


def make_trainer(parts, cfg, optimizer, batch=32, seq_len=256):
    """train_chunk(params, opt_state, biases, data, hashes, key, n_steps): n_steps under one jit (lax.scan),
    bf16 compute, bias balancing after every step. eval_loss(params, biases, x, y, hw) in f32."""

    def loss_fn(params, biases, x, y, hw):
        logits, loads = forward(parts, _bf16(params), x, hw, biases, cfg)
        return cross_entropy(logits, y), loads

    def step(carry, key):
        params, opt_state, biases, data, hashes = carry
        x, y, hw = get_windows(key, data, hashes, batch, seq_len)
        (loss, loads), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, biases, x, y, hw)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        biases = update_bias(biases, loads, cfg["gamma"])
        return (params, opt_state, biases, data, hashes), (loss, loads)

    @partial(jax.jit, static_argnames="n_steps", donate_argnums=(0, 1))
    def train_chunk(params, opt_state, biases, data, hashes, key, n_steps):
        (params, opt_state, biases, _, _), (losses, loads) = jax.lax.scan(
            step, (params, opt_state, biases, data, hashes), jax.random.split(key, n_steps))
        return params, opt_state, biases, losses, loads

    @jax.jit
    def eval_loss(params, biases, x, y, hw):
        return cross_entropy(forward(parts, params, x, hw, biases, cfg)[0], y)

    return train_chunk, eval_loss


def val_batches(val, hashes, batch, seq_len, n=8, seed=1234):
    """Fixed validation windows (the same for every variant): list of (x, y, hash rows or None)."""
    rng = np.random.default_rng(seed)
    out = []
    for row in rng.integers(0, len(val) - seq_len - 1, (n, batch)):
        w = np.stack([val[s:s + seq_len + 1] for s in row]).astype(np.int32)
        hw = None if hashes is None else jnp.asarray(np.stack([hashes[s:s + seq_len] for s in row]))
        out.append((jnp.asarray(w[:, :-1]), jnp.asarray(w[:, 1:]), hw))
    return out


# ------------------------------------------------------------------------------- checkpoints
def save(params, biases, cfg, out, extra=None):
    import json

    out.mkdir(parents=True, exist_ok=True)
    flat = {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    flat["__biases__"] = np.asarray(biases)
    tmp = out / "params.tmp.npz"
    np.savez(tmp, **flat)
    tmp.replace(out / "params.npz")
    cfg_json = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    if "engram" in cfg_json:
        cfg_json["engram"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg["engram"].items()}
    (out / "config.json").write_text(json.dumps({"model": cfg_json, **(extra or {})}, indent=2))


def load(out, vocab=8192):
    """(params, biases, cfg, meta) from a run directory written by train.py."""
    import json

    meta = json.loads((out / "config.json").read_text())
    cfg = meta["model"]
    if "engram" in cfg:
        cfg["engram"]["orders"] = tuple(cfg["engram"]["orders"])
    arrays = np.load(out / "params.npz")
    template = jax.eval_shape(lambda: init_model(jax.random.key(0), vocab, cfg))
    params = jax.tree_util.tree_map_with_path(
        lambda p, _: jnp.asarray(arrays[jax.tree_util.keystr(p)]), template)
    return params, jnp.asarray(arrays["__biases__"]), cfg, meta


def read_metrics(path):
    import json

    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

