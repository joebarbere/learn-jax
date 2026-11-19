"""Module 42 model code: the SmolLM2 family as teacher, reference and pruning source; a small student.

Provided (not exercises): the detective corpus re-tokenized with SmolLM2's tokenizer (distillation needs the
teacher and the student to share a vocabulary), the teacher pass that stores its top-k next-token predictions
once, a SmolLM2-shaped student, the training loop (cross-entropy, distillation, or both), Muon built on your
`newton_schulz`, and the Minitron-style pruning pass that uses your importance scores.

Everything is a learnjax/llama.py model, so the teacher, the reference SmolLM2-135M, the pruned model and the
students are scored on the *same* validation tokens, in the same units (nats per token).
"""
from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from learnjax import llama

TEACHER = "HuggingFaceTB/SmolLM2-360M"          # 360M parameters, 4T tokens, Apache-2.0
REFERENCE = "HuggingFaceTB/SmolLM2-135M"        # 135M parameters, 2T tokens
SEQ = 512
TOP_K = 16
STUDENT = llama.Config(vocab=49152, d_model=384, n_layers=8, n_heads=6, n_kv_heads=2, d_ff=1024,
                       rope_theta=100000.0, tied=True, max_seq=SEQ, dtype="float32")
CACHE = Path("/cache/learn-jax/smol-detective")


# ---------------------------------------------------------------- data
def tokenizer_path() -> Path:
    return llama.download(REFERENCE) / "tokenizer.json"


def smol_corpus() -> dict:
    """The detective corpus in SmolLM2 tokens: {"train", "val"} uint16 arrays (cached after the first call).
    The original 8K-token BPE decodes losslessly to text, which SmolLM2's tokenizer re-encodes."""
    from tokenizers import Tokenizer

    from learnjax.data import detective_corpus

    CACHE.mkdir(parents=True, exist_ok=True)
    if not (CACHE / "val.npy").exists():
        src = detective_corpus()
        old = Tokenizer.from_file(str(src["tokenizer_path"]))
        new = Tokenizer.from_file(str(llama.download(REFERENCE) / "tokenizer.json"))
        for split in ("train", "val"):
            ids = src[split]
            text = "".join(old.decode(ids[i:i + 1_000_000].tolist()) for i in range(0, len(ids), 1_000_000))
            parts = text.split("\n\n")
            enc = new.encode_batch([p + "\n\n" if i < len(parts) - 1 else p for i, p in enumerate(parts)])
            out = np.concatenate([np.asarray(e.ids, np.uint16) for e in enc])
            np.save(CACHE / f"{split}.npy", out)
    return {s: np.load(CACHE / f"{s}.npy") for s in ("train", "val")}


def windows(tokens, seq=SEQ):
    """Non-overlapping windows of seq + 1 tokens (inputs and targets): (n, seq + 1)."""
    n = (len(tokens) - 1) // seq
    return np.lib.stride_tricks.as_strided(tokens, (n, seq + 1), (tokens.strides[0] * seq, tokens.strides[0]))


def val_windows(n=64, seq=SEQ):
    w = windows(smol_corpus()["val"], seq)
    return np.ascontiguousarray(w[np.linspace(0, len(w) - 1, n).astype(int)]).astype(np.int32)


# ---------------------------------------------------------------- the teacher pass
def teacher_pass(out: Path, repo=TEACHER, batch=8, top_k=TOP_K, log=print, limit=None):
    """Run the teacher once over every training window; store its top-k next-token ids (uint16) and
    log-probabilities (float16), and its loss on each window (for the data-quality experiment)."""
    params, cfg, _ = llama.load(repo, max_seq=SEQ)
    w = windows(smol_corpus()["train"])
    n = min(len(w), limit or len(w))
    n -= n % batch
    out.mkdir(parents=True, exist_ok=True)
    ids = np.lib.format.open_memmap(out / "top_ids.npy", "w+", np.uint16, (n, SEQ, top_k))
    logp = np.lib.format.open_memmap(out / "top_logp.npy", "w+", np.float16, (n, SEQ, top_k))
    nll = np.zeros(n, np.float32)

    @jax.jit
    def scores(params, x, y):
        lp = jax.nn.log_softmax(llama.forward(params, x, cfg)[0], -1)
        return lp, -jnp.take_along_axis(lp, y[..., None], -1)[..., 0].mean(-1)

    @jax.jit
    def top(lp):     # a separate program: forward + top_k in one fails on ROCm (docs/ROCM_GOTCHAS.md, #20b)
        v, i = jax.lax.top_k(lp, top_k)
        return i.astype(jnp.uint16), v.astype(jnp.float16)

    def run(params, x, y):
        lp, nll = scores(params, x, y)
        return (*top(lp), nll)

    t0 = time.perf_counter()
    for s in range(0, n, batch):
        b = jnp.asarray(w[s:s + batch].astype(np.int32))
        i, v, l = run(params, b[:, :-1], b[:, 1:])
        ids[s:s + batch], logp[s:s + batch], nll[s:s + batch] = np.asarray(i), np.asarray(v), np.asarray(l)
        if s % (batch * 500) == 0:
            log(f"teacher: {s:,}/{n:,} windows, {(s + batch) * SEQ / (time.perf_counter() - t0):,.0f} tok/s")
    ids.flush()
    logp.flush()
    np.save(out / "nll.npy", nll)
    (out / "meta.json").write_text(json.dumps({"repo": repo, "windows": n, "top_k": top_k, "seq": SEQ,
                                               "seconds": time.perf_counter() - t0}))


# ---------------------------------------------------------------- students
def init_params(key, cfg: llama.Config):
    """A fresh model in llama.py's layout (layers stacked), float32 master weights."""
    D, L, F = cfg.d_model, cfg.n_layers, cfg.d_ff
    H, KV, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    ks = jax.random.split(key, 8)
    n = lambda k, shape, fan_in, scale=1.0: jax.random.normal(k, shape) * scale / np.sqrt(fan_in)
    out = 1 / np.sqrt(2 * L)
    return {"embed": jax.random.normal(ks[0], (cfg.vocab, D)) * 0.02, "final_norm": jnp.ones((D,)),
            "layers": {"attn_norm": jnp.ones((L, D)), "mlp_norm": jnp.ones((L, D)),
                       "wq": n(ks[1], (L, D, H * hd), D), "wk": n(ks[2], (L, D, KV * hd), D),
                       "wv": n(ks[3], (L, D, KV * hd), D), "wo": n(ks[4], (L, H * hd, D), H * hd, out),
                       "w_gate": n(ks[5], (L, D, F), D), "w_up": n(ks[6], (L, D, F), D),
                       "w_down": n(ks[7], (L, F, D), F, out)}}


def cross_entropy(logits, y):
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits.astype(jnp.float32), -1), y[..., None], -1))


def _bf16(params):
    return jax.tree.map(lambda a: a.astype(jnp.bfloat16), params)


def make_optimizer(params, lr, steps, warmup=200, kind="adamw", parts=None):
    """AdamW (warmup-cosine, clip 1.0, wd 0.1), or Muon for the hidden matrices + AdamW for the rest."""
    warmup = min(warmup, steps // 2)
    sched = lambda peak: optax.warmup_cosine_decay_schedule(0.0, peak, warmup, max(steps, warmup + 1), peak * 0.1)
    adamw = optax.adamw(sched(lr), b1=0.9, b2=0.95, weight_decay=0.1)
    if kind == "adamw":
        return optax.chain(optax.clip_by_global_norm(1.0), adamw)
    labels = jax.tree_util.tree_map_with_path(
        lambda p, a: "muon" if a.ndim == 3 and "layers" in jax.tree_util.keystr(p) else "adam", params)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.multi_transform(
        {"muon": muon(parts.newton_schulz, sched(0.02)), "adam": adamw}, labels))


def muon(newton_schulz, lr, momentum=0.95, weight_decay=0.0):
    """Muon (Keller Jordan, 2024) for stacked (L, in, out) matrices: Nesterov momentum, then each layer's
    update orthogonalized by Newton–Schulz and scaled by √max(1, out/in) (the blog's shape correction)."""

    def init(params):
        return {"m": jax.tree.map(jnp.zeros_like, params), "count": jnp.zeros((), jnp.int32)}

    def update(grads, state, params):
        m = jax.tree.map(lambda m, g: momentum * m + g, state["m"], grads)
        step_lr = lr(state["count"])

        def one(g, mm, p):
            o = jax.vmap(newton_schulz)(g + momentum * mm)                  # Nesterov look-ahead, per layer
            scale = jnp.sqrt(max(1.0, p.shape[2] / p.shape[1]))
            return -step_lr * (scale * o + weight_decay * p)

        return jax.tree.map(one, grads, m, params), {"m": m, "count": state["count"] + 1}

    return optax.GradientTransformation(init, update)


def train_forward(params, x, cfg):
    """llama.forward for training: the same layers, each wrapped in jax.checkpoint, so the backward pass
    recomputes a layer's activations instead of storing them all (a 30-layer model's attention scores alone
    would need ~9 GB at batch 8 × 512)."""
    B, T = x.shape
    pos = jnp.broadcast_to(jnp.arange(T), (B, T))
    layer = jax.checkpoint(lambda h, lp: (llama._layer(cfg, h, lp, pos, None, 0)[0], None))
    h, _ = jax.lax.scan(layer, params["embed"][x], params["layers"])
    h = llama.rms_norm(h, params["final_norm"], cfg.norm_eps)
    return (h @ (params["embed"].T if cfg.tied else params["head"])).astype(jnp.float32)


def make_trainer(parts, cfg, optimizer, kd_weight=0.0, T=1.0, batch=16):
    """train_chunk(params, opt_state, data, key, n_steps) and eval_loss(params, x, y).

    data = (windows (n, SEQ+1) int32, top_ids or None, top_logp or None, allowed window indices). With
    kd_weight w the loss is (1 − w)·cross-entropy + w·your kd_loss against the teacher's stored top-k."""

    def loss_fn(params, x, y, ti, tl):
        logits = train_forward(_bf16(params), x, cfg)
        ce = cross_entropy(logits, y)
        if kd_weight == 0.0:
            return ce, ce
        return (1 - kd_weight) * ce + kd_weight * parts.kd_loss(logits, ti.astype(jnp.int32),
                                                                tl.astype(jnp.float32), T), ce

    def step(carry, key):
        params, opt_state, data = carry
        win, top_ids, top_logp, allowed = data
        idx = allowed[jax.random.randint(key, (batch,), 0, allowed.shape[0])]
        w = win[idx]
        ti = None if top_ids is None else top_ids[idx]
        tl = None if top_logp is None else top_logp[idx]
        (_, ce), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, w[:, :-1], w[:, 1:], ti, tl)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, data), ce

    @partial(jax.jit, static_argnames="n_steps", donate_argnums=(0, 1))
    def train_chunk(params, opt_state, data, key, n_steps):
        (params, opt_state, _), ces = jax.lax.scan(step, (params, opt_state, data), jax.random.split(key, n_steps))
        return params, opt_state, ces

    return train_chunk, make_eval(cfg)


def make_eval(cfg):
    @jax.jit
    def eval_loss(params, w):
        return cross_entropy(llama.forward(_bf16(params), w[:, :-1], cfg)[0], w[:, 1:])
    return eval_loss


def val_loss(params, cfg, vw, batch=8):
    ev = make_eval(cfg)
    return float(np.mean([ev(params, jnp.asarray(vw[i:i + batch])) for i in range(0, len(vw), batch)]))


# ---------------------------------------------------------------- pruning (Minitron-style)
def calibration_stats(parts, params, cfg, w):
    """Run the model over calibration windows w (n, SEQ+1); return each layer's feed-forward neuron
    importance (L, F) from your `ffn_importance`, and each layer's block influence (L,) from your
    `layer_importance`."""

    @jax.jit
    def run(params, x):
        B, T = x.shape
        pos = jnp.broadcast_to(jnp.arange(T), (B, T))
        h = params["embed"][x]

        def body(h, lp):
            h_in = h
            h_attn, _ = llama._layer(cfg, h, {**lp, "w_gate": jnp.zeros_like(lp["w_gate"]),
                                              "w_up": jnp.zeros_like(lp["w_up"])}, pos, None, 0)
            m = llama.rms_norm(h_attn, lp["mlp_norm"], cfg.norm_eps)
            acts = jax.nn.silu(m @ lp["w_gate"]) * (m @ lp["w_up"])
            h_out = h_attn + acts @ lp["w_down"]
            return h_out, (parts.ffn_importance(acts), parts.layer_importance(h_in, h_out))

        _, (imp, bi) = jax.lax.scan(body, h, params["layers"])
        return imp, bi

    outs = [run(params, jnp.asarray(w[i:i + 4, :-1])) for i in range(0, len(w), 4)]
    return np.mean([np.asarray(o[0]) for o in outs], 0), np.mean([np.asarray(o[1]) for o in outs], 0)


def prune(parts, params, cfg, ffn_imp, block_inf, keep_layers: int, keep_ffn: int):
    """Drop the layers with the lowest block influence, then keep each remaining layer's most important
    feed-forward neurons (your `prune_ffn`). Returns (params, cfg)."""
    layers_kept = np.sort(np.argsort(block_inf)[::-1][:keep_layers])
    lay = jax.tree.map(lambda a: a[layers_kept], params["layers"])
    keep = np.sort(np.argsort(ffn_imp[layers_kept], -1)[:, ::-1][:, :keep_ffn], -1)
    lay = parts.prune_ffn(lay, jnp.asarray(keep))
    new_cfg = llama.Config(**{**cfg.__dict__, "n_layers": keep_layers, "d_ff": keep_ffn})
    return {**params, "layers": lay}, new_cfg, layers_kept


# ---------------------------------------------------------------- I/O
def save(params, out: Path, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    flat = {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    np.savez(out / "params.tmp.npz", **flat)
    (out / "params.tmp.npz").replace(out / "params.npz")
    if extra is not None:
        (out / "config.json").write_text(json.dumps(extra, indent=2))


def load(out: Path):
    meta = json.loads((out / "config.json").read_text())
    cfg = llama.Config(**meta["cfg"])
    arrays = np.load(out / "params.npz")
    flat, tree = jax.tree_util.tree_flatten_with_path(jax.eval_shape(lambda: init_params(jax.random.key(0), cfg)))
    params = jax.tree_util.tree_unflatten(tree, [jnp.asarray(arrays[jax.tree_util.keystr(p)]) for p, _ in flat])
    return params, cfg, meta


def read_metrics(path):
    try:
        return [json.loads(line) for line in open(path)]
    except OSError:
        return []
