"""Module 45 model code: a masked-diffusion language model (MDLM) on the detective corpus.

Provided (not exercises): the denoiser — a *bidirectional* Transformer the same size as Module 41's attention
model (8 layers of width 384, RoPE, SwiGLU) with one extra token, [MASK] — its training loop built on your
`mask_tokens` and `mdlm_loss`, the likelihood bound used to compare it with an autoregressive model, and the
samplers built on your `unmask_step`, `reveal_schedule` and `block_order`.

The autoregressive baseline is Module 41's `attention` run: same data, same size, same 98M training tokens.
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

VOCAB = 8192
MASK = VOCAB                       # the extra token: id 8192
CFG = {"d_model": 384, "n_layers": 8, "n_heads": 6, "d_ff": 1024, "rope_theta": 1e4}
HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent / "runs" / "45-diffusion-lm"


# ---------------------------------------------------------------- the denoiser
def rms_norm(x, w, eps=1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps)).astype(x.dtype) * w


def rope(x, theta=1e4):
    T, half = x.shape[-2], x.shape[-1] // 2
    ang = jnp.arange(T)[:, None] * (1.0 / theta ** (jnp.arange(half) / half))[None]
    cos, sin = jnp.cos(ang).astype(x.dtype), jnp.sin(ang).astype(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def init_model(key, cfg=CFG, vocab=VOCAB + 1):
    d, L, F = cfg["d_model"], cfg["n_layers"], cfg["d_ff"]
    ks = jax.random.split(key, 6)
    n = lambda k, shape, fan, s=1.0: jax.random.normal(k, shape) * s / np.sqrt(fan)
    out = 1 / np.sqrt(2 * L)
    return {"embed": jax.random.normal(ks[0], (vocab, d)) * 0.02, "norm": jnp.ones((d,)),
            "layers": {"n1": jnp.ones((L, d)), "n2": jnp.ones((L, d)), "wqkv": n(ks[1], (L, d, 3 * d), d),
                       "wo": n(ks[2], (L, d, d), d, out), "w1": n(ks[3], (L, d, F), d), "w3": n(ks[4], (L, d, F), d),
                       "w2": n(ks[5], (L, F, d), F, out)}}


def forward(params, tokens, cfg=CFG, causal=False):
    """Logits (B, T, vocab + 1) — bidirectional by default: every position sees every other."""
    B, T = tokens.shape
    H = cfg["n_heads"]
    dh = cfg["d_model"] // H
    mask = jnp.tril(jnp.ones((T, T), bool)) if causal else jnp.ones((T, T), bool)

    def layer(x, lp):
        h = rms_norm(x, lp["n1"])
        q, k, v = (a.reshape(B, T, H, dh).transpose(0, 2, 1, 3) for a in jnp.split(h @ lp["wqkv"], 3, -1))
        q, k = rope(q, cfg["rope_theta"]), rope(k, cfg["rope_theta"])
        s = jnp.einsum("bhtd,bhsd->bhts", q, k).astype(jnp.float32) / np.sqrt(dh)
        a = jax.nn.softmax(jnp.where(mask, s, -jnp.inf), -1).astype(x.dtype)
        x = x + jnp.einsum("bhts,bhsd->bhtd", a, v).transpose(0, 2, 1, 3).reshape(B, T, -1) @ lp["wo"]
        h = rms_norm(x, lp["n2"])
        return x + (jax.nn.silu(h @ lp["w1"]) * (h @ lp["w3"])) @ lp["w2"], None

    x, _ = jax.lax.scan(layer, params["embed"][tokens], params["layers"])
    return rms_norm(x, params["norm"]) @ params["embed"].T


def count_params(params) -> int:
    return sum(int(np.prod(a.shape)) for a in jax.tree.leaves(params))


def _bf16(params):
    return jax.tree.map(lambda a: a.astype(jnp.bfloat16), params)


# ---------------------------------------------------------------- training
def sample_t(key, batch, eps=1e-3):
    """Low-discrepancy masking rates in (eps, 1]: one per sequence, spread evenly (MDLM's trick to cut the
    variance of the loss)."""
    u = (jax.random.uniform(key) + jnp.arange(batch) / batch) % 1.0
    return eps + (1 - eps) * u


def make_trainer(parts, optimizer, batch=16, seq_len=512, cfg=CFG):
    def loss_fn(params, x, key):
        kt, km = jax.random.split(key)
        t = sample_t(kt, x.shape[0])
        z = parts.mask_tokens(km, x, t, MASK)
        return parts.mdlm_loss(forward(_bf16(params), z, cfg), x, z, t, MASK)

    def step(carry, key):
        params, opt_state, data = carry
        kw, kl = jax.random.split(key)
        starts = jax.random.randint(kw, (batch,), 0, data.shape[0] - seq_len)
        x = jax.vmap(lambda s: jax.lax.dynamic_slice(data, (s,), (seq_len,)))(starts)
        loss, grads = jax.value_and_grad(loss_fn)(params, x, kl)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, data), loss

    @partial(jax.jit, static_argnames="n_steps", donate_argnums=(0, 1))
    def train_chunk(params, opt_state, data, key, n_steps):
        (params, opt_state, _), losses = jax.lax.scan(step, (params, opt_state, data), jax.random.split(key, n_steps))
        return params, opt_state, losses

    return train_chunk


def make_optimizer(lr, steps, warmup=200):
    warmup = min(warmup, steps // 2)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(steps, warmup + 1), lr * 0.1)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=0.1))


def nll_bound(parts, params, windows, n_t=16, seed=0, cfg=CFG):
    """MDLM's bound on the negative log-likelihood, in nats per token: the training loss averaged over n_t
    evenly spread masking rates per window. An upper bound on the true NLL, so it can only flatter the
    autoregressive model it's compared with."""

    @jax.jit
    def one(params, x, key):
        kt, km = jax.random.split(key)
        t = 1e-3 + (1 - 1e-3) * ((jax.random.uniform(kt) + jnp.arange(n_t) / n_t) % 1.0)
        xs = jnp.repeat(x[None], n_t, 0)
        z = parts.mask_tokens(km, xs, t, MASK)
        return parts.mdlm_loss(forward(_bf16(params), z, cfg), xs, z, t, MASK)

    key = jax.random.key(seed)
    return float(np.mean([one(params, jnp.asarray(w), jax.random.fold_in(key, i)) for i, w in enumerate(windows)]))


# ---------------------------------------------------------------- sampling
def sample(parts, params, prompt, length, steps, key, block=None, temperature=1.0, by_confidence=True, cfg=CFG):
    """Generate `length` tokens after `prompt` (B, P) in `steps` denoising steps. With `block`, fill the
    answer block by block, left to right (LLaDA's semi-autoregressive sampling), splitting the steps evenly."""
    B, P = prompt.shape
    z = jnp.concatenate([prompt, jnp.full((B, length), MASK, prompt.dtype)], 1)
    T = P + length
    blocks = parts.block_order(T, P, block) if block else [(P, T)]
    per_block = max(1, steps // len(blocks))
    fwd = jax.jit(lambda p, z: forward(_bf16(p), z, cfg))
    reveal = {}
    for s, e in blocks:
        for n in parts.reveal_schedule(e - s, per_block):
            if n == 0:
                continue
            key, sub = jax.random.split(key)
            logits = fwd(params, z)
            outside = jnp.zeros((T,), bool).at[s:e].set(True)              # only reveal inside this block
            zl = jnp.where(outside[None], z, -1)                            # -1: not maskable this round
            if n not in reveal:
                reveal[n] = jax.jit(partial(parts.unmask_step, n_reveal=n, mask_id=MASK,
                                            temperature=temperature, by_confidence=by_confidence))
            new = reveal[n](sub, logits, zl)
            z = jnp.where(outside[None], new, z)
    return z


def ar_model():
    """Module 41's trained attention model (autoregressive, same data and size) and its forward function."""
    path = HERE.parent / "41-linear-attention-hybrids"
    spec = importlib.util.spec_from_file_location("m41_models", path / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from learnjax.course import load_exercises

    ex41 = load_exercises(path)
    run = HERE.parent.parent / "runs" / "41-linear-attention-hybrids" / "attention"
    if not (run / "params.npz").exists():
        return None
    params, cfg, _ = m.load(run)
    return jax.jit(lambda p, x: m.forward(ex41, p, x, cfg)), params


def ar_sample(fwd, params, prompt, length, key, temperature=1.0):
    """Plain autoregressive sampling with the AR model (re-running the prefix each step: simple, not fast)."""
    B, P = prompt.shape
    x = jnp.concatenate([prompt, jnp.zeros((B, length), prompt.dtype)], 1)
    for i in range(length):
        key, sub = jax.random.split(key)
        logits = fwd(params, x)[:, P + i - 1]
        x = x.at[:, P + i].set(jax.random.categorical(sub, logits.astype(jnp.float32) / temperature))
    return x


def gen_nll(fwd, params, seqs, prompt_len):
    """'Generative perplexity': the AR model's loss on generated tokens (lower = more like real text to it).
    Easy to game with dull, repetitive text, so read it with the entropy column."""
    lp = jax.nn.log_softmax(fwd(params, jnp.asarray(seqs))[:, :-1].astype(jnp.float32), -1)
    nll = -jnp.take_along_axis(lp, jnp.asarray(seqs)[:, 1:, None], -1)[..., 0]
    return float(nll[:, prompt_len - 1:].mean())


def repeat_4grams(seqs):
    """Share of 4-grams in each sample that already appeared earlier in the same sample, averaged: real prose
    repeats few; a sampler stuck in a loop repeats many."""
    out = []
    for row in np.asarray(seqs):
        grams = [tuple(row[i:i + 4]) for i in range(len(row) - 3)]
        out.append(1 - len(set(grams)) / max(len(grams), 1))
    return float(np.mean(out))


def unigram_entropy(seqs):
    _, counts = np.unique(np.asarray(seqs).ravel(), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------- I/O
def save(params, out: Path, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    flat = {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    np.savez(out / "params.tmp.npz", **flat)
    (out / "params.tmp.npz").replace(out / "params.npz")
    if extra is not None:
        (out / "config.json").write_text(json.dumps(extra, indent=2))


def load(out: Path):
    arrays = np.load(out / "params.npz")
    flat, tree = jax.tree_util.tree_flatten_with_path(jax.eval_shape(lambda: init_model(jax.random.key(0))))
    return jax.tree_util.tree_unflatten(tree, [jnp.asarray(arrays[jax.tree_util.keystr(p)]) for p, _ in flat])


def read_metrics(path):
    try:
        return [json.loads(line) for line in open(path)]
    except OSError:
        return []


def timed(fn, *a):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*a))
    return out, time.perf_counter() - t0
