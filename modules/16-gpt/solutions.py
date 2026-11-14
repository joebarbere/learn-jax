"""Module 16 reference solutions. Try the exercises first — `make test N=16` grades yours.

Everything is plain functions over a parameter pytree, in the style of deepseek-v3-jax:

    params = {
        "wte": (vocab, d_model),          # token embedding (also the output head when tied)
        "wpe": (seq_len, d_model),        # learned position embedding
        "blocks": [ {                     # one dict per layer
            "ln1_g": (d,), "ln1_b": (d,),
            "w_qkv": (d, 3d), "b_qkv": (3d,), "w_o": (d, d), "b_o": (d,),
            "ln2_g": (d,), "ln2_b": (d,),
            "w_fc": (d, 4d), "b_fc": (4d,), "w_proj": (4d, d), "b_proj": (d,),
        }, ... ],
        "lnf_g": (d,), "lnf_b": (d,),     # final LayerNorm
        "w_head": (d, vocab),             # ONLY when cfg.tie_embeddings is False
    }

`layer_norm`, `causal_self_attention` and `mlp` are provided (they're Modules 11 and 15). Your job is
to assemble them into a model, count it, feed it data, and sample from it. When these work, the lesson
trains your GPT on Shakespeare live, and `train.py` trains a bigger one on 207 detective novels.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GPTConfig:
    """Model shape. Frozen (hashable) so it can be a static argument to jax.jit."""

    vocab: int = 65
    seq_len: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    tie_embeddings: bool = True


# ---------------------------------------------------------------------------------------------------
# Provided building blocks (from Modules 11 and 15). Read them; don't change them.
# ---------------------------------------------------------------------------------------------------
def layer_norm(x, g, b, eps: float = 1e-5):
    """LayerNorm over the last axis, computed in float32 and returned in x's dtype (Module 11)."""
    x32 = x.astype(jnp.float32)
    mean = x32.mean(-1, keepdims=True)
    var = ((x32 - mean) ** 2).mean(-1, keepdims=True)
    y = (x32 - mean) * jax.lax.rsqrt(var + eps) * g.astype(jnp.float32) + b.astype(jnp.float32)
    return y.astype(x.dtype)


def causal_self_attention(x, w_qkv, b_qkv, w_o, b_o, n_heads: int):
    """Multi-head causal self-attention (Module 15). x: (B, T, D) -> (B, T, D).

    One matmul makes queries, keys and values for every head; scores are masked so position t only
    sees positions <= t; softmax runs in float32 for stability (Module 08).
    """
    B, T, D = x.shape
    hd = D // n_heads
    q, k, v = jnp.split(x @ w_qkv + b_qkv, 3, axis=-1)
    q, k, v = (a.reshape(B, T, n_heads, hd) for a in (q, k, v))
    scores = jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32) / jnp.sqrt(hd)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    weights = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), axis=-1).astype(x.dtype)
    out = jnp.einsum("bhts,bshd->bthd", weights, v).reshape(B, T, D)
    return out @ w_o + b_o


def mlp(x, w_fc, b_fc, w_proj, b_proj):
    """The block's feed-forward network: expand 4x, GELU, project back. (B, T, D) -> (B, T, D)."""
    return jax.nn.gelu(x @ w_fc + b_fc) @ w_proj + b_proj


# ---------------------------------------------------------------------------------------------------
# Solutions
# ---------------------------------------------------------------------------------------------------
def count_params(cfg: GPTConfig) -> int:
    d, v, t, n = cfg.d_model, cfg.vocab, cfg.seq_len, cfg.n_layers
    per_block = 12 * d * d + 13 * d
    total = v * d + t * d + n * per_block + 2 * d
    if not cfg.tie_embeddings:
        total += d * v
    return total


def init_gpt(key, cfg: GPTConfig) -> dict:
    d, n = cfg.d_model, cfg.n_layers
    std, proj_std = 0.02, 0.02 / (2 * n) ** 0.5
    keys = iter(jax.random.split(key, 3 + 4 * n))

    def normal(shape, s=std):
        return jax.random.normal(next(keys), shape, jnp.float32) * s

    zeros, ones = (lambda m: jnp.zeros((m,), jnp.float32)), (lambda m: jnp.ones((m,), jnp.float32))
    blocks = [{
        "ln1_g": ones(d), "ln1_b": zeros(d),
        "w_qkv": normal((d, 3 * d)), "b_qkv": zeros(3 * d),
        "w_o": normal((d, d), proj_std), "b_o": zeros(d),
        "ln2_g": ones(d), "ln2_b": zeros(d),
        "w_fc": normal((d, 4 * d)), "b_fc": zeros(4 * d),
        "w_proj": normal((4 * d, d), proj_std), "b_proj": zeros(d),
    } for _ in range(n)]
    params = {
        "wte": normal((cfg.vocab, d)),
        "wpe": normal((cfg.seq_len, d)),
        "blocks": blocks,
        "lnf_g": ones(d), "lnf_b": zeros(d),
    }
    if not cfg.tie_embeddings:
        params["w_head"] = normal((d, cfg.vocab))
    return params


def block_apply(p: dict, x, n_heads: int):
    x = x + causal_self_attention(layer_norm(x, p["ln1_g"], p["ln1_b"]),
                                  p["w_qkv"], p["b_qkv"], p["w_o"], p["b_o"], n_heads)
    return x + mlp(layer_norm(x, p["ln2_g"], p["ln2_b"]), p["w_fc"], p["b_fc"], p["w_proj"], p["b_proj"])


def gpt_apply(params: dict, tokens, cfg: GPTConfig):
    T = tokens.shape[1]
    x = params["wte"][tokens] + params["wpe"][:T]
    for p in params["blocks"]:
        x = block_apply(p, x, cfg.n_heads)
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    head = params["wte"].T if cfg.tie_embeddings else params["w_head"]
    return (x @ head).astype(jnp.float32)


def loss_fn(params: dict, tokens, cfg: GPTConfig):
    logits = gpt_apply(params, tokens[:, :-1], cfg)
    logp = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)
    return -picked.mean()


def get_batch(key, data, batch_size: int, seq_len: int):
    data = jnp.asarray(data)
    starts = jax.random.randint(key, (batch_size,), 0, data.shape[0] - seq_len)
    windows = jax.vmap(lambda s: jax.lax.dynamic_slice(data, (s,), (seq_len + 1,)))(starts)
    return windows.astype(jnp.int32)


_window_logits = jax.jit(gpt_apply, static_argnums=2)


def sample(key, params: dict, prompt, n: int, cfg: GPTConfig, temperature: float = 1.0, top_k=None):
    tokens = [int(t) for t in prompt]
    for _ in range(n):
        window = tokens[-cfg.seq_len:]
        buf = jnp.array([window + [0] * (cfg.seq_len - len(window))], jnp.int32)
        logits = _window_logits(params, buf, cfg)[0, len(window) - 1]
        if temperature == 0:
            nxt = jnp.argmax(logits)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = jax.lax.top_k(logits, top_k)[0][-1]
                logits = jnp.where(logits < kth, -jnp.inf, logits)
            key, sub = jax.random.split(key)
            nxt = jax.random.categorical(sub, logits)
        tokens.append(int(nxt))
    return jnp.array(tokens, jnp.int32)
