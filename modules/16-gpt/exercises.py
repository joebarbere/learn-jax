"""Module 16 exercises: a GPT, assembled by hand in pure JAX.

Replace each `raise NotImplementedError` with your code, then run `make test N=16`.

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
# Your turn
# ---------------------------------------------------------------------------------------------------
def count_params(cfg: GPTConfig) -> int:
    """Exact number of parameters `init_gpt(key, cfg)` creates — in closed form, without building it.

    Add up the shapes in the module docstring: the two embeddings, n_layers blocks, the final
    LayerNorm, and the (d_model × vocab) head only if the embeddings are NOT tied.
    Hint: each block works out to 12·d² + 13·d.
    """
    raise NotImplementedError


def init_gpt(key, cfg: GPTConfig) -> dict:
    """Create the parameter pytree described in the module docstring (all float32).

    * Every weight matrix and both embeddings: normal(0, 0.02).
    * The two matrices that write *into* the residual stream — each block's `w_o` and `w_proj` —
      use the smaller std 0.02 / sqrt(2 · n_layers) (GPT-2's scaled init: 2·n_layers additions to
      the residual stream, so their variances add up; Module 11).
    * Biases are zeros; LayerNorm gains (`ln1_g`, `ln2_g`, `lnf_g`) are ones, LayerNorm biases zeros.
    * Include "w_head" (normal 0.02) only when cfg.tie_embeddings is False.
    Split `key` so every random array gets its own key.
    """
    raise NotImplementedError


def block_apply(p: dict, x, n_heads: int):
    """One pre-LayerNorm transformer block. x: (B, T, D) -> (B, T, D).

        x = x + attention(LayerNorm₁(x))
        x = x + mlp(LayerNorm₂(x))

    Use the provided layer_norm, causal_self_attention and mlp with the weights in `p`.
    Note: if w_o, b_o, w_proj and b_proj are all zero, the block is exactly the identity —
    that's the residual stream at work.
    """
    raise NotImplementedError


def gpt_apply(params: dict, tokens, cfg: GPTConfig):
    """Forward pass. tokens: (B, T) ints with T <= cfg.seq_len -> logits (B, T, vocab) in float32.

    1. x = wte[tokens] + wpe[:T]                 (token embedding + position embedding)
    2. x = block_apply(...) for every block, in order
    3. x = layer_norm(x, lnf_g, lnf_b)
    4. logits = x @ wte.T  if cfg.tie_embeddings  else  x @ w_head
    Return logits as float32 even when params are bfloat16 (the lesson trains in mixed precision).
    """
    raise NotImplementedError


def loss_fn(params: dict, tokens, cfg: GPTConfig):
    """Mean next-token cross-entropy. tokens: (B, T + 1) ints.

    Inputs are tokens[:, :-1], targets are tokens[:, 1:] — every position predicts the next token.
    Use gpt_apply, then jax.nn.log_softmax, then pick the target's log-probability (Module 04).
    Return a float32 scalar.
    """
    raise NotImplementedError


def get_batch(key, data, batch_size: int, seq_len: int):
    """Sample `batch_size` random windows of seq_len + 1 consecutive tokens from the 1-D array `data`.

    Return int32 of shape (batch_size, seq_len + 1). Every window must lie fully inside `data`.
    Use jax.random.randint for the start positions and jax.vmap over jax.lax.dynamic_slice — this
    is jit-friendly (no Python indexing with traced values), so the lesson can sample batches inside
    a jitted training loop.
    """
    raise NotImplementedError


def sample(key, params: dict, prompt, n: int, cfg: GPTConfig, temperature: float = 1.0, top_k=None):
    """Generate n tokens after `prompt` (a list/1-D array of ints). Return int32 (len(prompt) + n,).

    Each step: take the last cfg.seq_len tokens as the context, run gpt_apply, keep the logits at the
    last real position, then pick the next token:
      * temperature == 0  -> greedy: argmax
      * otherwise         -> divide logits by temperature; if top_k is set, keep only the top_k
                             largest logits (set the rest to -inf); draw with jax.random.categorical,
                             splitting `key` each step.
    Speed tip: jit gpt_apply once (cfg static) and always feed a full (1, seq_len) window — pad the
    context on the right with zeros. Causal masking means padding *after* the last real token cannot
    change its logits, and a fixed shape compiles only once. (Module 18 replaces this recompute-
    everything loop with a KV cache.)
    """
    raise NotImplementedError
