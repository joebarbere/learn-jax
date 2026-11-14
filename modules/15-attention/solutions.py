"""Module 15 reference solutions. Try the exercises first — `make test N=15` grades yours."""
import jax
import jax.numpy as jnp


def causal_mask(T: int):
    return jnp.tril(jnp.ones((T, T), dtype=bool))


def scaled_dot_product(q, k, v, mask=None):
    d = q.shape[-1]
    scores = jnp.einsum("...td,...sd->...ts", q, k) / jnp.sqrt(d).astype(q.dtype)
    if mask is not None:
        scores = jnp.where(mask, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("...ts,...sd->...td", weights, v), weights


def split_heads(x, n_heads: int):
    B, T, D = x.shape
    if D % n_heads:
        raise ValueError(f"width {D} is not divisible by {n_heads} heads")
    return x.reshape(B, T, n_heads, D // n_heads)


def merge_heads(x):
    B, T, H, d = x.shape
    return x.reshape(B, T, H * d)


def attention_weights(params, x, n_heads: int, causal: bool = True):
    q = split_heads(x @ params["wq"], n_heads)
    k = split_heads(x @ params["wk"], n_heads)
    d = q.shape[-1]
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(d).astype(q.dtype)
    if causal:
        scores = jnp.where(causal_mask(x.shape[1]), scores, -jnp.inf)
    return jax.nn.softmax(scores, axis=-1)


def multi_head_attention(params, x, n_heads: int, causal: bool = True):
    w = attention_weights(params, x, n_heads, causal)
    v = split_heads(x @ params["wv"], n_heads)
    out = jnp.einsum("bhts,bshd->bthd", w, v)
    return merge_heads(out) @ params["wo"]


def sinusoidal_positions(T: int, D: int):
    pos = jnp.arange(T, dtype=jnp.float32)[:, None]
    i = jnp.arange(D // 2, dtype=jnp.float32)[None, :]
    angles = pos / 10000.0 ** (2 * i / D)
    pe = jnp.zeros((T, D), jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(angles))
    return pe.at[:, 1::2].set(jnp.cos(angles))


def is_permutation_equivariant(f, x, perm, atol: float = 1e-4) -> bool:
    perm = jnp.asarray(perm)
    return bool(jnp.allclose(f(x[:, perm]), f(x)[:, perm], atol=atol))
