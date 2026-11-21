"""Module 44 reference solutions. Same signatures as exercises.py."""
import jax
import jax.numpy as jnp


def window_mask(T: int, window, n_sink=0):
    i = jnp.arange(T)[:, None]
    j = jnp.arange(T)[None, :]
    return (j <= i) & ((i - j < window) | (j < n_sink))


def kv_cache_bytes(context: int, n_layers: int, n_global: int, window: int, n_kv_heads: int, d_head: int,
                   bytes_per: int = 2) -> int:
    per_token = 2 * n_kv_heads * d_head * bytes_per
    return per_token * (n_global * context + (n_layers - n_global) * min(context, window))


def banded_attention(q, k, v, window: int):
    B, H, T, d = q.shape
    n = T // window
    qb = q.reshape(B, H, n, window, d)
    kb = k.reshape(B, H, n, window, d)
    vb = v.reshape(B, H, n, window, -1)
    pad = lambda a: jnp.concatenate([jnp.zeros_like(a[:, :, :1]), a[:, :, :-1]], 2)   # the previous block
    k2 = jnp.concatenate([pad(kb), kb], 3)                                            # (B, H, n, 2w, d)
    v2 = jnp.concatenate([pad(vb), vb], 3)
    s = jnp.einsum("bhnqd,bhnkd->bhnqk", qb, k2).astype(jnp.float32) / jnp.sqrt(d)
    qi = jnp.arange(window)[:, None] + window                    # query position inside [prev | own]
    kj = jnp.arange(2 * window)[None, :]
    allowed = (kj <= qi) & (qi - kj < window)
    first = jnp.arange(n)[:, None, None] == 0                     # block 0 has no previous block
    allowed = allowed[None] & ~(first & (kj < window)[None])
    s = jnp.where(allowed, s, -jnp.inf)
    o = jnp.einsum("bhnqk,bhnkd->bhnqd", jax.nn.softmax(s, -1).astype(v.dtype), v2)
    return o.reshape(B, H, T, -1)


def moba_mask(q, k, block: int, top: int):
    B, H, T, d = q.shape
    nb = T // block
    means = k.reshape(B, H, nb, block, d).mean(3)                         # (B, H, nb, d)
    s = jnp.einsum("bhtd,bhnd->bhtn", q, means)
    qblock = jnp.arange(T) // block
    past = jnp.arange(nb)[None, :] < qblock[:, None]                      # (T, nb): strictly earlier blocks
    s = jnp.where(past, s, -jnp.inf)
    _, idx = jax.lax.top_k(s, min(top, nb))
    chosen = jax.nn.one_hot(idx, nb, dtype=jnp.int32).sum(-2) > 0          # (B, H, T, nb)
    chosen = (chosen & past) | (jnp.arange(nb)[None, :] == qblock[:, None])
    kblock = jnp.arange(T) // block
    return chosen[..., kblock] & (jnp.arange(T)[None, :] <= jnp.arange(T)[:, None])


def indexer_scores(q_idx, k_idx, w):
    dots = jnp.einsum("bthd,bsd->bths", q_idx, k_idx)
    return jnp.einsum("bth,bths->bts", w, jax.nn.relu(dots))


def topk_causal_mask(scores, k: int):
    T = scores.shape[-1]
    causal = jnp.tril(jnp.ones((T, T), bool))
    s = jnp.where(causal, scores, -jnp.inf)
    kth = jax.lax.top_k(s, min(k, T))[0][..., -1:]                   # each row's k-th largest score
    return (s >= kth) & causal                                       # (ties aside, exactly the top k)
