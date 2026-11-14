"""Module 17 reference solutions. Try the exercises first — `make test N=17` grades yours."""
import jax
import jax.numpy as jnp


def rms_norm(x, w, eps: float = 1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * w


def _angles(x, positions, theta):
    """(1, T, 1, ..., D/2) rotation angles broadcastable against x of shape (B, T, ..., D)."""
    d = x.shape[-1]
    freqs = theta ** -(jnp.arange(0, d, 2, dtype=jnp.float32) / d)                 # (D/2,)
    ang = positions.astype(jnp.float32)[:, None] * freqs[None, :]                  # (T, D/2)
    return ang.reshape((1, ang.shape[0]) + (1,) * (x.ndim - 3) + (d // 2,))


def rope_complex(x, positions, theta: float = 10000.0):
    ang = _angles(x, positions, theta)
    xc = jax.lax.complex(x[..., 0::2].astype(jnp.float32), x[..., 1::2].astype(jnp.float32))
    rot = xc * jnp.exp(1j * ang)                                                   # rotate each pair
    return jnp.stack([rot.real, rot.imag], axis=-1).reshape(x.shape).astype(x.dtype)


def rope_rotate_half(x, positions, theta: float = 10000.0):
    ang = _angles(x, positions, theta)
    cos, sin = jnp.cos(ang).astype(x.dtype), jnp.sin(ang).astype(x.dtype)
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def swiglu(params, x):
    return (jax.nn.silu(x @ params["w1"]) * (x @ params["w3"])) @ params["w2"]


def repeat_kv(x, n_rep: int):
    return x if n_rep == 1 else jnp.repeat(x, n_rep, axis=2)


def gqa_attention(params, x, positions, cfg, cache=None):
    B, T, _ = x.shape
    H, Hkv, D = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    q = (x @ params["wq"]).reshape(B, T, H, D)
    k = (x @ params["wk"]).reshape(B, T, Hkv, D)
    v = (x @ params["wv"]).reshape(B, T, Hkv, D)
    q = rope_rotate_half(q, positions, cfg["rope_theta"])
    k = rope_rotate_half(k, positions, cfg["rope_theta"])
    if cache is not None:
        k = jnp.concatenate([cache["k"], k], axis=1)                               # sequence axis
        v = jnp.concatenate([cache["v"], v], axis=1)
    new_cache = {"k": k, "v": v}
    S = k.shape[1]
    k, v = repeat_kv(k, H // Hkv), repeat_kv(v, H // Hkv)
    scores = jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32) / jnp.sqrt(D)
    allowed = jnp.arange(S)[None, :] <= positions[:, None]                         # (T, S)
    attn = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1).astype(x.dtype)
    out = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(B, T, H * D)
    return out @ params["wo"], new_cache


def buggy_cached_attention(q, k_new, v_new, cache_k, cache_v, max_len: int = 64):
    """PROVIDED (it works, but it's wrong). Attention for new queries against a KV cache.

    This follows code from a published "Llama 3 from scratch" tutorial and contains two real bugs.
    q, k_new, v_new: (B, T, H, D) for the T new tokens (already rotated). cache_k, cache_v: (B, P, H, D)
    for the P earlier tokens. Should return (out (B, T, H, D), (all_k, all_v)).
    """
    k = jnp.concatenate([cache_k, k_new], axis=-1)
    v = jnp.concatenate([cache_v, v_new], axis=-1)
    T, S = q.shape[1], k.shape[1]
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(q.shape[-1])
    mask = jnp.tril(jnp.ones((max_len, max_len), dtype=bool))[:T, :S]
    scores = jnp.where(mask, scores, -jnp.inf)
    out = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(scores, axis=-1), v)
    return out, (k, v)


def fix_buggy_cached_attention(q, k_new, v_new, cache_k, cache_v):
    k = jnp.concatenate([cache_k, k_new], axis=1)        # bug 1 fixed: append along the sequence axis
    v = jnp.concatenate([cache_v, v_new], axis=1)
    P, T, S = cache_k.shape[1], q.shape[1], k.shape[1]
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(q.shape[-1])
    q_pos = P + jnp.arange(T)                            # bug 2 fixed: queries sit *after* the cache
    allowed = jnp.arange(S)[None, :] <= q_pos[:, None]
    scores = jnp.where(allowed, scores, -jnp.inf)
    out = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(scores, axis=-1), v)
    return out, (k, v)


def moe_route(scores, bias, top_k: int):
    _, idx = jax.lax.top_k(scores + jax.lax.stop_gradient(bias), top_k)
    picked = jax.nn.one_hot(idx, scores.shape[-1]).sum(-2) > 0
    gated = jnp.where(picked, scores, 0.0)
    return gated / gated.sum(-1, keepdims=True)


def update_bias(bias, load, gamma: float):
    return bias + gamma * jnp.sign(load.mean(-1, keepdims=True) - load)


def mha_cache_floats(n_kv_heads: int, k_dim: int, v_dim: int) -> int:
    return n_kv_heads * (k_dim + v_dim)


def mla_cache_floats(cfg: dict) -> int:
    return cfg["kv_rank"] + cfg["d_rope"]
