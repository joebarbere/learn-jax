"""Module 32 reference solutions. Try the exercises first — `make test N=32` grades yours."""
import jax
import jax.numpy as jnp


def causal_mask(T: int):
    return jnp.tril(jnp.ones((T, T), dtype=bool))


def bidirectional_mask(T: int):
    return jnp.ones((T, T), dtype=bool)


def mask_tokens(key, ids, mask_id: int, vocab_size: int, rate: float = 0.15):
    k_pick, k_kind, k_rand = jax.random.split(key, 3)
    ids = jnp.asarray(ids)
    is_target = jax.random.uniform(k_pick, ids.shape) < rate
    kind = jax.random.uniform(k_kind, ids.shape)
    random_ids = jax.random.randint(k_rand, ids.shape, 0, vocab_size, dtype=ids.dtype)
    replaced = jnp.where(kind < 0.8, jnp.asarray(mask_id, ids.dtype),
                         jnp.where(kind < 0.9, random_ids, ids))
    return jnp.where(is_target, replaced, ids), is_target


def mlm_loss(logits, targets, is_target):
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    ce = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    w = is_target.astype(jnp.float32)
    return (ce * w).sum() / jnp.maximum(w.sum(), 1.0)


def cls_pool(hidden):
    return hidden[:, 0]


def last_token_pool(hidden, lengths=None):
    if lengths is None:
        return hidden[:, -1]
    return hidden[jnp.arange(hidden.shape[0]), jnp.asarray(lengths) - 1]


def contextual_similarity(vectors):
    v = vectors.astype(jnp.float32)
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v @ v.T
