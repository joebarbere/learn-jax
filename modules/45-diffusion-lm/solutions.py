"""Module 45 reference solutions. Same signatures as exercises.py."""
import jax
import jax.numpy as jnp


def mask_tokens(key, x, t, mask_id: int):
    keep = jax.random.uniform(key, x.shape) >= jnp.asarray(t)[..., None]
    return jnp.where(keep, x, mask_id)


def mdlm_loss(logits, x, z, t, mask_id: int):
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    nll = -jnp.take_along_axis(logp, x[..., None], -1)[..., 0]
    masked = (z == mask_id).astype(jnp.float32)
    per_seq = jnp.sum(nll * masked, -1) / (jnp.asarray(t) * x.shape[-1])
    return jnp.mean(per_seq)


def unmask_step(key, logits, z, n_reveal: int, mask_id: int, temperature: float = 1.0, by_confidence: bool = True):
    k1, k2 = jax.random.split(key)
    logits = logits.astype(jnp.float32).at[..., mask_id].set(-jnp.inf)
    sample = jax.random.categorical(k1, logits / temperature, -1)
    p = jax.nn.softmax(logits, -1)
    conf = jnp.take_along_axis(p, sample[..., None], -1)[..., 0]
    score = conf if by_confidence else jax.random.uniform(k2, conf.shape)
    score = jnp.where(z == mask_id, score, -jnp.inf)
    _, idx = jax.lax.top_k(score, n_reveal)
    reveal = jnp.zeros(z.shape, bool).at[jnp.arange(z.shape[0])[:, None], idx].set(True)
    reveal = reveal & (z == mask_id)
    return jnp.where(reveal, sample, z)


def reveal_schedule(n_masked: int, steps: int):
    base, extra = divmod(n_masked, steps)
    return [base + (1 if i < extra else 0) for i in range(steps)]


def block_order(T: int, prompt_len: int, block: int):
    out = []
    for s in range(prompt_len, T, block):
        out.append((s, min(s + block, T)))
    return out
