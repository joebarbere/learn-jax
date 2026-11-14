"""Module 04 reference solutions. Try the exercises first — `make test N=04` grades yours."""
import jax
import jax.numpy as jnp


def stable_softmax(x, axis: int = -1):
    x = jnp.asarray(x)
    e = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return e / jnp.sum(e, axis=axis, keepdims=True)


def log_softmax(x, axis: int = -1):
    x = jnp.asarray(x)
    m = jnp.max(x, axis=axis, keepdims=True)
    return x - (m + jnp.log(jnp.sum(jnp.exp(x - m), axis=axis, keepdims=True)))


def cross_entropy(logits, labels):
    logp = log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(logp, jnp.asarray(labels)[:, None], axis=-1)[:, 0]
    return -jnp.mean(picked)


def entropy(p):
    p = jnp.asarray(p)
    return -jnp.sum(jnp.where(p > 0, p * jnp.log(jnp.where(p > 0, p, 1.0)), 0.0), axis=-1)


def kl_divergence(p, q):
    p, q = jnp.asarray(p), jnp.asarray(q)
    safe_p = jnp.where(p > 0, p, 1.0)
    safe_q = jnp.where(p > 0, q, 1.0)
    return jnp.sum(jnp.where(p > 0, p * jnp.log(safe_p / safe_q), 0.0), axis=-1)


def sample_categorical(key, logits, n: int):
    logits = jnp.asarray(logits)
    u = jax.random.uniform(key, (n, logits.shape[-1]), minval=1e-20, maxval=1.0)
    gumbel = -jnp.log(-jnp.log(u))
    return jnp.argmax(logits + gumbel, axis=-1)


def perplexity(mean_cross_entropy):
    return jnp.exp(mean_cross_entropy)
