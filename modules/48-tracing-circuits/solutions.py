"""Module 48 reference solutions. Don't peek until you've tried (see exercises.py)."""
import jax
import jax.numpy as jnp


def logit_diff(logits, io, s):
    last = logits[:, -1].astype(jnp.float32)
    pick = lambda ids: jnp.take_along_axis(last, ids[:, None], 1)[:, 0]
    return (pick(io) - pick(s)).mean()


def patching_effect(patched, clean, corrupt):
    return (patched - corrupt) / (clean - corrupt)


def attribution_patch(grad, clean, corrupt):
    return (grad * (clean - corrupt)).sum(-1)


def edge_weights(jac, values):
    return jac * values[None, :]


def normalize_rows(A):
    a = jnp.abs(A)
    s = a.sum(1, keepdims=True)
    return jnp.where(s > 0, a / jnp.where(s > 0, s, 1.0), 0.0)


def total_influence(A_norm, w):
    n = A_norm.shape[0]

    def body(carry, _):
        x, v = carry
        return (x + v, v @ A_norm), None

    (x, _), _ = jax.lax.scan(body, (jnp.zeros_like(w), w @ A_norm), None, length=n)
    return x
