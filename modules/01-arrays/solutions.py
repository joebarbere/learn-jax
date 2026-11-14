"""Module 01 reference solutions. Try the exercises first — `make test N=01` grades yours."""
import jax.numpy as jnp


def broadcast_shape(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> tuple[int, ...]:
    ndim = max(len(a_shape), len(b_shape))
    a = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
    b = (1,) * (ndim - len(b_shape)) + tuple(b_shape)
    out = []
    for x, y in zip(a, b):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"shapes {a_shape} and {b_shape} are not broadcastable")
    return tuple(out)


def normalize_rows(x: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(x, axis=1, keepdims=True)
    return jnp.where(norms > 0, x / jnp.where(norms > 0, norms, 1.0), x)


def pairwise_sq_dists(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return jnp.sum(diff**2, axis=-1)


def one_hot(labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    return (labels[:, None] == jnp.arange(num_classes)[None, :]).astype(jnp.float32)


def batched_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("bnk,bkm->bnm", a, b)


def moving_average(x: jnp.ndarray, window: int) -> jnp.ndarray:
    if window < 1 or window > x.shape[0]:
        raise ValueError(f"window must be in [1, {x.shape[0]}], got {window}")
    c = jnp.concatenate([jnp.zeros(1, x.dtype), jnp.cumsum(x)])
    return (c[window:] - c[:-window]) / window
