"""Module 02 reference solutions. Try the exercises first — `make test N=02` grades yours."""
import jax.numpy as jnp


def cosine_similarity(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    dot = jnp.sum(a * b, axis=-1)
    return dot / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1))


def rotation_matrix(theta: float) -> jnp.ndarray:
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]])


def transform_points(m: jnp.ndarray, points: jnp.ndarray) -> jnp.ndarray:
    return points @ m.T


def project(v: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    return (jnp.dot(v, u) / jnp.dot(u, u)) * u


def low_rank_approx(m: jnp.ndarray, k: int) -> jnp.ndarray:
    u, s, vt = jnp.linalg.svd(m, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vt[:k]


def gram_schmidt(a: jnp.ndarray) -> jnp.ndarray:
    cols = []
    for j in range(a.shape[1]):
        v = a[:, j]
        for q in cols:
            v = v - jnp.dot(v, q) * q
        cols.append(v / jnp.linalg.norm(v))
    return jnp.stack(cols, axis=1)
