"""Module 47 reference solutions. Same signatures as exercises.py."""
import jax
import jax.numpy as jnp


def logit_lens(h, norm_w, W_U, eps: float = 1e-5):
    h = jnp.asarray(h, jnp.float32)
    x = h * jax.lax.rsqrt((h**2).mean(-1, keepdims=True) + eps) * jnp.asarray(norm_w, jnp.float32)
    return x @ jnp.asarray(W_U, jnp.float32)


def topk_activation(pre, k: int):
    kth = jax.lax.top_k(pre, k)[0][..., -1:]
    return jnp.where(pre >= kth, jax.nn.relu(pre), 0.0)


def fvu(x, x_hat):
    x, x_hat = jnp.asarray(x, jnp.float32), jnp.asarray(x_hat, jnp.float32)
    return ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()


def mean_difference(pos, neg):
    d = jnp.asarray(pos, jnp.float32).mean(0) - jnp.asarray(neg, jnp.float32).mean(0)
    return d / jnp.linalg.norm(d)


def ablate_direction(W, u, alpha: float = 1.0):
    return W - alpha * jnp.outer(W @ u, u)


def mutual_knn(X, Y, k: int):
    def knn(A):
        A = jnp.asarray(A, jnp.float32)
        A = A / jnp.linalg.norm(A, axis=1, keepdims=True)
        S = A @ A.T
        S = jnp.where(jnp.eye(S.shape[0], dtype=bool), -jnp.inf, S)
        return jax.lax.top_k(S, k)[1]
    a, b = knn(X), knn(Y)
    hits = (a[:, :, None] == b[:, None, :]).any(-1).sum(-1)
    return float(hits.mean() / k)
