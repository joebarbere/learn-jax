"""Module 46 reference solutions. Same signatures as exercises.py."""
import jax.numpy as jnp
import numpy as np


def ridge_filter(theta: float, size: int = 5, width: float = 1.0):
    c = (size - 1) / 2
    y, x = np.mgrid[:size, :size] - c
    across = -np.sin(theta) * x + np.cos(theta) * y
    along = np.cos(theta) * x + np.sin(theta) * y
    f = np.exp(-across**2 / (2 * width**2)) - 0.5 * np.exp(-across**2 / (2 * (2.5 * width) ** 2))
    f = f * np.exp(-along**2 / 8)
    f = f - f.mean()
    return jnp.asarray(f / np.abs(f).sum(), jnp.float32)


def firing_rates(fires, labels, n_classes: int, alpha: float = 1.0):
    onehot = jnp.asarray(labels)[:, None] == jnp.arange(n_classes)
    counts = onehot.T.astype(jnp.float32) @ jnp.asarray(fires, jnp.float32)
    n = onehot.sum(0).astype(jnp.float32)[:, None]
    return (counts + alpha) / (n + 2 * alpha)


def log_odds_readout(rates, prior):
    rates = jnp.asarray(rates)
    W = (jnp.log(rates) - jnp.log1p(-rates)).T
    b = jnp.log1p(-rates).sum(1) + jnp.log(jnp.asarray(prior))
    return W, b


def position_selector(target, scale: float = 50.0):
    L = len(target)
    return scale * jnp.eye(L)[jnp.asarray(target)], jnp.eye(L)


def ties_merge(taus, density: float = 0.2, lam: float = 1.0):
    taus = jnp.asarray(taus)
    k = max(1, round(density * taus.shape[1]))
    kth = -jnp.sort(-jnp.abs(taus), axis=1)[:, k - 1:k]
    trimmed = jnp.where(jnp.abs(taus) >= kth, taus, 0.0)
    sign = jnp.sign(trimmed.sum(0))
    agree = (jnp.sign(trimmed) == sign) & (trimmed != 0)
    n = agree.sum(0)
    return lam * jnp.where(n > 0, jnp.where(agree, trimmed, 0.0).sum(0) / jnp.maximum(n, 1), 0.0)


def rank_one_edit(W, k, v, C_inv):
    u = C_inv @ k
    return W + jnp.outer(u, v - k @ W) / (u @ k)
