"""Module 05 reference solutions. Try the exercises first — `make test N=05` grades yours."""
import math

import jax
import jax.numpy as jnp


def sgd_step(params, grads, lr: float):
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


def momentum_step(params, grads, velocity, lr: float, beta: float = 0.9):
    velocity = jax.tree.map(lambda g, v: g + beta * v, grads, velocity)
    params = jax.tree.map(lambda p, v: p - lr * v, params, velocity)
    return params, velocity


def adam_step(params, grads, state, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999,
              eps: float = 1e-8):
    m, v, t = state
    t = t + 1
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
    c1, c2 = 1 - b1**t, 1 - b2**t
    params = jax.tree.map(lambda p, m_, v_: p - lr * (m_ / c1) / (jnp.sqrt(v_ / c2) + eps), params, m, v)
    return params, (m, v, t)


def warmup_cosine(step: int, peak: float, warmup: int, total: int, final: float) -> float:
    if step < warmup:
        return peak * step / warmup
    if step > total:
        return float(final)
    frac = (step - warmup) / (total - warmup)
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * frac))


def clip_by_global_norm(grads, max_norm: float):
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    scale = jnp.where(norm > max_norm, max_norm / jnp.maximum(norm, 1e-30), 1.0)
    return jax.tree.map(lambda g: g * scale, grads)


def fit(loss_fn, params, data, steps: int, lr: float):
    @jax.jit
    def step(p):
        return sgd_step(p, jax.grad(loss_fn)(p, data), lr)

    for _ in range(steps):
        params = step(params)
    return params


NS_COEFFS = (3.4445, -4.775, 2.0315)


def newton_schulz(G, steps: int = 5, coeffs=NS_COEFFS, eps: float = 1e-7):
    a, b, c = coeffs
    X = G / (jnp.linalg.norm(G) + eps)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if transposed else X


def muon_step(param, grad, momentum, lr: float, beta: float = 0.95, nesterov: bool = True,
              eps: float = 1e-8):
    mu, t = momentum
    mu = beta * mu + (1 - beta) * grad
    t = t + 1
    if nesterov:
        m_hat = beta * mu / (1 - beta ** (t + 1)) + (1 - beta) * grad / (1 - beta**t)
    else:
        m_hat = mu / (1 - beta**t)
    update = newton_schulz(m_hat, eps=eps)
    fan_in, fan_out = param.shape
    scale = math.sqrt(max(1.0, fan_out / fan_in))
    return param - lr * scale * update, (mu, t)
