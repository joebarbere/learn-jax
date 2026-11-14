"""Module 11 reference solutions. Try the exercises first — `make test N=11` grades yours."""
import math
import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp


def init_scale(fan_in: int, fan_out: int, kind: str) -> float:
    if kind == "he":
        return math.sqrt(2 / fan_in)
    if kind == "xavier":
        return math.sqrt(2 / (fan_in + fan_out))
    raise ValueError(f"unknown init kind {kind!r}")


def layer_norm(x, gamma, beta, eps: float = 1e-5):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * gamma + beta


def rms_norm(x, w, eps: float = 1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * w


def dropout(key, x, rate: float, train: bool):
    if not train or rate == 0:
        return x
    if rate >= 1:
        return jnp.zeros_like(x)
    keep = jax.random.bernoulli(key, 1 - rate, x.shape)
    return jnp.where(keep, x / (1 - rate), 0)


def residual_block(params: dict, x):
    h = jax.nn.relu(rms_norm(x, params["norm"]) @ params["w1"])
    return x + h @ params["w2"]


def global_grad_norm(grads) -> jnp.ndarray:
    leaves = jax.tree.leaves(grads)
    return jnp.sqrt(sum((jnp.sum(jnp.square(g)) for g in leaves), jnp.float32(0)))


def save_and_restore(state, path):
    path = os.path.abspath(path)
    ckpt = ocp.StandardCheckpointer()
    ckpt.save(path, state)
    ckpt.wait_until_finished()
    return ckpt.restore(path)
