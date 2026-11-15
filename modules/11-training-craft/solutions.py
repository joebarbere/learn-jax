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


def batch_norm(x, gamma, beta, running_mean, running_var, momentum: float = 0.1, train: bool = True,
               eps: float = 1e-5):
    if not train:
        y = (x - running_mean) / jnp.sqrt(running_var + eps) * gamma + beta
        return y, running_mean, running_var
    mean = x.mean(0)
    var = ((x - mean) ** 2).mean(0)
    y = (x - mean) / jnp.sqrt(var + eps) * gamma + beta
    return (y, (1 - momentum) * running_mean + momentum * mean,
            (1 - momentum) * running_var + momentum * var)


def forward_with_cache(params: dict, x, y):
    h_pre = x @ params["w1"] + params["b1"]
    h = jnp.tanh(h_pre)
    logits = h @ params["w2"] + params["b2"]
    probs = jax.nn.softmax(logits, axis=-1)
    loss = -jnp.mean(jnp.log(probs[jnp.arange(x.shape[0]), y]))
    return loss, {"x": x, "h_pre": h_pre, "h": h, "logits": logits, "probs": probs}


def manual_backward(params: dict, cache: dict, y) -> dict:
    n = cache["x"].shape[0]
    dlogits = (cache["probs"] - jax.nn.one_hot(y, cache["probs"].shape[-1])) / n
    dw2 = cache["h"].T @ dlogits
    db2 = dlogits.sum(0)
    dh = dlogits @ params["w2"].T
    dh_pre = dh * (1 - cache["h"] ** 2)
    dw1 = cache["x"].T @ dh_pre
    db1 = dh_pre.sum(0)
    return {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2}
