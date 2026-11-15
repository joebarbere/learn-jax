"""Module 11 exercises: the small tools that make deep networks actually train.

Replace each `raise NotImplementedError` with your code, then run `make test N=11`.
All array functions take and return `jax.numpy` arrays. layer_norm / rms_norm normalize over the **last**
axis; batch_norm normalizes over the **batch** (first) axis. manual_backward must not call jax.grad.
"""
import jax
import jax.numpy as jnp


def init_scale(fan_in: int, fan_out: int, kind: str) -> float:
    """Standard deviation to initialize a dense layer's weights with.

    kind="he"     → sqrt(2 / fan_in)              (for ReLU networks)
    kind="xavier" → sqrt(2 / (fan_in + fan_out))  (Glorot; for tanh / linear networks)
    Any other kind: raise ValueError.
    """
    raise NotImplementedError


def layer_norm(x, gamma, beta, eps: float = 1e-5):
    """LayerNorm over the last axis: (x - mean) / sqrt(var + eps) * gamma + beta.

    x: (..., d); gamma, beta: (d,). Use the biased variance (mean of squared deviations).
    """
    raise NotImplementedError


def rms_norm(x, w, eps: float = 1e-6):
    """RMSNorm over the last axis, as in deepseek-v3-jax: x / sqrt(mean(x²) + eps) * w.

    No mean subtraction and no bias. x: (..., d); w: (d,).
    """
    raise NotImplementedError


def dropout(key, x, rate: float, train: bool):
    """Inverted dropout.

    When train is False, or rate == 0, return x unchanged.
    Otherwise zero each element independently with probability `rate` and scale the survivors by
    1 / (1 - rate) so the expected value is unchanged. rate == 1 returns all zeros (no division by 0).
    Same key → same mask.
    """
    raise NotImplementedError


def residual_block(params: dict, x):
    """A pre-norm residual MLP block: x + relu(rms_norm(x, params["norm"]) @ params["w1"]) @ params["w2"].

    params: {"norm": (d,), "w1": (d, h), "w2": (h, d)}; x: (..., d). Returns (..., d).
    Use rms_norm with its default eps. (If you haven't done rms_norm yet, do that first.)
    """
    raise NotImplementedError


def global_grad_norm(grads) -> jnp.ndarray:
    """The L2 norm of all gradient values in a pytree, as if flattened into one long vector.

    sqrt(sum over every leaf of sum(leaf²)). An empty pytree has norm 0.
    """
    raise NotImplementedError


def save_and_restore(state, path):
    """Save a pytree of arrays with Orbax to `path`, then restore and return it.

    Use `orbax.checkpoint.StandardCheckpointer`: `.save(path, state)`, `.wait_until_finished()`, then
    `.restore(path)`. Orbax needs an absolute path that doesn't exist yet — convert with
    `os.path.abspath`. The result should equal `state` leaf for leaf.
    """
    raise NotImplementedError


def batch_norm(x, gamma, beta, running_mean, running_var, momentum: float = 0.1, train: bool = True,
               eps: float = 1e-5):
    """BatchNorm over the **batch** axis (axis 0) of x: (N, d). Returns (y, new_running_mean, new_running_var).

    train=True:  normalize with this batch's per-feature mean and *biased* variance (mean of squared
                 deviations over axis 0): y = (x - mean) / sqrt(var + eps) * gamma + beta.
                 Update the running statistics as exponential moving averages, PyTorch-style —
                 `momentum` is the weight on the *new* batch statistic:
                     new_running = (1 - momentum) * running + momentum * batch_stat
                 (use the same biased variance for the running variance).
    train=False: normalize with running_mean / running_var instead, and return them unchanged.
    gamma, beta, running_mean, running_var: (d,).
    """
    raise NotImplementedError


def forward_with_cache(params: dict, x, y):
    """Provided (already done): a 2-layer tanh MLP with mean cross-entropy, returning (loss, cache).

    params: {"w1": (D, H), "b1": (H,), "w2": (H, C), "b2": (C,)}; x: (N, D); y: (N,) int labels.
    cache holds every intermediate the backward pass needs: x, h_pre, h, logits, probs.
    """
    h_pre = x @ params["w1"] + params["b1"]
    h = jnp.tanh(h_pre)
    logits = h @ params["w2"] + params["b2"]
    probs = jax.nn.softmax(logits, axis=-1)
    loss = -jnp.mean(jnp.log(probs[jnp.arange(x.shape[0]), y]))
    return loss, {"x": x, "h_pre": h_pre, "h": h, "logits": logits, "probs": probs}


def manual_backward(params: dict, cache: dict, y) -> dict:
    """Backprop Ninja: the gradient of forward_with_cache's loss, written BY HAND — no jax.grad.

    Return a dict with the same keys and shapes as params ("w1", "b1", "w2", "b2"). Work backwards:
      dlogits = (probs - onehot(y)) / N             (softmax + mean cross-entropy, combined)
      dw2 = hᵀ · dlogits,   db2 = Σ_rows dlogits,   dh = dlogits · w2ᵀ
      dh_pre = dh * (1 - h²)                        (tanh' = 1 - tanh²)
      dw1 = xᵀ · dh_pre,    db1 = Σ_rows dh_pre
    The lesson and tests compare every entry with jax.grad.
    """
    raise NotImplementedError
