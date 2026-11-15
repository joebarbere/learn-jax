"""Module 05 exercises: build the optimizers yourself.

Replace each `raise NotImplementedError` with your code, then run `make test N=05`.

Every optimizer here works on *pytrees* — nested dicts/lists/tuples of arrays, the way JAX models
store parameters. `jax.tree.map(f, tree_a, tree_b, ...)` applies `f` leaf by leaf across trees with
the same structure; that's the only tool you need to go from "one array" to "a whole model".
"""
import jax  # noqa: F401  (you will need these)
import jax.numpy as jnp  # noqa: F401


def sgd_step(params, grads, lr: float):
    """Plain gradient descent: return new params = params - lr * grads, leaf by leaf."""
    raise NotImplementedError


def momentum_step(params, grads, velocity, lr: float, beta: float = 0.9):
    """Heavy-ball momentum, in the form optax.sgd(lr, momentum=beta) uses.

    velocity <- grads + beta * velocity      (a running sum of past gradients)
    params   <- params - lr * velocity

    `velocity` has the same structure as params; start it at zeros.
    Return (new_params, new_velocity).
    """
    raise NotImplementedError


def adam_step(params, grads, state, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999,
              eps: float = 1e-8):
    """One Adam step. `state` is a tuple (m, v, t):

    m: running mean of gradients (pytree like params, starts at zeros)
    v: running mean of squared gradients (pytree like params, starts at zeros)
    t: number of steps taken so far (int, starts at 0)

    t      <- t + 1
    m      <- b1 * m + (1 - b1) * g
    v      <- b2 * v + (1 - b2) * g**2
    m_hat  <- m / (1 - b1**t)          (bias correction: m and v start at 0, so early on they're
    v_hat  <- v / (1 - b2**t)           too small — dividing undoes that)
    params <- params - lr * m_hat / (sqrt(v_hat) + eps)

    Return (new_params, (m, v, t)). Must match optax.adam to float32 precision.
    """
    raise NotImplementedError


def warmup_cosine(step: int, peak: float, warmup: int, total: int, final: float) -> float:
    """Learning rate at `step`: linear warmup from 0 to `peak` over `warmup` steps, then cosine
    decay from `peak` down to `final`, reached at step `total` and held after.

        step <  warmup:          peak * step / warmup
        warmup <= step <= total: final + (peak - final) * 0.5 * (1 + cos(pi * (step - warmup) / (total - warmup)))
        step >  total:           final

    This is optax.warmup_cosine_decay_schedule(0, peak, warmup, total, final). Return a float.
    """
    raise NotImplementedError


def clip_by_global_norm(grads, max_norm: float):
    """If the global norm of all gradients (sqrt of the sum of every squared element, across every
    leaf) exceeds `max_norm`, scale every leaf by max_norm / global_norm; otherwise return them
    unchanged. Direction is preserved — only the length shrinks. All-zero gradients must not
    produce NaNs."""
    raise NotImplementedError


def fit(loss_fn, params, data, steps: int, lr: float):
    """Train with plain gradient descent: repeat `steps` times
        grads = jax.grad(loss_fn)(params, data); params = sgd_step(params, grads, lr)
    and return the final params. `loss_fn(params, data)` returns a scalar.
    Tip: jax.jit the step so the loop is fast."""
    raise NotImplementedError


# --- Advanced sidebar: Muon (Section 11) -------------------------------------------------------------

NS_COEFFS = (3.4445, -4.775, 2.0315)  # Keller Jordan's quintic Newton–Schulz coefficients (a, b, c)


def newton_schulz(G, steps: int = 5, coeffs=NS_COEFFS, eps: float = 1e-7):
    """Approximate the polar factor U·Vᵀ of a 2-D matrix G (from its SVD G = U·S·Vᵀ) using only matmuls.

    1. X = G / (‖G‖_F + eps)   — Frobenius normalization puts every singular value in [0, 1].
    2. If G has more rows than columns, work on X.T (so X·Xᵀ is the *small* Gram matrix) and transpose
       the result back at the end.
    3. Repeat `steps` times, with (a, b, c) = coeffs and A = X·Xᵀ:
           X ← a·X + (b·A + c·A·A)·X
    The result has the same shape as G and singular values pushed into a band around 1 (≈0.7–1.2 with
    these coefficients — deliberately not exactly 1; that's what makes 5 steps enough).
    """
    raise NotImplementedError


def muon_step(param, grad, momentum, lr: float, beta: float = 0.95, nesterov: bool = True,
              eps: float = 1e-8):
    """One Muon update of a single 2-D weight `param` of shape (fan_in, fan_out) — the x @ W layout used
    throughout this course. Matches `optax.contrib.muon` (its default 'frobenius' preconditioning and
    width scaling) exactly.

    `momentum` is the state tuple (mu, t): the momentum buffer (zeros like `param` at the start) and the
    number of steps taken so far (0 at the start). Return (new_param, (new_mu, new_t)).

        mu ← beta·mu + (1 − beta)·grad;  t ← t + 1
        nesterov:  m̂ = beta·mu/(1 − beta^(t+1)) + (1 − beta)·grad/(1 − beta^t)
        otherwise: m̂ = mu/(1 − beta^t)                      (bias-corrected, like Adam's m)
        O = newton_schulz(m̂, eps=eps)
        scale = sqrt(max(1, fan_out / fan_in))              (optax's "width transfer" scaling)
        param ← param − lr · scale · O
    """
    raise NotImplementedError
