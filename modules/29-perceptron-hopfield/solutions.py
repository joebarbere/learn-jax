"""Module 29 reference solutions. Try the exercises first — `make test N=29` grades yours."""

import jax
import jax.numpy as jnp


def perceptron_update(w, b, x, y, lr=1.0):
    mistake = y * (jnp.dot(w, x) + b) <= 0
    w = jnp.where(mistake, w + lr * y * x, w)
    b = jnp.where(mistake, b + lr * y, b)
    return w, b


def train_perceptron(X, y, epochs, lr=1.0):
    X = jnp.asarray(X, jnp.float32)
    y = jnp.asarray(y, jnp.float32)

    def sample_step(carry, xy):
        w, b = carry
        x, t = xy
        mistake = t * (jnp.dot(w, x) + b) <= 0
        return perceptron_update(w, b, x, t, lr), mistake

    def epoch(carry, _):
        carry, mistakes = jax.lax.scan(sample_step, carry, (X, y))
        return carry, mistakes.sum()

    init = (jnp.zeros(X.shape[1], jnp.float32), jnp.float32(0.0))
    (w, b), mistakes = jax.lax.scan(epoch, init, None, length=int(epochs))
    return w, b, mistakes.astype(jnp.int32)


def is_linearly_separable(X, y, max_epochs=200):
    _, _, mistakes = train_perceptron(X, y, max_epochs)
    return bool((mistakes == 0).any())


def hebbian_weights(patterns):
    patterns = jnp.asarray(patterns, jnp.float32)
    n = patterns.shape[1]
    W = patterns.T @ patterns / n
    return W * (1.0 - jnp.eye(n, dtype=W.dtype))


def hopfield_energy(W, s):
    s = jnp.asarray(s, W.dtype)
    return -0.5 * s @ W @ s


def hopfield_recall(W, s, steps, key):
    n = s.shape[0]

    def body(t, state):
        i = jax.random.randint(jax.random.fold_in(key, t), (), 0, n)
        h = jnp.dot(W[i], state.astype(W.dtype))
        new = jnp.where(h > 0, 1, jnp.where(h < 0, -1, state[i])).astype(state.dtype)
        return state.at[i].set(new)

    return jax.lax.fori_loop(0, steps, body, s)


def modern_hopfield_update(X, xi, beta):
    weights = jax.nn.softmax(beta * (xi @ X.T), axis=-1)
    return weights @ X
