"""Module 29 exercises: the perceptron (1958) and the Hopfield network (1982), in JAX.

Replace each `raise NotImplementedError` with your code, then run `make test N=29`.
Conventions used throughout:
  * Perceptron labels are +1 / -1 (Rosenblatt's convention), not 0 / 1.
  * Hopfield states are vectors of +1 / -1 ("spins"), shape (N,).
Everything should work on JAX arrays (and inside `jax.jit`/`jax.vmap` where noted).
"""

import jax  # noqa: F401 - you'll want jax.lax and jax.random
import jax.numpy as jnp  # noqa: F401


def perceptron_update(w, b, x, y, lr=1.0):
    """One step of Rosenblatt's mistake-driven learning rule.

    w: (d,) weights, b: scalar bias, x: (d,) input, y: label in {+1, -1}.
    If the example is misclassified — y * (w·x + b) <= 0 (a point exactly on the boundary counts as a
    mistake) — move the boundary toward it: w ← w + lr·y·x, b ← b + lr·y. Otherwise change nothing.
    Return (w, b). Must be jit-friendly: no Python `if` on traced values (use jnp.where).
    """
    raise NotImplementedError


def train_perceptron(X, y, epochs, lr=1.0):
    """Train a perceptron from w = 0, b = 0 by sweeping the data in order, `epochs` times.

    X: (n, d), y: (n,) in {+1, -1}. Visit examples in the given order each epoch, applying
    perceptron_update to each one. Return (w, b, mistakes_per_epoch) where mistakes_per_epoch is an int
    array of shape (epochs,) counting how many updates (mistakes) happened in each epoch.
    Hint: `jax.lax.scan` over examples inside a `jax.lax.scan` over epochs.
    """
    raise NotImplementedError


def is_linearly_separable(X, y, max_epochs=200):
    """Heuristic: True if a perceptron makes a full mistake-free pass within `max_epochs` epochs.

    The perceptron convergence theorem (Novikoff, 1962) says that if the data *are* separable with
    margin γ and radius R, the rule makes at most (R/γ)² mistakes — so it eventually stops. The converse
    test is only a heuristic: failing to converge in max_epochs doesn't *prove* the data aren't separable
    (the margin might just be tiny). Return a Python bool.
    """
    raise NotImplementedError


def hebbian_weights(patterns):
    """Hopfield's storage rule: "neurons that fire together wire together".

    patterns: (P, N) array of ±1. Return the (N, N) weight matrix W = (1/N) Σ_p x_p x_pᵀ with a zero
    diagonal (no neuron connects to itself). W must be symmetric.
    """
    raise NotImplementedError


def hopfield_energy(W, s):
    """The network's energy for state s (±1 vector): E = -½ sᵀ W s. Return a scalar."""
    raise NotImplementedError


def hopfield_recall(W, s, steps, key):
    """Asynchronous recall: `steps` single-neuron updates starting from state s.

    At step t pick a neuron i uniformly at random (derive the randomness from `key`, e.g.
    jax.random.fold_in(key, t)), compute its local field h = W[i]·s, and set s[i] = +1 if h > 0,
    -1 if h < 0, and leave it unchanged if h == 0. Return the final state (same shape and dtype as s).
    Use jax.lax.fori_loop so it compiles to one loop and can be vmapped over many starting states.
    """
    raise NotImplementedError


def modern_hopfield_update(X, xi, beta):
    """One step of the *modern* (continuous) Hopfield network of Ramsauer et al. (2020).

    X: (P, d) stored patterns (rows), xi: (d,) current state or (M, d) several states, beta: inverse
    temperature. New state = softmax(beta · xi Xᵀ) X, with the softmax over the P stored patterns.
    That is exactly attention with query xi, keys X, values X and scale beta. Return the new state(s),
    same shape as xi.
    """
    raise NotImplementedError
