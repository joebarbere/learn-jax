"""Module 04 exercises: softmax, cross-entropy, entropy, KL, sampling and perplexity.

Replace each `raise NotImplementedError` with your code, then run `make test N=04`.
Write these with `jax.numpy` (jnp) so they work on the GPU and under `jax.grad`. Don't call
`jax.nn.softmax` / `jax.nn.log_softmax` / `jax.random.categorical` — the point is to build them.
"""
import jax  # noqa: F401  (jax.random for sample_categorical)
import jax.numpy as jnp  # noqa: F401


def stable_softmax(x, axis: int = -1):
    """Softmax along `axis`: exp(x_i) / sum_j exp(x_j), without overflowing.

    Subtract the max along `axis` first (keepdims=True) — softmax is unchanged by adding a constant,
    and the largest exponent becomes exp(0) = 1. Must handle x = [1000, 1001, 1002].
    """
    raise NotImplementedError


def log_softmax(x, axis: int = -1):
    """log(softmax(x)) computed stably: x - logsumexp(x), with logsumexp = m + log(sum(exp(x - m))).

    Don't compute log(stable_softmax(x)) — tiny probabilities underflow to 0 and log(0) = -inf.
    """
    raise NotImplementedError


def cross_entropy(logits, labels):
    """Mean cross-entropy loss for classification.

    logits: (batch, classes) float; labels: (batch,) integer class indices.
    Loss per example = -log softmax(logits)[label]; return the mean over the batch (a scalar).
    Hint: log_softmax, then pick each row's label with jnp.take_along_axis.
    """
    raise NotImplementedError


def entropy(p):
    """Shannon entropy H(p) = -sum_i p_i log p_i in nats (natural log), over the last axis.

    Convention: 0 · log 0 = 0. Use jnp.where so zeros contribute 0 (not nan).
    """
    raise NotImplementedError


def kl_divergence(p, q):
    """KL(p || q) = sum_i p_i log(p_i / q_i) over the last axis, in nats.

    Terms where p_i = 0 contribute 0. Assume q_i > 0 wherever p_i > 0.
    """
    raise NotImplementedError


def sample_categorical(key, logits, n: int):
    """Draw n samples (shape (n,), integer class indices) from softmax(logits) using the Gumbel-max trick.

    logits: (classes,). Add independent Gumbel noise g = -log(-log(u)), u ~ Uniform(0, 1), to the
    logits — one noise vector per sample, shape (n, classes) — and take the argmax of each row.
    Use jax.random.uniform with minval=1e-20 (or similar) so log(0) never happens.
    """
    raise NotImplementedError


def perplexity(mean_cross_entropy):
    """Perplexity = exp(mean cross-entropy in nats): the 'effective number of equally likely choices'."""
    raise NotImplementedError

