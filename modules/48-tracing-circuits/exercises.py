"""Module 48 exercises: finding the circuit behind one behaviour.

Replace each `raise NotImplementedError` with your code, then run `make test N=48`. The lesson scores SmolLM2-135M
on the indirect-object task with your logit difference, ranks its layers and heads with your patching effect,
estimates every head at once with your attribution patching, and builds and prunes an attribution graph with
your edge weights, row normalisation and total influence.
"""
import jax  # noqa: F401  (for your code)
import jax.numpy as jnp  # noqa: F401


# ---------------------------------------------------------------- 1. measuring a behaviour
def logit_diff(logits, io, s):
    """How much the model prefers the right name over the wrong one, averaged over a batch.

    logits: (B, T, V) any float dtype. io, s: (B,) int token ids of the indirect object (the right answer) and the
    subject (the wrong one). Use the last position only:  mean over b of  logits[b, -1, io[b]] − logits[b, -1, s[b]].
    Return a float32 scalar.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2. activation patching
def patching_effect(patched, clean, corrupt):
    """Normalise a patched run's metric: 0 = no better than the corrupted run, 1 = fully restores the clean run.

    All three are scalars or arrays of the same shape (e.g. a whole grid of patched metrics against two scalars).
    Return (patched − corrupt) / (clean − corrupt). Values outside [0, 1] are allowed and meaningful.
    """
    raise NotImplementedError


def attribution_patch(grad, clean, corrupt):
    """Attribution patching: the first-order (linear) estimate of what patching clean activations into the corrupted
    run would change, for many activations at once from one backward pass.

    grad: (..., d) the gradient of the metric with respect to an activation, taken on the *corrupted* run.
    clean, corrupt: (..., d) that activation on the clean and corrupted runs.
    Return (...): the estimated change in the metric, Σ_d grad · (clean − corrupt).
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3. attribution graphs
def edge_weights(jac, values):
    """Edges of an attribution graph. In the frozen, linear replacement model, a source node's direct effect on a
    target is (∂ target / ∂ source) × the source's value.

    jac: (n_targets, n_sources) the derivatives. values: (n_sources,) the source nodes' values.
    Return (n_targets, n_sources).
    """
    raise NotImplementedError


def normalize_rows(A):
    """Prepare a graph for influence: take |A| and scale each row (a target's incoming edges) to sum to 1.
    A row of all zeros (a node with no inputs, like an embedding) stays all zeros — don't divide by zero.
    A: (n, n). Return (n, n).
    """
    raise NotImplementedError


def total_influence(A_norm, w):
    """Direct plus indirect influence of every node on a weighted set of targets.

    A_norm: (n, n) row-normalised graph, A_norm[t, s] = share of target t's input that comes from source s. Nodes
    are causally ordered so A_norm is nilpotent (no cycles: some power of it is all zeros).
    w: (n,) weights on the targets (e.g. the logit nodes' probabilities, zeros elsewhere).
    Return x = w A + w A² + w A³ + …  (n,), which equals w A (I − A)⁻¹. Sum the series: n terms are always enough.
    """
    raise NotImplementedError
