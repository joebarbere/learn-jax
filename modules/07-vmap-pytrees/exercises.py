"""Module 07 exercises: batching for free, and models as data.

Replace each `raise NotImplementedError` with your code, then run `make test N=07`.

Two ideas carry this module: write a function for *one* example and let `jax.vmap` batch it; and
keep every bit of model state in *pytrees* (nested dicts/lists/tuples of arrays) that you pass in and
return — no hidden state anywhere.
"""
import jax  # noqa: F401  (you will need these)
import jax.numpy as jnp  # noqa: F401


def batched_dot(a, b):
    """Row-wise dot products: a and b are (N, D); return (N,) with out[i] = a[i] · b[i].

    Use jax.vmap over jnp.dot — don't write a loop and don't use einsum/sum tricks.
    """
    raise NotImplementedError


def pairwise_distances(x, y):
    """Euclidean distances between every row of x (N, D) and every row of y (M, D) -> (N, M).

    Write the distance between two single vectors, then vmap it twice (once over x, once over y).
    """
    raise NotImplementedError


def per_example_grads(loss_fn, params, xs, ys):
    """Gradient of loss_fn for each example separately.

    loss_fn(params, x, y) computes the loss of ONE example (x, y) and returns a scalar.
    xs, ys have a leading batch axis of size B. Return a pytree shaped like params where every leaf
    gets an extra leading axis of size B: leaf[i] is the gradient for example i.
    Hint: jax.vmap(jax.grad(loss_fn), in_axes=...) — params are shared, the data is batched.
    """
    raise NotImplementedError


def tree_l2_norm(tree):
    """The global L2 norm of every number in a pytree: sqrt(sum of squares over all leaves)."""
    raise NotImplementedError


def tree_axpy(a, x, y):
    """Return a * x + y leaf by leaf, for scalar a and pytrees x, y of the same structure."""
    raise NotImplementedError


def init_mlp(key, sizes):
    """Parameters of a multi-layer perceptron with layer widths `sizes`, e.g. [784, 128, 10].

    Return a list with one dict per layer: {"w": (fan_in, fan_out), "b": (fan_out,)}.
    w ~ Normal(0, 1) * sqrt(2 / fan_in)   (He initialization — Module 11 explains why),
    b = zeros. Use a *different* key for every layer: split `key`, never reuse it.
    The same key must always give the same parameters.
    """
    raise NotImplementedError


def mlp_forward(params, x):
    """Run the MLP: x @ w + b for each layer, with ReLU after every layer except the last.

    x is a single example (D,) or a batch (B, D) — the same code handles both.
    """
    raise NotImplementedError


def count_params(params) -> int:
    """Total number of scalars in a pytree of arrays (as a Python int)."""
    raise NotImplementedError
