"""Module 01 exercises: shapes, broadcasting and vectorized array code.

Replace each `raise NotImplementedError` with your code, then run `make test N=01`.
Rule for this module: **no Python loops over array elements** (loops over a shape tuple are fine
in `broadcast_shape`). Use jax.numpy operations and broadcasting instead.
"""
import jax.numpy as jnp


def broadcast_shape(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> tuple[int, ...]:
    """The shape NumPy/JAX broadcasting gives for operands of shapes a_shape and b_shape.

    Implement the rules by hand (don't call jnp.broadcast_shapes):
      1. Right-align the shapes; pad the shorter one on the left with 1s.
      2. Compare dimension by dimension: equal sizes match, and a size of 1 stretches to the other.
      3. Any other mismatch is an error: raise ValueError.
    Example: (8, 1, 3) and (5, 1) -> (8, 5, 3).
    """
    raise NotImplementedError


def normalize_rows(x: jnp.ndarray) -> jnp.ndarray:
    """Scale each row of a 2-D array to unit L2 norm (length 1).

    x has shape (n, d); return the same shape. A row that is all zeros has no direction —
    return it unchanged (all zeros) instead of dividing by zero.
    """
    raise NotImplementedError


def pairwise_sq_dists(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Squared Euclidean distance between every row of a (n, d) and every row of b (m, d).

    Return shape (n, m) where out[i, j] = sum_k (a[i, k] - b[j, k])**2.
    Hint: give a and b extra axes so their difference broadcasts to (n, m, d).
    """
    raise NotImplementedError


def one_hot(labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """Turn integer class labels of shape (n,) into a float32 one-hot matrix of shape (n, num_classes).

    Row i is all zeros except a 1.0 in column labels[i]. Hint: compare labels against
    jnp.arange(num_classes) with broadcasting.
    """
    raise NotImplementedError


def batched_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Multiply a stack of matrices: a is (batch, n, k), b is (batch, k, m) -> (batch, n, m).

    Write it with jnp.einsum (the subscripts read like the math).
    """
    raise NotImplementedError


def moving_average(x: jnp.ndarray, window: int) -> jnp.ndarray:
    """Mean of every length-`window` slice of a 1-D array, with no Python loop.

    For x of length n, return length n - window + 1, where out[i] = mean(x[i : i + window]).
    Hint: differences of jnp.cumsum (prepend a 0) give every window sum at once.
    Raise ValueError if window < 1 or window > len(x).
    """
    raise NotImplementedError
