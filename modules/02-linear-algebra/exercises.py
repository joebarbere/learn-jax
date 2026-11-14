"""Module 02 exercises: linear algebra as geometry.

Replace each `raise NotImplementedError` with your code, then run `make test N=02`.
Use jax.numpy (jnp.linalg has norm, svd, solve, eigh, ...). Points and vectors are rows:
a set of n points in the plane has shape (n, 2).
"""
import jax.numpy as jnp


def cosine_similarity(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Cosine of the angle between vectors a and b along the last axis.

    a and b have the same shape (..., d); return shape (...). 1 = same direction, 0 = perpendicular,
    -1 = opposite. Formula: a·b / (|a| |b|). You may assume no vector is all zeros.
    """
    raise NotImplementedError


def rotation_matrix(theta: float) -> jnp.ndarray:
    """2×2 matrix that rotates column vectors counter-clockwise by `theta` radians.

    It sends the x-axis (1, 0) to (cos θ, sin θ) and the y-axis (0, 1) to (-sin θ, cos θ):
    those images are its columns.
    """
    raise NotImplementedError


def transform_points(m: jnp.ndarray, points: jnp.ndarray) -> jnp.ndarray:
    """Apply the 2×2 matrix m to every point in `points` (shape (n, 2)); return shape (n, 2).

    For one column vector p the result is m @ p. Do all n points at once without a loop
    (think about which side of the matrix the rows go on, and the transpose).
    """
    raise NotImplementedError


def project(v: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Orthogonal projection of vector v onto the line through vector u (both shape (d,)).

    The result points along u, and v - project(v, u) is perpendicular to u.
    Formula: (v·u / u·u) u.
    """
    raise NotImplementedError


def low_rank_approx(m: jnp.ndarray, k: int) -> jnp.ndarray:
    """Best rank-k approximation of a 2-D matrix m, via the SVD (keep the k largest singular values).

    Use jnp.linalg.svd(m, full_matrices=False) -> (U, s, Vt) and rebuild from the first k columns
    of U, the first k values of s and the first k rows of Vt. Returns the same shape as m.
    """
    raise NotImplementedError


def gram_schmidt(a: jnp.ndarray) -> jnp.ndarray:
    """Orthonormalize the columns of a (shape (n, k), k <= n, columns linearly independent).

    Return q of shape (n, k) whose columns are unit length and mutually perpendicular, where column
    j spans the same space as the first j+1 columns of a (so column 0 points the same way as a[:, 0]).
    Algorithm: for each column, subtract its projections onto the already-built q columns, then
    normalize. A Python loop over the k columns is fine here.
    """
    raise NotImplementedError
