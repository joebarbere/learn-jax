"""Module 09 exercises: your first models, from scratch.

Replace each `raise NotImplementedError` with your code, then run `make test N=09`.
Use jax.numpy throughout; every function should also accept NumPy arrays.
"""
import jax  # noqa: F401  (you will need these)
import jax.numpy as jnp  # noqa: F401


def train_val_test_split(key, n: int, fractions=(0.7, 0.15, 0.15)):
    """Shuffle the indices 0..n-1 with `key` and cut them into train / val / test index arrays.

    n_train = int(fractions[0] * n), n_val = int(fractions[1] * n), test gets the rest.
    Return (train_idx, val_idx, test_idx): disjoint, together covering every index exactly once.
    Hint: jax.random.permutation.
    """
    raise NotImplementedError


def polynomial_features(x, degree: int):
    """Turn a 1-D array x of shape (n,) into columns [x**0, x**1, ..., x**degree], shape (n, degree + 1).

    The column of ones is the bias term, so a linear model on these features is a polynomial in x.
    """
    raise NotImplementedError


def ridge_closed_form(X, y, lam: float):
    """Ridge regression weights: the w minimising ||X w - y||² + lam * ||w||².

    Setting the gradient to zero gives (Xᵀ X + lam·I) w = Xᵀ y. Solve that with jnp.linalg.solve
    (never invert a matrix explicitly — Module 02). X is (n, d), y is (n,); return w of shape (d,).
    lam = 0 is ordinary least squares.
    """
    raise NotImplementedError


def bce_with_logits(logits, labels):
    """Mean binary cross-entropy from raw scores (logits), computed stably.

    For one example with logit z and label y in {0, 1}:
        loss = -[y·log σ(z) + (1 - y)·log(1 - σ(z))]
    which simplifies to the stable form  max(z, 0) - z·y + log(1 + exp(-|z|)).
    It must stay finite for |z| = 1000. Return the mean over all examples (a scalar).
    """
    raise NotImplementedError


def logistic_fit(X, y, steps: int = 500, lr: float = 0.5):
    """Fit logistic regression p(y=1|x) = σ(x·w + b) by full-batch gradient descent on bce_with_logits.

    X is (n, d), y is (n,) of 0/1. Start from w = zeros(d), b = 0. Use jax.grad; jit the step if you like.
    Return (w, b).
    """
    raise NotImplementedError


def accuracy(predictions, labels):
    """Fraction of predictions equal to labels, as a Python float in [0, 1]."""
    raise NotImplementedError


def confusion_matrix(y_true, y_pred, n_classes: int):
    """Counts M[i, j] = number of examples whose true class is i and predicted class is j.

    Return an integer array of shape (n_classes, n_classes). No Python loop over examples —
    hint: `.at[...].add(1)` with two index arrays, or one-hot vectors and a matmul.
    """
    raise NotImplementedError
