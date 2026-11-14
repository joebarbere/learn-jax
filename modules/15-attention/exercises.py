"""Module 15 exercises: attention from scratch.

Replace each `raise NotImplementedError` with your code, then run `make test N=15`.
The lesson calls these functions, so its "Your turn" cells come alive as you finish them.

Layout conventions (the same as deepseek-v3-jax, so you can read that code afterwards):
  B = batch, T = query positions, S = key/value positions, D = model width,
  H = number of heads, d = D // H = width of one head.
  Activations are (B, T, D); split into heads they are (B, T, H, d) — "bthd".
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def causal_mask(T: int):
    """Boolean (T, T) mask: entry [t, s] is True when query position t may attend to key position s.

    A language model may only look at the present and the past: s <= t. The diagonal is True.
    """
    raise NotImplementedError


def scaled_dot_product(q, k, v, mask=None):
    """Single-head attention. Returns (out, weights).

    q: (..., T, d)   k: (..., S, d)   v: (..., S, dv)
    mask: optional boolean array broadcastable to (..., T, S); True = allowed.

    scores  = q @ kᵀ / sqrt(d)                 shape (..., T, S)
    masked positions get -inf *before* the softmax (so they receive exactly zero weight)
    weights = softmax(scores) over the last axis (each row sums to 1)
    out     = weights @ v                      shape (..., T, dv)

    You may assume every row of the mask allows at least one position.
    """
    raise NotImplementedError


def split_heads(x, n_heads: int):
    """(B, T, D) -> (B, T, H, D // H). Head h owns features [h*d, (h+1)*d). Raise ValueError if
    D is not divisible by n_heads."""
    raise NotImplementedError


def merge_heads(x):
    """(B, T, H, d) -> (B, T, H*d). The exact inverse of split_heads."""
    raise NotImplementedError


def attention_weights(params, x, n_heads: int, causal: bool = True):
    """The attention pattern of multi-head self-attention: (B, H, T, T), rows sum to 1.

    params: {"wq", "wk", "wv", "wo"}, each (D, D). Project q = x @ wq and k = x @ wk, split both
    into heads, scores = einsum("bthd,bshd->bhts", q, k) / sqrt(d), apply the causal mask if
    `causal`, softmax over the last axis.
    """
    raise NotImplementedError


def multi_head_attention(params, x, n_heads: int, causal: bool = True):
    """Multi-head self-attention, (B, T, D) -> (B, T, D).

    Use attention_weights for the pattern, v = split_heads(x @ wv), mix the values with
    einsum("bhts,bshd->bthd", weights, v), merge the heads, then project with wo.
    """
    raise NotImplementedError


def sinusoidal_positions(T: int, D: int):
    """The Transformer paper's position encodings, (T, D) float32, D even.

    For position p and pair index i in [0, D/2):
        PE[p, 2i]   = sin(p / 10000**(2i / D))
        PE[p, 2i+1] = cos(p / 10000**(2i / D))
    """
    raise NotImplementedError


def is_permutation_equivariant(f, x, perm, atol: float = 1e-4) -> bool:
    """True if shuffling the tokens of x just shuffles f's output the same way.

    x: (B, T, D); perm: a permutation of range(T). Check f(x[:, perm]) ≈ f(x)[:, perm] within
    `atol` (absolute). Return a Python bool.
    """
    raise NotImplementedError
