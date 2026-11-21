"""Module 47 exercises: reading what a language model computes, and nudging it.

Replace each `raise NotImplementedError` with your code, then run `make test N=47`. The lesson reads SmolLM2-135M's
residual stream with your logit lens, builds its sparse autoencoder's code with your TopK, scores it with your
FVU, steers the model along your mean-difference direction, bakes that direction out of the weights with your
ablation, and compares two models' representations with your mutual nearest neighbours.

Convention: activations are row vectors and weights act as x @ W, so a matrix that writes into the residual
stream has shape (d_in, d_model), as in the rest of the course.
"""
import jax  # noqa: F401  (for your code)
import jax.numpy as jnp  # noqa: F401


# ---------------------------------------------------------------- 1. lenses
def logit_lens(h, norm_w, W_U, eps: float = 1e-5):
    """Decode a residual-stream state as if it were the last one: the model's final RMSNorm, then its output
    matrix.

    h: (..., d) activations (any float dtype). norm_w: (d,) the final norm's gain. W_U: (d, V) the unembedding.
      rms(h) = sqrt(mean(h², over the last axis) + eps)
      logits = (h / rms(h) · norm_w) @ W_U
    Compute in float32 and return (..., V) float32 logits.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2. sparse autoencoders
def topk_activation(pre, k: int):
    """The TopK SAE's nonlinearity (Gao et al. 2024): in each row of pre (N, F), keep the k largest entries,
    pass them through a ReLU (so a kept negative value becomes 0), and set every other entry to 0.

    Return an array shaped like pre. Assume no ties within a row. k may equal F (then it is just a ReLU).
    """
    raise NotImplementedError


def fvu(x, x_hat):
    """Fraction of variance unexplained by a reconstruction: how much of x's spread the SAE misses.

    x, x_hat: (N, d). FVU = Σ ‖x − x_hat‖² / Σ ‖x − mean(x)‖², where mean(x) is the (d,) mean over the N rows
    and the sums run over all N rows. 0 = perfect; 1 = no better than predicting the mean. Return a float32
    scalar.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3. steering
def mean_difference(pos, neg):
    """A steering direction from contrasting examples (CAA, Rimsky et al. 2023; also Fan et al.'s self-edits).

    pos: (N, d) activations on examples with the property (positive reviews); neg: (M, d) without it.
    Return the unit vector (d,) float32 along mean(pos) − mean(neg).
    """
    raise NotImplementedError


def ablate_direction(W, u, alpha: float = 1.0):
    """Edit a weight matrix so that what it writes has less (or none) of direction u.

    W: (d_in, d) writes x @ W into the residual stream; u: (d,) unit vector.
    Return W − alpha · (W u) uᵀ, shape (d_in, d). With alpha = 1, every output x @ W' has zero component
    along u; with alpha = 2 the component is reflected (flipped in sign); alpha = 0 changes nothing.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4. two models, one space?
def mutual_knn(X, Y, k: int):
    """How much two representations agree about which inputs are similar (Huh et al. 2024, "Platonic").

    X: (N, d1) and Y: (N, d2) are two models' vectors for the same N inputs (row i ↔ row i). For each input i,
    find its k nearest neighbours among the *other* inputs by cosine similarity, once in X and once in Y.
    Return the mean over i of |neighbours_X(i) ∩ neighbours_Y(i)| / k, a float in [0, 1]. Exclude i itself.
    """
    raise NotImplementedError
