"""Module 46 exercises: setting, compiling, merging and editing weights without training them.

Replace each `raise NotImplementedError` with your code, then run `make test N=46`. The lesson builds a digit
classifier from your filters and readout, compiles a program into attention weights with your selector, merges
two fine-tunes of SmolLM2 with your TIES, and rewrites one fact in SmolLM2-360M with your rank-one edit.

Convention: weights act on row vectors, x @ W (so a layer's W is (d_in, d_out)), as in the rest of the course.
"""
import jax  # noqa: F401  (for your code)
import jax.numpy as jnp  # noqa: F401
import numpy as np  # noqa: F401


# ---------------------------------------------------------------- 1. a network with no training
def ridge_filter(theta: float, size: int = 5, width: float = 1.0):
    """A size × size filter that responds to a bright line (a pen stroke) at angle theta (radians, 0 = horizontal,
    increasing counter-clockwise when y points down the image: theta = π/2 is a vertical stroke).

    With pixel coordinates centred on the middle (x = column − c, y = row − c, c = (size − 1) / 2):
      across = −sin(theta)·x + cos(theta)·y     distance from the line through the centre
      along  =  cos(theta)·x + sin(theta)·y     position along it
      f = (exp(−across² / (2 width²)) − 0.5 · exp(−across² / (2 (2.5 width)²))) · exp(−along² / 8)
    A narrow bump minus a wide one: positive on the line, negative beside it. Then subtract the mean (so a
    uniform patch gives 0) and divide by the sum of absolute values (so every orientation has the same scale).
    Return a (size, size) float32 array.
    """
    raise NotImplementedError


def firing_rates(fires, labels, n_classes: int, alpha: float = 1.0):
    """How often each feature fires for each class: the table you'd use to unit-test one feature at a time.

    fires: (N, F) bool, whether feature f fired on image n. labels: (N,) int in [0, n_classes).
    rate[c, f] = (number of class-c images where f fired + alpha) / (number of class-c images + 2 alpha)
    (Laplace smoothing: never exactly 0 or 1, so the log-odds below stay finite, even for an empty class).
    Return (n_classes, F) float32.
    """
    raise NotImplementedError


def log_odds_readout(rates, prior):
    """A linear readout built from counts, not gradients: naive Bayes for binary features.

    rates: (C, F), P(feature f fires | class c). prior: (C,), P(class c).
    Treating the features as independent given the class, the log-probability of class c given a 0/1 feature
    vector x is, up to a constant shared by all classes,
      Σ_f [x_f log r_cf + (1 − x_f) log(1 − r_cf)] + log prior_c  =  x @ W[:, c] + b[c]
    with W[f, c] = log r_cf − log(1 − r_cf) and b[c] = Σ_f log(1 − r_cf) + log prior_c.
    Return (W, b): (F, C) and (C,).
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2. compiling a program into attention
def position_selector(target, scale: float = 50.0):
    """Query and key weights for an attention head that makes position i read position target[i].

    The head sees one-hot positions: P = eye(L), row i is position i. Return (W_Q, W_K), each (L, L), so that the
    scores S = (P @ W_Q) @ (P @ W_K).T equal `scale` where key j == target[i] and 0 elsewhere. After softmax the
    head attends (almost) only to target[i]: with scale 50 the leftover weight is about e^-50.
    target: (L,) int. Reverse is target = L-1-i; "look at the previous token" is target = i-1 (Module 37).
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3. merging fine-tunes
def ties_merge(taus, density: float = 0.2, lam: float = 1.0):
    """TIES-Merging (Yadav et al. 2023) of n task vectors (fine-tuned weights minus base weights).

    taus: (n, P), one flattened task vector per row. Three steps:
      1. trim: in each row keep the round(density · P) entries with the largest |value|, set the rest to 0
         (ties at the threshold: keep any entry with |value| ≥ the k-th largest |value| of its row)
      2. elect: sign of each column's sum of the trimmed values (+1, −1, or 0)
      3. disjoint mean: average, over rows, only the trimmed entries that are non-zero and have the elected sign
         (0 where no row qualifies)
    Return lam × that average: (P,). With one task and density 1 it is lam × the task vector.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4. editing one fact
def rank_one_edit(W, k, v, C_inv):
    """ROME's rank-one update (Meng et al. 2022): change W as little as possible so that k @ W' = v.

    W: (d_in, d_out), the MLP's output matrix (w_down); k: (d_in,), the key, its input at the subject's last
    token; v: (d_out,), the value you want written there; C_inv: (d_in, d_in), the inverse of the keys'
    covariance (C = mean of k kᵀ over ordinary text), so the edit moves other keys as little as possible.
      u  = C_inv @ k
      W' = W + outer(u, v − k @ W) / (u · k)
    Return W' (d_in, d_out). Check: k @ W' = v exactly; a key orthogonal (under C_inv) to k is unchanged.
    """
    raise NotImplementedError
