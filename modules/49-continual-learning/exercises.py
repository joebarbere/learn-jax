"""Module 49 exercises: measuring forgetting, and three ways to learn without it.

Replace each `raise NotImplementedError` with your code, then run `make test N=49`. The lesson scores every
experiment with your `forgetting_metrics`, protects old weights with your `ewc_penalty`, adds new capacity with your
`expert_out`, distils with your `token_kl`, rewards imitation with your `levenshtein_reward` and picks dreams with
your `select_dreams`.
"""
import jax  # noqa: F401  (for your code)
import jax.numpy as jnp  # noqa: F401
import numpy as np  # noqa: F401


# ---------------------------------------------------------------- 1. measuring forgetting
def forgetting_metrics(R):
    """Summarise a continual-learning accuracy matrix.

    R: (T, T) array-like of floats. R[i, j] = accuracy on task j's test set after training on tasks 0 … i (rows are
    "after day i", columns are "facts of day j"; entries above the diagonal are tasks not yet trained on).
    Return a dict of Python floats:
      "avg"        mean of the last row: what the model knows at the end, over all T tasks
      "bwt"        backward transfer (Lopez-Paz & Ranzato 2017): mean over j < T−1 of R[T−1, j] − R[j, j].
                   Negative means later training hurt earlier tasks.
      "forgetting" (Chaudhry et al. 2018): mean over j < T−1 of (max over i < T−1 of R[i, j]) − R[T−1, j].
    With T = 1 there is nothing earlier to forget: bwt = forgetting = 0.0.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2. protecting old weights
def ewc_penalty(params, anchor, fisher):
    """Elastic weight consolidation's penalty (Kirkpatrick et al. 2017): Σ_i F_i (θ_i − θ*_i)².

    params, anchor, fisher: pytrees with the same structure (dicts of arrays, any float dtypes). anchor = θ*, the
    weights to stay near; fisher = F, each weight's importance (≥ 0). Compute in float32 and return a float32
    scalar summed over every leaf and every element. Equal params and anchor give exactly 0.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3. adding capacity
def expert_out(h, A, B, gate):
    """The output of a layer's low-rank experts: Σ_e gate[e] · silu(h @ A[e]) @ B[e].

    h: (batch, T, d). A: (n, d, r). B: (n, r, d). gate: (n,) with 1.0 for an expert that is switched on and 0.0 for
    one that is allocated but not active yet. Return (batch, T, d) in h's dtype. An all-zero gate returns zeros,
    and an expert whose B is zero adds nothing (that is how a new expert starts).
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4. distillation
def token_kl(teacher_logits, student_logits, kind="forward"):
    """Per-token divergence between two next-token distributions, over the whole vocabulary.

    teacher_logits, student_logits: (..., V), any float dtype; compute with log_softmax in float32. With
    p = softmax(teacher), q = softmax(student):
      kind="forward": KL(p ‖ q) = Σ_v p (log p − log q)
      kind="reverse": KL(q ‖ p) = Σ_v q (log q − log p)
      kind="jsd":     ½ KL(p ‖ m) + ½ KL(q ‖ m), m = (p + q) / 2
    Return shape (...), float32. All three are 0 when the logits are equal (up to a constant shift).
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 5. learning to imitate
def levenshtein_reward(pred, target, z0):
    """The Sleep paper's token-level imitation reward (Behrouz et al. 2026, Eq. 4).

    pred, target: sequences (lists of token ids, or strings). z = the Levenshtein (edit) distance between them: the
    fewest single-element insertions, deletions and substitutions that turn one into the other.
    Return 1 − z / max(len(pred), len(target)) if z ≤ z0, else 0.0, as a Python float. Two empty sequences are
    identical: return 1.0.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 6. choosing dreams
def select_dreams(scores, k, b, seed=0):
    """Pick which self-generated examples ("dreams") to train on: the k with the highest importance score, plus b
    others chosen at random to keep variety (the Sleep paper's dream selection, §3.4).

    scores: (n,) floats. Return a sorted Python list of distinct indices: the k largest scores (ties: lower index
    first), then b indices drawn uniformly without replacement from the rest with np.random.default_rng(seed).
    If k + b ≥ n, return every index.
    """
    raise NotImplementedError
