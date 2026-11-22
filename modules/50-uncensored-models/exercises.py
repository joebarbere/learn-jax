"""Module 50 exercises: removing refusals two ways, from the data and from the weights.

Replace each `raise NotImplementedError` with your code, then run `make test N=50`. The lesson reruns Eric
Hartford's dataset filter with your `unwanted`, shows what your `response_mask` trains on, scores refusals with your
`refusal_score`, finds the refusal direction with your `diff_in_means`, and removes it with your `ablate_dir` (at run
time) and `orthogonalize` (in the weights).
"""
import jax  # noqa: F401  (for your code)
import jax.numpy as jnp  # noqa: F401


# ---------------------------------------------------------------- 1. the data route
def unwanted(text, phrases):
    """Hartford's filter test: True if `text` contains any of `phrases`, ignoring upper/lower case.

    text: str. phrases: a list of str (may repeat, may differ only in case). Plain substring matching, so "bias"
    also matches "biases" and "unbiased". An empty list matches nothing. Return a Python bool.
    """
    raise NotImplementedError


def response_mask(lengths, starts, T):
    """Which next-token targets count in the fine-tuning loss: only the response's tokens.

    A batch of B examples is left-padded to T tokens: example b's n = lengths[b] real tokens sit at the end of row b
    (columns T − n … T − 1), and its response starts at token starts[b] of those n (prompt = the first starts[b]).
    The model predicts column t + 1 from column t, so the targets are tokens[:, 1:].
    lengths, starts: (B,) ints with 1 ≤ starts[b] ≤ lengths[b] ≤ T. Return a (B, T − 1) bool array, True where the
    target is a response token: for row b, target index j (token j + 1) with j + 1 ≥ T − n + starts[b].
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2. measuring refusal
def refusal_score(logits, token_ids):
    """Arditi et al.'s refusal score: log-odds that the next token is one that starts a refusal.

    logits: (..., V) any float dtype. token_ids: a list or (K,) array of distinct ids (e.g. the first tokens of
    "I'm sorry", "As an AI"). With p = softmax(logits) in float32 and P = Σ_k p[token_ids[k]],
    return log P − log(1 − P), shape (...), float32. Add 1e-8 inside both logs so P = 0 or 1 stays finite.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3. the weight route
def diff_in_means(harmful, harmless):
    """One candidate refusal direction per layer: mean activation on harmful prompts − mean on harmless ones.

    harmful: (L, N1, d), harmless: (L, N2, d), any float dtype (N1 and N2 may differ). Return (L, d) float32.
    Do not normalise.
    """
    raise NotImplementedError


def ablate_dir(x, u):
    """Remove direction u from activations: x − (x · û) û, where û = u / ‖u‖.

    x: (..., d), u: (d,) (any length, not zero). Compute in float32 and return x's dtype. Afterwards x · û = 0
    (to float precision) and the part of x orthogonal to u is unchanged.
    """
    raise NotImplementedError


def orthogonalize(W, u):
    """Bake `ablate_dir` into a weight matrix that writes to the residual stream.

    W: (d_in, d) — a layer computes y = h @ W, a (…, d) write into the stream. Return W' = W − (W û) ûᵀ (û = u/‖u‖),
    shape (d_in, d), W's dtype, computed in float32, so that h @ W' == ablate_dir(h @ W, u) for every h.
    """
    raise NotImplementedError
