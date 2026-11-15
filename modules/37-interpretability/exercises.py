"""Module 37 exercises: the measuring instruments of interpretability.

Replace each `raise NotImplementedError`, then run `make test N=37`. The lesson uses your functions to find
induction heads, knock them out, and read the board out of an Othello-playing GPT.
"""
import jax  # noqa: F401  (you will need it)
import jax.numpy as jnp  # noqa: F401


def attention_entropy(weights):
    """Entropy (in nats) of each attention row.

    weights: [..., T_q, T_k], each row a probability distribution (rows sum to 1; masked keys have weight 0).
    Returns [..., T_q]. Treat 0·log 0 as 0. A head that always looks at one token has entropy ≈ 0; one that
    spreads evenly over n tokens has entropy log n.
    """
    raise NotImplementedError


def previous_token_score(attn):
    """How much a head attends to the immediately preceding token.

    attn: [..., T, T] causal attention weights (query rows, key columns). Return the mean of attn[..., i, i-1]
    over i = 1..T-1 (and over any leading dimensions) as a scalar.
    """
    raise NotImplementedError


def induction_score(attn, tokens):
    """Olsson et al. (2022)'s induction score for one head.

    attn: [B, T, T] attention weights; tokens: [B, T] the input tokens.
    For a query at position i, the "induction targets" are the key positions j with 1 ≤ j ≤ i whose
    *previous* token equals the current one: tokens[j-1] == tokens[i] and j-1 < i. An induction head puts its
    attention there: on the token that came *after* an earlier occurrence of the current token.

    Return the mean, over all (b, i) that have at least one induction target, of the total attention weight
    that query puts on its targets. A scalar in [0, 1].
    """
    raise NotImplementedError


def ablate_head(head_mask, layer: int, head: int):
    """Return a copy of head_mask ([n_layers, n_heads], 1.0 = on) with that head switched off (0.0).

    jax arrays are immutable: use .at[...].set(...).
    """
    raise NotImplementedError


def train_linear_probe(acts, labels, n_classes: int, steps: int = 300, lr: float = 1e-2):
    """Fit a linear probe: softmax regression from activations to labels, for several outputs at once.

    acts: [N, d] float activations. labels: [N, K] ints in [0, n_classes) — K separate classification problems
    (e.g. K = 64 board squares), all read linearly from the same activation vector.
    Model: logits[n, k, c] = acts[n] @ W[:, k, c] + b[k, c]. Train with mean cross-entropy over n and k, using
    full-batch Adam (optax.adam(lr)) for `steps` steps from W = 0, b = 0, inside one jitted lax.scan — and
    pass acts/labels into the jitted function as arguments (don't close over them; Module 16 pitfall).
    Return (W [d, K, n_classes], b [K, n_classes]).
    """
    raise NotImplementedError


def probe_accuracy(W, b, acts, labels):
    """Mean accuracy of the probe's argmax predictions over all N·K labels (a Python float)."""
    raise NotImplementedError


def to_relative_board(abs_board, player):
    """Convert an absolute board to the player-relative view.

    abs_board: [..., 64] ints, 0 empty, 1 black, 2 white. player: [...] ints, 1 if the viewpoint is black,
    2 if white (broadcast against abs_board's leading dims). Return [..., 64] ints: 0 empty, 1 mine,
    2 theirs.
    """
    raise NotImplementedError
