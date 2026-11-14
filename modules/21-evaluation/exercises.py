"""Module 21 exercises: the arithmetic of evaluating language models.

Replace each `raise NotImplementedError` with your code, then run `make test N=21`. Everything here is
NumPy / plain Python — the model calls live in the lesson; these functions turn model outputs into
numbers you can trust (or distrust).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def sliding_window_nll(token_logprob_fn: Callable[[np.ndarray], np.ndarray], ids: Sequence[int],
                       window: int, stride: int) -> tuple[float, int]:
    """Total negative log-likelihood of a long token sequence, scoring every token exactly once.

    `token_logprob_fn(chunk)` takes a 1-D int array of length L (2 <= L <= window) and returns L-1
    log-probabilities: entry i is log p(chunk[i+1] | chunk[:i+1]). The model only sees the chunk, so a
    token near the start of a chunk gets little context.

    Slide a window of length `window` along `ids` in steps of `stride` (1 <= stride <= window):
    window w covers ids[begin : min(begin + window, len(ids))] with begin = w * stride. Each window
    scores only tokens after the previous window's end (each token at most once), taking the *last*
    `new` entries of its log-probs, where new = (this window's end - previous end), capped at the
    number of log-probs the window returned. The first window scores all its tokens but the first.
    Stop once a window reaches the end of `ids`.

    With stride < window every token from ids[1] onward is scored. With stride == window (plain
    non-overlapping chunks) the first token of each later window has no context inside its window, so
    it can't be scored — n_scored comes out smaller. That's how naive chunked evaluation behaves.

    Returns (total_nll, n_scored): total_nll = -sum of the scored log-probs (nats, a float), n_scored =
    how many tokens were scored. If len(ids) < 2, return (0.0, 0). Compare runs by total_nll / n_scored.
    """
    raise NotImplementedError


def bits_per_byte(total_nll_nats: float, n_bytes: int) -> float:
    """Convert a total NLL in nats over some text into bits per UTF-8 byte of that text.

    Tokenizer-independent: two models with different vocabularies can be compared on the same text.
    1 nat = 1 / ln 2 bits. Raise ValueError if n_bytes <= 0.
    """
    raise NotImplementedError


def choice_logprob(token_logprobs: np.ndarray, choice_mask: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Score each candidate answer by the log-probability of its tokens.

    token_logprobs: [N, T] float — log p(token t | everything before it) for N padded sequences.
    choice_mask:    [N, T] bool  — True where token t belongs to the answer being scored.
    Returns [N]: the sum of the masked log-probs, or (normalize=True) their mean over the answer's
    tokens — "length normalization", so long answers aren't penalized just for having more tokens.
    A row with no answer tokens scores -inf (it can never win).
    """
    raise NotImplementedError


def mc_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Multiple-choice accuracy. scores: [n_items, n_choices] (higher = more likely); labels: [n_items]
    index of the correct choice. The prediction is the argmax (ties → the lowest index). Returns a float."""
    raise NotImplementedError


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Accuracy with a bootstrap (1 - alpha) confidence interval.

    correct: [n] bool/0-1 array of per-item results. Resample n items with replacement `n_boot` times
    (np.random.default_rng(seed)), compute each resample's mean, and return
    (mean, lower, upper) where lower/upper are the alpha/2 and 1 - alpha/2 quantiles of those means.
    """
    raise NotImplementedError


def paired_bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int = 2000,
                          seed: int = 0) -> tuple[float, float, float, float]:
    """Is model B really better than model A on the *same* items?

    Resample item indices with replacement (the same indices for both models — that's the "paired"
    part), compute mean(b) - mean(a) per resample. Return (diff, lower, upper, p) where diff is the
    observed mean(b) - mean(a), lower/upper the 2.5% / 97.5% quantiles of the resampled differences,
    and p the fraction of resampled differences <= 0 (small p → B's lead is unlikely to be luck).
    """
    raise NotImplementedError


def ngram_overlap(a_tokens: Sequence, b_tokens: Sequence, n: int) -> float:
    """Fraction of a's n-grams (contiguous length-n tuples) that also occur anywhere in b.

    The core of a contamination check: a = a benchmark item, b = training text. Count n-grams of `a`
    with multiplicity (a repeated n-gram counts each time). If `a` has no n-grams (len(a) < n), return 0.0.
    """
    raise NotImplementedError


def position_bias_rate(judgments_ab: Sequence[str], judgments_ba: Sequence[str]) -> float:
    """How often a judge picks a *position* rather than an *answer*.

    Each item was judged twice: with answers in order (X, Y) → judgments_ab[i] in {"A", "B"}, and
    swapped (Y, X) → judgments_ba[i]. A consistent judge prefers the same answer both times, so its
    verdict letter flips ("A" then "B", or "B" then "A"). Picking the same letter both times means the
    order decided. Return the fraction of items where judgments_ab[i] == judgments_ba[i].
    Raise ValueError if the lists differ in length or are empty.
    """
    raise NotImplementedError
