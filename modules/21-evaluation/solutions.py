"""Module 21 reference solutions. Try the exercises first — `make test N=21` grades yours."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np


def sliding_window_nll(token_logprob_fn: Callable[[np.ndarray], np.ndarray], ids: Sequence[int],
                       window: int, stride: int) -> tuple[float, int]:
    ids = np.asarray(ids)
    n = len(ids)
    if n < 2:
        return 0.0, 0
    total, scored, prev_end, begin = 0.0, 0, 1, 0  # ids[0] is never scored (no context)
    while True:
        end = min(begin + window, n)
        lp = np.asarray(token_logprob_fn(ids[begin:end]), dtype=np.float64)  # predicts ids[begin+1 : end]
        new = min(end - prev_end, len(lp))       # tokens this window hasn't scored (and can score)
        if new > 0:
            total -= float(lp[-new:].sum())
            scored += new
        prev_end = end
        if end == n:
            return total, scored
        begin += stride


def bits_per_byte(total_nll_nats: float, n_bytes: int) -> float:
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")
    return total_nll_nats / (n_bytes * math.log(2))


def choice_logprob(token_logprobs: np.ndarray, choice_mask: np.ndarray, normalize: bool = False) -> np.ndarray:
    lp = np.asarray(token_logprobs, dtype=np.float64)
    m = np.asarray(choice_mask, dtype=bool)
    total = np.where(m, lp, 0.0).sum(-1)
    count = m.sum(-1)
    if normalize:
        total = total / np.maximum(count, 1)
    return np.where(count > 0, total, -np.inf)


def mc_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(np.asarray(scores), axis=-1) == np.asarray(labels)))


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    c = np.asarray(correct, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = c[rng.integers(0, len(c), size=(n_boot, len(c)))].mean(-1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(c.mean()), float(lo), float(hi)


def paired_bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int = 2000,
                          seed: int = 0) -> tuple[float, float, float, float]:
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = b[idx].mean(-1) - a[idx].mean(-1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(b.mean() - a.mean()), float(lo), float(hi), float(np.mean(diffs <= 0))


def ngram_overlap(a_tokens: Sequence, b_tokens: Sequence, n: int) -> float:
    a, b = list(a_tokens), list(b_tokens)
    a_grams = [tuple(a[i:i + n]) for i in range(len(a) - n + 1)]
    if not a_grams:
        return 0.0
    b_grams = Counter(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
    return sum(g in b_grams for g in a_grams) / len(a_grams)


def position_bias_rate(judgments_ab: Sequence[str], judgments_ba: Sequence[str]) -> float:
    if len(judgments_ab) != len(judgments_ba) or not judgments_ab:
        raise ValueError("need two equal-length, non-empty lists of judgments")
    return sum(x == y for x, y in zip(judgments_ab, judgments_ba, strict=True)) / len(judgments_ab)
