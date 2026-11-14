import math

import numpy as np
import pytest


def _context_logprob_fn(calls):
    """Fake model: log p(next) = -1 / (1 + context length). Records every chunk it sees."""

    def fn(chunk):
        calls.append(np.asarray(chunk).copy())
        return np.array([-1.0 / (1 + i + 1) for i in range(len(chunk) - 1)])

    return fn


def test_sliding_window_scores_each_token_once(impl):
    ids = np.arange(100)
    calls = []
    total, n = impl.sliding_window_nll(_context_logprob_fn(calls), ids, window=10, stride=5)
    assert n == 99                                   # overlapping windows: every token from ids[1] on
    assert all(len(c) <= 10 for c in calls)
    # Non-overlapping chunks: each window restarts context and its first token can't be scored.
    total, n = impl.sliding_window_nll(_context_logprob_fn([]), ids, window=10, stride=10)
    assert n == 99 - 9
    assert total == pytest.approx(10 * sum(1.0 / (i + 2) for i in range(9)))


def test_sliding_window_smaller_stride_more_context(impl):
    ids = np.arange(200)
    big, n1 = impl.sliding_window_nll(_context_logprob_fn([]), ids, window=32, stride=32)
    small, n2 = impl.sliding_window_nll(_context_logprob_fn([]), ids, window=32, stride=8)
    assert n2 == 199 and n1 < n2
    assert small / n2 < big / n1  # more context per scored token → lower mean NLL with this fake model


def test_sliding_window_edge_cases(impl):
    assert impl.sliding_window_nll(_context_logprob_fn([]), [7], window=8, stride=4) == (0.0, 0)
    total, n = impl.sliding_window_nll(_context_logprob_fn([]), [1, 2, 3], window=8, stride=4)
    assert n == 2 and total == pytest.approx(1 / 2 + 1 / 3)
    # stride == 1 and a window longer than the text both still score every token once
    _, n = impl.sliding_window_nll(_context_logprob_fn([]), np.arange(20), window=5, stride=1)
    assert n == 19


def test_bits_per_byte(impl):
    assert impl.bits_per_byte(math.log(2) * 100, 100) == pytest.approx(1.0)
    assert impl.bits_per_byte(0.0, 5) == 0.0
    with pytest.raises(ValueError):
        impl.bits_per_byte(1.0, 0)


def test_choice_logprob(impl):
    lp = np.array([[-1.0, -2.0, -3.0, -4.0], [-0.5, -0.5, -9.0, -9.0], [-1.0, -1.0, -1.0, -1.0]])
    mask = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
    np.testing.assert_allclose(impl.choice_logprob(lp, mask)[:2], [-5.0, -1.0])
    np.testing.assert_allclose(impl.choice_logprob(lp, mask, normalize=True)[:2], [-2.5, -0.5])
    assert impl.choice_logprob(lp, mask)[2] == -np.inf
    assert impl.choice_logprob(lp, mask, normalize=True)[2] == -np.inf


def test_mc_accuracy(impl):
    scores = np.array([[0.1, 0.9, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    assert impl.mc_accuracy(scores, np.array([1, 0, 0])) == pytest.approx(1.0)
    assert impl.mc_accuracy(scores, np.array([1, 0, 1])) == pytest.approx(2 / 3)  # tie → index 0


def test_bootstrap_ci(impl):
    correct = np.array([1] * 70 + [0] * 30, dtype=bool)
    mean, lo, hi = impl.bootstrap_ci(correct, n_boot=4000, seed=1)
    assert mean == pytest.approx(0.7)
    assert lo < 0.7 < hi
    assert 0.58 < lo < 0.65 and 0.75 < hi < 0.82  # ~±0.09 for n = 100
    assert impl.bootstrap_ci(np.ones(10, bool))[1:] == (1.0, 1.0)
    assert impl.bootstrap_ci(correct, seed=3) == impl.bootstrap_ci(correct, seed=3)


def test_paired_bootstrap_diff(impl):
    rng = np.random.default_rng(0)
    a = rng.random(400) < 0.5
    b = a.copy()
    b[:60] = True  # B fixes some of A's mistakes and never does worse
    diff, lo, hi, p = impl.paired_bootstrap_diff(a, b, n_boot=2000, seed=0)
    assert diff == pytest.approx(b.mean() - a.mean())
    assert 0 < lo < diff < hi
    assert p < 0.01
    diff, lo, hi, p = impl.paired_bootstrap_diff(a, a, n_boot=500)
    assert diff == 0 and lo == 0 and hi == 0 and p == 1.0


def test_ngram_overlap(impl):
    a = ["the", "cat", "sat", "on", "the", "mat"]
    b = ["yesterday", "the", "cat", "sat", "on", "a", "chair"]
    assert impl.ngram_overlap(a, b, 2) == pytest.approx(3 / 5)   # "the cat", "cat sat", "sat on"
    assert impl.ngram_overlap(a, a, 3) == 1.0
    assert impl.ngram_overlap(a, b, 7) == 0.0                     # a has no 7-grams
    assert impl.ngram_overlap([1, 2, 1, 2], [9, 1, 2], 2) == pytest.approx(2 / 3)  # multiplicity


def test_position_bias_rate(impl):
    assert impl.position_bias_rate(["A", "B", "A", "A"], ["B", "A", "A", "B"]) == pytest.approx(0.25)
    assert impl.position_bias_rate(["A"] * 5, ["A"] * 5) == 1.0
    with pytest.raises(ValueError):
        impl.position_bias_rate(["A"], ["A", "B"])
    with pytest.raises(ValueError):
        impl.position_bias_rate([], [])
