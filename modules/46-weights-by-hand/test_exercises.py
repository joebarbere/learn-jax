import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(46)


def _rand(i, shape, scale=1.0):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32) * scale


# ---------------------------------------------------------------- ridge_filter
def test_ridge_filter_normalized(impl):
    for theta in (0.0, 0.4, np.pi / 2, 2.0):
        f = np.asarray(impl.ridge_filter(theta))
        assert f.shape == (5, 5)
        assert f.sum() == pytest.approx(0.0, abs=1e-5)
        assert np.abs(f).sum() == pytest.approx(1.0, rel=1e-5)


def test_ridge_filter_orientation(impl):
    h, v = np.asarray(impl.ridge_filter(0.0)), np.asarray(impl.ridge_filter(np.pi / 2))
    line_h = np.zeros((5, 5))
    line_h[2] = 1.0                                   # a horizontal stroke through the middle row
    assert (h * line_h).sum() > 5 * abs((v * line_h).sum())   # the matching orientation responds far more
    np.testing.assert_allclose(v, h.T, atol=1e-6)     # a vertical stroke detector is the transpose
    assert h[2, 2] == h.max()


def test_ridge_filter_size(impl):
    assert np.asarray(impl.ridge_filter(1.0, size=7, width=1.5)).shape == (7, 7)


# ---------------------------------------------------------------- firing_rates
def test_firing_rates(impl):
    fires = np.array([[1, 0], [1, 1], [0, 0], [1, 0]], bool)
    labels = np.array([0, 0, 1, 1])
    r = np.asarray(impl.firing_rates(fires, labels, 3))
    assert r.shape == (3, 2)
    np.testing.assert_allclose(r[0], [(2 + 1) / 4, (1 + 1) / 4], rtol=1e-6)
    np.testing.assert_allclose(r[1], [(1 + 1) / 4, (0 + 1) / 4], rtol=1e-6)
    np.testing.assert_allclose(r[2], [0.5, 0.5], rtol=1e-6)      # empty class: smoothing alone


def test_firing_rates_alpha(impl):
    r = np.asarray(impl.firing_rates(np.ones((4, 1), bool), np.zeros(4, int), 1, alpha=0.0))
    assert r[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------- log_odds_readout
def test_log_odds_readout_is_naive_bayes(impl):
    rng = np.random.default_rng(0)
    rates = rng.uniform(0.05, 0.95, (4, 6))
    prior = rng.dirichlet(np.ones(4))
    W, b = map(np.asarray, impl.log_odds_readout(rates, prior))
    assert W.shape == (6, 4) and b.shape == (4,)
    for x in rng.integers(0, 2, (5, 6)):
        direct = (x * np.log(rates) + (1 - x) * np.log1p(-rates)).sum(1) + np.log(prior)
        np.testing.assert_allclose(x @ W + b, direct, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------- position_selector
def test_position_selector_reverse(impl):
    L = 6
    target = np.arange(L)[::-1]
    Wq, Wk = impl.position_selector(target)
    P = jnp.eye(L)
    attn = jax.nn.softmax((P @ Wq) @ (P @ Wk).T, -1)
    np.testing.assert_array_equal(np.asarray(attn).argmax(1), target)
    assert float(attn.max(1).min()) > 0.999


def test_position_selector_scores(impl):
    target = np.array([0, 0, 1, 2])                        # previous token (position 0 reads itself)
    Wq, Wk = impl.position_selector(target, scale=7.0)
    S = np.asarray((jnp.eye(4) @ Wq) @ (jnp.eye(4) @ Wk).T)
    expect = np.zeros((4, 4))
    expect[np.arange(4), target] = 7.0
    np.testing.assert_allclose(S, expect, atol=1e-6)


# ---------------------------------------------------------------- ties_merge
def test_ties_merge_hand(impl):
    taus = np.array([[3.0, -1.0, 0.5, 2.0],
                     [1.0, 2.0, -0.2, -4.0]])
    # density 0.5 keeps 2 per row: row 0 -> [3, 0, 0, 2], row 1 -> [0, 2, 0, -4]
    # signs of sums: [+, +, 0, -]; disjoint means: [3, 2, 0, -4]
    np.testing.assert_allclose(np.asarray(impl.ties_merge(taus, 0.5, 1.0)), [3.0, 2.0, 0.0, -4.0])
    np.testing.assert_allclose(np.asarray(impl.ties_merge(taus, 0.5, 0.5)), [1.5, 1.0, 0.0, -2.0])


def test_ties_merge_single_task(impl):
    tau = np.asarray(_rand(1, (1, 50)))
    np.testing.assert_allclose(np.asarray(impl.ties_merge(tau, 1.0, 0.7)), 0.7 * tau[0], rtol=1e-6)


def test_ties_merge_conflict(impl):
    taus = np.array([[1.0, 5.0], [-3.0, 5.0]])            # column 0: sum −2 wins, only row 1 agrees
    np.testing.assert_allclose(np.asarray(impl.ties_merge(taus, 1.0)), [-3.0, 5.0])


# ---------------------------------------------------------------- rank_one_edit
def test_rank_one_edit_writes_value(impl):
    W, k, v = _rand(2, (8, 5)), _rand(3, (8,)), _rand(4, (5,))
    A = _rand(5, (8, 8))
    C_inv = jnp.linalg.inv(A @ A.T / 8 + jnp.eye(8))
    W2 = impl.rank_one_edit(W, k, v, C_inv)
    np.testing.assert_allclose(np.asarray(k @ W2), np.asarray(v), atol=1e-4)
    assert int(jnp.linalg.matrix_rank(W2 - W)) == 1


def test_rank_one_edit_leaves_others(impl):
    W, k, v = _rand(6, (6, 3)), _rand(7, (6,)), _rand(8, (3,))
    C_inv = jnp.eye(6)
    other = _rand(9, (6,))
    other = other - (other @ k) / (k @ k) * k                # orthogonal to k (C = I)
    W2 = impl.rank_one_edit(W, k, v, C_inv)
    np.testing.assert_allclose(np.asarray(other @ W2), np.asarray(other @ W), atol=1e-4)
