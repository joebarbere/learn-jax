import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(47)


def _rand(i, shape, scale=1.0):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32) * scale


# ---------------------------------------------------------------- logit_lens
def test_logit_lens_matches_formula(impl):
    h, w, W = _rand(0, (3, 4, 8), 5.0), _rand(1, (8,)), _rand(2, (8, 11))
    out = np.asarray(impl.logit_lens(h, w, W))
    hn = np.asarray(h) / np.sqrt((np.asarray(h) ** 2).mean(-1, keepdims=True) + 1e-5) * np.asarray(w)
    assert out.shape == (3, 4, 11) and out.dtype == np.float32
    np.testing.assert_allclose(out, hn @ np.asarray(W), rtol=1e-4, atol=1e-4)


def test_logit_lens_scale_invariant_and_bf16(impl):
    h, w, W = _rand(3, (5, 8)), _rand(4, (8,)), _rand(5, (8, 6))
    a = np.asarray(impl.logit_lens(h, w, W))
    np.testing.assert_allclose(np.asarray(impl.logit_lens(100.0 * h, w, W)), a, rtol=1e-3, atol=1e-3)
    b = impl.logit_lens(h.astype(jnp.bfloat16), w, W)
    assert np.asarray(b).dtype == np.float32


# ---------------------------------------------------------------- topk_activation
def test_topk_keeps_k_largest(impl):
    pre = jnp.array([[0.5, -1.0, 3.0, 2.0, 0.1], [1.0, 2.0, 3.0, 4.0, 5.0]])
    z = np.asarray(impl.topk_activation(pre, 2))
    np.testing.assert_allclose(z, [[0, 0, 3.0, 2.0, 0], [0, 0, 0, 4.0, 5.0]])


def test_topk_relu_on_kept_negatives(impl):
    pre = jnp.array([[-0.5, -2.0, -3.0, 1.0]])
    np.testing.assert_allclose(np.asarray(impl.topk_activation(pre, 2)), [[0, 0, 0, 1.0]])


def test_topk_k_equals_width_is_relu(impl):
    pre = _rand(6, (4, 7))
    np.testing.assert_allclose(np.asarray(impl.topk_activation(pre, 7)), np.maximum(np.asarray(pre), 0), atol=1e-6)


# ---------------------------------------------------------------- fvu
def test_fvu_values(impl):
    x = _rand(7, (50, 6), 3.0) + 10.0
    assert float(impl.fvu(x, x)) == pytest.approx(0.0, abs=1e-6)
    assert float(impl.fvu(x, jnp.broadcast_to(x.mean(0), x.shape))) == pytest.approx(1.0, rel=1e-5)
    noisy = x + _rand(8, x.shape, 0.3)
    expect = float(((x - noisy) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
    assert float(impl.fvu(x, noisy)) == pytest.approx(expect, rel=1e-4)


# ---------------------------------------------------------------- mean_difference
def test_mean_difference(impl):
    pos = _rand(9, (40, 5)) + jnp.array([3.0, 0, 0, 4.0, 0])
    neg = _rand(10, (30, 5))
    u = np.asarray(impl.mean_difference(pos, neg))
    d = np.asarray(pos).mean(0) - np.asarray(neg).mean(0)
    np.testing.assert_allclose(u, d / np.linalg.norm(d), rtol=1e-5, atol=1e-6)
    assert np.linalg.norm(u) == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------- ablate_direction
def test_ablate_direction_removes_and_reflects(impl):
    W, x = _rand(11, (7, 5)), _rand(12, (9, 7))
    u = _rand(13, (5,))
    u = u / jnp.linalg.norm(u)
    W1 = impl.ablate_direction(W, u, 1.0)
    assert W1.shape == W.shape
    np.testing.assert_allclose(np.asarray(x @ W1 @ u), 0.0, atol=1e-5)
    W2 = impl.ablate_direction(W, u, 2.0)
    np.testing.assert_allclose(np.asarray(x @ W2 @ u), -np.asarray(x @ W @ u), rtol=1e-4, atol=1e-5)
    perp = jnp.array([u[1], -u[0], 0.0, 0.0, 0.0])          # orthogonal to u: untouched
    np.testing.assert_allclose(np.asarray(x @ W1 @ perp), np.asarray(x @ W @ perp), rtol=1e-4, atol=1e-5)


def test_ablate_direction_alpha_zero(impl):
    W, u = _rand(14, (3, 4)), jnp.array([0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(impl.ablate_direction(W, u, 0.0)), np.asarray(W))


# ---------------------------------------------------------------- mutual_knn
def test_mutual_knn_same_and_rotated(impl):
    X = _rand(15, (60, 8))
    Q = jnp.linalg.qr(_rand(16, (8, 8)))[0]
    assert impl.mutual_knn(X, X, 5) == pytest.approx(1.0)
    assert impl.mutual_knn(X, 3.0 * X @ Q, 5) == pytest.approx(1.0)       # cosine kNN ignores scale and rotation


def test_mutual_knn_unrelated_is_low(impl):
    X, Y = _rand(17, (200, 16)), _rand(18, (200, 16))
    assert impl.mutual_knn(X, Y, 10) < 0.2                                   # chance is k/(N−1) = 0.05


def test_mutual_knn_hand_example(impl):
    # unit vectors at angles 0°, 10°, 30°, 60° in X; in Y the last two swap angles. Nearest neighbours (k = 1):
    # X: 0→1, 1→0, 2→1, 3→2.  Y: 0→1, 1→0, 2→3, 3→1.  Two of four agree.
    def at(degrees):
        r = np.deg2rad(degrees)
        return jnp.asarray(np.stack([np.cos(r), np.sin(r)], 1), jnp.float32)
    assert impl.mutual_knn(at([0, 10, 30, 60]), at([0, 10, 60, 30]), 1) == pytest.approx(0.5)
