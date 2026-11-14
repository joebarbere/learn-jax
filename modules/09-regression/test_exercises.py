import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_train_val_test_split(impl):
    tr, va, te = impl.train_val_test_split(jax.random.key(0), 101)
    assert (len(tr), len(va), len(te)) == (70, 15, 16)
    allidx = np.concatenate([np.asarray(tr), np.asarray(va), np.asarray(te)])
    assert sorted(allidx.tolist()) == list(range(101))
    assert not np.array_equal(np.asarray(tr)[:10], np.arange(10))  # actually shuffled
    tr2, _, _ = impl.train_val_test_split(jax.random.key(0), 101)
    np.testing.assert_array_equal(tr, tr2)  # same key → same split
    tr3, va3, te3 = impl.train_val_test_split(jax.random.key(1), 10, (0.5, 0.5, 0.0))
    assert (len(tr3), len(va3), len(te3)) == (5, 5, 0)


def test_polynomial_features(impl):
    f = impl.polynomial_features(jnp.array([2.0, -1.0, 0.0]), 3)
    np.testing.assert_allclose(f, [[1, 2, 4, 8], [1, -1, 1, -1], [1, 0, 0, 0]])
    assert impl.polynomial_features(jnp.ones(5), 0).shape == (5, 1)


def test_ridge_closed_form(impl):
    k1, k2 = jax.random.split(jax.random.key(2))
    X = jax.random.normal(k1, (200, 4))
    w_true = jnp.array([1.0, -2.0, 0.5, 3.0])
    y = X @ w_true + 0.01 * jax.random.normal(k2, (200,))
    np.testing.assert_allclose(impl.ridge_closed_form(X, y, 0.0), w_true, atol=5e-3)
    w_big = impl.ridge_closed_form(X, y, 1e4)
    assert float(jnp.linalg.norm(w_big)) < 0.2 * float(jnp.linalg.norm(w_true))  # heavy shrinkage
    ref = np.linalg.solve(np.asarray(X.T @ X) + 3.0 * np.eye(4), np.asarray(X.T @ y))
    np.testing.assert_allclose(impl.ridge_closed_form(X, y, 3.0), ref, rtol=1e-3, atol=1e-4)


def test_bce_with_logits(impl):
    z = jnp.array([0.0, 2.0, -3.0])
    y = jnp.array([1.0, 0.0, 0.0])
    p = jax.nn.sigmoid(z)
    naive = -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))
    assert float(impl.bce_with_logits(z, y)) == pytest.approx(float(naive), rel=1e-5)
    extreme = impl.bce_with_logits(jnp.array([1000.0, -1000.0]), jnp.array([0.0, 1.0]))
    assert np.isfinite(float(extreme)) and float(extreme) == pytest.approx(1000.0, rel=1e-5)
    assert float(impl.bce_with_logits(jnp.array([50.0]), jnp.array([1.0]))) < 1e-6


def test_logistic_fit(impl):
    k1, k2 = jax.random.split(jax.random.key(3))
    a = jax.random.normal(k1, (100, 2)) + jnp.array([2.0, 2.0])
    b = jax.random.normal(k2, (100, 2)) - jnp.array([2.0, 2.0])
    X = jnp.concatenate([a, b])
    y = jnp.concatenate([jnp.ones(100), jnp.zeros(100)])
    w, bias = impl.logistic_fit(X, y, steps=300, lr=0.5)
    assert w.shape == (2,) and jnp.shape(bias) == ()
    pred = (X @ w + bias > 0).astype(jnp.float32)
    assert float(jnp.mean(pred == y)) > 0.97
    assert float(w[0]) > 0 and float(w[1]) > 0  # class 1 lives up and to the right


def test_accuracy(impl):
    assert impl.accuracy(jnp.array([1, 2, 3, 4]), jnp.array([1, 2, 0, 4])) == pytest.approx(0.75)
    assert isinstance(impl.accuracy(jnp.array([0]), jnp.array([0])), float)


def test_confusion_matrix(impl):
    yt = jnp.array([0, 0, 1, 2, 2, 2])
    yp = jnp.array([0, 1, 1, 2, 0, 2])
    m = impl.confusion_matrix(yt, yp, 3)
    np.testing.assert_array_equal(m, [[1, 1, 0], [0, 1, 0], [1, 0, 2]])
    assert jnp.issubdtype(m.dtype, jnp.integer)
    r = jax.random.randint(jax.random.key(4), (1000,), 0, 10)
    assert int(impl.confusion_matrix(r, r, 10).sum()) == 1000
