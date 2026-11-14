import jax.numpy as jnp
import numpy as np
import pytest


def test_broadcast_shape(impl):
    assert impl.broadcast_shape((8, 1, 3), (5, 1)) == (8, 5, 3)
    assert impl.broadcast_shape((3,), (4, 3)) == (4, 3)
    assert impl.broadcast_shape((), (2, 2)) == (2, 2)
    assert impl.broadcast_shape((1, 4), (3, 1)) == (3, 4)
    assert impl.broadcast_shape((2, 3), (2, 3)) == (2, 3)
    with pytest.raises(ValueError):
        impl.broadcast_shape((3,), (4,))
    with pytest.raises(ValueError):
        impl.broadcast_shape((2, 3), (3, 2))


def test_normalize_rows(impl):
    x = jnp.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    out = np.asarray(impl.normalize_rows(x))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0], [1.0, 0.0]], atol=1e-6)
    assert not np.isnan(out).any()
    r = np.asarray(impl.normalize_rows(jnp.arange(12.0).reshape(4, 3) + 1))
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), np.ones(4), rtol=1e-5)


def test_pairwise_sq_dists(impl):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 3)), rng.normal(size=(7, 3))
    expected = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    out = impl.pairwise_sq_dists(jnp.asarray(a), jnp.asarray(b))
    assert out.shape == (5, 7)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)
    same = impl.pairwise_sq_dists(jnp.asarray(a), jnp.asarray(a))
    np.testing.assert_allclose(np.diag(np.asarray(same)), 0.0, atol=1e-5)


def test_one_hot(impl):
    out = impl.one_hot(jnp.array([0, 2, 1, 2]), 3)
    assert out.shape == (4, 3)
    assert out.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(out), [[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
    assert impl.one_hot(jnp.array([4]), 5)[0, 4] == 1.0


def test_batched_matmul(impl):
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(4, 2, 3)), rng.normal(size=(4, 3, 5))
    out = impl.batched_matmul(jnp.asarray(a), jnp.asarray(b))
    assert out.shape == (4, 2, 5)
    np.testing.assert_allclose(np.asarray(out), a @ b, rtol=1e-4, atol=1e-4)


def test_moving_average(impl):
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(np.asarray(impl.moving_average(x, 2)), [1.5, 2.5, 3.5, 4.5], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(impl.moving_average(x, 5)), [3.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(impl.moving_average(x, 1)), np.asarray(x), rtol=1e-6)
    with pytest.raises(ValueError):
        impl.moving_average(x, 6)
    with pytest.raises(ValueError):
        impl.moving_average(x, 0)
