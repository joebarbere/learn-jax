import itertools
import math

import jax.numpy as jnp
import numpy as np


def test_cosine_similarity(impl):
    a = jnp.array([1.0, 0.0])
    np.testing.assert_allclose(float(impl.cosine_similarity(a, jnp.array([2.0, 0.0]))), 1.0, rtol=1e-6)
    np.testing.assert_allclose(float(impl.cosine_similarity(a, jnp.array([0.0, 3.0]))), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(impl.cosine_similarity(a, jnp.array([-1.0, 0.0]))), -1.0, atol=1e-6)
    np.testing.assert_allclose(float(impl.cosine_similarity(jnp.array([1.0, 1.0]), a)), 1 / math.sqrt(2),
                               rtol=1e-5)
    batch = impl.cosine_similarity(jnp.ones((4, 3)), jnp.ones((4, 3)) * 5)
    assert batch.shape == (4,)
    np.testing.assert_allclose(np.asarray(batch), 1.0, rtol=1e-5)


def test_rotation_matrix(impl):
    r = np.asarray(impl.rotation_matrix(math.pi / 2))
    assert r.shape == (2, 2)
    np.testing.assert_allclose(r @ [1.0, 0.0], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(r @ [0.0, 1.0], [-1.0, 0.0], atol=1e-6)
    r30 = np.asarray(impl.rotation_matrix(0.5))
    np.testing.assert_allclose(r30.T @ r30, np.eye(2), atol=1e-6)  # rotations preserve lengths
    np.testing.assert_allclose(np.linalg.det(r30), 1.0, atol=1e-6)


def test_transform_points(impl):
    m = jnp.array([[2.0, 0.0], [1.0, 3.0]])
    pts = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = impl.transform_points(m, pts)
    assert out.shape == (3, 2)
    expected = np.array([np.asarray(m) @ np.asarray(p) for p in pts])
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-6)


def test_project(impl):
    p = impl.project(jnp.array([3.0, 4.0]), jnp.array([1.0, 0.0]))
    np.testing.assert_allclose(np.asarray(p), [3.0, 0.0], atol=1e-6)
    v, u = jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, -1.0, 1.0])
    p = impl.project(v, u)
    np.testing.assert_allclose(float(jnp.dot(v - p, u)), 0.0, atol=1e-5)  # residual is perpendicular
    np.testing.assert_allclose(np.asarray(impl.project(u * 2, u)), np.asarray(u * 2), atol=1e-6)


def test_low_rank_approx(impl):
    rng = np.random.default_rng(0)
    rank2 = rng.normal(size=(8, 2)) @ rng.normal(size=(2, 6))
    m = jnp.asarray(rank2, jnp.float32)
    np.testing.assert_allclose(np.asarray(impl.low_rank_approx(m, 2)), rank2, atol=1e-4)
    full = jnp.asarray(rng.normal(size=(8, 6)), jnp.float32)
    a1 = np.asarray(impl.low_rank_approx(full, 1))
    assert a1.shape == (8, 6)
    assert np.linalg.matrix_rank(a1, tol=1e-4) == 1
    errs = [np.linalg.norm(np.asarray(full) - np.asarray(impl.low_rank_approx(full, k))) for k in range(1, 7)]
    assert all(e1 >= e2 - 1e-5 for e1, e2 in itertools.pairwise(errs))  # more rank, never worse
    assert errs[-1] < 1e-4


def test_gram_schmidt(impl):
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.normal(size=(5, 3)), jnp.float32)
    q = np.asarray(impl.gram_schmidt(a))
    assert q.shape == (5, 3)
    np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-5)
    first = np.asarray(a[:, 0]) / np.linalg.norm(np.asarray(a[:, 0]))
    np.testing.assert_allclose(q[:, 0], first, atol=1e-5)
    r = q.T @ np.asarray(a)  # a = q r with r upper-triangular
    np.testing.assert_allclose(np.tril(r, -1), 0.0, atol=1e-4)
