import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(48)


def _rand(i, shape, scale=1.0):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32) * scale


# ---------------------------------------------------------------- logit_diff
def test_logit_diff_last_position(impl):
    lg = _rand(0, (4, 5, 9))
    io, s = jnp.array([1, 2, 3, 4]), jnp.array([0, 0, 8, 7])
    want = np.mean([float(lg[b, -1, io[b]] - lg[b, -1, s[b]]) for b in range(4)])
    got = impl.logit_diff(lg, io, s)
    assert float(got) == pytest.approx(want, rel=1e-5)
    assert jnp.asarray(got).shape == ()


def test_logit_diff_bf16_and_sign(impl):
    lg = jnp.zeros((2, 3, 5)).at[:, -1, 2].set(4.0).at[:, 0, 1].set(100.0).astype(jnp.bfloat16)
    assert float(impl.logit_diff(lg, jnp.array([2, 2]), jnp.array([1, 1]))) == pytest.approx(4.0)
    assert float(impl.logit_diff(lg, jnp.array([1, 1]), jnp.array([2, 2]))) == pytest.approx(-4.0)
    assert jnp.asarray(impl.logit_diff(lg, jnp.array([2, 2]), jnp.array([1, 1]))).dtype == jnp.float32


# ---------------------------------------------------------------- patching_effect
def test_patching_effect_endpoints_and_grid(impl):
    assert float(impl.patching_effect(4.0, 4.0, -2.0)) == pytest.approx(1.0)
    assert float(impl.patching_effect(-2.0, 4.0, -2.0)) == pytest.approx(0.0)
    grid = jnp.array([[1.0, 7.0], [-5.0, 4.0]])
    np.testing.assert_allclose(impl.patching_effect(grid, 4.0, -2.0), (np.asarray(grid) + 2) / 6, rtol=1e-6)


# ---------------------------------------------------------------- attribution_patch
def test_attribution_patch_exact_for_linear_metric(impl):
    w = _rand(1, (6,))
    clean, corrupt = _rand(2, (3, 4, 6)), _rand(3, (3, 4, 6))
    f = lambda a: (a * w).sum()
    g = jax.grad(f)(corrupt)
    est = impl.attribution_patch(g, clean, corrupt)
    assert est.shape == (3, 4)
    # for a linear metric the estimate of patching one (row, col) is exact
    true = np.array([[float(f(corrupt.at[i, j].set(clean[i, j])) - f(corrupt)) for j in range(4)] for i in range(3)])
    np.testing.assert_allclose(est, true, rtol=1e-4, atol=1e-5)


def test_attribution_patch_zero_when_equal(impl):
    a = _rand(4, (5, 3))
    np.testing.assert_allclose(impl.attribution_patch(_rand(5, (5, 3)), a, a), np.zeros(5), atol=1e-7)


# ---------------------------------------------------------------- edge_weights
def test_edge_weights_scale_columns(impl):
    J = _rand(6, (4, 3))
    v = jnp.array([2.0, 0.0, -1.0])
    out = np.asarray(impl.edge_weights(J, v))
    np.testing.assert_allclose(out, np.asarray(J) * np.array([2.0, 0.0, -1.0]), rtol=1e-6)
    assert np.all(out[:, 1] == 0)  # an inactive source has no edges


def test_edges_sum_to_target_for_linear_model(impl):
    M = _rand(7, (5, 8))
    x = _rand(8, (8,))
    E = np.asarray(impl.edge_weights(M, x))  # target = M @ x, jacobian = M
    np.testing.assert_allclose(E.sum(1), np.asarray(M @ x), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- normalize_rows
def test_normalize_rows(impl):
    A = jnp.array([[0.0, 0.0, 0.0], [2.0, -2.0, 0.0], [1.0, 1.0, -2.0]])
    out = np.asarray(impl.normalize_rows(A))
    np.testing.assert_allclose(out, [[0, 0, 0], [0.5, 0.5, 0], [0.25, 0.25, 0.5]], atol=1e-7)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------- total_influence
def test_total_influence_chain(impl):
    # 3 → 2 → 1 → 0 (row = target): every node's whole input comes from the next one
    A = jnp.zeros((4, 4)).at[0, 1].set(1.0).at[1, 2].set(1.0).at[2, 3].set(1.0)
    w = jnp.array([1.0, 0, 0, 0])
    np.testing.assert_allclose(impl.total_influence(A, w), [0, 1, 1, 1], atol=1e-6)


def test_total_influence_closed_form(impl):
    rng = np.random.default_rng(0)
    A = np.triu(rng.random((7, 7)), 1)
    An = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    w = rng.random(7)
    want = w @ An @ np.linalg.inv(np.eye(7) - An)
    np.testing.assert_allclose(impl.total_influence(jnp.asarray(An, jnp.float32), jnp.asarray(w, jnp.float32)), want,
                               rtol=1e-4, atol=1e-5)
