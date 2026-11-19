import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(42)


def _rand(i, shape, scale=1.0):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32) * scale


# ---------------------------------------------------------------- training_cost
def test_training_cost(impl):
    c = impl.training_cost(135e6, 2e12, 100.0, mfu=0.5)
    assert c["flops"] == pytest.approx(6 * 135e6 * 2e12)
    assert c["hours"] == pytest.approx(6 * 135e6 * 2e12 / (100e12 * 0.5) / 3600)
    assert c["tokens_per_param"] == pytest.approx(2e12 / 135e6)
    assert c["chinchilla_tokens"] == pytest.approx(20 * 135e6)


def test_training_cost_default_mfu(impl):
    assert impl.training_cost(1e9, 1e9, 1.0)["hours"] == pytest.approx(6e18 / (1e12 * 0.4) / 3600)


# ---------------------------------------------------------------- kd_loss
def _teacher(i, shape, k):
    lp = jax.nn.log_softmax(_rand(i, shape, 3.0))
    vals, ids = jax.lax.top_k(lp, k)
    return lp, ids, vals


def test_kd_loss_full_k_is_kl(impl):
    lp, ids, vals = _teacher(1, (2, 3, 40), 40)
    s = _rand(2, (2, 3, 40))
    kl = jnp.mean(jnp.sum(jnp.exp(lp) * (lp - jax.nn.log_softmax(s)), -1))
    assert float(impl.kd_loss(s, ids, vals)) == pytest.approx(float(kl), rel=1e-4)


def test_kd_loss_zero_at_teacher(impl):
    lp, ids, vals = _teacher(3, (4, 30), 30)
    assert float(impl.kd_loss(lp, ids, vals)) == pytest.approx(0.0, abs=1e-5)


def test_kd_loss_temperature(impl):
    lp, ids, vals = _teacher(4, (6, 20), 20)
    s = _rand(5, (6, 20))
    T = 2.0
    pt, ls = jax.nn.softmax(lp / T), jax.nn.log_softmax(s / T)
    expect = T * T * jnp.mean(jnp.sum(pt * (jnp.log(pt) - ls), -1))
    assert float(impl.kd_loss(s, ids, vals, T)) == pytest.approx(float(expect), rel=1e-4)


def test_kd_loss_top_k(impl):
    _lp, ids, vals = _teacher(6, (5, 50), 8)
    s = _rand(7, (5, 50))
    p = jax.nn.softmax(vals, -1)
    ls = jnp.take_along_axis(jax.nn.log_softmax(s), ids, -1)
    expect = jnp.mean(jnp.sum(p * (jnp.log(p) - ls), -1))
    assert float(impl.kd_loss(s, ids, vals)) == pytest.approx(float(expect), rel=1e-4)


def test_kd_loss_grad_pulls_toward_teacher(impl):
    lp, ids, vals = _teacher(8, (3, 25), 25)
    s = jnp.zeros((3, 25))
    g = jax.grad(lambda x: impl.kd_loss(x, ids, vals))(s)
    _close = np.allclose(np.asarray(g), np.asarray((jax.nn.softmax(s) - jnp.exp(lp)) / 3), atol=1e-5)
    assert _close


# ---------------------------------------------------------------- pruning
def test_ffn_importance(impl):
    a = _rand(9, (2, 5, 7))
    np.testing.assert_allclose(impl.ffn_importance(a), np.abs(np.asarray(a)).reshape(-1, 7).mean(0), rtol=1e-5)
    assert impl.ffn_importance(a).shape == (7,)


def test_prune_ffn_matches_masking(impl):
    L, D, F = 3, 5, 8
    layers = {"w_gate": _rand(10, (L, D, F)), "w_up": _rand(11, (L, D, F)), "w_down": _rand(12, (L, F, D)),
              "attn_norm": jnp.ones((L, D))}
    keep = jnp.array([[0, 3, 7], [6, 1, 2], [4, 5, 0]])
    out = impl.prune_ffn(layers, keep)
    assert out["w_gate"].shape == (L, D, 3) and out["w_up"].shape == (L, D, 3) and out["w_down"].shape == (L, 3, D)
    assert out["attn_norm"] is layers["attn_norm"]
    x = _rand(13, (4, D))
    for i in range(L):
        mask = jnp.zeros(F).at[keep[i]].set(1.0)
        ref = (jax.nn.silu(x @ layers["w_gate"][i]) * (x @ layers["w_up"][i]) * mask) @ layers["w_down"][i]
        got = (jax.nn.silu(x @ out["w_gate"][i]) * (x @ out["w_up"][i])) @ out["w_down"][i]
        np.testing.assert_allclose(got, ref, atol=1e-5)


def test_layer_importance(impl):
    h = _rand(14, (2, 6, 16))
    assert float(impl.layer_importance(h, h)) == pytest.approx(0.0, abs=1e-6)
    assert float(impl.layer_importance(h, 3 * h)) == pytest.approx(0.0, abs=1e-6)     # scale doesn't count
    assert float(impl.layer_importance(h, -h)) == pytest.approx(2.0, abs=1e-6)
    o = _rand(15, (2, 6, 16))
    a, b = np.asarray(h).reshape(-1, 16), np.asarray(o).reshape(-1, 16)
    cos = (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    assert float(impl.layer_importance(h, o)) == pytest.approx(1 - cos.mean(), rel=1e-4)


# ---------------------------------------------------------------- Muon
@pytest.mark.parametrize("shape", [(32, 32), (64, 16), (16, 48)])
def test_newton_schulz(impl, shape):
    G = _rand(16, shape)
    O = impl.newton_schulz(G)
    assert O.shape == shape
    U, _s, Vt = np.linalg.svd(np.asarray(G, np.float64), full_matrices=False)
    so = np.linalg.svd(np.asarray(O, np.float64), compute_uv=False)
    assert 0.5 < so.min() and so.max() < 1.3                              # all pushed toward 1
    # same singular vectors: O ≈ U diag(so') Vᵀ, so Uᵀ O V is (nearly) diagonal
    M = U.T @ np.asarray(O, np.float64) @ Vt.T
    assert np.abs(M - np.diag(np.diag(M))).max() < 1e-3


def test_newton_schulz_matches_reference(impl):
    G = _rand(17, (24, 40))
    a, b, c = 3.4445, -4.7750, 2.0315
    X = np.asarray(G, np.float64) / np.linalg.norm(np.asarray(G, np.float64))
    for _ in range(5):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    np.testing.assert_allclose(impl.newton_schulz(G), X, atol=1e-4)
