import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_init_scale(impl):
    assert impl.init_scale(512, 256, "he") == pytest.approx(math.sqrt(2 / 512))
    assert impl.init_scale(512, 256, "xavier") == pytest.approx(math.sqrt(2 / 768))
    with pytest.raises(ValueError):
        impl.init_scale(10, 10, "lecun-ish")


def test_layer_norm(impl):
    x = jax.random.normal(jax.random.key(0), (4, 7, 16)) * 3 + 5
    out = impl.layer_norm(x, jnp.ones(16), jnp.zeros(16))
    np.testing.assert_allclose(out.mean(-1), 0, atol=1e-5)
    np.testing.assert_allclose(out.std(-1), 1, atol=1e-3)
    gamma, beta = jnp.arange(16.0), jnp.full(16, 0.5)
    ref = (x - x.mean(-1, keepdims=True)) / jnp.sqrt(x.var(-1, keepdims=True) + 1e-5) * gamma + beta
    np.testing.assert_allclose(impl.layer_norm(x, gamma, beta), ref, rtol=1e-5, atol=1e-5)
    # a constant row has zero variance: eps keeps it finite
    assert jnp.all(jnp.isfinite(impl.layer_norm(jnp.ones((2, 8)), jnp.ones(8), jnp.zeros(8))))


def test_rms_norm(impl):
    x = jax.random.normal(jax.random.key(1), (3, 32)) * 4
    out = impl.rms_norm(x, jnp.ones(32))
    np.testing.assert_allclose(jnp.sqrt(jnp.mean(out**2, -1)), 1, atol=1e-4)
    w = jnp.linspace(0.5, 2, 32)
    np.testing.assert_allclose(impl.rms_norm(x, w), x / jnp.sqrt(jnp.mean(x**2, -1, keepdims=True) + 1e-6) * w,
                               rtol=1e-5)
    # RMSNorm does not remove the mean: a constant positive row stays positive
    assert jnp.all(impl.rms_norm(jnp.full((1, 4), 3.0), jnp.ones(4)) > 0)


def test_dropout(impl):
    key = jax.random.key(2)
    x = jnp.ones((200, 500))
    out = impl.dropout(key, x, 0.3, True)
    zeros = float(jnp.mean(out == 0))
    assert zeros == pytest.approx(0.3, abs=0.01)
    np.testing.assert_allclose(out[out != 0], 1 / 0.7, rtol=1e-6)
    assert float(out.mean()) == pytest.approx(1.0, abs=0.01)
    np.testing.assert_array_equal(impl.dropout(key, x, 0.3, True), out)  # same key, same mask
    assert not jnp.array_equal(impl.dropout(jax.random.key(3), x, 0.3, True), out)
    np.testing.assert_array_equal(impl.dropout(key, x, 0.3, False), x)  # eval mode: identity
    np.testing.assert_array_equal(impl.dropout(key, x, 0.0, True), x)
    np.testing.assert_array_equal(impl.dropout(key, x, 1.0, True), jnp.zeros_like(x))


def test_residual_block(impl):
    k1, k2, k3 = jax.random.split(jax.random.key(4), 3)
    d, h = 8, 32
    params = {"norm": jnp.ones(d), "w1": jax.random.normal(k1, (d, h)) * 0.1,
              "w2": jax.random.normal(k2, (h, d)) * 0.1}
    x = jax.random.normal(k3, (5, d))
    xn = x / jnp.sqrt(jnp.mean(x**2, -1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(impl.residual_block(params, x),
                               x + jax.nn.relu(xn @ params["w1"]) @ params["w2"], rtol=1e-5, atol=1e-6)
    # with w2 = 0 the block is exactly the identity — why residual nets start out easy to train
    zero = dict(params, w2=jnp.zeros((h, d)))
    np.testing.assert_array_equal(impl.residual_block(zero, x), x)


def test_global_grad_norm(impl):
    grads = {"a": jnp.array([3.0, 0.0]), "b": [jnp.array([[4.0]]), jnp.zeros((2, 2))]}
    assert float(impl.global_grad_norm(grads)) == pytest.approx(5.0)
    assert float(impl.global_grad_norm({})) == 0.0


def test_save_and_restore(impl, tmp_path):
    state = {"params": {"w": jnp.arange(6.0).reshape(2, 3), "b": jnp.ones(3)},
             "step": jnp.array(42, jnp.int32)}
    restored = impl.save_and_restore(state, tmp_path / "ckpt")
    assert jax.tree.structure(restored) == jax.tree.structure(state)
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert (tmp_path / "ckpt").exists()


def test_batch_norm(impl):
    x = jax.random.normal(jax.random.key(20), (64, 10)) * 3 + jnp.arange(10.0)
    g, b = jnp.ones(10), jnp.zeros(10)
    rm, rv = jnp.zeros(10), jnp.ones(10)
    y, nm, nv = impl.batch_norm(x, g, b, rm, rv, momentum=0.1, train=True)
    np.testing.assert_allclose(y.mean(0), 0, atol=1e-5)          # per *feature*, over the batch
    np.testing.assert_allclose(y.std(0), 1, atol=1e-3)
    np.testing.assert_allclose(nm, 0.1 * x.mean(0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(nv, 0.9 + 0.1 * x.var(0), rtol=1e-5)
    # eval mode: running stats, unchanged
    rm2, rv2 = jnp.full(10, 2.0), jnp.full(10, 4.0)
    ye, em, ev = impl.batch_norm(x, g * 3, b + 1, rm2, rv2, train=False)
    np.testing.assert_allclose(ye, (x - 2.0) / jnp.sqrt(4.0 + 1e-5) * 3 + 1, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(em, rm2)
    np.testing.assert_array_equal(ev, rv2)
    # many training steps on same-distribution batches: running stats converge to the data's stats
    m, v = rm, rv
    for i in range(200):
        xb = jax.random.normal(jax.random.key(100 + i), (256, 10)) * 2 + 5
        _, m, v = impl.batch_norm(xb, g, b, m, v, momentum=0.1, train=True)
    np.testing.assert_allclose(m, 5, atol=0.1)
    np.testing.assert_allclose(v, 4, atol=0.3)


def test_manual_backward(impl):
    k = jax.random.split(jax.random.key(21), 5)
    params = {"w1": jax.random.normal(k[0], (12, 32)) * 0.3, "b1": jax.random.normal(k[1], (32,)) * 0.1,
              "w2": jax.random.normal(k[2], (32, 5)) * 0.3, "b2": jnp.zeros(5)}
    x = jax.random.normal(k[3], (40, 12))
    y = jax.random.randint(k[4], (40,), 0, 5)

    def loss(p):  # an independent reference forward pass
        logits = jnp.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(40), y])

    with jax.default_matmul_precision("highest"):
        ref = jax.grad(loss)(params)
        _, cache = impl.forward_with_cache(params, x, y)
        mine = impl.manual_backward(params, cache, y)
    assert set(mine) == set(params)
    for name, value in params.items():
        assert mine[name].shape == value.shape, name
        np.testing.assert_allclose(mine[name], ref[name], rtol=1e-4, atol=1e-5, err_msg=name)
