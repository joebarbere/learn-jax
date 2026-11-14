import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest


def _params(key=0):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(key), 4)
    return [
        {"w": jax.random.normal(k1, (5, 7)), "b": jax.random.normal(k2, (7,))},
        {"w": jax.random.normal(k3, (7, 3)), "b": jax.random.normal(k4, (3,))},
    ]


def test_he_init(impl):
    w = impl.he_init(jax.random.key(0), 1000, 500)
    assert w.shape == (1000, 500)
    assert w.dtype == jnp.float32
    assert float(w.std()) == pytest.approx(np.sqrt(2 / 1000), rel=0.03)
    assert abs(float(w.mean())) < 0.002
    assert not np.allclose(w, impl.he_init(jax.random.key(1), 1000, 500))


def test_init_mlp(impl):
    params = impl.init_mlp(jax.random.key(0), [784, 256, 256, 10])
    assert len(params) == 3
    assert [p["w"].shape for p in params] == [(784, 256), (256, 256), (256, 10)]
    assert [p["b"].shape for p in params] == [(256,), (256,), (10,)]
    assert all(float(jnp.abs(p["b"]).max()) == 0.0 for p in params)
    assert float(params[0]["w"].std()) == pytest.approx(np.sqrt(2 / 784), rel=0.05)
    # each layer gets its own key: the two 256x256-ish layers must differ
    a = impl.init_mlp(jax.random.key(0), [64, 64, 64])
    assert not np.allclose(a[0]["w"], a[1]["w"])


def test_mlp_apply(impl):
    params = _params()
    x = jax.random.normal(jax.random.key(9), (4, 5))
    h = jax.nn.relu(x @ params[0]["w"] + params[0]["b"])
    expected = h @ params[1]["w"] + params[1]["b"]
    out = impl.mlp_apply(params, x)
    assert out.shape == (4, 3)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
    assert float(out.min()) < 0  # last layer must not be ReLU'd


def test_loss_fn(impl):
    params = _params()
    x = jax.random.normal(jax.random.key(3), (6, 5))
    y = jnp.array([0, 1, 2, 2, 1, 0])
    ref = optax.softmax_cross_entropy_with_integer_labels(
        jax.nn.relu(x @ params[0]["w"] + params[0]["b"]) @ params[1]["w"] + params[1]["b"], y).mean()
    assert float(impl.loss_fn(params, x, y)) == pytest.approx(float(ref), rel=1e-5)
    # all-zero last layer -> uniform prediction -> loss = ln(3)
    flat = [params[0], {"w": jnp.zeros((7, 3)), "b": jnp.zeros(3)}]
    assert float(impl.loss_fn(flat, x, y)) == pytest.approx(np.log(3), rel=1e-6)


def test_data_batches(impl):
    batches = impl.data_batches(jax.random.key(0), 10, 3)
    assert len(batches) == 3
    assert all(len(b) == 3 for b in batches)
    seen = np.concatenate([np.asarray(b) for b in batches])
    assert len(set(seen.tolist())) == 9 and set(seen.tolist()) <= set(range(10))
    again = impl.data_batches(jax.random.key(0), 10, 3)
    assert all(np.array_equal(a, b) for a, b in zip(batches, again, strict=True))
    big = np.concatenate([np.asarray(b) for b in impl.data_batches(jax.random.key(1), 100, 10)])
    assert not np.array_equal(big, np.arange(100))  # shuffled


def test_accuracy(impl):
    params = _params()
    x = jax.random.normal(jax.random.key(5), (25, 5))
    preds = (jax.nn.relu(x @ params[0]["w"] + params[0]["b"]) @ params[1]["w"] + params[1]["b"]).argmax(-1)
    y = preds.at[:5].set((preds[:5] + 1) % 3)  # exactly 5 wrong
    acc = impl.accuracy(params, x, y, batch_size=4)  # 25 rows, chunks of 4: last chunk has 1 row
    assert isinstance(acc, float)
    assert acc == pytest.approx(20 / 25)


def test_train_step(impl):
    params = _params()
    x = jax.random.normal(jax.random.key(7), (16, 5))
    y = jnp.arange(16) % 3
    opt = optax.sgd(0.1)
    state = opt.init(params)
    new_params, _new_state, loss = impl.train_step(params, state, x, y, opt)
    # reference: one manual SGD step
    def ref_loss(p):
        logits = jax.nn.relu(x @ p[0]["w"] + p[0]["b"]) @ p[1]["w"] + p[1]["b"]
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    l0, g = jax.value_and_grad(ref_loss)(params)
    assert float(loss) == pytest.approx(float(l0), rel=1e-5)
    for a, p, gr in zip(jax.tree.leaves(new_params), jax.tree.leaves(params), jax.tree.leaves(g), strict=True):
        np.testing.assert_allclose(a, p - 0.1 * gr, rtol=1e-4, atol=1e-5)
    assert jax.tree.structure(new_params) == jax.tree.structure(params)
    # repeated steps reduce the loss
    p, s = params, opt.init(params)
    losses = []
    for _ in range(30):
        p, s, lo = impl.train_step(p, s, x, y, opt)
        losses.append(float(lo))
    assert losses[-1] < losses[0] * 0.7
