import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _rand(shape, seed=0):
    return jax.random.normal(jax.random.key(seed), shape)


def test_batched_dot(impl):
    a, b = _rand((5, 3), 0), _rand((5, 3), 1)
    out = impl.batched_dot(a, b)
    assert out.shape == (5,)
    np.testing.assert_allclose(out, np.sum(np.asarray(a) * np.asarray(b), axis=1), rtol=1e-5)
    np.testing.assert_allclose(jax.jit(impl.batched_dot)(a, b), out, rtol=1e-6)


def test_pairwise_distances(impl):
    x, y = _rand((4, 3), 2), _rand((6, 3), 3)
    out = impl.pairwise_distances(x, y)
    assert out.shape == (4, 6)
    ref = np.linalg.norm(np.asarray(x)[:, None, :] - np.asarray(y)[None, :, :], axis=-1)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.diag(impl.pairwise_distances(x, x)), np.zeros(4), atol=1e-6)


def _loss(params, x, y):
    pred = x @ params["w"] + params["b"]
    return (pred - y) ** 2


def test_per_example_grads(impl):
    params = {"w": jnp.array([0.5, -1.0]), "b": jnp.array(0.2)}
    xs, ys = _rand((7, 2), 4), _rand((7,), 5)
    g = impl.per_example_grads(_loss, params, xs, ys)
    assert g["w"].shape == (7, 2) and g["b"].shape == (7,)
    for i in range(7):
        gi = jax.grad(_loss)(params, xs[i], ys[i])
        np.testing.assert_allclose(g["w"][i], gi["w"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(g["b"][i], gi["b"], rtol=1e-5, atol=1e-6)
    # the mean of per-example grads is the ordinary batch gradient
    batch = jax.grad(lambda p: jnp.mean(jax.vmap(_loss, in_axes=(None, 0, 0))(p, xs, ys)))(params)
    np.testing.assert_allclose(g["w"].mean(0), batch["w"], rtol=1e-5, atol=1e-6)


def test_tree_l2_norm(impl):
    tree = {"a": jnp.array([3.0]), "b": [jnp.array([[4.0, 0.0]]), jnp.array(12.0)]}
    assert float(impl.tree_l2_norm(tree)) == pytest.approx(13.0)
    assert float(impl.tree_l2_norm({"z": jnp.zeros(5)})) == 0.0


def test_tree_axpy(impl):
    x = {"a": jnp.array([1.0, 2.0]), "b": (jnp.array(3.0),)}
    y = {"a": jnp.array([10.0, 20.0]), "b": (jnp.array(30.0),)}
    out = impl.tree_axpy(2.0, x, y)
    assert jax.tree.structure(out) == jax.tree.structure(x)
    np.testing.assert_allclose(out["a"], [12.0, 24.0])
    np.testing.assert_allclose(out["b"][0], 36.0)


def test_init_mlp(impl):
    params = impl.init_mlp(jax.random.key(0), [784, 256, 10])
    assert isinstance(params, list) and len(params) == 2
    assert params[0]["w"].shape == (784, 256) and params[0]["b"].shape == (256,)
    assert params[1]["w"].shape == (256, 10) and params[1]["b"].shape == (10,)
    np.testing.assert_allclose(params[0]["b"], 0.0)
    assert float(jnp.std(params[0]["w"])) == pytest.approx(np.sqrt(2 / 784), rel=0.05)
    again = impl.init_mlp(jax.random.key(0), [784, 256, 10])
    np.testing.assert_allclose(again[0]["w"], params[0]["w"])  # deterministic
    square = impl.init_mlp(jax.random.key(1), [8, 8, 8])
    assert not np.allclose(square[0]["w"], square[1]["w"])  # each layer has its own key


def test_mlp_forward(impl):
    params = [
        {"w": jnp.array([[1.0, -1.0], [2.0, 0.5]]), "b": jnp.array([0.0, -1.0])},
        {"w": jnp.array([[1.0], [3.0]]), "b": jnp.array([0.5])},
    ]
    x = jnp.array([1.0, 1.0])
    # hidden = relu([3.0, -1.5]) = [3, 0]; out = 3 * 1 + 0 * 3 + 0.5
    np.testing.assert_allclose(impl.mlp_forward(params, x), [3.5])
    batch = jnp.stack([x, -x])
    out = impl.mlp_forward(params, batch)
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out[0], [3.5])


def test_count_params(impl):
    params = [{"w": jnp.zeros((784, 256)), "b": jnp.zeros(256)}, {"w": jnp.zeros((256, 10)), "b": jnp.zeros(10)}]
    n = impl.count_params(params)
    assert n == 784 * 256 + 256 + 256 * 10 + 10 and isinstance(n, int)
