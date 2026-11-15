import jax
import jax.numpy as jnp
import numpy as np


def _blobs(seed=0, n=60):
    rng = np.random.default_rng(seed)
    a = rng.normal([2.0, 2.0], 0.6, (n // 2, 2))
    b = rng.normal([-2.0, -1.5], 0.6, (n // 2, 2))
    X = np.concatenate([a, b]).astype(np.float32)
    y = np.concatenate([np.ones(n // 2), -np.ones(n // 2)]).astype(np.float32)
    return jnp.asarray(X), jnp.asarray(y)


XOR_X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_Y = jnp.array([-1.0, 1.0, 1.0, -1.0])
AND_Y = jnp.array([-1.0, -1.0, -1.0, 1.0])


def test_perceptron_update(impl):
    w, b = jnp.array([1.0, 0.0]), jnp.float32(0.0)
    # correctly classified: no change
    w2, b2 = impl.perceptron_update(w, b, jnp.array([2.0, 1.0]), 1.0)
    np.testing.assert_allclose(w2, w)
    assert float(b2) == 0.0
    # mistake: w += lr*y*x, b += lr*y
    w3, b3 = impl.perceptron_update(w, b, jnp.array([2.0, 1.0]), -1.0, lr=0.5)
    np.testing.assert_allclose(w3, [0.0, -0.5])
    assert float(b3) == -0.5
    # a point exactly on the boundary counts as a mistake
    w4, b4 = impl.perceptron_update(jnp.zeros(2), jnp.float32(0.0), jnp.array([1.0, 1.0]), 1.0)
    np.testing.assert_allclose(w4, [1.0, 1.0])
    assert float(b4) == 1.0
    # jit-friendly
    jax.jit(impl.perceptron_update)(w, b, jnp.array([2.0, 1.0]), -1.0)


def test_train_perceptron_converges_on_separable_data(impl):
    X, y = _blobs()
    w, b, mistakes = impl.train_perceptron(X, y, 20)
    assert mistakes.shape == (20,)
    assert int(mistakes[0]) > 0
    assert int(mistakes[-1]) == 0
    assert bool(jnp.all(y * (X @ w + b) > 0))


def test_train_perceptron_never_converges_on_xor(impl):
    _, _, mistakes = impl.train_perceptron(XOR_X, XOR_Y, 100)
    assert bool(jnp.all(mistakes > 0))


def test_is_linearly_separable(impl):
    assert impl.is_linearly_separable(XOR_X, AND_Y) is True
    assert impl.is_linearly_separable(XOR_X, XOR_Y) is False


def test_hebbian_weights(impl):
    rng = np.random.default_rng(1)
    P = jnp.asarray(rng.choice([-1.0, 1.0], (3, 50)))
    W = impl.hebbian_weights(P)
    assert W.shape == (50, 50)
    np.testing.assert_allclose(W, W.T, atol=1e-6)
    np.testing.assert_allclose(jnp.diag(W), 0.0, atol=1e-7)
    expected = (P.T @ P / 50) * (1 - np.eye(50))
    np.testing.assert_allclose(W, expected, atol=1e-6)


def test_hopfield_energy(impl):
    W = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    assert float(impl.hopfield_energy(W, jnp.array([1.0, 1.0]))) == -1.0
    assert float(impl.hopfield_energy(W, jnp.array([1.0, -1.0]))) == 1.0


def test_hopfield_recall_cleans_up_and_lowers_energy(impl):
    rng = np.random.default_rng(2)
    n = 100
    P = jnp.asarray(rng.choice([-1.0, 1.0], (3, n)))
    W = impl.hebbian_weights(P)
    noisy = np.array(P[0])
    flip = rng.choice(n, 10, replace=False)
    noisy[flip] *= -1
    noisy = jnp.asarray(noisy)
    out = impl.hopfield_recall(W, noisy, 10 * n, jax.random.key(0))
    assert out.shape == noisy.shape
    np.testing.assert_array_equal(out, P[0])
    assert float(impl.hopfield_energy(W, out)) <= float(impl.hopfield_energy(W, noisy))
    # a stored pattern is a fixed point
    np.testing.assert_array_equal(impl.hopfield_recall(W, P[1], 5 * n, jax.random.key(1)), P[1])
    # vmappable over starting states
    batch = jnp.stack([noisy, P[1]])
    outs = jax.vmap(lambda s: impl.hopfield_recall(W, s, 10 * n, jax.random.key(2)))(batch)
    assert outs.shape == (2, n)


def test_modern_hopfield_update_is_attention(impl):
    k1, k2 = jax.random.split(jax.random.key(3))
    X = jax.random.normal(k1, (8, 16))
    xi = jax.random.normal(k2, (5, 16))
    beta = 0.7
    out = impl.modern_hopfield_update(X, xi, beta)
    attn = jax.nn.dot_product_attention(xi[None, :, None, :], X[None, :, None, :], X[None, :, None, :],
                                        scale=beta)[0, :, 0, :]
    np.testing.assert_allclose(out, attn, atol=1e-5)
    # single state works too, and large beta retrieves the nearest stored pattern
    single = impl.modern_hopfield_update(X, X[3] + 0.05, 50.0)
    assert single.shape == (16,)
    np.testing.assert_allclose(single, X[3], atol=1e-3)
