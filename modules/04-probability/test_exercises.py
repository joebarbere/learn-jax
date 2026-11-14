import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_stable_softmax(impl):
    x = jnp.array([[1.0, 2.0, 3.0], [0.5, 0.5, -1.0]])
    p = impl.stable_softmax(x)
    np.testing.assert_allclose(p, jax.nn.softmax(x, axis=-1), rtol=1e-5)
    np.testing.assert_allclose(jnp.sum(p, axis=-1), 1.0, rtol=1e-6)
    big = impl.stable_softmax(jnp.array([1000.0, 1001.0, 1002.0]))
    assert bool(jnp.all(jnp.isfinite(big)))
    np.testing.assert_allclose(big, jax.nn.softmax(jnp.array([0.0, 1.0, 2.0])), rtol=1e-5)
    np.testing.assert_allclose(impl.stable_softmax(x, axis=0), jax.nn.softmax(x, axis=0), rtol=1e-5)


def test_log_softmax(impl):
    x = jnp.array([[1.0, 2.0, 3.0], [-200.0, 0.0, 200.0]])
    out = impl.log_softmax(x)
    np.testing.assert_allclose(out, jax.nn.log_softmax(x, axis=-1), rtol=1e-5, atol=1e-5)
    assert bool(jnp.all(jnp.isfinite(out)))  # log(softmax) would give -inf for the -200 entry


def test_cross_entropy(impl):
    logits = jnp.zeros((4, 10))
    labels = jnp.array([0, 3, 9, 5])
    assert float(impl.cross_entropy(logits, labels)) == pytest.approx(math.log(10), rel=1e-5)
    logits = jnp.array([[2.0, 0.5, -1.0], [0.1, 0.2, 3.0]])
    labels = jnp.array([0, 1])
    expected = -(jax.nn.log_softmax(logits)[0, 0] + jax.nn.log_softmax(logits)[1, 1]) / 2
    assert float(impl.cross_entropy(logits, labels)) == pytest.approx(float(expected), rel=1e-5)
    assert jnp.shape(impl.cross_entropy(logits, labels)) == ()


def test_entropy(impl):
    assert float(impl.entropy(jnp.ones(8) / 8)) == pytest.approx(math.log(8), rel=1e-5)
    assert float(impl.entropy(jnp.array([1.0, 0.0, 0.0]))) == pytest.approx(0.0, abs=1e-7)
    h = impl.entropy(jnp.array([[0.5, 0.5, 0.0], [0.25, 0.25, 0.5]]))
    np.testing.assert_allclose(h, [math.log(2), 1.5 * math.log(2)], rtol=1e-5)


def test_kl_divergence(impl):
    p = jnp.array([0.1, 0.4, 0.5])
    q = jnp.array([0.3, 0.3, 0.4])
    expected = sum(a * math.log(a / b) for a, b in zip([0.1, 0.4, 0.5], [0.3, 0.3, 0.4]))
    assert float(impl.kl_divergence(p, q)) == pytest.approx(expected, rel=1e-5)
    assert float(impl.kl_divergence(p, p)) == pytest.approx(0.0, abs=1e-7)
    assert float(impl.kl_divergence(q, p)) != pytest.approx(expected, rel=1e-3)  # not symmetric
    assert float(impl.kl_divergence(jnp.array([0.5, 0.5, 0.0]), q)) == pytest.approx(
        0.5 * math.log(0.5 / 0.3) * 2, rel=1e-5)


def test_sample_categorical(impl):
    logits = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.4]))
    key = jax.random.key(0)
    s = impl.sample_categorical(key, logits, 40_000)
    assert s.shape == (40_000,)
    assert jnp.issubdtype(s.dtype, jnp.integer)
    freq = np.bincount(np.asarray(s), minlength=4) / 40_000
    np.testing.assert_allclose(freq, [0.1, 0.2, 0.3, 0.4], atol=0.01)
    np.testing.assert_array_equal(impl.sample_categorical(key, logits, 50), impl.sample_categorical(key, logits, 50))
    assert not np.array_equal(impl.sample_categorical(key, logits, 50),
                              impl.sample_categorical(jax.random.key(1), logits, 50))


def test_perplexity(impl):
    assert float(impl.perplexity(math.log(33.0))) == pytest.approx(33.0, rel=1e-5)
    assert float(impl.perplexity(0.0)) == pytest.approx(1.0)
