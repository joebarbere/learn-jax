import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_bigram_counts(impl):
    ids = jnp.array([0, 1, 1, 2, 0, 1])
    c = np.asarray(impl.bigram_counts(ids, 3))
    expected = np.zeros((3, 3), int)
    for a, b in [(0, 1), (1, 1), (1, 2), (2, 0), (0, 1)]:
        expected[a, b] += 1
    np.testing.assert_array_equal(c, expected)
    assert c.sum() == len(ids) - 1
    assert c.shape == (3, 3)
    np.testing.assert_array_equal(np.asarray(impl.bigram_counts(jnp.array([2]), 3)), np.zeros((3, 3)))


def test_bigram_nll(impl):
    vocab = 4
    zeros = jnp.zeros((vocab, vocab), jnp.int32)
    # No counts: smoothing alone gives the uniform distribution → ln(vocab)
    assert float(impl.bigram_nll(zeros, jnp.array([0, 1, 2, 3]))) == pytest.approx(math.log(vocab), rel=1e-5)
    ids = jnp.array([0, 1, 0, 1, 0, 1])
    counts = jnp.zeros((vocab, vocab), jnp.int32).at[jnp.array([0, 1, 0, 1, 0]), jnp.array([1, 0, 1, 0, 1])].add(1)
    # P(1|0) = (3+1)/(3+4), P(0|1) = (2+1)/(2+4)
    expected = -(3 * math.log(4 / 7) + 2 * math.log(3 / 6)) / 5
    assert float(impl.bigram_nll(counts, ids, 1.0)) == pytest.approx(expected, rel=1e-5)
    # Less smoothing on seen pairs → lower loss, and still finite
    low = float(impl.bigram_nll(counts, ids, 0.01))
    assert math.isfinite(low) and low < expected


def test_get_batch(impl):
    data = jnp.arange(100, dtype=jnp.int32)
    x, y = impl.get_batch(jax.random.key(0), data, 8, 10)
    assert x.shape == (8, 10) and y.shape == (8, 10)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(x) + 1)          # y is x shifted by one
    np.testing.assert_array_equal(np.diff(np.asarray(x), axis=1), 1)         # contiguous windows
    assert int(y.max()) <= 99 and int(x.min()) >= 0                          # never runs off the end
    x2, _ = impl.get_batch(jax.random.key(1), data, 8, 10)
    assert not np.array_equal(np.asarray(x), np.asarray(x2))                 # the key matters
    # Works under jit (static sizes, traced key)
    jx, _ = jax.jit(impl.get_batch, static_argnums=(2, 3))(jax.random.key(0), data, 8, 10)
    np.testing.assert_array_equal(np.asarray(jx), np.asarray(x))
    # Tight fit: only one possible window
    tx, ty = impl.get_batch(jax.random.key(3), jnp.arange(5), 4, 4)
    np.testing.assert_array_equal(np.asarray(tx[0]), [0, 1, 2, 3])
    np.testing.assert_array_equal(np.asarray(ty[0]), [1, 2, 3, 4])


def test_mlp_lm_init(impl):
    p = impl.mlp_lm_init(jax.random.key(0), vocab=65, context=3, emb_dim=10, hidden=32)
    assert set(p) == {"emb", "w1", "b1", "w2", "b2"}
    assert p["emb"].shape == (65, 10)
    assert p["w1"].shape == (30, 32) and p["b1"].shape == (32,)
    assert p["w2"].shape == (32, 65) and p["b2"].shape == (65,)
    assert float(jnp.abs(p["b1"]).sum()) == 0.0
    assert not np.allclose(np.asarray(p["emb"][0]), np.asarray(p["emb"][1]))
    assert 0.05 < float(jnp.std(p["emb"])) < 0.2


def test_mlp_lm_apply(impl):
    p = impl.mlp_lm_init(jax.random.key(0), vocab=20, context=4, emb_dim=6, hidden=16)
    ctx = jnp.array([[1, 2, 3, 4], [0, 0, 0, 0], [19, 18, 17, 16]])
    logits = impl.mlp_lm_apply(p, ctx)
    assert logits.shape == (3, 20)
    flat = p["emb"][ctx].reshape(3, -1)
    expected = jnp.tanh(flat @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]
    np.testing.assert_allclose(np.asarray(logits), np.asarray(expected), rtol=1e-5, atol=1e-6)
    # initial predictions are close to uniform → loss near ln(vocab)
    lp = jax.nn.log_softmax(logits)
    assert float(-lp.mean()) == pytest.approx(math.log(20), rel=0.05)
    # differentiable
    g = jax.grad(lambda q: impl.mlp_lm_apply(q, ctx).sum())(p)
    assert g["emb"].shape == p["emb"].shape


def test_sample(impl):
    vocab = 5

    def always_next(window):  # deterministic "model": strongly predicts (last + 1) % vocab
        return jax.nn.one_hot((window[-1] + 1) % vocab, vocab) * 50.0

    out = impl.sample(jax.random.key(0), always_next, jnp.array([0, 1]), 7, temperature=0)
    np.testing.assert_array_equal(np.asarray(out), [2, 3, 4, 0, 1, 2, 3])
    out_t = impl.sample(jax.random.key(0), always_next, jnp.array([0, 1]), 7, temperature=1.0)
    np.testing.assert_array_equal(np.asarray(out_t), [2, 3, 4, 0, 1, 2, 3])  # 50-logit gap ≈ certain
    assert len(impl.sample(jax.random.key(0), always_next, jnp.array([3]), 0)) == 0

    def uniform(window):
        return jnp.zeros(vocab)

    many = np.asarray(impl.sample(jax.random.key(1), uniform, jnp.array([0]), 400, temperature=1.0))
    assert many.shape == (400,)
    assert set(many.tolist()) == set(range(vocab))  # explores every token
    a = impl.sample(jax.random.key(7), uniform, jnp.array([0]), 20)
    b = impl.sample(jax.random.key(7), uniform, jnp.array([0]), 20)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))  # same key → same text
