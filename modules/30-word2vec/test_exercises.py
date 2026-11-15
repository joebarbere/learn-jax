import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_build_vocab(impl):
    toks = ["the", "cat", "sat", "on", "the", "mat", "the", "cat", "ran", "a", "b"]
    w2i, i2w, counts = impl.build_vocab(toks, min_count=2)
    assert i2w == ["the", "cat"]
    assert w2i == {"the": 0, "cat": 1}
    np.testing.assert_array_equal(counts, [3, 2])
    _, i2w1, c1 = impl.build_vocab(toks, min_count=1)
    assert i2w1[:2] == ["the", "cat"] and i2w1[2:] == sorted(i2w1[2:])   # ties alphabetical
    assert int(np.sum(c1)) == len(toks)


def test_subsample_probs(impl):
    counts = np.array([1_000_000, 1_000, 1])
    keep = impl.subsample_probs(counts, t=1e-4)
    f = counts / counts.sum()
    np.testing.assert_allclose(keep, np.minimum(1, np.sqrt(1e-4 / f) + 1e-4 / f))
    assert keep[0] < 0.02 and keep[2] == 1.0
    assert np.all(np.diff(keep) >= 0)


def test_negative_sampling_table(impl):
    counts = np.array([16, 1])
    p = impl.negative_sampling_table(counts, 0.75)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] / p[1] == pytest.approx(8.0)            # 16^0.75 = 8
    np.testing.assert_allclose(impl.negative_sampling_table(counts, 0.0), [0.5, 0.5])


def test_skipgram_pairs(impl):
    ids = np.array([10, 11, 12, 13])
    c, x = impl.skipgram_pairs(ids, window=1, key=jax.random.key(0))
    pairs = sorted(zip(np.asarray(c).tolist(), np.asarray(x).tolist()))
    assert pairs == sorted([(10, 11), (11, 10), (11, 12), (12, 11), (12, 13), (13, 12)])
    c2, x2 = impl.skipgram_pairs(ids, window=2, key=jax.random.key(1))
    assert len(c2) == 6 + 4
    assert (13, 11) in set(zip(np.asarray(c2).tolist(), np.asarray(x2).tolist()))


def test_sgns_loss(impl):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    E_in = jax.random.normal(k1, (20, 8)) * 0.3
    E_out = jax.random.normal(k2, (20, 8)) * 0.3
    center, context = jnp.array([1, 2, 3]), jnp.array([4, 5, 6])
    negs = jnp.array([[7, 8], [9, 10], [11, 12]])
    got = impl.sgns_loss(center, context, negs, E_in, E_out)
    v = np.asarray(E_in)[[1, 2, 3]]
    up, un = np.asarray(E_out)[[4, 5, 6]], np.asarray(E_out)[np.asarray(negs)]

    def ls(z):
        return -np.log1p(np.exp(-z))
    want = -np.mean(ls((v * up).sum(-1)) + ls(-np.einsum("bd,bkd->bk", v, un)).sum(-1))
    assert float(got) == pytest.approx(want, rel=1e-5)
    # zero vectors: every logit is 0, loss = (1 + K) * log 2
    z = jnp.zeros((20, 8))
    assert float(impl.sgns_loss(center, context, negs, z, z)) == pytest.approx(3 * np.log(2), rel=1e-6)


def test_cosine_neighbors(impl):
    E = jnp.array([[1.0, 0.0], [10.0, 1.0], [0.0, 1.0], [-1.0, 0.0]])
    nb = np.asarray(impl.cosine_neighbors(E, 0, 2))
    assert nb.tolist() == [1, 2]                         # scale doesn't matter; excludes itself
    assert 0 not in np.asarray(impl.cosine_neighbors(E, 0, 3)).tolist()


def test_analogy(impl):
    # man, woman, king, queen, other — queen = king - man + woman exactly
    man, woman = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    king = np.array([1.0, 0.0, 1.0])
    E = jnp.array([man, woman, king, king - man + woman, [0.0, 0.0, -1.0]])
    assert int(impl.analogy(E, 0, 1, 2, k=1)[0]) == 3
    top = np.asarray(impl.analogy(E, 0, 1, 2, k=2)).tolist()
    assert not set(top) & {0, 1, 2}
