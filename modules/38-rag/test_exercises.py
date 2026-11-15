import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_chunk_sentences(impl):
    assert impl.chunk_sentences([3, 3, 3, 3], 6, 0) == [(0, 2), (2, 4)]
    assert impl.chunk_sentences([3, 3, 3, 3], 6, 1) == [(0, 2), (1, 3), (2, 4)]
    assert impl.chunk_sentences([10, 2, 2], 5, 0) == [(0, 1), (1, 3)]          # long sentence stays whole
    assert impl.chunk_sentences([2, 2, 2], 100, 1) == [(0, 3)]                 # everything fits
    assert impl.chunk_sentences([5, 5, 5], 5, 3) == [(0, 1), (1, 2), (2, 3)]   # overlap can't stall
    assert impl.chunk_sentences([], 5, 0) == []


def test_bm25_weights(impl):
    w = np.asarray(impl.bm25_weights(np.array([1, 2, 1]), np.array([10, 10, 20]), np.array([1, 1, 50]),
                                     100, 10.0, k1=1.5, b=0.75))
    idf = np.log1p((100 - np.array([1, 1, 50]) + 0.5) / (np.array([1, 1, 50]) + 0.5))
    tf, dl = np.array([1, 2, 1]), np.array([10, 10, 20])
    np.testing.assert_allclose(w, idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * dl / 10)), rtol=1e-5)
    assert w[1] > w[0] > w[2]          # more occurrences > fewer; common terms count less
    assert w.dtype == np.float32


def test_info_nce(impl):
    eye = jnp.eye(4)
    assert float(impl.info_nce(eye, eye, 0.05)) < 1e-6                     # perfect alignment
    q = jax.random.normal(jax.random.key(0), (8, 16))
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    d = jnp.roll(q, 1, axis=0)
    logits = q @ d.T / 0.1
    expect = -jnp.mean(jnp.diagonal(jax.nn.log_softmax(logits, -1)))
    assert float(impl.info_nce(q, d, 0.1)) == pytest.approx(float(expect), rel=1e-5)
    extra = jnp.concatenate([d, -q])                                         # 8 hard negatives appended
    logits2 = q @ extra.T / 0.1
    expect2 = -jnp.mean(jnp.diagonal(jax.nn.log_softmax(logits2, -1)))
    assert float(impl.info_nce(q, extra, 0.1)) == pytest.approx(float(expect2), rel=1e-5)
    uniform = jnp.ones((4, 3)) / jnp.sqrt(3.0)
    assert float(impl.info_nce(uniform, uniform, 1.0)) == pytest.approx(np.log(4), rel=1e-5)


def test_topk_cosine(impl):
    D = jnp.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    s, i = impl.topk_cosine(jnp.array([[5.0, 0.1], [0.0, 1.0]]), D, 2)
    np.testing.assert_array_equal(np.asarray(i), [[0, 2], [1, 2]])
    np.testing.assert_allclose(np.asarray(s)[1], [1.0, np.sqrt(0.5)], rtol=1e-5)


def test_rrf_fuse(impl):
    assert impl.rrf_fuse([[1, 2, 3], [3, 1]], k=60) == [1, 3, 2]
    assert impl.rrf_fuse([[5], [7]], k=60) == [5, 7]                         # tie -> smaller id first
    assert impl.rrf_fuse([[2, 1]], k=0) == [2, 1]


def test_minmax_fuse(impl):
    out = np.asarray(impl.minmax_fuse(jnp.array([0.0, 5.0, 10.0]), jnp.array([3.0, 2.0, 1.0]), 0.25))
    np.testing.assert_allclose(out, [0.75, 0.25 * 0.5 + 0.75 * 0.5, 0.25], rtol=1e-6)
    flat = np.asarray(impl.minmax_fuse(jnp.array([2.0, 2.0]), jnp.array([0.0, 4.0]), 0.5))
    np.testing.assert_allclose(flat, [0.0, 0.5])


def test_metrics(impl):
    ranked = np.array([[7, 3, 9], [1, 2, 3], [4, 5, 6]])
    relevant = np.array([3, 1, 0])           # rank 2, rank 1, missing
    assert impl.recall_at_k(ranked, relevant, 1) == pytest.approx(1 / 3)
    assert impl.recall_at_k(ranked, relevant, 3) == pytest.approx(2 / 3)
    assert impl.mrr_at_k(ranked, relevant, 3) == pytest.approx((0.5 + 1.0) / 3)
    assert impl.mrr_at_k(ranked, relevant, 1) == pytest.approx(1 / 3)
    assert impl.ndcg_at_k(ranked, relevant, 3) == pytest.approx((1 / np.log2(3) + 1.0) / 3)
