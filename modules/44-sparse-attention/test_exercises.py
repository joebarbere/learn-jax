import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(44)


def _rand(i, shape):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32)


def _dense(q, k, v, mask):
    s = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(q.shape[-1])
    return jnp.einsum("bhts,bhsd->bhtd", jax.nn.softmax(jnp.where(mask, s, -jnp.inf), -1), v)


# ---------------------------------------------------------------- window_mask
def test_window_mask(impl):
    m = np.asarray(impl.window_mask(6, 3))
    expect = np.array([[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                       [0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1]], bool)
    assert m.dtype == bool and np.array_equal(m, expect)


def test_window_mask_sinks_and_full(impl):
    m = np.asarray(impl.window_mask(6, 2, 1))
    assert m[5].tolist() == [True, False, False, False, True, True]
    assert np.array_equal(np.asarray(impl.window_mask(5, 99)), np.tril(np.ones((5, 5), bool)))


def test_window_mask_traced(impl):
    f = jax.jit(lambda w, s: impl.window_mask(8, w, s))
    assert np.array_equal(np.asarray(f(3, 2)), np.asarray(impl.window_mask(8, 3, 2)))


# ---------------------------------------------------------------- kv_cache_bytes
def test_kv_cache_bytes(impl):
    assert impl.kv_cache_bytes(1000, 4, 4, 10, 2, 8) == 2 * 2 * 8 * 2 * 1000 * 4
    assert impl.kv_cache_bytes(1000, 6, 1, 100, 2, 8) == 2 * 2 * 8 * 2 * (1000 + 5 * 100)
    assert impl.kv_cache_bytes(50, 6, 1, 100, 2, 8, bytes_per=1) == 2 * 2 * 8 * 1 * 50 * 6


# ---------------------------------------------------------------- banded_attention
@pytest.mark.parametrize("window", [4, 8, 16])
def test_banded_matches_dense(impl, window):
    q, k, v = _rand(1, (2, 3, 32, 8)), _rand(2, (2, 3, 32, 8)), _rand(3, (2, 3, 32, 5))
    T = 32
    i, j = np.arange(T)[:, None], np.arange(T)[None]
    ref = _dense(q, k, v, (j <= i) & (i - j < window))
    np.testing.assert_allclose(np.asarray(impl.banded_attention(q, k, v, window)), np.asarray(ref),
                               atol=2e-5, rtol=2e-5)


# ---------------------------------------------------------------- moba_mask
def test_moba_mask_structure(impl):
    q, k = _rand(4, (2, 2, 32, 8)), _rand(5, (2, 2, 32, 8))
    m = np.asarray(impl.moba_mask(q, k, 8, 2))
    assert m.shape == (2, 2, 32, 32) and m.dtype == bool
    assert not np.triu(m, 1).any()                                       # causal
    for t in range(32):
        own = t // 8
        assert m[:, :, t, own * 8:t + 1].all()                           # own block, up to itself
        blocks = {b for b in range(own) if m[0, 0, t, b * 8]}
        assert len(blocks) == min(2, own)
        for b in blocks:
            assert m[0, 0, t, b * 8:(b + 1) * 8].all()                   # whole blocks


def test_moba_mask_picks_best_block(impl):
    T, d = 32, 4
    k = jnp.zeros((1, 1, T, d)).at[0, 0, 8:16, 0].set(5.0)              # block 1 points along e0
    q = jnp.zeros((1, 1, T, d)).at[0, 0, :, 0].set(1.0)
    m = np.asarray(impl.moba_mask(q, k, 8, 1))
    assert m[0, 0, 31, 8:16].all() and not m[0, 0, 31, :8].any() and not m[0, 0, 31, 16:24].any()


# ---------------------------------------------------------------- indexer
def test_indexer_scores(impl):
    B, T, Hi, di = 2, 5, 3, 4
    q, kk, w = _rand(6, (B, T, Hi, di)), _rand(7, (B, T, di)), _rand(8, (B, T, Hi))
    out = np.asarray(impl.indexer_scores(q, kk, w))
    qn, kn, wn = map(np.asarray, (q, kk, w))
    expect = np.zeros((B, T, T))
    for b in range(B):
        for t in range(T):
            for s in range(T):
                expect[b, t, s] = sum(wn[b, t, h] * max(0.0, qn[b, t, h] @ kn[b, s]) for h in range(Hi))
    np.testing.assert_allclose(out, expect, rtol=1e-5, atol=1e-5)


def test_topk_causal_mask(impl):
    s = _rand(9, (2, 10, 10))
    m = np.asarray(impl.topk_causal_mask(s, 3))
    sn = np.asarray(s)
    assert m.shape == (2, 10, 10) and not np.triu(m, 1).any()
    for b in range(2):
        for t in range(10):
            n = min(3, t + 1)
            assert m[b, t].sum() == n
            assert set(np.flatnonzero(m[b, t])) == set(np.argsort(-sn[b, t, :t + 1])[:n])
