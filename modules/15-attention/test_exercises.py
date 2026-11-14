import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _params(key, D):
    ks = jax.random.split(key, 4)
    return {n: jax.random.normal(k, (D, D)) / np.sqrt(D) for n, k in zip(("wq", "wk", "wv", "wo"), ks)}


def test_causal_mask(impl):
    m = np.asarray(impl.causal_mask(4))
    assert m.dtype == bool and m.shape == (4, 4)
    np.testing.assert_array_equal(m, np.tril(np.ones((4, 4), bool)))
    assert np.asarray(impl.causal_mask(1)).tolist() == [[True]]


def test_scaled_dot_product_matches_formula(impl):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    q, k, v = (jax.random.normal(kk, (2, 5, 8)) for kk in (k1, k2, k3))
    out, w = impl.scaled_dot_product(q, k, v)
    ref_w = jax.nn.softmax(q @ jnp.swapaxes(k, -1, -2) / np.sqrt(8), axis=-1)
    np.testing.assert_allclose(w, ref_w, atol=1e-5)
    np.testing.assert_allclose(out, ref_w @ v, atol=1e-5)
    np.testing.assert_allclose(np.asarray(w).sum(-1), 1.0, atol=1e-5)


def test_scaled_dot_product_mask_gives_zero_weight(impl):
    k1, k2 = jax.random.split(jax.random.key(1))
    q = jax.random.normal(k1, (4, 6))
    kv = jax.random.normal(k2, (4, 6))
    mask = np.tril(np.ones((4, 4), bool))
    out, w = impl.scaled_dot_product(q, kv, kv, mask)
    w = np.asarray(w)
    assert np.all(w[~mask] == 0.0)  # exactly zero, not just small
    np.testing.assert_allclose(np.asarray(out)[0], np.asarray(kv)[0], atol=1e-5)  # first row sees only itself
    # different value width
    _, w2 = impl.scaled_dot_product(q, kv, jnp.ones((4, 3)), mask)
    assert np.asarray(w2).shape == (4, 4)


def test_split_merge_heads(impl):
    x = jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8)
    h = impl.split_heads(x, 4)
    assert h.shape == (2, 3, 4, 2)
    np.testing.assert_array_equal(h[0, 0, 1], x[0, 0, 2:4])  # head 1 owns features 2..3
    np.testing.assert_array_equal(impl.merge_heads(h), x)
    with pytest.raises(ValueError):
        impl.split_heads(x, 3)


def test_attention_weights(impl):
    key = jax.random.key(2)
    p = _params(key, 16)
    x = jax.random.normal(jax.random.key(3), (2, 7, 16))
    w = np.asarray(impl.attention_weights(p, x, 4, causal=True))
    assert w.shape == (2, 4, 7, 7)
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-5)
    upper = np.triu(np.ones((7, 7), bool), k=1)
    assert np.all(w[..., upper] == 0.0)
    wf = np.asarray(impl.attention_weights(p, x, 4, causal=False))
    assert np.all(wf[..., upper] > 0.0)


def test_multi_head_attention_matches_jax(impl):
    D, H = 32, 4
    p = _params(jax.random.key(4), D)
    x = jax.random.normal(jax.random.key(5), (2, 9, D))
    for causal in (True, False):
        got = impl.multi_head_attention(p, x, H, causal=causal)
        q, k, v = (impl.split_heads(x @ p[n], H) for n in ("wq", "wk", "wv"))
        ref = jax.nn.dot_product_attention(q, k, v, is_causal=causal)
        ref = ref.reshape(2, 9, D) @ p["wo"]
        np.testing.assert_allclose(got, ref, atol=1e-4)


def test_causal_output_ignores_future(impl):
    D = 16
    p = _params(jax.random.key(6), D)
    x = jax.random.normal(jax.random.key(7), (1, 6, D))
    x2 = x.at[:, 4:].set(99.0)  # change only the future of position 3
    a = impl.multi_head_attention(p, x, 2, causal=True)
    b = impl.multi_head_attention(p, x2, 2, causal=True)
    np.testing.assert_allclose(a[:, :4], b[:, :4], atol=1e-5)


def test_sinusoidal_positions(impl):
    pe = np.asarray(impl.sinusoidal_positions(50, 16))
    assert pe.shape == (50, 16) and pe.dtype == np.float32
    np.testing.assert_allclose(pe[0, 0::2], 0.0, atol=1e-6)
    np.testing.assert_allclose(pe[0, 1::2], 1.0, atol=1e-6)
    np.testing.assert_allclose(pe[7, 0], np.sin(7.0), atol=1e-5)
    np.testing.assert_allclose(pe[7, 3], np.cos(7.0 / 10000 ** (2 / 16)), atol=1e-5)
    assert np.abs(pe).max() <= 1.0 + 1e-6


def test_permutation_equivariance(impl):
    D = 16
    p = _params(jax.random.key(8), D)
    x = jax.random.normal(jax.random.key(9), (2, 6, D))
    perm = [3, 0, 5, 1, 4, 2]
    full = lambda z: impl.multi_head_attention(p, z, 4, causal=False)
    assert impl.is_permutation_equivariant(full, x, perm) is True
    causal = lambda z: impl.multi_head_attention(p, z, 4, causal=True)
    assert impl.is_permutation_equivariant(causal, x, perm) is False
    pe = impl.sinusoidal_positions(6, D)
    with_pos = lambda z: impl.multi_head_attention(p, z + pe, 4, causal=False)
    assert impl.is_permutation_equivariant(with_pos, x, perm) is False
    assert impl.is_permutation_equivariant(full, x, list(range(6))) is True
