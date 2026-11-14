import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(17)


def _rand(i, shape):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32)


def _to_halves(x):
    """Interleaved pairs (0,1),(2,3),... -> halves layout: evens then odds."""
    return jnp.concatenate([x[..., 0::2], x[..., 1::2]], axis=-1)


def test_rms_norm(impl):
    x = _rand(0, (3, 5, 16))
    w = _rand(1, (16,))
    ref = x / np.sqrt(np.mean(np.asarray(x) ** 2, -1, keepdims=True) + 1e-6) * w
    np.testing.assert_allclose(impl.rms_norm(x, w), ref, rtol=1e-5, atol=1e-6)
    out = impl.rms_norm(x, jnp.ones(16))
    np.testing.assert_allclose(jnp.sqrt(jnp.mean(out**2, -1)), 1.0, rtol=1e-4)  # unit RMS


def test_rope_complex(impl):
    x = _rand(2, (2, 6, 3, 8))
    pos = jnp.arange(6)
    out = impl.rope_complex(x, pos)
    assert out.shape == x.shape and out.dtype == x.dtype
    np.testing.assert_allclose(out[:, 0], x[:, 0], atol=1e-6)            # position 0: no rotation
    np.testing.assert_allclose(jnp.linalg.norm(out, axis=-1), jnp.linalg.norm(x, axis=-1), rtol=1e-5)
    # pair 0 rotates by angle p (θ_0 = 1): check position 3 by hand
    a, b = x[0, 3, 0, 0], x[0, 3, 0, 1]
    np.testing.assert_allclose([out[0, 3, 0, 0], out[0, 3, 0, 1]],
                               [a * np.cos(3) - b * np.sin(3), a * np.sin(3) + b * np.cos(3)], rtol=1e-5)


def test_rope_rotate_half_and_equivalence(impl):
    x = _rand(3, (2, 7, 4, 16))
    pos = jnp.arange(7) + 5
    half = impl.rope_rotate_half(x, pos, 500000.0)
    assert half.shape == x.shape
    np.testing.assert_allclose(jnp.linalg.norm(half, axis=-1), jnp.linalg.norm(x, axis=-1), rtol=1e-5)
    # the two conventions differ only by which features are paired up
    cplx = impl.rope_complex(x, pos, 500000.0)
    np.testing.assert_allclose(impl.rope_rotate_half(_to_halves(x), pos, 500000.0), _to_halves(cplx),
                               atol=1e-5)
    # also works on (B, T, D) without a head axis
    assert impl.rope_rotate_half(x[:, :, 0], pos).shape == (2, 7, 16)


def test_rope_relative_position(impl):
    q, k = _rand(4, (1, 1, 1, 32)), _rand(5, (1, 1, 1, 32))

    def score(m, n):
        qm = impl.rope_rotate_half(q, jnp.array([m]))
        kn = impl.rope_rotate_half(k, jnp.array([n]))
        return float(jnp.sum(qm * kn))

    assert score(10, 7) == pytest.approx(score(103, 100), rel=1e-4, abs=1e-4)  # same offset, same score


def test_swiglu(impl):
    d, h = 8, 12
    p = {"w1": _rand(6, (d, h)), "w3": _rand(7, (d, h)), "w2": _rand(8, (h, d))}
    x = _rand(9, (4, d))
    ref = (jax.nn.silu(x @ p["w1"]) * (x @ p["w3"])) @ p["w2"]
    np.testing.assert_allclose(impl.swiglu(p, x), ref, rtol=1e-4, atol=1e-5)
    swapped = (jax.nn.silu(x @ p["w3"]) * (x @ p["w1"])) @ p["w2"]
    assert not np.allclose(impl.swiglu(p, x), swapped, atol=1e-3)            # naming matters


def test_repeat_kv(impl):
    x = jnp.arange(2 * 3 * 2 * 4, dtype=jnp.float32).reshape(2, 3, 2, 4)
    r = impl.repeat_kv(x, 3)
    assert r.shape == (2, 3, 6, 4)
    for h in range(6):
        np.testing.assert_array_equal(r[:, :, h], x[:, :, h // 3])
    assert impl.repeat_kv(x, 1) is x or np.array_equal(impl.repeat_kv(x, 1), x)


def _gqa_setup(n_heads=4, n_kv=2, D=8, d=16):
    cfg = {"n_heads": n_heads, "n_kv_heads": n_kv, "head_dim": D, "rope_theta": 10000.0}
    p = {"wq": _rand(10, (d, n_heads * D)) * 0.3, "wk": _rand(11, (d, n_kv * D)) * 0.3,
         "wv": _rand(12, (d, n_kv * D)) * 0.3, "wo": _rand(13, (n_heads * D, d)) * 0.3}
    return cfg, p


def test_gqa_attention_matches_mha_with_shared_heads(impl):
    cfg, p = _gqa_setup()
    x = _rand(14, (2, 9, 16))
    pos = jnp.arange(9)
    out, cache = impl.gqa_attention(p, x, pos, cfg)
    assert out.shape == (2, 9, 16)
    assert cache["k"].shape == (2, 9, 2, 8) and cache["v"].shape == (2, 9, 2, 8)
    # GQA == MHA whose k/v weights repeat each kv head's columns
    D = 8
    rep = lambda w: jnp.concatenate([w[:, (h // 2) * D:(h // 2 + 1) * D] for h in range(4)], axis=1)
    p_mha = dict(p, wk=rep(p["wk"]), wv=rep(p["wv"]))
    out_mha, _ = impl.gqa_attention(p_mha, x, pos, dict(cfg, n_kv_heads=4))
    np.testing.assert_allclose(out, out_mha, rtol=1e-4, atol=1e-5)


def test_gqa_attention_is_causal(impl):
    cfg, p = _gqa_setup()
    x = _rand(15, (1, 8, 16))
    x2 = x.at[:, 5:].set(_rand(16, (1, 3, 16)))      # change only the future
    a, _ = impl.gqa_attention(p, x, jnp.arange(8), cfg)
    b, _ = impl.gqa_attention(p, x2, jnp.arange(8), cfg)
    np.testing.assert_allclose(a[:, :5], b[:, :5], rtol=1e-5, atol=1e-6)
    assert not np.allclose(a[:, 5:], b[:, 5:])


def test_gqa_attention_cache_matches_full(impl):
    cfg, p = _gqa_setup()
    x = _rand(17, (2, 10, 16))
    full, _ = impl.gqa_attention(p, x, jnp.arange(10), cfg)
    out, cache = impl.gqa_attention(p, x[:, :6], jnp.arange(6), cfg)        # prefill
    outs = [out]
    for t in range(6, 10):                                                  # decode one token at a time
        o, cache = impl.gqa_attention(p, x[:, t:t + 1], jnp.array([t]), cfg, cache)
        outs.append(o)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, rtol=1e-4, atol=1e-5)
    assert cache["k"].shape == (2, 10, 2, 8)


def test_fix_buggy_cached_attention(impl):
    B, L, H, D = 2, 9, 3, 8
    q, k, v = _rand(18, (B, L, H, D)), _rand(19, (B, L, H, D)), _rand(20, (B, L, H, D))
    empty = jnp.zeros((B, 0, H, D))
    full, _ = impl.fix_buggy_cached_attention(q, k, v, empty, empty)       # all at once, no cache
    allowed = jnp.tril(jnp.ones((L, L), bool))
    s = jnp.where(allowed, jnp.einsum("bthd,bshd->bhts", q, k) / np.sqrt(D), -jnp.inf)
    ref = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(s, -1), v)
    np.testing.assert_allclose(full, ref, rtol=1e-5, atol=1e-6)
    ck, cv = k[:, :4], v[:, :4]                                             # prefill 4, then decode
    outs = [full[:, :4]]
    for t in range(4, L):
        o, (ck, cv) = impl.fix_buggy_cached_attention(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], ck, cv)
        outs.append(o)
    np.testing.assert_allclose(jnp.concatenate(outs, 1), full, rtol=1e-5, atol=1e-6)
    assert ck.shape == (B, L, H, D)


def test_moe_route(impl):
    scores = jnp.array([[0.9, 0.1, 0.8, 0.2], [0.3, 0.6, 0.5, 0.4]])
    g = impl.moe_route(scores, jnp.zeros(4), 2)
    np.testing.assert_allclose(g.sum(-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(g[0], [0.9 / 1.7, 0, 0.8 / 1.7, 0], rtol=1e-5)
    # bias changes WHO is picked, but gates still use the raw scores
    g2 = impl.moe_route(scores, jnp.array([0.0, 1.0, 0.0, 0.0]), 2)
    np.testing.assert_allclose(g2[0], [0.9 / 1.0, 0.1 / 1.0, 0, 0], rtol=1e-5)
    # no gradient flows into the bias
    gb = jax.grad(lambda b: impl.moe_route(scores, b, 2).sum())(jnp.zeros(4))
    np.testing.assert_array_equal(gb, 0.0)


def test_update_bias(impl):
    b = impl.update_bias(jnp.zeros(4), jnp.array([0.4, 0.1, 0.25, 0.25]), 0.01)
    np.testing.assert_allclose(b, [-0.01, 0.01, 0.0, 0.0], atol=1e-7)
    b2 = impl.update_bias(jnp.zeros((2, 3)), jnp.array([[0.5, 0.25, 0.25], [0.2, 0.2, 0.6]]), 0.1)
    assert b2.shape == (2, 3)
    np.testing.assert_allclose(b2[1], [0.1, 0.1, -0.1], atol=1e-6)


def test_cache_floats(impl):
    assert impl.mha_cache_floats(6, 32 + 16, 32) == 480       # deepseek-v3-jax without MLA
    assert impl.mha_cache_floats(8, 128, 128) == 2048         # Llama-3-8B (GQA, 8 KV heads)
    assert impl.mla_cache_floats({"kv_rank": 128, "d_rope": 16}) == 144
    assert impl.mla_cache_floats({"kv_rank": 512, "d_rope": 64, "n_heads": 128}) == 576   # DeepSeek-V3
