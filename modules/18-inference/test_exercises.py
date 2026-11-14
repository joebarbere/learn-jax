from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_kv_cache_bytes(impl):
    smol = SimpleNamespace(n_layers=32, n_kv_heads=5, head_dim=64)
    # 2 (K,V) * 32 layers * 1 seq * 8192 pos * 5 heads * 64 dims * 2 bytes
    assert impl.kv_cache_bytes(smol, 8192, 1) == 2 * 32 * 8192 * 5 * 64 * 2
    assert impl.kv_cache_bytes(smol, 100, 4, "float32") == 2 * 32 * 4 * 100 * 5 * 64 * 4
    assert impl.kv_cache_bytes(smol, 0, 8) == 0
    assert isinstance(impl.kv_cache_bytes(smol, 10, 1), int)


def test_decode_roofline(impl):
    assert impl.decode_roofline_tok_s(1_000_000_000, 2, 288) == pytest.approx(144.0)
    assert impl.decode_roofline_tok_s(361_821_120, 2, 288) == pytest.approx(397.99, rel=1e-3)
    # halving bytes per parameter doubles the bound
    a = impl.decode_roofline_tok_s(10**8, 2, 100)
    assert impl.decode_roofline_tok_s(10**8, 1, 100) == pytest.approx(2 * a)


def test_top_k_filter(impl):
    x = jnp.array([1.0, 5.0, 3.0, 4.0, 2.0])
    out = np.asarray(impl.top_k_filter(x, 2))
    np.testing.assert_array_equal(np.isfinite(out), [False, True, False, True, False])
    np.testing.assert_allclose(out[[1, 3]], [5.0, 4.0])
    np.testing.assert_array_equal(np.asarray(impl.top_k_filter(x, 0)), np.asarray(x))
    np.testing.assert_array_equal(np.asarray(impl.top_k_filter(x, 10)), np.asarray(x))
    batch = jnp.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    np.testing.assert_array_equal(np.isfinite(np.asarray(impl.top_k_filter(batch, 1))),
                                  [[False, False, True], [True, False, False]])


def test_top_p_filter(impl):
    probs = jnp.array([0.05, 0.5, 0.3, 0.15])
    logits = jnp.log(probs)
    keep = lambda p: np.isfinite(np.asarray(impl.top_p_filter(logits, p)))
    np.testing.assert_array_equal(keep(0.4), [False, True, False, False])   # top token alone reaches 0.4
    np.testing.assert_array_equal(keep(0.7), [False, True, True, False])    # 0.5 < 0.7, so add 0.3
    np.testing.assert_array_equal(keep(0.81), [False, True, True, True])
    np.testing.assert_array_equal(keep(1e-6), [False, True, False, False])  # always keep the top token
    big = jax.random.normal(jax.random.key(2), (49152,)) * 3.0
    assert bool(jnp.all(jnp.isfinite(impl.top_p_filter(big, 1.0))))         # p = 1 keeps everything
    out = np.asarray(impl.top_p_filter(logits, 0.7))
    np.testing.assert_allclose(out[[1, 2]], np.asarray(logits)[[1, 2]], rtol=1e-6)   # kept values unchanged
    batch = jnp.stack([logits, logits[::-1]])
    np.testing.assert_array_equal(np.isfinite(np.asarray(impl.top_p_filter(batch, 0.4))),
                                  [[False, True, False, False], [False, False, True, False]])


def test_repetition_penalty(impl):
    logits = jnp.array([2.0, -1.0, 3.0, 0.5])
    out = np.asarray(impl.apply_repetition_penalty(logits, jnp.array([0, 1, 1]), 2.0))
    np.testing.assert_allclose(out, [1.0, -2.0, 3.0, 0.5])
    np.testing.assert_allclose(np.asarray(impl.apply_repetition_penalty(logits, jnp.array([0, 2]), 1.0)),
                               np.asarray(logits))


def test_int8_round_trip(impl):
    w = jax.random.normal(jax.random.key(0), (64, 32)) * jnp.linspace(0.01, 3.0, 32)
    w = w.at[:, 5].set(0.0)                                   # an all-zero column
    q = impl.quantize_int8(w)
    assert q["q"].dtype == jnp.int8 and q["q"].shape == (64, 32) and q["s"].shape == (32,)
    assert int(jnp.max(jnp.abs(q["q"]))) <= 127
    back = np.asarray(impl.dequantize_int8(q))
    assert np.all(np.isfinite(back))
    # error at most half a quantization step per column
    step = np.asarray(q["s"])
    assert np.all(np.abs(back - np.asarray(w)) <= step / 2 + 1e-6)
    np.testing.assert_allclose(back[:, 5], 0.0)


def test_int4_grouped(impl):
    w = jax.random.normal(jax.random.key(1), (128, 16))
    w = w.at[:32, 3].set(0.0)                                  # an all-zero block (group 32)
    q = impl.quantize_int4_grouped(w, 32)
    assert q["q"].shape == (128, 16) and q["s"].shape == (4, 16)
    qv = np.asarray(q["q"])
    assert qv.min() >= -8 and qv.max() <= 7
    back = np.asarray(impl.dequantize_int4_grouped(q))
    assert np.all(np.isfinite(back))
    steps = np.repeat(np.asarray(q["s"]), 32, axis=0)
    assert np.all(np.abs(back - np.asarray(w)) <= steps / 2 + 1e-6)
    # smaller groups track the weights more closely
    err = lambda g: float(jnp.mean(jnp.abs(impl.dequantize_int4_grouped(impl.quantize_int4_grouped(w, g)) - w)))
    assert err(16) < err(128)


def test_speculative_accept(impl):
    V = 10
    def logits_for(tokens):
        return jax.nn.one_hot(jnp.array(tokens), V) * 5.0
    target = logits_for([3, 7, 2, 9])                          # target's greedy choices at 4 positions
    n, nxt = impl.speculative_accept(jnp.array([3, 7, 2], jnp.int32), target)
    assert (int(n), int(nxt)) == (3, 9)                        # all accepted + bonus token
    n, nxt = impl.speculative_accept(jnp.array([3, 1, 2], jnp.int32), target)
    assert (int(n), int(nxt)) == (1, 7)                        # mismatch at 1: later match doesn't count
    n, nxt = impl.speculative_accept(jnp.array([4, 7, 2], jnp.int32), target)
    assert (int(n), int(nxt)) == (0, 3)
    n, nxt = jax.jit(impl.speculative_accept)(jnp.array([3, 7, 0], jnp.int32), target)
    assert (int(n), int(nxt)) == (2, 2)
