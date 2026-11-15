import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl


def naive_attention(q, k, v, causal=False):
    s = q @ k.T / jnp.sqrt(q.shape[-1])
    if causal:
        T = q.shape[0]
        s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -jnp.inf)
    return jax.nn.softmax(s, axis=-1) @ v


def qkv(T=64, S=64, d=16, dv=8, seed=0):
    ks = jax.random.split(jax.random.key(seed), 3)
    return (jax.random.normal(ks[0], (T, d)), jax.random.normal(ks[1], (S, d)),
            jax.random.normal(ks[2], (S, dv)))


def test_workgroup_occupancy(impl):
    # 256 threads = 8 waves = 4 per SIMD; 32 VGPRs → 32 waves by registers, capped at 16 → 4 workgroups
    assert impl.workgroup_occupancy(32, 0, 256) == {
        "fits": True, "waves_per_simd": 16, "workgroups_per_cu": 4, "limited_by": "waves"}
    # same, but 20 KB of LDS each → only 3 workgroups fit in 64 KB
    assert impl.workgroup_occupancy(32, 20000, 256) == {
        "fits": True, "waves_per_simd": 12, "workgroups_per_cu": 3, "limited_by": "lds"}
    # 1024 threads need 16 waves per SIMD; 72 VGPRs → 1024 // 72 = 14 on gfx1102: doesn't fit...
    assert impl.workgroup_occupancy(72, 16512, 1024)["fits"] is False
    assert impl.workgroup_occupancy(72, 16512, 1024)["limited_by"] == "vgprs"
    # ...but fits on gfx1100's 1536-entry register file
    r = impl.workgroup_occupancy(72, 16512, 1024, vgpr_file=1536)
    assert r["fits"] is True and r["waves_per_simd"] == 16
    # rounding up to the granule: 65 VGPRs counts as 72
    assert impl.workgroup_occupancy(65, 0, 64)["waves_per_simd"] == 14
    assert impl.workgroup_occupancy(65, 0, 64)["limited_by"] == "vgprs"
    # hard limits
    assert impl.workgroup_occupancy(32, 0, 2048)["limited_by"] == "wg_size"
    assert impl.workgroup_occupancy(32, 70000, 64)["limited_by"] == "lds"
    assert impl.workgroup_occupancy(300, 0, 64)["limited_by"] == "vgprs"


def test_attention_memory_bytes(impl):
    assert impl.attention_memory_bytes(1024, 64) == 1024 * 1024 * 4
    assert impl.attention_memory_bytes(1024, 64, block=128) == (1024 * 128 + 1024 * 66) * 4
    assert impl.attention_memory_bytes(4096, 128, dtype_bytes=2) == 4096 * 4096 * 2
    # tiled grows linearly in T, standard quadratically
    a, b = impl.attention_memory_bytes(2048, 64, 128), impl.attention_memory_bytes(4096, 64, 128)
    assert b == 2 * a


def test_online_softmax_update_streams_exactly(impl):
    q, k, v = qkv(T=8, S=40, d=4, dv=3)
    scores = q @ k.T
    expected = jax.nn.softmax(scores, axis=-1) @ v
    m, l, acc = jnp.full((8,), -jnp.inf), jnp.zeros(8), jnp.zeros((8, 3))
    for s0 in range(0, 40, 10):
        m, l, acc = impl.online_softmax_update(m, l, acc, scores[:, s0:s0 + 10], v[s0:s0 + 10])
    np.testing.assert_allclose(acc / l[:, None], expected, atol=1e-5)
    np.testing.assert_allclose(m, scores.max(-1), atol=1e-6)


def test_online_softmax_update_fully_masked_block(impl):
    m0, l0, acc0 = jnp.full((2,), -jnp.inf), jnp.zeros(2), jnp.zeros((2, 3))
    masked = jnp.full((2, 4), -jnp.inf)
    m, l, acc = impl.online_softmax_update(m0, l0, acc0, masked, jnp.ones((4, 3)))
    assert not np.any(np.isnan(np.asarray(acc))) and not np.any(np.isnan(np.asarray(l)))
    np.testing.assert_allclose(l, 0.0)
    # a real block afterwards still gives the right answer
    s = jnp.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
    vv = jnp.arange(12.0).reshape(4, 3)
    m, l, acc = impl.online_softmax_update(m, l, acc, s, vv)
    np.testing.assert_allclose(acc / l[:, None], jax.nn.softmax(s, -1) @ vv, atol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_matches_standard(impl, causal):
    q, k, v = qkv(T=64, S=64)
    out = impl.flash_attention(q, k, v, block=16, causal=causal)
    np.testing.assert_allclose(out, naive_attention(q, k, v, causal), atol=1e-5)
    # block size doesn't change the answer, and it works under jit
    f = jax.jit(lambda q, k, v: impl.flash_attention(q, k, v, block=32, causal=causal))
    np.testing.assert_allclose(f(q, k, v), out, atol=1e-5)


def test_flash_attention_rectangular_and_errors(impl):
    q, k, v = qkv(T=10, S=48, dv=5)
    np.testing.assert_allclose(impl.flash_attention(q, k, v, block=12), naive_attention(q, k, v), atol=1e-5)
    with pytest.raises(ValueError):
        impl.flash_attention(q, k, v, block=10)


def test_pallas_kernels_interpret(impl):
    x = jax.random.normal(jax.random.key(1), (16, 128))
    y = jax.random.normal(jax.random.key(2), (16, 128))
    out = pl.pallas_call(impl.add_relu_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                         interpret=True)(x, y)
    np.testing.assert_allclose(out, jnp.maximum(x + y, 0.0), atol=1e-6)
    sm = pl.pallas_call(impl.row_softmax_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                        interpret=True)(x * 30.0)   # large values: must be numerically stable
    np.testing.assert_allclose(sm, jax.nn.softmax(x * 30.0, -1), atol=1e-6)
    assert not np.any(np.isnan(np.asarray(sm)))
