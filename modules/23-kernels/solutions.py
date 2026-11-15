"""Module 23 reference solutions. Try the exercises first — `make test N=23` grades yours."""
import math

import jax
import jax.numpy as jnp

WAVE = 32
SIMDS_PER_CU = 2
MAX_WAVES_PER_SIMD = 16
MAX_WORKGROUP = 1024
LDS_PER_CU = 65536
MAX_VGPRS = 256
VGPR_GRANULE = 8


def workgroup_occupancy(vgprs_per_thread: int, lds_bytes: int, wg_size: int, vgpr_file: int = 1024) -> dict:
    if wg_size < 1 or wg_size > MAX_WORKGROUP:
        return {"fits": False, "waves_per_simd": 0, "workgroups_per_cu": 0, "limited_by": "wg_size"}
    if lds_bytes > LDS_PER_CU:
        return {"fits": False, "waves_per_simd": 0, "workgroups_per_cu": 0, "limited_by": "lds"}
    if vgprs_per_thread > MAX_VGPRS:
        return {"fits": False, "waves_per_simd": 0, "workgroups_per_cu": 0, "limited_by": "vgprs"}
    vgprs = math.ceil(vgprs_per_thread / VGPR_GRANULE) * VGPR_GRANULE
    per_simd = math.ceil(math.ceil(wg_size / WAVE) / SIMDS_PER_CU)
    vgpr_waves = min(MAX_WAVES_PER_SIMD, vgpr_file // vgprs)
    if per_simd > vgpr_waves:
        return {"fits": False, "waves_per_simd": 0, "workgroups_per_cu": 0, "limited_by": "vgprs"}
    by_vgpr = vgpr_waves // per_simd
    by_lds = LDS_PER_CU // lds_bytes if lds_bytes > 0 else by_vgpr
    wgs = min(by_vgpr, by_lds)
    if by_lds < by_vgpr:
        limit = "lds"
    elif vgpr_waves < MAX_WAVES_PER_SIMD:
        limit = "vgprs"
    else:
        limit = "waves"
    return {"fits": True, "waves_per_simd": wgs * per_simd, "workgroups_per_cu": wgs, "limited_by": limit}


def attention_memory_bytes(T: int, d: int, block: int | None = None, dtype_bytes: int = 4) -> int:
    if block is None:
        return T * T * dtype_bytes
    return T * block * dtype_bytes + T * (d + 2) * dtype_bytes


def online_softmax_update(m, l, acc, scores_block, v_block):
    m_new = jnp.maximum(m, jnp.max(scores_block, axis=-1))
    m_safe = jnp.where(jnp.isfinite(m_new), m_new, 0.0)
    p = jnp.exp(scores_block - m_safe[:, None])
    corr = jnp.exp(jnp.where(jnp.isfinite(m), m - m_safe, -jnp.inf))
    l_new = l * corr + jnp.sum(p, axis=-1)
    acc_new = acc * corr[:, None] + p @ v_block
    return m_new, l_new, acc_new


def flash_attention(q, k, v, block: int, causal: bool = False):
    T, d = q.shape
    S = k.shape[0]
    if S % block:
        raise ValueError(f"key length {S} is not a multiple of block {block}")
    nb = S // block
    kb = k.reshape(nb, block, d)
    vb = v.reshape(nb, block, v.shape[-1])
    starts = jnp.arange(nb) * block
    q_pos = jnp.arange(T)
    scale = 1.0 / jnp.sqrt(jnp.asarray(d, q.dtype))

    def step(state, blk):
        kk, vv, s0 = blk
        s = (q @ kk.T) * scale
        if causal:
            k_pos = s0 + jnp.arange(block)
            s = jnp.where(k_pos[None, :] <= q_pos[:, None], s, -jnp.inf)
        return online_softmax_update(*state, s, vv), None

    init = (jnp.full((T,), -jnp.inf, q.dtype), jnp.zeros((T,), q.dtype), jnp.zeros((T, v.shape[-1]), q.dtype))
    (_, l, acc), _ = jax.lax.scan(step, init, (kb, vb, starts))
    return acc / l[:, None]


def add_relu_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = jnp.maximum(x_ref[...] + y_ref[...], 0.0)


def row_softmax_kernel(x_ref, o_ref):
    x = x_ref[...]
    e = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    o_ref[...] = e / jnp.sum(e, axis=-1, keepdims=True)
