"""Module 23 exercises: kernels, occupancy, and FlashAttention's streaming softmax.

Replace each `raise NotImplementedError` with your code, then run `make test N=23`.
The lesson calls these functions, so its "Your turn" cells come alive as you finish them.

Attention shapes here are for ONE head (the lesson vmaps over batch and heads):
  q: (T, d)   k: (S, d)   v: (S, dv)   scores: (T, S)
"""
import math  # noqa: F401

import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401

# A simplified model of one RDNA3 compute unit (CU), as used by the lesson. Real hardware has more
# rules (WGP mode, allocation granules per wave, scalar registers), but these capture the big three.
WAVE = 32                 # threads per wavefront (wave32)
SIMDS_PER_CU = 2          # each CU has two SIMD32 units; a workgroup's waves are split across them
MAX_WAVES_PER_SIMD = 16   # hardware wave slots per SIMD
MAX_WORKGROUP = 1024      # max threads in one workgroup
LDS_PER_CU = 65536        # local data share (shared memory) available to workgroups on a CU, bytes
MAX_VGPRS = 256           # max vector registers one thread may use
VGPR_GRANULE = 8          # VGPRs are allocated in blocks of 8


def workgroup_occupancy(vgprs_per_thread: int, lds_bytes: int, wg_size: int, vgpr_file: int = 1024) -> dict:
    """Can one workgroup launch on a CU, and how many waves per SIMD can be resident?

    `vgpr_file` is the number of VGPRs per SIMD lane shared by all resident waves: 1024 on gfx1102
    (RX 7600 XT), 1536 on gfx1100 (RX 7900). Model, in order:

      1. Reject (fits=False) if wg_size is not in 1..MAX_WORKGROUP → limited_by "wg_size";
         else if lds_bytes > LDS_PER_CU → "lds"; else if vgprs_per_thread > MAX_VGPRS → "vgprs".
      2. Round vgprs_per_thread UP to a multiple of VGPR_GRANULE.
      3. waves in the workgroup = ceil(wg_size / WAVE); waves it needs per SIMD = ceil(that / SIMDS_PER_CU).
      4. waves a SIMD can hold by registers = min(MAX_WAVES_PER_SIMD, vgpr_file // rounded_vgprs).
         If the workgroup needs more per SIMD than that → fits=False, limited_by "vgprs".
      5. workgroups per CU by registers = (step 4) // (waves needed per SIMD);
         by LDS = LDS_PER_CU // lds_bytes (if lds_bytes == 0, LDS doesn't limit: use the register count).
         workgroups_per_cu = the smaller; waves_per_simd = workgroups_per_cu * (waves needed per SIMD).
      6. limited_by: "lds" if the LDS count is strictly smaller than the register count; otherwise
         "vgprs" if step 4 was below MAX_WAVES_PER_SIMD; otherwise "waves".

    Returns {"fits": bool, "waves_per_simd": int, "workgroups_per_cu": int, "limited_by": str};
    when fits is False, both counts are 0.
    """
    raise NotImplementedError


def attention_memory_bytes(T: int, d: int, block: int | None = None, dtype_bytes: int = 4) -> int:
    """Working memory for one head's attention scores, in bytes (a simple model, f32 by default).

    block is None  → standard attention materializes the full (T, T) score matrix: T*T*dtype_bytes.
    block = B      → tiled (Flash) attention keeps one (T, B) score tile plus the running state for every
                     query row — max m, denominator l and accumulator acc (d values): T*B + T*(d+2) values.
    """
    raise NotImplementedError


def online_softmax_update(m, l, acc, scores_block, v_block):
    """One step of the streaming softmax at the heart of FlashAttention. Returns (m_new, l_new, acc_new).

    State per query row (start: m = -inf, l = 0, acc = 0):
      m   (T,)     the largest score seen so far
      l   (T,)     sum of exp(score - m) over keys seen so far
      acc (T, dv)  sum of exp(score - m) * v over keys seen so far
    New block: scores_block (T, B) and its values v_block (B, dv).

      m_new = max(m, rowmax(scores_block))
      p     = exp(scores_block - m_new)            (B values per row)
      corr  = exp(m - m_new)                       rescales the old sums to the new max
      l_new = l * corr + rowsum(p);   acc_new = acc * corr + p @ v_block

    After the last block, attention output = acc / l. Edge case: a row whose scores are all -inf so far
    (fully masked) must stay at m = -inf, l = 0, acc = 0 with no NaNs — guard the exp(-inf - (-inf)).
    """
    raise NotImplementedError


def flash_attention(q, k, v, block: int, causal: bool = False):
    """Exact softmax attention computed block by block with `jax.lax.scan` and online_softmax_update.

    q (T, d), k (S, d), v (S, dv) → (T, dv), equal to softmax(q kᵀ / √d) v up to float rounding.
    Split k and v into S // block blocks and scan over them, never building the (T, S) score matrix.
    causal=True: query t may attend to key s only if s <= t (assume T == S then).
    Raise ValueError if S is not a multiple of block.
    """
    raise NotImplementedError


def add_relu_kernel(x_ref, y_ref, o_ref):
    """A Pallas kernel body: o = relu(x + y), elementwise.

    Read whole blocks with `x_ref[...]`, write with `o_ref[...] = ...`. The tests run it with
    `pl.pallas_call(add_relu_kernel, out_shape=..., interpret=True)`, which works on any backend.
    """
    raise NotImplementedError


def row_softmax_kernel(x_ref, o_ref):
    """A Pallas kernel body: numerically stable softmax over the last axis of the block.

    Subtract the row max before exp (Module 04), then divide by the row sum.
    """
    raise NotImplementedError
