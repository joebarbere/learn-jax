"""Module 44 exercises: attention that reads only part of the past.

Replace each `raise NotImplementedError` with your code, then run `make test N=44`. models.py builds the
race's attention layers from your masks, selections and indexer.

Conventions: q, k, v are head-major, (B, H, T, d). A mask is boolean, True = "this query may read this key",
shaped (T, T) or broadcastable to (B, H, T, T); rows are queries i, columns keys j. Everything is causal:
j ≤ i always.
"""
import jax  # noqa: F401  (for your code)
import jax.numpy as jnp  # noqa: F401


def window_mask(T: int, window, n_sink=0):
    """Sliding-window attention with optional attention sinks: query i may read key j when j ≤ i and either
    i − j < window (the last `window` tokens, itself included) or j < n_sink (the first n_sink tokens of the
    sequence, StreamingLLM's sinks). A (T, T) bool array.

    `window` and `n_sink` may be traced integers inside jit/scan (models.py passes one per layer), so build
    the mask with array comparisons, not Python `if`s. window ≥ T gives ordinary causal attention.
    """
    raise NotImplementedError


def kv_cache_bytes(context: int, n_layers: int, n_global: int, window: int, n_kv_heads: int, d_head: int,
                   bytes_per: int = 2) -> int:
    """Bytes of KV cache for one sequence of `context` tokens. Global layers keep every token; the other
    n_layers − n_global local (sliding-window) layers keep only the last min(context, window). Each kept
    token costs a key and a value: 2 × n_kv_heads × d_head × bytes_per bytes per layer.
    """
    raise NotImplementedError


def banded_attention(q, k, v, window: int):
    """Sliding-window causal attention that costs O(T × window), not O(T²). Same result as dense softmax
    attention under window_mask(T, window) (scale 1/√d, softmax in float32).

    Cut the sequence into T // window blocks of `window` queries (assume T divisible by window). Any key a
    query in block n may read lies in block n or block n − 1, so attend from each query block to the 2·window
    keys [block n−1 | block n] only, masking the ones outside the window or in the future. Block 0 has no
    previous block (pad with anything, but mask it). Returns (B, H, T, dv).
    """
    raise NotImplementedError


def moba_mask(q, k, block: int, top: int):
    """MoBA (Lu et al. 2025): each query reads its own block (causally) plus the `top` earlier blocks whose
    *mean key* scores highest against it.

    1. Mean-pool k over each block of `block` keys: (B, H, T // block, d).
    2. Score every query against every block mean (dot product). Only blocks strictly before the query's
       own block are candidates.
    3. Keep the `top` highest-scoring candidates (fewer exist for early queries: keep what exists).
    4. The mask: query i reads key j if j ≤ i and (j is in i's block or j's block was kept for i).
    Returns (B, H, T, T) bool. Assume T divisible by block.
    """
    raise NotImplementedError


def indexer_scores(q_idx, k_idx, w):
    """DeepSeek Sparse Attention's "lightning indexer": a cheap scorer that decides which past tokens each
    query reads. Index score of query t for key s:

        I[t, s] = Σ_h  w[t, h] · ReLU(q_idx[t, h] · k_idx[s])

    q_idx (B, T, Hi, di): a few small indexer heads per query; k_idx (B, T, di): one indexer key per token,
    shared by the heads; w (B, T, Hi): per-query head weights. Returns (B, T, T). No causal masking here.
    """
    raise NotImplementedError


def topk_causal_mask(scores, k: int):
    """For each query t, select the k highest-scoring keys among s ≤ t (all of them while t + 1 ≤ k).
    scores (..., T, T) → bool mask of the same shape, True where selected. Use jax.lax.top_k.
    """
    raise NotImplementedError
