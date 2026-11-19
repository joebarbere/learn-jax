"""Module 41 exercises: linear attention, Mamba-2's chunked algorithm, and the delta rule.

Replace each `raise NotImplementedError` with your code, then run `make test N=41`. The lesson's language
models (models.py) call your kernels, so later cells come alive as you finish these.

Conventions: arrays are head-major. q and k are (..., T, dk), v is (..., T, dv), where "..." is any
leading batch and head axes, e.g. (B, H). Per-step scalars (a decay, a write strength) are (..., T). The
state S is (..., dk, dv): it maps a key to a value, so reading it with a query q gives qᵀS, a (dv,) vector.
No softmax, no 1/√d and no normalizing denominator anywhere here: the layers in models.py scale q and k
before calling you.
"""
import jax  # noqa: F401  (for your code)
import jax.numpy as jnp  # noqa: F401


# ---------------------------------------------------------------- memory
def memory_costs(n_layers: int, attn_every: int, n_heads: int, d_head: int, context: int,
                 kv_bytes: int = 2, state_bytes: int = 4) -> dict:
    """What a model must keep in memory per sequence while generating, after `context` tokens.

    attn_every: 0 = no attention layers at all; 1 = every layer is attention; r > 1 = one attention layer
    in every r (so n_layers // r of them), the rest are linear-attention layers. Every attention layer
    caches a key and a value of n_heads × d_head numbers per token (kv_bytes each). Every linear layer
    keeps one state of n_heads × d_head × d_head numbers (state_bytes each), whatever the context.

    Return a dict of ints: "attn_layers", "linear_layers", "kv_cache_bytes", "state_bytes", "total_bytes".
    """
    raise NotImplementedError


# ---------------------------------------------------------------- linear attention, two ways
def causal_linear_attention(q, k, v):
    """The parallel ("attention") form: out_t = Σ_{s ≤ t} (q_t · k_s) v_s.

    Like Module 15's causal attention with the softmax removed: scores Q Kᵀ (..., T, T), zero above the
    diagonal (not −∞: there's no softmax), times V. Returns (..., T, dv).
    """
    raise NotImplementedError


def linear_attention_scan(q, k, v, log_a=None):
    """The recurrent ("RNN") form, with an optional per-step decay a_t = exp(log_a_t):

        S_t = a_t S_{t−1} + k_t v_tᵀ          (S_0 = 0; S_t is (..., dk, dv))
        out_t = q_tᵀ S_t

    log_a: (..., T) or None (None = no decay, a_t = 1). Use jax.lax.scan over time (move T to the front).
    Returns (..., T, dv). With log_a=None this must equal causal_linear_attention.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- decay: Mamba-2's chunked algorithm
def segsum(log_a):
    """The decay between every pair of steps, in log space: L[..., i, j] = Σ_{j < m ≤ i} log_a[..., m] for
    i ≥ j (so 0 on the diagonal) and −inf above the diagonal. exp(L) is then the matrix of decays a_{j+1}···a_i
    that turns decayed linear attention into masked attention: out = (Q Kᵀ ⊙ exp(L)) V.

    log_a: (..., T) → (..., T, T). Hint: with c = cumsum(log_a), the sum from j+1 to i is c_i − c_j.
    """
    raise NotImplementedError


def chunked_linear_attention(q, k, v, log_a, chunk: int):
    """Decayed linear attention (the scan above with log_a), computed chunk by chunk, as Mamba-2's SSD does.

    Split the T steps into T // chunk chunks (T is a multiple of chunk). Then, with c = the cumulative sum of
    log_a *inside each chunk*:
      1. inside every chunk at once: y = (Q Kᵀ ⊙ exp(segsum(log_a))) V          (a matmul per chunk)
      2. each chunk's own contribution to the state at its end: Σ_j exp(c_last − c_j) k_j v_jᵀ
      3. a short scan over chunks: S_in[0] = 0, S_in[n+1] = exp(c_last of chunk n) S_in[n] + (2.) of chunk n
      4. add what each step reads from the state that came in: y_i += exp(c_i) q_iᵀ S_in
    Returns (..., T, dv), equal to linear_attention_scan(q, k, v, log_a).
    """
    raise NotImplementedError


# ---------------------------------------------------------------- the delta rule
def gated_delta_scan(q, k, v, beta, log_a=None):
    """The gated delta rule (Gated DeltaNet), step by step:

        S ← a_t S                               forget a little (a_t = exp(log_a_t); no decay if log_a is None)
        S ← S + β_t k_t (v_t − kₜᵀ S)ᵀ          write only the *error*: what v_t adds to what S already says
        out_t = q_tᵀ S

    beta: (..., T), the write strength in (0, 1). The keys arrive unit-length. Returns (..., T, dv).
    """
    raise NotImplementedError
