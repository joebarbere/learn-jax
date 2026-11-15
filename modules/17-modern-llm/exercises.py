"""Module 17 exercises: the parts of a modern LLM, built one at a time.

Replace each `raise NotImplementedError` with your code, then run `make test N=17`.
The lesson assembles your parts into a Llama-style and a DeepSeek-style model and trains both, so the
later cells come alive as you finish these.

Layout conventions (the same as Module 15 and deepseek-v3-jax):
  B = batch, T = query positions, S = key/value positions, H = query heads, Hkv = key/value heads,
  D = width of one head. Activations are (B, T, d_model); split into heads they are (B, T, H, D).
"""
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------- normalization
def rms_norm(x, w, eps: float = 1e-6):
    """RMSNorm over the last axis: x / sqrt(mean(x²) + eps) * w.   x: (..., d), w: (d,).

    (Same as Module 11 — copy yours over if you like.)
    """
    raise NotImplementedError


# ---------------------------------------------------------------- RoPE, two ways
def rope_complex(x, positions, theta: float = 10000.0):
    """Rotary position embedding, *interleaved* convention, via complex numbers.

    x: (B, T, ..., D) with T on axis 1 and D even (e.g. (B, T, H, D)); positions: (T,) integers.
    Pair up neighbouring features (x[..., 0], x[..., 1]), (x[..., 2], x[..., 3]), … and treat each pair
    as one complex number a + ib. Pair i at position p is rotated by the angle p · θ_i, where
    θ_i = theta ** (-2i / D) for i = 0 … D/2-1 — i.e. multiply it by exp(i · p · θ_i).
    Write the rotated pairs back in the same interleaved order. Same shape and dtype as x.

    Hint: jax.lax.complex(a, b) builds a + ib; .real and .imag take it apart; jnp.stack(..., -1) then a
    reshape re-interleaves.
    """
    raise NotImplementedError


def rope_rotate_half(x, positions, theta: float = 10000.0):
    """Rotary position embedding, *halves* convention, with cos/sin only (deepseek-v3-jax's `apply_rope`).

    Same inputs and frequencies θ_i as rope_complex, but pair i is (x[..., i], x[..., i + D/2]) — the first
    half of the features with the second half. With x1, x2 = the two halves, c = cos(p·θ), s = sin(p·θ):
        out = concatenate([x1·c − x2·s,  x1·s + x2·c], axis=-1)
    """
    raise NotImplementedError


def rope_partial(x, positions, rotary_dim: int, theta: float = 10000.0):
    """Stretch — *partial* RoPE (GPT-NeoX, Phi, GLM, Qwen3-Next): rotate only some of each head's features.

    Rotate the first `rotary_dim` features of the last axis with rope_rotate_half (frequencies computed as if
    the head were `rotary_dim` wide), and pass the remaining D − rotary_dim features through unchanged.
    rotary_dim is even, 0 <= rotary_dim <= D. rotary_dim == D is ordinary RoPE; rotary_dim == 0 returns x
    unchanged (no position signal at all: "NoPE"). Same shape and dtype as x.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- feed-forward
def swiglu(params, x):
    """The gated feed-forward block of Llama and DeepSeek: w2( silu(x @ w1) * (x @ w3) ).

    params: {"w1": (d, h), "w3": (d, h), "w2": (h, d)}; x: (..., d) -> (..., d).
    w1 goes through the SiLU (the "gate"), w3 does not (the "up" projection). Some tutorials swap the
    names w1/w3 — the tests use the convention written here.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- grouped-query attention
def repeat_kv(x, n_rep: int):
    """(B, S, Hkv, D) -> (B, S, Hkv * n_rep, D): repeat each key/value head n_rep times *consecutively*.

    Head order after repeating: kv0, kv0, …, kv1, kv1, … — so query head h uses kv head h // n_rep.
    n_rep == 1 returns x unchanged.
    """
    raise NotImplementedError


def gqa_attention(params, x, positions, cfg, cache=None):
    """Causal grouped-query attention with RoPE and an optional KV cache. Returns (out, new_cache).

    x: (B, T, d_model). positions: (T,) absolute positions of these T tokens.
    cfg: {"n_heads": H, "n_kv_heads": Hkv, "head_dim": D, "rope_theta": float}; H is a multiple of Hkv.
    params: {"wq": (d_model, H·D), "wk": (d_model, Hkv·D), "wv": (d_model, Hkv·D), "wo": (H·D, d_model)}.
    cache: None, or {"k": (B, P, Hkv, D), "v": (B, P, Hkv, D)} holding the *rotated* keys and the values
    of positions 0 … P-1 (so here positions == P + arange(T)).

    Steps: project q, k, v and split heads → apply rope_rotate_half to q and k → append k, v to the cache
    along the sequence axis → new_cache = {"k": all keys, "v": all values} (return it even when cache is
    None) → repeat_kv so every query head has keys → scores = q·k / sqrt(D) (compute the softmax in
    float32) → causal mask: query at absolute position p may see key s only if s <= p → softmax →
    weighted sum of values → merge heads → @ wo.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- find the bugs
def buggy_cached_attention(q, k_new, v_new, cache_k, cache_v, max_len: int = 64):
    """PROVIDED (it works, but it's wrong). Attention for new queries against a KV cache.

    This follows code from a published "Llama 3 from scratch" tutorial and contains two real bugs.
    q, k_new, v_new: (B, T, H, D) for the T new tokens (already rotated). cache_k, cache_v: (B, P, H, D)
    for the P earlier tokens. Should return (out (B, T, H, D), (all_k, all_v)).
    """
    k = jnp.concatenate([cache_k, k_new], axis=-1)
    v = jnp.concatenate([cache_v, v_new], axis=-1)
    T, S = q.shape[1], k.shape[1]
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(q.shape[-1])
    mask = jnp.tril(jnp.ones((max_len, max_len), dtype=bool))[:T, :S]
    scores = jnp.where(mask, scores, -jnp.inf)
    out = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(scores, axis=-1), v)
    return out, (k, v)


def fix_buggy_cached_attention(q, k_new, v_new, cache_k, cache_v):
    """Same contract as buggy_cached_attention, with both bugs fixed (and no max_len needed).

    The test: run a whole sequence through attention at once, then run the same sequence as a prefill
    plus one-token decode steps through your function — the outputs must match.
    Bug 1 shows up as a crash; bug 2 is silent (wrong numbers, right shapes). Which mask row should a
    query at absolute position P + t use?
    """
    raise NotImplementedError


# ---------------------------------------------------------------- mixture of experts
def moe_route(scores, bias, top_k: int):
    """DeepSeek-style routing. Returns gate weights (N, E).

    scores: (N, E) router affinities in (0, 1) (already through a sigmoid). bias: (E,) balancing bias.
    Pick each token's top_k experts by (scores + bias) — the bias decides WHO is picked — but use the raw
    scores (no bias) as gate values. Unpicked experts get 0. Normalize each row's gates to sum to 1.
    The bias must receive no gradient (jax.lax.stop_gradient).
    """
    raise NotImplementedError


def update_bias(bias, load, gamma: float):
    """Aux-loss-free balancing step: bias + gamma · sign(mean(load) − load), mean over the last axis.

    load: (E,) (or (L, E) for L layers) — the fraction of routed slots each expert received this step.
    Overloaded experts (load above the mean) get pushed down; underloaded ones up. Same shape as bias.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- KV-cache arithmetic
def mha_cache_floats(n_kv_heads: int, k_dim: int, v_dim: int) -> int:
    """Numbers cached per token per layer by standard attention: one key and one value per KV head.

    (MHA: n_kv_heads = n_heads. GQA: fewer. MQA: 1.)
    """
    raise NotImplementedError


def mla_cache_floats(cfg: dict) -> int:
    """Numbers cached per token per layer by Multi-head Latent Attention.

    cfg has "kv_rank" (width of the compressed joint key/value latent) and "d_rope" (width of the single
    RoPE key shared by all heads). MLA caches only those two things.
    """
    raise NotImplementedError
