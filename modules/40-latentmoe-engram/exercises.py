"""Module 40 exercises: two new axes of sparsity, LatentMoE and Engram.

Replace each `raise NotImplementedError` with your code, then run `make test N=40`. The lesson builds a
DeepSeek-V3-style model (Module 17's MLA + DeepSeekMoE, provided in models.py) and upgrades it with your parts,
so later cells come alive as you finish these.

Two of the functions run on the *host* in NumPy, not on the GPU: Engram's lookups depend only on the token ids,
so they can be computed in the data pipeline (in 64-bit integers, which JAX doesn't use by default) before the
model ever runs. That's what makes Engram's tables prefetchable.

Conventions: d = model width, ℓ = latent width (d / α), m = an expert's hidden width, E = number of routed
experts, N = number of tokens (batch × time, flattened).
"""
import jax  # noqa: F401  (for your code)
import jax.numpy as jnp  # noqa: F401
import numpy as np  # noqa: F401


# ---------------------------------------------------------------- LatentMoE
def latent_moe_costs(d: int, m: int, n_experts: int, top_k: int, alpha: int, variant: str,
                     bytes_per_value: int = 2) -> dict:
    """Parameter, compute and communication costs of one MoE layer's *routed* experts, per token.

    variant: "standard" (experts work at width d), "eff" or "acc" (LatentMoE: width ℓ = d // alpha, and
    alpha × n_experts experts; "eff" keeps top_k, "acc" uses alpha × top_k). Every expert is a SwiGLU:
    three matrices of width × m. LatentMoE adds two projections, d → ℓ and ℓ → d, which every token uses.
    Ignore the router and the shared experts (they're the same in every variant).

    Return a dict with these int values:
      "width"               the experts' input/output width (d or ℓ)
      "experts"             number of routed experts
      "active"              experts picked per token
      "expert_params"       parameters in one expert
      "routed_params"       all experts' parameters + the projections (if any)
      "active_params"       parameters one token uses: its active experts + the projections (if any)
      "flops_per_token"     2 × active_params (one multiply and one add per weight)
      "a2a_bytes_per_token" bytes one token sends across devices under expert parallelism: its width-sized
                            vector to each active expert and the result back (dispatch + combine), at
                            bytes_per_value bytes per number
    Example: latent_moe_costs(4096, 2688, 128, 6, 4, "acc")["active"] == 24
    """
    raise NotImplementedError


def latent_moe(p, x, gates):
    """LatentMoE's forward pass (Elango et al., eq. 2) for N tokens, with Module 17's dense compute.

    x: (N, d). gates: (N, E), the router's output — already computed from the *full* x (the router never sees
    the latent), zero for experts a token didn't pick. p:
      "down": (d, ℓ), "up": (ℓ, d)                                   the latent projections
      "experts": {"w1": (E, ℓ, m), "w3": (E, ℓ, m), "w2": (E, m, ℓ)}  routed SwiGLU experts, in the latent
      "shared":  {"w1": (d, h), "w3": (d, h), "w2": (h, d)}          the shared expert, at full width d
    Project x down to ℓ, run every expert on every token (SwiGLU: silu(z W1) * (z W3), then W2), weight each
    expert's output by its gate and sum, project the sum up to d *once*, then add the shared expert's output on
    the full x. Returns (N, d).
    """
    raise NotImplementedError


# ---------------------------------------------------------------- Engram, on the host (NumPy)
def compress_vocab(texts):
    """Engram's tokenizer compression: merge tokens whose text means the same thing.

    texts: one string per token id. Normalize each, in this order (DeepSeek's demo): Unicode NFKC, then NFD,
    then drop accent marks (characters whose `unicodedata.category` is "Mn"), lowercase, replace every run of
    spaces/tabs/newlines with one space, then — unless the result is exactly " " — strip leading and trailing
    whitespace. If the result is empty, use the original text instead.
    Tokens with the same normalized text share a new id; new ids are 0, 1, 2, … in order of first appearance.
    One exception: a text containing the replacement character "\ufffd" (a byte-level token that is only part
    of a multi-byte UTF-8 character, so it doesn't decode on its own) is never merged — it gets an id of its own.

    Returns (lookup, n_new): lookup is an int64 NumPy array, lookup[old_id] = new_id; n_new the number of
    distinct new ids. Example: ["The", " the", "THE", "a"] -> ([0, 0, 0, 1], 2).
    """
    raise NotImplementedError


def ngram_hashes(ids, multipliers, primes, pad_id):
    """Hash every position's suffix n-grams to table rows (DeepSeek's multiplicative-XOR hash).

    ids: (B, T) int64 NumPy array of (compressed) token ids. multipliers: (max_n,) int64, odd.
    primes: a list with one entry per n-gram order n = 2, 3, …, max_n; entry n-2 is a list of that order's
    head table sizes (distinct primes). pad_id: the id used for positions before the start.

    For order n at position t, the n-gram is ids[t-n+1 .. t] (positions before 0 read pad_id). Mix it:
        mix = (ids[t] * m[0]) XOR (ids[t-1] * m[1]) XOR … XOR (ids[t-n+1] * m[n-1])      (int64)
    and head j of that order reads row  mix % primes[n-2][j].
    Returns (B, T, total heads) int64, columns ordered by order, then head:
    [n=2 head 0, n=2 head 1, …, n=3 head 0, …].
    """
    raise NotImplementedError


# ---------------------------------------------------------------- Engram, on the GPU (JAX)
def engram_gate(h, e, p):
    """Engram's context-aware gate (paper eq. 3–4). Decide, per token, how much of the looked-up memory to use.

    h: (B, T, d) the hidden state. e: (B, T, d_mem) the concatenated table rows. p: "w_k", "w_v" (d_mem, d);
    "norm_h", "norm_k" (d,) RMSNorm weights.
        k = e W_k,   v = e W_v
        alpha = sigmoid( sum over d of RMSNorm(h) * RMSNorm(k)  /  sqrt(d) )      (RMSNorm with eps = 1e-6)
    Returns (alpha * v, alpha): shapes (B, T, d) and (B, T). Compute alpha in float32.
    """
    raise NotImplementedError


def causal_dwconv(x, w, dilation: int):
    """A depthwise, causal, dilated 1-D convolution over time.

    x: (B, T, C). w: (K, C) — one K-tap filter per channel, no mixing between channels. Output (B, T, C):
        y[:, t, c] = sum over i in 0..K-1 of  w[i, c] * x[:, t - (K-1-i) * dilation, c]
    with x taken as 0 before time 0. So w[K-1] weighs the current step, w[K-2] the step `dilation` back, …
    (This is PyTorch's Conv1d(groups=C, padding=(K-1)*dilation) keeping the first T outputs.)
    """
    raise NotImplementedError
