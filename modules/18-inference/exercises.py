"""Module 18 exercises: the arithmetic and the small functions behind fast, controllable inference.

Replace each `raise NotImplementedError` with your code, then run `make test N=18`. The lesson calls
these on a real model (SmolLM2), so a working function lights up the matching experiment.
All functions take and return jax.numpy arrays unless the docstring says otherwise; none of them need
Python loops over tokens or vocabulary entries.
"""
import jax  # noqa: F401  (hint: jax.nn.softmax)
import jax.numpy as jnp  # noqa: F401

BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1}


def kv_cache_bytes(cfg, seq_len: int, batch: int, dtype: str = "bfloat16") -> int:
    """Memory for a KV cache holding `seq_len` positions for `batch` sequences.

    `cfg` has `n_layers`, `n_kv_heads` and `head_dim`. Every layer stores a key AND a value vector of
    `head_dim` numbers per KV head, per position, per sequence. Use BYTES[dtype]. Return a Python int.
    """
    raise NotImplementedError


def decode_roofline_tok_s(n_params: int, bytes_per_param: float, bandwidth_gbs: float) -> float:
    """Upper bound on single-sequence decode speed when every weight is read once per token.

    bandwidth_gbs is in GB/s (10**9 bytes per second). Return tokens per second as a float.
    """
    raise NotImplementedError


def top_k_filter(logits, k: int):
    """Keep the k largest logits along the last axis; set every other entry to -inf.

    Works on [..., V]. Ties with the k-th largest value are kept. If k <= 0 or k >= V, return the
    logits unchanged.
    """
    raise NotImplementedError


def top_p_filter(logits, p: float):
    """Nucleus filtering along the last axis: keep the smallest set of most-likely tokens whose total
    probability reaches p; set the rest to -inf.

    Rule: sort by probability (descending); a token is kept if the probability mass of the tokens
    strictly ahead of it is < p. So the most likely token is always kept. If p >= 1, return the logits
    unchanged (summing 49,152 float32 probabilities can overshoot 1.0 and wrongly drop the tail).
    Works on [..., V]; the output is in the ORIGINAL vocabulary order.
    """
    raise NotImplementedError


def apply_repetition_penalty(logits, prev_ids, penalty: float):
    """Make already-generated tokens less likely (the CTRL-paper rule used by most libraries).

    logits: [V]. prev_ids: int array of token ids seen so far (may contain repeats — penalize once).
    For each seen token: positive logits are divided by `penalty`, negative ones multiplied by it.
    Unseen tokens are unchanged. penalty = 1.0 changes nothing.
    """
    raise NotImplementedError


def quantize_int8(w):
    """Symmetric per-output-channel int8 quantization of a weight matrix w: [in, out].

    For each output column j: s[j] = max_i |w[i, j]| / 127, q = round(w / s) clipped to [-127, 127].
    Return {"q": int8 [in, out], "s": float32 [out]}. An all-zero column must not produce NaN or inf
    (use s = 1 there).
    """
    raise NotImplementedError


def dequantize_int8(qw):
    """Inverse of quantize_int8: float32 [in, out] = q * s (s broadcast over rows)."""
    raise NotImplementedError


def quantize_int4_grouped(w, group: int = 64):
    """Symmetric int4 quantization with one scale per `group` consecutive INPUT rows, per output column.

    w: [in, out] with in divisible by group. Rows [g*group, (g+1)*group) share s[g, j] =
    max |w| in that block / 7, and q = round(w / s) clipped to [-8, 7] (int4's range).
    Return {"q": int8 [in, out] holding values in [-8, 7], "s": float32 [in // group, out]}.
    An all-zero block must not produce NaN or inf (use s = 1 there).
    """
    raise NotImplementedError


def dequantize_int4_grouped(qw):
    """Inverse of quantize_int4_grouped (infer `group` from the shapes): float32 [in, out]."""
    raise NotImplementedError


def speculative_accept(draft_tokens, target_logits):
    """Greedy speculative-decoding verification for one round.

    draft_tokens: int32 [k] — what the small model proposed after the current token.
    target_logits: [k + 1, V] — the big model's logits at the current position and after each draft
    token (one forward pass). target_logits[i] predicts the token that should follow draft i - 1.

    Accept drafts left to right while draft_tokens[i] == argmax(target_logits[i]); stop at the first
    mismatch. Return (n_accepted, next_token): n_accepted in 0..k, and next_token =
    argmax(target_logits[n_accepted]) — the big model's correction (or a free bonus token when all k
    were accepted). Must work under jax.jit: no Python `if` on array values.
    """
    raise NotImplementedError
