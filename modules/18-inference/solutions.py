"""Module 18 reference solutions. Try the exercises first — `make test N=18` grades yours."""
import jax
import jax.numpy as jnp

BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1}


def kv_cache_bytes(cfg, seq_len: int, batch: int, dtype: str = "bfloat16") -> int:
    # K and V, for every layer, batch row, position and KV head: head_dim numbers each
    return 2 * cfg.n_layers * batch * seq_len * cfg.n_kv_heads * cfg.head_dim * BYTES[dtype]


def decode_roofline_tok_s(n_params: int, bytes_per_param: float, bandwidth_gbs: float) -> float:
    return bandwidth_gbs * 1e9 / (n_params * bytes_per_param)


def top_k_filter(logits, k: int):
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = jnp.sort(logits, axis=-1)[..., -k][..., None]
    return jnp.where(logits < kth, -jnp.inf, logits)


def top_p_filter(logits, p: float):
    if p >= 1.0:
        return logits
    order = jnp.argsort(-logits, axis=-1)
    sorted_logits = jnp.take_along_axis(logits, order, -1)
    probs = jax.nn.softmax(sorted_logits, axis=-1)
    before = jnp.cumsum(probs, -1) - probs                 # probability mass strictly above this token
    keep_sorted = before < p                                # always keeps the top token (before = 0)
    keep = jnp.take_along_axis(keep_sorted, jnp.argsort(order, axis=-1), -1)   # undo the sort
    return jnp.where(keep, logits, -jnp.inf)


def apply_repetition_penalty(logits, prev_ids, penalty: float):
    seen = jnp.zeros(logits.shape[-1], bool).at[prev_ids].set(True)
    penalized = jnp.where(logits > 0, logits / penalty, logits * penalty)
    return jnp.where(seen, penalized, logits)


def quantize_int8(w):
    s = jnp.max(jnp.abs(w), axis=0) / 127.0
    s = jnp.where(s == 0, 1.0, s)
    q = jnp.clip(jnp.round(w / s), -127, 127).astype(jnp.int8)
    return {"q": q, "s": s.astype(jnp.float32)}


def dequantize_int8(qw):
    return qw["q"].astype(jnp.float32) * qw["s"]


def quantize_int4_grouped(w, group: int = 64):
    n_in, n_out = w.shape
    g = w.reshape(n_in // group, group, n_out)
    s = jnp.max(jnp.abs(g), axis=1) / 7.0
    s = jnp.where(s == 0, 1.0, s)
    q = jnp.clip(jnp.round(g / s[:, None, :]), -8, 7).astype(jnp.int8).reshape(n_in, n_out)
    return {"q": q, "s": s.astype(jnp.float32)}


def dequantize_int4_grouped(qw):
    q, s = qw["q"], qw["s"]
    n_in, n_out = q.shape
    group = n_in // s.shape[0]
    g = q.reshape(s.shape[0], group, n_out).astype(jnp.float32) * s[:, None, :]
    return g.reshape(n_in, n_out)


def speculative_accept(draft_tokens, target_logits):
    greedy = jnp.argmax(target_logits, axis=-1).astype(jnp.int32)        # [k+1]
    k = draft_tokens.shape[0]
    matches = (draft_tokens == greedy[:k]).astype(jnp.int32)
    n_accepted = jnp.sum(jnp.cumprod(matches))                           # length of the matching prefix
    return n_accepted, greedy[n_accepted]
