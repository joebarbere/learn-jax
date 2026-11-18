"""Module 40 reference solutions. Same signatures as exercises.py."""
import re
import unicodedata

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------- LatentMoE
def latent_moe_costs(d: int, m: int, n_experts: int, top_k: int, alpha: int, variant: str,
                     bytes_per_value: int = 2) -> dict:
    w = d if variant == "standard" else d // alpha
    n = n_experts if variant == "standard" else n_experts * alpha
    k = top_k * alpha if variant == "acc" else top_k
    expert = 3 * w * m
    proj = 0 if variant == "standard" else 2 * d * w
    active = k * expert + proj
    return {"width": w, "experts": n, "active": k, "expert_params": expert, "routed_params": n * expert + proj,
            "active_params": active, "flops_per_token": 2 * active,
            "a2a_bytes_per_token": 2 * k * w * bytes_per_value}


def latent_moe(p, x, gates):
    z = x @ p["down"]                                                          # (N, ℓ)
    w = p["experts"]
    h = jax.nn.silu(jnp.einsum("nl,elm->nem", z, w["w1"])) * jnp.einsum("nl,elm->nem", z, w["w3"])
    routed = jnp.einsum("ne,nel->nl", gates, jnp.einsum("nem,eml->nel", h, w["w2"]))   # (N, ℓ)
    s = p["shared"]
    shared = (jax.nn.silu(x @ s["w1"]) * (x @ s["w3"])) @ s["w2"]
    return routed @ p["up"] + shared


# ---------------------------------------------------------------- Engram, on the host
def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[ \t\r\n]+", " ", s)
    return " " if s == " " else s.strip()


def compress_vocab(texts):
    lookup, key_to_new = np.empty(len(texts), np.int64), {}
    for i, text in enumerate(texts):
        key = ("\ufffd", i) if "\ufffd" in text else (_normalize(text) or text)
        lookup[i] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


def ngram_hashes(ids, multipliers, primes, pad_id):
    x = np.asarray(ids, np.int64)
    T = x.shape[1]
    max_n = len(primes) + 1
    shifts = [x] + [np.pad(x, ((0, 0), (k, 0)), constant_values=pad_id)[:, :T] for k in range(1, max_n)]
    out = []
    for n, heads in zip(range(2, max_n + 1), primes, strict=True):
        mix = shifts[0] * multipliers[0]
        for k in range(1, n):
            mix = np.bitwise_xor(mix, shifts[k] * multipliers[k])
        out += [mix % p for p in heads]
    return np.stack(out, axis=2)


# ---------------------------------------------------------------- Engram, on the GPU
def _rms(x, w, eps=1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps)).astype(x.dtype) * w


def engram_gate(h, e, p):
    k, v = e @ p["w_k"], e @ p["w_v"]
    score = jnp.sum((_rms(h, p["norm_h"]) * _rms(k, p["norm_k"])).astype(jnp.float32), -1)
    alpha = jax.nn.sigmoid(score / jnp.sqrt(h.shape[-1]))
    return v * alpha[..., None].astype(v.dtype), alpha


def causal_dwconv(x, w, dilation: int):
    K, T = w.shape[0], x.shape[1]
    pad = (K - 1) * dilation
    xp = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))
    return sum(w[i] * xp[:, i * dilation:i * dilation + T] for i in range(K))
