"""Module 30 reference solutions. Try the exercises first — `make test N=30` grades yours."""
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np


def build_vocab(tokens: list[str], min_count: int):
    counts = Counter(tokens)
    kept = sorted((w for w, c in counts.items() if c >= min_count), key=lambda w: (-counts[w], w))
    word_to_id = {w: i for i, w in enumerate(kept)}
    return word_to_id, kept, np.array([counts[w] for w in kept], dtype=np.int64)


def subsample_probs(counts: np.ndarray, t: float = 1e-4) -> np.ndarray:
    f = np.asarray(counts, dtype=np.float64) / np.sum(counts)
    return np.minimum(1.0, np.sqrt(t / f) + t / f)


def negative_sampling_table(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    p = np.asarray(counts, dtype=np.float64) ** power
    return p / p.sum()


def skipgram_pairs(ids, window: int, key):
    ids = np.asarray(ids)
    centers, contexts = [], []
    for off in range(-window, window + 1):
        if off == 0:
            continue
        lo, hi = max(0, -off), min(len(ids), len(ids) - off)
        if hi <= lo:
            continue
        centers.append(ids[lo:hi])
        contexts.append(ids[lo + off:hi + off])
    c = jnp.asarray(np.concatenate(centers) if centers else np.zeros(0, np.int32), jnp.int32)
    x = jnp.asarray(np.concatenate(contexts) if contexts else np.zeros(0, np.int32), jnp.int32)
    perm = jax.random.permutation(key, c.shape[0])
    return c[perm], x[perm]


def sgns_loss(center, context, negatives, emb_in, emb_out):
    v = emb_in[center]                                   # (B, D)
    u_pos = emb_out[context]                             # (B, D)
    u_neg = emb_out[negatives]                           # (B, K, D)
    pos = jax.nn.log_sigmoid(jnp.sum(v * u_pos, -1))
    neg = jax.nn.log_sigmoid(-jnp.einsum("bd,bkd->bk", v, u_neg)).sum(-1)
    return -jnp.mean(pos + neg)


def _unit(E):
    E = jnp.asarray(E, jnp.float32)
    return E / (jnp.linalg.norm(E, axis=-1, keepdims=True) + 1e-8)


def cosine_neighbors(E, idx: int, k: int):
    U = _unit(E)
    sims = U @ U[idx]
    sims = sims.at[idx].set(-jnp.inf)
    return jax.lax.top_k(sims, k)[1]


def analogy(E, a: int, b: int, c: int, k: int = 1):
    U = _unit(E)
    q = U[b] - U[a] + U[c]
    sims = U @ (q / (jnp.linalg.norm(q) + 1e-8))
    sims = sims.at[jnp.array([a, b, c])].set(-jnp.inf)
    return jax.lax.top_k(sims, k)[1]
