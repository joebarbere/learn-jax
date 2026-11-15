"""Module 38 reference solutions. Don't read these until you've tried exercises.py."""
import jax
import jax.numpy as jnp
import numpy as np


def chunk_sentences(n_tokens: list[int], max_tokens: int, overlap: int = 0) -> list[tuple[int, int]]:
    chunks, start, n = [], 0, len(n_tokens)
    while start < n:
        end, size = start, 0
        while end < n and (end == start or size + n_tokens[end] <= max_tokens):
            size += n_tokens[end]
            end += 1
        chunks.append((start, end))
        if end == n:
            break
        start = max(start + 1, end - overlap)
    return chunks


def bm25_weights(tf, doc_len, df, n_docs: int, avg_len: float, k1: float = 1.5, b: float = 0.75):
    tf = jnp.asarray(tf, jnp.float32)
    df = jnp.asarray(df, jnp.float32)
    idf = jnp.log1p((n_docs - df + 0.5) / (df + 0.5))
    norm = k1 * (1.0 - b + b * jnp.asarray(doc_len, jnp.float32) / avg_len)
    return idf * tf * (k1 + 1.0) / (tf + norm)


def info_nce(q, d, temperature: float = 0.05):
    logits = (q @ d.T).astype(jnp.float32) / temperature
    return -jnp.mean(jnp.diagonal(jax.nn.log_softmax(logits, axis=-1)))


def topk_cosine(Q, D, k: int):
    Qn = Q / jnp.linalg.norm(Q, axis=-1, keepdims=True)
    Dn = D / jnp.linalg.norm(D, axis=-1, keepdims=True)
    return jax.lax.top_k(Qn @ Dn.T, k)


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[int]:
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + r)
    return sorted(score, key=lambda d: (-score[d], d))


def minmax_fuse(a, b, w: float):
    def norm(x):
        x = jnp.asarray(x, jnp.float32)
        span = x.max() - x.min()
        return jnp.where(span > 0, (x - x.min()) / jnp.where(span > 0, span, 1.0), 0.0)

    return w * norm(a) + (1.0 - w) * norm(b)


def _ranks(ranked, relevant):
    """1-based rank of the relevant doc in each row, or 0 if absent."""
    hit = np.asarray(ranked) == np.asarray(relevant)[:, None]
    return np.where(hit.any(1), hit.argmax(1) + 1, 0)


def recall_at_k(ranked, relevant, k: int) -> float:
    r = _ranks(ranked, relevant)
    return float(np.mean((r > 0) & (r <= k)))


def mrr_at_k(ranked, relevant, k: int) -> float:
    r = _ranks(ranked, relevant)
    return float(np.mean(np.where((r > 0) & (r <= k), 1.0 / np.maximum(r, 1), 0.0)))


def ndcg_at_k(ranked, relevant, k: int) -> float:
    r = _ranks(ranked, relevant)
    return float(np.mean(np.where((r > 0) & (r <= k), 1.0 / np.log2(np.maximum(r, 1) + 1.0), 0.0)))
