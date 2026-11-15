"""Module 38 exercises: retrieval-augmented generation, built from its parts.

Replace each `raise NotImplementedError`, then run `make test N=38`. The scoring functions (BM25, InfoNCE,
cosine top-k, min-max fusion) should use jax.numpy so they run on the GPU; the metrics and RRF may use
NumPy or plain Python.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401
import numpy as np  # noqa: F401


def chunk_sentences(n_tokens: list[int], max_tokens: int, overlap: int = 0) -> list[tuple[int, int]]:
    """Pack whole sentences into chunks. `n_tokens[i]` is sentence i's length.

    Greedy: start a chunk at sentence `start`, keep adding sentences while the chunk stays within
    `max_tokens` (a single sentence longer than `max_tokens` becomes a chunk on its own — never split it).
    The next chunk starts `overlap` sentences before the previous one ended, but always at least one
    sentence later than the previous start (so the loop always moves forward).
    Return half-open (start, end) sentence ranges, in order. Stop once a chunk reaches the last sentence.

    chunk_sentences([3, 3, 3, 3], max_tokens=6, overlap=0) -> [(0, 2), (2, 4)]
    chunk_sentences([3, 3, 3, 3], max_tokens=6, overlap=1) -> [(0, 2), (1, 3), (2, 4)]
    """
    raise NotImplementedError


def bm25_weights(tf, doc_len, df, n_docs: int, avg_len: float, k1: float = 1.5, b: float = 0.75):
    """The BM25 weight of a term in a document, for arrays of (term, document) pairs.

    tf: how often the term occurs in the document; doc_len: the document's length in terms;
    df: how many documents contain the term. All three are aligned arrays (one entry per pair).
        idf    = log(1 + (n_docs - df + 0.5) / (df + 0.5))
        weight = idf · tf · (k1 + 1) / (tf + k1 · (1 - b + b · doc_len / avg_len))
    Return a float32 array of weights. A query's BM25 score for a document is the sum of the weights of
    the query's terms in that document.
    """
    raise NotImplementedError


def info_nce(q, d, temperature: float = 0.05):
    """InfoNCE with in-batch negatives (DPR, 2020). q, d: (B, D), L2-normalized; row i of d is the
    positive for row i of q and every other row of d is a negative.

    logits = q @ d.T / temperature; loss = mean over i of cross-entropy(logits[i], label i).
    (Module 33's CLIP loss is this, averaged in both directions.) Return a float32 scalar.
    d may have MORE rows than q (shape (B + M, D)): rows B.. are extra negatives (hard negatives) that are
    nobody's positive. The same formula handles it: row i's label is still column i.
    """
    raise NotImplementedError


def topk_cosine(Q, D, k: int):
    """Exact nearest neighbours by cosine similarity. Q: (nq, dim) queries, D: (n, dim) documents
    (neither needs to be normalized). Return (scores, indices), each (nq, k), best first —
    `jax.lax.top_k` on the (nq, n) similarity matrix.
    """
    raise NotImplementedError


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion (Cormack, Clarke and Büttcher, 2009) for ONE query.

    `rankings` holds several ranked lists of document ids (best first; they may contain different
    documents). A document's fused score is the sum over the lists that contain it of 1 / (k + rank),
    with rank starting at 1. Return every document that appears in any list, best fused score first;
    break ties by smaller document id.
    """
    raise NotImplementedError


def minmax_fuse(a, b, w: float):
    """Convex combination of two score arrays over the SAME candidate documents.

    Rescale each array to [0, 1] with (x - min) / (max - min) (an array whose values are all equal
    becomes all zeros), then return w · a' + (1 - w) · b'.
    """
    raise NotImplementedError


def recall_at_k(ranked, relevant, k: int) -> float:
    """ranked: (nq, K) document ids, best first. relevant: (nq,) the one relevant id per query.
    Fraction of queries whose relevant document is in the top k. (With one relevant document per
    query this is also hit@k.)"""
    raise NotImplementedError


def mrr_at_k(ranked, relevant, k: int) -> float:
    """Mean reciprocal rank: average over queries of 1 / rank of the relevant document if it is in the
    top k (rank starts at 1), else 0."""
    raise NotImplementedError


def ndcg_at_k(ranked, relevant, k: int) -> float:
    """Normalized discounted cumulative gain with one relevant document per query: 1 / log2(rank + 1)
    if the relevant document is in the top k, else 0 (the ideal ranking scores 1), averaged."""
    raise NotImplementedError
