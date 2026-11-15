"""Module 38 helpers: a small retrieval stack in JAX — BM25, a trained bi-encoder, IVF, a reranker, RAG.

The learner's exercise functions are passed in (`ex`), so every number in the lesson comes from their code.
Module 39 imports this file for long-term memory:

    idx = rag.build_index(texts, ex)            # BM25 (no training needed); add encoder=... for hybrid
    idx.search("pulsar timing arrays", k=5)     # -> [(doc_id, score), ...]
"""
from __future__ import annotations

import collections
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import sparse

RUNS = Path("/course/runs/38-rag")
WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "may", "more", "not", "of", "on", "or", "our", "such", "than", "that",
    "the", "their", "these", "this", "to", "using", "use", "used", "was", "we", "were", "which", "with",
    "within"])
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def words(text: str) -> list[str]:
    """BM25's tokenizer: lowercase alphanumeric runs, minus a few stopwords. No stemming."""
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def sentences(text: str) -> list[str]:
    return [s for s in SENT.split(text.strip()) if s]


# ------------------------------------------------------------------------------------------- the data

def split(papers: list[dict], n_test: int = 1000, seed: int = 0):
    """(test ids, train ids): the test papers' titles are the benchmark queries; their (title, abstract)
    pairs never train the bi-encoder. Every abstract stays in the index."""
    perm = np.random.default_rng(seed).permutation(len(papers))
    return np.sort(perm[:n_test]), np.sort(perm[n_test:])


# ------------------------------------------------------------------------------------------- BM25

@dataclass
class BM25:
    """Sparse doc-term matrix of BM25 weights, as a JAX BCOO array. Scoring a batch of queries is one
    sparse x dense matmul: (n_docs, V) @ (V, batch)."""

    vocab: dict
    W: sparse.BCOO
    n_docs: int
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, texts: list[str], bm25_weights, k1: float = 1.5, b: float = 0.75) -> BM25:
        vocab: dict[str, int] = {}
        di, ti, tf, dl = [], [], [], []
        for d, text in enumerate(texts):
            c = collections.Counter(words(text))
            dl.append(sum(c.values()))
            for w, n in c.items():
                di.append(d)
                ti.append(vocab.setdefault(w, len(vocab)))
                tf.append(n)
        di, ti, tf, dl = (np.asarray(x, np.int32) for x in (di, ti, tf, dl))
        df = np.bincount(ti, minlength=len(vocab))
        w = bm25_weights(tf, dl[di], df[ti], len(texts), float(dl.mean()), k1, b)
        W = sparse.BCOO((jnp.asarray(w, jnp.float32), jnp.stack([jnp.asarray(di), jnp.asarray(ti)], 1)),
                        shape=(len(texts), len(vocab)))
        return cls(vocab, W, len(texts), k1, b)

    def query_matrix(self, queries: list[str]) -> jnp.ndarray:
        Q = np.zeros((len(self.vocab), len(queries)), np.float32)
        for j, q in enumerate(queries):
            for w in set(words(q)):
                if w in self.vocab:
                    Q[self.vocab[w], j] = 1.0
        return jnp.asarray(Q)

    def scores(self, queries: list[str]) -> jnp.ndarray:
        """(len(queries), n_docs) BM25 scores."""
        return _bcoo_scores(self.W, self.query_matrix(queries))

    def topk(self, queries: list[str], k: int = 100, batch: int = 64):
        out_s, out_i = [], []
        for i in range(0, len(queries), batch):
            s, ix = _bcoo_topk(self.W, self.query_matrix(queries[i:i + batch]), k)
            out_s.append(np.asarray(s))
            out_i.append(np.asarray(ix))
        return np.concatenate(out_s), np.concatenate(out_i)

    def scores_for(self, queries: list[str], cand: np.ndarray, batch: int = 64) -> np.ndarray:
        """BM25 scores of each query's candidate documents: cand (nq, c) ids -> (nq, c) scores."""
        out = []
        for i in range(0, len(queries), batch):
            out.append(np.asarray(_bcoo_gather(self.W, self.query_matrix(queries[i:i + batch]),
                                               jnp.asarray(cand[i:i + batch]))))
        return np.concatenate(out)


def topk_wide(x, k: int, block: int = 32768):
    """`jax.lax.top_k` along the last axis, in blocks: on ROCm, top_k over very wide rows (hundreds of thousands
    of columns) fails with "Failed to set shared memory size". Top-k of each block, then top-k of the winners."""
    n = x.shape[-1]
    if n <= block:
        return jax.lax.top_k(x, k)
    pad = (-n) % block
    xp = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)], constant_values=-jnp.inf)
    xb = xp.reshape(*x.shape[:-1], -1, block)
    v, i = jax.lax.top_k(xb, min(k, block))                                  # (..., n_blocks, k)
    i = i + (jnp.arange(xb.shape[-2]) * block)[:, None]
    v, j = jax.lax.top_k(v.reshape(*x.shape[:-1], -1), k)
    return v, jnp.take_along_axis(i.reshape(*x.shape[:-1], -1), j, axis=-1)


@jax.jit
def _bcoo_scores(W, Q):
    return (W @ Q).T


def _bcoo_gather(W, Q, cand):
    return jnp.take_along_axis(_bcoo_scores(W, Q), cand, axis=1)


_topk_jit = jax.jit(topk_wide, static_argnames=("k", "block"))


def _bcoo_topk(W, Q, k):
    # Two separate programs, scoring then top-k. Compiled as ONE program, XLA on ROCm sometimes emits a kernel that
    # asks for more shared memory than the GPU has ("Failed to set shared memory size", hipError 98) — depending
    # on the shapes; an optimization_barrier between them was not enough. Each half alone always works.
    return _topk_jit(_bcoo_scores(W, Q), k=k)


# ------------------------------------------------------------------------------------------- the bi-encoder

@dataclass(frozen=True)
class EncCfg:
    vocab: int = 49152          # SmolLM2's tokenizer
    d: int = 256
    layers: int = 4
    heads: int = 4
    q_len: int = 32             # titles
    d_len: int = 160            # abstracts (truncated: the opening carries most of the topic)
    pretrained: bool = True     # token vectors from SmolLM2-135M's embedding table (frozen), projected to d
    d_emb: int = 576            # SmolLM2-135M's width


def tokenizer():
    from tokenizers import Tokenizer

    from learnjax import llama

    return Tokenizer.from_file(str(llama.download(llama.SMOLLM_135M) / "tokenizer.json"))


def encode_ids(tok, texts: list[str], length: int) -> np.ndarray:
    """Token ids, truncated/right-padded to `length`; 0 = padding (SmolLM2's id 0 is <|endoftext|>, which
    never appears inside these texts)."""
    out = np.zeros((len(texts), length), np.int32)
    for i, e in enumerate(tok.encode_batch(texts)):
        ids = e.ids[:length]
        out[i, :len(ids)] = ids
    return out


def enc_init(key, cfg: EncCfg) -> dict:
    d, n = cfg.d, cfg.layers
    ks = iter(jax.random.split(key, 4 + 4 * n))

    def normal(shape, s=0.02):
        return jax.random.normal(next(ks), shape, jnp.float32) * s

    blocks = [{"ln1": jnp.ones(d), "w_qkv": normal((d, 3 * d)), "w_o": normal((d, d), 0.02 / (2 * n) ** 0.5),
               "ln2": jnp.ones(d), "w_fc": normal((d, 4 * d)), "w_proj": normal((4 * d, d), 0.02 / (2 * n) ** 0.5)}
              for _ in range(n)]
    p = {"wpe": normal((max(cfg.q_len, cfg.d_len), d)), "blocks": blocks, "lnf": jnp.ones(d),
         "proj": normal((d, d), d ** -0.5)}
    if cfg.pretrained:
        p["emb_proj"] = normal((cfg.d_emb, d), cfg.d_emb ** -0.5)
    else:
        p["wte"] = normal((cfg.vocab, d))
    return p


def pretrained_table():
    """SmolLM2-135M's input embedding table (49,152 x 576, bf16): token vectors that already encode meaning."""
    from learnjax import llama

    params, _, _ = llama.load(llama.SMOLLM_135M, max_seq=256)
    return params["embed"]


def _ln(x, g):
    x32 = x.astype(jnp.float32)
    x32 = (x32 - x32.mean(-1, keepdims=True)) * jax.lax.rsqrt(x32.var(-1, keepdims=True) + 1e-5)
    return (x32 * g).astype(x.dtype)


def enc_apply(p, ids, cfg: EncCfg, table=None):
    """ids (B, T) -> L2-normalized embeddings (B, d). Bidirectional attention over non-padding tokens,
    mean-pooled (Sentence-BERT's recipe), then a linear projection."""
    B, T = ids.shape
    p = jax.tree.map(lambda a: a.astype(jnp.bfloat16), p)
    pad = ids > 0
    # Project the frozen table first, then gather rows. (Gathering 576-wide bf16 rows and projecting them in one
    # fused kernel page-faulted the GPU on ROCm under jit + scan + grad; this order is the same math.)
    tokens = (table.astype(jnp.bfloat16) @ p["emb_proj"])[ids] if cfg.pretrained else p["wte"][ids]
    x = tokens + p["wpe"][:T]
    hd = cfg.d // cfg.heads
    for blk in p["blocks"]:
        h = _ln(x, blk["ln1"])
        q, k, v = (a.reshape(B, T, cfg.heads, hd) for a in jnp.split(h @ blk["w_qkv"], 3, -1))
        s = jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32) / jnp.sqrt(hd)
        s = jnp.where(pad[:, None, None, :], s, -1e30)
        a = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(s, -1).astype(x.dtype), v).reshape(B, T, cfg.d)
        x = x + a @ blk["w_o"]
        x = x + jax.nn.gelu(_ln(x, blk["ln2"]) @ blk["w_fc"]) @ blk["w_proj"]
    x = _ln(x, p["lnf"]).astype(jnp.float32)
    m = pad[..., None].astype(jnp.float32)
    pooled = (x * m).sum(1) / jnp.maximum(m.sum(1), 1.0)
    e = pooled @ p["proj"].astype(jnp.float32)
    return e / jnp.linalg.norm(e, axis=-1, keepdims=True)


def ict_pairs(key, D, q_len: int, p_blank: float = 0.9):
    """Inverse Cloze Task (Lee, Chang and Toutanova, 2019): a random q_len-token window of each abstract is a
    pseudo-query for its abstract, with the window blanked out of the document most of the time (so the model
    must match topic, not copy text). Free training pairs: every abstract yields new ones each epoch."""
    k_start, k_blank = jax.random.split(key)
    lens = (D > 0).sum(1)
    start = (jax.random.uniform(k_start, lens.shape) * jnp.maximum(lens - q_len, 1)).astype(jnp.int32)
    q = jax.vmap(lambda row, s: jax.lax.dynamic_slice(row, (s,), (q_len,)))(D, start)
    pos = jnp.arange(D.shape[1])
    window = (pos >= start[:, None]) & (pos < start[:, None] + q_len)
    blank = (jax.random.uniform(k_blank, lens.shape) < p_blank) & (lens >= 2 * q_len)   # never empty a short doc
    return q, jnp.where(window & blank[:, None], 0, D)


def make_trainer(cfg: EncCfg, info_nce, steps: int, batch: int = 256, lr: float = 1e-3, temperature: float = 0.05,
                 chunk: int = 50, ict: float = 0.0):
    """The frozen table is a jit *argument* (not a closed-over constant: Module 06). `ict` is the fraction of each
    batch made of Inverse Cloze Task pairs instead of (title, abstract) pairs."""
    n_ict = int(batch * ict)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, max(steps // 20, 1), steps, lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))

    def loss(p, qi, di, table):
        return info_nce(enc_apply(p, qi, cfg, table), enc_apply(p, di, cfg, table), temperature)

    @jax.jit
    def run(p, st, Q, D, key, table, NEG):
        def step(carry, k):
            p, st = carry
            k_idx, k_ict = jax.random.split(k)
            idx = jax.random.choice(k_idx, Q.shape[0], (batch,), replace=False)
            q, d = Q[idx], D[idx]
            if n_ict:                       # the last n_ict rows become ICT pairs (different papers, so no clash)
                iq, idoc = ict_pairs(k_ict, d[batch - n_ict:], cfg.q_len)
                q, d = q.at[batch - n_ict:].set(iq), d.at[batch - n_ict:].set(idoc)
            if NEG is not None:             # hard negatives: d becomes [positives; BM25 look-alikes], (2B, len)
                d = jnp.concatenate([d, NEG[idx]])
            l, g = jax.value_and_grad(loss)(p, q, d, table)
            u, st = opt.update(g, st, p)
            return (optax.apply_updates(p, u), st), l
        (p, st), ls = jax.lax.scan(step, (p, st), jax.random.split(key, chunk))
        return p, st, ls

    return opt, run, chunk


def train_encoder(info_nce, Q_ids, D_ids, steps: int, cfg: EncCfg = EncCfg(), key=None, log_every: int = 0,
                  table=None, neg_ids=None, **kw):
    """Train on (title, abstract) id pairs; `neg_ids` (same rows as D_ids) adds one hard negative per pair.
    Returns (params, losses per step, seconds after the first chunk)."""
    key = jax.random.key(0) if key is None else key
    opt, run, chunk = make_trainer(cfg, info_nce, steps, **kw)
    kp, kt = jax.random.split(key)
    p = enc_init(kp, cfg)
    st = opt.init(p)
    Q, D = jnp.asarray(Q_ids), jnp.asarray(D_ids)
    losses, t_run = [], 0.0
    for i, k in enumerate(jax.random.split(kt, steps // chunk)):
        t0 = time.perf_counter()
        p, st, ls = run(p, st, Q, D, k, table, None if neg_ids is None else jnp.asarray(neg_ids))
        ls = np.asarray(ls)
        if i:
            t_run += time.perf_counter() - t0
        losses.append(ls)
        if not np.isfinite(ls).all():
            raise FloatingPointError(f"bi-encoder loss became non-finite near step {(i + 1) * chunk}")
        if log_every and ((i + 1) * chunk) % log_every == 0:
            print(f"step {(i + 1) * chunk:5d}  loss {ls.mean():.3f}", flush=True)
    n = steps // chunk
    return p, np.concatenate(losses), t_run * n / max(n - 1, 1)


def mine_hard_negatives(bm25: BM25, titles: list[str], own: np.ndarray, exclude: np.ndarray, depth: int = 10):
    """For each title, the highest-ranked BM25 abstract that is neither its own paper nor in `exclude` (the test
    papers, so their abstracts never enter training). DPR's recipe: negatives that look right but aren't."""
    _, top = bm25.topk(titles, depth, batch=256)
    banned = np.zeros(bm25.n_docs, bool)
    banned[exclude] = True
    neg = np.empty(len(titles), np.int64)
    rng = np.random.default_rng(0)
    for i, row in enumerate(top):
        ok = [int(d) for d in row if d != own[i] and not banned[d]]
        neg[i] = ok[0] if ok else rng.integers(bm25.n_docs)
    return neg


@jax.jit(static_argnames="cfg")
def _embed_batch(p, ids, cfg, table):
    return enc_apply(p, ids, cfg, table)


def embed(p, ids: np.ndarray, cfg: EncCfg = EncCfg(), table=None, batch: int = 512) -> np.ndarray:
    n = len(ids)
    pad = (-n) % batch                       # one shape -> one compile
    ids = np.concatenate([ids, np.zeros((pad, ids.shape[1]), ids.dtype)]) if pad else ids
    out = [np.asarray(_embed_batch(p, jnp.asarray(ids[i:i + batch]), cfg, table)) for i in range(0, len(ids), batch)]
    return np.concatenate(out)[:n]


def save_params(path: Path, p):
    flat = {jax.tree_util.keystr(k): np.asarray(v) for k, v in jax.tree_util.tree_flatten_with_path(p)[0]}
    np.savez(path, **flat)


def load_params(path: Path, cfg: EncCfg = EncCfg()):
    z = np.load(path)
    return jax.tree_util.tree_map_with_path(lambda k, _: jnp.asarray(z[jax.tree_util.keystr(k)]),
                                            enc_init(jax.random.key(0), cfg))


# ------------------------------------------------------------------------------------------- zero-shot LLM embeddings

def llm_embed(params, cfg, ids: np.ndarray, batch: int = 64) -> np.ndarray:
    """Mean-pooled final hidden states of a decoder LLM (learnjax.llama), L2-normalized. No training."""
    from learnjax import llama

    @jax.jit
    def f(params, ids):
        B, T = ids.shape
        positions = jnp.broadcast_to(jnp.arange(T), (B, T))
        x = params["embed"][ids]
        x, _ = jax.lax.scan(lambda x, lp: (llama._layer(cfg, x, lp, positions, None, 0)[0], None), x, params["layers"])
        x = llama.rms_norm(x, params["final_norm"], cfg.norm_eps).astype(jnp.float32)
        m = (ids > 0)[..., None].astype(jnp.float32)
        e = (x * m).sum(1) / jnp.maximum(m.sum(1), 1.0)
        return e / jnp.linalg.norm(e, axis=-1, keepdims=True)

    n = len(ids)
    pad = (-n) % batch
    ids = np.concatenate([ids, np.zeros((pad, ids.shape[1]), ids.dtype)]) if pad else ids
    return np.concatenate([np.asarray(f(params, jnp.asarray(ids[i:i + batch]))) for i in range(0, len(ids), batch)])[:n]


# ------------------------------------------------------------------------------------------- exact and approximate search

def exact_topk(topk_cosine, Q: np.ndarray, D: np.ndarray, k: int = 100, batch: int = 256):
    Dj = jnp.asarray(D)
    f = jax.jit(lambda q, d: topk_cosine(q, d, k))
    out = [f(jnp.asarray(Q[i:i + batch]), Dj) for i in range(0, len(Q), batch)]
    return np.concatenate([np.asarray(s) for s, _ in out]), np.concatenate([np.asarray(i) for _, i in out])


def kmeans(X: np.ndarray, n_clusters: int, iters: int = 20, seed: int = 0):
    """Spherical k-means (cosine) in JAX: assign to the nearest centroid, average, renormalize."""
    X = jnp.asarray(X)
    C = X[jax.random.choice(jax.random.key(seed), X.shape[0], (n_clusters,), replace=False)]

    @jax.jit
    def step(C):
        a = jnp.argmax(X @ C.T, -1)
        S = jax.ops.segment_sum(X, a, n_clusters)
        C_new = jnp.where(jnp.linalg.norm(S, axis=-1, keepdims=True) > 0, S, C)
        return C_new / jnp.linalg.norm(C_new, axis=-1, keepdims=True), a

    for _ in range(iters):
        C, a = step(C)
    return np.asarray(C), np.asarray(a)


@dataclass
class IVF:
    """Inverted file index: documents grouped by their nearest centroid; a query scans only the `nprobe`
    closest groups. Lists are padded to the longest one so the scan is a fixed-shape gather (-1 = empty)."""

    centroids: np.ndarray
    lists: np.ndarray
    D: np.ndarray = field(repr=False)

    @classmethod
    def build(cls, D: np.ndarray, n_lists: int, seed: int = 0) -> IVF:
        C, assign = kmeans(D, n_lists, seed=seed)
        groups = [np.flatnonzero(assign == c) for c in range(n_lists)]
        width = max(len(g) for g in groups)
        lists = np.full((n_lists, width), -1, np.int32)
        for c, g in enumerate(groups):
            lists[c, :len(g)] = g
        return cls(C, lists, D)

    def search(self, Q: np.ndarray, k: int, nprobe: int, budget: float = 2**27):
        """Batched over queries so the gathered candidates (batch, nprobe * width, dim) stay near `budget` floats
        (512 MB): all queries at once would be tens of GB at large nprobe."""
        width, dim = self.lists.shape[1], self.D.shape[1]
        bs = max(1, int(budget // (nprobe * width * dim)))
        C, L, D = jnp.asarray(self.centroids), jnp.asarray(self.lists), jnp.asarray(self.D)
        pad = (-len(Q)) % bs                                   # equal batches -> one compile per nprobe
        Qp = np.concatenate([Q, np.repeat(Q[:1], pad, 0)]) if pad else Q
        outs = [_ivf_search(jnp.asarray(Qp[i:i + bs]), C, L, D, k, nprobe) for i in range(0, len(Qp), bs)]
        return tuple(np.concatenate([np.asarray(o[j]) for o in outs])[:len(Q)] for j in range(3))


@jax.jit(static_argnames="nprobe")
def _ivf_scores(Q, C, lists, D, nprobe):
    _, probe = jax.lax.top_k(Q @ C.T, nprobe)                       # (nq, nprobe) closest lists
    cand = lists[probe].reshape(Q.shape[0], -1)                     # (nq, nprobe * width) doc ids, -1 = pad
    s = jnp.einsum("qd,qnd->qn", Q, D[jnp.maximum(cand, 0)])
    return jnp.where(cand >= 0, s, -jnp.inf), cand


def _ivf_search(Q, C, lists, D, k, nprobe):
    s, cand = _ivf_scores(Q, C, lists, D, nprobe)                   # separate programs: see _bcoo_topk
    top, j = _topk_jit(s, k=k)
    return top, jnp.take_along_axis(cand, j, 1), (cand >= 0).sum(1)


# ------------------------------------------------------------------------------------------- reranking and generation

RERANK_PROMPT = ("Paper title: {title}\n\nAbstract: {abstract}\n\n"
                 "Is this the abstract of the paper with that title? Answer yes or no.")


def make_reranker(params, cfg, tok, max_len: int = 256):
    """Score (title, abstract) pairs with SmolLM2-Instruct: logit("yes") - logit("no") at the answer position.
    A cross-encoder: the model reads both texts together, unlike the bi-encoder's two separate vectors."""
    from learnjax import llama

    yes = jnp.asarray([tok.encode(w).ids[0] for w in ("Yes", "yes")])
    no = jnp.asarray([tok.encode(w).ids[0] for w in ("No", "no")])

    @jax.jit
    def f(params, ids, last):
        logits, _ = llama.forward(params, ids, cfg)
        lg = jax.nn.log_softmax(logits[jnp.arange(ids.shape[0]), last], -1)
        return jax.nn.logsumexp(lg[:, yes], -1) - jax.nn.logsumexp(lg[:, no], -1)

    def score(title: str, abstracts: list[str]) -> np.ndarray:
        rows = []
        for a in abstracts:
            # Truncate the ABSTRACT, never the prompt: the question comes last and must survive.
            short = tok.decode(tok.encode(a).ids[:max_len - 96])
            ids = llama.chat_prompt(tok, RERANK_PROMPT.format(title=title, abstract=short))[:max_len]
            rows.append(ids)
        arr = np.zeros((len(rows), max_len), np.int32)
        last = np.zeros(len(rows), np.int32)
        for i, r in enumerate(rows):
            arr[i, :len(r)] = r
            last[i] = len(r) - 1
        return np.asarray(f(params, jnp.asarray(arr), jnp.asarray(last)))

    return score


def make_ql_reranker(params, cfg, tok, max_len: int = 256, title_max: int = 48):
    """Query likelihood (Ponte and Croft, 1998; UPR, Sachan et al., 2022): score each abstract by how probable
    the language model finds the *title* after reading it — mean log p(title tokens | "Abstract: ... Title:").
    It asks the LM to do what it was trained for (predict text) instead of judging relevance."""
    from learnjax import llama

    @jax.jit
    def f(params, ids, mask):
        logits, _ = llama.forward(params, ids, cfg)
        lp = jax.nn.log_softmax(logits[:, :-1], -1)
        tok_lp = jnp.take_along_axis(lp, ids[:, 1:, None], -1)[..., 0]
        return (tok_lp * mask[:, :-1]).sum(1) / jnp.maximum(mask[:, :-1].sum(1), 1)

    def score(title: str, abstracts: list[str]) -> np.ndarray:
        tgt = tok.encode(" " + title).ids[:title_max]
        arr = np.zeros((len(abstracts), max_len), np.int32)
        mask = np.zeros((len(abstracts), max_len), np.float32)
        for i, a in enumerate(abstracts):
            body = tok.encode("Abstract: " + a).ids[: max_len - len(tgt) - 8]
            prefix = body + tok.encode("\nTitle:").ids
            ids = prefix + tgt
            arr[i, :len(ids)] = ids
            mask[i, len(prefix) - 1:len(ids) - 1] = 1.0      # logits at t predict token t+1: the title's tokens
        return np.asarray(f(params, jnp.asarray(arr), jnp.asarray(mask)))

    return score


ANSWER_PROMPT = ("Sources:\n\n{sources}\n\nQuestion: {question}\n\nAnswer in two or three sentences using only the "
                 "sources above. After each sentence, write the number of the source it came from in brackets, like [2]. "
                 "If the sources don't answer the question, say so.")


def answer(params, cfg, tok, question: str, sources: list[str], n_new: int = 120, src_tokens: int = 110) -> str:
    from learnjax import llama

    def clip(text):
        ids = tok.encode(text).ids
        return tok.decode(ids[:src_tokens]) + (" …" if len(ids) > src_tokens else "")

    src = "\n\n".join(f"[{i + 1}] {clip(s)}" for i, s in enumerate(sources))
    prompt = llama.chat_prompt(tok, ANSWER_PROMPT.format(sources=src, question=question)) if sources else \
        llama.chat_prompt(tok, question)
    end = tok.token_to_id("<|im_end|>")
    out = generate_fixed(params, cfg, prompt, n_new, stop_ids=(end,))
    return tok.decode([t for t in out if t != end]).strip()


def generate_fixed(params, cfg, prompt_ids, n_new: int, stop_ids=(), cache_len: int = 1024, chunk: int = 64):
    """Greedy decoding with FIXED shapes, so prompts of any length reuse two compiled programs.

    `llama.generate` sizes its cache to len(prompt) + n_new, so every new prompt length recompiles (Module 18).
    Here the cache is always `cache_len` long and the prompt is prefilled in `chunk`-token pieces. The last piece
    is padded; its padding lands in cache slots *after* the prompt, which the causal mask hides and decoding
    overwrites one slot at a time.
    """
    from learnjax import llama

    T = len(prompt_ids)
    if T + n_new > cache_len:
        raise ValueError(f"prompt ({T}) + n_new ({n_new}) exceeds cache_len ({cache_len})")
    cache = llama.init_cache(cfg, 1, max_len=cache_len)
    padded = np.zeros(-(-T // chunk) * chunk, np.int32)
    padded[:T] = prompt_ids
    for start in range(0, len(padded), chunk):
        logits, cache = llama.forward_jit(params, jnp.asarray(padded[None, start:start + chunk]), cfg, cache, start)
    nxt = int(jnp.argmax(logits[0, (T - 1) - (len(padded) - chunk)]))
    out = []
    for i in range(n_new):
        out.append(nxt)
        if nxt in stop_ids:
            break
        logits, cache = llama.forward_jit(params, jnp.asarray([[nxt]], jnp.int32), cfg, cache, T + i)
        nxt = int(jnp.argmax(logits[0, -1]))
    return out


def groundedness(text: str, sources: list[str]) -> dict:
    """Cheap checks, not proof: which [n] citations appear (and are valid), and what fraction of the answer's
    content words (4+ letters) occur somewhere in the retrieved sources (`support`)."""
    cited = sorted({int(c) for c in re.findall(r"\[(\d+)\]", text)})
    valid = [c for c in cited if 1 <= c <= len(sources)]
    pool = set(words(" ".join(sources)))
    content = [w for w in words(text) if len(w) >= 4]
    support = float(np.mean([w in pool for w in content])) if content else 0.0
    return {"cited": cited, "valid": valid, "support": support}


# ------------------------------------------------------------------------------------------- evaluation

def ranks_of(ranked: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    hit = np.asarray(ranked) == np.asarray(relevant)[:, None]
    return np.where(hit.any(1), hit.argmax(1) + 1, 0)


def bootstrap_ci(per_query: np.ndarray, n: int = 2000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap over queries (Module 21): resample queries with replacement, recompute the mean."""
    rng = np.random.default_rng(seed)
    means = per_query[rng.integers(0, len(per_query), (n, len(per_query)))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(per_query.mean()), float(lo), float(hi)


def rr_at(ranked, relevant, k: int = 10) -> np.ndarray:
    r = ranks_of(ranked, relevant)
    return np.where((r > 0) & (r <= k), 1.0 / np.maximum(r, 1), 0.0)


def fuse_rrf(rrf_fuse, lists_a: np.ndarray, lists_b: np.ndarray, k: int = 60, depth: int = 100) -> np.ndarray:
    out = np.full((len(lists_a), depth), -1, np.int64)
    for i, (a, b) in enumerate(zip(lists_a, lists_b, strict=True)):
        f = rrf_fuse([list(map(int, a)), list(map(int, b))], k)[:depth]
        out[i, :len(f)] = f
    return out


def fuse_minmax_union(minmax_fuse, bm25_rows: np.ndarray, dense_rows: np.ndarray, cand: np.ndarray, w: float):
    """bm25_rows / dense_rows: (nq, n_docs) full score matrices for the queries; cand: (nq, c) candidate ids.
    Returns (nq, c) candidate ids re-ordered by the fused score."""
    out = np.empty_like(cand)
    f = jax.jit(lambda a, b: minmax_fuse(a, b, w))
    for i in range(len(cand)):
        s = np.asarray(f(jnp.asarray(bm25_rows[i, cand[i]]), jnp.asarray(dense_rows[i, cand[i]])))
        out[i] = cand[i][np.argsort(-s, kind="stable")]
    return out


# ------------------------------------------------------------------------------------------- module 39's API

@dataclass
class Index:
    """A searchable memory: BM25 over `texts`, optionally fused with a trained bi-encoder by min-max (w on BM25).

    Built by `build_index`; `ex` is the learner's exercises module (bm25_weights, minmax_fuse, topk_cosine).
    Starts empty if you like (an agent's memory does); `add` appends one text and rebuilds BM25 (fine for
    hundreds or thousands of memories; for 50k documents build once instead).
    """

    texts: list[str]
    ex: object
    bm25: BM25 | None = None
    encoder: dict | None = None
    doc_emb: np.ndarray | None = None
    tok: object = None
    cfg: EncCfg = EncCfg()
    table: object = None
    w: float = 0.6

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Top-k (doc_id, score) for one query, best first. doc_id indexes `texts`."""
        if not self.texts:
            return []
        depth = min(max(k, 50), len(self.texts))
        s, i = self.bm25.topk([query], k=depth)
        if self.encoder is None:
            return [(int(d), float(v)) for d, v in zip(i[0][:k], s[0][:k], strict=True) if v > 0]
        q = embed(self.encoder, encode_ids(self.tok, [query], self.cfg.q_len), self.cfg, self.table, batch=1)
        _, di = jax.lax.top_k(jnp.asarray(q @ self.doc_emb.T), depth)
        cand = np.unique(np.concatenate([i[0], np.asarray(di)[0]]))
        bm = self.bm25.scores_for([query], cand[None])[0]
        dn = self.doc_emb[cand] @ q[0]
        fused = np.asarray(self.ex.minmax_fuse(jnp.asarray(bm), jnp.asarray(dn), self.w))
        order = np.argsort(-fused, kind="stable")[:k]
        return [(int(cand[j]), float(fused[j])) for j in order]

    def add(self, text: str) -> int:
        """Append one text; returns its doc_id."""
        self.texts.append(text)
        self.bm25 = BM25.build(self.texts, self.ex.bm25_weights)
        if self.encoder is not None:
            e = embed(self.encoder, encode_ids(self.tok, [text], self.cfg.d_len), self.cfg, self.table, batch=1)
            self.doc_emb = e if self.doc_emb is None else np.concatenate([self.doc_emb, e])
        return len(self.texts) - 1


def load_encoder(cfg: EncCfg = EncCfg()):
    """train.py's bi-encoder parameters, or None if it hasn't been run."""
    path = RUNS / "encoder.npz"
    return load_params(path, cfg) if path.exists() else None


def build_index(texts: list[str], ex, encoder: dict | None = None, cfg: EncCfg = EncCfg(), w: float = 0.6) -> Index:
    """BM25 index over `texts` (may be empty); pass `encoder` (e.g. `load_encoder()`) for min-max hybrid search."""
    texts = list(texts)
    idx = Index(texts, ex, BM25.build(texts, ex.bm25_weights) if texts else None, cfg=cfg, w=w)
    if encoder is not None:
        idx.encoder, idx.tok = encoder, tokenizer()
        idx.table = pretrained_table() if cfg.pretrained else None
        if texts:
            idx.doc_emb = embed(encoder, encode_ids(idx.tok, texts, cfg.d_len), cfg, idx.table)
    return idx
