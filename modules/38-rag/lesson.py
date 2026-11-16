import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="38 · Retrieval-augmented generation")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import json
    import sys
    import time
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    _spec = importlib.util.spec_from_file_location("rag38", mo.notebook_dir() / "rag.py")
    rag = importlib.util.module_from_spec(_spec)
    sys.modules["rag38"] = rag
    _spec.loader.exec_module(rag)
    RUNS = rag.RUNS
    return RUNS, ex, jax, jnp, json, lj, mo, np, plt, rag, time


@app.cell
def _(mo):
    mo.md(r"""
    # 38 · Retrieval-augmented generation, from BM25 to rerankers

    *Part 11 — Systems around the model. Take Modules 21 (evaluation), 27 (LM Studio) and 33 (CLIP) first.*

    **Goal:** build every stage of a retrieval stack in JAX on your card, measure each one honestly, and see
    where a production knowledge base (your [aws-ai](https://github.com/joebarbere/aws-ai) `local-kb`) makes different choices.

    **You will:** turn 49,950 radio-astronomy abstracts into a benchmark · score them with BM25 as one sparse
    matrix product · train a bi-encoder with InfoNCE · fuse the two (RRF and min-max) · compare exact search
    with a hand-written IVF index · try chunk sizes · rerank with SmolLM2 reading query and document together ·
    let SmolLM2 answer questions with citations, and check whether it stuck to its sources.
    """)
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **Where this came from.** Information retrieval is older than neural networks. Gerard Salton's SMART
    system (1960s) ranked documents by vector similarity of word counts, and Karen Spärck Jones (1972) added
    *inverse document frequency*: rare words matter more. Stephen Robertson's probabilistic model became
    **BM25** at TREC in 1994, and BM25 still powers Lucene, Elasticsearch and OpenSearch. Neural retrieval
    took over question answering in 2020: **DPR** (Karpukhin et al.) trained a BERT bi-encoder with in-batch
    negatives, **ColBERT** (Khattab and Zaharia) kept one vector per token, and **RAG** (Lewis et al., Meta)
    fed retrieved passages to a generator. DeepMind's **RETRO** (2021) put retrieval inside the model. After
    ChatGPT, "RAG" came to mean any LLM answering from a search index, and vector databases followed.
    Timeline: Module 00b.
    """), kind="info")
    return


@app.cell
def _(mo, rag, time):
    from learnjax.data import radio_abstracts

    _t = time.perf_counter()
    try:
        papers = radio_abstracts()
        _err = None
    except FileNotFoundError as _e:
        papers, _err = None, str(_e)
    if papers is None:
        test_ids = train_ids = None
        _out = mo.callout(mo.md(f"**The corpus isn't in the cache yet.** {_err}\n\n"
                                "`make rag-data` copies it from your aws-ai checkout if it's there, otherwise it "
                                "fetches it from the arXiv API at one request every 3 s (~15 min)."), kind="warn")
    else:
        test_ids, train_ids = rag.split(papers)
        _p = next(papers[int(i)] for i in test_ids if "$" not in papers[int(i)]["abstract"])   # no LaTeX in the example
        _years = sorted(int(p["published"][:4]) for p in papers)
        _out = mo.md(f"""
    ## 1 · The corpus and the benchmark

    **{len(papers):,} arXiv abstracts** about radio astronomy, {_years[0]}–{_years[-1]}, loaded in
    {time.perf_counter() - _t:.1f} s. It's the "large" corpus of your aws-ai `local-kb` benchmark, so the two
    can be compared. One of them:

    > **{_p['title']}** ({_p['published'][:4]}, {_p['category']})
    >
    > {_p['abstract'][:600]}…

    **The task:** the *title* is the query and the paper's own *abstract* is the one right answer, searched
    among all {len(papers):,} abstracts. **{len(test_ids):,} papers**, picked at random, are the test set: their
    titles are the queries, and their (title, abstract) pairs never train anything. Every abstract stays in
    the index, so each query has 49,949 wrong answers to beat.

    One relevant document per query keeps the metrics simple: **recall@k** (is it in the top k?), **MRR@10**
    (1/rank, so rank 1 scores 1 and rank 2 scores 0.5), and **nDCG@10** (1/log₂(rank + 1), a gentler discount).
    """)
    _out
    return papers, test_ids, train_ids


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Your turn: chunks, BM25 weights and the metrics

    * **`chunk_sentences`** packs whole sentences into chunks of at most `max_tokens`, with `overlap`
      sentences repeated between neighbours. `local-kb` chunks the same way (300 tokens, 20% overlap, whole
      sentences) so that a chunk never ends mid-thought.
    * **`bm25_weights`** is the heart of BM25. A term's weight in a document grows with how often it appears
      there ($tf$), but *saturates* (the 10th "pulsar" adds less than the 2nd), is damped in long documents,
      and is scaled by how rare the term is across the corpus ($idf$):

    $$w(t, d) = \underbrace{\log\left(1 + \frac{N - df_t + 0.5}{df_t + 0.5}\right)}_{idf}\cdot
    \frac{tf\,(k_1 + 1)}{tf + k_1\left(1 - b + b\,\frac{|d|}{\overline{|d|}}\right)}$$

    A query's score for a document is the sum of the weights of the query's words. $k_1 = 1.5$ sets how fast
    $tf$ saturates and $b = 0.75$ how much length matters (Lucene's defaults).
    * **`recall_at_k`, `mrr_at_k`, `ndcg_at_k`** turn ranked lists into the numbers above.
    """)
    return


@app.cell
def _(ex, lj, mo, np):
    _rows = []
    _c = lj.attempt(ex.chunk_sentences, [12, 30, 25, 8, 40, 15], 60, 1)
    _rows.append(mo.md(f"**`chunk_sentences([12, 30, 25, 8, 40, 15], max_tokens=60, overlap=1)`** → "
                       f"`{_c.value}`") if _c.ok else _c.view(mo))
    _w = lj.attempt(ex.bm25_weights, np.array([1, 3, 1, 1]), np.array([140, 140, 140, 400]),
                    np.array([20, 20, 30000, 20]), 49950, 140.0)
    if _w.ok:
        _v = np.asarray(_w.value)
        _rows.append(mo.md(f"""**`bm25_weights`** for four (term, document) pairs in a 49,950-document corpus:

| term | appears in | tf | doc length | weight |
|---|---|---|---|---|
| a rare word ("magnetar") | 20 docs | 1 | 140 | {_v[0]:.2f} |
| the same word, 3 times | 20 docs | 3 | 140 | {_v[1]:.2f} |
| a common word ("observations") | 30,000 docs | 1 | 140 | {_v[2]:.2f} |
| the rare word in a long document | 20 docs | 1 | 400 | {_v[3]:.2f} |

Three times the occurrences is worth well under 3× the weight (saturation), a word in most documents is
worth almost nothing, and the same word counts less in a long document."""))
    else:
        _rows.append(_w.view(mo))
    _rk = np.array([[7, 3, 9], [1, 2, 3], [4, 5, 6]])
    _rel = np.array([3, 1, 0])
    _m = [lj.attempt(f, _rk, _rel, 3) for f in (ex.recall_at_k, ex.mrr_at_k, ex.ndcg_at_k)]
    if all(a.ok for a in _m):
        _rows.append(mo.md(f"**Metrics** on three toy queries (relevant doc at rank 2, rank 1, missing): "
                           f"recall@3 {_m[0].value:.3f}, MRR@3 {_m[1].value:.3f}, nDCG@3 {_m[2].value:.3f}."))
    else:
        _rows.extend(a.view(mo) for a in _m if not a.ok)
    metrics_ready = _w.ok and all(a.ok for a in _m)
    mo.vstack(_rows)
    return (metrics_ready,)


@app.cell
def _(ex, metrics_ready, mo, np, papers, rag, test_ids, time):
    if papers is None or not metrics_ready:
        bm25 = bm25_top = bm25_scores = None
        _out = mo.md("*(BM25 runs once `bm25_weights` and the metrics work.)*")
    else:
        _queries = [papers[int(i)]["title"] for i in test_ids]
        _t = time.perf_counter()
        bm25 = rag.BM25.build([p["abstract"] for p in papers], ex.bm25_weights)
        _build = time.perf_counter() - _t
        _t = time.perf_counter()
        bm25_scores, bm25_top = bm25.topk(_queries, 100)
        _search = time.perf_counter() - _t
        _easy = rag.BM25.build([f"{p['title']}. {p['abstract']}" for p in papers], ex.bm25_weights)
        _, _easy_top = _easy.topk(_queries, 100)
        _nnz = bm25.W.nse
        _V = len(bm25.vocab)
        _q = _queries[0]
        _hits = "\n".join(f"{r + 1}. {'**✓**' if int(d) == int(test_ids[0]) else ''} {papers[int(d)]['title']}"
                          for r, d in enumerate(bm25_top[0][:3]))
        _out = mo.md(f"""
## 3 · BM25 as one sparse matrix product

Every abstract becomes a row of a **{len(papers):,} × {_V:,}** matrix of BM25 weights (documents × words).
Almost all entries are zero, so it's stored as a JAX **BCOO** sparse array ([`jax.experimental.sparse.BCOO`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO)): {_nnz:,} non-zeros, {_nnz / (len(papers) * _V):.3%}
of the full matrix. A query is a 0/1 column marking its words, so a whole batch of queries scores against the whole
corpus as **one sparse × dense matmul**: `W @ Q`, shape ({len(papers):,}, {_V:,}) @ ({_V:,}, 64). An inverted
index (term → documents) is the same matrix stored by columns; search engines walk it term by term.

Built in **{_build:.1f} s** (Python tokenizing and counting), then {len(test_ids):,} queries ranked against every
abstract in **{_search:.2f} s** on the GPU, compile included.

| index text | MRR@10 | recall@1 | recall@10 | recall@100 |
|---|---|---|---|---|
| abstract only (**this module's benchmark**) | **{ex.mrr_at_k(bm25_top, test_ids, 10):.3f}** | {ex.recall_at_k(bm25_top, test_ids, 1):.3f} | {ex.recall_at_k(bm25_top, test_ids, 10):.3f} | {ex.recall_at_k(bm25_top, test_ids, 100):.3f} |
| "title. abstract" (`local-kb`'s setup) | {ex.mrr_at_k(_easy_top, test_ids, 10):.3f} | {ex.recall_at_k(_easy_top, test_ids, 1):.3f} | {ex.recall_at_k(_easy_top, test_ids, 10):.3f} | {ex.recall_at_k(_easy_top, test_ids, 100):.3f} |

**Same algorithm, very different numbers — because of how the benchmark was built.** `local-kb` indexes
"title. abstract", so each query is copied word for word into the start of its own answer; its code says
so ("easy for keyword search, a fair test of ranking at scale"). That's a fine test of a vector store's
plumbing, but it can't tell good retrievers apart. Indexing abstracts alone makes the query and its answer
share *topic*, not text, which is what real questions look like. Module 21's lesson again: the test set
decides what a number means. The rest of this module uses the abstract-only index.

Query: *"{_q}"* — BM25's top 3:

{_hits}
""")
    _out
    return bm25, bm25_scores, bm25_top


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Dense retrieval: train a bi-encoder

    BM25 only matches words that literally appear. A **bi-encoder** maps any text to a vector so that a title
    and its own abstract land close together, even when they share no words ("FRB" vs "fast radio burst").
    One encoder embeds both sides (Sentence-BERT's recipe): token vectors, 4 Transformer layers with
    *bidirectional* attention (Module 32's BERT), mean-pooling, a projection, and L2 normalization.

    It learns from (title, abstract) pairs with **InfoNCE and in-batch negatives** (DPR, 2020): in a batch of
    256 pairs, title *i*'s positive is abstract *i* and the other 255 abstracts are free negatives. That's a
    256-way classification per row — exactly half of Module 33's CLIP loss, and a cousin of Module 30's
    word2vec negative sampling.

    **Your turn:** `info_nce(q, d, temperature)` and `topk_cosine(Q, D, k)` (exact nearest neighbours).
    """)
    return


@app.cell
def _(ex, jnp, lj, mo):
    _e = jnp.eye(4)
    _a = lj.attempt(ex.info_nce, _e, _e, 0.05)
    _b = lj.attempt(ex.topk_cosine, jnp.array([[5.0, 0.1]]), jnp.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]]), 2)
    dense_ready = _a.ok and _b.ok
    _items = [mo.md(f"`info_nce` on perfectly aligned pairs: {abs(float(_a.value)):.1e} (should be ~0)") if _a.ok else _a.view(mo),
              mo.md(f"`topk_cosine` → indices {[int(i) for i in _b.value[1][0]]} (should be [0, 2])") if _b.ok else _b.view(mo)]
    mo.vstack(_items)
    return (dense_ready,)


@app.cell
def _(RUNS, dense_ready, ex, json, mo, np, papers, rag, test_ids, time, train_ids):
    LIVE_STEPS = 150
    enc_params = enc_table = dense_docs = dense_queries = enc_losses = None
    enc_source, enc_candidates = "", 256
    if papers is not None and dense_ready:
        _t0 = time.perf_counter()
        enc_cfg = rag.EncCfg()
        enc_tok = rag.tokenizer()
        enc_table = rag.pretrained_table() if enc_cfg.pretrained else None
        _Qi = rag.encode_ids(enc_tok, [p["title"] for p in papers], enc_cfg.q_len)
        if (RUNS / "encoder.npz").exists() and (RUNS / "meta.json").exists():
            _meta = json.loads((RUNS / "meta.json").read_text())
            enc_params = rag.load_params(RUNS / "encoder.npz", enc_cfg)
            dense_docs = np.load(RUNS / "encoder_docs.npy").astype(np.float32)
            enc_losses = np.repeat(np.asarray(_meta["losses"]), 10)
            enc_candidates = 512 if _meta.get("hard_negatives") else 256
            enc_source = (f"Loaded `train.py`'s encoder: **{_meta['steps']:,} steps** ({_meta['run_seconds'] / 60:.0f} min "
                          "of training), with all abstracts already embedded.")
        else:
            _Di = rag.encode_ids(enc_tok, [p["abstract"] for p in papers], enc_cfg.d_len)
            enc_params, enc_losses, _run = rag.train_encoder(ex.info_nce, _Qi[train_ids], _Di[train_ids], LIVE_STEPS,
                                                             enc_cfg, table=enc_table)
            dense_docs = rag.embed(enc_params, _Di, enc_cfg, enc_table)
            enc_source = (f"Trained live for **{LIVE_STEPS} steps** with your `info_nce` ({_run:.0f} s), then embedded all "
                          f"{len(papers):,} abstracts. For the full model run `containers/run.sh python "
                          "modules/38-rag/train.py` once (~20 min); this cell then loads it.")
        dense_queries = rag.embed(enc_params, _Qi[test_ids], enc_cfg, enc_table)
        enc_source += f" ({time.perf_counter() - _t0:.0f} s for this cell.)"
    else:
        enc_cfg = enc_tok = None
    mo.md(enc_source)
    return dense_docs, dense_queries, enc_candidates, enc_cfg, enc_losses, enc_params, enc_table, enc_tok


@app.cell
def _(RUNS, dense_docs, dense_queries, enc_candidates, enc_losses, ex, mo, np, plt, rag, test_ids):
    if dense_docs is None:
        dense_top = None
        _out = mo.md("")
    else:
        _, dense_top = rag.exact_topk(ex.topk_cosine, dense_queries, dense_docs, 100)
        _rows = [("your bi-encoder", dense_top)]
        _llm_note = ""
        if (RUNS / "llm_docs.npy").exists():
            _llm_note = ("An LLM's hidden states, averaged, are *not* an embedding: untrained mean pooling of SmolLM2 "
                         "barely beats chance. Contrastive training is what makes vectors comparable by cosine.")
            _, _llm_top = rag.exact_topk(ex.topk_cosine, np.load(RUNS / "llm_queries.npy").astype(np.float32),
                                         np.load(RUNS / "llm_docs.npy").astype(np.float32), 100)
            _rows.insert(0, ("SmolLM2-135M, mean-pooled, untrained", _llm_top))
        _tbl = "\n".join(f"| {n} | {ex.mrr_at_k(t, test_ids, 10):.3f} | {ex.recall_at_k(t, test_ids, 10):.3f} | "
                         f"{ex.recall_at_k(t, test_ids, 100):.3f} |" for n, t in _rows)
        _fig, _ax = plt.subplots(figsize=(6.5, 2.8))
        _ax.plot(np.convolve(enc_losses, np.ones(20) / 20, mode="valid"))
        _ax.axhline(np.log(enc_candidates), color="gray", ls=":", label=f"chance: ln {enc_candidates}")
        _ax.set(xlabel="step", ylabel="InfoNCE loss",
                title=f"bi-encoder training (each title vs {enc_candidates} abstracts)")
        _ax.legend()
        _ax.grid(alpha=0.3)
        _fig.tight_layout()
        _out = mo.vstack([_fig, mo.md(f"""
| dense retriever | MRR@10 | recall@10 | recall@100 |
|---|---|---|---|
{_tbl}

{_llm_note}

The training loss measures a {enc_candidates}-way choice inside one batch; retrieval is a 49,950-way choice, which is
much harder. What was tried while building this module (2,000 steps each, MRR@10 on these test titles,
*measured when written*):

| bi-encoder variant | MRR@10 |
|---|---|
| all weights from scratch (400 steps) | 0.085 |
| token vectors from SmolLM2-135M's embedding table, frozen | 0.294 |
| + Inverse Cloze Task pairs (a random window of the abstract as a pseudo-query; Lee et al., 2019) | 0.269 |
| + one **BM25 hard negative** per pair (`train.py`'s recipe; DPR, 2020) | 0.301 |

Pretrained token vectors matter most; hard negatives (abstracts that BM25 thinks match, but don't) help a
little; ICT didn't help here. **The honest summary: this bi-encoder is well below BM25 on this task.** DPR beat
BM25 because it started from BERT, pretrained on 3.3 billion words; only our *token vectors* are pretrained,
and the 4 layers above them learn from ~49,000 titles. Closing the gap takes a pretrained encoder trained on
far more pairs, which is what Qwen3-Embedding is (`local-kb`'s embedder: a whole pretrained LLM fine-tuned
with this same loss on hundreds of millions of pairs). Section 5 asks whether a weaker retriever can still
help a stronger one.
""")])
    _out
    return (dense_top,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Hybrid search: fuse the two rankings

    BM25 and the bi-encoder fail differently: BM25 misses paraphrases, the bi-encoder blurs rare exact terms
    (a source name, an instrument). Two standard ways to combine them:

    * **Reciprocal rank fusion** (`rrf_fuse`): ignore the scores and add $1/(k + \text{rank})$ from each list.
      No tuning, and robust because BM25 scores (0–40) and cosines (−1–1) are never compared directly.
    * **Min-max fusion** (`minmax_fuse`): rescale each retriever's scores over the candidates to [0, 1] and mix
      with a weight $w$. One knob to tune, and it uses *how much* better a document scored, not just its rank.

    `local-kb` uses min-max 0.5/0.5 in OpenSearch and RRF with $k = 60$ in pgvector.
    """)
    return


@app.cell
def _(bm25, bm25_top, dense_docs, dense_queries, dense_top, enc_cfg, enc_params, enc_table, enc_tok, ex, jax, jnp,
      lj, mo, np, papers, plt, rag, test_ids):
    _a = lj.attempt(ex.rrf_fuse, [[1, 2, 3], [3, 1]], 60)
    _b = lj.attempt(ex.minmax_fuse, jnp.array([0.0, 5.0, 10.0]), jnp.array([3.0, 2.0, 1.0]), 0.25)
    _items = [mo.md(f"`rrf_fuse([[1, 2, 3], [3, 1]])` → {_a.value} (should be [1, 3, 2])") if _a.ok else _a.view(mo),
              mo.md(f"`minmax_fuse(..., w=0.25)` → {np.round(np.asarray(_b.value), 3).tolist()} "
                    "(should be [0.75, 0.5, 0.25])") if _b.ok else _b.view(mo)]
    retrieve, first_stage, results = None, "BM25", None
    if bm25 is not None:
        def retrieve(queries, k):                                     # BM25 only until the dense side works
            return bm25.topk(queries, k)[1]
    if dense_top is not None and _a.ok and _b.ok:
        _queries = [papers[int(i)]["title"] for i in test_ids]

        def _fused(b_top, d_top, d_q, queries, w):
            """Min-max fusion over the union of both top lists (fixed width 200: duplicates allowed, removed after)."""
            _cand = np.concatenate([b_top, d_top], axis=1)
            _bm = bm25.scores_for(queries, _cand)
            _dn = np.einsum("qd,qcd->qc", d_q, dense_docs[_cand])
            try:
                _f = np.asarray(jax.vmap(lambda a, b: ex.minmax_fuse(a, b, w))(jnp.asarray(_bm), jnp.asarray(_dn)))
            except Exception:  # noqa: BLE001 - a learner's version may not vmap; fall back to a loop
                _f = np.stack([np.asarray(ex.minmax_fuse(jnp.asarray(a), jnp.asarray(b), w))
                               for a, b in zip(_bm, _dn, strict=True)])
            _order = np.argsort(-_f, axis=1, kind="stable")
            _out = np.full((len(_cand), 100), -1)
            for _q in range(len(_cand)):
                _ids = list(dict.fromkeys(_cand[_q, _order[_q]].tolist()))[:100]
                _out[_q, :len(_ids)] = _ids
            return _out

        _rrf = rag.fuse_rrf(ex.rrf_fuse, bm25_top, dense_top)
        _ws = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
        _mm = {w: _fused(bm25_top, dense_top, dense_queries, _queries, w) for w in _ws}
        _best_w = max(_ws, key=lambda w: ex.mrr_at_k(_mm[w], test_ids, 10))
        results = {"BM25": bm25_top, "bi-encoder": dense_top, "RRF (k = 60)": _rrf,
                   "min-max, w = 0.5": _mm[0.5], f"min-max, best w = {_best_w}": _mm[_best_w]}
        _fig, _ax = plt.subplots(figsize=(6.5, 2.8))
        _ax.plot(_ws, [ex.mrr_at_k(_mm[w], test_ids, 10) for w in _ws], "o-", label="min-max fusion")
        _ax.axhline(ex.mrr_at_k(_rrf, test_ids, 10), color="C1", ls="--", label="RRF")
        _ax.set(xlabel="w (weight on BM25; 0 = dense only, 1 = BM25 only)", ylabel="MRR@10", title="mixing the two")
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.3)
        _fig.tight_layout()
        _ci = {n: rag.bootstrap_ci(rag.rr_at(t, test_ids, 10)) for n, t in results.items()}
        _tbl = "\n".join(f"| {n} | **{_ci[n][0]:.3f}** ({_ci[n][1]:.3f}–{_ci[n][2]:.3f}) | "
                         f"{ex.recall_at_k(t, test_ids, 10):.3f} | {ex.ndcg_at_k(t, test_ids, 10):.3f} |"
                         for n, t in results.items())

        def _dense(queries):
            _q = rag.embed(enc_params, rag.encode_ids(enc_tok, list(queries), enc_cfg.q_len), enc_cfg, enc_table)
            return _q, rag.exact_topk(ex.topk_cosine, _q, dense_docs, 100)[1]

        _pick = max(["BM25", "RRF (k = 60)", f"min-max, best w = {_best_w}"], key=lambda n: _ci[n][0])
        if _pick == "BM25":
            first_stage = "BM25"
        elif _pick.startswith("RRF"):
            first_stage = "RRF(BM25, bi-encoder)"

            def retrieve(queries, k):                                 # noqa: F811
                return rag.fuse_rrf(ex.rrf_fuse, bm25.topk(queries, 100)[1], _dense(queries)[1])[:, :k]
        else:
            first_stage = f"min-max(BM25, bi-encoder), w = {_best_w}"

            def retrieve(queries, k):                                 # noqa: F811
                _q, _d_top = _dense(queries)
                return _fused(bm25.topk(queries, 100)[1], _d_top, _q, list(queries), _best_w)[:, :k]

        _items += [_fig, mo.md(f"""
**{len(test_ids):,} test titles, abstract-only index** (95% bootstrap intervals over queries, Module 21):

| method | MRR@10 | recall@10 | nDCG@10 |
|---|---|---|---|
{_tbl}

**Measured in `local-kb` on its easier "title. abstract" index** (200 titles; not comparable to the rows above,
see section 3): Qwen3-Embedding 0.925, OpenSearch BM25 0.955, OpenSearch hybrid (min-max) 0.966, pgvector
hybrid (RRF) 0.988.

Read the intervals before the bold numbers: two methods whose intervals overlap heavily haven't been shown
to differ. Fusion only helps when the second retriever is *right where the first is wrong*; a weak dense
model mostly adds noise to RRF, while min-max with a tuned weight can lean on the stronger side. (Tuning $w$ on
the test queries, as this plot does, flatters it slightly; properly you'd tune on a separate validation set.)

From here on the first stage is the best of these: **{first_stage}**.
""")]
    mo.vstack(_items)
    return first_stage, results, retrieve

@app.cell
def _(dense_docs, dense_queries, ex, mo, np, plt, rag, test_ids, time):
    if dense_docs is None:
        ivf_rows = None
        _out = mo.md("")
    else:
        _Q = np.asarray(dense_queries, np.float32)
        _D = np.asarray(dense_docs, np.float32)
        _, _exact = rag.exact_topk(ex.topk_cosine, _Q, _D, 10)
        rag.exact_topk(ex.topk_cosine, _Q, _D, 10)                     # warm
        _t = time.perf_counter()
        rag.exact_topk(ex.topk_cosine, _Q, _D, 10)
        _exact_ms = (time.perf_counter() - _t) * 1e3
        _t = time.perf_counter()
        _ivf = rag.IVF.build(_D, n_lists=224)
        _build = time.perf_counter() - _t
        ivf_rows = []
        for _np_ in (1, 2, 4, 8, 16, 32):
            _ivf.search(_Q, 10, _np_)                                   # compile this nprobe
            _t = time.perf_counter()
            _s, _i, _n = _ivf.search(_Q, 10, _np_)
            _i = np.asarray(_i)
            _ms = (time.perf_counter() - _t) * 1e3
            _overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(_i, _exact, strict=True)])
            ivf_rows.append({"nprobe": _np_, "recall": _overlap, "scanned": float(np.mean(np.asarray(_n))) / len(_D),
                             "ms": _ms, "mrr": ex.mrr_at_k(_i, test_ids, 10)})
        _fig, _ax = plt.subplots(figsize=(6.5, 3))
        _ax.plot([r["scanned"] * 100 for r in ivf_rows], [r["recall"] for r in ivf_rows], "o-")
        for _r in ivf_rows:
            _ax.annotate(f"nprobe {_r['nprobe']}", (_r["scanned"] * 100, _r["recall"]), fontsize=7,
                         xytext=(4, -10), textcoords="offset points")
        _ax.set(xscale="log", xlabel="% of the corpus scanned per query (log)", ylabel="recall@10 vs exact",
                title="IVF: scan less, miss more", ylim=(0, 1.05))
        _ax.grid(alpha=0.3)
        _fig.tight_layout()
        _tbl = "\n".join(f"| {r['nprobe']} | {r['scanned']:.1%} | {r['recall']:.3f} | {r['mrr']:.3f} | {r['ms']:.1f} ms |"
                         for r in ivf_rows)
        _out = mo.vstack([mo.md(f"""
## 6 · Exact search, and the approximate kind vector databases use

Exact search is your `topk_cosine`: one ({len(_Q):,} × {_D.shape[1]}) @ ({_D.shape[1]} × {len(_D):,}) matmul and a
top-k. All {len(_Q):,} queries take **{_exact_ms:.1f} ms** on the GPU.

Vector databases can't afford that at a billion vectors, so they search **approximately**. The simplest
scheme is an **inverted file (IVF)**, the core of Faiss: cluster the documents with k-means (here 224
clusters, built in {_build:.1f} s with a few lines of JAX in `rag.kmeans`), store each cluster's members as a
list, and at query time scan only the `nprobe` lists whose centroids are closest. HNSW, which both
OpenSearch and pgvector use in `local-kb`, is a graph instead of clusters, but trades recall for speed the
same way.

| nprobe | corpus scanned | recall@10 vs exact | MRR@10 | time, {len(_Q):,} queries |
|---|---|---|---|---|
{_tbl}
"""), _fig, mo.md(f"""
**At {len(_D):,} vectors, approximate search barely matters on a GPU.** IVF at nprobe 8 is
{_exact_ms / ivf_rows[3]['ms']:.1f}× faster than exact search, but exact search is already {_exact_ms / len(_Q) * 1e3:.0f} µs per query,
and the speedup costs {1 - ivf_rows[3]['recall']:.0%} of the true top 10. ANN earns its keep at 10⁸–10⁹ vectors, or on a CPU (`local-kb`'s stores run on the CPU, which is why HNSW matters there: its pgvector store
reached recall@20 0.964 at the default `ef_search=40` and 1.000 at 512). What the table *does* show is the
trade every vector database tunes: the fewer lists you scan, the more true neighbours you miss.
""")])
    _out
    return (ivf_rows,)


@app.cell
def _(ex, metrics_ready, mo, np, papers, rag, test_ids, time):
    if papers is None or not metrics_ready:
        chunk_rows = None
        _out = mo.md("")
    else:
        _queries = [papers[int(i)]["title"] for i in test_ids]
        _sents = [rag.sentences(p["abstract"]) for p in papers]
        chunk_rows = []
        _t0 = time.perf_counter()
        for _max_tok, _ov in ((16, 0), (40, 0), (40, 1), (80, 1), (None, 0)):
            if _max_tok is None:
                _texts, _owner = [p["abstract"] for p in papers], np.arange(len(papers))
            else:
                _texts, _owner = [], []
                for _d, _ss in enumerate(_sents):
                    _n = [len(rag.words(s)) for s in _ss]
                    for _a, _b in ex.chunk_sentences(_n, _max_tok, _ov):
                        _texts.append(" ".join(_ss[_a:_b]))
                        _owner.append(_d)
                _owner = np.asarray(_owner)
            _idx = rag.BM25.build(_texts, ex.bm25_weights)
            _, _top = _idx.topk(_queries, 100)
            _docs = np.full((len(_queries), 10), -1)
            for _q, _row in enumerate(_owner[_top]):          # best chunk wins: a document ranks by its best chunk
                _seen = list(dict.fromkeys(_row.tolist()))[:10]
                _docs[_q, :len(_seen)] = _seen
            chunk_rows.append({"max": _max_tok, "overlap": _ov, "chunks": len(_texts),
                               "mrr": ex.mrr_at_k(_docs, test_ids, 10), "r10": ex.recall_at_k(_docs, test_ids, 10)})
        _label = {r["max"]: ("whole abstract" if r["max"] is None else f"≤ {r['max']} words") for r in chunk_rows}
        _tbl = "\n".join(f"| {_label[r['max']]} | {r['overlap']} | {r['chunks']:,} | {r['mrr']:.3f} | {r['r10']:.3f} |"
                         for r in chunk_rows)
        _out = mo.md(f"""
## 7 · Chunk size

Abstracts are short, but real documents (`local-kb`'s PDFs, your notes) are not: they get cut into
chunks, each chunk is indexed on its own, and a document ranks by its best chunk. With your
`chunk_sentences`, BM25 on abstracts cut at different sizes ({time.perf_counter() - _t0:.0f} s):

| chunk | overlap (sentences) | chunks | MRR@10 | recall@10 |
|---|---|---|---|---|
{_tbl}

For *this* task — a title that summarizes the whole paper — the whole abstract wins: a title's words are
spread across the abstract, and small chunks each see only a few of them. A question about one specific
fact ("what frequency did they observe at?") would favour small chunks, where that fact isn't diluted.
There is no universally right chunk size, only one that suits your queries — so measure it, which
`local-kb` hasn't done yet for its four chunking strategies.
""")
    _out
    return (chunk_rows,)


@app.cell
def _(mo, time):
    from learnjax import llama

    _t = time.perf_counter()
    lm_params, lm_cfg, lm_tok = llama.load(llama.SMOLLM_360M, max_seq=1024)
    mo.md(f"*SmolLM2-360M-Instruct loaded in {time.perf_counter() - _t:.1f} s (Module 18's `learnjax.llama`).*")
    return lm_cfg, lm_params, lm_tok


@app.cell
def _(ex, first_stage, lm_cfg, lm_params, lm_tok, mo, np, papers, rag, retrieve, test_ids, time):
    if retrieve is None:
        rerank_result = None
        _out = mo.md("")
    else:
        _n, _depth = 50, 20
        _qids = test_ids[:_n]
        _titles = [papers[int(i)]["title"] for i in _qids]
        _cand = retrieve(_titles, _depth)
        _abs = [[papers[int(d)]["abstract"] for d in c] for c in _cand]
        _scorers = {"yes/no judge": rag.make_reranker(lm_params, lm_cfg, lm_tok),
                    "query likelihood": rag.make_ql_reranker(lm_params, lm_cfg, lm_tok)}
        rerank_result = {"first stage": ex.mrr_at_k(_cand, _qids, 10), "ceiling": ex.recall_at_k(_cand, _qids, _depth)}
        _moves, _secs = {}, {}
        for _name, _score in _scorers.items():
            _score(_titles[0], _abs[0])                                           # compile
            _t = time.perf_counter()
            _rr = np.stack([c[np.argsort(-_score(t, a), kind="stable")] for t, a, c in zip(_titles, _abs, _cand, strict=True)])
            _secs[_name] = time.perf_counter() - _t
            rerank_result[_name] = ex.mrr_at_k(_rr, _qids, 10)
            _rb, _ra = rag.ranks_of(_cand, _qids), rag.ranks_of(_rr, _qids)
            _moves[_name] = (int(np.sum((_ra > 0) & ((_rb == 0) | (_ra < _rb)))), int(np.sum((_rb > 0) & (_ra > _rb))))
        _out = mo.md(f"""
## 8 · Reranking: read the query and the document together

A bi-encoder squeezes each text into one vector *before* it sees the query, so it can be precomputed for
50,000 documents. A **cross-encoder** reads query and document *together* — every query word can attend to
every document word — which is far more accurate and far too slow to run on the whole corpus. So production
systems retrieve ~20–100 candidates cheaply, then rerank only those (Nogueira and Cho, 2019).

SmolLM2-360M-Instruct as the cross-encoder, two ways, on the top {_depth} candidates of {_n} test queries:

* **yes/no judge:** *"Is this the abstract of the paper with that title? Answer yes or no."*, scored by
  log p("Yes") − log p("No") at the answer position.
* **query likelihood** (Ponte and Croft, 1998; UPR, Sachan et al., 2022): no question at all; score each
  abstract by how probable the model finds the *title* after reading it, log p(title | abstract).

| | MRR@10 | right abstract moved up / down | time |
|---|---|---|---|
| first stage: {first_stage} | {rerank_result['first stage']:.3f} | | |
| + yes/no judge | **{rerank_result['yes/no judge']:.3f}** | {_moves['yes/no judge'][0]} / {_moves['yes/no judge'][1]} | {_secs['yes/no judge']:.0f} s |
| + query likelihood | **{rerank_result['query likelihood']:.3f}** | {_moves['query likelihood'][0]} / {_moves['query likelihood'][1]} | {_secs['query likelihood']:.0f} s |
| best possible (right abstract is in the top {_depth}) | {rerank_result['ceiling']:.3f} | | |

The yes/no judge asks a 360M chat model for a judgment it was never trained to make, and its answers barely
depend on which abstract it read. Query likelihood asks it to do what it *was* trained for, predicting text,
which is why it works even with a small model. Real rerankers (BGE-reranker, Qwen3-Reranker, Cohere Rerank)
are cross-encoders *fine-tuned* on millions of relevance labels. `local-kb` has no reranker yet; this is
the stage to add next.
""")
    _out
    return (rerank_result,)


@app.cell
def _(lm_cfg, lm_params, lm_tok, mo, papers, rag, retrieve):
    QUESTIONS = ["How do radio telescopes remove radio frequency interference from their data?",
                 "What can pulsar timing arrays detect?",
                 "Why is the 21 cm line of hydrogen useful for cosmology?"]
    if retrieve is None:
        rag_answers = None
        _out = mo.md("")
    else:
        rag_answers = []
        _blocks = []
        _top = retrieve(QUESTIONS, 3)
        for _q, _ids in zip(QUESTIONS, _top, strict=True):
            _srcs = [papers[int(d)]["abstract"] for d in _ids]
            _with = rag.answer(lm_params, lm_cfg, lm_tok, _q, _srcs)
            _without = rag.answer(lm_params, lm_cfg, lm_tok, _q, [])
            _g = rag.groundedness(_with, _srcs)
            _g0 = rag.groundedness(_without, _srcs)
            rag_answers.append({"q": _q, "with": _with, "without": _without, "ground": _g, "ground_closed": _g0,
                                "titles": [papers[int(d)]["title"] for d in _ids]})
            _src_list = "\n".join(f"  [{i + 1}] *{papers[int(d)]['title']}*" for i, d in enumerate(_ids))
            _blocks.append(f"""**Q: {_q}**

Retrieved:
{_src_list}

* **With retrieval:** {_with}
* **Closed book (no sources):** {_without}
* *Check:* the RAG answer cites {_g['cited'] or 'nothing'}{f" (valid: {_g['valid']})" if _g['cited'] else ''};
  **{_g['support']:.0%}** of its content words appear in the retrieved sources, vs {_g0['support']:.0%} for the
  closed-book answer.""")
        _out = mo.md("## 9 · Generation: answer from the sources, and cite them\n\n"
                     "SmolLM2-360M-Instruct gets the top 3 abstracts (clipped to ~110 tokens each), numbered, and is "
                     "told to answer only from them and cite like [1]. For contrast, the same model answers *closed "
                     "book*, from its weights alone.\n\n" + "\n\n---\n\n".join(_blocks) + """

---

Two cheap checks, not proof: are the citations real source numbers, and do the answer's words come from
the sources? (The closed-book answer is scored against the same sources, as a baseline.) Retrieval raised
the overlap for every question, which is the point of RAG. But read the answers yourself: overlap isn't
truth, and a 360M model still mixes in claims its sources don't make, or copies a source verbatim. Neither
check catches that. Production
systems add an LLM judge for *faithfulness* (does every claim follow from the sources?), with the caveats
from Module 21 (the judge has biases). And be honest about scale: a 360M model follows the "cite your
sources" instruction only loosely. The same pipeline with a 7–9B model (`local-kb` uses Qwen3.5-9B in LM
Studio) is where RAG answers become dependable.""")
    _out
    return QUESTIONS, rag_answers


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 · From this notebook to a knowledge base

    Your [aws-ai](https://github.com/joebarbere/aws-ai) [`local-kb`](https://github.com/joebarbere/aws-ai/tree/main/local-kb) is the same pipeline built for production, as a local copy of Amazon Bedrock
    Knowledge Bases (same API shapes, prompts and citations, so stock `boto3` code runs against it):

    | stage | this module | `local-kb` |
    |---|---|---|
    | chunking | `chunk_sentences`, words | whole sentences, 300 tokens, 20% overlap; also hierarchical and semantic |
    | sparse | BM25 as a BCOO matmul | OpenSearch BM25 / Postgres `tsquery` |
    | dense | your bi-encoder (256-d) | Qwen3-Embedding-0.6B in LM Studio (1024-d), via `/v1/embeddings` |
    | index | exact matmul; IVF | HNSW (OpenSearch faiss, pgvector) |
    | fusion | RRF, min-max with a weight | min-max 0.5/0.5 (OpenSearch), RRF k=60 (pgvector) |
    | rerank | SmolLM2 yes/no | none yet |
    | generate | SmolLM2-360M | Qwen3.5-9B, Bedrock's prompt templates |
    | evaluate | MRR, recall, nDCG + bootstrap CIs | hit@5, MRR@10, keyword match, LLM judge |

    Its measured verdict on 50k abstracts: pgvector to run next to a local LLM (300 MB loaded vs 3.4 GB,
    0.4 s cold start vs 13.6 s); OpenSearch for hybrid search on natural-language questions.

    Module 27's LM Studio server exposes `/v1/embeddings`, so the dense stage can swap in a real embedding
    model without touching the rest. **Module 39** reuses this module's `rag.build_index(...)` /
    `index.search(query, k)` as an agent's long-term memory.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"BM25 already matches the title's words inside its own abstract, so the benchmark can't separate good retrievers": "a",
                 "BM25 is a better algorithm than dense retrieval": "b",
                 "OpenSearch's BM25 implementation is more accurate than ours": "c"},
        label="**Check yourself:** why does BM25 score far higher on `local-kb`'s \"title. abstract\" index than on "
              "this module's abstract-only index?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {"a": mo.callout(mo.md("Right. Each query is copied into the start of its own answer, so any keyword "
                                 "search finds it. The benchmark measures plumbing, not retrieval quality."),
                           kind="success"),
           "b": mo.callout(mo.md("Not in general: which wins depends on the queries and the training data. "
                                 "Here the gap comes from the index text, since the algorithm is the same."), kind="warn"),
           "c": mo.callout(mo.md("Same formula, same defaults ($k_1 = 1.5$-ish, $b = 0.75$). The difference is what "
                                 "was indexed."), kind="warn")}
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(38))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Christopher D. Manning, Prabhakar Raghavan & Hinrich Schütze, *Introduction to Information Retrieval*
      (Cambridge University Press, 2008; the authors host it free on Stanford's NLP site) — inverted indexes,
      tf-idf, evaluation.
    * Pinecone, [Retrieval-Augmented Generation](https://www.pinecone.io/learn/retrieval-augmented-generation/) — a practitioner's overview.
    * Nils Reimers, [Sentence-Transformers documentation](https://www.sbert.net/) — bi-encoders, cross-encoders and how they're trained.
    * Anthropic, [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (2024) —
      prepend chunk-specific context before embedding; BM25 + embeddings + reranking, measured.
    * Your [aws-ai](https://github.com/joebarbere/aws-ai) repo: [`local-kb/README.md`](https://github.com/joebarbere/aws-ai/blob/main/local-kb/README.md) and [`local-kb/bench/`](https://github.com/joebarbere/aws-ai/tree/main/local-kb/bench) — the production-shaped version and its benchmark.

    **Next:** Module 39 — Agents: tools, memory, workflows and multi-agent systems.
    """)
    return


if __name__ == "__main__":
    app.run()
