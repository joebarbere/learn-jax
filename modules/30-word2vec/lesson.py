import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="30 · word2vec")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import time
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    _spec = importlib.util.spec_from_file_location("w2v30", mo.notebook_dir() / "w2v.py")
    w2v = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(w2v)
    return ex, jax, jnp, lj, mo, np, plt, time, w2v


@app.cell
def _(mo):
    mo.md(r"""
    # 30 · Word embeddings: word2vec (2013)

    *Part 10 — Milestones, rebuilt in JAX. Take Modules 13 and 14 first.*

    **Goal:** rebuild the idea that made "a word is a vector" mainstream, the way it was done in 2013, and
    see what it can and can't do on your own detective-fiction corpus.

    **You will:** see why one-hot vectors know nothing about meaning · build word2vec's vocabulary,
    frequent-word subsampling and noise distribution · write the skip-gram negative-sampling loss and train
    it on the GPU · look up nearest neighbours and try the famous analogies, honestly measured · factorize
    a co-occurrence matrix (the GloVe/LSA family) and get similar vectors without any neural network ·
    compare with the token embeddings inside SmolLM2, a 2024 language model.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from.** Linguists Zellig Harris (1954) and J. R. Firth (1957) argued that a word's
    meaning shows in the company it keeps — the *distributional hypothesis*. In 1990, Latent Semantic
    Analysis turned that into linear algebra by factorizing word–document counts. Bengio and colleagues'
    2003 neural language model (Module 14) learned word vectors as a side effect, but training it was slow.
    In 2013 Tomas Mikolov and colleagues at Google stripped the model down to almost nothing — word2vec —
    so it could read billions of words on ordinary CPUs, and its vectors showed surprising arithmetic
    structure. GloVe (Pennington, Socher and Manning, Stanford, 2014) reached similar vectors from global
    co-occurrence counts. Contextual embeddings (ELMo and BERT, 2018 — Module 32) and LLMs later replaced
    them. Timeline: Module 00b.
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Two ways to write a word down

    **One-hot:** give each word its own dimension. "revolver" is $[0,\dots,0,1,0,\dots]$, "pistol" another
    axis. Every pair of words is orthogonal — the representation knows *nothing* about meaning, and it has
    as many dimensions as the vocabulary.

    **Distributed:** give each word a short dense vector (here 128 numbers) and let training place words
    used in similar contexts near each other. Similarity becomes a cosine.

    word2vec's trick is the training task. **Skip-gram** asks: given a center word, predict the words
    around it. A full softmax over the whole vocabulary would be expensive, so **negative sampling** turns it
    into many cheap yes/no questions: *is this word a real neighbour, or a random noise word?*

    For a center word with input vector $v_c$, a real context word with output vector $u_o$, and $K$ noise
    words $u_1,\dots,u_K$:

    $$\mathcal{L} = -\log\sigma(v_c\cdot u_o) - \sum_{k=1}^{K}\log\sigma(-v_c\cdot u_k)$$

    Push real pairs together, random pairs apart. That's the whole model: two embedding tables and a dot
    product. No hidden layer — which is why it was so fast.
    """)
    return


@app.cell
def _(mo, time, w2v):
    _t = time.perf_counter()
    words = w2v.corpus_words(12_000_000)
    _secs = time.perf_counter() - _t
    mo.md(f"""
    ## 2 · The corpus, as words

    We decode the first 12M BPE tokens of the detective corpus you trained on in Module 16 back to text and
    split it into lowercase words: **{len(words):,} words** in {_secs:.1f} s. For scale, word2vec's
    published vectors were trained on billions of words — ours is a few hundred times smaller.

    First words: *{" ".join(words[:25])}…*
    """)
    return (words,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Your turn: vocabulary, subsampling and the noise distribution

    Three counting helpers, straight from the 2013 C code:

    * `build_vocab` — keep words seen at least `min_count` times, most frequent first.
    * `subsample_probs` — "the" appears thousands of times more often than "revolver" but tells us little;
      keep each occurrence with probability $\min\left(1, \sqrt{t/f} + t/f\right)$ where $f$ is the word's
      frequency.
    * `negative_sampling_table` — draw noise words from the unigram distribution raised to the $3/4$ power,
      which boosts rare words a little.
    """)
    return


@app.cell
def _(ex, lj, mo, words):
    vocab_a = lj.attempt(ex.build_vocab, words, 10)
    _items = [mo.md("**`build_vocab(words, min_count=10)`**")]
    if vocab_a.ok:
        _w2i, _i2w, _counts = vocab_a.value
        _top = ", ".join(f"{w} ({c:,})" for w, c in zip(_i2w[:12], _counts[:12]))
        _items.append(mo.md(f"{len(_i2w):,} words kept. Most frequent: {_top}"))
    else:
        _items.append(vocab_a.view(mo))
    mo.vstack(_items)
    return (vocab_a,)


@app.cell
def _(ex, lj, mo, np, plt, vocab_a):
    keep_a = lj.attempt(ex.subsample_probs, vocab_a.value[2], 1e-4) if vocab_a.ok else None
    noise_a = lj.attempt(ex.negative_sampling_table, vocab_a.value[2], 0.75) if vocab_a.ok else None
    _out = mo.md("Finish `build_vocab` first.")
    if keep_a is not None and keep_a.ok and noise_a.ok:
        _counts = vocab_a.value[2]
        _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.4))
        _a1.loglog(np.arange(1, len(_counts) + 1), keep_a.value)
        _a1.set(xlabel="word rank", ylabel="P(keep)", title="frequent words are mostly dropped")
        _r = np.arange(1, 41)
        _uni = _counts / _counts.sum()
        _a2.semilogy(_r, _uni[:40], label="unigram")
        _a2.semilogy(_r, noise_a.value[:40], label="unigram$^{3/4}$")
        _a2.semilogy(_r, np.full(40, 1 / len(_counts)), label="uniform")
        _a2.set(xlabel="word rank", ylabel="noise probability", title="the 3/4 power flattens the noise")
        _a2.legend()
        _fig.tight_layout()
        _kept = float((keep_a.value * _counts).sum() / _counts.sum())
        _out = mo.vstack([_fig, mo.md(f"Subsampling keeps about **{_kept:.0%}** of word occurrences; "
                                      f"P(keep \"the\") = {keep_a.value[0]:.3f}.")])
    elif keep_a is not None:
        _out = mo.vstack([keep_a.view(mo), noise_a.view(mo)])
    _out
    return keep_a, noise_a


@app.cell
def _(ex, jax, lj, mo):
    _sent = "the inspector examined the revolver".split()
    _ids = list(range(len(_sent)))
    _a = lj.attempt(ex.skipgram_pairs, _ids, 2, jax.random.key(0))
    _items = [mo.md("## 4 · Skip-gram pairs\n\n**`skipgram_pairs`** on *the inspector examined the revolver*, "
                    "window 2:")]
    if _a.ok:
        _pairs = sorted(zip(_a.value[0].tolist(), _a.value[1].tolist()))
        _items.append(mo.md(", ".join(f"({_sent[c]} → {_sent[x]})" for c, x in _pairs)))
        _items.append(mo.md("Each pair is one training example. The trainer below samples pairs on the GPU "
                            "instead of materializing them all, with word2vec's *dynamic window*: nearer "
                            "words are picked more often."))
    else:
        _items.append(_a.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _k1, _k2 = jax.random.split(jax.random.key(1))
    _Ei = jax.random.normal(_k1, (50, 16)) * 0.3
    _Eo = jax.random.normal(_k2, (50, 16)) * 0.3
    loss_a = lj.attempt(ex.sgns_loss, jnp.array([1, 2]), jnp.array([3, 4]), jnp.array([[5, 6], [7, 8]]),
                        _Ei, _Eo)
    _z = jnp.zeros((50, 16))
    _items = [mo.md("## 5 · Your turn: the loss\n\n**`sgns_loss`** on random vectors:"), loss_a.view(mo)]
    if loss_a.ok:
        _zl = float(ex.sgns_loss(jnp.array([1]), jnp.array([3]), jnp.array([[5, 6]]), _z, _z))
        _items.append(mo.md(f"With all-zero vectors every logit is 0 and the loss is (1 + K)·ln 2 = "
                            f"**{_zl:.4f}** for K = 2 — a good sanity check."))
    mo.vstack(_items)
    return (loss_a,)


@app.cell
def _(ex, keep_a, loss_a, mo, noise_a, np, plt, vocab_a, w2v, words):
    trained = None
    _out = mo.callout(mo.md("Finish `build_vocab`, `subsample_probs`, `negative_sampling_table` and "
                            "`sgns_loss`, then re-run this cell to train."), kind="info")
    if vocab_a.ok and keep_a is not None and keep_a.ok and noise_a.ok and loss_a.ok:
        _w2i, _i2w, _counts = vocab_a.value
        _ids = w2v.encode(words, _w2i)
        _stream = w2v.subsample(_ids, keep_a.value)
        _E_in, _E_out, _losses, _secs, _pps = w2v.train(ex.sgns_loss, _stream, noise_a.value, len(_i2w),
                                                        steps=5000)
        trained = {"E": _E_in, "w2i": _w2i, "i2w": _i2w, "ids": _ids}
        _fig, _ax = plt.subplots(figsize=(7, 3))
        _sm = np.convolve(_losses, np.ones(50) / 50, mode="valid")
        _ax.plot(_sm)
        _ax.set(xlabel="step (4,096 pairs each)", ylabel="SGNS loss", title="training skip-gram on your GPU")
        _fig.tight_layout()
        _out = mo.vstack([
            mo.md(f"""## 6 · Train it

    {len(_ids):,} words → {len(_stream):,} after subsampling. 5,000 steps × 4,096 pairs × 5 negatives,
    128-dimensional vectors, Adam: **{_secs:.0f} s** ({_pps / 1e6:.2f}M pairs/s, measured live — slower
    if other jobs share the GPU). Loss {_losses[:50].mean():.2f} → {_losses[-200:].mean():.2f}.

    The 2013 C code used plain SGD with a decaying learning rate on many CPU threads, updating only the
    rows a pair touches. Here every step updates two dense 21k × 128 tables with Adam — wasteful in
    arithmetic, but the GPU doesn't mind."""),
            _fig])
    _out
    return (trained,)


@app.cell
def _(mo, trained):
    _choices = ["holmes", "revolver", "inspector", "murder", "window", "poison", "london", "letter", "night",
                "detective", "doctor", "train"]
    pick = mo.ui.dropdown([w for w in _choices if trained is None or w in trained["w2i"]], value="revolver",
                          label="word")
    pick
    return (pick,)


@app.cell
def _(ex, lj, mo, pick, trained):
    _out = mo.md("Train the model above first.")
    if trained is not None:
        _i2w, _w2i = trained["i2w"], trained["w2i"]
        _a = lj.attempt(ex.cosine_neighbors, trained["E"], _w2i[pick.value], 10)
        _out = mo.vstack([
            mo.md(f"## 7 · Nearest neighbours\n\n**`cosine_neighbors`** of *{pick.value}*:"),
            mo.md(", ".join(_i2w[int(i)] for i in _a.value)) if _a.ok else _a.view(mo),
            mo.md("Nobody told the model that a revolver is a weapon or that Lestrade works with Holmes. "
                  "It only saw which words appear near which."),
        ])
    _out
    return


@app.cell
def _(ex, lj, mo, trained):
    _ANALOGIES = [
        ("man", "woman", "he", "she"), ("man", "woman", "his", "her"), ("he", "she", "him", "her"),
        ("mr", "mrs", "sir", "madam"), ("father", "mother", "son", "daughter"),
        ("brother", "sister", "husband", "wife"), ("king", "queen", "man", "woman"),
        ("door", "doors", "window", "windows"), ("room", "rooms", "house", "houses"),
        ("man", "men", "woman", "women"), ("walk", "walked", "look", "looked"),
        ("say", "said", "tell", "told"), ("go", "went", "come", "came"),
        ("good", "better", "bad", "worse"), ("big", "bigger", "small", "smaller"),
        ("france", "paris", "england", "london"), ("france", "french", "england", "english"),
        ("one", "two", "three", "four"), ("monday", "tuesday", "wednesday", "thursday"),
        ("revolver", "revolvers", "pistol", "pistols"),
    ]
    _out = mo.md("Train the model above first.")
    if trained is not None:
        _w2i, _i2w = trained["w2i"], trained["i2w"]
        _rows, _top1, _top5, _n = [], 0, 0, 0
        for _a, _b, _c, _d in _ANALOGIES:
            if not all(w in _w2i for w in (_a, _b, _c, _d)):
                continue
            _r = lj.attempt(ex.analogy, trained["E"], _w2i[_a], _w2i[_b], _w2i[_c], 5)
            if not _r.ok:
                _out = _r.view(mo)
                break
            _ans = [_i2w[int(i)] for i in _r.value]
            _n += 1
            _top1 += _ans[0] == _d
            _top5 += _d in _ans
            _rows.append(f"| {_a} : {_b} :: {_c} : **{_d}** | {', '.join(_ans)} | "
                         f"{'✅' if _ans[0] == _d else ('top 5' if _d in _ans else '✗')} |")
        else:
            _out = mo.md("## 8 · The famous analogies, honestly measured\n\n"
                         "word2vec made headlines because vector arithmetic seemed to capture meaning: "
                         "*king − man + woman ≈ queen*. Your **`analogy`** on 20 hand-written analogies "
                         "suited to this corpus:\n\n"
                         "| analogy | model's top 5 | |\n|---|---|---|\n" + "\n".join(_rows) +
                         f"\n\n**Top-1 {_top1}/{_n}, top-5 {_top5}/{_n}.** Pronouns and the commonest grammatical "
                         "patterns (tenses, plurals, comparatives) come through some of the time; facts like "
                         "capitals need far more text than a shelf of detective novels. The original vectors were trained "
                         "on billions of words, and analogy accuracy rose steadily with corpus size.")
    _out
    return


@app.cell
def _(jnp, mo, plt, trained):
    _out = mo.md("Train the model above first.")
    if trained is not None:
        _groups = {"weapons": ["revolver", "pistol", "gun", "knife", "dagger", "rifle"],
                   "police": ["inspector", "sergeant", "constable", "detective", "superintendent"],
                   "rooms": ["kitchen", "bedroom", "library", "study", "hall", "parlour"],
                   "family": ["father", "mother", "brother", "sister", "wife", "husband"],
                   "time": ["morning", "evening", "night", "afternoon", "midnight"]}
        _w2i = trained["w2i"]
        _sel = [(g, w) for g, ws in _groups.items() for w in ws if w in _w2i]
        _X = trained["E"][jnp.array([_w2i[w] for _, w in _sel])]
        _X = _X - _X.mean(0)
        _U, _S, _Vt = jnp.linalg.svd(_X, full_matrices=False)
        _P = _X @ _Vt[:2].T
        _fig, _ax = plt.subplots(figsize=(7, 5))
        for _g in _groups:
            _pts = [(float(_P[i, 0]), float(_P[i, 1]), w) for i, (g, w) in enumerate(_sel) if g == _g]
            _ax.scatter([p[0] for p in _pts], [p[1] for p in _pts], label=_g)
            for _x, _y, _w in _pts:
                _ax.annotate(_w, (_x, _y), fontsize=8, xytext=(3, 3), textcoords="offset points")
        _ax.legend(fontsize=8)
        _ax.set(title="128-d word vectors, projected to 2-D with PCA (Module 02's SVD)")
        _fig.tight_layout()
        _out = mo.vstack([mo.md("## 9 · A picture of the space"), _fig,
                          mo.md("The groups were never labelled — they come from co-occurrence alone. Two "
                                "dimensions throw away most of the 128, so overlaps here don't mean much.")])
    _out
    return


@app.cell
def _(ex, lj, mo, time, trained, w2v):
    _out = mo.md("Train the model above first.")
    if trained is not None and not lj.attempt(ex.cosine_neighbors, trained["E"], 0, 1).ok:
        _out = mo.md("Finish `cosine_neighbors` to compare neighbourhoods.")
    elif trained is not None:
        _t = time.perf_counter()
        _V = 3000
        _E = w2v.ppmi_svd(trained["ids"], _V, window=5, dim=128)
        _secs = time.perf_counter() - _t
        _i2w, _w2i = trained["i2w"], trained["w2i"]
        _rows = []
        for _w in ["revolver", "inspector", "night", "letter"]:
            if _w2i.get(_w, _V) < _V:
                _a = [_i2w[int(i)] for i in ex.cosine_neighbors(_E, _w2i[_w], 6)]
                _b = [_i2w[int(i)] for i in ex.cosine_neighbors(trained["E"], _w2i[_w], 6)]
                _rows.append(f"| {_w} | {', '.join(_a)} | {', '.join(_b)} |")
        _out = mo.md(
            "## 10 · The counting route: PMI + SVD (the GloVe family)\n\n"
            f"Is the neural network essential? No. Count how often each pair of the {_V:,} most frequent words "
            "appears within 5 words of each other, convert the counts to **positive pointwise mutual "
            "information** (how much more often two words co-occur than chance), and factorize that matrix with "
            "an SVD, the descendant of 1990's Latent Semantic Analysis. GloVe (2014) fits a weighted "
            "least-squares model to the same co-occurrence statistics, and Levy and Goldberg (2014) showed "
            "skip-gram with negative sampling implicitly factorizes a shifted PMI matrix. "
            f"Built here in **{_secs:.1f} s**:\n\n"
            "| word | PPMI + SVD neighbours | word2vec neighbours |\n|---|---|---|\n" + "\n".join(_rows) +
            "\n\nSimilar neighbourhoods from two very different procedures: the information is in the counts.")
    _out
    return


@app.cell
def _(ex, lj, mo):
    from learnjax import llama

    _params, _cfg, _tok = llama.load(llama.SMOLLM_135M)
    _E = _params["embed"]
    _rows = []
    for _w in [" revolver", " inspector", " murder", " London"]:
        _ids = _tok.encode(_w).ids
        if len(_ids) == 1:
            _a = lj.attempt(ex.cosine_neighbors, _E, _ids[0], 8)
            _txt = (", ".join(repr(_tok.decode([int(i)])) for i in _a.value) if _a.ok
                    else "(finish `cosine_neighbors`)")
            _rows.append(f"| {_w.strip()} | {_txt} |")
    _table = "\n".join(_rows)
    mo.md(f"""## 11 · What came next

word2vec gives **one vector per word, whatever the sentence**: "bank" in "river bank" and "bank robbery"
share a vector. ELMo (2018) and BERT (2018, Module 32) made embeddings **contextual** — computed by a deep
network reading the whole sentence. Every LLM still starts with a word2vec-like **input embedding table**,
though: one learned vector per token, refined by the layers above (Modules 16–17).

Here is SmolLM2-135M's input table ({_cfg.vocab:,} BPE tokens × {_cfg.d_model}), searched with your
`cosine_neighbors`:

| token | nearest tokens in SmolLM2's embedding table |
|---|---|
{_table}

Near-duplicates (capitalization, leading spaces, plurals) dominate: in an LLM much of the meaning lives in
the layers, not the table.""")
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"It removed the hidden layer and replaced the full softmax with a few yes/no noise "
                 "comparisons": "a",
                 "It used a much deeper network than Bengio's 2003 model": "b",
                 "It used labelled data for word meanings": "c"},
        label="**Check yourself:** why could word2vec train on billions of words in 2013?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {"a": mo.callout(mo.md("Right — two embedding tables and a dot product, plus negative sampling "
                                 "instead of a softmax over the whole vocabulary."), kind="success"),
           "b": mo.callout(mo.md("The opposite: it was *shallower* — no hidden layer at all."), kind="warn"),
           "c": mo.callout(mo.md("No labels: only raw text and which words appear near which."), kind="warn")}
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(30))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Chris McCormick, [Word2Vec Tutorial — The Skip-Gram Model](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
    * Omer Levy & Yoav Goldberg, [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization) (NeurIPS 2014)
    * Jay Alammar, [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)
    * Dan Jurafsky & James H. Martin, [Speech and Language Processing, ch. 6: Vector Semantics and Embeddings](https://web.stanford.edu/~jurafsky/slp3/6.pdf)

    **Next:** Module 31 — RNNs, LSTMs and seq2seq: reading whole sequences, the road to attention.
    """)
    return


if __name__ == "__main__":
    app.run()
