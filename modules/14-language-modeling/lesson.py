import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="14 · Language modeling")


@app.cell
def _():
    import marimo as mo
    import math
    import time
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax

    import learnjax as lj
    from learnjax.data import tiny_shakespeare

    ex = lj.load_exercises(mo.notebook_dir())
    return ex, jax, jnp, lj, math, mo, np, optax, plt, time, tiny_shakespeare


@app.cell
def _(mo):
    mo.md(r"""
    # 14 · Language modeling basics: bigrams to MLP LMs

    **Goal:** understand what a language model *is* — a machine that assigns a probability to the next
    token — and build two of them: one by counting, one by gradient descent. Everything in Modules 15–17
    (attention, GPT, DeepSeek-style models) is a better answer to the same question.

    **You will:** count which character follows which in Shakespeare and sample text from the counts ·
    measure a language model with the cross-entropy and perplexity from Module 04 · rebuild the bigram
    model as a neural network and watch gradient descent find the same answer as counting · write the
    windowed-batch sampler used by most pure-JAX LLM code ([`vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap) + `dynamic_slice`) · train an MLP that
    reads several previous characters, and see what longer context buys · find the limit that motivates
    attention.

    We work with **characters** this time (vocabulary 65), so every probability table is small enough to
    look at. Module 16 switches to BPE tokens from Module 13.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Shannon's 1948 paper already built text from letter and word statistics, the first n-gram language
    models. Through the 1980s–2000s, n-gram models powered speech recognition and machine translation.
    Bengio et al. (2003) replaced the counts with a neural network over learned word embeddings, the MLP
    language model you build here, and word2vec (Mikolov et al., 2013) made embeddings famous on their own.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(jnp, mo, np, tiny_shakespeare):
    text = tiny_shakespeare()
    chars = sorted(set(text))
    vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = jnp.asarray(np.array([stoi[c] for c in text], dtype=np.int32))
    n_train = int(0.9 * len(data))
    train_ids, val_ids = data[:n_train], data[n_train:]

    def decode(ids):
        return "".join(chars[int(i)] for i in ids)

    def show(c):
        return {"\n": "↵", " ": "␣"}.get(c, c)

    mo.md(f"""
    ## 1 · A language model predicts the next token

    Given the text so far, a language model outputs a probability for **every** possible next token. To
    *generate*, sample one, append it, repeat. To *train*, show it real text and minimize the
    cross-entropy of the token that actually came next (Module 04): the average of
    $-\\log P(\\text{{actual next token}})$.

    Our data: Tiny Shakespeare as **{len(text):,} characters** from a vocabulary of **{vocab}**.
    The first 90% ({n_train:,}) is for training, the last 10% ({len(data) - n_train:,}) for validation —
    split **by position**, not at random, because neighbouring windows overlap and a random split would
    leak training text into validation.

    **Baseline:** a model that knows nothing gives every character probability 1/{vocab}, a loss of
    $\\ln {vocab} = {np.log(vocab):.2f}$ nats (perplexity {vocab}). Every model below has to beat that.
    """)
    return chars, decode, n_train, show, stoi, text, train_ids, val_ids, vocab


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · The simplest model: count pairs

    A **bigram** model only looks at the previous character: $P(\text{next} \mid \text{prev})$. Estimate it by
    counting: how often is `h` followed by `e`? Divide each row of the count table by its total and you have
    a probability table. Implement `bigram_counts` in `exercises.py` (one [`.at[].add`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at) — no loops).
    """)
    return


@app.cell
def _(ex, lj, train_ids, vocab):
    counts_try = lj.attempt(ex.bigram_counts, train_ids, vocab)
    counts = counts_try.value if counts_try.ok else None
    return counts, counts_try


@app.cell
def _(chars, counts, counts_try, mo, np, plt, show):
    if counts is None:
        _out = counts_try.view(mo)
    else:
        _c = np.asarray(counts)
        _fig, _ax = plt.subplots(figsize=(7, 7))
        _ax.imshow(np.log1p(_c), cmap="viridis")
        _labels = [show(c) for c in chars]
        _ax.set_xticks(range(len(chars)), _labels, fontsize=5)
        _ax.set_yticks(range(len(chars)), _labels, fontsize=5)
        _ax.set(xlabel="next character", ylabel="previous character",
                title="log(1 + count): which character follows which")
        _flat = np.argsort(_c.ravel())[::-1][:8]
        _top = ", ".join(f"`{show(chars[i // len(chars)])}{show(chars[i % len(chars)])}` ({_c.ravel()[i]:,})"
                         for i in _flat)
        _out = mo.vstack([_fig, mo.md(f"""
Most common pairs: {_top}. Bright columns are common characters (space, `e`); rows show what
can follow each character — after `q`, almost only `u`; after a newline, mostly capitals (speaker names).
""")])
    _out
    return


@app.cell
def _(counts, ex, lj, math, mo, np, train_ids, val_ids, vocab):
    if counts is None:
        _out = mo.md("*Implement `bigram_counts` first.*")
    else:
        _rows = []
        for _k in [1e-3, 0.1, 1.0, 10.0, 100.0]:
            _tr = lj.attempt(ex.bigram_nll, counts, train_ids, _k)
            _va = lj.attempt(ex.bigram_nll, counts, val_ids, _k)
            if not (_tr.ok and _va.ok):
                _rows = None
                _bad = _tr if not _tr.ok else _va
                break
            _rows.append((_k, float(_tr.value), float(_va.value)))
        if _rows is None:
            _out = _bad.view(mo)
        else:
            _table = "\n    ".join(f"| {k:g} | {tr:.3f} | {va:.3f} | {math.exp(va):.1f} |" for k, tr, va in _rows)
            _out = mo.md(f"""
    ### How good is it? Your `bigram_nll`

    | smoothing k | train loss | **val loss** | val perplexity |
    |---|---|---|---|
    {_table}

    From {np.log(vocab):.2f} nats (uniform) to about **{_rows[2][2]:.2f}**: knowing just the previous character
    cuts the "effective number of choices" from {vocab} to about {math.exp(_rows[2][2]):.0f}.
    Smoothing barely matters here because a million characters see almost every plausible pair; with
    heavy smoothing (k = 100) the table drifts back toward uniform and the loss rises. The train and
    val losses are nearly equal: a bigram model has too few parameters to memorize anything.
    """)
    _out
    return


@app.cell
def _(counts, decode, ex, jax, jnp, lj, mo, stoi):
    if counts is None:
        _out = mo.md("")
    else:
        _logp = jnp.log((counts + 1.0) / (counts.sum(axis=1, keepdims=True) + counts.shape[0]))
        _s = lj.attempt(ex.sample, jax.random.key(0), lambda w: _logp[w[-1]], jnp.array([stoi["\n"]]), 300, 1.0)
        _out = (mo.vstack([mo.md("### Text sampled from the bigram counts (your `sample`, context = 1 character)"),
                           mo.md("```\n" + decode(_s.value) + "\n```"),
                           mo.md("Letters, spaces and capitalised names in the right places — and no words. "
                                 "One character of memory is not enough to spell.")])
                if _s.ok else _s.view(mo))
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · The same model, learned by gradient descent

    Put the bigram table in a matrix $W$ of shape (65, 65) and call row $i$ the **logits** for "what follows
    character $i$". Predicting = one-hot($i$) $\times W$ — which just *selects row $i$*. That row lookup is
    exactly an **embedding layer**, the first layer of every language model. Train $W$ with Adam on
    cross-entropy and it should rediscover the counts: the best possible logits are $\log P$ from counting.
    """)
    return


@app.cell
def _(jax, jnp, optax, plt, train_ids, vocab):
    _prev, _next = train_ids[:-1], train_ids[1:]
    _opt = optax.adam(0.1)
    _W = jnp.zeros((vocab, vocab))

    def _loss(W, prev, nxt):
        logits = W[prev]                             # embedding lookup = one-hot @ W
        return optax.softmax_cross_entropy_with_integer_labels(logits, nxt).mean()

    @jax.jit
    def _train(W, key, prev, nxt):
        def step(carry, k):
            W, s = carry
            idx = jax.random.randint(k, (65_536,), 0, prev.shape[0])
            loss, g = jax.value_and_grad(_loss)(W, prev[idx], nxt[idx])
            upd, s = _opt.update(g, s)
            return (optax.apply_updates(W, upd), s), loss
        (W, _), losses = jax.lax.scan(step, (W, _opt.init(W)), jax.random.split(key, 300))
        return W, losses

    bigram_W, _losses = _train(_W, jax.random.key(0), _prev, _next)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_losses)
    _ax.axhline(float(jnp.log(vocab)), color="gray", ls=":", label="uniform")
    _ax.set(xlabel="step", ylabel="train loss (nats)", title="bigram as a neural net (Adam, batches of 65,536 pairs)")
    _ax.legend()
    bigram_nn_fig = _fig
    return bigram_W, bigram_nn_fig


@app.cell
def _(bigram_W, bigram_nn_fig, chars, counts, ex, jax, jnp, lj, mo, val_ids):
    _items = [bigram_nn_fig]
    _nn_val = float(jnp.mean(-jax.nn.log_softmax(bigram_W[val_ids[:-1]])[jnp.arange(val_ids.shape[0] - 1), val_ids[1:]]))
    if counts is not None:
        _cnt = lj.attempt(ex.bigram_nll, counts, val_ids, 1.0)
        if _cnt.ok:
            _p_nn = jax.nn.softmax(bigram_W, axis=1)
            _p_cnt = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + counts.shape[0])
            _tv = 0.5 * jnp.abs(_p_nn - _p_cnt).sum(axis=1)          # per-row total variation, 0..1
            _w = counts.sum(axis=1) / counts.sum()                      # how often each row is used
            _typical = float((_w * _tv).sum())
            _worst = int(jnp.argmax(_tv))
            _items.append(mo.md(
                f"Validation loss: neural net **{_nn_val:.3f}**, counting **{float(_cnt.value):.3f}** — the same "
                f"model reached by two routes. Weighted by how often each character occurs, the two probability "
                f"tables differ by only {_typical:.3f} (total variation, 0 = identical, 1 = disjoint). The "
                f"differences live in rare rows such as the one after `{chars[_worst]!r}` "
                f"(seen {int(counts.sum(axis=1)[_worst])}× in training): few examples mean few gradient updates, and the "
                f"count table's +1 smoothing dominates. Counting only works for tiny models like this; gradient "
                f"descent is what scales to everything that follows."))
    else:
        _items.append(mo.md(f"Validation loss of the neural bigram: **{_nn_val:.3f}** nats."))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Batches of windows: `vmap` + `dynamic_slice`

    Longer context needs training examples that are *windows* of text. The standard pure-JAX recipe:

    1. draw `batch_size` random start positions with [`jax.random.randint`](https://docs.jax.dev/en/latest/_autosummary/jax.random.randint.html#jax.random.randint),
    2. cut `block_size + 1` tokens at each with [`jax.lax.dynamic_slice`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html#jax.lax.dynamic_slice) — the start can be a traced value
       but the **size must be a static Python int**, so the whole thing compiles under [`jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit),
    3. `jax.vmap` the slice over the starts (Module 07) to get a `(batch, block + 1)` array in one op,
    4. inputs are the first `block` tokens, targets the last `block` — **shifted by one**, so every
       position has a "next token" to predict.

    Implement `get_batch`. The data never leaves the GPU; there is no Python loop over examples.
    """)
    return


@app.cell
def _(decode, ex, jax, lj, mo, show, train_ids):
    batch_try = lj.attempt(ex.get_batch, jax.random.key(42), train_ids, 4, 24)
    if batch_try.ok:
        _x, _y = batch_try.value
        _lines = "\n".join(f"x: {''.join(show(c) for c in decode(_x[i]))}\ny: {''.join(show(c) for c in decode(_y[i]))}\n"
                           for i in range(2))
        _out = mo.vstack([mo.md(f"`x` and `y` have shape {tuple(_x.shape)}. Two of the windows (␣ = space, ↵ = newline):"),
                          mo.md("```\n" + _lines + "```"),
                          mo.md("Read down a column: under each input character is the character that follows it.")])
    else:
        _out = batch_try.view(mo)
    _out
    return (batch_try,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · An MLP that reads several characters (Bengio et al., 2003)

    To predict from the last $k$ characters: look up each one's **embedding** (a learned 16-number vector),
    concatenate the $k$ vectors, feed them through a hidden layer (tanh), and output 65 logits. That is the
    neural language model of Bengio et al. — the first to beat counting methods. Implement `mlp_lm_init` and
    `mlp_lm_apply`; the cell below trains one per context length on the GPU (8,000 steps of 512 windows
    each, with the whole training loop compiled as one [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan), Module 10's trick).
    """)
    return


@app.cell
def _(batch_try, ex, jax, jnp, lj, mo, optax, time, train_ids, val_ids, vocab):
    _probe_p = lj.attempt(ex.mlp_lm_init, jax.random.key(0), vocab, 3, 16, 32)
    _probe_a = lj.attempt(ex.mlp_lm_apply, _probe_p.value, jnp.zeros((2, 3), jnp.int32)) if _probe_p.ok else _probe_p
    mlp_ready = batch_try.ok and _probe_p.ok and _probe_a.ok
    mlp_results = {}
    if mlp_ready:
        _opt = optax.adamw(3e-3, weight_decay=1e-4)

        def _make_trainer(k, steps=8000, batch=512):
            def loss_fn(p, x, y):
                logits = ex.mlp_lm_apply(p, x)
                return optax.softmax_cross_entropy_with_integer_labels(logits, y[:, -1]).mean()

            @jax.jit
            def train(p, key, data):
                def step(carry, kk):
                    p, s = carry
                    x, y = ex.get_batch(kk, data, batch, k)
                    loss, g = jax.value_and_grad(loss_fn)(p, x, y)
                    upd, s = _opt.update(g, s, p)
                    return (optax.apply_updates(p, upd), s), loss
                (p, _), losses = jax.lax.scan(step, (p, _opt.init(p)), jax.random.split(key, steps))
                return p, losses

            @jax.jit
            def evaluate(p, ids, key):
                x, y = ex.get_batch(key, ids, 8192, k)
                return loss_fn(p, x, y)

            return train, evaluate

        for _k in [1, 3, 8, 16]:
            _p = ex.mlp_lm_init(jax.random.key(_k), vocab, _k, 16, 256)
            _train, _eval = _make_trainer(_k)
            _t0 = time.perf_counter()
            _p, _losses = _train(_p, jax.random.key(100 + _k), train_ids)
            _losses.block_until_ready()
            _secs = time.perf_counter() - _t0
            mlp_results[_k] = {
                "params": _p,
                "curve": _losses,
                "train": float(_eval(_p, train_ids, jax.random.key(1))),
                "val": float(_eval(_p, val_ids, jax.random.key(2))),
                "secs": _secs,
                "n_params": sum(int(a.size) for a in jax.tree.leaves(_p)),
            }
        _out = mo.md(f"Trained {len(mlp_results)} models.")
    else:
        _bad = _probe_a if _probe_p.ok else _probe_p
        _out = _bad.view(mo) if not _bad.ok else mo.md("*Implement `get_batch` first (Section 4).*")
    _out
    return mlp_ready, mlp_results


@app.cell
def _(math, mlp_ready, mlp_results, mo, np, plt):
    if mlp_ready:
        _fig, _ax = plt.subplots(figsize=(7, 3.2))
        for _k, _r in mlp_results.items():
            _c = np.asarray(_r["curve"])
            _sm = np.convolve(_c, np.ones(50) / 50, mode="valid")
            _ax.plot(_sm, label=f"context {_k}")
        _ax.set(xlabel="step", ylabel="train loss (50-step avg)", title="MLP language models on Tiny Shakespeare",
                ylim=(1.4, 3.2))
        _ax.legend()
        _rows = "\n".join(
            f"| {k} | {r['n_params']:,} | {r['train']:.3f} | **{r['val']:.3f}** | {math.exp(r['val']):.1f} | {r['secs']:.1f} s |"
            for k, r in mlp_results.items())
        _out = mo.vstack([_fig, mo.md(f"""
| context (chars) | parameters | train loss | val loss | val perplexity | train time |
|---|---|---|---|---|---|
{_rows}

* **Context 1** is a bigram model again, and lands where counting did (~2.48).
* **Three characters is a big jump** — enough to finish common syllables and short words.
* **Then the returns vanish.** Going from 8 to 16 characters adds parameters but doesn't lower the
  validation loss — this architecture can't make use of the extra context (Section 6 explains why).
* **Train < val** once there is real context, and the gap grows with the parameter count: the first
  sign of memorizing. More data, dropout or weight decay (Module 11) are the remedies.
* Each model trains in seconds because the whole loop is one compiled `lax.scan`.
""")])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mlp_ready, mo):
    temperature = mo.ui.slider(steps=[0.0, 0.3, 0.6, 0.8, 1.0, 1.3, 2.0], value=0.8, label="temperature",
                               show_value=True)
    temperature if mlp_ready else mo.md("")
    return (temperature,)


@app.cell
def _(decode, ex, jax, jnp, lj, mlp_ready, mlp_results, mo, stoi, temperature):
    if mlp_ready:
        _p = mlp_results[8]["params"]
        _logits = jax.jit(lambda w: ex.mlp_lm_apply(_p, w[None])[0])
        _start = jnp.array([stoi[c] for c in "ROMEO:\nI"], jnp.int32)
        _s = lj.attempt(ex.sample, jax.random.key(3), _logits, _start, 400, temperature.value)
        _out = (mo.vstack([
            mo.md(f"### Samples from the context-8 model, temperature {temperature.value}"),
            mo.md("```\nROMEO:\nI" + decode(_s.value) + "\n```"),
            mo.md("**Temperature** divides the logits before the softmax (Module 04). Near 0 the model always "
                  "picks its favourite and loops; around 0.8 the text looks most like Shakespeare; above 1 it "
                  "takes risks and the spelling falls apart. Same model, same probabilities — only how "
                  "boldly we sample changes."),
        ]) if _s.ok else _s.view(mo))
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · The wall: why fixed windows don't scale

    The MLP gives every one of its $k$ input slots its own block of weights in `w1`
    (shape $k \cdot d_{emb} \times d_{hidden}$). That has three consequences:

    * **Parameters grow with context.** Doubling $k$ doubles `w1` — and, as the table showed, the extra
      weights did not buy a better model. A 2,000-token context this way would need a weight block for
      every one of 2,000 positions.
    * **Nothing is shared across positions.** The pattern "`the` → space" must be learned separately for
      each slot it can appear in.
    * **Every position gets the same treatment.** The model cannot decide that, for *this* prediction,
      the speaker's name 200 characters back matters more than the last three letters.

    **Attention** (Module 15) fixes all three: the same weights are applied at every position, and each
    token computes, from the data, *which* earlier tokens to look at and how much. That's the idea behind
    every transformer — GPT, Llama, and the DeepSeek-style model you trained.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Neighbouring windows overlap, so a random split leaks training text into validation": "a",
            "Random splits are slower to compute": "b",
            "The model can only learn from the start of a text": "c",
        },
        label="**Check yourself:** why split train/validation by *position* instead of shuffling windows?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _feedback = {
        "a": mo.callout(mo.md("Right. Two windows starting one character apart share almost all their text; "
                              "if one is in training and the other in validation, the validation score "
                              "measures memory, not generalisation."), kind="success"),
        "b": mo.callout(mo.md("Speed isn't the issue — it's *leakage*: overlapping windows would put the same "
                              "text on both sides of the split."), kind="warn"),
        "c": mo.callout(mo.md("No — `get_batch` samples windows from anywhere in the training part. The issue "
                              "is that overlapping windows leak text across a random split."), kind="warn"),
    }
    _feedback.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(14))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Bengio, Ducharme, Vincent & Jauvin, [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
      (2003) — the MLP language model of Section 5.
    * Andrej Karpathy, [makemore](https://github.com/karpathy/makemore) and the
      [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) videos 2–5 — bigrams to MLPs in
      PyTorch, a good second angle on this module.
    * Jurafsky & Martin, [Speech and Language Processing, ch. 3: N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
      — smoothing, perplexity and evaluation done carefully.
    * Claude Shannon, [Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf)
      (1951) — the original "guess the next letter" experiment.

    **Next:** Module 15 — attention from scratch: letting every token decide which earlier tokens matter.
    """)
    return


if __name__ == "__main__":
    app.run()
