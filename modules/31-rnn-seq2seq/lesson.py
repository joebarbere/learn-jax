import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="31 · RNNs and seq2seq")


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
    from learnjax.data import tiny_shakespeare

    ex = lj.load_exercises(mo.notebook_dir())
    _spec = importlib.util.spec_from_file_location("m31_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    return M, ex, jax, jnp, lj, mo, np, plt, time, tiny_shakespeare


@app.cell
def _(mo):
    mo.md(r"""
    # 31 · RNNs, LSTMs and seq2seq with attention (1986–2014)

    **Take first:** Module 08 (`lax.scan`) and Module 14 (language modeling). This module sits *between*
    14 and 15: it's the road the field actually took from fixed-window models to attention.

    **Goal:** rebuild the recurrent era the way it was done then — a network that reads one token at a
    time and carries a memory forward — see exactly why it struggled, and watch attention get invented
    as the fix.

    **You will:** write the Elman RNN, LSTM and GRU cells and run them with
    [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan) · train
    three character language models with the same parameter budget and compare them with Module 14's MLP
    and Module 16's GPT · *measure* vanishing gradients · build a 2014-style encoder–decoder that reverses
    digit strings, and watch it fail on long inputs until you add Bahdanau attention · time recurrence
    against attention to see why the Transformer won.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    In the late 1980s Michael Jordan (1986) and Jeffrey Elman (1990) gave networks a loop: the hidden state
    from the last step is fed back in, so a network can read a sequence one token at a time. Training them
    turned out to be hard — Sepp Hochreiter's 1991 thesis and Bengio, Simard & Frasconi (1994) showed that
    gradients shrink (or blow up) exponentially as they travel back through time. Hochreiter & Schmidhuber's
    LSTM (1997) fixed it with a gated memory cell, and from the late 2000s LSTMs ran speech and handwriting
    recognition. In 2014 encoder–decoder RNNs (Sutskever, Vinyals & Le; Cho et al.) learned to translate,
    and Bahdanau, Cho & Bengio added *attention* to get past their fixed-size bottleneck — the idea the
    Transformer (2017) kept while throwing away the recurrence.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The story: memory, and why it was hard

    **1986–1990: a loop in the network.** Module 14's MLP saw a fixed window of characters and nothing
    else. Jordan's and Elman's *recurrent* networks read one token at a time and keep a hidden state
    $h_t = \tanh(x_t W_x + h_{t-1} W_h + b)$ that summarizes everything so far. Training uses ordinary
    backprop on the network *unrolled* through time — **backpropagation through time** (BPTT).

    **1991–1994: the gradients die.** To learn that the character 50 steps ago matters, the gradient has
    to travel back 50 steps, getting multiplied by $W_h$ (and $\tanh'$) at each one. Products of many
    numbers below 1 shrink exponentially; above 1 they explode. Hochreiter's 1991 diploma thesis and
    Bengio, Simard & Frasconi (1994) pinned this down: plain RNNs effectively can't learn long-range
    dependencies. You'll measure it in Section 5.

    **1997–2000: the LSTM.** Hochreiter & Schmidhuber's fix was a separate *cell state* $c_t$ updated by
    **addition**, $c_t = f \odot c_{t-1} + i \odot g$, with learned gates deciding what to write, keep
    and read. Gers, Schmidhuber & Cummins added the **forget gate** $f$ in 2000. When $f \approx 1$,
    gradients flow back through $c$ almost untouched — the same trick as the residual connections of
    Module 11, two decades earlier. The **GRU** (Cho et al., 2014) is a lighter two-gate version.

    **2014: sequence to sequence.** Sutskever, Vinyals & Le trained one LSTM to *read* a sentence into
    a vector and another to *write* the translation from it (they also found that feeding the source in
    reverse helped). It worked — but the whole input has to squeeze through one fixed-size vector.
    Bahdanau, Cho & Bengio's fix, the same year: let the decoder **look back at every encoder state**
    and take a learned, weighted average at each step. That weighted average is **attention**, and
    Module 15 is what happened when Vaswani et al. (2017) kept only that part.
    """)
    return


@app.cell
def _(jnp, mo, np, tiny_shakespeare):
    text = tiny_shakespeare()
    chars = sorted(set(text))
    vocab = len(chars)
    _stoi = {c: i for i, c in enumerate(chars)}
    _data = jnp.asarray(np.array([_stoi[c] for c in text], dtype=np.int32))
    _n = int(0.9 * len(_data))
    train_ids, val_ids = _data[:_n], _data[_n:]
    mo.md(f"""
    Data, as in Module 14: Tiny Shakespeare as **{len(text):,} characters**, vocabulary **{vocab}**,
    90/10 split by position. Module 14's best MLP reached **1.82** nats/char on validation; Module 16's
    small GPT reached about **1.54**.
    """)
    return chars, train_ids, val_ids, vocab


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · The Elman cell, and `scan` as the loop

    One step of the vanilla RNN is one line: $h_{\text{new}} = \tanh(x W_x + h W_h + b)$. Running it over
    a sequence is a loop that carries $h$ forward — exactly what `lax.scan` is for (Module 08). We keep
    sequences **time-major**, shape `(T, batch, features)`, so `scan` walks the first axis.

    **Your turn:** implement `rnn_cell` and `run_rnn` in `exercises.py`.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _p = {"Wx": jax.random.normal(jax.random.key(0), (3, 5)), "Wh": 0.5 * jnp.eye(5), "b": jnp.zeros(5)}
    _xs = jax.random.normal(jax.random.key(1), (10, 2, 3))
    rnn_try = lj.attempt(ex.rnn_cell, _p, jnp.zeros((2, 5)), _xs[0])
    run_try = lj.attempt(ex.run_rnn, ex.rnn_cell, _p, jnp.zeros((2, 5)), _xs)
    _items = [mo.md("**`rnn_cell` on a batch of 2:**"), rnn_try.view(mo, show_value=False)]
    if run_try.ok:
        _items.append(mo.md(f"**`run_rnn` over 10 steps:** final h {tuple(run_try.value[0].shape)}, all "
                            f"hidden states {tuple(run_try.value[1].shape)} — one per step."))
    else:
        _items.append(run_try.view(mo))
    mo.vstack(_items)
    return rnn_try, run_try


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Gates: the LSTM and the GRU

    **LSTM.** From one matrix product, four gates (in this order): input $i$, forget $f$, candidate $g$,
    output $o$:

    $$c_t = \sigma(z_f) \odot c_{t-1} + \sigma(z_i) \odot \tanh(z_g), \qquad h_t = \sigma(z_o) \odot \tanh(c_t)$$

    The carry is now a pair $(h, c)$. The key line is the first: $c$ changes by *addition*, scaled by a
    forget gate that can sit near 1, so information — and gradient — can survive many steps.
    (We initialize the forget-gate bias to 1, as Gers et al. recommended, so cells start out remembering.)

    **GRU.** Two gates — update $z$ and reset $r$ — and no separate cell state:
    $h_t = (1 - z) \odot n + z \odot h_{t-1}$. Fewer parameters for the same width.

    **Your turn:** implement `lstm_cell` and `gru_cell`.
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lj, mo):
    _x = jax.random.normal(jax.random.key(2), (2, 3))
    _h = jnp.zeros((2, 4))
    lstm_try = lj.attempt(ex.lstm_cell, M.init_cell(jax.random.key(3), "lstm", 3, 4), (_h, _h), _x)
    gru_try = lj.attempt(ex.gru_cell, M.init_cell(jax.random.key(4), "gru", 3, 4), _h, _x)
    budget = {"rnn": 280, "gru": 150, "lstm": 128}
    _rows = "\n".join(f"| {k.upper()} | {H} | {M.cell_params(k, 64, H):,} |" for k, H in budget.items())
    mo.vstack([
        mo.md("**`lstm_cell`:**"), lstm_try.view(mo, show_value=False),
        mo.md("**`gru_cell`:**"), gru_try.view(mo, show_value=False),
        mo.md("For a fair race, each cell gets about the same number of recurrent parameters (input width 64):\n\n"
              "| cell | hidden size H | recurrent params |\n|---|---|---|\n" + _rows +
              "\n\nThe LSTM has 4 gate blocks per unit, the GRU 3, the vanilla RNN 1 — so the LSTM gets the "
              "smallest hidden state."),
    ])
    return budget, gru_try, lstm_try


@app.cell
def _(gru_try, lstm_try, mo, rnn_try, run_try):
    lm_ready = rnn_try.ok and run_try.ok and lstm_try.ok and gru_try.ok
    mo.md(r"""
    ## 4 · Three character language models, one budget

    Each model: embed the character (64 numbers) → recurrent cell → read out 65 logits. Trained with
    Adam on random 64-character windows (truncated BPTT), 2,000 steps each, with **your** cells and
    **your** `run_rnn`. Each training cell takes up to a minute.
    """ if lm_ready else r"""
    ## 4 · Three character language models, one budget

    The training cells below use **your** `rnn_cell`, `lstm_cell`, `gru_cell` and `run_rnn`. Finish those
    four exercises and they will run.
    """)
    return (lm_ready,)


@app.cell
def _(M, budget, ex, lm_ready, train_ids, val_ids, vocab):
    lm_rnn = (M.train_lm(ex.rnn_cell, ex.run_rnn, "rnn", vocab, train_ids, val_ids, budget["rnn"], steps=2000)
              if lm_ready else None)
    return (lm_rnn,)


@app.cell
def _(M, budget, ex, lm_ready, train_ids, val_ids, vocab):
    lm_gru = (M.train_lm(ex.gru_cell, ex.run_rnn, "gru", vocab, train_ids, val_ids, budget["gru"], steps=2000)
              if lm_ready else None)
    return (lm_gru,)


@app.cell
def _(M, budget, ex, lm_ready, train_ids, val_ids, vocab):
    lm_lstm = (M.train_lm(ex.lstm_cell, ex.run_rnn, "lstm", vocab, train_ids, val_ids, budget["lstm"],
                          steps=2000) if lm_ready else None)
    return (lm_lstm,)


@app.cell
def _(lm_gru, lm_lstm, lm_ready, lm_rnn, mo, plt):
    if lm_ready:
        _fig, _ax = plt.subplots(figsize=(7, 3.4))
        for _name, _r in [("vanilla RNN", lm_rnn), ("GRU", lm_gru), ("LSTM", lm_lstm)]:
            _steps, _, _val = zip(*_r["curve"])
            _ax.plot(_steps, _val, label=f"{_name} (final {_r['val']:.3f})")
        _ax.axhline(1.82, color="gray", ls="--", lw=1, label="Module 14 MLP (1.82)")
        _ax.axhline(1.54, color="black", ls=":", lw=1, label="Module 16 GPT (≈1.54)")
        _ax.set(xlabel="training step", ylabel="validation loss (nats/char)", ylim=(1.4, 2.6))
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.3)
        _best = min([("vanilla RNN", lm_rnn), ("GRU", lm_gru), ("LSTM", lm_lstm)], key=lambda t: t[1]["val"])
        _rows = "\n".join(f"| {n} | {r['val']:.3f} | {r['seconds']:.0f} s |"
                          for n, r in [("vanilla RNN", lm_rnn), ("GRU", lm_gru), ("LSTM", lm_lstm)])
        _out = mo.vstack([_fig, mo.md(
            "| model | val loss | train time |\n|---|---|---|\n" + _rows + "\n| Module 14 MLP (8 chars) | 1.82 | — |"
            "\n| Module 16 GPT | ≈1.54 | — |\n\n"
            f"All three recurrent models beat the fixed-window MLP: a hidden state can carry more than 8 "
            f"characters of context. Best here: **{_best[0]}**. On 64-character windows the vanilla RNN does "
            "fine — most of what predicts the next character is nearby. The gates earn their keep when the "
            "*relevant* information is far away, which the next section measures directly. (MLP and GPT "
            "numbers were measured when Modules 14 and 16 were written; these runs are short — more steps "
            "lower all three curves.)")])
    else:
        _out = mo.md("*(waiting for your cells)*")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Measuring vanishing gradients

    Feed a 128-character window through a trained model, read a single number off the **final** hidden
    state, and ask: how big is its gradient with respect to the input at each earlier step? If that
    gradient is tiny, the model *cannot* learn to use that input — no learning signal reaches it.

    **Your turn:** implement `grad_norm_by_distance` (it's [`jax.grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html#jax.grad) through your `run_rnn`).
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lj, lm_gru, lm_lstm, lm_ready, lm_rnn, mo, np, plt, train_ids):
    _T = 128
    _starts = jax.random.randint(jax.random.key(5), (8,), 0, train_ids.shape[0] - _T)
    _win = jnp.stack([jax.lax.dynamic_slice(train_ids, (s,), (_T,)) for s in _starts])
    _vg = None
    if lm_ready:
        _curves = {}
        for _name, _kind, _cell, _r in [("vanilla RNN", "rnn", ex.rnn_cell, lm_rnn), ("GRU", "gru", ex.gru_cell, lm_gru),
                                        ("LSTM", "lstm", ex.lstm_cell, lm_lstm)]:
            _p = _r["params"]
            _H = _p["Wo"].shape[0]
            _xs = _p["emb"][_win.T]                                   # (T, 8, E)
            _w = jax.random.normal(jax.random.key(6), (_H,)) / np.sqrt(_H)
            _vg = lj.attempt(ex.grad_norm_by_distance, _cell, _p["cell"], M.zero_carry(_kind, (8,), _H), _xs, _w)
            if not _vg.ok:
                break
            _curves[_name] = np.asarray(_vg.value)[::-1]          # index 0 = the last input
        if _vg.ok:
            _fig, _ax = plt.subplots(figsize=(7, 3.2))
            for _name, _c in _curves.items():
                _ax.semilogy(np.arange(_T), _c / _c[0], label=_name)
            _ax.set(xlabel="steps back from the output", ylabel="gradient norm (relative to last step)")
            _ax.legend()
            _ax.grid(alpha=0.3, which="both")
            _at = {n: c[50] / c[0] for n, c in _curves.items()}
            _out = mo.vstack([_fig, mo.md(
                "Relative gradient 50 steps back: " + ", ".join(f"**{n}** {v:.1e}" for n, v in _at.items()) +
                ".\n\nThe vanilla RNN's signal from distant inputs collapses exponentially — a straight line "
                "falling on this log scale — which is the 1991/1994 discovery in one plot. The gated cells hold "
                "on to it far longer, because the cell state (LSTM) or the $z$-gated update (GRU) passes "
                "gradient back by addition rather than through a squashing nonlinearity every step.")])
        else:
            _out = _vg.view(mo)
    else:
        _out = mo.md("*(needs the trained models from Section 4)*")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Sequence to sequence, and the bottleneck

    A toy translation task, small enough to train in seconds: **reverse a string of digits** (length 4 to
    24). `3 1 4 1 5` → `5 1 4 1 3`.

    * **Encoder–decoder (Sutskever et al. 2014):** an LSTM reads the source; its final state is the only
      thing handed to a decoder LSTM, which writes the output one digit at a time, feeding back its own
      previous digit. Everything must fit through one 128-number vector.
    * **With attention (Bahdanau et al. 2014):** at every output step the decoder scores *every* encoder
      state with a small network, $\text{score}_s = v^\top \tanh(q W_q + k_s W_k)$, softmaxes the scores,
      and reads a weighted average (the *context*). Module 15's dot-product attention is the same idea
      with a cheaper score.

    Generation at test time is **greedy**: pick the most likely digit, feed it back, repeat.

    **Your turn:** implement `additive_attention` and `greedy_decode`. Both models also use your
    `lstm_cell` and `run_rnn`.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, lm_ready, mo):
    _ks = jax.random.split(jax.random.key(8), 4)
    _p = {"Wq": jax.random.normal(_ks[0], (4, 8)), "Wk": jax.random.normal(_ks[1], (4, 8)), "v": jnp.ones(8)}
    att_try = lj.attempt(ex.additive_attention, _p, jax.random.normal(_ks[2], (2, 4)),
                         jax.random.normal(_ks[3], (2, 6, 4)), jax.random.normal(_ks[3], (2, 6, 4)))

    def _toy(state, tok):
        return jax.nn.one_hot((tok + 1) % 5, 5), state

    dec_try = lj.attempt(ex.greedy_decode, _toy, jnp.zeros(()), jnp.array([0, 2]), 4)
    s2s_ready = lm_ready and att_try.ok and dec_try.ok
    _items = [mo.md("**`additive_attention`:**"), att_try.view(mo, show_value=False)]
    if att_try.ok:
        _items.append(mo.md(f"weights sum to {[round(float(s), 4) for s in att_try.value[1].sum(1)]} per row."))
    _items += [mo.md("**`greedy_decode`** on a toy decoder that always predicts *previous + 1*:"),
               dec_try.view(mo) if not dec_try.ok else mo.md(f"`{dec_try.value.tolist()}`")]
    mo.vstack(_items)
    return (s2s_ready,)


@app.cell
def _(M, ex, s2s_ready):
    s2s_fns = {k: getattr(ex, k) for k in ["lstm_cell", "run_rnn", "additive_attention", "greedy_decode"]}
    s2s_plain = M.train_s2s(s2s_fns, attention=False, steps=1500) if s2s_ready else None
    return s2s_fns, s2s_plain


@app.cell
def _(M, s2s_fns, s2s_ready):
    s2s_att = M.train_s2s(s2s_fns, attention=True, steps=1500) if s2s_ready else None
    return (s2s_att,)


@app.cell
def _(M, mo, np, plt, s2s_att, s2s_fns, s2s_plain, s2s_ready):
    if s2s_ready:
        _acc_plain = M.s2s_accuracy_by_length(s2s_fns, s2s_plain["params"], 4, 24)
        _acc_att = M.s2s_accuracy_by_length(s2s_fns, s2s_att["params"], 4, 24)
        _L = sorted(_acc_plain)
        _fig, _axs = plt.subplots(1, 2, figsize=(9, 3.2))
        for _acc, _name in [(_acc_plain, "no attention"), (_acc_att, "Bahdanau attention")]:
            _axs[0].plot(_L, [_acc[n][0] for n in _L], marker="o", ms=3, label=_name)
            _axs[1].plot(_L, [_acc[n][1] for n in _L], marker="o", ms=3, label=_name)
        _axs[0].set(xlabel="string length", ylabel="digit accuracy", ylim=(0, 1.05))
        _axs[1].set(xlabel="string length", ylabel="whole string exactly right", ylim=(0, 1.05))
        for _a in _axs:
            _a.grid(alpha=0.3)
            _a.legend(fontsize=8)
        _fig.tight_layout()
        _long = [n for n in _L if n >= 20]
        _p_long = np.mean([_acc_plain[n][1] for n in _long])
        _a_long = np.mean([_acc_att[n][1] for n in _long])
        _out = mo.vstack([_fig, mo.md(
            f"Training: {s2s_plain['seconds']:.0f} s without attention, {s2s_att['seconds']:.0f} s with.\n\n"
            f"Both models get most short strings right. For strings of 20–24 digits, the plain encoder–decoder gets "
            f"**{_p_long:.0%}** of them exactly right; with attention, **{_a_long:.0%}**. The plain model has to "
            "cram up to 24 digits into one 128-number state; the attention model can simply *look* at the "
            "digit it needs. That gap is the bottleneck Bahdanau et al. removed.")])
    else:
        _out = mo.md("*(finish `additive_attention` and `greedy_decode` — plus the cells — to train these)*")
    _out
    return


@app.cell
def _(M, jax, mo, np, plt, s2s_att, s2s_fns, s2s_ready):
    if s2s_ready:
        _src, _tgt, _len = M.reverse_batch(jax.random.key(11), 64, 16, 16)
        _, _w = M.s2s_teacher_forced(s2s_fns, s2s_att["params"], _src, _tgt)
        _w = np.asarray(_w[0])                                      # (target step, source position)
        _digits = [str(int(t) - M.OFFSET) for t in np.asarray(_src[0]) if t != M.PAD]
        _shift = int(np.asarray(_src[0] == M.PAD).sum())
        _fig, _ax = plt.subplots(figsize=(5, 4.4))
        _ax.imshow(_w[:16, _shift:], cmap="Blues", aspect="auto")
        _ax.set_xticks(range(len(_digits)), _digits)
        _ax.set_yticks(range(16), [str(int(t) - M.OFFSET) for t in np.asarray(_tgt[0])[:16]])
        _ax.set(xlabel="source digit (read left to right)", ylabel="output digit (written top to bottom)",
                title="where the decoder looks")
        _fig.tight_layout()
        _arg = _w[:16, _shift:].argmax(1)
        _off = _arg - (15 - np.arange(16))                         # 0 = looking at the digit being written
        _near = float(np.mean(np.abs(_off) <= 1))
        _mode = int(np.bincount(_off + 16).argmax()) - 16
        _where = ("exactly the digit it is about to write" if _mode == 0 else
                  f"{abs(_mode)} position{'s' if abs(_mode) > 1 else ''} to the "
                  f"{'right' if _mode > 0 else 'left'} of the digit it is about to write")
        _out = mo.vstack([_fig, mo.md(
            f"Each row is one output step; dark = high attention weight. The decoder walks **backwards** "
            f"through the source — an anti-diagonal. Most often it looks at {_where}; on **{_near:.0%}** of "
            "the output steps its strongest weight is within one position of that digit. (An offset of one is "
            "fine: the encoder LSTM reads left to right, so its state at a position still carries the digit "
            "it read just before.) Nobody told the model to align; the alignment fell out of training, which "
            "is what Bahdanau et al.'s title — *jointly learning to align and translate* — promises.")])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Why attention won: sequential vs parallel

    A recurrent network must finish step $t-1$ before starting step $t$ — `scan` is a loop the GPU runs
    one step at a time. Self-attention (Module 15) computes every position at once with big matrix
    multiplies. Below: one forward + backward pass of an LSTM layer versus one causal self-attention layer
    of the same width, batch 32, at growing sequence length, on your GPU.
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lj, lm_ready, mo, plt):
    if lm_ready:
        _D, _B = 128, 32
        _lp = M.init_cell(jax.random.key(12), "lstm", _D, _D)
        _ks = jax.random.split(jax.random.key(13), 3)
        _ap = [jax.random.normal(k, (_D, _D)) / _D**0.5 for k in _ks]

        def _lstm_loss(p, x):
            _, hs = ex.run_rnn(ex.lstm_cell, p, M.zero_carry("lstm", (_B,), _D), x)
            return jnp.sum(hs**2)

        def _att_loss(p, x):
            q, k, v = (x @ w for w in p)
            s = jnp.einsum("tbd,sbd->bts", q, k) / _D**0.5
            T = x.shape[0]
            s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -jnp.inf)
            return jnp.sum(jnp.einsum("bts,sbd->tbd", jax.nn.softmax(s, -1), v) ** 2)

        _lg, _ag = jax.jit(jax.grad(_lstm_loss)), jax.jit(jax.grad(_att_loss))
        _timing = []
        for _T in [64, 128, 256, 512, 1024]:
            _x = jax.random.normal(jax.random.key(_T), (_T, _B, _D))
            _timing.append((_T, lj.timeit(_lg, _lp, _x, iters=5)["median"] * 1e3,
                           lj.timeit(_ag, _ap, _x, iters=5)["median"] * 1e3))
        _T, _l, _a = zip(*_timing)
        _fig, _ax = plt.subplots(figsize=(6, 3.2))
        _ax.loglog(_T, _l, marker="o", label="LSTM (scan)")
        _ax.loglog(_T, _a, marker="s", label="causal self-attention")
        _ax.set(xlabel="sequence length T", ylabel="forward+backward (ms)")
        _ax.legend()
        _ax.grid(alpha=0.3, which="both")
        _rows = "\n".join(f"| {t} | {l:.1f} | {a:.1f} | {l / a:.0f}× |" for t, l, a in _timing)
        _out = mo.vstack([_fig, mo.md(
            "| T | LSTM ms | attention ms | LSTM / attention |\n|---|---|---|---|\n" + _rows + "\n\n"
            "Attention's cost grows as $T^2$ and the LSTM's only as $T$, yet attention was faster at "
            f"**{sum(l > a for _, l, a in _timing)} of {len(_timing)}** lengths here — its work is a few large, "
            "parallel matrix multiplies, while the LSTM issues $T$ small "
            "sequential steps that can't fill the GPU. That trade (more total arithmetic, far less waiting) "
            "is why the Transformer replaced RNNs on GPUs.")])
    else:
        _out = mo.md("*(needs your `lstm_cell` and `run_rnn`)*")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Then and now

    **Then (2014–2017):** LSTM encoder–decoders with attention were the state of the art; Google's 2016
    neural translation system was a deep stack of them. **Now:** the Transformer (Modules 15–17) kept the
    attention and dropped the recurrence, trading $O(T)$ sequential steps for $O(T^2)$ parallel work.

    But the recurrent idea came back. Inference with a KV cache (Module 18) *is* a recurrence over a
    growing state, and **state-space models** like Mamba (Module 36) are RNNs again — linear ones,
    designed so the loop can be computed as a parallel scan. The lesson of this module, the tension
    between carrying a compact memory and looking at everything, is still the central design question.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "The LSTM's cell state is updated by addition through a forget gate, so gradients don't pass "
            "through a squashing nonlinearity at every step": "a",
            "LSTMs have more parameters, so they remember more": "b",
            "LSTMs use ReLU instead of tanh": "c",
        },
        label="**Check yourself:** why does an LSTM keep gradient signal from distant steps better than a "
              "vanilla RNN?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right — the additive cell-state path (the 'constant error carousel') is the "
                              "fix, the same idea as a residual connection."), kind="success"),
        "b": mo.callout(mo.md("Not the reason — in Section 4 every cell had the same parameter budget. "
                              "It's *how* the state is updated."), kind="warn"),
        "c": mo.callout(mo.md("No — the LSTM uses sigmoid gates and tanh. The difference is the additive "
                              "cell-state update."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(31))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    * Andrej Karpathy, [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
      (2015) — character-level RNNs, the post that made them famous.
    * Christopher Olah, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
      (2015) — the classic diagrams of the gates.
    * Chris Olah & Shan Carter (Distill), [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
      (2016) — attention as it looked from inside the RNN era.
    * Jay Alammar, [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
      — seq2seq with attention, animated.

    **Next:** Module 32 — BERT vs GPT: what happened when the Transformer split into encoders and decoders.
    """)
    return


if __name__ == "__main__":
    app.run()
