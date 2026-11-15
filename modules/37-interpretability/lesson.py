import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="37 · Interpretability")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import json
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    ex = lj.load_exercises(mo.notebook_dir())
    ref = lj.load_exercises(mo.notebook_dir(), solutions=True)
    _spec = importlib.util.spec_from_file_location("interp37", mo.notebook_dir() / "interp.py")
    interp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(interp)
    RUNS = Path("/course/runs/37-interpretability")
    return RUNS, ex, interp, jax, jnp, json, lj, mo, np, plt, ref


@app.cell
def _(mo):
    mo.md(r"""
    # 37 · Interpretability: induction heads and world models (2020–2023)

    **Part 10, Milestones rebuilt in JAX.** Take Modules 15–17 first.

    **Goal:** open up models you train yourself and find *how* they do what they do. Not by asking them,
    but by measuring their insides.

    **You will:** build the instruments interpretability researchers use (attention entropy,
    previous-token and induction scores, head ablation, linear probes) · train a 2-layer attention-only
    transformer and watch an **induction circuit** appear in a sudden phase change · find the two heads
    that form it and switch them off · train a GPT on nothing but Othello move lists and read the **board**
    out of its activations with a linear probe · see why "mine vs theirs" beats "black vs white" · watch
    5 features squeeze into 2 dimensions as a pentagon (**superposition**).
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from.** For most of the deep-learning boom, trained networks were treated as black
    boxes: they worked, nobody could say how. Starting around 2017, Chris Olah and colleagues published a
    series in the online journal *Distill* on feature visualization and then on *circuits*, arguing that
    networks could be read like code (Olah et al., "Zoom In", 2020). The team carried the program to
    transformers at Anthropic: a mathematical framework for reading attention heads (Elhage et al., 2021),
    *induction heads* as the circuit behind in-context learning (Olsson et al., 2022), and *superposition*,
    why single neurons mix several features (Elhage et al., 2022). In parallel, Kenneth Li and colleagues
    trained a GPT on Othello moves and found a board inside it (2022); Neel Nanda and colleagues showed the
    board is *linearly* readable once you ask the right question (2023). Timeline: Module 00b.
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The story: can we read a neural network?

    **The black-box worry.** A trained network is millions of numbers. It can be right for the wrong
    reasons, it can fail in ways nobody anticipated, and "it got 95% on the test set" says nothing about
    *what* it learned. Interpretability asks for mechanisms: which internal parts compute what.

    **Circuits (2017–2020).** Olah and colleagues visualized what individual units in image models respond
    to (curve detectors, dog-head detectors) and traced how earlier units combine into later ones. Their
    claim, in "Zoom In" (2020): features are the units of computation, circuits are the wiring between them,
    and similar circuits recur across models. Module 12's learned edge filters were a first glimpse.

    **Transformer circuits (2021–).** Elhage et al. (2021) showed how to read small *attention-only*
    transformers exactly: each head writes into a shared **residual stream**, and heads in later layers can
    read what earlier heads wrote. Olsson et al. (2022) found a two-head pattern, the **induction head**,
    that completes `… A B … A → B` by finding the earlier `A` and copying what followed it. It appears
    suddenly during training, at the same moment the model gets better at using its context. They argue it
    is a major mechanism behind *in-context learning*: prompting with examples works partly because of it.

    **World models (2022–2023).** Do language models just stitch together surface statistics ("stochastic
    parrots", in Bender, Gebru, McMillan-Major & Shmitchell's 2021 phrase), or do they build internal models
    of the world their text describes? Li et al. (2022) trained a GPT on random legal Othello games, given
    only move sequences and never the board, and found that the board state could be *decoded* from its
    activations; editing those activations changed its predicted moves. Nanda, Lee & Wattenberg (2023)
    showed the board is encoded *linearly*, but as "my pieces vs theirs", not "black vs white". The debate
    is not settled for large language models. Othello is a clean case where a world model demonstrably
    forms.

    **Superposition (2022–).** Networks represent more features than they have dimensions, by storing them
    in nearly-orthogonal directions that interfere a little (Elhage et al., 2022). That's why single neurons
    are often "polysemantic", and why recent work uses **sparse autoencoders** to pull features back apart.

    **Why it matters.** Interpretability is one of the main hopes for *verifying* what a model is doing
    rather than trusting its outputs: catching deception, finding bugs, auditing a model before deployment.
    This lesson rebuilds the three classic results at small scale on your RX 7600 XT.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · The instruments

    Interpretability is measurement. Your four instruments (all in `exercises.py`):

    * **`attention_entropy`**: is a head focused (near 0) or spread out (near $\log n$)?
    * **`previous_token_score`**: mean attention from each token to the one just before it. A
      *previous-token head* scores high.
    * **`induction_score`** (Olsson et al.'s definition): for each token, find earlier occurrences of the
      same token and measure attention on the token that came *after* each one. An *induction head* scores
      high.
    * **`ablate_head`**: switch one head off (zero its output) and see what breaks.

    Checked on hand-made attention patterns:
    """)
    return


@app.cell
def _(ex, jnp, lj, mo, np):
    _T = 8
    _prev = np.zeros((_T, _T), np.float32)
    _prev[0, 0] = 1
    _prev[np.arange(1, _T), np.arange(_T - 1)] = 1
    _uniform = np.tril(np.ones((_T, _T), np.float32)) / np.arange(1, _T + 1)[:, None]
    _toks = np.array([[5, 9, 2, 7, 5, 9, 2, 7]])
    _ind = np.zeros((1, _T, _T), np.float32)
    _ind[0, :4, 0] = 1
    for _i in range(4, _T):
        _ind[0, _i, _i - 3] = 1                                # attend to the token after the earlier copy
    _e = lj.attempt(ex.attention_entropy, jnp.asarray(np.stack([_prev, _uniform])))
    _p = lj.attempt(ex.previous_token_score, jnp.asarray(_prev))
    _s1 = lj.attempt(ex.induction_score, jnp.asarray(_ind), jnp.asarray(_toks))
    _s2 = lj.attempt(ex.induction_score, jnp.asarray(_prev[None]), jnp.asarray(_toks))
    _rows = []
    if _e.ok:
        _rows.append(f"| `attention_entropy`, last row | previous-token head {float(_e.value[0, -1]):.2f} · uniform "
                     f"{float(_e.value[1, -1]):.2f} (= log 8 = {np.log(8):.2f}) |")
    if _p.ok:
        _rows.append(f"| `previous_token_score` of the previous-token head | {_p.value:.2f} |")
    if _s1.ok and _s2.ok:
        _rows.append(f"| `induction_score` on `5 9 2 7 5 9 2 7` | induction head {_s1.value:.2f} · previous-token "
                     f"head {_s2.value:.2f} |")
    _todo = [a.view(mo) for a in (_e, _p, _s1) if not a.ok]
    mo.vstack(([mo.md("| instrument | reading |\n|---|---|\n" + "\n".join(_rows))] if _rows else []) + _todo)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Induction heads, grown from scratch

    The model: **two layers of attention and nothing else**: no MLPs, no LayerNorm, 4 heads per layer,
    width 64 (Elhage et al. study exactly this kind of model because it can be read completely).

    The data: sequences `A A noise`, where `A` is a random run of tokens of random length (8–28) that
    immediately repeats. On the second copy, every next token is predictable, *but only* by remembering
    what followed the same token the first time. Because the length of `A` varies, a fixed "look back 20
    positions" trick doesn't work; the model has to *search by content*.

    We train a 2-layer model and, for comparison, a 1-layer one, and track the loss on the repeated part.
    Random guessing costs $\log 64 = 4.16$ nats.
    """)
    return


@app.cell
def _(interp, jax):
    ind2, ind2_hist, ind2_secs = interp.train_induction(jax.random.key(1), n_layers=2, steps=3000, lr=3e-3)
    ind1, ind1_hist, ind1_secs = interp.train_induction(jax.random.key(1), n_layers=1, steps=3000, lr=3e-3)
    return ind1_hist, ind1_secs, ind2, ind2_hist, ind2_secs


@app.cell
def _(ind1_hist, ind1_secs, ind2_hist, ind2_secs, mo, np, plt):
    _fig, _ax = plt.subplots(figsize=(7.5, 3.4))
    _ax.plot(ind2_hist["step"], ind2_hist["repeated"], label="2 layers: repeated half", lw=2)
    _ax.plot(ind1_hist["step"], ind1_hist["repeated"], label="1 layer: repeated half", lw=2)
    _ax.plot(ind2_hist["step"], ind2_hist["other"], ":", color="gray", label="2 layers: non-repeated tokens")
    _ax.axhline(np.log(64), color="k", lw=0.6)
    _ax.set(xlabel="training step", ylabel="loss (nats/token)", title="the induction phase change")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _r2 = np.array(ind2_hist["repeated"])
    _drop = ind2_hist["step"][int(np.argmax(_r2 < (_r2[0] + _r2[-1]) / 2))]
    mo.vstack([_fig, mo.md(
        f"The 2-layer model's loss on repeated tokens falls from {_r2[0]:.2f} to **{_r2[-1]:.2f}** nats, and "
        f"most of the fall happens over a few hundred steps around step **{_drop}**: a phase change, as Olsson "
        f"et al. saw. The 1-layer model ends at **{ind1_hist['repeated'][-1]:.2f}**: it can learn some token "
        f"statistics but not the two-step trick. Training took {ind2_secs:.0f} s and {ind1_secs:.0f} s.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Finding the circuit

    The two-step trick needs two heads in sequence:

    1. **Layer 0, a previous-token head**: each position copies in *which token came just before it*.
       Now the position holding `B` in `… A B …` "knows" it follows an `A`.
    2. **Layer 1, an induction head**: at the second `A`, its query looks for keys that "follow an `A`" (what
       the previous-token head wrote), finds the old `B`, and copies `B` into the prediction.

    Below: every head's scores on test sequences with a 20-token repeat, measured with *your* instruments,
    and the attention patterns of the strongest head in each layer.
    """)
    return


@app.cell
def _(ex, ind2, interp, jax, lj, mo, np, ref):
    probe_toks, probe_rep = interp.repeated_batch(jax.random.key(9), 128, 64, 64, 20, 20)
    _, ind2_attn = interp.attn_only_apply(ind2, probe_toks, 4)        # [L, B, H, T, T]
    _impl = ex
    _check = lj.attempt(ex.induction_score, ind2_attn[1, :, 0], probe_toks)
    if not (_check.ok and lj.attempt(ex.previous_token_score, ind2_attn[0, :, 0]).ok):
        _impl = ref
    head_scores = np.array([[[_impl.previous_token_score(ind2_attn[l, :, h]),
                              _impl.induction_score(ind2_attn[l, :, h], probe_toks),
                              float(_impl.attention_entropy(ind2_attn[l, :, h]).mean())]
                             for h in range(4)] for l in range(2)])
    _rows = "\n".join(f"| L{l}H{h} | {head_scores[l, h, 0]:.2f} | {head_scores[l, h, 1]:.2f} | {head_scores[l, h, 2]:.2f} |"
                      for l in range(2) for h in range(4))
    _note = "" if _impl is ex else "\n\n*(Scored with the reference instruments: finish yours to use them here.)*"
    mo.md("| head | previous-token score | induction score | mean attention entropy |\n|---|---|---|---|\n"
          + _rows + _note)
    return head_scores, ind2_attn, probe_rep, probe_toks


@app.cell
def _(head_scores, ind2_attn, mo, np, plt):
    _lp, _hp = 0, int(np.argmax(head_scores[0, :, 0]))
    _li, _hi = 1, int(np.argmax(head_scores[1, :, 1]))
    _fig, _axs = plt.subplots(1, 2, figsize=(9, 4.2))
    for _ax, (_l, _h, _name) in zip(_axs, [(_lp, _hp, "previous-token head"), (_li, _hi, "induction head")]):
        _ax.imshow(np.asarray(ind2_attn[_l, 0, _h, :48, :48]), cmap="magma")
        _ax.set_title(f"L{_l}H{_h}: {_name}", fontsize=10)
        _ax.set_xlabel("key position")
        _ax.set_ylabel("query position")
        _ax.axvline(20, color="w", lw=0.5, ls=":")
        _ax.axhline(20, color="w", lw=0.5, ls=":")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "Left: a thin diagonal just below the main one: every token attends to its predecessor. Right: in the "
        "second copy (queries below the dotted line), attention sits on a diagonal 19 positions back, the token "
        "*after* each earlier occurrence. That off-diagonal stripe is the signature of an induction head.")])
    return


@app.cell
def _(mo):
    head_choice = mo.ui.dropdown(
        options=["none"] + [f"L{_l}H{_h}" for _l in range(2) for _h in range(4)] + ["all of layer 1"],
        value="none", label="switch off")
    mo.vstack([mo.md("### Ablation: break the circuit\n\nSwitch heads off and re-measure the loss on repeated "
                     "tokens (uses your `ablate_head`)."), head_choice])
    return (head_choice,)


@app.cell
def _(ex, head_choice, head_scores, ind2, interp, jnp, lj, mo, np, probe_rep, probe_toks, ref):
    _abl = ex.ablate_head if lj.attempt(ex.ablate_head, jnp.ones((2, 4)), 0, 0).ok else ref.ablate_head

    def _rep_loss(mask):
        _lg, _ = interp.attn_only_apply(ind2, probe_toks, 4, head_mask=mask)
        return float(interp.split_losses(_lg, probe_toks, probe_rep)[1])

    _base = _rep_loss(jnp.ones((2, 4)))
    _prev_head = int(np.argmax(head_scores[0, :, 0]))
    _ind_head = int(np.argmax(head_scores[1, :, 1]))
    _m = jnp.ones((2, 4))
    if head_choice.value == "all of layer 1":
        for _h in range(4):
            _m = _abl(_m, 1, _h)
    elif head_choice.value != "none":
        _m = _abl(_m, int(head_choice.value[1]), int(head_choice.value[3]))
    _now = _rep_loss(_m)
    _pm = _abl(jnp.ones((2, 4)), 0, _prev_head)
    _im = _abl(jnp.ones((2, 4)), 1, _ind_head)
    _all1 = jnp.ones((2, 4)).at[1].set(0.0)
    mo.md(f"""
    Loss on repeated tokens: **{_base:.2f}** with every head on → **{_now:.2f}** with *{head_choice.value}* off.

    | switched off | repeated-token loss |
    |---|---|
    | nothing | {_base:.2f} |
    | the previous-token head L0H{_prev_head} | {_rep_loss(_pm):.2f} |
    | the strongest induction head L1H{_ind_head} | {_rep_loss(_im):.2f} |
    | all four layer-1 heads | {_rep_loss(_all1):.2f} |

    Removing the single previous-token head destroys copying almost completely: every induction head
    depends on what it wrote. Removing one induction head barely matters, because this model grew several
    (look at the induction scores above): **redundancy**. Removing all of layer 1 takes the model back to
    guessing. A circuit, found by measuring and confirmed by breaking it.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Othello-GPT: is there a board in there?

    **The setup (after Li et al., 2022).** Othello is played on an 8×8 board; a move places a disc and flips
    the opponent's discs it brackets. We generate random *legal* games on the GPU with
    [pgx](https://github.com/sotetsuk/pgx) (Koyamada et al.; thousands of games in parallel under
    [`jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit)), and train a small GPT to predict the next move from the move list alone: tokens 0–63 are
    squares, 64 is "pass". **The model never sees a board.**

    If it predicts *legal* moves well, it must be tracking which squares are occupied and by whom. Does it
    build an internal board, or memorize move statistics? Train a probe and find out.

    The full model (6 layers, width 128, 60,000 games, about 10 minutes) trains in `train.py`:

        containers/run.sh python modules/37-interpretability/train.py

    The lesson loads it if it exists, and otherwise trains a small one live.
    """)
    return


@app.cell
def _(RUNS, interp, jax, json, mo):
    othello_games = interp.gen_games(jax.random.key(123), 1200)               # held out: a different seed
    _meta_path, _w_path = RUNS / "othello_gpt.json", RUNS / "othello_gpt.npz"
    if _w_path.exists() and _meta_path.exists():
        othello_meta = json.loads(_meta_path.read_text())
        _like = interp.gpt_init(jax.random.key(0), interp.OTHELLO_VOCAB, othello_meta["d"], othello_meta["n_layers"],
                                othello_meta["n_heads"], 64)
        othello_params = interp.load_params(_w_path, _like)
        othello_source = (f"Loaded `train.py`'s model: {othello_meta['n_layers']} layers, width {othello_meta['d']}, "
                          f"{othello_meta['steps']} steps on {othello_meta['games']:,} games "
                          f"({othello_meta['train_seconds']:.0f} s).")
    else:
        _train = interp.gen_games(jax.random.key(7), 12000)
        othello_params, _h, _s = interp.train_othello_gpt(jax.random.key(1), _train, d=128, n_layers=4, steps=600)
        othello_meta = {"d": 128, "n_layers": 4, "n_heads": 4, "steps": 600, "games": 12000, "loss": _h["loss"]}
        othello_source = (f"No `train.py` checkpoint yet, so a small model was trained live (4 layers, 600 steps, "
                          f"{_s:.0f} s). Its board probes will be weaker; run `train.py` for the full result.")
    othello_heads = othello_meta["n_heads"]
    othello_acc = interp.legal_move_accuracy(othello_params, othello_games, othello_heads, n_eval=1000)
    _nxt = othello_games["valid"][:, 1:]
    _random_legal = float((othello_games["legal"][:, :-1].sum(-1) / 65)[_nxt].mean())
    mo.md(f"{othello_source}\n\n**Legal-move accuracy on held-out games: {othello_acc:.1%}** (the model's single "
          f"top-ranked next move is legal). Picking uniformly among the 65 possible moves would be legal "
          f"{_random_legal:.1%} of the time.")
    return othello_acc, othello_games, othello_heads, othello_meta, othello_params, othello_source


@app.cell
def _(mo):
    mo.md(r"""
    ### Linear probes: reading the board

    A **linear probe** is a deliberately weak instrument: logistic regression from a layer's activations to
    some property. If a linear probe can read a property, the model has it laid out along directions in
    its residual stream, available to anything downstream. A powerful nonlinear probe could find structure
    the model doesn't use, which is why linearity matters here.

    For every position in 1,200 held-out games, we take the residual stream after each layer and train a
    probe to predict all 64 squares at once (3 classes each), with *your* `train_linear_probe`, in two
    framings:

    * **absolute**: empty / black / white;
    * **relative**: empty / *mine* / *theirs*, from the view of the player about to move.

    The board is the same; only the labels' point of view differs. As a control, the same probes are
    trained on an *untrained* copy of the model.
    """)
    return


@app.cell
def _(ex, interp, jax, lj, mo, np, othello_games, othello_heads, othello_meta, othello_params, ref):
    _acts, _rel, _abs = interp.collect_resid(othello_params, othello_games, othello_heads, 1200)
    _rand = interp.gpt_init(jax.random.key(99), interp.OTHELLO_VOCAB, othello_meta["d"], othello_meta["n_layers"],
                            othello_heads, 64)
    _racts, _, _ = interp.collect_resid(_rand, othello_games, othello_heads, 1200)
    _n = _acts.shape[1]
    _tr = int(_n * 0.8)
    _impl = ex
    _t = lj.attempt(ex.train_linear_probe, _acts[0][:200], _rel[:200], 3, 5, 1e-2)
    if not (_t.ok and lj.attempt(ex.probe_accuracy, *_t.value, _acts[0][:200], _rel[:200]).ok):
        _impl = ref

    def _score(a, y):
        _W, _b = _impl.train_linear_probe(a[:_tr], y[:_tr], 3, 300, 1e-2)
        return _impl.probe_accuracy(_W, _b, a[_tr:], y[_tr:]), (_W, _b)

    probe_rel, probe_abs, probe_rand, probe_models = [], [], [], []
    for _l in range(_acts.shape[0]):
        _r, _m = _score(_acts[_l], _rel)
        probe_rel.append(_r)
        probe_models.append(_m)
        probe_abs.append(_score(_acts[_l], _abs)[0])
        probe_rand.append(_score(_racts[_l], _rel)[0])
    _counts = np.stack([(_rel[_tr:] == c).mean(0) for c in range(3)])           # per-square class frequencies
    probe_majority = float(_counts.max(0).mean())
    probe_data = {"acts": _acts, "rel": _rel, "tr": _tr}
    _note = "" if _impl is ex else " *(Reference probes used: finish `train_linear_probe` / `probe_accuracy` to use yours.)*"
    mo.md(f"Probes trained on {_tr:,} positions, tested on {_n - _tr:,}.{_note}")
    return probe_abs, probe_data, probe_majority, probe_models, probe_rand, probe_rel


@app.cell
def _(mo, np, plt, probe_abs, probe_majority, probe_rand, probe_rel):
    _L = np.arange(len(probe_rel))
    _fig, _ax = plt.subplots(figsize=(7.5, 3.6))
    _ax.plot(_L, probe_rel, "o-", lw=2, label="relative (empty / mine / theirs)")
    _ax.plot(_L, probe_abs, "s-", lw=2, label="absolute (empty / black / white)")
    _ax.plot(_L, probe_rand, "^:", color="gray", label="relative, untrained model")
    _ax.axhline(probe_majority, color="k", lw=0.8, ls="--", label="always guess the most common class")
    _ax.set(xlabel="residual stream after layer (0 = embeddings)", ylabel="square accuracy",
            title="reading the board out of Othello-GPT", ylim=(0.4, 1.0))
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _best = int(np.argmax(probe_rel))
    _gap = probe_rel[_best] - probe_abs[_best]
    _verdict = ("the relative framing is clearly easier to read linearly, as Nanda et al. found"
                if _gap > 0.03 else "the two framings are close at this scale; Nanda et al.'s larger model shows a wide gap")
    mo.vstack([_fig, mo.md(
        f"Best layer: **{_best}**, relative probe **{probe_rel[_best]:.1%}** vs absolute **{probe_abs[_best]:.1%}** "
        f"(gap {_gap:+.1%}); an untrained model gives {probe_rand[_best]:.1%} and guessing the most common class "
        f"{probe_majority:.1%}. So {_verdict}. Accuracy climbs through the middle layers and can dip at the end, "
        f"where the model turns the board into a move prediction.")])
    return


@app.cell
def _(mo, np, plt, probe_data, probe_models, probe_rel):
    _best = int(np.argmax(probe_rel))
    _W, _b = probe_models[_best]
    _i = probe_data["tr"] + 40                                                   # a held-out position
    _x = probe_data["acts"][_best][_i]
    _pred = np.asarray(np.argmax(np.einsum("d,dkc->kc", _x, np.asarray(_W)) + np.asarray(_b), -1))
    _true = probe_data["rel"][_i]
    _cmap = plt.matplotlib.colors.ListedColormap(["#2e7d32", "#f5f5f5", "#212121"])
    _fig, _axs = plt.subplots(1, 2, figsize=(7, 3.6))
    for _ax, _board, _t in zip(_axs, [_true, _pred], ["true board", f"read from layer {_best}"]):
        _ax.imshow(_board.reshape(8, 8), cmap=_cmap, vmin=0, vmax=2)
        _ax.set_title(_t, fontsize=10)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.suptitle("green = empty · white = mine · black = theirs", fontsize=9)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"{int((_pred == _true).sum())} of 64 squares read correctly for this position, from a "
                           f"{probe_data['acts'].shape[-1]}-number activation vector the model computed only from "
                           "the list of moves so far.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Li et al. went one step further: they **edited** the residual stream to change one square in the internal
    board, and the model's predicted legal moves changed to match the edited board. That is evidence the
    board is not just *decodable* but *used*. Nanda et al. did the same with simple vector arithmetic along
    the probe's "mine minus theirs" direction. It's the natural next experiment for your model: `gpt_apply`
    takes an `edit(layer, x)` hook for exactly that.

    **What this does and doesn't show.** It shows that next-token prediction on a symbolic sequence can
    produce a structured internal model of the process that generated the sequence. It does not settle
    whether large language models build world models of everything they talk about: text is far messier
    than Othello, and "decodable by a probe" is weaker than "used by the model".
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Superposition: more features than dimensions

    Elhage et al.'s toy model (2022): 5 features, each active (uniform in [0, 1]) only rarely, must pass
    through a **2-dimensional** bottleneck and come back out: $x' = \mathrm{ReLU}(W^\top W x + b)$. With
    dense features the model keeps the 2 it can fit and drops the rest. With **sparse** features, collisions
    are rare, so it can afford to store all 5 in nearly-opposed directions and let the ReLU clean up the
    interference. Each arrow below is one feature's direction (a column of $W$).
    """)
    return


@app.cell
def _(interp, jax):
    superposition_runs = {_sp: interp.train_superposition(jax.random.key(0), sparsity=_sp)
                          for _sp in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99)}
    return (superposition_runs,)


@app.cell
def _(mo):
    sparsity_choice = mo.ui.slider(steps=[0.0, 0.5, 0.8, 0.9, 0.95, 0.99], value=0.95, label="feature sparsity",
                                   show_value=True)
    sparsity_choice
    return (sparsity_choice,)


@app.cell
def _(mo, np, plt, sparsity_choice, superposition_runs):
    _W, _b, _loss = superposition_runs[sparsity_choice.value]
    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    for _i in range(_W.shape[1]):
        _ax.arrow(0, 0, _W[0, _i], _W[1, _i], width=0.02, length_includes_head=True, color=f"C{_i}")
        _ax.text(_W[0, _i] * 1.15, _W[1, _i] * 1.15, f"f{_i}", ha="center", va="center", fontsize=9)
    _ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal", title=f"sparsity {sparsity_choice.value}")
    _ax.axhline(0, color="gray", lw=0.4)
    _ax.axvline(0, color="gray", lw=0.4)
    _fig.tight_layout()
    _norms = np.linalg.norm(_W, axis=0)
    mo.vstack([_fig, mo.md(f"Features represented (length > 0.5): **{int((_norms > 0.5).sum())} of 5** · loss {_loss:.4f}. "
                           "At 0.95 and above all five sit 72° apart: the pentagon.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Why this matters, and where it went next

    * **In-context learning has a mechanism.** Induction heads are a concrete, measurable part of why
      prompting with examples works. The "few-shot" ability of GPT-3 (Module 32's story) is partly this
      circuit, at scale.
    * **Models can learn latent structure.** Othello-GPT shows next-token prediction producing an internal
      model of the game. Probing and editing are now standard tools for asking what a model represents.
    * **Superposition is why it's hard.** If features share neurons, you can't read a model one neuron at a
      time. Sparse autoencoders (2023–24) learn an over-complete set of directions that untangle features,
      and have been scaled to production language models.
    * **Safety.** Interpretability is one of the main research bets for checking what a model will do before
      trusting it, rather than only judging its outputs.

    Your own models are now open to the same instruments: try `induction_score` on the heads of Module 16's
    GPT, or probe Module 26's reasoning model for whether it "knows" the answer before it writes it.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Removing the previous-token head breaks copying more than removing one induction head": "a",
            "A 1-layer attention-only model can learn induction if trained long enough": "b",
            "The probe proves the model uses the board to choose moves": "c",
        },
        label="**Check yourself:** which statement do this lesson's measurements support?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right: every induction head reads what the single previous-token head wrote, while "
                              "the induction heads back each other up (redundancy)."), kind="success"),
        "b": mo.callout(mo.md("No: an induction head needs a previous-token head *before* it to know what each "
                              "token followed. One layer can't compose the two steps (Elhage et al. 2021)."), kind="warn"),
        "c": mo.callout(mo.md("Not on its own: a probe shows the board is *decodable*. Showing it's *used* needs an "
                              "intervention: edit the activations and watch predictions change (Li et al. 2022)."),
                        kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(37))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Neel Nanda, [A Comprehensive Mechanistic Interpretability Explainer & Glossary](https://www.neelnanda.io/mechanistic-interpretability/glossary).
    * Kenneth Li, [Do Large Language Models learn world models or just surface statistics?](https://thegradient.pub/othello/)
      (The Gradient, 2023): the Othello-GPT story by its first author.
    * Bender, Gebru, McMillan-Major & Shmitchell, [On the Dangers of Stochastic Parrots](https://doi.org/10.1145/3442188.3445922)
      (FAccT 2021): the other side of the world-model debate.
    * Anthropic, [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
      (Bricken et al., 2023): sparse autoencoders on a real transformer.
    * Sotetsu Koyamada et al., [pgx](https://github.com/sotetsuk/pgx): the GPU game environments used here.

    **Next:** that's the last milestone. Go back to the capstone (Module 25) with these instruments in hand.
    """)
    return


if __name__ == "__main__":
    app.run()
