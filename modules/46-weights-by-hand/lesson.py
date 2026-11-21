import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="46 · Weights by hand")


@app.cell
def _():
    import marimo as mo
    import importlib.util

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import rosetta

    ex = lj.load_exercises(mo.notebook_dir())
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")  # re-runs skip XLA compilation
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    _spec = importlib.util.spec_from_file_location("m46_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    Xtr, ytr, Xte, yte = M.mnist_arrays()
    return M, Xte, Xtr, ex, jax, jnp, lj, mo, np, plt, rosetta, yte, ytr


@app.cell
def _(mo):
    mo.md(r"""
    # 46 · Weights by hand: compiling, merging and editing networks (1990s–2026)

    **Take first:** Module 12 (CNNs), Module 15 (attention), Module 20 (fine-tuning SmolLM2) and Module 37
    (reading a model's internals).

    **Goal:** every network in this course got its weights from gradient descent. This module sets weights
    some other way, four times: by hand from what we know about the task, by compiling a program, by
    adding and subtracting other networks' weights, and by solving for a single edit. Each one shows what
    weights *are*: numbers a person can read, write and reason about. The last section looks at a 2026 idea
    that does the same with an LLM in the chair: **weight-space programming**.

    **You will:** build a digit classifier with no training and score it against Adam · compile two small
    programs into a transformer's weights and check them exactly · merge two fine-tunes of SmolLM2 by weight
    arithmetic · move the Eiffel Tower to Rome in SmolLM2-360M with one rank-one change · see what LLMs did
    when asked to write weights themselves.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Before backpropagation won, people wrote features by hand: 1980s and 1990s character recognizers used
    stroke and corner detectors that engineers drew, with a statistical readout on top; LeCun's 1989 work
    showed that learning those features beat drawing them. Hand-set weights came back as a research tool:
    **RASP** (Weiss et al., 2021) described what a transformer can compute as a little programming language,
    and DeepMind's **Tracr** (2023) compiled such programs into real weights. Editing a trained model came
    next: **ROME** (Meng et al., 2022) located a fact in one MLP and rewrote it with a rank-one change, and
    **task arithmetic** (Ilharco et al., 2022), **TIES** (2023) and **DARE** (2023) merged fine-tunes by
    adding their weight changes. In November 2026 **weight-space programming** (Takehi et al.) had an LLM
    read and edit another model's weights with no gradients at all.

    *Timeline: Module 00b.*
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · A network with no training

    The network is small and fixed: Conv 1→8 (5 × 5), ReLU, max-pool; Conv 8→16 (5 × 5), max-pool; Linear
    256→10. That is **5,994 weights**, the same CNN Lex Toumbourou used to test LLM-written weights (section 5).
    Instead of training it, we decide what each weight should do.

    **Layer 1: stroke detectors.** A digit is pen strokes, and a stroke is a bright line with dark on both
    sides. A filter that is positive along a line and negative beside it lights up on strokes at that angle
    and stays quiet on flat paper. Eight of them, every 22.5°, cover all directions.

    **Your turn:** `ridge_filter(theta)`, a narrow bump minus a wide one across the line, faded along it.
    """)
    return


@app.cell
def _(M, ex, lj, mo, np):
    _a = lj.attempt(lambda: np.stack([np.asarray(ex.ridge_filter(t)) for t in M.THETAS]))
    if _a.ok:
        _f = _a.value
        _ok = np.allclose(_f.sum((1, 2)), 0, atol=1e-5) and np.allclose(np.abs(_f).sum((1, 2)), 1, atol=1e-4)
        _line = np.zeros((5, 5))
        _line[2] = 1
        _resp = (_f * _line).sum((1, 2))
        _out = mo.md(f"**`ridge_filter`** {'✅' if _ok else '❌ (each filter should sum to 0 with |·| summing to 1)'}: "
                     f"8 filters. Response to a horizontal stroke through the centre, by filter angle: "
                     + ", ".join(f"{d:.1f}°: {r:+.3f}" for d, r in zip(np.degrees(M.THETAS), _resp))
                     + ". The 0° filter wins, the 90° one barely reacts.")
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(M, ex, lj, mo, np, plt):
    _a = lj.attempt(lambda: np.stack([np.asarray(ex.ridge_filter(t)) for t in M.THETAS]))
    _adam = M.read_json(M.RUNS / "mnist.json")
    if not _a.ok:
        _out = mo.md("*Your filters appear here once `ridge_filter` works.*")
    else:
        _rows = 2 if _adam else 1
        _fig, _axs = plt.subplots(_rows, 8, figsize=(9, 1.3 * _rows + 0.3), squeeze=False)
        for _i in range(8):
            _axs[0, _i].imshow(_a.value[_i], cmap="RdBu_r", vmin=-0.12, vmax=0.12)
            _axs[0, _i].set_title(f"{np.degrees(M.THETAS[_i]):.1f}°", fontsize=7)
        if _adam:
            _w = np.asarray(_adam["seed0"]["5000"]["w1"])[:, :, 0, :]
            for _i in range(8):
                _m = np.abs(_w[:, :, _i]).max()
                _axs[1, _i].imshow(_w[:, :, _i], cmap="RdBu_r", vmin=-_m, vmax=_m)
            _axs[1, 0].set_ylabel("Adam,\n5,000 steps", fontsize=7)
        _axs[0, 0].set_ylabel("yours", fontsize=7)
        for _ax in _axs.flat:
            _ax.set_xticks([])
            _ax.set_yticks([])
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Adam's filters (bottom row, if `train.py --mnist` has run) are messier, but several are the same idea:
    a light band beside a dark one, at some angle. Learned or drawn, layer 1 finds edges and strokes.

    **Layer 2: lines and corners** (provided, in `models.py`). Each of the 16 filters looks at the eight
    stroke maps (12 × 12 after pooling). Eight are **long lines**: stroke angle *o* present along a 5-cell line
    at angle *o*, a real stroke rather than a speck. Eight are **corners**: angle *o* in the top-left and the
    perpendicular angle in the bottom-right. Their weights are large and their activation is a ReLU capped at
    1, so each becomes a switch: after pooling, 256 features (16 detectors × 4 × 4 places) that are each
    either *fired* or *not*.

    **Unit-testing a feature.** Before trusting a feature, measure it: on the 60,000 training images, how often
    does it fire for each digit? A feature that fires for every digit, or for none, is useless.

    **Your turn:** `firing_rates(fires, labels, n_classes)`, smoothed so no rate is exactly 0 or 1.
    """)
    return


@app.cell
def _(M, Xte, Xtr, ex, lj, mo, np):
    _a = lj.attempt(M.hand_layers, ex)
    if _a.ok:
        hand = _a.value
        F_train = M.features(hand, Xtr) > 0.5
        F_test = M.features(hand, Xte) > 0.5
        _out = mo.md(f"Layers 1–2 built: **{M.count_weights(hand):,} weights**. On average a training image fires "
                     f"**{F_train.sum(1).mean():.0f}** of the 256 features ({F_train.mean():.0%}).")
    else:
        hand, F_train, F_test = None, None, None
        _out = _a.view(mo)
    _out
    return F_test, F_train, hand


@app.cell
def _(F_train, ex, lj, mo, np, plt, ytr):
    rates = None
    if F_train is None:
        _out = mo.md("*Firing rates need `ridge_filter`.*")
    else:
        _a = lj.attempt(ex.firing_rates, F_train, ytr, 10)
        if _a.ok:
            rates = np.asarray(_a.value)
            _by_det = rates.reshape(10, 4, 4, 16).mean((1, 2))           # average over the 16 places
            _fig, _ax = plt.subplots(figsize=(7.5, 3))
            _im = _ax.imshow(_by_det, cmap="viridis", aspect="auto")
            _ax.set_yticks(range(10))
            _ax.set_ylabel("digit")
            _ax.set_xticks(range(16), [f"line {i}" for i in range(8)] + [f"corner {i}" for i in range(8)],
                           rotation=60, fontsize=7)
            _fig.colorbar(_im, ax=_ax, label="firing rate")
            _ax.set_title("How often each detector fires, by digit (averaged over its 16 places)", fontsize=9)
            _fig.tight_layout()
            _spread = _by_det.max(0) - _by_det.min(0)
            _best = int(_spread.argmax())
            _name = f"line {_best}" if _best < 8 else f"corner {_best - 8}"
            _few, _many = int(_by_det.mean(1).argmin()), int(_by_det.mean(1).argmax())
            _out = mo.vstack([_fig, mo.md(
                f"The most selective detector is **{_name}**: it fires on {_by_det[:, _best].max():.0%} of "
                f"{int(_by_det[:, _best].argmax())}s and {_by_det[:, _best].min():.0%} of "
                f"{int(_by_det[:, _best].argmin())}s. Each row is how a digit looks to the network: {_few}s fire "
                f"the fewest detectors on average ({_by_det[_few].mean():.0%}), {_many}s the most "
                f"({_by_det[_many].mean():.0%}).")])
        else:
            _out = _a.view(mo)
    _out
    return (rates,)


@app.cell
def _(mo):
    mo.md(r"""
    **The readout, from counting.** The last layer turns 256 fired-or-not features into 10 scores. Treat the
    features as independent once you know the digit (they aren't, quite) and Bayes' rule gives

    $$\log P(c \mid x) = \sum_f \big[x_f \log r_{cf} + (1 - x_f)\log(1 - r_{cf})\big] + \log P(c) + \text{const},$$

    which is a linear layer: $W_{fc} = \log r_{cf} - \log(1-r_{cf})$, the log-odds that feature $f$ fires for
    digit $c$, and a bias per digit. This is **naive Bayes**, and it is how 1990s character recognizers
    combined hand-made features. No gradient anywhere: the weights are counts.

    **Your turn:** `log_odds_readout(rates, prior)`.
    """)
    return


@app.cell
def _(F_test, M, Xte, ex, hand, jnp, lj, mo, np, rates, yte, ytr):
    if rates is None:
        _out = mo.md("*The readout needs `firing_rates`.*")
    else:
        _a = lj.attempt(ex.log_odds_readout, rates, np.bincount(ytr, minlength=10) / len(ytr))
        if _a.ok:
            _W, _b = _a.value
            _net = dict(hand, w3=jnp.asarray(_W, jnp.float32), b3=jnp.asarray(_b, jnp.float32))
            _hand_acc = M.accuracy(_net, Xte, yte)
            _bin = float(((F_test @ np.asarray(_W) + np.asarray(_b)).argmax(1) == yte).mean())
            _adam = M.read_json(M.RUNS / "mnist.json")
            _rows = ""
            if _adam:
                for _s in ("50", "500", "5000"):
                    _v = [r[_s]["test_acc"] for r in _adam.values()]
                    _rows += (f"| Adam, {int(_s):,} steps × 64 images ({len(_v)} seeds) | "
                              f"{min(_v):.1%}–{max(_v):.1%} | {int(_s) * 64:,} |\n")
            _out = mo.md(f"""
    | network (same 5,994 weights) | MNIST test accuracy | training images seen |
    |---|---|---|
    | **set by hand: your filters, line and corner detectors, your counted readout** | **{_hand_acc:.1%}** | 60,000 (counted once) |
    {_rows}
    The whole network runs as a normal JAX CNN ({_bin:.1%} when the readout is fed the exact 0/1 features). It
    {"beats" if _adam and _hand_acc > max(r["50"]["test_acc"] for r in _adam.values()) else "is close to"}
    Adam after 50 steps and loses to Adam after a few hundred. Toumbourou's
    note on fairness applies: Adam at 50 steps has seen 3,200 images, while counting saw all 60,000. And we
    chose these detectors knowing what digits look like; a network that learns them needs no such knowledge,
    which is why learning won.
    """)
        else:
            _hand_acc = None
            _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "naive Bayes readout for binary features", solves="log_odds_readout")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Compiling a program into a transformer

    Section 1 set weights from knowledge of the data. You can also set them from a **program**. Weiss et al.
    noticed that attention is a "select, then aggregate" operation and wrote a tiny language for it, **RASP**;
    Tracr (DeepMind, 2023) compiles RASP programs into actual transformer weights.

    The trick is to treat the residual stream as named variables. Here each position's vector is
    `[token one-hot (6) | position one-hot (9)]`, and a head's query and key weights read whichever variables
    the program's `select` compares. To **reverse** a sequence, position $i$ must read position $L-1-i$:
    compare positions, put a large score on the matching pair, and the softmax does the rest.

    **Your turn:** `position_selector(target)`, the $W_Q, W_K$ that make position $i$ attend to `target[i]`.
    """)
    return


@app.cell
def _(M, ex, jnp, lj, mo, np, plt):
    _seq = np.array([3, 1, 4, 1, 0, 2, 2, 3])
    _a = lj.attempt(M.reverse_head, ex, len(_seq))
    if _a.ok:
        _out_v, _attn = M.attention(M.embed(_seq), *_a.value)
        _got = np.asarray(_out_v).argmax(1)
        _rng = np.random.default_rng(0)
        _exact = 0
        for _ in range(1000):
            _s = _rng.integers(0, 5, 8)
            _exact += int(np.array_equal(np.asarray(M.attention(M.embed(_s), *_a.value)[0]).argmax(1), _s[::-1]))
        _fig, _ax = plt.subplots(figsize=(2.8, 2.6))
        _ax.imshow(np.asarray(_attn), cmap="Greys")
        _ax.set_xlabel("reads position")
        _ax.set_ylabel("query position")
        _ax.set_title("the compiled reverse head", fontsize=9)
        _fig.tight_layout()
        _ok = np.array_equal(_got, _seq[::-1])
        _out = mo.hstack([_fig, mo.md(
            f"**`position_selector`** {'✅' if _ok else '❌'}: input `{_seq.tolist()}` → `{_got.tolist()}`. "
            f"Exactly right on **{_exact:,} of 1,000** random sequences; the attention left over for every other "
            f"position is {float(1 - jnp.max(_attn, 1).min()):.1e} (e^-50 is below float32's resolution next to 1). "
            "With `target = i - 1` the same "
            "function builds a **previous-token head**, the first half of Module 37's induction circuit.")],
            widths=[1, 2])
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(M, mo, np):
    _seq = np.array([3, 1, 4, 1, 0, 2, 2, 3])
    _counts, _share, _ = M.run_histogram(_seq)
    _rng = np.random.default_rng(1)
    _exact = 0
    for _ in range(1000):
        _s = _rng.integers(0, 5, _rng.integers(1, 9))
        _exact += int(np.array_equal(M.run_histogram(_s)[0], [(_s == t).sum() for t in _s]))
    mo.md(f"""
    **A program that needs an MLP: histogram** (provided). "How many times does my token appear?" Step one is a
    head where every token attends equally to all copies of itself *and* to a BOS token at the start; the share
    of attention that lands on BOS is $1/(c+1)$ for a token that appears $c$ times. Step two is an MLP that
    turns $1/(c+1)$ back into $c$: pairs of ReLUs make sharp steps at the midpoints between $1/(c+1)$ and
    $1/c$, and differences of steps make a one-hot count (`count_mlp` in `models.py`, 16 hidden units).

    On `{_seq.tolist()}` the BOS shares are {", ".join(f"{x:.3f}" for x in _share)} and the network outputs
    counts `{_counts.tolist()}`. Exactly right on **{_exact:,} of 1,000** random sequences of 1–8 tokens.

    This is what "a transformer is a program" means, made literal. It also shows why reading learned weights
    is hard: Tracr's weights are clean one-hots, while trained weights spread the same computation over many
    directions at once (Module 47).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Arithmetic on weights: merging fine-tunes

    Fine-tune a model and subtract the original weights: what's left is a **task vector**
    $\tau = \theta_{\text{fine-tuned}} - \theta_{\text{base}}$, the change the task made. Ilharco et al. (2022)
    found you can do arithmetic with these. Add two and the model does both tasks; subtract one and it
    forgets that task. It works because fine-tuning moves a big model only a little, and the moves for
    different tasks barely overlap.

    `train.py --merge` fine-tunes SmolLM2-135M (the base model from Module 20) twice from the same weights:
    once on Shakespeare, once on radio-astronomy abstracts (Module 38's corpus). It then evaluates every merge
    on both, plus detective novels that neither fine-tune saw, as a check on what merging costs elsewhere.

    **TIES** (Yadav et al., 2023) adds three steps for when tasks fight: keep each task vector's largest 20% of
    changes, let the tasks vote on each weight's sign, and average only the changes that agree.

    **Your turn:** `ties_merge(taus, density, lam)` on flattened task vectors (`models.py` runs it on every
    weight matrix of the model, layer by layer).
    """)
    return


@app.cell
def _(ex, lj, mo, np):
    _taus = np.array([[3.0, -1.0, 0.5, 2.0], [1.0, 2.0, -0.2, -4.0]])
    _a = lj.attempt(ex.ties_merge, _taus, 0.5, 1.0)
    if _a.ok:
        _got = np.asarray(_a.value)
        _ok = np.allclose(_got, [3.0, 2.0, 0.0, -4.0])
        _out = mo.md(f"""
    **`ties_merge`** {'✅' if _ok else '❌ (expected [3, 2, 0, −4])'} on two 4-weight task vectors, keeping half:

    | | weight 1 | weight 2 | weight 3 | weight 4 |
    |---|---|---|---|---|
    | task A | 3.0 | −1.0 | 0.5 | 2.0 |
    | task B | 1.0 | 2.0 | −0.2 | −4.0 |
    | trimmed A / B | 3 / 0 | 0 / 2 | 0 / 0 | 2 / −4 |
    | merged | {" | ".join(f"{x:+.1f}" for x in _got)} |

    Weight 4 is a conflict: A pushed it up by 2, B down by 4. Plain averaging would give −1, a change neither
    task asked for; TIES lets the bigger push win and keeps it at full size. Plain averaging also halves
    every change only one task made (weights 1 and 2); TIES keeps those at full size too.
    """)
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "TIES merging of task vectors", solves="ties_merge")
    return


@app.cell
def _(M, mo):
    _r = M.read_json(M.RUNS / "merge.json")
    if not _r:
        _out = mo.md("*Run `train.py --merge` (about 15 minutes) to fill this in.*")
    else:
        _names = [k for k in _r if k not in ("curves", "stats")]
        _b = _r["base"]
        _ft_sh, _ft_as = _r["fine-tuned on shakespeare"], _r["fine-tuned on astro"]
        _gain_sh = _b["shakespeare"] - _ft_sh["shakespeare"]
        _gain_as = _b["astro"] - _ft_as["astro"]

        def _kept(row):
            return ((_b["shakespeare"] - row["shakespeare"]) / _gain_sh, (_b["astro"] - row["astro"]) / _gain_as)

        _rows = "\n".join(
            f"| {n} | {_r[n]['shakespeare']:.3f} | {_r[n]['astro']:.3f} | {_r[n]['detective']:.3f} | "
            + (" | ".join(f"{x:.0%}" for x in _kept(_r[n])) if n != "base" and not n.startswith("negation")
               else "— | —") + " |" for n in _names)
        _merges = [n for n in _names if n not in ("base", "fine-tuned on shakespeare", "fine-tuned on astro")
                   and not n.startswith("negation")]
        _best = max(_merges, key=lambda n: min(_kept(_r[n])))
        _ties = max((n for n in _merges if n.startswith("TIES")), key=lambda n: min(_kept(_r[n])))
        _avg = "average (λ = 0.5 each)"
        _neg = next(n for n in _names if n.startswith("negation"))
        _s = _r["stats"]
        _ta7, _ta10 = _r["task arithmetic, λ = 0.7"], _r["task arithmetic, λ = 1.0"]
        _out = mo.md(f"""
    Validation loss (nats per token, lower is better). "Kept" = the share of each fine-tune's own improvement
    that the merged model keeps.

    | weights | Shakespeare | astro abstracts | detective (neither) | kept: Shakespeare | kept: astro |
    |---|---|---|---|---|---|
    {_rows}

    * **One model, both tasks.** The best merge, *{_best}*, keeps {_kept(_r[_best])[0]:.0%} and
      {_kept(_r[_best])[1]:.0%} of the two fine-tunes' gains, with no training after the fine-tunes, just
      adding weights. Plain averaging keeps {_kept(_r[_avg])[0]:.0%} and {_kept(_r[_avg])[1]:.0%}.
    * **Why it works here:** the two task vectors are almost orthogonal (cosine {_s['cosine']:+.3f}, over
      {_s['n_params'] / 1e6:.0f}M weights). Where both made one of their biggest changes, the signs agree only
      {1 - _s['sign_conflict_in_both_top20']:.0%} of the time, a coin flip, so the overlap is noise rather than
      a real disagreement.
    * **TIES {"helps" if min(_kept(_r[_ties])) > min(_kept(_r[_best])) else "doesn't help"} here**: its best
      setting keeps {_kept(_r[_ties])[0]:.0%} and {_kept(_r[_ties])[1]:.0%}. It was built for merging many
      tasks whose changes conflict. Two unrelated tasks barely interfere, so trimming 80% of each change mostly
      throws away useful signal. DARE, which drops 90% of each vector at random, does worse still.
    * **Merging has a cost elsewhere:** detective-novel loss goes from {_b['detective']:.3f} (base) to
      {_r[_best]['detective']:.3f} for the best merge. From λ = 0.7 to 1.0 (task arithmetic), astro goes
      {_ta7['astro']:.3f} → {_ta10['astro']:.3f}, Shakespeare {_ta7['shakespeare']:.3f} → {_ta10['shakespeare']:.3f},
      and the detective novels pay: {_ta7['detective']:.3f} → {_ta10['detective']:.3f}.
    * **Subtracting forgets:** base − Shakespeare vector raises Shakespeare loss by
      {_r[_neg]['shakespeare'] - _b['shakespeare']:+.2f}, astro by only
      {_r[_neg]['astro'] - _b['astro']:+.3f}, but detective prose by
      {_r[_neg]['detective'] - _b['detective']:+.2f}: what the Shakespeare vector learned was partly "older
      English prose", and the novels share some of it.

    Model merging is in real use: mergekit (Goddard et al.) is the standard tool, and merged models have
    topped open leaderboards since 2024. This experiment is small (two fine-tunes of
    {_s['steps']} steps), but the pattern, that orthogonal task vectors add, is the one those merges rely on.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Editing one fact

    Meng et al. (2022) asked where a model like GPT-2 stores "The Eiffel Tower is in Paris", by corrupting the
    subject and restoring one layer's activations at a time (**causal tracing**). The answer: in the **MLPs of
    early-to-middle layers, at the subject's last token**. An MLP's output matrix acts like a key-value memory:
    its input there (the **key** $k$, the hidden activation after the nonlinearity) says "we are reading about
    the Eiffel Tower", and $k^\top W$ is the **value** it writes into the residual stream for later layers to
    turn into "Paris".

    **ROME** rewrites that memory in three steps:

    1. **Key:** run "The Eiffel Tower" after a few different prefixes and average the MLP's hidden activation
       at "Tower".
    2. **Value:** find the output $v^*$ that makes the model say "Rome", by optimizing *one vector* added to
       the MLP's output there (60 Adam steps), with a penalty for changing what "The Eiffel Tower is a …"
       predicts.
    3. **Write it:** change $W$ so that $k^\top W' = v^*$, as little as possible. "Little" is measured on the
       keys the layer usually sees: $C$ is their covariance over ordinary text (detective novels here; ROME
       used Wikipedia), so keys unlike the Eiffel Tower's barely move.

    $$W' = W + \frac{(C^{-1}k)\,(v^* - k^\top W)}{(C^{-1}k)^\top k}.$$

    **Your turn:** `rank_one_edit(W, k, v, C_inv)`.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np):
    _k = jax.random.split(jax.random.key(0), 4)
    _W = jax.random.normal(_k[0], (64, 16))
    _key = jax.random.normal(_k[1], (64,))
    _v = jax.random.normal(_k[2], (16,))
    _A = jax.random.normal(_k[3], (64, 64))
    _C_inv = jnp.linalg.inv(_A @ _A.T / 64 + jnp.eye(64))
    _a = lj.attempt(ex.rank_one_edit, _W, _key, _v, _C_inv)
    if _a.ok:
        _err = float(jnp.abs(_key @ _a.value - _v).max())
        _rank = int(jnp.linalg.matrix_rank(_a.value - _W))
        _out = mo.md(f"**`rank_one_edit`** {'✅' if _err < 1e-3 and _rank == 1 else '❌'} on a random 64 × 16 "
                     f"layer: the key now produces the value (max error {_err:.1e}), and the change has rank "
                     f"{_rank}.")
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "rank-one edit of a weight matrix (ROME)", solves="rank_one_edit")
    return


@app.cell
def _(M, mo, np, plt):
    _r = M.read_json(M.RUNS / "edit.json")
    best = None
    if not _r:
        _out = mo.md("*Run `train.py --edit` (about 5 minutes) to fill this in.*")
    else:
        _L = sorted(_r["layers"], key=int)
        _rome = "ROME (C from text)"
        _p = [_r["layers"][l][_rome]["probe"] for l in _L]
        _fig, _ax = plt.subplots(figsize=(6, 2.8))
        _ax.plot([int(l) for l in _L], [p["the edited prompt"]["p_new"] for p in _p], "o-", label="edited prompt: Rome")
        _ax.plot([int(l) for l in _L], [p["paraphrase: stands in"]["p_new"] for p in _p], "s-",
                 label="paraphrase: Rome")
        _ax.plot([int(l) for l in _L], [p["neighbour: Louvre"]["p_old"] for p in _p], "^-",
                 label="Louvre: still Paris")
        _ax.set_xlabel("edited layer (of 32)")
        _ax.set_ylabel("probability")
        _ax.legend(fontsize=7)
        _ax.set_title("Which layer to edit", fontsize=9)
        _fig.tight_layout()
        best = max(_L, key=lambda l: _r["layers"][l][_rome]["probe"]["the edited prompt"]["p_new"])
        _lo, _hi = _L[0], _L[-1]
        _out = mo.vstack([_fig, mo.md(
            f"The edit takes best in early layers (layer {best}: "
            f"{_r['layers'][best][_rome]['probe']['the edited prompt']['p_new']:.0%} Rome) and fades by layer "
            f"{_hi} ({_r['layers'][_hi][_rome]['probe']['the edited prompt']['p_new']:.0%}), the pattern causal "
            "tracing predicts: the fact is looked up early, and late layers only pass it along.")])
    _out
    return (best,)


@app.cell
def _(M, best, mo):
    _r = M.read_json(M.RUNS / "edit.json")
    if not _r or best is None:
        _out = mo.md("")
    else:
        _e = _r["layers"][best]
        _rome, _id = _e["ROME (C from text)"], _e["plain rank-one (C = I)"]
        _base = _r["base"]
        _rows = "\n".join(
            f"| {n} | {_base['probe'][n]['top']!s} ({_base['probe'][n]['p_old']:.0%} Paris) | "
            f"{_rome['probe'][n]['top']!s} ({_rome['probe'][n]['p_new']:.0%} Rome, {_rome['probe'][n]['p_old']:.0%} Paris) | "
            f"{_id['probe'][n]['p_new']:.0%} Rome, {_id['probe'][n]['p_old']:.0%} Paris |"
            for n in _base["probe"])
        _dl = _rome["detective_loss"] - _base["detective_loss"]
        _cdiff = max(abs(_rome["probe"][n][k] - _id["probe"][n][k]) for n in _base["probe"] for k in ("p_new", "p_old"))
        _out = mo.md(f"""
    SmolLM2-360M, layer {best}'s MLP output matrix, changed by {_rome['rel_change']:.1%} (relative norm):

    | prompt | before (top word) | after ROME | after plain rank-one (C = I) |
    |---|---|---|---|
    {_rows}

    *Before:* "{_base['sample']}…"
    *After:* "{_rome['sample']}…"

    * **The edit holds where the wording is close:** the edited prompt and the "stands in the city of"
      paraphrase both say Rome{', and the model even continues with "Italy"' if "Italy" in _rome["sample"] else ""}.
    * **Looser paraphrases lose Paris but don't fully gain Rome** ("The Eiffel Tower is in": the top word becomes
      something generic). ROME's key is one averaged activation; a phrasing that reaches the subject differently
      reads a different key. MEMIT spreads the edit over several layers partly for this.
    * **Neighbours stay put:** the Louvre and the capital of France still say Paris, the Colosseum Rome. On
      ordinary text the loss moves by {_dl:+.4f} nats.
    * **The covariance {"made no visible difference" if _cdiff < 0.05 else "mattered"} in this small test:**
      across these eight prompts, ROME and plain rank-one (C = I) differ by at most {_cdiff:.0%} in any
      probability. {"$C$ matters when you check thousands of other facts (ROME's paper) or make many edits; "
      "eight prompts are too coarse a probe to see it." if _cdiff < 0.05 else "Measuring 'small' on real keys "
      "protects the facts the layer usually handles."}
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Weight-space programming: when an LLM writes the weights

    Every edit above was designed by a person who knew what each weight was for. In November 2026, Takehi, Teoh,
    Zhang and Khattab (MIT CSAIL) tried putting an LLM in that chair, calling it
    **weight-space programming**: the model gets another network's raw weights in a Python session, can run a
    limited number of forward passes, and edits the numbers. No gradients, no optimizer.

    **What it did** (Takehi et al., ["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/)):

    | task | result | what the edit actually was |
    |---|---|---|
    | GPT-6 Astra edits Qwen3.5-0.8B, 50 forward passes | MATH-500 7.9% → 48.1%, GSM8K 13.8% → 67.1% | found that the model got stuck repeating itself, traced it to one direction written by the MLPs of layers 15 and 19, and subtracted it: 131,072 weights (0.016%) |
    | same, with 10 or 25 passes | 12.3%, 14.1% | not enough looking to find the direction |
    | same protocol on Gemma 3 270M (which rarely loops) | 6.8% → 7.7% at best | nothing to fix |
    | a 9 × 9 Go network (1.48M weights) written from zeros | beats a same-size PPO network trained on 20M games, after 50 passes | a rule-based Go program compiled into weights |

    The authors call these "very early results" and say plainly that removing a loop "is not the same as
    making a model better at math": the base model knew the math and kept running out of tokens.

    **Toumbourou's replication** ([notesbylex.com](https://notesbylex.com/weight-space-programming-1990s),
    code under MIT) gave Claude Fable 5.2 *this module's* CNN and the same rules: write `build_weights()` with
    NumPy only, no fitting, at most 50 tries, see the training loss and confusion matrix after each.
    MNIST test accuracy was 88–91% over three runs, against Adam's 89–92% at 50 steps and ~98% at 5,000. Reading
    the weights it wrote, he found section 1 of this module: stroke and corner detectors and a log-odds readout
    "rebuilt from measured rates", in his words like a 1990s OCR researcher.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **A model editing itself.** Fan et al. ([arXiv 2611.21885](https://arxiv.org/abs/2611.21885), November 2026)
    gave Qwen3.5 its *own* weights, 20 forward passes and 200 code runs:

    * Only the 27B model improved itself; 2B made no edit, and 4B and 9B broke themselves with random noise.
    * Its best edit (+6.6 points on 490 held-out math problems, 57.2% → 63.8%) was one uniform change:
      raise the time-step bias of all 48 linear-attention layers (Module 41) by 0.5, so they forget faster.
      Fewer answers ran out of tokens.
    * Its other gains (+1.1 and +4.0) were Takehi's move: $W \leftarrow W - \alpha\, u u^\top W$, removing
      one direction $u$, the difference between the hidden states of finished and unfinished answers. That is
      a rank-one edit like section 4's, aimed at a behaviour instead of a fact, and the same projection
      Module 47 uses to steer.
    * Given a copy with one corrupted matrix, it clipped every weight over 15× the median: image benchmarks
      went from 7.3% back to 70.2% (the undamaged model: 84.3%).
    * In 5 of 8 sessions it looked at its own test results and withdrew the edit.

    **How to read all this.** The skeptical papers are worth reading alongside: Ahmed
    ([arXiv 2611.08070](https://arxiv.org/abs/2611.08070)) finds LLMs reading weights do accurate local
    arithmetic but lean on dataset guesses for meaning, and **WeightBench**
    ([arXiv 2611.18553](https://arxiv.org/abs/2611.18553)) scores frontier models at about 0.4–0.57 of 1 across
    reading, fixing and writing tasks; none could patch a language model or tell an impossible edit from a
    hard one. What these systems did is what this module did: find a direction, remove it; draw detectors;
    count; clip outliers. The novelty is who did it and how fast, not new kinds of edits. The weights were
    never magic; they were numbers a careful reader, human or not, could change on purpose.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"a": "It finds where the fact is stored by gradient descent over the whole model",
                 "b": "It writes a new output for one key in one MLP matrix, changing that matrix as little as "
                      "possible on the keys the layer usually sees",
                 "c": "It fine-tunes the model on the new fact with a small learning rate"},
        label="**Check yourself:** what does a ROME edit change?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("The only gradient steps are on one vector, v*, to find what to write. Where to "
                              "write came from causal tracing and a layer sweep."), kind="warn"),
        "b": mo.callout(mo.md("Right: one rank-one change to one matrix, sized by the keys' covariance."),
                        kind="success"),
        "c": mo.callout(mo.md("That is the alternative ROME was compared against: fine-tuning on one fact tends "
                              "to damage neighbours or not generalize. ROME edits one matrix in closed form."),
                        kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(46))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Lex Toumbourou, [Weight-Space Programming builds neural networks like a 1990s researcher](https://notesbylex.com/weight-space-programming-1990s)
      and [the experiments' code](https://github.com/lextoumbourou/weight-space-programming-experiments) —
      section 1's network, written by an LLM, read by a person.
    * Takehi, Teoh, Zhang, Khattab, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
      — the original weight-space programming experiments, with the edits explained.
    * Meng et al., [ROME project page](https://rome.baulab.info/) — causal tracing and the edit, with
      interactive figures.
    * DeepMind, [Tracr](https://github.com/google-deepmind/tracr) — the RASP compiler; its examples are
      section 2 at full size.
    * Charles Goddard et al., [mergekit](https://github.com/arcee-ai/mergekit) — the merging toolkit behind
      most merged open models (task arithmetic, TIES, DARE and more).

    **Next:** Module 47 (planned): reading activations, with lenses, sparse autoencoders and steering.
    """)
    return


if __name__ == "__main__":
    app.run()
