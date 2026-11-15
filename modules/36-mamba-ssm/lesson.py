import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="36 · State-space models and Mamba")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax.data import tiny_shakespeare

    ex = lj.load_exercises(mo.notebook_dir())
    _spec = importlib.util.spec_from_file_location("m36_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    return M, ex, jax, jnp, lj, mo, np, plt, tiny_shakespeare


@app.cell
def _(mo):
    mo.md(r"""
    # 36 · State-space models and Mamba (2021–2023)

    **Take first:** Module 08 ([`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan)) and Module 31 (RNNs). Module 18 helps for the last section.

    **Goal:** see where a 2023 language-model architecture came from — 1960s control theory — and rebuild
    it in JAX: a linear state-space model run three different ways, then Mamba's *selective* version,
    trained on the same Shakespeare as Modules 14, 16 and 31.

    **You will:** discretize a continuous system with a zero-order hold · run it as a recurrence
    (`lax.scan`), as a convolution (FFT) and as a parallel prefix scan
    ([`jax.lax.associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html#jax.lax.associative_scan))
    and prove all three agree · time them against attention on your GPU · write Mamba's selective scan ·
    train a small Mamba-style language model · watch selectivity solve a copying task an ordinary SSM
    can't · decode with a constant-size state instead of a growing KV cache.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    State-space models are older than deep learning: Rudolf Kálmán's 1960 filter tracked a system's hidden
    state $x' = Ax + Bu$ from noisy measurements $y = Cx$, and the same equations run control loops to
    this day. In 2020–21 Albert Gu and colleagues (HiPPO, then S4 with Karan Goel and Christopher Ré)
    showed that a carefully initialized linear SSM layer could model very long sequences, trained in
    parallel as a convolution. Orvieto et al. (2023) found much of the benefit in simple linear
    recurrences. Then Gu and Tri Dao's Mamba (December 2023) made the SSM's parameters depend on the input
    — *selectivity* — and computed it with a hardware-aware parallel scan; Mamba-2 (2024) tied SSMs back
    to attention. Many recent models mix SSM layers with attention.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The story: from control loops to language models

    **1960: the state-space view.** Control engineers describe a system — a motor, a circuit, a rocket —
    by a hidden **state** $h(t)$ that evolves under a linear rule and is observed through an output:

    $$h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t) + D\,x(t).$$

    If you've tuned a PID loop or written a transfer function, this is the same object: the state is the
    capacitor voltages and inductor currents, $A$ how they decay and couple, $B$ how the input drives
    them, $C$ what you measure. Kálmán's filter (1960) estimated $h$ from noisy $y$.

    **2020–2021: the same equations as a neural layer.** Treat each feature channel of a sequence as an
    input signal $x(t)$, give it an $N$-dimensional state, and *learn* $A, B, C$. Gu et al.'s **HiPPO**
    (2020) worked out an $A$ whose state optimally compresses the history of the input; **S4** (2021)
    made it fast and trainable. Because the system is **linear and time-invariant** (LTI), the whole
    output is a convolution of the input with a kernel — so training parallelizes like a CNN, while
    generation runs like an RNN with a fixed-size state. S4 set records on long-range benchmarks.

    **2023: selectivity.** An LTI system treats every input the same way; it can't decide to *remember
    this token and ignore that one*. **Mamba** (Gu & Dao) made the step size $\Delta$ and the matrices
    $B, C$ functions of the current input. That breaks the convolution trick — but a linear recurrence
    can still be computed in parallel with a **prefix scan** (Blelloch, 1990), and Mamba's GPU kernel did
    exactly that. **Mamba-2** (2024) showed a precise duality between these SSMs and a form of attention.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Discretization: from $h'$ to $h_t$

    A sequence model sees samples, not continuous time. Holding the input constant over each step of
    length $\Delta$ (a **zero-order hold** — exactly what a DAC does) gives an exact discrete system:

    $$h_t = \bar A\, h_{t-1} + \bar B\, x_t, \qquad y_t = C\, h_t, \qquad
      \bar A = e^{\Delta A}, \quad \bar B = (\bar A - 1)\,A^{-1} B.$$

    With a **diagonal** $A$ (what S4D, S5 and Mamba use) everything is elementwise: each of the $N$ state
    dimensions is a first-order low-pass filter with its own time constant $-1/A_n$. Slide $\Delta$:
    small steps make the state change slowly (long memory); large steps make it forget fast.
    """)
    return


@app.cell
def _(mo):
    dt_slider = mo.ui.slider(steps=[0.01, 0.03, 0.1, 0.3, 1.0], value=0.1, label="step size Δ", show_value=True)
    dt_slider
    return (dt_slider,)


@app.cell
def _(dt_slider, ex, jnp, lj, mo, np, plt):
    _A = -jnp.array([0.2, 1.0, 5.0])
    _B = jnp.ones(3)
    _r = lj.attempt(ex.discretize_zoh, _A, _B, dt_slider.value)
    if _r.ok:
        _Ab, _Bb = _r.value
        _fig, _ax = plt.subplots(figsize=(7, 2.8))
        _t = np.arange(40)
        for _i, _a in enumerate(np.asarray(_A)):
            _ax.plot(_t * dt_slider.value, np.asarray(_Bb[_i] * _Ab[_i] ** _t), "o-", ms=3,
                     label=f"A = {_a:g}:  Ā = {float(_Ab[_i]):.3f}")
            _tt = np.linspace(0, 40 * dt_slider.value, 200)
            _ax.plot(_tt, dt_slider.value * np.exp(_a * _tt), color="gray", lw=0.6)
        _ax.set(xlabel="time", ylabel="impulse response of each state", title="discrete (dots) vs continuous (lines)")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _out = _fig
    else:
        _out = _r.view(mo)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · One system, three ways to run it

    Unroll the recurrence from $h_{-1} = 0$:
    $y_t = \sum_{k=0}^{t} \underbrace{C \bar A^k \bar B}_{K_k}\, x_{t-k}$. So an LTI SSM is three things
    at once:

    1. **A recurrence** — step through time with `lax.scan`. $O(L)$ work, but $L$ *sequential* steps.
    2. **A convolution** with kernel $K = (C\bar B, C\bar A\bar B, C\bar A^2\bar B, \dots)$ — compute $K$
       once, then convolve with the FFT in $O(L \log L)$, fully parallel. This is how S4 trained.
    3. **A prefix scan.** Each step is the affine map $h \mapsto a_t h + b_t$. Composing two of them gives
       another affine map, $(a_1, b_1) \circ (a_2, b_2) = (a_1 a_2,\; a_2 b_1 + b_2)$, and that operation
       is **associative** — which is all
       [`jax.lax.associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html#jax.lax.associative_scan)
       needs to compute every prefix in $O(\log L)$ parallel depth (Blelloch's algorithm). Unlike the
       convolution, this still works when $a_t, b_t$ change at every step — Mamba's case.

    **Your turn:** `ssm_scan`, `ssm_kernel`, `ssm_conv` and `linear_recurrence_assoc` in `exercises.py`.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    def _three_ways():
        A = -jnp.exp(jax.random.normal(jax.random.key(0), (16,)))
        B = jax.random.normal(jax.random.key(1), (16,))
        C = jax.random.normal(jax.random.key(2), (16,))
        A_bar, B_bar = ex.discretize_zoh(A, B, 0.05)
        xs = jax.random.normal(jax.random.key(3), (2048,))
        y_scan = ex.ssm_scan(A_bar, B_bar, C, xs)
        y_conv = ex.ssm_conv(ex.ssm_kernel(A_bar, B_bar, C, 2048), xs)
        h = ex.linear_recurrence_assoc(jnp.broadcast_to(A_bar, (2048, 16)), B_bar[None] * xs[:, None])
        y_assoc = h @ C
        scale = float(jnp.max(jnp.abs(y_scan)))
        return {"conv": float(jnp.max(jnp.abs(y_conv - y_scan))), "assoc": float(jnp.max(jnp.abs(y_assoc - y_scan))),
                "scale": scale}

    three = lj.attempt(_three_ways)
    if three.ok:
        _v = three.value
        _out = mo.md(f"""
    A 16-state SSM on a 2,048-step random input (outputs up to ±{_v['scale']:.2f}):

    | method | max \\|y − y_scan\\| |
    |---|---|
    | FFT convolution | {_v['conv']:.1e} |
    | associative scan | {_v['assoc']:.1e} |

    Three algorithms, one function. The FFT's error is float32 round-off in a length-4,096 transform; the
    scan-based answers agree almost exactly because they do the same multiply-adds in a different order.
    """)
    else:
        _out = three.view(mo)
    _out
    return (three,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Which is fastest on your GPU?

    Below, a 256-channel, 16-state SSM run each way at growing sequence lengths, next to one causal
    attention head (64 dims) for reference. *Scratch* is the compiler's planned temporary memory.
    """)
    return


@app.cell
def _(M, ex, mo, three):
    if three.ok:
        bench = M.bench_lengths(ex)

        def _fmt(cell):
            ms, mib = cell
            if ms is None:
                return "—"
            return f"{ms:.2f} ms" + (f" · {mib:.0f} MiB" if mib is not None else "")

        _rows = "\n    ".join(f"| {r['L']:,} | {_fmt(r['scan'])} | {_fmt(r['assoc'])} | {_fmt(r['fft'])} | {_fmt(r['attention'])} |"
                          for r in bench)
        _out = mo.md(f"""
    | length L | `lax.scan` (recurrent) | `associative_scan` | FFT convolution | attention (1 head) |
    |---|---|---|---|---|
    {_rows}

    What to look for:

    * **`lax.scan` grows linearly but pays per step** — each of the L steps is a tiny kernel launch, so on a
      GPU it's limited by overhead, not arithmetic (Module 10's lesson again).
    * **`associative_scan` does about twice the arithmetic but in log L rounds**, so it wins once L is long.
      It also stores the (L, D, N) intermediates, which is why its scratch memory grows with L·D·N.
    * **The FFT convolution** is fast but only works for LTI systems, and building the kernel costs memory.
      Compile time is a cost too: the first version of this benchmark built the kernel as `A_bar ** k` from
      closed-over arrays, and XLA spent **over three minutes** compiling it (constant folding, plus a slow
      ROCm code path for that power-inside-a-sum). Passing the arrays as arguments and writing
      $\\bar A^k = e^{{k \\log \\bar A}}$ brought it to about a second; see `bench_lengths` in `models.py`.
    * **Attention's cost and memory grow with L²** — at 16,384 tokens one head's score matrix alone would be
      1 GiB, so it's skipped there (as are the sequential scan and the FFT, to keep this cell quick; the
      sequential scan took about 2.5 s at 16,384 when this was written). That quadratic is the reason SSMs exist.

    Mamba's real kernel fuses the whole scan into one GPU kernel that keeps the state in on-chip memory
    and never writes the (L, D, N) tensor out — the same move as FlashAttention (Module 23). JAX's
    `associative_scan` is the portable version; it doesn't do that fusion.
    """)
    else:
        bench = None
        _out = mo.md("*Finish Section 3's exercises to run the benchmark.*")
    _out
    return (bench,)


@app.cell
def _(bench, np, plt):
    if bench:
        _fig, _ax = plt.subplots(figsize=(7, 3.2))
        for _name, _style in [("scan", "o-"), ("assoc", "s-"), ("fft", "^-"), ("attention", "x--")]:
            _pts = [(r["L"], r[_name][0]) for r in bench if r[_name][0] is not None]
            if _pts:
                _ax.plot(*np.array(_pts).T, _style, label=_name)
        _ax.set(xscale="log", yscale="log", xlabel="sequence length L", ylabel="ms per forward pass",
                title="time vs length (log–log)")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _plot = _fig
    else:
        _plot = None
    _plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Selectivity: Mamba's one big idea

    In an LTI SSM, $\Delta$, $B$ and $C$ are the same at every step, so every token is filtered the same
    way. Mamba computes them **from the input**:

    $$\Delta_t = \text{softplus}(x_t W_\Delta + b_\Delta), \qquad B_t = x_t W_B, \qquad C_t = x_t W_C.$$

    A large $\Delta_t$ means "reset the state and take in this token"; a tiny one means "ignore this token,
    keep remembering". $B_t$ chooses *what* to write into the state, $C_t$ *what* to read out. It's a gate,
    like an LSTM's (Module 31) — but linear in the state, so it still parallelizes with a prefix scan.

    **Your turn:** `selective_scan` — the reference version, with `lax.scan`, one sequence at a time.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    def _check_selective():
        L, D, N = 64, 8, 16
        ks = jax.random.split(jax.random.key(4), 5)
        x = jax.random.normal(ks[0], (L, D))
        delta = jax.nn.softplus(jax.random.normal(ks[1], (L, D)))
        A = -jnp.exp(jax.random.normal(ks[2], (D, N)))
        B = jax.random.normal(ks[3], (L, N))
        C = jax.random.normal(ks[4], (L, N))
        ref = ex.selective_scan(x, delta, A, B, C)
        a = jnp.exp(delta[..., None] * A)
        b = delta[..., None] * B[:, None, :] * x[..., None]
        par = jnp.einsum("ldn,ln->ld", ex.linear_recurrence_assoc(a, b), C)
        return float(jnp.max(jnp.abs(ref - par)))

    _r = lj.attempt(_check_selective)
    (mo.md(f"Your `selective_scan` vs the same computation with your `linear_recurrence_assoc`: "
           f"max difference **{_r.value:.1e}**. The parallel scan handles input-dependent steps fine.")
     if _r.ok else _r.view(mo))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · A small Mamba, trained on Shakespeare

    `models.py` assembles a Mamba-style block the way the paper does: project the input to two streams,
    run one through a short **causal depthwise convolution** and SiLU, compute $\Delta, B, C$ from it, run
    the **selective scan** (your `linear_recurrence_assoc`), add a skip term $D \cdot u$, gate with the
    other stream, and project back — with RMSNorm and residuals around it. Three blocks, character-level,
    1,200 steps.
    """)
    return


@app.cell
def _(tiny_shakespeare, np):
    _text = tiny_shakespeare()
    chars = sorted(set(_text))
    stoi = {c: i for i, c in enumerate(chars)}
    _data = np.array([stoi[c] for c in _text], np.int32)
    _n = int(0.9 * len(_data))
    train_ids, val_ids = _data[:_n], _data[_n:]
    return chars, stoi, train_ids, val_ids


@app.cell
def _(M, chars, ex, mo, three, train_ids, val_ids):
    if three.ok:
        lm = M.train_lm(ex.linear_recurrence_assoc, len(chars), train_ids, val_ids, steps=1200)
        _out = mo.md(f"""
    Trained **{lm['n_params']:,} parameters** for 1,200 steps in **{lm['seconds']:.0f} s**.

    | model (character level, Shakespeare) | val loss (nats/char) |
    |---|---|
    | Module 14 MLP, fixed 8-char window *(measured when written)* | 1.82 |
    | Module 31 RNN / GRU / LSTM, ~100K recurrent params *(measured when written)* | 1.64–1.71 |
    | **this Mamba-style model** | **{lm['val']:.3f}** |
    | Module 16 GPT, ~0.8M params *(measured when written)* | ≈1.54 |

    These are short runs with different budgets and context lengths, so read the ordering, not the
    decimals. The point: a linear recurrence with input-dependent gates is a real sequence model, in the
    same league as the recurrent and attention models you've built.
    """)
    else:
        lm = None
        _out = mo.md("*Finish the exercises in Sections 3 and 5 to train the model.*")
    _out
    return (lm,)


@app.cell
def _(M, chars, jax, lm, mo, stoi):
    if lm:
        _ids = M.generate(lm["params"], [stoi[c] for c in "ROMEO:\n"], 240, jax.random.key(0))
        _sample = mo.md("**Sample** (recurrent decoding, temperature 0.8):\n\n```\nROMEO:\n" + "".join(chars[i] for i in _ids) + "\n```")
    else:
        _sample = None
    _sample
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Where selectivity matters: selective copying

    Here's a task from the Mamba paper, simplified: a sequence of 128 positions holds 8 coloured tokens at
    *random* places among noise; then come 8 markers, and at each marker the model must say the colours
    in order. An LTI SSM's kernel is fixed — it can only weight positions by *how far back* they are — so
    it can't pick out tokens whose positions change every time. A selective SSM can open its gate when it
    sees a colour and hold still on noise. Same architecture both times; only selectivity differs.
    """)
    return


@app.cell
def _(M, ex, mo, three):
    if three.ok:
        copy_lti = M.train_copy(ex.linear_recurrence_assoc, selective=False)
        copy_sel = M.train_copy(ex.linear_recurrence_assoc, selective=True)
        _out = mo.md(f"""
    | model | colour accuracy on 2,048 fresh sequences | train time |
    |---|---|---|
    | LTI SSM (fixed Δ, B, C) | {copy_lti['test_acc']:.1%} | {copy_lti['seconds']:.0f} s |
    | **selective SSM** (Δ, B, C from the input) | **{copy_sel['test_acc']:.1%}** | {copy_sel['seconds']:.0f} s |

    Chance is 12.5% (8 colours).
    """)
    else:
        copy_lti = copy_sel = None
        _out = mo.md("*Finish the exercises to run the comparison.*")
    _out
    return copy_lti, copy_sel


@app.cell
def _(copy_lti, copy_sel, np, plt):
    if copy_lti and copy_sel:
        _fig, _ax = plt.subplots(figsize=(6.5, 2.8))
        _x = np.arange(1, len(copy_lti["curve"]) + 1) * 100  # one point per 100 steps
        _ax.plot(_x, copy_lti["curve"], "o-", label="LTI SSM")
        _ax.plot(_x, copy_sel["curve"], "s-", label="selective SSM")
        _ax.axhline(1 / 8, color="gray", ls=":", lw=1)
        _ax.set(xlabel="training step", ylabel="train accuracy", ylim=(0, 1.02), title="selective copying")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _p = _fig
    else:
        _p = None
    _p
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Generating with a constant-size state

    A Transformer's KV cache grows with every token (Module 18). An SSM carries a **fixed** state — per
    layer, the $E \times N$ scan state plus the last $K-1$ convolution inputs — no matter how long the
    text gets. The same trained weights run either way: the parallel scan for training, one step at a time
    for generation.
    """)
    return


@app.cell
def _(M, ex, jnp, lm, mo, val_ids):
    if lm:
        _p = lm["params"]
        _toks = jnp.asarray(val_ids[:64])[None]
        _par = M.forward(_p, _toks, ex.linear_recurrence_assoc)[0, -1]
        _st = M.init_state(_p)
        for _t in _toks[0]:
            _lg, _st = M.step(_p, _st, _t[None])
        _diff = float(jnp.max(jnp.abs(_lg[0] - _par)))
        _sf = M.state_floats(_p)
        _kv = lambda L: 2 * 32 * 320 * L                     # SmolLM2-360M: 32 layers, 5 KV heads × 64 dims
        _rows = "\n    ".join(f"| {L:,} | {_sf:,} floats ({_sf * 4 / 2**10:.0f} KiB) | {_kv(L):,} floats ({_kv(L) * 2 / 2**20:.0f} MiB in bf16) |"
                          for L in (1_000, 8_000, 64_000))
        _out = mo.md(f"""
    Recurrent decoding (64 tokens fed one at a time) vs the parallel forward pass on the same tokens:
    max logit difference **{_diff:.1e}** — the same model, two algorithms.

    | context length | this model's state (constant) | SmolLM2-360M's KV cache (grows) |
    |---|---|---|
    {_rows}

    The comparison isn't size-matched (SmolLM2 is ~1,000× bigger), but the *shape* is the point: one
    column is flat, the other grows linearly. That's why SSM and hybrid models are attractive for long
    contexts — and why they must compress: a fixed state can't remember everything verbatim, which is
    exactly the trade-off selectivity manages.
    """)
    else:
        _out = mo.md("*Train the model in Section 6 first.*")
    _out
    return


@app.cell
def _(M, ex, mo, three):
    if not three.ok:
        _out = mo.md("*(Section 9 trains with your `linear_recurrence_assoc`; implement it in Section 3 first.)*")
    else:
        _runs = {_name: M.train_mqar(ex.linear_recurrence_assoc, _attn, n_pairs=16, steps=600)
                 for _name, _attn in [("Mamba × 4", ()), ("hybrid: Mamba, Mamba, **attention**, Mamba", (2,))]}
        _rows = "\n".join(f"| {_k} | {_r['n_params']:,} | **{_r['test_acc']:.0%}** | {_r['seconds']:.0f} s |"
                          for _k, _r in _runs.items())
        _out = mo.md(f"""
## 9 · Hybrids: a little attention goes a long way (Jamba)

A fixed-size state has to *compress*, and some jobs don't compress. **Multi-query associative recall**
(Arora et al., 2023, "Zoology") is the cleanest example: show 16 key→value pairs, then ask for the keys
again in a new order; the model must answer each with its value. Attention solves it by looking back at
the pair. An SSM must have squeezed all 16 pairs into its state as they went by.

Two models of the same width, 4 layers each, trained for 600 steps; in the hybrid, one of the four layers is
causal attention (Module 15's), with **no positional encoding**, because the Mamba layers around it already
know where each token is (the Jamba paper found the same):

| model | parameters | recall accuracy (1,024 fresh sequences; chance 1.6%) | train time |
|---|---|---|---|
{_rows}

One attention layer turns a task the SSM barely starts on into a solved one, with *fewer* parameters. When
this was written, with 1,000 steps, the gap held at 32 and 64 pairs too (Mamba 8% and 6%, the hybrid 100% at both). Pure
Mamba isn't hopeless here: Zoology shows a recurrent model's state must grow with the number of pairs it has
to hold, and with more width and training it climbs. Attention does it with one layer at any size.

**Jamba** (AI21 Labs, 2024) is this idea at scale: blocks of one attention layer per seven Mamba layers, with
mixture-of-experts feed-forwards (Module 17) in some of them. The payoff is memory. Only the attention layers
keep a KV cache, so at 1:7 the cache is **one eighth** of a same-depth transformer's (Section 8's table,
divided by eight), while the attention layers supply the exact look-up that recall needs. The big open
models are still mostly pure transformers (Llama, Qwen, DeepSeek), but a growing family is hybrid: Samba,
Zamba, NVIDIA's Nemotron-H and IBM's Granite 4.0 all put SSM layers in the bulk of the stack and keep a few
attention layers for recall.
""")
    _out
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "It makes Δ, B and C depend on the input, so the model can choose what to remember": "a",
            "It replaces the recurrence with attention": "b",
            "It uses the FFT convolution instead of a scan": "c",
        },
        label="**Check yourself:** what did Mamba change relative to S4?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right — selectivity. It also means the convolution trick no longer applies, so "
                              "Mamba computes the recurrence with a parallel scan instead."), kind="success"),
        "b": mo.callout(mo.md("No — Mamba has no attention at all. Mamba-2 later showed a mathematical duality "
                              "with a form of attention, and hybrids mix the two, but Mamba itself is an SSM."),
                        kind="warn"),
        "c": mo.callout(mo.md("The opposite: S4 used the FFT convolution; input-dependent parameters make the "
                              "kernel change at every step, so Mamba can't."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(36))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Sasha Rush & Sidd Karamcheti, [The Annotated S4](https://srush.github.io/annotated-s4/) — S4 line by
      line, in JAX.
    * Maarten Grootendorst, [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
    * Albert Gu & Tri Dao, [state-spaces/mamba](https://github.com/state-spaces/mamba) — the reference
      implementation and its fused scan kernel.
    * JAX docs: [`jax.lax.associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html)

    **Next:** Module 37 — interpretability: what's going on inside the models you've trained.
    """)
    return


if __name__ == "__main__":
    app.run()
