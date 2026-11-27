import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="31b · Multi-dimensional RNNs")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import json
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import rosetta

    HERE = mo.notebook_dir()
    ex = lj.load_exercises(HERE)
    _spec = importlib.util.spec_from_file_location("m31b_models", HERE / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    RUNS = HERE.parent.parent / "runs" / "31b-mdrnn"
    return M, RUNS, ex, jax, jnp, json, lj, mo, np, plt, rosetta


@app.cell
def _(mo):
    mo.md(r"""
    # 31b · Multi-dimensional RNNs: one cell, four directions (2007–2024)

    **Take first:** Module 31 (RNN cells and [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan)) and Module 12 (convolution). This module sits
    between 31 and 32: what happened when the recurrent network of 31 was pointed at an image instead of a
    sentence.

    **Goal:** rebuild Graves, Fernández & Schmidhuber's multi-dimensional RNN (2007) the way it was done then,
    a recurrent cell swept across an image from each of its four corners; see why it lost to convolution on
    the clock even when it won on accuracy; and test an idea the original mostly didn't: **one set of
    weights for all four directions**. Then watch the idea come back in 2024 with a linear recurrence and a
    parallel scan.

    **You will:** write the 2-D sweep cell by cell · turn it into a *diagonal wavefront* (R + C − 1 steps
    instead of R × C) with a skew · sweep from four corners with separate or shared weights and *see* what
    each cell can see · train MDRNNs against an MLP and a CNN on Fashion-MNIST, at full size, with little data
    and with part of each image blacked out · build VMamba's cross-scan and time a parallel scan against a
    sequential one.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Networks with lateral and top-down connections are as old as the perceptron: Rosenblatt's *Principles of
    Neurodynamics* (1962) already described "cross-coupled" and "back-coupled" perceptrons. Fukushima's
    Neocognitron (1980) and LeCun's convolutional networks (1989) made one set of weights do the work at
    every *position*. Schuster & Paliwal's bidirectional RNN (1997) read a sequence in two directions. In
    2007 Alex Graves, Santiago Fernández and Jürgen Schmidhuber gave a recurrent cell one connection per
    dimension and swept an image from all four corners; MDLSTM went on to win handwriting competitions.
    By 2017 Puigcerver showed a CNN with a 1-D LSTM matched it at 6–7× less training time, and the line
    faded, until 2024's vision state-space models (Vim, VMamba) brought back multi-direction scans built on
    a linear recurrence that a GPU can run in parallel.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The story: a recurrent network that reads in two dimensions

    **An MLP is two-dimensional:** each layer is a vector, stacked along depth. Two ways to add space were
    tried. A **convolution** (Module 12) reuses one small set of weights at every *position*: Fukushima's
    Neocognitron (1980) copied each learned weight to every cell of its layer, and LeCun's 1989 network had
    64,660 connections but only 9,760 free parameters. A **recurrence over the grid** instead lets
    information *flow* across it, the way Module 31's RNN carries a state along a sentence.

    **2007: the multi-dimensional RNN.** Graves, Fernández & Schmidhuber replaced the RNN's single recurrent
    connection with "as many recurrent connections as there are dimensions in the data". In 2-D, cell
    $(i, j)$ reads its input and the hidden states of the cell above and the cell to its left:

    $$h_{i,j} = \tanh\big(x_{i,j} W_x + h_{i-1,j} W_u + h_{i,j-1} W_l + b\big).$$

    One such sweep only sees what lies above and to the left. So they ran $2^d$ sweeps, one from each corner
    (four in 2-D), with **a separate hidden layer per corner**, all feeding one output layer. On MNIST it was
    slightly worse than a CNN on clean digits (1.1 % vs 0.9 % error) but far better on elastically *warped*
    ones (6.8 % vs 11.3 %). With LSTM cells (MDLSTM) and CTC it read handwriting from raw pixels: it beat
    every entry of the ICDAR 2007 Arabic competition (91.4 % vs 87.2 %), and MDLSTM and LSTM systems from
    the same group won three ICDAR 2009 competitions.

    **2015–2017: making it fit a GPU, and losing anyway.** Within one sweep, a cell must wait for its two
    neighbours, so only the cells on one anti-diagonal can run at once. Voigtlaender, Doetsch & Ney (2016)
    exploited exactly that and cut an IAM handwriting epoch from about three days on a CPU to about 15
    minutes on a GPU. Then Puigcerver (2017) replaced the 2-D LSTM layers with convolutions and 1-D LSTMs:
    error rates equal or better, **3.8 vs 24.3 minutes per epoch**, though with 9.3M parameters against
    2.6M. Cheaper parameters, more expensive time: the trade this module measures.

    **Almost everyone untied the directions.** MDRNN, ReNet (Visin et al. 2015, sweeps instead of
    convolutions), PixelRNN (van den Oord et al. 2016) and the 2-D LSTM scene labeller all gave each
    direction its own weights. The exception is Grid LSTM (Kalchbrenner, Danihelka & Graves 2015), which
    tied weights across dimensions, depth included, and found the tied version better (on a memorisation
    task, >99 % with 0.55M parameters against 67 % untied with 5M). Whether **one set of weights swept in
    four directions** is a good image model is a question you can answer on your own GPU in this module.
    """)
    return


@app.cell
def _(M, mo):
    data = M.load_data()
    mo.md(f"""
    Data: Fashion-MNIST (Module 10), **{len(data[0]):,}** training and **{len(data[2]):,}** test images, each
    28×28 pixels cut into a **14×14 grid of 2×2 patches**, so each cell has 4 features. Fewer cells keep the
    sweeps short (14 + 14 − 1 = 27 wavefront steps).
    """)
    return (data,)


@app.cell
def _(M, data, ex, jax, jnp, lj, mo):
    _p = M.init_sweep(jax.random.key(0), 4, 8)
    _x = jnp.asarray(data[0][:2, :5, :6])
    _a = lj.attempt(ex.mdrnn_sweep_naive, _p, _x)
    mo.vstack([mo.md(r"""
    ## 2 · One sweep, cell by cell

    **Your turn:** `mdrnn_sweep_naive`: the recurrence above, read literally. Visit row by row, left to
    right, so $(i-1, j)$ and $(i, j-1)$ are always done before $(i, j)$; a neighbour outside the grid is
    zeros. It's slow on purpose: $R \times C$ steps, each waiting for the last.
    """), _a.view(mo) if not _a.ok else mo.md(f"Your sweep on a 5×6 corner of two images: output shape "
                                               f"`{tuple(_a.value.shape)}` (B, R, C, H).")])
    return


@app.cell
def _(M, data, ex, jax, jnp, lj, mo, np, plt):
    def _field():
        p = M.init_sweep(jax.random.key(1), 4, 16)
        x = jnp.asarray(data[0][:1])

        def cell(x):
            return ex.mdrnn_sweep_naive(p, x)[0, 7, 7].sum()

        g = np.abs(np.asarray(jax.grad(cell)(x)[0])).sum(-1)
        fig, ax = plt.subplots(figsize=(3.2, 3))
        ax.imshow(g > 1e-9, cmap="Blues")
        ax.plot(7, 7, "r*", ms=12)
        ax.set_title("inputs cell (7, 7) depends on", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig, int((g > 1e-9).sum())

    _a = lj.attempt(_field)
    if _a.ok:
        _fig, _n = _a.value
        _out = mo.hstack([_fig, mo.md(f"""
    Take the gradient of one cell's state, (7, 7) (red star), with respect to every input cell. It is nonzero
    for **{_n}** of the 196 cells: exactly the rectangle above and to the left, (0..7) × (0..7) = 64.
    Information only flows down and right, so one sweep sees a quadrant. That is why the 2007 model sweeps
    from all four corners.
    """)], justify="start")
    else:
        _out = mo.md("*Finish `mdrnn_sweep_naive` to see which inputs one cell depends on.*")
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "2-D RNN sweep (MDRNN)", solves=("mdrnn_sweep_naive", "mdrnn_sweep"))
    return


@app.cell
def _(ex, jnp, lj, mo, np):
    _x = (jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4, 1) + 1)
    _a = lj.attempt(ex.skew, _x)
    _b = lj.attempt(lambda: ex.unskew(ex.skew(_x), 4))
    if _a.ok:
        _s = np.asarray(_a.value[0, :, :, 0]).astype(int)
        _grid = "\n    ".join("| " + " | ".join(str(v) if v else "·" for v in r) + " |" for r in _s)
        _orig = "\n    ".join("| " + " | ".join(str(v) for v in r) + " |" for r in np.asarray(_x[0, :, :, 0]).astype(int))
        _show = mo.md(f"""
    A 3×4 grid (cells numbered row by row) and its skew. Each skewed **column** is one anti-diagonal of the
    original, and every cell's up and left neighbours are in the column just before it:

    | | | | |
    |---|---|---|---|
    {_orig}

    | | | | | | |
    |---|---|---|---|---|---|
    {_grid}

    `unskew` brings it back: {"✅ round trip exact" if _b.ok and bool(jnp.array_equal(_b.value, _x)) else "*(your turn)*"}.
    """)
    else:
        _show = _a.view(mo)
    mo.vstack([mo.md(r"""
    ## 3 · The diagonal wavefront

    Cell $(i, j)$ needs $(i-1, j)$ and $(i, j-1)$, so all cells with the same $i + j$, one **anti-diagonal**,
    are independent of each other: compute them together and a 14×14 grid takes 27 steps, not 196. (A
    "wavefront" here is this order of computation, not Module 23's GPU wavefront of 32 threads.)

    The trick that makes it a plain `lax.scan`: **skew** the grid, shifting row $i$ right by $i$ cells. Then
    an anti-diagonal is a column, and the scan walks the columns. PixelRNN's Diagonal BiLSTM (2016) used the
    same skew. **Your turn:** `skew` and `unskew`.
    """), _show])
    return


@app.cell
def _(M, data, ex, jax, jnp, lj, mo, np):
    def _check():
        p = M.init_sweep(jax.random.key(2), 4, 32)
        x = jnp.asarray(data[0][:256])
        diff = float(jnp.abs(ex.mdrnn_sweep(p, x[:8]) - M.sweep_cellwise(p, x[:8])).max())
        t_cell = M.time_forward_backward(M.sweep_cellwise, (p, x))
        t_wave = M.time_forward_backward(ex.mdrnn_sweep, (p, x))
        return diff, t_cell, t_wave

    _a = lj.attempt(_check)
    if _a.ok:
        _d, _tc, _tw = _a.value
        _out = mo.md(f"""
    Your wavefront agrees with a cell-by-cell sweep to **{_d:.1e}** (float32 rounding). One forward and
    backward pass over a batch of 256 images, H = 32, on the GPU:

    | order | sequential steps | time |
    |---|---|---|
    | cell by cell (a scan of rows of scans of cells) | 196 | {1000 * _tc:.1f} ms |
    | diagonal wavefront (your `mdrnn_sweep`) | 27 | **{1000 * _tw:.1f} ms** |

    {_tc / _tw:.1f}× faster, for {196 / 27:.1f}× fewer steps: each step now does 14 cells' work in one matmul,
    which the GPU barely notices. That is Voigtlaender's 2016 speedup in miniature, and also its limit: a
    convolution layer is *one* step, however large the image.
    """)
    else:
        _out = _a.view(mo)
    mo.vstack([mo.md(r"""
    **Your turn:** `mdrnn_sweep`: project all inputs at once, skew, `lax.scan` over the skewed columns,
    mask the padding (or $\tanh(b)$ leaks in as a fake neighbour), unskew.
    """), _out])
    return


@app.cell
def _(M, data, ex, jax, jnp, lj, mo, np, plt):
    def _fields():
        x = jnp.asarray(data[0][:1])
        ps = [M.init_sweep(jax.random.key(10 + k), 4, 16) for k in range(4)]

        def cell(x, k):
            z = ex.four_direction_sweeps(ps, x, ex.mdrnn_sweep)[0, 7, 7]
            return z[16 * k: 16 * (k + 1)].sum() if k < 4 else z.sum()

        fig, axs = plt.subplots(1, 5, figsize=(9, 2.1))
        names = ["from top-left", "from bottom-left", "from top-right", "from bottom-right", "all four"]
        for k, ax in enumerate(axs):
            g = np.abs(np.asarray(jax.grad(cell)(x, k)[0])).sum(-1)
            ax.imshow(g > 1e-9, cmap="Blues", vmin=0, vmax=1)
            ax.plot(7, 7, "r*", ms=8)
            ax.set_title(names[k], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return fig

    _a = lj.attempt(_fields)
    mo.vstack([mo.md(r"""
    ## 4 · Four corners, and one set of weights

    **Your turn:** `four_direction_sweeps`: flip the grid so each corner becomes the top-left, sweep, flip
    back, and stack the four results. Pass four parameter sets for Graves' MDRNN, or one for **the same
    weights in all four directions**. Below, what cell (7, 7) can see through each direction, and through
    all four together: every cell sees the whole image.
    """), _a.value if _a.ok else _a.view(mo)])
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "flip an axis")
    return


@app.cell
def _(M, mo):
    _rows = "\n    ".join(f"| `{k}` | {d} |" for k, (d, _h) in M.KINDS.items())
    mo.md(r"""
    ## 5 · Training them

    Six models, all small (6k–16k parameters), all ending in global average pooling and a linear layer, all
    trained with Adam at 3e-3, batch 256:

    | name | model |
    |---|---|
    """ + _rows + r"""

    The MDRNNs run your `four_direction_sweeps` and `mdrnn_sweep`; `linscan` is §7's, built on your
    `cross_scan`. Below, **2 epochs each, one seed** (~1 minute), for the shape of things; `train.py` runs
    the full comparison (3 seeds, 5 epochs, plus the small-data runs) and §6 reads it.
    """)
    return


@app.cell
def _(M, data, ex, lj, mo):
    fns = {"sweep": ex.mdrnn_sweep, "four_direction_sweeps": ex.four_direction_sweeps,
           "cross_scan": ex.cross_scan, "cross_merge": ex.cross_merge}

    def _run():
        out = {}
        for k in M.KINDS:
            out[k] = M.train(k, fns, data, epochs=2, seed=0)[1]
        return out

    _a = lj.attempt(_run)
    live = _a.value if _a.ok else None
    if _a.ok:
        _r = "\n    ".join(f"| `{k}` | {v['params']:,} | {v['acc']:.1%} | {v['acc_occluded_4']:.1%} | {v['s_per_epoch']:.2f} s |"
                           for k, v in live.items())
        _out = mo.md(r"""
    | model | parameters | test accuracy (2 epochs) | with an 8×8-pixel square blacked out | time / epoch |
    |---|---|---|---|---|
    """ + _r)
    else:
        _out = mo.vstack([mo.md("*Finish §2–4 and §7's exercises to train the models.*"), _a.view(mo)])
    _out
    return fns, live


@app.cell
def _(RUNS, json, mo, np, plt):
    _p = RUNS / "results.json"
    long_run = json.loads(_p.read_text()) if _p.exists() else None
    if long_run is None:
        _out = mo.callout(mo.md("The full comparison isn't here yet: run `containers/run.sh python "
                                "modules/31b-mdrnn/train.py` (~20 min; 3 seeds × 6 models × 3 training-set "
                                "sizes), then re-run this cell."), kind="info")
    else:
        _runs = long_run["runs"]
        _kinds = list(dict.fromkeys(r["kind"] for r in _runs))
        _sizes = sorted({r["n_train"] for r in _runs})

        def _stat(k, n, f="acc"):
            v = [r[f] for r in _runs if r["kind"] == k and r["n_train"] == n]
            return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0, len(v), np.mean(
                [r["params"] for r in _runs if r["kind"] == k]))

        _rows = []
        for k in _kinds:
            cells = [f"{100 * m:.1f} ± {100 * s:.1f}" for m, s, _, _ in (_stat(k, n) for n in _sizes)]
            occ = _stat(k, max(_sizes), "acc_occluded_4")
            _rows.append(f"| `{k}` | {int(_stat(k, max(_sizes))[3]):,} | " + " | ".join(cells)
                         + f" | {100 * occ[0]:.1f} ± {100 * occ[1]:.1f} |")
        _fig, _ax = plt.subplots(figsize=(5.6, 2.8))
        for k in _kinds:
            _m = [_stat(k, n)[0] * 100 for n in _sizes]
            _s = [_stat(k, n)[1] * 100 for n in _sizes]
            _ax.errorbar(_sizes, _m, yerr=_s, marker="o", ms=3, capsize=2, label=k)
        _ax.set_xscale("log")
        _ax.set_xlabel("training examples")
        _ax.set_ylabel("test accuracy (%)")
        _ax.legend(fontsize=7)
        _fig.tight_layout()
        def _findings(st, sizes):
            lo, mid, hi = sizes[0], sizes[len(sizes) // 2], sizes[-1]

            def a(k, n, f="acc"):
                return 100 * st(k, n, f)[0]

            return f"""
    What it says, in this run:

    * **One set of weights for all four directions holds up, per parameter.** With half the parameters,
      `mdrnn_tied` scores {a('mdrnn_tied', lo):.1f} % against `mdrnn`'s {a('mdrnn', lo):.1f} % on {lo:,} examples
      and {a('mdrnn_tied', hi):.1f} vs {a('mdrnn', hi):.1f} % on {hi:,}. Spending the saved parameters on a wider
      hidden state (`mdrnn_tied_wide`, the same count as four separate sets) gives {a('mdrnn_tied_wide', lo):.1f},
      {a('mdrnn_tied_wide', mid):.1f} and {a('mdrnn_tied_wide', hi):.1f} % against {a('mdrnn', lo):.1f},
      {a('mdrnn', mid):.1f} and {a('mdrnn', hi):.1f} %: at least as good everywhere. Compare the ± columns
      before reading more into one-point gaps.
    * **Against the CNN** of about the same size, the MDRNNs are ahead from {mid:,} examples up
      ({a('mdrnn', hi):.1f} vs {a('cnn', hi):.1f} % at {hi:,}) and lose less to the blacked-out square
      ({a('mdrnn', hi, 'acc_occluded_4'):.1f} vs {a('cnn', hi, 'acc_occluded_4'):.1f} %): every cell sees the whole
      image, so one missing patch matters less.
    * **But the MLP beats all of them**, even with {lo:,} examples and under occlusion
      ({a('mlp', lo):.1f} % and {a('mlp', hi, 'acc_occluded_4'):.1f} %). Read that as a lesson about the experiment,
      not about spatial models: every model here except the MLP ends in **global average pooling**, which throws
      away *where* each feature was, and Fashion-MNIST's items are centred, so position is very informative. The
      MLP keeps position for free. A fairer comparison pools less (or flattens the grid) and uses images where
      position isn't given away: an exercise worth running.
    * **The linear cross-scan** is the smallest and weakest here and the most hurt by occlusion: one layer of a
      linear recurrence is a weak feature extractor. VMamba stacks dozens, with convolutions and MLPs between.
    """

        _t = long_run["timing"]
        _out = mo.vstack([mo.md(r"""
    ## 6 · Where shared weights should help: little data, damaged images

    The research on weight sharing says its real advantage is not accuracy on a big clean dataset but
    **learning from less data** and **robustness**. `train.py` tests both: every model trained on 60,000,
    5,000 and 1,000 examples (each seeing 300,000 examples in total, so small-data runs aren't just shorter),
    3 seeds each, and every test image also scored with a random 8×8-pixel square blacked out. Mean ± standard
    deviation over seeds:

    | model | parameters | """ + " | ".join(f"{n:,} examples" for n in _sizes) + r""" | occluded (60k) |
    |---|---|""" + "---|" * len(_sizes) + r"""---|
    """ + "\n    ".join(_rows)), _fig, mo.md(_findings(_stat, _sizes))])
    _out
    return (long_run,)


@app.cell
def _(ex, jnp, lj, mo):
    _x = jnp.arange(6, dtype=jnp.float32).reshape(1, 2, 3, 1)
    _a = lj.attempt(ex.cross_scan, _x)
    _b = lj.attempt(lambda: ex.cross_merge(ex.cross_scan(_x), 2, 3))
    if _a.ok:
        _orders = "\n    ".join(f"| {n} | {' '.join(str(int(v)) for v in _a.value[k, 0, :, 0])} |"
                               for k, n in enumerate(["row-major", "column-major", "row-major, reversed",
                                                      "column-major, reversed"]))
        _show = mo.md(r"""
    The 2×3 grid `0 1 2 / 3 4 5` as four sequences:

    | order | sequence |
    |---|---|
    """ + _orders + f"""

    `cross_merge` puts them back and sums: {"✅ 4 × the grid" if _b.ok and bool(jnp.array_equal(_b.value, 4 * _x)) else "*(your turn)*"}.
    """)
    else:
        _show = _a.view(mo)
    mo.vstack([mo.md(r"""
    ## 7 · Then and now: the same directions, a linear recurrence

    Why didn't MDLSTM just get a faster kernel? Its gates depend *nonlinearly* on the previous state, so each
    step must wait for the last. A **linear** recurrence $h_t = a_t \odot h_{t-1} + b_t$ is different: any
    stretch of steps composes into one step of the same form, so the whole sequence can be computed by a
    **parallel scan** in $O(\log T)$ rounds (Martin & Cundy 2018; Module 36 builds it with
    [`jax.lax.associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html#jax.lax.associative_scan)). That is what Mamba (Module 36) and the Gated DeltaNet of Module 41 run on.

    In 2024 VMamba (Liu et al.) brought MDRNN's four directions back on that footing: its **cross-scan**
    reads the image as four 1-D sequences, row-major, column-major and both reversed, runs a selective
    state-space scan along each, and **cross-merges** them back onto the grid. Vision-LSTM (ViL, Alkin et
    al. 2024) measured how many directions pay: ImageNet accuracy 72.2 % with one, 73.7 % with two, 73.8 %
    with four, 73.5 % with eight, while four roughly doubled its runtime. **Your turn:** `cross_scan` and
    `cross_merge`.
    """), _show])
    return


@app.cell
def _(M, data, fns, jax, jnp, lj, mo):
    def _time():
        p = M.init("linscan", jax.random.key(0))
        x = jnp.asarray(data[0][:256])
        par = M.time_forward_backward(lambda p, x: M.forward("linscan", p, x, fns, True), (p, x))
        seq = M.time_forward_backward(lambda p, x: M.forward("linscan", p, x, fns, False), (p, x))
        diff = float(jnp.abs(M.forward("linscan", p, x[:16], fns, True) - M.forward("linscan", p, x[:16], fns, False)).max())
        return par, seq, diff

    _a = lj.attempt(_time)
    if _a.ok:
        _par, _seq, _d = _a.value
        _out = mo.md(f"""
    `linscan` (§5's sixth model): your cross-scan, a gated linear recurrence along each of the four 196-step
    orders, your cross-merge. Forward and backward over 256 images, the same recurrence computed two ways
    (they agree to {_d:.1e}):

    | computed as | rounds | time |
    |---|---|---|
    | `lax.scan`, one step at a time | 196 | {1000 * _seq:.1f} ms |
    | `lax.associative_scan` (parallel) | ~2 log₂ 196 ≈ 15 | **{1000 * _par:.1f} ms** |

    {_seq / _par:.1f}× faster. The parallel scan does more arithmetic in total, but in far fewer dependent
    rounds; on a GPU, rounds are what cost.
    """)
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Then and now

    | | 2007 (MDRNN / MDLSTM) | 2024 (VMamba, ViL) |
    |---|---|---|
    | directions | 2^d sweeps, one from each corner | 2–4 1-D orders (cross-scan) |
    | weights per direction | separate | separate |
    | recurrence | nonlinear (tanh, LSTM gates) | linear in the state, input-dependent gates |
    | parallelism | one anti-diagonal at a time (R + C − 1 steps) | parallel scan over each order |
    | won on | warped digits, handwriting, segmentation | high-resolution efficiency; accuracy ≈ ViT |
    | lost on | wall-clock: 6–7× slower per epoch than CNN + 1-D LSTM | classification: MambaOut drops the scan and still wins |

    The question this module adds, one set of weights for all directions, is the cheap half of the idea:
    it halves or quarters the parameters of a sweep at no cost in time. Whether it costs accuracy is what
    §6 measures.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"Each step of its sweep depends nonlinearly on the previous one, so a GPU can only compute one anti-diagonal at a time": "a",
                 "It needs more parameters than a CNN": "b",
                 "It can only see the part of the image above and to the left of each pixel": "c"},
        label="**Check yourself:** why did MDLSTM lose to CNN + 1-D LSTM by 2017?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {"a": mo.callout(mo.md("Right: R + C − 1 dependent steps per sweep against one for a convolution layer. "
                                 "A *linear* recurrence escapes this with a parallel scan (§7)."), kind="success"),
           "b": mo.callout(mo.md("The opposite: Puigcerver's 2-D LSTM had 2.6M parameters against 9.3M for the CNN "
                                 "+ 1-D LSTM. It lost on time, not parameters."), kind="warn"),
           "c": mo.callout(mo.md("One sweep does (§2), but four sweeps from the four corners see everything (§4)."),
                           kind="warn")}
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(51))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    * Graves, Fernández & Schmidhuber, [Multi-Dimensional Recurrent Neural Networks](https://arxiv.org/abs/0705.2011)
      (2007): the model, in 10 pages.
    * Voigtlaender, Doetsch & Ney, [Handwriting Recognition with Large Multidimensional Long Short-Term Memory
      Recurrent Neural Networks](https://vision.rwth-aachen.de/media/papers/MDLSTM_final.pdf) (2016): the
      GPU wavefront implementation.
    * Puigcerver, [Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text
      Recognition?](http://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf) (2017): the measured case against.
    * Alkin et al., [Vision-LSTM: xLSTM as Generic Vision Backbone](https://arxiv.org/abs/2406.04303) (2024):
      the direction ablation in §7.
    * Martin & Cundy, [Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057)
      (2018): why linear recurrences escape the sequential trap.

    **Next:** Module 32 — BERT vs GPT, back on the main road to the Transformer.
    """)
    return


if __name__ == "__main__":
    app.run()
