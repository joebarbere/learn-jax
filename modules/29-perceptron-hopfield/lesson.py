import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="29 · Perceptron to Hopfield")


@app.cell
def _():
    import marimo as mo
    import time

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import rosetta

    ex = lj.load_exercises(mo.notebook_dir())
    return ex, jax, jnp, lj, mo, np, plt, rosetta, time


@app.cell
def _(mo):
    mo.md(r"""
    # 29 · Perceptron to Hopfield networks (1958–1982)

    **Part 10, Milestones rebuilt in JAX.** Take Module 10 (MLPs) first.

    **Goal:** rebuild the two neural networks that bracket the field's first boom and bust, the way their
    inventors ran them, on your RX 7600 XT, and see how each one lives on inside the models you've
    already trained.

    **You will:** train Rosenblatt's perceptron with its original mistake-driven rule · watch it fail,
    forever, on XOR · fix XOR with one hidden layer (the 1986 answer) · store pictures in a Hopfield
    network and recall them from noise · measure its famous capacity limit (~0.14 patterns per neuron) ·
    show that one step of a *modern* Hopfield network is exactly the attention you built in Module 15.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from.** In 1943 Warren McCulloch and Walter Pitts described a neuron as a simple
    threshold unit. In 1958 Frank Rosenblatt, a psychologist at Cornell, gave such units a *learning rule*:
    the perceptron, built in hardware as the Mark I Perceptron. It learned to tell simple shapes apart, and
    the press promised thinking machines. In 1969 Marvin Minsky and Seymour Papert's book *Perceptrons*
    proved what a single layer cannot do (XOR is the classic example), and funding drifted to symbolic AI.
    In 1982 the physicist John Hopfield showed that a network of such units could act as a memory, with an
    energy that only goes down; he shared the 2024 Nobel Prize in Physics with Geoffrey Hinton.
    Timeline: Module 00b.
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The story: a neuron that learns (1943–1969)

    **1943.** McCulloch and Pitts model a neuron as a unit that sums weighted inputs and *fires* if the
    sum passes a threshold. Networks of such units can compute any logical function, but someone has to
    set the weights by hand.

    **1958.** Rosenblatt's perceptron learns its own weights from examples. The rule is almost
    embarrassingly simple: classify an example; if you got it wrong, nudge the weights toward the right
    answer; otherwise do nothing. The Mark I Perceptron did this in hardware, with a camera of 400
    photocells and motor-driven potentiometers as weights. Press coverage at the time promised machines
    that would soon walk, talk and think.

    **1962.** The *perceptron convergence theorem* (Novikoff): if two classes can be separated by a
    straight line (a hyperplane) with some margin $\gamma$, and the data fit in a ball of radius $R$, the
    rule makes at most $(R/\gamma)^2$ mistakes. Then it stops, having found a separating line.

    **1969.** Minsky and Papert's *Perceptrons* turned the theorem around: a single layer can *only*
    learn linearly separable problems. XOR (output 1 when exactly one input is 1) has no separating line.
    Multi-layer networks could do it, but nobody knew how to train them. Funding and attention moved to
    symbolic AI, and neural networks entered the first **AI winter**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Rosenblatt's rule, rebuilt

    Labels are $y \in \{+1, -1\}$. Predict $\operatorname{sign}(w \cdot x + b)$. On a mistake, i.e.
    $y\,(w \cdot x + b) \le 0$:

    $$w \leftarrow w + \eta\, y\, x \qquad b \leftarrow b + \eta\, y$$

    Note what's *not* here: no loss function, no derivative, no learning-rate schedule. This is not
    gradient descent (that came later, Modules 03 and 05). Each mistake rotates and shifts the boundary
    toward the misclassified point. Drag the slider to replay training one example at a time.
    """)
    return


@app.cell
def _(jax, jnp, np):
    def _blobs(seed=4, n=40, margin=0.25):
        """Separable by the line x + 0.6y = 0.3, with every point at least `margin` away from it."""
        _rng = np.random.default_rng(seed)
        _pts = []
        while len(_pts) < n:
            _p = _rng.uniform(-3, 3, 2)
            _d = (_p[0] + 0.6 * _p[1] - 0.3) / np.hypot(1, 0.6)
            if abs(_d) >= margin:
                _pts.append((_p, 1.0 if _d > 0 else -1.0))
        _X = np.array([p for p, _ in _pts], np.float32)
        _y = np.array([t for _, t in _pts], np.float32)
        return jnp.asarray(_X), jnp.asarray(_y)

    sep_X, sep_y = _blobs()

    def trace_perceptron(X, y, epochs):
        """Reference replay: (w, b, mistake) after every single example, for plotting."""
        def _step(carry, xy):
            _w, _b = carry
            _x, _t = xy
            _m = _t * (jnp.dot(_w, _x) + _b) <= 0
            _w = jnp.where(_m, _w + _t * _x, _w)
            _b = jnp.where(_m, _b + _t, _b)
            return (_w, _b), (_w, _b, _m)
        _Xs = jnp.tile(X, (epochs, 1))
        _ys = jnp.tile(y, epochs)
        _, (_W, _B, _M) = jax.lax.scan(_step, (jnp.zeros(2), jnp.float32(0.0)), (_Xs, _ys))
        return np.asarray(_W), np.asarray(_B), np.asarray(_M)

    sep_trace = trace_perceptron(sep_X, sep_y, 8)
    return sep_X, sep_trace, sep_y, trace_perceptron


@app.cell
def _(mo, sep_trace):
    replay = mo.ui.slider(1, len(sep_trace[0]), value=12, label="examples seen", show_value=True)
    replay
    return (replay,)


@app.cell
def _(np, plt, replay, sep_X, sep_trace, sep_y):
    def _boundary(ax, w, b, color="k", label=None):
        _xs = np.linspace(-3.5, 3.5, 2)
        if abs(w[1]) > 1e-6:
            ax.plot(_xs, -(w[0] * _xs + b) / w[1], color=color, lw=2, label=label)
        elif abs(w[0]) > 1e-6:
            ax.axvline(-b / w[0], color=color, lw=2, label=label)

    _k = replay.value - 1
    _W, _B, _M = sep_trace
    _fig, _ax = plt.subplots(figsize=(5.2, 4.2))
    _X, _y = np.asarray(sep_X), np.asarray(sep_y)
    _ax.scatter(*_X[_y > 0].T, c="tab:blue", label="+1")
    _ax.scatter(*_X[_y < 0].T, c="tab:orange", label="−1")
    _i = _k % len(_X)
    _ax.scatter(*_X[_i], s=220, facecolors="none", edgecolors="red" if _M[_k] else "green", lw=2,
                label="this example (red = mistake)")
    _boundary(_ax, _W[_k], _B[_k], label="boundary")
    _ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5),
            title=f"after {_k + 1} examples · {int(_M[: _k + 1].sum())} mistakes so far")
    _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, sep_trace):
    _M = sep_trace[2].reshape(8, -1).sum(1)
    mo.md(f"""
    Mistakes per pass over the 40 points: **{", ".join(str(int(m)) for m in _M)}**. Once a pass makes no
    mistakes, the rule never changes the weights again: it has converged, exactly as the theorem
    promises for separable data.
    """)
    return


@app.cell
def _(ex, lj, mo, sep_X, sep_y):
    _upd = lj.attempt(ex.perceptron_update, sep_X[0] * 0, 0.0, sep_X[0], sep_y[0])
    _tr = lj.attempt(ex.train_perceptron, sep_X, sep_y, 8)
    _items = [mo.md("## 3 · Your turn: the perceptron\n\n**`perceptron_update`** on a first example "
                    "(starting from zero weights, so it must be a mistake):"), _upd.view(mo)]
    _items.append(mo.md("**`train_perceptron(sep_X, sep_y, 8)`**, mistakes per epoch:"))
    if _tr.ok:
        _w, _b, _m = _tr.value
        _items.append(mo.md(f"`{[int(v) for v in _m]}` → final w = {[round(float(v), 2) for v in _w]}, "
                            f"b = {float(_b):.2f}. Matches the replay above if your loop visits the data in order."))
    else:
        _items.append(_tr.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "perceptron update", solves=("perceptron_update", "train_perceptron"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · XOR: the problem that froze the field

    Four points: $(0,0)\to -1$, $(0,1)\to +1$, $(1,0)\to +1$, $(1,1)\to -1$. No straight line puts the two
    $+1$ corners on one side and the two $-1$ corners on the other. The perceptron rule never notices that
    it's impossible; it just keeps making mistakes, cycling forever. AND, by contrast, is separable.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo, np, plt):
    xor_X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    xor_y = jnp.array([-1.0, 1.0, 1.0, -1.0])
    _and_y = jnp.array([-1.0, -1.0, -1.0, 1.0])
    _x = lj.attempt(ex.train_perceptron, xor_X, xor_y, 60)
    _a = lj.attempt(ex.train_perceptron, xor_X, _and_y, 60)
    _sx = lj.attempt(ex.is_linearly_separable, xor_X, xor_y)
    _sa = lj.attempt(ex.is_linearly_separable, xor_X, _and_y)
    if _x.ok and _a.ok:
        _fig, _ax = plt.subplots(figsize=(6.5, 2.8))
        _ax.plot(np.asarray(_a.value[2]), "o-", ms=3, label="AND (separable)")
        _ax.plot(np.asarray(_x.value[2]), "o-", ms=3, label="XOR (not separable)")
        _ax.set(xlabel="epoch", ylabel="mistakes per epoch", title="The perceptron on 4 points")
        _ax.legend()
        _fig.tight_layout()
        _view = mo.vstack([_fig, mo.md(
            f"AND stops making mistakes after **{int(np.argmax(np.asarray(_a.value[2]) == 0))}** epochs. "
            f"XOR still makes **{int(_x.value[2][-1])}** mistakes in epoch 60, and will for ever. "
            + (f"`is_linearly_separable`: AND → **{_sa.value}**, XOR → **{_sx.value}**." if _sa.ok and _sx.ok
               else ""))])
    else:
        _view = mo.vstack([_x.view(mo), _a.view(mo)])
    _view
    return xor_X, xor_y


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · The 1986 answer: one hidden layer, trained by backpropagation

    Minsky and Papert knew that *multi-layer* perceptrons could represent XOR. What was missing was a way
    to train the hidden units. Backpropagation (popularized by Rumelhart, Hinton and Williams in 1986, and
    built from scratch in Module 03) gives every weight a gradient. Two hidden units with a smooth
    nonlinearity are enough. This is the same fix Module 10 opened with, now in its historical place.
    """)
    return


@app.cell
def _(jax, jnp, np, plt, xor_X, xor_y):
    def _mlp(p, x):
        _h = jnp.tanh(x @ p["w1"] + p["b1"])
        return jnp.tanh(_h @ p["w2"] + p["b2"])[..., 0]

    def _loss(p):
        return jnp.mean((_mlp(p, xor_X) - xor_y) ** 2)

    _k1, _k2 = jax.random.split(jax.random.key(7))
    _p = {"w1": jax.random.normal(_k1, (2, 2)), "b1": jnp.zeros(2),
          "w2": jax.random.normal(_k2, (2, 1)), "b2": jnp.zeros(1)}
    _step = jax.jit(lambda p: jax.tree.map(lambda a, g: a - 0.5 * g, p, jax.grad(_loss)(p)))
    for _ in range(600):
        _p = _step(_p)
    _g = np.linspace(-0.5, 1.5, 120)
    _gx, _gy = np.meshgrid(_g, _g)
    _z = np.asarray(_mlp(_p, jnp.stack([_gx.ravel(), _gy.ravel()], 1))).reshape(_gx.shape)
    _fig, _ax = plt.subplots(figsize=(4.4, 3.8))
    _ax.contourf(_gx, _gy, _z, levels=[-1, 0, 1], colors=["#f5c99b", "#a9c8e8"], alpha=0.8)
    _Xn, _yn = np.asarray(xor_X), np.asarray(xor_y)
    _ax.scatter(*_Xn[_yn > 0].T, c="tab:blue", s=90, label="+1")
    _ax.scatter(*_Xn[_yn < 0].T, c="tab:orange", s=90, label="−1")
    _pred = np.sign(np.asarray(_mlp(_p, xor_X)))
    _ax.set(title=f"2-2-1 tanh net: {int((_pred == _yn).sum())}/4 correct, loss {float(_loss(_p)):.3f}")
    _ax.legend(loc="center", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · 1982: Hopfield's memory

    John Hopfield was a physicist. He noticed that a network of binary neurons with *symmetric*
    connections behaves like a magnet's spins: there is an **energy**

    $$E(s) = -\tfrac12\, s^\top W s,$$

    and if you update one neuron at a time, setting $s_i \leftarrow \operatorname{sign}(W_i \cdot s)$, the
    energy can only go down or stay the same. So the network rolls downhill and stops in a valley. Store
    memories by *making them the valleys*, with Hebb's rule ("neurons that fire together wire
    together"):

    $$W = \frac1N \sum_p x_p x_p^\top, \quad W_{ii} = 0.$$

    Start from a corrupted memory and the network rolls down into the clean one. Memory is a place you
    fall into, not an address you look up: **content-addressable memory**.
    """)
    return


@app.cell
def _(jnp, np):
    _GLYPHS = {
        "J": ["....######..", ".......##...", ".......##...", ".......##...", ".......##...", ".......##...",
              ".......##...", ".##....##...", ".##....##...", "..######....", "............", "............"],
        "A": ["....####....", "...##..##...", "..##....##..", "..##....##..", "..########..", "..##....##..",
              "..##....##..", "..##....##..", "..##....##..", "..##....##..", "............", "............"],
        "X": ["##........##", ".##......##.", "..##....##..", "...##..##...", "....####....", "....####....",
              "...##..##...", "..##....##..", ".##......##.", "##........##", "............", "............"],
        "+": ["............", ".....##.....", ".....##.....", ".....##.....", "############", "############",
              ".....##.....", ".....##.....", ".....##.....", ".....##.....", "............", "............"],
    }
    glyph_names = list(_GLYPHS)
    glyphs = jnp.asarray(np.array([[1.0 if c == "#" else -1.0 for c in "".join(_GLYPHS[k])] for k in glyph_names],
                                  np.float32))
    return glyph_names, glyphs


@app.cell
def _(mo):
    noise = mo.ui.slider(0, 40, value=25, step=5, label="corruption (% of pixels flipped)", show_value=True)
    noise
    return (noise,)


@app.cell
def _(ex, glyph_names, glyphs, jax, jnp, lj, mo, noise, np, plt):
    _W = lj.attempt(ex.hebbian_weights, glyphs)
    if not _W.ok:
        _view = _W.view(mo)
    else:
        _rng = np.random.default_rng(0)
        _n = glyphs.shape[1]
        _fig, _axes = plt.subplots(3, len(glyph_names), figsize=(1.6 * len(glyph_names), 5))
        _energies = []
        _ok = True
        for _j, _name in enumerate(glyph_names):
            _noisy = np.array(glyphs[_j])
            _flip = _rng.choice(_n, int(_n * noise.value / 100), replace=False)
            _noisy[_flip] *= -1
            _state = jnp.asarray(_noisy)
            _trace = []
            for _sweep in range(6):                       # 6 sweeps of N single-neuron updates
                _e = lj.attempt(ex.hopfield_energy, _W.value, _state)
                _r = lj.attempt(ex.hopfield_recall, _W.value, _state, _n, jax.random.fold_in(jax.random.key(1), _sweep))
                if not (_e.ok and _r.ok):
                    _ok = False
                    break
                _trace.append(float(_e.value))
                _state = _r.value
            if not _ok:
                break
            _energies.append(_trace)
            for _row, _img in enumerate([glyphs[_j], _noisy, _state]):
                _axes[_row, _j].imshow(np.asarray(_img).reshape(12, 12), cmap="gray_r")
                _axes[_row, _j].set_xticks([])
                _axes[_row, _j].set_yticks([])
            _axes[2, _j].set_xlabel("✓" if bool(jnp.all(_state == glyphs[_j])) else "✗", fontsize=14)
        if _ok:
            for _row, _lab in enumerate(["stored", f"{noise.value}% flipped", "recalled"]):
                _axes[_row, 0].set_ylabel(_lab)
            _fig.tight_layout()
            _e_lines = "; ".join(f"{_nm}: " + " → ".join(f"{_v:.0f}" for _v in _tr)
                                 for _nm, _tr in zip(glyph_names, _energies))
            _ov = np.asarray(glyphs @ glyphs.T) / _n
            _pairs = sorted(((abs(_ov[_a, _b]), glyph_names[_a], glyph_names[_b])
                             for _a in range(len(glyph_names)) for _b in range(_a + 1, len(glyph_names))),
                            reverse=True)
            _view = mo.vstack([_fig, mo.md(
                f"Energy after each sweep (it never rises): {_e_lines}.\n\n"
                f"When a glyph fails at higher noise, look at its neighbours: the most similar pair is "
                f"**{_pairs[0][1]}** and **{_pairs[0][2]}** (overlap {_pairs[0][0]:.2f}, where 0 means unrelated "
                "and 1 identical). Correlated memories share valleys, so a noisy start can roll into the wrong "
                "one or a blend. Hopfield's capacity math assumes random, uncorrelated patterns; real images "
                "rarely are.")])
        else:
            _view = mo.callout(mo.md("**Your turn:** implement `hopfield_energy` and `hopfield_recall` in "
                                     "`exercises.py` to run the recall demo."), kind="info")
    _view
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "Hopfield network: energy and update", solves=("hebbian_weights", "hopfield_energy", "hopfield_recall"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · How much can it remember?

    Every stored pattern adds noise ("crosstalk") to the others' valleys. Hopfield found, and physicists
    later derived, that random patterns are recalled reliably up to about **0.14 patterns per neuron**.
    Past that, recall collapses. Below we measure it: $N = 120$ neurons, $P$ random patterns, each probe
    starts from a stored pattern with 5% of bits flipped, and success means recovering it exactly. All the
    recalls for one $P$ run as a single [`vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap)ped, compiled loop on the GPU.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np, plt, time):
    _N, _probes = 120, 16
    _Ps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 34, 40]
    _rng = np.random.default_rng(3)
    _probe_W = lj.attempt(ex.hebbian_weights, jnp.ones((2, _N)))
    _probe_r = lj.attempt(ex.hopfield_recall, jnp.zeros((_N, _N)), jnp.ones(_N), 1, jax.random.key(0))
    if not (_probe_W.ok and _probe_r.ok):
        _view = mo.callout(mo.md("**Your turn:** `hebbian_weights` and `hopfield_recall` power this "
                                 "experiment."), kind="info")
        capacity_curve = None
    else:
        _recall_many = jax.jit(jax.vmap(lambda W, s, k: ex.hopfield_recall(W, s, 8 * _N, k), in_axes=(None, 0, 0)))
        _rates = []
        _t0 = time.perf_counter()
        for _P in _Ps:
            _pats = jnp.asarray(_rng.choice([-1.0, 1.0], (_P, _N)).astype(np.float32))
            _W = ex.hebbian_weights(_pats)
            _targets = _pats[jnp.arange(_probes) % _P]
            _flips = np.ones((_probes, _N), np.float32)
            for _i in range(_probes):
                _flips[_i, _rng.choice(_N, 6, replace=False)] = -1.0
            _keys = jax.random.split(jax.random.key(_P), _probes)
            _out = _recall_many(_W, _targets * _flips, _keys)
            _rates.append(float(jnp.mean(jnp.all(_out == _targets, axis=1))))
        _dt = time.perf_counter() - _t0
        capacity_curve = (np.array(_Ps) / _N, np.array(_rates))
        _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
        _ax.plot(*capacity_curve, "o-")
        _ax.axvline(0.138, color="gray", ls="--", label="theory ≈ 0.138 N")
        _ax.set(xlabel="patterns stored per neuron (P / N)", ylabel="exact recall rate",
                title=f"Hopfield capacity, N = {_N}")
        _ax.legend()
        _fig.tight_layout()
        _good = capacity_curve[0][capacity_curve[1] >= 0.9]
        _view = mo.vstack([_fig, mo.md(
            f"Recall stays at or above 90% up to **P/N ≈ {(_good.max() if len(_good) else 0):.2f}**, then falls "
            f"off. {len(_Ps) * _probes:,} recalls of {8 * _N} updates each took {_dt:.1f} s, including "
            "compilation. Finite N blurs the edge, but the cliff sits near the theory's 0.138.")])
    _view
    return (capacity_curve,)


@app.cell
def _(ex, glyphs, jax, jnp, lj, mo):
    _W = lj.attempt(ex.hebbian_weights, glyphs[:3])
    if _W.ok:
        _mix = jnp.sign(glyphs[0] + glyphs[1] + glyphs[2])
        _r = lj.attempt(ex.hopfield_recall, _W.value, _mix, 20 * glyphs.shape[1], jax.random.key(5))
        if _r.ok:
            _stays = float(jnp.mean(_r.value == _mix))
            _near = [float(jnp.mean(_r.value == glyphs[_i])) for _i in range(3)]
            _view = mo.md(f"""
            ### Spurious memories

            Hebb's rule also creates valleys you never stored. Take the pixel-wise majority vote of three
            stored glyphs, $\\operatorname{{sign}}(x_J + x_A + x_X)$, a *mixture* state. After 20 sweeps of
            recall it still agrees with itself on **{_stays:.0%}** of pixels, and matches each stored glyph
            on only {", ".join(f"{_v:.0%}" for _v in _near)}. The network happily "remembers" a blend it was
            never shown. Spurious states are why classical Hopfield networks are a toy today, and why the
            modern version below changed the energy.
            """)
        else:
            _view = _r.view(mo)
    else:
        _view = _W.view(mo)
    _view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · The modern view: Hopfield networks are attention

    In 2020 Hubert Ramsauer, Sepp Hochreiter and colleagues ("Hopfield Networks is All You Need") gave
    Hopfield networks continuous states and an exponential energy. Its update rule, one step from state
    $\xi$ with stored patterns as the rows of $X$, is

    $$\xi^{\text{new}} = \operatorname{softmax}(\beta\, \xi X^\top)\, X.$$

    Read that against Module 15: queries $\xi$, keys $X$, values $X$, scale $\beta$. **One modern
    Hopfield step is one attention head.** Storage capacity becomes exponential in the dimension, and one
    step usually lands on a stored pattern. The 1982 memory and the 2017 Transformer turned out to be the
    same idea.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _k1, _k2 = jax.random.split(jax.random.key(11))
    _X = jax.random.normal(_k1, (32, 64))
    _xi = jax.random.normal(_k2, (8, 64))
    _beta = 0.125
    _mh = lj.attempt(ex.modern_hopfield_update, _X, _xi, _beta)
    if _mh.ok:
        _attn = jax.nn.dot_product_attention(_xi[None, :, None, :], _X[None, :, None, :], _X[None, :, None, :],
                                             scale=_beta)[0, :, 0, :]
        _err = float(jnp.max(jnp.abs(_mh.value - _attn)))
        _noisy = _X[5] + 0.4 * jax.random.normal(jax.random.key(12), (64,))
        _back = ex.modern_hopfield_update(_X, _noisy, 2.0)
        _err_ret = float(jnp.max(jnp.abs(_back - _X[5])))
        _ret = ("lands exactly on the clean pattern (the softmax puts essentially all its weight on it)"
                if _err_ret < 1e-6 else f"lands within **{_err_ret:.1e}** of the clean pattern")
        _view = mo.md(f"""
        Your `modern_hopfield_update` against JAX's
        [`jax.nn.dot_product_attention`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html)
        on 8 queries over 32 stored 64-d patterns: max difference **{_err:.1e}**. Same computation.

        Retrieval: start from pattern 5 plus heavy noise (σ = 0.4 per coordinate); one update with
        β = 2 {_ret}. One step, no rolling downhill.
        """)
    else:
        _view = _mh.view(mo)
    _view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Epilogue: 2024

    The 2024 Nobel Prize in Physics went to John Hopfield and Geoffrey Hinton for foundational work that
    made machine learning with artificial neural networks possible: Hopfield for the associative memory
    you just built, Hinton for the Boltzmann machine that grew out of it and his later work on training
    deep networks ([NobelPrize.org, 2024 physics press release](https://www.nobelprize.org/prizes/physics/2024/press-release/)).
    The perceptron's mistake rule survives too: it is the ancestor of the hinge loss and of every linear
    layer you've trained. The line from 1958 to your GPU is unbroken.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "XOR's classes can't be split by any single straight line": "a",
            "The perceptron's learning rate was too small": "b",
            "Hopfield's energy function had no minimum": "c",
        },
        label="**Check yourself:** why can't a single perceptron learn XOR?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right. A single layer draws one line (hyperplane), and XOR needs two. "
                              "Hidden layers fix it (Section 5)."), kind="success"),
        "b": mo.callout(mo.md("No: the rule doesn't converge at *any* learning rate, because no separating "
                              "line exists."), kind="warn"),
        "c": mo.callout(mo.md("That's a different network. The perceptron has no energy function; the "
                              "issue is linear separability."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(29))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Wikipedia: [Perceptron](https://en.wikipedia.org/wiki/Perceptron) (history and the Mark I) and
      [*Perceptrons* (book)](https://en.wikipedia.org/wiki/Perceptrons_(book)).
    * Wikipedia: [AI winter](https://en.wikipedia.org/wiki/AI_winter).
    * Michael Nielsen, [Neural Networks and Deep Learning, ch. 1](http://neuralnetworksanddeeplearning.com/chap1.html)
      — perceptrons to sigmoid neurons.
    * Johannes Brandstetter et al. (JKU Linz), [Hopfield Networks is All You Need](https://ml-jku.github.io/hopfield-layers/)
      — the authors' blog post on modern Hopfield networks.
    * The Royal Swedish Academy of Sciences, [The Nobel Prize in Physics 2024, popular information](https://www.nobelprize.org/prizes/physics/2024/popular-information/).

    **Next:** Module 30 — word embeddings, and how word2vec (2013) taught networks what words mean.
    """)
    return


if __name__ == "__main__":
    app.run()
