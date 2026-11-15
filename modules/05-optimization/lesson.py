import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="05 · Optimization")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    return ex, jax, jnp, lj, mo, np, optax, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 05 · Optimization: gradient descent to Adam

    **Goal:** understand exactly what happens on each line of a training loop's update, and build the
    optimizer that trained your deepseek-v3-jax model from parts you write yourself.

    **You will:** see a loss as a landscape · watch gradient descent converge, oscillate and diverge as you
    move one slider · learn why long narrow valleys are slow · derive momentum, RMSProp and Adam and race
    them · see why minibatch gradients are noisy · plot the exact learning-rate schedule from
    deepseek-v3-jax · clip gradients · understand AdamW's weight decay · map all of it onto `optax`.

    Module 03 gave you gradients. This module is about what to *do* with them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · A loss is a landscape

    Training means: choose parameters $\theta$ that make a loss $L(\theta)$ small. Picture $L$ as terrain
    and $\theta$ as your position. The gradient $\nabla L$ points **uphill**, steepest first, so the
    simplest possible strategy is to step a little bit downhill, over and over:

    $$\theta \leftarrow \theta - \eta\,\nabla L(\theta)$$

    $\eta$ (eta) is the **learning rate**: how big a step to take. Real models have millions of
    parameters, so you can't draw their landscape — but 1-D and 2-D toys behave the same way, and every
    idea in this module is visible in them. Left: a bumpy 1-D loss with its gradient as arrows (they point
    *downhill* here, i.e. $-\nabla L$). Right: a 2-D bowl seen from above as contour lines.
    """)
    return


@app.cell
def _(jax, jnp, np, plt):
    def _f1(x):
        return x**4 - 3 * x**2 + x

    _xs = jnp.linspace(-2.2, 2.2, 200)
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.6))
    _a.plot(_xs, _f1(_xs), color="k")
    _pts = jnp.linspace(-2, 2, 11)
    _g = jax.vmap(jax.grad(_f1))(_pts)
    _a.quiver(_pts, _f1(_pts), -_g / (1 + np.abs(_g)), 0 * _g, angles="xy", scale_units="xy", scale=2,
              color="tab:red", width=0.006)
    _a.set(title="1-D loss: two valleys, arrows = −gradient", xlabel="θ", ylabel="L(θ)")
    _X, _Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    _b.contour(_X, _Y, 0.5 * _X**2 + 2 * _Y**2, levels=15, cmap="viridis")
    _b.set(title="2-D loss 0.5·x² + 2·y², from above", xlabel="θ₁", ylabel="θ₂", aspect="equal")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Notice on the left: two valleys. Gradient descent goes downhill from wherever it starts, so it finds
    *a* minimum, not necessarily *the* minimum. For big neural networks this turns out to matter less
    than you'd fear — there are many roughly-equally-good valleys — but it's why initialization and random
    seeds change results a little.

    ## 2 · The learning rate, one slider

    Take the simplest bowl, $L(x)=x^2$, so $\nabla L = 2x$ and one step is
    $x \leftarrow x - \eta\cdot 2x = (1-2\eta)\,x$. Everything depends on the factor $(1-2\eta)$:

    * $0<\eta<0.5$: the factor is between 0 and 1 — **smooth convergence** (slow if $\eta$ is tiny).
    * $0.5<\eta<1$: the factor is negative — you **overshoot** and **oscillate** across the valley, but still shrink.
    * $\eta>1$: $|1-2\eta|>1$ — every step lands farther away: **divergence**. In a real run this is the
      loss shooting to `inf`/`nan`.
    """)
    return


@app.cell
def _(mo):
    lr_gd = mo.ui.slider(steps=[0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 1.02], value=0.1, label="learning rate η",
                         show_value=True)
    lr_gd
    return (lr_gd,)


@app.cell
def _(lr_gd, np, plt):
    _eta = lr_gd.value
    _x = [2.0]
    for _ in range(15):
        _x.append(_x[-1] - _eta * 2 * _x[-1])
    _x = np.array(_x)
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.4))
    _grid = np.linspace(-3, 3, 200)
    _a.plot(_grid, _grid**2, color="k", lw=1)
    _a.plot(_x, _x**2, "o-", color="tab:orange", ms=4)
    _a.set(xlim=(-3, 3), ylim=(-0.3, 9), title=f"path on L(x) = x², η = {_eta}", xlabel="x")
    _b.semilogy(np.arange(len(_x)), _x**2 + 1e-12, "o-", ms=3)
    _b.set(title="loss per step (log scale)", xlabel="step")
    _regime = ("diverges" if _eta > 1 else "oscillates but converges" if _eta > 0.5
               else "converges in one step" if _eta == 0.5 else "converges smoothly")
    _fig.suptitle(f"factor (1 − 2η) = {1 - 2 * _eta:+.2f} → {_regime}")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Why long narrow valleys are slow

    Real losses curve much more steeply in some directions than others. Take
    $L(x,y)=\tfrac12(x^2+\kappa\,y^2)$: curvature 1 along $x$, curvature $\kappa$ along $y$. The ratio
    $\kappa$ is the **condition number**.

    The learning rate must keep the *steep* direction stable ($\eta<2/\kappa$, by Section 2's argument),
    but then the *shallow* direction moves by only a factor $(1-\eta)\approx(1-2/\kappa)$ per step. Large
    $\kappa$ → tiny progress along the valley floor, while zig-zagging across it. Move the slider: the step
    count grows roughly in proportion to $\kappa$.

    This is the single most important picture in optimization. Every trick that follows — momentum,
    Adam, normalization layers (Module 11) — is, at heart, a way to fight a bad condition number.
    """)
    return


@app.cell
def _(mo):
    kappa = mo.ui.slider(steps=[1, 5, 20, 50, 100], value=20, label="condition number κ", show_value=True)
    kappa
    return (kappa,)


@app.cell
def _(kappa, np, plt):
    _k = kappa.value
    _eta = 1.9 / _k  # just under the stability limit 2/κ
    _p = np.array([-3.0, 1.5])
    _path = [_p.copy()]
    _steps = None
    for _i in range(2000):
        _p = _p - _eta * np.array([_p[0], _k * _p[1]])
        _path.append(_p.copy())
        if _steps is None and 0.5 * (_p[0] ** 2 + _k * _p[1] ** 2) < 1e-3:
            _steps = _i + 1
    _path = np.array(_path[:80])
    _X, _Y = np.meshgrid(np.linspace(-3.5, 3.5, 120), np.linspace(-2, 2, 120))
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.contour(_X, _Y, 0.5 * (_X**2 + _k * _Y**2), levels=20, cmap="viridis", linewidths=0.8)
    _ax.plot(_path[:, 0], _path[:, 1], "o-", color="tab:red", ms=2.5, lw=1)
    _ax.set(title=f"κ = {_k}, η = 1.9/κ = {_eta:.3f}: {_steps} steps to reach loss < 1e-3", aspect="equal")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Momentum, RMSProp, Adam

    **Momentum (heavy ball).** Keep a running sum of past gradients — a *velocity* — and step along that:

    $$v \leftarrow g + \beta v,\qquad \theta \leftarrow \theta - \eta\,v \qquad(\beta\approx0.9)$$

    Across the narrow valley the gradient flips sign every step, so those components **cancel** in $v$.
    Along the valley floor they all point the same way, so they **add up** — effectively a learning rate
    ~$1/(1-\beta)=10\times$ bigger in the direction that was too slow. A ball rolling downhill.

    **RMSProp.** Give every parameter its own step size: keep a running average of the *squared*
    gradient, $s \leftarrow \beta_2 s + (1-\beta_2)g^2$, and divide by its root:
    $\theta \leftarrow \theta - \eta\, g/(\sqrt{s}+\epsilon)$. Steep directions (big $g$) get damped,
    shallow ones get boosted — a cheap, per-parameter fix for the condition number.

    **Adam = momentum + RMSProp**, with one more detail. Both running averages start at zero:

    $$m \leftarrow \beta_1 m + (1-\beta_1)g,\qquad v \leftarrow \beta_2 v + (1-\beta_2)g^2$$

    so for the first steps they're biased toward 0 (after one step $m = 0.1\,g$, not $g$). Adam divides
    that bias out — $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$ — and steps by
    $\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$. A side effect worth remembering: the first step has size
    ≈ $\eta$ in every coordinate *regardless of the gradient's scale*. Adam's step size is set by $\eta$,
    not by how big your gradients happen to be.
    """)
    return


@app.cell
def _(np, plt):
    _t = np.arange(1, 61)
    _b1, _b2 = 0.9, 0.999
    _m = 1 - _b1**_t   # m_t when every gradient is exactly 1
    _v = 1 - _b2**_t
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_t, _m, label="m (β₁=0.9), raw")
    _ax.plot(_t, _v, label="v (β₂=0.999), raw")
    _ax.plot(_t, _m / (1 - _b1**_t), "--", label="either one, bias-corrected")
    _ax.axhline(1, color="gray", lw=0.6)
    _ax.set(title="Running averages when the true gradient is always 1", xlabel="step t", ylim=(0, 1.1))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    `v` with $\beta_2=0.999$ needs *thousands* of steps to warm up without correction — Adam would take
    huge steps early because it divides by a too-small $\sqrt v$. Bias correction fixes that exactly.

    ### The race: Rosenbrock's banana

    $L(x,y)=(1-x)^2+100\,(y-x^2)^2$ is the classic hard case: a curved, very narrow valley with the minimum
    at $(1,1)$. All four optimizers start at $(-1.5,\,2)$ and get 1,000 steps, each with a learning rate
    tuned to be roughly as large as it can stably be. (These run as one compiled [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan) each —
    Module 08 explains `scan`.)
    """)
    return


@app.cell
def _(jax, jnp, optax):
    def rosen(p):
        return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2

    def run_optax(opt, loss, x0, steps):
        """Run an optax optimizer on `loss` from x0; return the path of parameters, shape (steps+1, dim)."""
        p0 = jnp.asarray(x0)

        def body(carry, _):
            p, s = carry
            upd, s = opt.update(jax.grad(loss)(p), s, p)
            p = optax.apply_updates(p, upd)
            return (p, s), p

        _, path = jax.lax.scan(body, (p0, opt.init(p0)), None, length=steps)
        return jnp.concatenate([p0[None], path])

    return rosen, run_optax


@app.cell
def _(mo, np, optax, plt, rosen, run_optax):
    _opts = {
        "gradient descent (η=1e-3)": optax.sgd(1e-3),
        "momentum (η=1e-3, β=0.9)": optax.sgd(1e-3, momentum=0.9),
        "RMSProp (η=1e-2)": optax.rmsprop(1e-2),
        "Adam (η=5e-2)": optax.adam(5e-2),
    }
    _X, _Y = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-1, 3, 200))
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4))
    _a.contour(_X, _Y, np.log10(rosen((_X, _Y)) + 1e-3), levels=25, cmap="Greys", linewidths=0.6)
    _a.plot(1, 1, "*", color="gold", ms=15, mec="k")
    _rows = []
    for _name, _opt in _opts.items():
        _path = np.asarray(run_optax(_opt, rosen, [-1.5, 2.0], 1000))
        _loss = np.array([float(rosen(_q)) for _q in _path[::10]])
        _a.plot(_path[:, 0], _path[:, 1], lw=1.2, label=_name)
        _b.semilogy(np.arange(0, len(_path), 10), _loss + 1e-12, label=_name)
        _rows.append(f"| {_name} | ({_path[-1, 0]:.3f}, {_path[-1, 1]:.3f}) | {_loss[-1]:.2e} |")
    _a.set(title="paths on Rosenbrock (★ = minimum)", xlim=(-2, 2), ylim=(-1, 3))
    _a.legend(fontsize=7, loc="lower right")
    _b.set(title="loss vs step", xlabel="step")
    _fig.tight_layout()
    _table = "| optimizer | final (x, y) | final loss |\n|---|---|---|\n" + "\n".join(_rows)
    mo.vstack([_fig, mo.md(_table)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Plain gradient descent creeps down the curved valley and is still far from $(1,1)$ after 1,000 steps;
    momentum and Adam make it. That's condition number in action — and it's why almost every model you'll
    meet is trained with Adam or a close relative.

    ## 5 · Stochastic gradients: why a minibatch is noisy

    A real loss is an average over the dataset, $L=\frac1N\sum_i \ell_i$. Computing the full gradient
    every step is wasteful, so we estimate it from a random **minibatch** of $B$ examples. The estimate
    is unbiased but noisy, and its noise shrinks like $1/\sqrt B$: 4× the batch → half the noise.

    Below: a linear-regression problem with 8,192 examples. At a fixed point, sample 200 minibatch
    gradients per batch size and measure how far they scatter from the true gradient.
    deepseek-v3-jax used $B=32$ sequences of 256 tokens — each step saw 8,192 tokens.
    """)
    return


@app.cell
def _(jax, jnp, np, plt):
    _k1, _k2, _k3 = jax.random.split(jax.random.key(0), 3)
    _N, _D = 8192, 8
    _X = jax.random.normal(_k1, (_N, _D))
    _y = _X @ jnp.arange(1.0, _D + 1) + 0.5 * jax.random.normal(_k2, (_N,))
    _w = jnp.zeros(_D)

    def _grad_on(idx):
        _xb, _yb = _X[idx], _y[idx]
        return 2 * _xb.T @ (_xb @ _w - _yb) / idx.shape[0]

    _full = _grad_on(jnp.arange(_N))
    _sizes = [4, 16, 64, 256, 1024]
    _noise = []
    for _B in _sizes:
        _idx = jax.random.randint(jax.random.fold_in(_k3, _B), (200, _B), 0, _N)
        _gs = jax.vmap(_grad_on)(_idx)
        _noise.append(float(jnp.mean(jnp.linalg.norm(_gs - _full, axis=1)) / jnp.linalg.norm(_full)))
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.loglog(_sizes, _noise, "o-", label="measured")
    _ax.loglog(_sizes, _noise[0] * np.sqrt(_sizes[0] / np.array(_sizes)), "--", label="∝ 1/√B")
    _ax.set(xlabel="batch size B", ylabel="relative gradient error", title="minibatch gradient noise")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Noise isn't all bad — it helps escape sharp, narrow minima — but it does mean that late in training,
    a fixed learning rate keeps you bouncing around the bottom. Hence…

    ## 6 · Learning-rate schedules

    deepseek-v3-jax's `train.py` builds its schedule in one line:

    ```python
    optax.warmup_cosine_decay_schedule(0.0, 6e-4, 200, 12000, 6e-5)
    ```

    * **Warmup** (steps 0→200): ramp linearly from 0 to the peak $6\times10^{-4}$. At initialization the
      gradients are large and Adam's statistics aren't settled; full-size steps there can blow up a
      transformer.
    * **Cosine decay** (200→12,000): glide down a half cosine to $6\times10^{-5}$ (10% of peak), so late
      steps are small enough to settle into the minimum instead of bouncing around it.

    Your `warmup_cosine` (dashed) is drawn on top of optax's once you've implemented it.
    """)
    return


@app.cell
def _(ex, lj, mo, np, optax, plt):
    _sched = optax.warmup_cosine_decay_schedule(0.0, 6e-4, 200, 12000, 6e-5)
    _steps = np.arange(0, 12001, 20)
    _ref = np.array([float(_sched(int(_s))) for _s in _steps])
    _mine = lj.attempt(lambda: np.array([ex.warmup_cosine(int(_s), 6e-4, 200, 12000, 6e-5) for _s in _steps]))
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.2))
    for _ax, _lim in ((_a, 12000), (_b, 400)):
        _ax.plot(_steps, _ref, label="optax")
        if _mine.ok:
            _ax.plot(_steps, _mine.value, "--", label="yours")
        _ax.set(xlim=(0, _lim), xlabel="step", ylabel="learning rate")
    _a.set_title("full run")
    _b.set_title("first 400 steps (warmup)")
    _a.legend()
    _fig.tight_layout()
    _out = [_fig]
    if _mine.ok:
        _out.append(mo.md(f"Max difference from optax: **{np.max(np.abs(_mine.value - _ref)):.2e}**"))
    else:
        _out.append(_mine.view(mo))
    mo.vstack(_out)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Gradient clipping

    Occasionally one batch produces an enormous gradient (an unlucky batch, a spike in attention logits).
    One giant step can undo hours of training. **Clipping by global norm** caps the *length* of the whole
    gradient — all parameters together, treated as one long vector — while keeping its *direction*:

    $$\text{if } \|g\| > c:\quad g \leftarrow g\cdot\frac{c}{\|g\|}$$

    deepseek-v3-jax uses $c=1.0$. Clipping the global norm (not each value separately) matters: clipping
    values one at a time would change the direction you step in.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _g = {"embed": jnp.full((4, 3), 2.0), "head": [jnp.array([-6.0, 1.0]), jnp.array(3.0)]}

    def _norm(t):
        return float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(t))))

    _c = lj.attempt(ex.clip_by_global_norm, _g, 1.0)
    _items = [mo.md(f"A gradient pytree with global norm **{_norm(_g):.2f}**, clipped to max norm 1.0:")]
    if _c.ok:
        _cos = float(sum(jnp.sum(a * b) for a, b in zip(jax.tree.leaves(_g), jax.tree.leaves(_c.value)))
                     / (_norm(_g) * _norm(_c.value)))
        _items.append(mo.md(f"norm after: **{_norm(_c.value):.4f}** · cosine with original direction: "
                            f"**{_cos:.6f}** (1 = same direction)"))
    else:
        _items.append(_c.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Weight decay vs L2 regularization (the "W" in AdamW)

    *Weight decay* shrinks every weight a little each step, nudging the model toward small weights —
    simpler functions that tend to generalize better. With plain gradient descent there are two
    equivalent ways to do it: add $\tfrac{\lambda}{2}\|\theta\|^2$ to the loss (**L2**), or subtract
    $\eta\lambda\theta$ in the update (**decay**).

    With Adam they are **not** equivalent. If you add L2 to the loss, the extra gradient $\lambda\theta$
    goes through Adam's $1/\sqrt{\hat v}$ scaling — so parameters with big gradients barely decay and
    parameters with tiny gradients decay a lot. **AdamW** *decouples* the two: it does the Adam step on
    the loss gradient alone, then separately applies $\theta \leftarrow \theta - \eta\lambda\theta$, so
    every weight decays at the same rate. deepseek-v3-jax uses `adamw(..., weight_decay=0.1)`.
    """)
    return


@app.cell
def _(jax, jnp, mo, optax):
    _sched = optax.warmup_cosine_decay_schedule(0.0, 6e-4, 200, 12000, 6e-5)
    _opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(_sched, b1=0.9, b2=0.95, weight_decay=0.1),
    )
    _params = {"embed": jnp.ones((8, 4)), "head": jnp.ones((4, 8))}
    _state = _opt.init(_params)
    _grads = jax.tree.map(lambda p: 0.01 * p, _params)
    _updates, _state = _opt.update(_grads, _state, _params)
    _new = optax.apply_updates(_params, _updates)
    _leaves = jax.tree.leaves(_state)
    mo.md(f"""
    ## 9 · The optax way — your deepseek-v3-jax optimizer, verbatim

    ```python
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),                                   # Section 7
        optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=0.1),            # Sections 4, 6, 8
    )
    state = opt.init(params)                          # m and v zeros, step count 0 — your (m, v, t)
    updates, state = opt.update(grads, state, params) # the math you're writing in adam_step
    params = optax.apply_updates(params, updates)     # params + updates  (updates are already negative)
    ```

    Optax splits "compute the step" (`update`, a pure function of gradients and state) from "apply it"
    (`apply_updates`). That separation is what lets `chain` stack transformations: clip first, then Adam,
    then weight decay — each one rewrites the `updates` pytree and passes it on.

    Running it once on a toy 2-matrix model: the optimizer state holds **{len(_leaves)} arrays/counters**
    (clip has no state; Adam keeps a step count plus `m` and `v` for each of 2 params; the schedule keeps
    its own count). The first step moved `embed[0,0]` from 1.0 to **{float(_new['embed'][0, 0]):.10f}** —
    tiny, because warmup's learning rate at step 0 is exactly 0.

    `b2=0.95` instead of the default 0.999 is common for transformers: `v` forgets faster, so Adam reacts
    sooner when gradient scales shift.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 · Your turn: build the optimizers

    Open `modules/05-optimization/exercises.py`. Six functions: `sgd_step`, `momentum_step`, `adam_step`,
    `warmup_cosine`, `clip_by_global_norm`, `fit`. Each works on pytrees via [`jax.tree.map`](https://docs.jax.dev/en/latest/_autosummary/jax.tree.map.html#jax.tree.map). The cells
    below check your step functions against optax over several steps on a small pytree.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, optax):
    def _tree(i):
        _k = jax.random.key(i)
        return {"w": jax.random.normal(_k, (3, 2)), "b": [jax.random.normal(jax.random.fold_in(_k, 1), (2,))]}

    def _maxdiff(a, b):
        return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))

    def _compare(mine_step, init_mine, opt):
        _p_mine = _p_ref = _tree(100)
        _s_mine, _s_ref = init_mine(_p_mine), opt.init(_p_ref)
        for _i in range(5):
            _g = _tree(_i)
            _p_mine, _s_mine = mine_step(_p_mine, _g, _s_mine)
            _u, _s_ref = opt.update(_g, _s_ref, _p_ref)
            _p_ref = optax.apply_updates(_p_ref, _u)
        return _maxdiff(_p_mine, _p_ref)

    def _zeros(p):
        return jax.tree.map(jnp.zeros_like, p)

    _checks = {
        "sgd_step vs optax.sgd(0.1)": lambda: _compare(
            lambda p, g, s: (ex.sgd_step(p, g, 0.1), s), lambda p: None, optax.sgd(0.1)),
        "momentum_step vs optax.sgd(0.1, momentum=0.9)": lambda: _compare(
            lambda p, g, s: ex.momentum_step(p, g, s, 0.1, 0.9), _zeros, optax.sgd(0.1, momentum=0.9)),
        "adam_step vs optax.adam(0.01)": lambda: _compare(
            lambda p, g, s: ex.adam_step(p, g, s, 0.01), lambda p: (_zeros(p), _zeros(p), 0),
            optax.adam(0.01)),
    }
    _rows = []
    for _name, _fn in _checks.items():
        _a = lj.attempt(_fn)
        if _a.ok:
            _rows.append(mo.md(f"**{_name}** — max difference after 5 steps: `{_a.value:.2e}` "
                               f"{'✅' if _a.value < 1e-5 else '❌'}"))
        else:
            _rows.append(mo.vstack([mo.md(f"**{_name}**"), _a.view(mo)]))
    mo.vstack(_rows)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Fitting a curve: your gradient descent vs optax's Adam

    Data: $y = x^3 - x + \text{noise}$ for $x\in[-1.5,1.5]$. Model: a cubic polynomial
    $\hat y = w_0 + w_1x + w_2x^2 + w_3x^3$ — four parameters, mean-squared-error loss. Your `fit`
    (plain gradient descent) and optax's Adam each get 300 steps at learning rate 0.1. The features
    $1, x, x^2, x^3$ have different scales, so gradient descent's safe learning rate is capped by the
    steepest direction (Section 3) and it ends a little behind Adam, which rescales each weight.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np, optax, plt):
    _k1, _k2 = jax.random.split(jax.random.key(5))
    _x = jnp.linspace(-1.5, 1.5, 128)
    _y = _x**3 - _x + 0.1 * jax.random.normal(_k1, _x.shape)
    _data = (_x, _y)

    def _loss(p, d):
        _xs, _ys = d
        _pred = p["w"][0] + p["w"][1] * _xs + p["w"][2] * _xs**2 + p["w"][3] * _xs**3
        return jnp.mean((_pred - _ys) ** 2)

    _p0 = {"w": jnp.zeros(4)}
    _opt = optax.adam(0.1)

    @jax.jit
    def _adam_step(p, s):
        _u, s = _opt.update(jax.grad(_loss)(p, _data), s, p)
        return optax.apply_updates(p, _u), s

    _pa, _sa = _p0, _opt.init(_p0)
    for _ in range(300):
        _pa, _sa = _adam_step(_pa, _sa)

    _mine = lj.attempt(ex.fit, _loss, _p0, _data, 300, 0.1)
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.scatter(np.asarray(_x), np.asarray(_y), s=6, color="gray", label="data")
    _ax.plot(_x, _x**3 - _x, "k:", lw=1, label="true curve")

    def _curve(p):
        return p["w"][0] + p["w"][1] * _x + p["w"][2] * _x**2 + p["w"][3] * _x**3

    _ax.plot(_x, _curve(_pa), label=f"optax Adam (loss {float(_loss(_pa, _data)):.4f})")
    _items = []
    if _mine.ok:
        _ax.plot(_x, _curve(_mine.value), "--", label=f"your fit, GD (loss {float(_loss(_mine.value, _data)):.4f})")
        _items.append(mo.md(f"True weights (0, −1, 0, 1). Adam: `{np.round(np.asarray(_pa['w']), 3)}` · "
                            f"yours: `{np.round(np.asarray(_mine.value['w']), 3)}`"))
    else:
        _items.append(_mine.view(mo))
    _ax.legend(fontsize=8)
    _ax.set(title="cubic fit after 300 steps")
    _fig.tight_layout()
    mo.vstack([_fig, *_items])
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Lower the learning rate (or add warmup / clipping)": "a",
            "Raise the learning rate so it escapes faster": "b",
            "Switch from Adam to plain gradient descent": "c",
        },
        label="**Check yourself:** your loss falls for a while, then starts oscillating wildly and finally "
              "becomes `nan`. What's the first thing to try?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _feedback = {
        "a": mo.callout(mo.md("Right. Growing oscillation is Section 2's η > 1 regime in many dimensions. "
                              "Smaller steps, a warmup, or gradient clipping keep the update stable."),
                        kind="success"),
        "b": mo.callout(mo.md("That's the direction that *causes* divergence — the factor |1 − 2η| grows "
                              "with η."), kind="warn"),
        "c": mo.callout(mo.md("Plain gradient descent has the same stability limit and copes worse with "
                              "poor conditioning. The learning rate is the culprit."), kind="warn"),
    }
    _feedback.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(5))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Gabriel Goh, [Why Momentum Really Works](https://distill.pub/2017/momentum/) — interactive, the
      condition-number story told beautifully.
    * Kingma & Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — short
      and readable; Algorithm 1 is your `adam_step`.
    * Loshchilov & Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — AdamW.
    * Google DeepMind, [Optax documentation](https://optax.readthedocs.io/) — especially "Gradient transformations".
    * Sebastian Ruder, [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/).

    **Next:** Module 06 — [`jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit) and XLA: what actually happens when you compile a training step.
    """)
    return


if __name__ == "__main__":
    app.run()
