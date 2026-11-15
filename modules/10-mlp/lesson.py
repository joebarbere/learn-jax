import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="10 · MLPs")


@app.cell
def _():
    import marimo as mo
    import time

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax
    from flax import nnx

    import learnjax as lj
    from learnjax.data import FASHION_CLASSES, fashion_mnist

    ex = lj.load_exercises(mo.notebook_dir())
    return FASHION_CLASSES, ex, fashion_mnist, jax, jnp, lj, mo, nnx, np, optax, plt, time


@app.cell
def _(mo):
    mo.md(r"""
    # 10 · MLPs on Fashion-MNIST, then Flax NNX

    **Goal:** build your first real neural network — a multilayer perceptron (MLP) — by hand in pure JAX,
    train it on 60,000 images on the GPU, then rebuild it in **Flax NNX**, the library most JAX models
    (and the rest of this course) use, and see that the library is the same pytree-of-arrays underneath.

    **You will:** see why a linear model can't separate XOR · watch hidden layers + a nonlinearity bend
    decision boundaries · fit a curve with a width slider (universal approximation, felt not proved) ·
    write He init, the forward pass, the loss, a shuffling batch iterator, accuracy and a pure train step ·
    train to ~88% test accuracy in seconds · port it to `flax.nnx` and compare.

    Module 09 ended with softmax regression on these images at about **84%**. One hidden layer pair
    should beat it comfortably. A PyTorch twin of this notebook is in `torch_compare.py`.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Frank Rosenblatt's perceptron (1958) learned simple patterns and drew huge excitement. In 1969 Minsky
    and Papert's book "Perceptrons" showed a single-layer perceptron cannot compute XOR, and interest
    collapsed into the first "AI winter". Multi-layer networks came back when backpropagation (Rumelhart,
    Hinton & Williams, 1986) showed how to train hidden layers, and Cybenko (1989) proved they can
    approximate any continuous function. Later activations such as ReLU (around 2010) and GELU (2016) made
    deep stacks train better.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · What a linear model can't do

    A linear classifier draws a straight line (a hyperplane, in more dimensions) and says "class A on this
    side, class B on that side". That's Module 09. Now take the four points of **XOR**: label 1 when
    exactly one coordinate is positive. No single straight line puts both orange points on one side and both
    blue points on the other — try it in your head.

    The fix is to let the model **bend space first**. An MLP computes
    $$h = \mathrm{relu}(x W_1 + b_1), \qquad \text{logits} = h W_2 + b_2 .$$
    The hidden layer $h$ is a new set of coordinates in which the classes *are* linearly separable, and the
    last layer is just a linear classifier on those. The **nonlinearity** (relu) is essential: without it,
    $(xW_1+b_1)W_2+b_2 = x(W_1W_2) + \ldots$ collapses back to one linear layer, however many you stack.
    """)
    return


@app.cell
def _(jax, jnp, optax):
    def train_tiny(key, x, y, sizes, steps=600, lr=0.03, task="classify"):
        """Train a small MLP (lesson helper, independent of your exercises). Returns predict(x)."""
        keys = jax.random.split(key, len(sizes) - 1)
        params = [{"w": jax.random.normal(k, (a, b)) * jnp.sqrt(2.0 / a), "b": jnp.zeros(b)}
                  for k, a, b in zip(keys, sizes[:-1], sizes[1:], strict=True)]

        def apply(p, xx):
            for layer in p[:-1]:
                xx = jax.nn.relu(xx @ layer["w"] + layer["b"])
            return xx @ p[-1]["w"] + p[-1]["b"]

        def loss(p):
            out = apply(p, x)
            if task == "classify":
                return optax.softmax_cross_entropy_with_integer_labels(out, y).mean()
            return jnp.mean((out[:, 0] - y) ** 2)

        opt = optax.adam(lr)

        @jax.jit
        def run(p):
            def body(carry, _):
                p, s = carry
                g = jax.grad(loss)(p)
                u, s = opt.update(g, s, p)
                return (optax.apply_updates(p, u), s), None

            (p, _), _ = jax.lax.scan(body, (p, opt.init(p)), None, length=steps)
            return p

        trained = run(params)
        return lambda xx: apply(trained, xx)

    return (train_tiny,)


@app.cell
def _(jax, jnp, np, plt, train_tiny):
    _k1, _k2 = jax.random.split(jax.random.key(0))
    _x = jax.random.uniform(_k1, (400, 2), minval=-1, maxval=1)
    _y = ((_x[:, 0] > 0) ^ (_x[:, 1] > 0)).astype(jnp.int32)
    _lin = train_tiny(_k2, _x, _y, [2, 2])
    _mlp = train_tiny(_k2, _x, _y, [2, 16, 2])
    _g = np.linspace(-1, 1, 150)
    _G = jnp.asarray(np.stack(np.meshgrid(_g, _g), -1).reshape(-1, 2))
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4))
    for _ax, _f, _name in zip(_axes, [_lin, _mlp], ["linear model", "MLP, 16 hidden units"], strict=True):
        _pred = np.asarray(_f(_G).argmax(-1)).reshape(150, 150)
        _acc = float((_f(_x).argmax(-1) == _y).mean())
        _ax.contourf(_g, _g, _pred, levels=[-0.5, 0.5, 1.5], colors=["#cfe3f5", "#fbd9b6"])
        _ax.scatter(_x[:, 0], _x[:, 1], c=np.where(np.asarray(_y) == 1, "tab:orange", "tab:blue"), s=8)
        _ax.set(title=f"{_name}: {_acc:.0%} correct", aspect="equal", xticks=[], yticks=[])
    _fig.suptitle("XOR: label 1 when exactly one coordinate is positive")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(jax, jnp, plt):
    _x = jnp.linspace(-4, 4, 400)
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.2))
    for _name, _f in [("relu", jax.nn.relu), ("gelu", jax.nn.gelu), ("tanh", jnp.tanh), ("sigmoid", jax.nn.sigmoid)]:
        _a.plot(_x, _f(_x), label=_name)
        _b.plot(_x, jax.vmap(jax.grad(_f))(_x), label=_name)
    _a.set(title="activation functions", xlabel="x", ylim=(-1.5, 4))
    _b.set(title="their derivatives", xlabel="x")
    _a.legend()
    _a.grid(alpha=0.3)
    _b.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Choosing a nonlinearity

    * **ReLU** $=\max(0,x)$: dirt cheap, gradient exactly 1 for positive inputs so signals don't shrink
      through many layers. The default for MLPs and CNNs.
    * **GELU**: a smooth ReLU. Transformers (Module 16) use it or its cousin SiLU/SwiGLU — your
      deepseek-v3-jax model uses `silu` inside SwiGLU.
    * **tanh / sigmoid**: the classics. Look at their derivatives: they're tiny away from 0, so gradients
      **vanish** when you stack many layers. Module 11 measures this.

    ## 3 · How expressive is one hidden layer?

    A ReLU network computes a **piecewise linear** function: each hidden unit adds one "kink". Give it
    enough units and it can approximate any reasonable curve — the *universal approximation theorem*. The
    slider sets the hidden width; each fit trains for 1,500 Adam steps on the GPU (well under a second,
    thanks to [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan), which Module 08 explains).
    """)
    return


@app.cell
def _(mo):
    width = mo.ui.slider(steps=[1, 2, 4, 8, 16, 64], value=4, label="hidden units", show_value=True)
    width
    return (width,)


@app.cell
def _(jax, jnp, plt, train_tiny, width):
    def _target(x):
        return jnp.sin(3 * x) + 0.3 * x**2

    _x = jnp.linspace(-2, 2, 256)[:, None]
    _f = train_tiny(jax.random.key(1), _x, _target(_x[:, 0]), [1, width.value, 1], steps=1500, lr=0.02,
                    task="regress")
    _pred = _f(_x)[:, 0]
    _mse = float(jnp.mean((_pred - _target(_x[:, 0])) ** 2))
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot(_x[:, 0], _target(_x[:, 0]), color="k", lw=2, label="target  sin(3x) + 0.3x²")
    _ax.plot(_x[:, 0], _pred, color="tab:orange", lw=2, label=f"MLP, {width.value} hidden ReLUs")
    _ax.set(title=f"one hidden layer, {width.value} units: mean squared error {_mse:.4f}", xlabel="x")
    _ax.legend(loc="upper center", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    With 1 unit you get a single bend; with 4, a few straight segments; by 16–64 the fit is nearly
    perfect. Width buys expressiveness — but notice nothing here says how well it **generalizes** to
    points it never saw. That's what the validation set below is for.

    ## 4 · Fashion-MNIST

    70,000 grayscale 28×28 images of clothing in 10 classes (60k train, 10k test) — a drop-in, harder
    replacement for handwritten digits. We flatten each image to 784 numbers, scale pixels to [0, 1], then
    standardize with the training set's mean and std, and hold out the last 5,000 training images as a
    **validation** set for watching overfitting. The test set is touched only at the end.
    """)
    return


@app.cell
def _(fashion_mnist, jnp, np):
    _d = fashion_mnist()
    _mean, _std = 0.2860, 0.3530  # training-set pixel mean / std after scaling to [0, 1]

    def _prep(images):
        return jnp.asarray((images.reshape(len(images), -1).astype(np.float32) / 255.0 - _mean) / _std)

    raw_images = _d["test_images"]
    x_train, y_train = _prep(_d["train_images"][:55000]), jnp.asarray(_d["train_labels"][:55000])
    x_val, y_val = _prep(_d["train_images"][55000:]), jnp.asarray(_d["train_labels"][55000:])
    x_test, y_test = _prep(_d["test_images"]), jnp.asarray(_d["test_labels"])
    return raw_images, x_test, x_train, x_val, y_test, y_train, y_val


@app.cell
def _(FASHION_CLASSES, mo, plt, raw_images, x_train, y_test):
    _fig, _axes = plt.subplots(2, 8, figsize=(10, 2.8))
    for _i, _ax in enumerate(_axes.flat):
        _ax.imshow(raw_images[_i], cmap="gray_r")
        _ax.set_title(FASHION_CLASSES[int(y_test[_i])], fontsize=7)
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"`x_train`: shape `{tuple(x_train.shape)}`, dtype `{x_train.dtype}`, "
                           f"mean {float(x_train.mean()):.3f}, std {float(x_train.std()):.3f} — on "
                           f"`{list(x_train.devices())[0]}`, so every batch below is sliced on the GPU.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Your turn: build the MLP by hand

    Open `modules/10-mlp/exercises.py`. Our network is **784 → 256 → 256 → 10**: two hidden ReLU layers
    and an output layer of 10 logits. Parameters are a list of `{"w", "b"}` dicts — a pytree.

    **He initialization.** Each hidden unit sums 784 inputs. If weights were standard normal, that sum would
    have variance ~784 and activations would explode layer after layer; too small and they'd vanish. Drawing
    weights with std $\sqrt{2/\text{fan\_in}}$ keeps the variance of ReLU activations roughly constant
    (the 2 compensates for ReLU zeroing half its inputs). Module 11 shows the alternative going wrong.
    """)
    return


@app.cell
def _(ex, jax, lj, mo, np):
    _w = lj.attempt(ex.he_init, jax.random.key(0), 784, 256)
    _items = [mo.md("**`he_init(key, 784, 256)`**"), _w.view(mo, show_value=False)]
    if _w.ok:
        _items.append(mo.md(f"shape `{tuple(_w.value.shape)}`, dtype `{_w.value.dtype}`, std "
                            f"**{float(_w.value.std()):.4f}** (target √(2/784) = {np.sqrt(2 / 784):.4f})"))
    mo.vstack(_items)
    return


@app.cell
def _(ex, jax, lj, mo, x_train, y_train):
    SIZES = [784, 256, 256, 10]
    _p = lj.attempt(ex.init_mlp, jax.random.key(0), SIZES)
    _items = [mo.md("**`init_mlp`, `mlp_apply`, `loss_fn`** on the first 128 training images:")]
    if not _p.ok:
        _items.append(_p.view(mo))
    else:
        _rows = "\n".join(f"| {i} | `{tuple(l['w'].shape)}` | `{tuple(l['b'].shape)}` | "
                          f"{l['w'].size + l['b'].size:,} |" for i, l in enumerate(_p.value))
        _n = sum(l["w"].size + l["b"].size for l in _p.value)
        _items.append(mo.md(f"| layer | w | b | params |\n|---|---|---|---|\n{_rows}\n| | | **total** | "
                            f"**{_n:,}** |"))
        _out = lj.attempt(ex.mlp_apply, _p.value, x_train[:128])
        _items.append(_out.view(mo, show_value=False))
        if _out.ok:
            _items.append(mo.md(f"logits shape `{tuple(_out.value.shape)}` — one score per class per image."))
        _l = lj.attempt(ex.loss_fn, _p.value, x_train[:128], y_train[:128])
        _items.append(_l.view(mo, show_value=False))
        if _l.ok:
            _items.append(mo.md(
                f"Initial loss **{float(_l.value):.3f}**. A model that knows nothing *and says so* would spread "
                f"probability evenly over 10 classes: ln 10 = 2.303 (Module 04). Ours starts a bit higher "
                f"because random weights make random, somewhat confident guesses. Checking the first loss is a "
                f"habit worth keeping: around 2–4 here is fine; 20 or `nan` means the init or the loss is broken."))
    mo.vstack(_items)
    return (SIZES,)


@app.cell
def _(mo):
    mo.md(r"""
    **Minibatches.** Each step uses 128 images, not all 55,000: the gradient is noisier (Module 05) but
    you get 430 updates per pass over the data (an **epoch**) instead of one. Shuffle every epoch with a
    fresh key so batches differ, and keep every batch the same size so [`jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit) compiles once.
    """)
    return


@app.cell
def _(ex, jax, lj, mo, np, x_train):
    _b = lj.attempt(ex.data_batches, jax.random.key(0), len(x_train), 128)
    _items = [mo.md("**`data_batches(key, 55000, 128)`**"), _b.view(mo, show_value=False)]
    if _b.ok:
        _all = np.concatenate([np.asarray(_x) for _x in _b.value])
        _items.append(mo.md(f"{len(_b.value)} batches of {len(_b.value[0])} · {len(np.unique(_all)):,} unique "
                            f"indices ({len(x_train) - len(_all)} dropped) · first batch starts "
                            f"`{np.asarray(_b.value[0][:6]).tolist()}`"))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Train it

    `train_step` is a **pure function**: parameters and optimizer state go in, new ones come out. That's
    what lets us wrap it in `jax.jit` and have XLA compile the forward pass, backward pass and AdamW update
    into a handful of GPU kernels. The optimizer is closed over (it's configuration, not data):

    ```python
    optimizer = optax.adamw(1e-3, weight_decay=1e-4)
    step = jax.jit(lambda p, s, x, y: ex.train_step(p, s, x, y, optimizer))
    for epoch in range(10):
        for idx in ex.data_batches(epoch_key, n, 128):
            params, opt_state, loss = step(params, opt_state, x_train[idx], y_train[idx])
    ```

    The cell below runs exactly that for 10 epochs (~10 s on the RX 7600 XT) once your functions work.
    """)
    return


@app.cell
def _(SIZES, ex, jax, jnp, lj, mo, optax, time, x_train, x_val, y_train, y_val):
    EPOCHS = 10
    _needed = [ex.he_init, ex.init_mlp, ex.mlp_apply, ex.loss_fn, ex.data_batches, ex.accuracy, ex.train_step]
    _optimizer = optax.adamw(1e-3, weight_decay=1e-4)

    def _train():
        params = ex.init_mlp(jax.random.key(0), SIZES)
        opt_state = _optimizer.init(params)
        step = jax.jit(lambda p, s, x, y: ex.train_step(p, s, x, y, _optimizer))
        key = jax.random.key(1)
        history = []
        t0 = time.perf_counter()
        train_time = 0.0
        for epoch in range(EPOCHS):
            key, sub = jax.random.split(key)
            losses = []
            t_epoch = time.perf_counter()
            for idx in ex.data_batches(sub, len(x_train), 128):
                params, opt_state, loss = step(params, opt_state, x_train[idx], y_train[idx])
                losses.append(loss)
            jax.block_until_ready(params)  # wait for the GPU before reading the clock
            epoch_seconds = time.perf_counter() - t_epoch
            train_time += epoch_seconds
            history.append({"epoch": epoch + 1, "train_loss": float(jnp.mean(jnp.stack(losses))),
                            "train_acc": ex.accuracy(params, x_train[:10000], y_train[:10000]),
                            "val_acc": ex.accuracy(params, x_val, y_val),
                            "epoch_seconds": epoch_seconds, "train_seconds": train_time,
                            "seconds": time.perf_counter() - t0})
        return params, history

    hand = lj.attempt(_train)
    if hand.ok:
        _h = hand.value[1]
        _view = mo.md(f"Trained {EPOCHS} epochs in **{_h[-1]['seconds']:.1f} s** (the first epoch includes "
                      f"compilation: {_h[0]['seconds']:.1f} s). Final validation accuracy "
                      f"**{_h[-1]['val_acc']:.1%}**.")
    elif hand.todo:
        _view = mo.callout(mo.md("**Your turn:** finish all seven functions in `exercises.py` "
                                 f"({', '.join(f'`{f.__name__}`' for f in _needed)}) and this cell trains "
                                 "your network. `make test N=10` tells you which are done."), kind="info")
    else:
        _view = hand.view(mo)
    _view
    return EPOCHS, hand


@app.cell
def _(hand, plt):
    if hand.ok:
        _h = hand.value[1]
        _ep = [r["epoch"] for r in _h]
        _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.2))
        _a.plot(_ep, [r["train_loss"] for r in _h], "o-")
        _a.set(title="training loss (mean over epoch)", xlabel="epoch")
        _b.plot(_ep, [r["train_acc"] for r in _h], "o-", label="train (first 10k)")
        _b.plot(_ep, [r["val_acc"] for r in _h], "o-", label="validation")
        _b.axhline(0.84, color="gray", ls="--", lw=1, label="Module 09 softmax regression")
        _b.set(title="accuracy", xlabel="epoch", ylim=(0.8, 1.0))
        _b.legend(fontsize=8)
        _fig.tight_layout()
        _out = _fig
    else:
        _out = None
    _out
    return


@app.cell
def _(ex, hand, mo, x_test, y_test):
    if hand.ok:
        _h = hand.value[1]
        hand_test = ex.accuracy(hand.value[0], x_test, y_test)
        _gap = _h[-1]["train_acc"] - _h[-1]["val_acc"]
        _out = mo.md(f"""
    **Test accuracy: {hand_test:.1%}** — up from ~84% for softmax regression, with 269k parameters and
    ten lines of model code. Look at the right-hand plot: training accuracy keeps climbing while validation
    flattens out (gap now {_gap:+.1%}). That gap is **overfitting** — the network starts memorizing
    training images. Module 11's tools (weight decay, dropout, early stopping) are about that gap.
    """)
    else:
        hand_test = None
        _out = mo.md("")
    _out
    return (hand_test,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6b · Making it fast the JAX way

    How long did one epoch of *training steps* take above — about a second? The model does ~0.2 GFLOP of
    math per step (forward + backward), roughly **10 µs** on this card, and 429 steps should be a few
    milliseconds of GPU work. The rest is **overhead**: every step, Python builds an index slice, slices
    the data, and calls the jitted function, which flattens and checks its pytree arguments and launches
    a couple of dozen small kernels. A model this small is **overhead-bound**: the GPU spends most of the
    epoch waiting for Python.

    The JAX fix is to hand XLA the *whole epoch*:

    1. Keep the data on the device (it already is) and pass it **as arguments**, not captured constants.
    2. Shuffle with a key *inside* the compiled function and reshape the permutation to `(steps, batch)`.
    3. Run the steps with `jax.lax.scan`, a compiled loop: `scan(body, carry, xs)` calls `body(carry, x)`
       for each row of `xs`, threading `carry` (here: params and optimizer state) through, and stacks the
       per-step outputs (here: losses). Module 08 covers `scan` properly.

    ```python
    @jax.jit
    def train_epoch(params, opt_state, key, x, y):
        idx = jax.random.permutation(key, len(x))[: STEPS * 128].reshape(STEPS, 128)
        def body(carry, batch_idx):
            params, opt_state = carry
            params, opt_state, loss = ex.train_step(params, opt_state, x[batch_idx], y[batch_idx], optimizer)
            return (params, opt_state), loss
        (params, opt_state), losses = jax.lax.scan(body, (params, opt_state), idx)
        return params, opt_state, losses.mean()
    ```

    Same `train_step` you wrote, same math — one Python call per epoch instead of 429.
    """)
    return


@app.cell
def _(EPOCHS, SIZES, ex, hand, jax, lj, mo, optax, time, x_train, x_val, y_train, y_val):
    _optimizer = optax.adamw(1e-3, weight_decay=1e-4)
    _steps = len(x_train) // 128

    @jax.jit
    def _train_epoch(params, opt_state, key, x, y):
        idx = jax.random.permutation(key, x.shape[0])[: _steps * 128].reshape(_steps, 128)

        def body(carry, batch_idx):
            p, s = carry
            p, s, loss = ex.train_step(p, s, x[batch_idx], y[batch_idx], _optimizer)
            return (p, s), loss

        (params, opt_state), losses = jax.lax.scan(body, (params, opt_state), idx)
        return params, opt_state, losses.mean()

    def _run():
        params = ex.init_mlp(jax.random.key(0), SIZES)
        opt_state = _optimizer.init(params)
        key = jax.random.key(1)
        times = []
        for _ in range(EPOCHS):
            key, sub = jax.random.split(key)
            t0 = time.perf_counter()
            params, opt_state, loss = _train_epoch(params, opt_state, sub, x_train, y_train)
            jax.block_until_ready(loss)
            times.append(time.perf_counter() - t0)
        return params, times

    scan_run = lj.attempt(_run) if hand.ok else None
    if scan_run is not None and scan_run.ok:
        _loop = hand.value[1]
        _loop_epoch = sum(r["epoch_seconds"] for r in _loop[1:]) / (len(_loop) - 1)
        _scan_epoch = sum(scan_run.value[1][1:]) / (len(scan_run.value[1]) - 1)
        _torch_epoch = 0.30
        _out = mo.md(f"""
    | training steps only, one epoch (429 steps), after the first | seconds per epoch | per step |
    |---|---|---|
    | Python loop over a jitted `train_step` (Section 6) | {_loop_epoch:.3f} s | {_loop_epoch / _steps * 1e3:.2f} ms |
    | **one jitted `lax.scan` per epoch** | **{_scan_epoch:.3f} s** | **{_scan_epoch / _steps * 1e3:.2f} ms** |
    | PyTorch eager, same model (`torch_compare.py`, measured when written) | ~{_torch_epoch:.2f} s | ~{_torch_epoch / _steps * 1e3:.2f} ms |

    The scan version is **{_loop_epoch / _scan_epoch:.0f}× faster** than the Python loop and about
    **{_torch_epoch / _scan_epoch:.1f}× faster** than PyTorch eager, with identical math (first epoch:
    {scan_run.value[1][0]:.1f} s, mostly compiling the scan). Validation accuracy after {EPOCHS} epochs:
    **{ex.accuracy(scan_run.value[0], x_val, y_val):.1%}**.

    Be honest about what this measures: all three run the same ~10 µs of matrix math per step. The
    differences are **overhead** — Python calls, argument checking, kernel launches. That's what dominates
    a 269k-parameter MLP, and it's why the "JAX vs PyTorch speed" question has no answer for tiny models:
    it depends on how much per-step Python you leave in the loop. For big models the picture flips:
    each step is milliseconds of real GPU work, overhead becomes a rounding error, and what matters is how
    well the kernels use the hardware (**compute-bound**). Module 16 trains a transformer that's firmly in
    that regime, and Module 22 profiles the difference.
    """)
    elif scan_run is not None:
        _out = scan_run.view(mo)
    else:
        _out = mo.md("*(This section runs once your exercises work.)*")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · The same network in Flax NNX

    Writing `init_mlp` and `mlp_apply` by hand is the best way to understand a network, and the worst way
    to build a big one. **Flax NNX** gives you Python objects for layers — closer to PyTorch's style — while
    keeping JAX's transformations. The model:

    ```python
    class MLP(nnx.Module):
        def __init__(self, sizes, rngs):
            self.layers = nnx.List([nnx.Linear(a, b, rngs=rngs, kernel_init=nnx.initializers.he_normal())
                                    for a, b in zip(sizes[:-1], sizes[1:])])
        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = nnx.relu(layer(x))
            return self.layers[-1](x)
    ```

    and the training step — note `optimizer.update(model, grads)` mutates the model *in place*, which is
    allowed because `nnx.jit` tracks the objects' state for you:

    ```python
    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(model):
            return optax.softmax_cross_entropy_with_integer_labels(model(x), y).mean()
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss
    ```

    This section runs regardless of your exercises.
    """)
    return


@app.cell
def _(nnx):
    class MLP(nnx.Module):
        def __init__(self, sizes, rngs):
            self.layers = nnx.List([nnx.Linear(a, b, rngs=rngs, kernel_init=nnx.initializers.he_normal())
                                    for a, b in zip(sizes[:-1], sizes[1:], strict=True)])

        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = nnx.relu(layer(x))
            return self.layers[-1](x)

    return (MLP,)


@app.cell
def _(EPOCHS, MLP, SIZES, jax, jnp, nnx, optax, time, x_train, x_val, y_train, y_val):
    @nnx.jit
    def nnx_train_step(model, optimizer, x, y):
        def _loss(m):
            return optax.softmax_cross_entropy_with_integer_labels(m(x), y).mean()

        loss, grads = nnx.value_and_grad(_loss)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def nnx_correct(model, x, y):
        return (model(x).argmax(-1) == y).sum()

    def nnx_accuracy(model, x, y, bs=1000):
        return sum(int(nnx_correct(model, x[i:i + bs], y[i:i + bs])) for i in range(0, len(x), bs)) / len(x)

    nnx_model = MLP(SIZES, nnx.Rngs(0))
    _opt = nnx.Optimizer(nnx_model, optax.adamw(1e-3, weight_decay=1e-4), wrt=nnx.Param)
    _key = jax.random.key(1)
    nnx_history = []
    _t0 = time.perf_counter()
    for _epoch in range(EPOCHS):
        _key, _sub = jax.random.split(_key)
        _perm = jax.random.permutation(_sub, len(x_train))
        _losses = [nnx_train_step(nnx_model, _opt, x_train[_idx], y_train[_idx])
                   for _idx in (_perm[_i * 128:(_i + 1) * 128] for _i in range(len(x_train) // 128))]
        nnx_history.append({"train_loss": float(jnp.mean(jnp.stack(_losses))),
                            "val_acc": nnx_accuracy(nnx_model, x_val, y_val),
                            "seconds": time.perf_counter() - _t0})
    return nnx_accuracy, nnx_history, nnx_model


@app.cell
def _(hand, hand_test, mo, nnx_accuracy, nnx_history, nnx_model, x_test, y_test):
    nnx_test = nnx_accuracy(nnx_model, x_test, y_test)
    _speed = (f"({nnx_history[-1]['seconds'] / hand.value[1][-1]['seconds']:.2f}× the hand-written time)"
              if hand.ok else "")
    _hand_row = (f"| your hand-written MLP (`jax.jit`) | {hand.value[1][-1]['seconds']:.1f} s | "
                 f"{hand.value[1][-1]['val_acc']:.1%} | {hand_test:.1%} |" if hand.ok
                 else "| your hand-written MLP | *finish the exercises* | | |")
    mo.md(f"""
    | {nnx_history.__len__()} epochs, same data, AdamW 1e-3 | time | val acc | test acc |
    |---|---|---|---|
    {_hand_row}
    | Flax NNX (`nnx.jit`) | {nnx_history[-1]['seconds']:.1f} s | {nnx_history[-1]['val_acc']:.1%} | {nnx_test:.1%} |

    Same architecture and optimizer, so the same accuracy within the noise of different random initial
    weights and shuffles. Both compile to nearly the same XLA program; the time difference
    {_speed} is per-step **Python** overhead: `nnx.jit` has to walk the model and optimizer objects to
    find their arrays on every call. With 430 tiny steps per epoch that shows; for a big model where each
    step takes milliseconds of GPU time it disappears (and `nnx.split` + a plain `jax.jit` removes it).
    """)
    return (nnx_test,)


@app.cell
def _(jax, mo, nnx, nnx_model):
    _graphdef, _state = nnx.split(nnx_model)
    _pure = nnx.to_pure_dict(_state)
    _shapes = jax.tree.map(lambda a: tuple(a.shape), _pure)
    _lines = "\n".join(f"    layers[{i}]: kernel {v['kernel']}, bias {v['bias']}" for i, v in _shapes["layers"].items())
    mo.md(f"""
    ## 8 · It's still a pytree

    `nnx.split(model)` separates a model into a **graph definition** (the Python structure: which layers,
    how they connect) and its **state** (the arrays). The state is exactly the pytree you built by hand —
    NNX calls `w` a `kernel` and stores it as a `Param` variable:

    ```
{_lines}
    ```

    `nnx.merge(graphdef, state)` puts them back together. This split is how NNX works with `jax.jit`,
    [`jax.grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html#jax.grad) and Orbax checkpoints (Module 11): transformations only ever see arrays.

    | you wrote | Flax NNX |
    |---|---|
    | `init_mlp(key, sizes)` → list of dicts | `MLP(sizes, nnx.Rngs(0))` — layers create their own params |
    | `he_init` | `kernel_init=nnx.initializers.he_normal()` |
    | `mlp_apply(params, x)` | `model(x)` |
    | `opt_state = optimizer.init(params)` | `nnx.Optimizer(model, tx, wrt=nnx.Param)` |
    | [`jax.value_and_grad(loss_fn)(params, x, y)`](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html#jax.value_and_grad) | `nnx.value_and_grad(loss_fn)(model)` |
    | `params = optax.apply_updates(...)` (returns new) | `optimizer.update(model, grads)` (updates in place) |
    | `jax.jit` on a pure function | `nnx.jit`, which handles the in-place updates for you |
    """)
    return


@app.cell
def _(FASHION_CLASSES, jnp, nnx_model, np, plt, raw_images, x_test, y_test):
    _pred = np.asarray(jnp.concatenate([nnx_model(x_test[_i:_i + 1000]).argmax(-1)
                                        for _i in range(0, len(x_test), 1000)]))
    _wrong = np.flatnonzero(_pred != np.asarray(y_test))[:12]
    _fig, _axes = plt.subplots(2, 6, figsize=(10, 4.6))
    for _ax, _i in zip(_axes.flat, _wrong, strict=False):
        _ax.imshow(raw_images[_i], cmap="gray_r")
        _ax.set_title(f"said {FASHION_CLASSES[_pred[_i]]}\nwas {FASHION_CLASSES[int(y_test[_i])]}", fontsize=7)
        _ax.axis("off")
    _fig.suptitle("Some test images the NNX model gets wrong")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Most mistakes are shirts vs T-shirts vs pullovers vs coats — hard even for people at 28×28. And the
    MLP treats an image as 784 unrelated numbers: shift a shirt two pixels and every input changes. Module
    12's convolutional networks build in the idea that nearby pixels belong together, and do better.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "It would collapse to a single linear layer, no more powerful than softmax regression": "a",
            "Training would be slower but reach the same accuracy": "b",
            "Gradients would explode": "c",
        },
        label="**Check yourself:** what happens if you delete the ReLUs from your 784→256→256→10 MLP?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _feedback = {
        "a": mo.callout(mo.md("Right. A product of matrices is a matrix: $xW_1W_2W_3$ is just $xW$. You'd be "
                              "back at Module 09's ~84%, with more parameters to train."), kind="success"),
        "b": mo.callout(mo.md("No — it can't reach the same accuracy at all. Without a nonlinearity the "
                              "layers multiply into one linear map."), kind="warn"),
        "c": mo.callout(mo.md("That's not the main effect. The real problem is expressiveness: stacked linear "
                              "layers equal one linear layer."), kind="warn"),
    }
    _feedback.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(10))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · The PyTorch side

    `modules/10-mlp/torch_compare.py` trains this exact network in PyTorch on ROCm, with the same
    hyperparameters, and ends with a line-by-line table of JAX vs PyTorch idioms. Open it with
    `make build-torch` (once; a ~26 GB image you can remove when you're done and rebuild later, see the
    README's *Disk space* section) then `make torch-lesson N=10`.

    ## Further reading
    * Michael Nielsen, [Neural Networks and Deep Learning, ch. 1–2](http://neuralnetworksanddeeplearning.com/)
      — the clearest first explanation of MLPs and backprop, plus a visual proof of universal approximation (ch. 4).
    * He et al., [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) — where He initialization comes from.
    * Flax docs: [Flax basics (NNX)](https://flax.readthedocs.io/en/latest/nnx_basics.html) and the
      [MNIST tutorial](https://flax.readthedocs.io/en/latest/mnist_tutorial.html).
    * Xiao et al., [Fashion-MNIST](https://arxiv.org/abs/1708.07747).

    **Next:** Module 11 — training craft: initialization, normalization, schedules, regularization and
    checkpoints.
    """)
    return


if __name__ == "__main__":
    app.run()
