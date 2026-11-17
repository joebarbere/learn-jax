import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="09 · Regression")


@app.cell
def _():
    import marimo as mo
    import time
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax
    from jax import lax

    import learnjax as lj
    from learnjax import rosetta
    from learnjax.data import FASHION_CLASSES, fashion_mnist

    ex = lj.load_exercises(mo.notebook_dir())
    return FASHION_CLASSES, ex, fashion_mnist, jax, jnp, lax, lj, mo, np, optax, plt, rosetta, time


@app.cell
def _(mo):
    mo.md(r"""
    # 09 · Linear and logistic regression from scratch

    **Goal:** train your first real models end to end — and learn the habits (data splits, a validation
    set, regularization, honest evaluation) that every later model depends on.

    **You will:** fit a line two ways and see they agree · overfit a polynomial on purpose with a slider ·
    learn what train / validation / test sets are *for* · tame overfitting with ridge regularization ·
    classify 2-D points with logistic regression and draw its decision boundary · train a 10-class
    classifier on 70,000 clothing images on your GPU and read its confusion matrix.

    Everything from Part 1 shows up here: arrays (01), matrices (02), gradients (03), cross-entropy (04)
    and optimizers (05). A "model" is just a function with parameters; "training" is Module 05 applied to it.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Least squares is over two hundred years old: Legendre published it in 1805 and Gauss soon after. The
    logistic curve dates to Verhulst in the 1830s and became a regression tool in mid-20th-century
    statistics. Hoerl & Kennard introduced ridge regression in 1970 to stabilize ill-conditioned problems.
    Fashion-MNIST (Xiao, Rasul & Vollgraf, 2017) was released as a harder drop-in replacement for the 1998
    MNIST digits.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Linear regression, two ways

    Model: $\hat y = w x + b$. Loss: mean squared error $\frac1n\sum_i (\hat y_i - y_i)^2$. Stack a column of
    ones next to $x$ and it becomes $\hat{\mathbf y} = X\mathbf w$ — a matrix problem with an exact answer,
    *least squares* ([`jnp.linalg.lstsq`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.lstsq.html#jax.numpy.linalg.lstsq)). It's also an optimization problem gradient descent can solve.
    For this tiny problem both work; the exact answer stops existing once models become nonlinear, and
    gradient descent is what's left. So it's worth seeing them agree once.
    """)
    return


@app.cell
def _(jax, jnp, mo, np, optax, plt):
    _k1, _k2 = jax.random.split(jax.random.key(0))
    _x = jax.random.uniform(_k1, (80,), minval=-3, maxval=3)
    _y = 2.0 * _x + 1.0 + 0.8 * jax.random.normal(_k2, (80,))
    _X = jnp.stack([_x, jnp.ones_like(_x)], axis=1)

    _w_exact = jnp.linalg.lstsq(_X, _y)[0]

    _opt = optax.sgd(0.05)
    _w = jnp.zeros(2)
    _state = _opt.init(_w)
    _loss = lambda w: jnp.mean((_X @ w - _y) ** 2)  # noqa: E731
    _history = []
    for _ in range(200):
        _l, _g = jax.value_and_grad(_loss)(_w)
        _upd, _state = _opt.update(_g, _state)
        _w = optax.apply_updates(_w, _upd)
        _history.append(float(_l))

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.4))
    _a.scatter(np.asarray(_x), np.asarray(_y), s=12, alpha=0.6)
    _xs = np.linspace(-3, 3, 50)
    _a.plot(_xs, _w_exact[0] * _xs + _w_exact[1], "C3", lw=2, label="least squares")
    _a.plot(_xs, _w[0] * _xs + _w[1], "k--", lw=1.5, label="gradient descent")
    _a.legend(fontsize=8)
    _a.set(title="data: y = 2x + 1 + noise")
    _b.semilogy(_history)
    _b.set(xlabel="step", ylabel="MSE", title="gradient descent loss")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Least squares: w = {float(_w_exact[0]):.3f}, b = {float(_w_exact[1]):.3f}. "
                           f"Gradient descent after 200 steps: w = {float(_w[0]):.3f}, b = {float(_w[1]):.3f}. "
                           "Same answer; the noise is why neither is exactly (2, 1).")])
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "least squares")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Overfitting, on purpose

    Linear models get much more expressive with **features**: feed in $1, x, x^2, \dots, x^d$ and a
    linear model in those columns is a degree-$d$ polynomial in $x$. More features → more flexible → the
    model can fit the training points better and better... including their noise.

    Below: 20 noisy samples of a smooth curve. The fit is judged on the **training** points it saw and on
    fresh **validation** points it didn't. Drag the degree up.
    """)
    return


@app.cell
def _(jax, jnp, np):
    def _truth(x):
        return np.sin(1.5 * x) + 0.3 * x

    _kx, _kn, _vx, _vn = jax.random.split(jax.random.key(1), 4)
    poly_train_x = np.asarray(jax.random.uniform(_kx, (20,), minval=-3, maxval=3))
    poly_train_y = _truth(poly_train_x) + 0.25 * np.asarray(jax.random.normal(_kn, (20,)))
    poly_val_x = np.asarray(jax.random.uniform(_vx, (200,), minval=-3, maxval=3))
    poly_val_y = _truth(poly_val_x) + 0.25 * np.asarray(jax.random.normal(_vn, (200,)))

    def poly_fit_eval(degree, lam=0.0):
        """Fit a degree-d polynomial (features scaled to [-1, 1]) with ridge λ; return weights + errors."""
        def feats(x):
            return jnp.asarray(x / 3.0)[:, None] ** jnp.arange(degree + 1)

        Xtr = feats(poly_train_x)
        w = jnp.linalg.solve(Xtr.T @ Xtr + lam * jnp.eye(degree + 1), Xtr.T @ poly_train_y)
        tr = float(jnp.mean((Xtr @ w - poly_train_y) ** 2))
        va = float(jnp.mean((feats(poly_val_x) @ w - poly_val_y) ** 2))
        return w, feats, tr, va

    poly_truth = _truth
    return poly_fit_eval, poly_train_x, poly_train_y, poly_truth, poly_val_x, poly_val_y


@app.cell
def _(mo):
    degree = mo.ui.slider(0, 15, value=3, label="polynomial degree", show_value=True)
    degree
    return (degree,)


@app.cell
def _(degree, np, plt, poly_fit_eval, poly_train_x, poly_train_y, poly_truth):
    _w, _feats, _tr, _va = poly_fit_eval(degree.value)
    _xs = np.linspace(-3, 3, 300)
    _degs = list(range(16))
    _errs = [poly_fit_eval(d)[2:] for d in _degs]
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.5))
    _a.scatter(poly_train_x, poly_train_y, s=18, label="training points", zorder=3)
    _a.plot(_xs, poly_truth(_xs), "k--", lw=1, label="true curve")
    _a.plot(_xs, np.asarray(_feats(_xs) @ _w), "C3", lw=2, label=f"degree {degree.value} fit")
    _a.set(ylim=(-3, 3), title=f"train MSE {_tr:.3f} · validation MSE {_va:.3f}")
    _a.legend(fontsize=8, loc="lower right")
    _b.semilogy(_degs, [e[0] for e in _errs], "o-", ms=3, label="train")
    _b.semilogy(_degs, [e[1] for e in _errs], "o-", ms=3, label="validation")
    _b.axvline(degree.value, color="0.6", lw=1)
    _b.set(xlabel="degree", ylabel="MSE", title="error vs model complexity")
    _b.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Training error keeps going **down** as the degree rises — with 20 points, degree 19 could pass through
    every one. (The small bumps at high degree are float32 round-off: those solves are badly conditioned.) Validation error falls, bottoms out, then climbs: past that point the model is
    memorizing noise (**overfitting**). Degree 0–1 is **underfitting**: too simple to follow the curve.

    ## 3 · Train, validation, test — what each one is for

    | Set | Used to… | Touched how often |
    |---|---|---|
    | **train** | fit the parameters (gradient descent sees only this) | every step |
    | **validation** | choose *hyperparameters*: degree, λ, learning rate, when to stop | many times |
    | **test** | report the final number, once, after every choice is made | once |

    Why three? Every time you pick something because it scored well on a set, that set's score becomes
    optimistic. Choosing the degree by validation error "spends" the validation set — so you keep a test
    set nobody has looked at. Your deepseek-v3-jax run did the same: training batches from 207 books,
    validation loss on 4 held-out books it never trained on.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Ridge regression: penalize big weights

    Overfit polynomials have huge, oscillating coefficients. **Ridge regression** adds a price on weight
    size: minimize $\lVert X\mathbf w - \mathbf y\rVert^2 + \lambda\lVert \mathbf w\rVert^2$. The solution is still
    closed-form, $(X^\top X + \lambda I)\,\mathbf w = X^\top\mathbf y$ — your `ridge_closed_form` exercise. This is
    the same idea as **weight decay** in AdamW (Module 05): pull weights toward zero unless the data
    really wants them big. Fix the degree at 15 and turn λ up:
    """)
    return


@app.cell
def _(mo):
    log_lam = mo.ui.slider(-8, 2, step=0.5, value=-8, label="log₁₀ λ", show_value=True)
    log_lam
    return (log_lam,)


@app.cell
def _(log_lam, np, plt, poly_fit_eval, poly_train_x, poly_train_y, poly_truth):
    _lam = 10.0 ** log_lam.value
    _w, _feats, _tr, _va = poly_fit_eval(15, _lam)
    _xs = np.linspace(-3, 3, 300)
    _grid = np.arange(-8, 2.5, 0.5)
    _vals = [poly_fit_eval(15, 10.0**g)[3] for g in _grid]
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.4))
    _a.scatter(poly_train_x, poly_train_y, s=18, zorder=3)
    _a.plot(_xs, poly_truth(_xs), "k--", lw=1)
    _a.plot(_xs, np.asarray(_feats(_xs) @ _w), "C3", lw=2)
    _a.set(ylim=(-3, 3), title=f"degree 15, λ = {_lam:.1e}: val MSE {_va:.3f}, |w| = {float(np.linalg.norm(_w)):.1f}")
    _b.semilogx(10.0**_grid, _vals, "o-", ms=3)
    _b.axvline(_lam, color="0.6", lw=1)
    _b.set(xlabel="λ", ylabel="validation MSE", yscale="log", title="pick λ on the validation set")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Too little λ: the wiggly overfit. Too much: everything is squashed toward zero and the model underfits.
    The validation curve tells you where in between to stop — then, and only then, you'd check the test set.

    ## 5 · Logistic regression: from numbers to classes

    For a yes/no question, predict a **probability** with the sigmoid,
    $p = \sigma(z) = 1/(1+e^{-z})$ where $z = \mathbf w\cdot\mathbf x + b$ is the **logit**, and train with
    **binary cross-entropy** — Module 04's cross-entropy with two classes. The decision boundary
    $z = 0$ is a straight line: the model is still linear, it just squashes its output.

    One numerical trap: computing `log(sigmoid(z))` naively gives $\log 0 = -\infty$ once $z$ is very
    negative. The stable form works from the logit directly:
    $\max(z,0) - zy + \log(1+e^{-|z|})$. That's your `bce_with_logits`.
    """)
    return


@app.cell
def _(jax, jnp, mo):
    _z = jnp.array([-200.0, -20.0, 0.0, 20.0])
    _y = jnp.ones(4)
    _naive = -(_y * jnp.log(jax.nn.sigmoid(_z)))
    _stable = jnp.maximum(_z, 0) - _z * _y + jnp.log1p(jnp.exp(-jnp.abs(_z)))
    _rows = "\n".join(f"| {float(a):g} | {float(b):g} | {float(c):g} |" for a, b, c in zip(_z, _naive, _stable))
    mo.md("Loss for label y = 1 at different logits (float32):\n\n| logit z | naive −log σ(z) | stable form |\n"
          "|---|---|---|\n" + _rows + "\n\nThe true loss at z = −200 is 200. The naive version says ∞ and "
          "its gradient is NaN — one such example and a whole training run turns to NaN.")
    return


@app.cell
def _(mo, rosetta):
    mo.vstack([
        rosetta.tabs(mo, "sigmoid, log-sigmoid and softplus"),
        rosetta.tabs(mo, "log1p and expm1 (accurate near zero)"),
    ])
    return


@app.cell
def _(jax, jnp, np, plt):
    _ka, _kb = jax.random.split(jax.random.key(5))
    blob_X = jnp.concatenate([jax.random.normal(_ka, (150, 2)) * 0.9 + jnp.array([1.2, 1.0]),
                              jax.random.normal(_kb, (150, 2)) * 0.9 - jnp.array([1.2, 1.0])])
    blob_y = jnp.concatenate([jnp.ones(150), jnp.zeros(150)])

    def _bce(z, y):
        return jnp.mean(jnp.maximum(z, 0) - z * y + jnp.log1p(jnp.exp(-jnp.abs(z))))

    @jax.jit
    def _step(p):
        g = jax.grad(lambda p: _bce(blob_X @ p[0] + p[1], blob_y))(p)
        return jax.tree.map(lambda a, b: a - 0.5 * b, p, g)

    _p = (jnp.zeros(2), jnp.zeros(()))
    for _ in range(300):
        _p = _step(_p)
    _gx, _gy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
    _prob = jax.nn.sigmoid(np.stack([_gx, _gy], -1) @ np.asarray(_p[0]) + float(_p[1]))
    _fig, _ax = plt.subplots(figsize=(5.2, 4.2))
    _cs = _ax.contourf(_gx, _gy, np.asarray(_prob), levels=20, cmap="RdBu_r", alpha=0.6)
    _ax.contour(_gx, _gy, np.asarray(_prob), levels=[0.5], colors="k")
    _ax.scatter(*np.asarray(blob_X[:150]).T, s=10, c="C3", label="class 1")
    _ax.scatter(*np.asarray(blob_X[150:]).T, s=10, c="C0", label="class 0")
    _fig.colorbar(_cs, ax=_ax, label="p(class 1)")
    _acc = float(jnp.mean((blob_X @ _p[0] + _p[1] > 0) == (blob_y > 0.5)))
    _ax.set(title=f"logistic regression · train accuracy {_acc:.1%}")
    _ax.legend(fontsize=8, loc="upper left")
    _fig.tight_layout()
    _fig
    return blob_X, blob_y


@app.cell
def _(mo):
    mo.md(r"""
    The black line is $p = 0.5$; the colour shows how *confident* the model is. Points in the overlap get
    misclassified — no straight line separates them, and no amount of training will fix that. Curved
    boundaries need nonlinear models: Module 10.

    ## 6 · Softmax regression on Fashion-MNIST — on your GPU

    Now real data: 70,000 grayscale 28×28 images of clothing in 10 classes. Flatten each image into 784
    numbers, and use **one weight vector per class**: logits $= X W + b$ with $W$ of shape 784×10. Softmax
    turns the 10 logits into probabilities and cross-entropy (Module 04) is the loss. This is logistic
    regression with 10 classes — still a linear model, 7,850 parameters.

    The 60,000 training images are split 55,000 train / 5,000 validation; the 10,000 test images are only
    used once, at the end.
    """)
    return


@app.cell
def _(FASHION_CLASSES, fashion_mnist, mo, np, plt):
    fmnist = fashion_mnist()
    _fig, _axes = plt.subplots(2, 10, figsize=(10, 2.4))
    for _c in range(10):
        _idx = np.where(fmnist["train_labels"] == _c)[0][:2]
        for _r in range(2):
            _axes[_r, _c].imshow(fmnist["train_images"][_idx[_r]], cmap="gray_r")
            _axes[_r, _c].axis("off")
        _axes[0, _c].set_title(FASHION_CLASSES[_c], fontsize=7)
    _fig.tight_layout()
    mo.vstack([mo.md(f"Train images {fmnist['train_images'].shape}, test images {fmnist['test_images'].shape}, "
                     "pixel values 0–255:"), _fig])
    return (fmnist,)


@app.cell
def _(fmnist, jax, jnp, lax, mo, optax, time):
    _X = jnp.asarray(fmnist["train_images"].reshape(-1, 784), jnp.float32) / 255.0
    _y = jnp.asarray(fmnist["train_labels"])
    fm_Xtr, fm_ytr, fm_Xval, fm_yval = _X[:55000], _y[:55000], _X[55000:], _y[55000:]
    fm_Xte = jnp.asarray(fmnist["test_images"].reshape(-1, 784), jnp.float32) / 255.0
    fm_yte = jnp.asarray(fmnist["test_labels"])

    def fm_logits(params, x):
        return x @ params["W"] + params["b"]

    def _loss(params, x, y):
        return optax.softmax_cross_entropy_with_integer_labels(fm_logits(params, x), y).mean()

    _opt = optax.adam(2e-3)
    _batch = 250
    _steps_per_epoch = 55000 // _batch

    @jax.jit
    def _epoch(params, opt_state, key):
        perm = jax.random.permutation(key, 55000).reshape(_steps_per_epoch, _batch)

        def step(carry, idx):
            p, s = carry
            loss, g = jax.value_and_grad(_loss)(p, fm_Xtr[idx], fm_ytr[idx])
            upd, s = _opt.update(g, s)
            return (optax.apply_updates(p, upd), s), loss

        (params, opt_state), losses = lax.scan(step, (params, opt_state), perm)
        return params, opt_state, losses.mean()

    def _acc(params, x, y):
        return float(jnp.mean(jnp.argmax(fm_logits(params, x), -1) == y))

    fm_params = {"W": jnp.zeros((784, 10)), "b": jnp.zeros(10)}
    _state = _opt.init(fm_params)
    _rows = []
    _t0 = time.perf_counter()
    for _e, _k in enumerate(jax.random.split(jax.random.key(6), 8)):
        fm_params, _state, _l = _epoch(fm_params, _state, _k)
        _rows.append(f"| {_e + 1} | {float(_l):.3f} | {_acc(fm_params, fm_Xtr, fm_ytr):.1%} | "
                     f"{_acc(fm_params, fm_Xval, fm_yval):.1%} |")
    _secs = time.perf_counter() - _t0
    fm_test_acc = _acc(fm_params, fm_Xte, fm_yte)
    _rows = "\n".join(_rows)
    mo.md(
        "Each epoch is one [`jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit)-compiled function: shuffle with a key, then [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan) over 220 minibatches "
        "(Module 08) with Adam (Module 05).\n\n"
        "| epoch | train loss | train acc | val acc |\n|---|---|---|---|\n" + _rows
        + f"\n\n8 epochs in **{_secs:.1f} s**. **Test accuracy: {fm_test_acc:.1%}** — measured once, now that "
        "training is done. Train and validation accuracy stay close: a linear model with 7,850 parameters is "
        "too simple to overfit 55,000 images. This is the **baseline Module 10's neural network has to beat.**"
    )
    return fm_Xte, fm_logits, fm_params, fm_test_acc, fm_yte


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "shuffle (random permutation)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What did it learn?

    Each class's weight column has 784 entries — one per pixel — so you can draw it as an image. Red
    pixels push *toward* the class, blue *away*. A linear classifier is, literally, matching each image
    against one template per class:
    """)
    return


@app.cell
def _(FASHION_CLASSES, fm_params, np, plt):
    _W = np.asarray(fm_params["W"])
    _fig, _axes = plt.subplots(1, 10, figsize=(10, 1.6))
    _lim = np.abs(_W).max()
    for _c in range(10):
        _axes[_c].imshow(_W[:, _c].reshape(28, 28), cmap="RdBu_r", vmin=-_lim, vmax=_lim)
        _axes[_c].set_title(FASHION_CLASSES[_c], fontsize=7)
        _axes[_c].axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(FASHION_CLASSES, fm_Xte, fm_logits, fm_params, fm_yte, jnp, mo, np, plt):
    fm_pred = jnp.argmax(fm_logits(fm_params, fm_Xte), -1)
    _cm = np.zeros((10, 10), int)
    np.add.at(_cm, (np.asarray(fm_yte), np.asarray(fm_pred)), 1)
    _fig, _ax = plt.subplots(figsize=(6.4, 5.4))
    _ax.imshow(_cm, cmap="Blues")
    for _i in range(10):
        for _j in range(10):
            _ax.text(_j, _i, _cm[_i, _j], ha="center", va="center", fontsize=6,
                     color="white" if _cm[_i, _j] > 500 else "black")
    _ax.set_xticks(range(10), FASHION_CLASSES, rotation=60, ha="right", fontsize=7)
    _ax.set_yticks(range(10), FASHION_CLASSES, fontsize=7)
    _ax.set(xlabel="predicted", ylabel="true", title="test-set confusion matrix")
    _fig.tight_layout()
    _worst = sorted(((int(_cm[i, j]), FASHION_CLASSES[i], FASHION_CLASSES[j]) for i in range(10) for j in range(10)
                     if i != j), reverse=True)[:3]
    _w = "; ".join(f"{t} → {p} ({n})" for n, t, p in _worst)
    mo.vstack([_fig, mo.md(f"The diagonal is correct predictions. Biggest confusions: **{_w}**. Shirts, T-shirts, "
                           "pullovers and coats share a silhouette — a pixel template can't tell them apart.")])
    return (fm_pred,)


@app.cell
def _(FASHION_CLASSES, fm_pred, fm_yte, fmnist, np, plt):
    _wrong = np.where(np.asarray(fm_pred) != np.asarray(fm_yte))[0][:10]
    _fig, _axes = plt.subplots(1, 10, figsize=(10, 1.9))
    for _a, _i in zip(_axes, _wrong):
        _a.imshow(fmnist["test_images"][_i], cmap="gray_r")
        _a.set_title(f"{FASHION_CLASSES[int(fm_yte[_i])]}\n→ {FASHION_CLASSES[int(fm_pred[_i])]}", fontsize=6)
        _a.axis("off")
    _fig.suptitle("misclassified test images (true → predicted)", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    split_quiz = mo.ui.radio(
        options={
            "To pick the learning rate and regularization strength": "tune",
            "To report the final accuracy once, after all choices are made": "report",
            "To get more training data at the end": "more",
        },
        label="**Check yourself:** what is the *test* set for?",
    )
    split_quiz
    return (split_quiz,)


@app.cell
def _(mo, split_quiz):
    _fb = {
        "report": mo.callout(mo.md("Right. Tuning happens on validation; the test set gives one honest number "
                                   "because nothing was chosen to make it look good."), kind="success"),
        "tune": mo.callout(mo.md("That's the validation set's job. Tune on test and the test score stops being "
                                 "an honest estimate — you've fit it too."), kind="warn"),
        "more": mo.callout(mo.md("Some people retrain on train+val at the end, but the test set stays untouched "
                                 "— otherwise there's nothing left to measure the final model on."), kind="warn"),
    }
    _fb.get(split_quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Your turn

    `modules/09-regression/exercises.py` has seven functions — together, a tiny ML library. The cells below
    use them on this lesson's data.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np):
    _sp = lj.attempt(ex.train_val_test_split, jax.random.key(0), 1000)
    _pf = lj.attempt(ex.polynomial_features, jnp.array([1.0, 2.0, 3.0]), 3)
    _items = [mo.md("**`train_val_test_split(key, 1000)`** — sizes should be 700 / 150 / 150:")]
    if _sp.ok:
        _sizes = [len(np.asarray(s)) for s in _sp.value]
        _items.append(mo.md(f"sizes {_sizes}, first train indices {np.asarray(_sp.value[0])[:8].tolist()}"))
    else:
        _items.append(_sp.view(mo))
    _items += [mo.md("**`polynomial_features([1, 2, 3], 3)`** — rows of 1, x, x², x³:"), _pf.view(mo)]
    mo.vstack(_items)
    return


@app.cell
def _(ex, jnp, lj, mo, poly_train_x, poly_train_y, poly_val_x, poly_val_y):
    _items = [mo.md("**`ridge_closed_form`** on your `polynomial_features` (degree 15, x scaled to [−1, 1]) — "
                    "validation MSE for a few λ:")]
    _f = lj.attempt(ex.polynomial_features, jnp.asarray(poly_train_x / 3.0), 15)
    _fv = lj.attempt(ex.polynomial_features, jnp.asarray(poly_val_x / 3.0), 15)
    if _f.ok and _fv.ok:
        _rows = []
        for _lam in (0.0, 1e-4, 1e-2, 1.0):
            _w = lj.attempt(ex.ridge_closed_form, _f.value, jnp.asarray(poly_train_y), _lam)
            if not _w.ok:
                _items.append(_w.view(mo))
                break
            _rows.append(f"| {_lam:g} | {float(jnp.mean((_fv.value @ _w.value - poly_val_y) ** 2)):.3f} |")
        else:
            _items.append(mo.md("| λ | validation MSE |\n|---|---|\n" + "\n".join(_rows)
                                + "\n\nλ = 0 is plain least squares: at degree 15 it explodes between the training "
                                "points. A little λ fixes it; too much starts to underfit again."))
    else:
        _items.append((_f if not _f.ok else _fv).view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(blob_X, blob_y, ex, jnp, lj, mo):
    _b = lj.attempt(ex.bce_with_logits, jnp.array([-200.0, 0.0, 200.0]), jnp.array([1.0, 1.0, 0.0]))
    _fit = lj.attempt(ex.logistic_fit, blob_X, blob_y, 300, 0.5)
    _items = [mo.md("**`bce_with_logits([−200, 0, 200], [1, 1, 0])`** — finite, ≈ (200 + 0.693 + 200)/3 ≈ 133.6:"),
              _b.view(mo), mo.md("**`logistic_fit`** on the two blobs from Section 5:")]
    if _fit.ok:
        _w, _bias = _fit.value
        _acc = lj.attempt(ex.accuracy, (blob_X @ _w + _bias > 0).astype(jnp.int32), blob_y.astype(jnp.int32))
        _items.append(mo.md(f"w = {[round(float(v), 3) for v in _w]}, b = {float(_bias):.3f}"))
        _items.append(mo.md(f"train accuracy via your `accuracy`: **{_acc.value:.1%}**") if _acc.ok else _acc.view(mo))
    else:
        _items.append(_fit.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "binary cross-entropy from logits", solves="bce_with_logits")
    return


@app.cell
def _(ex, fm_pred, fm_yte, lj, mo, np):
    _cm = lj.attempt(ex.confusion_matrix, fm_yte, fm_pred, 10)
    _ref = np.zeros((10, 10), int)
    np.add.at(_ref, (np.asarray(fm_yte), np.asarray(fm_pred)), 1)
    if _cm.ok:
        _match = np.array_equal(np.asarray(_cm.value), _ref)
        _out = mo.md(f"Your matrix {'**matches**' if _match else '**does not match**'} the one plotted above "
                     f"(diagonal sum {int(np.trace(np.asarray(_cm.value)))} correct out of 10,000).")
    else:
        _out = _cm.view(mo)
    mo.vstack([mo.md("**`confusion_matrix`** on the Fashion-MNIST test predictions:"), _out])
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(9))
    return


@app.cell
def _(fm_test_acc, mo):
    mo.md(f"""
    ## Further reading
    * Stanford CS229 notes, [Linear regression, logistic regression and GLMs](https://cs229.stanford.edu/main_notes.pdf)
      (chapters 1–3) — the same models, derived properly
    * *Dive into Deep Learning*, [Linear Neural Networks for Classification](https://d2l.ai/chapter_linear-classification/index.html)
    * Zalando Research, [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — why it replaced MNIST
      as the default "hello world" (it's harder) and a table of published accuracies

    **Next:** Module 10 — an MLP: add a hidden layer and a nonlinearity, and beat this lesson's
    **{fm_test_acc:.1%}** test accuracy.
    """)
    return


if __name__ == "__main__":
    app.run()
