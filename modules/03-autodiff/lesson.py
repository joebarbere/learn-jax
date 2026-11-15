import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="03 · Autodiff")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    return ex, jax, jnp, lj, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 03 · Derivatives and autodiff, built from scratch

    **Goal:** understand what [`jax.grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html#jax.grad) actually computes and how, well enough that you could write it.
    Training a neural network is "nudge every parameter in the direction that lowers the loss"; the
    gradient *is* that direction, and autodiff is how we get it for millions of parameters at once.

    **You will:** see a derivative as a slope · measure why finite differences are a poor substitute ·
    see a gradient as the direction of steepest ascent · walk the chain rule through a computational graph
    · build reverse-mode autodiff yourself (a `Value` class, micrograd-style) · learn why deep learning
    uses *reverse* mode · tour `jax.grad`, [`value_and_grad`](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html#jax.value_and_grad), [`jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html#jax.jvp), [`vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html#jax.vjp), [`jacobian`](https://docs.jax.dev/en/latest/_autosummary/jax.jacobian.html#jax.jacobian) and [`hessian`](https://docs.jax.dev/en/latest/_autosummary/jax.hessian.html#jax.hessian).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · A derivative is a slope

    $f'(x_0)$ is the slope of the straight line that just touches $f$ at $x_0$ — how much $f$ changes per
    unit change in $x$, *right there*. Positive slope: going right goes up. Zero: a flat spot (a minimum,
    maximum, or saddle). Drag the point and watch the tangent.

    The slope here is computed with `jax.grad(f)(x0)`. By the end of this lesson you'll know exactly how.
    """)
    return


@app.cell
def _(mo):
    x0 = mo.ui.slider(-3.0, 3.0, step=0.1, value=1.0, label="point x₀", show_value=True)
    x0
    return (x0,)


@app.cell
def _(jax, jnp, np, plt, x0):
    def f1(x):
        return jnp.sin(2 * x) + 0.3 * x**2

    _xs = np.linspace(-3, 3, 300)
    _x0 = float(x0.value)
    _y0 = float(f1(_x0))
    _slope = float(jax.grad(f1)(_x0))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    _ax.plot(_xs, f1(_xs), label="f(x) = sin(2x) + 0.3x²")
    _ax.plot(_xs, _y0 + _slope * (_xs - _x0), "--", label=f"tangent, slope f'(x₀) = {_slope:+.3f}")
    _ax.plot([_x0], [_y0], "o", color="black")
    _ax.set_ylim(-2, 4)
    _ax.legend(loc="upper center", fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return (f1,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Finite differences, and why we don't train with them

    The definition of a derivative suggests an algorithm: nudge $x$ by a small $h$ and measure.

    * forward difference: $\dfrac{f(x+h) - f(x)}{h}$ — error shrinks like $h$
    * central difference: $\dfrac{f(x+h) - f(x-h)}{2h}$ — error shrinks like $h^2$

    So make $h$ tiny? The plot says no. For large $h$ the **truncation** error (the formula is only an
    approximation) dominates. For tiny $h$, $f(x+h)$ and $f(x)$ agree in almost every digit, so their
    difference is mostly **round-off** noise, then divided by a tiny $h$. The sweet spot for central
    differences in float64 is $h \approx 10^{-5}$, giving ~10 correct digits; in float32 (what GPUs
    like) you get maybe 3–4 digits at best.

    Two more problems: each partial derivative needs its own pair of function evaluations, so a model
    with 36 million parameters would need 72 million forward passes *per step*. Autodiff gives the exact
    gradient (up to rounding) for roughly the cost of 2–3 forward passes. Finite differences survive as a
    **test** — checking that a hand-written gradient is right.
    """)
    return


@app.cell
def _(np, plt):
    _hs = np.logspace(-13, 0, 53)
    _x = 1.0
    _true = np.cos(_x)
    _fwd64 = np.abs((np.sin(_x + _hs) - np.sin(_x)) / _hs - _true)
    _ctr64 = np.abs((np.sin(_x + _hs) - np.sin(_x - _hs)) / (2 * _hs) - _true)
    _x32, _h32 = np.float32(_x), _hs.astype(np.float32)
    _ctr32 = np.abs((np.sin(_x32 + _h32) - np.sin(_x32 - _h32)) / (2 * _h32) - _true)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.loglog(_hs, _fwd64, label="forward, float64")
    _ax.loglog(_hs, _ctr64, label="central, float64")
    _ax.loglog(_hs, np.maximum(_ctr32, 1e-17), label="central, float32")
    _ax.set_xlabel("step size h")
    _ax.set_ylabel("|error| in d/dx sin(x) at x=1")
    _ax.set_ylim(1e-12, 10)
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3, which="both")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ex, lj, mo, np):
    _a = lj.attempt(ex.central_difference, np.sin, 1.0, 1e-5)
    _items = [mo.md("**Your turn — `central_difference(sin, 1.0)`** should match cos(1) = "
                    f"{np.cos(1.0):.10f}:"), _a.view(mo)]
    if _a.ok:
        _items.append(mo.md(f"Error: **{abs(_a.value - np.cos(1.0)):.1e}** — right at the bottom of the "
                            "float64 valley in the plot above."))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Many inputs: partial derivatives and the gradient

    A loss depends on many parameters. Hold all but one fixed and take the derivative: that's a
    **partial derivative** $\partial f / \partial x_i$. Stack them into a vector and you have the
    **gradient** $\nabla f$. Geometrically it points in the direction of **steepest ascent**, and it is
    perpendicular to the contour lines (the curves where $f$ is constant). Gradient *descent* walks the
    other way: $x \leftarrow x - \eta \nabla f(x)$ (Module 05).

    The arrows below are $\nabla f$ at each grid point for
    $f(x, y) = 0.5x^2 + 2y^2 + 0.5xy + \sin(2x)$. They are short where the surface is flat and long
    where it is steep. ([`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap) evaluates the gradient at all grid points at once; Module 07 covers it.)
    """)
    return


@app.cell
def _(jax, jnp, np, plt):
    def f2(p):
        return 0.5 * p[0] ** 2 + 2 * p[1] ** 2 + 0.5 * p[0] * p[1] + jnp.sin(2 * p[0])

    _g = np.linspace(-2.5, 2.5, 120)
    _X, _Y = np.meshgrid(_g, _g)
    _Z = np.asarray(jax.vmap(f2)(jnp.stack([_X.ravel(), _Y.ravel()], axis=1))).reshape(_X.shape)
    _q = np.linspace(-2.2, 2.2, 13)
    _QX, _QY = np.meshgrid(_q, _q)
    _pts = jnp.stack([_QX.ravel(), _QY.ravel()], axis=1)
    _G = np.asarray(jax.vmap(jax.grad(f2))(_pts))
    _fig, _ax = plt.subplots(figsize=(5.2, 4.4))
    _cs = _ax.contour(_X, _Y, _Z, levels=18, cmap="viridis", linewidths=0.8)
    _ax.quiver(_QX.ravel(), _QY.ravel(), _G[:, 0], _G[:, 1], color="crimson", angles="xy",
               scale_units="xy", scale=12, width=0.004)
    _ax.set_aspect("equal")
    _ax.set_title("contours of f and its gradient (arrows)", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: a gradient checker

    `grad_check(f, grad_f, x)` compares an analytic gradient with central differences, coordinate by
    coordinate, and returns the worst *relative* error. Below it judges two hand-derived gradients of
    $f(x) = \sum_i \sin(x_i)\,x_i^2$ — one correct, one that forgot half of the product rule.
    """)
    return


@app.cell
def _(ex, lj, mo, np):
    def _f(x):
        return float(np.sum(np.sin(x) * x**2))

    def _good(x):
        return np.cos(x) * x**2 + 2 * x * np.sin(x)

    def _buggy(x):
        return np.cos(x) * x**2

    _x = np.array([0.3, -1.2, 2.0, 0.9])
    _ok = lj.attempt(ex.grad_check, _f, _good, _x)
    _bad = lj.attempt(ex.grad_check, _f, _buggy, _x)
    _items = [_ok.view(mo, show_value=False)]
    if _ok.ok and _bad.ok:
        _items = [mo.md(f"""
    | gradient | worst relative error | verdict |
    |---|---|---|
    | correct: `cos(x)·x² + 2x·sin(x)` | {_ok.value:.1e} | {'✅ right' if _ok.value < 1e-7 else '❌'} |
    | buggy: `cos(x)·x²` | {_bad.value:.1e} | {'❌ bug caught' if _bad.value > 1e-3 else '?'} |
    """)]
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · The chain rule, walked through a graph

    Real functions are compositions of simple ones. For a single neuron with squared error:

    $$z = w x + b, \qquad a = \tanh(z), \qquad L = (a - y)^2$$

    The **chain rule** says derivatives multiply along the path:
    $\dfrac{\partial L}{\partial w} = \dfrac{\partial L}{\partial a}\cdot\dfrac{\partial a}{\partial z}
    \cdot\dfrac{\partial z}{\partial w}$. Draw the computation as a graph — inputs at the bottom, $L$ at
    the top — and autodiff is two sweeps over it:

    1. **Forward:** compute every node's value, remembering the graph.
    2. **Backward:** start with $\partial L/\partial L = 1$ at the top and walk down. Each node takes the
       gradient arriving from above, multiplies by its own *local* derivative, and passes it to its
       inputs. That's **backpropagation**.

    Here are both sweeps for $w=0.5,\ x=2,\ b=-0.3,\ y=0.8$:
    """)
    return


@app.cell
def _(jax, jnp, mo, np):
    _w, _x, _b, _y = 0.5, 2.0, -0.3, 0.8
    _z = _w * _x + _b
    _a = np.tanh(_z)
    _L = (_a - _y) ** 2
    _dL_da = 2 * (_a - _y)
    _dL_dz = _dL_da * (1 - _a**2)
    _dL_dw, _dL_db = _dL_dz * _x, _dL_dz
    _jw, _jb = jax.grad(lambda w, b: (jnp.tanh(w * _x + b) - _y) ** 2, argnums=(0, 1))(_w, _b)
    mo.md(f"""
    | node | forward value | local derivative | gradient ∂L/∂node (backward) |
    |---|---|---|---|
    | $L = (a-y)^2$ | {_L:.5f} | — | 1 |
    | $a = \\tanh(z)$ | {_a:.5f} | ∂L/∂a = 2(a−y) | {_dL_da:+.5f} |
    | $z = wx + b$ | {_z:.5f} | ∂a/∂z = 1−a² = {1 - _a**2:.5f} | {_dL_dz:+.5f} |
    | $w$ | {_w} | ∂z/∂w = x = {_x} | **{_dL_dw:+.5f}** |
    | $b$ | {_b} | ∂z/∂b = 1 | **{_dL_db:+.5f}** |

    `jax.grad` agrees: ∂L/∂w = **{float(_jw):+.5f}**, ∂L/∂b = **{float(_jb):+.5f}**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Build it: reverse-mode autodiff in ~80 lines

    Andrej Karpathy's *micrograd* shows that the heart of PyTorch and JAX autodiff fits on a page. You'll
    write the same thing in `exercises.py` as a `Value` class:

    * A `Value` wraps one number (`.data`) and has a `.grad`, initially 0.
    * Every operation (`+`, `*`, `**`, `tanh`, `exp`, …) returns a **new** `Value` that records its inputs
      (`_prev`) and a closure `_backward` that applies *that operation's* local derivative:
      for `out = a * b`, `a.grad += b.data * out.grad` and `b.grad += a.data * out.grad`.
    * `backward()` sorts the graph **topologically** (every node after its inputs), sets the output's
      grad to 1, and calls `_backward` on each node in **reverse** order — so a node's gradient is
      complete before it is passed down.
    * Gradients **accumulate** with `+=`: if a value feeds two places, both contribute (multivariate
      chain rule). Using `=` here is the classic bug; the tests check for it.

    Build `-`, `/` and negation out of `+`, `*` and `**` — they come for free.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    def _run():
        V = ex.Value
        w, x, b, y = V(0.5), V(2.0), V(-0.3), V(0.8)
        loss = ((w * x + b).tanh() - y) ** 2
        loss.backward()
        return loss, w, b

    _r = lj.attempt(_run)
    _r.name = "Value"
    if _r.ok:
        _loss, _w, _b = _r.value
        _jw, _jb = jax.grad(lambda w, b: (jnp.tanh(w * 2.0 + b) - 0.8) ** 2, argnums=(0, 1))(0.5, -0.3)
        _out = mo.md(f"""
    Your `Value` computed the same neuron: loss = {_loss.data:.5f},
    ∂L/∂w = **{_w.grad:+.5f}** (JAX: {float(_jw):+.5f}), ∂L/∂b = **{_b.grad:+.5f}**
    (JAX: {float(_jb):+.5f}). {'✅ match' if abs(_w.grad - float(_jw)) < 1e-4 else '❌ mismatch'}

    That is backpropagation — the same algorithm that trains every model in this course, just on
    scalars instead of arrays.
    """)
    else:
        _out = _r.view(mo)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: the same neuron, by hand, with vectors

    `neuron_grad(w, b, x, y)` uses vectors $w, x \in \mathbb{R}^n$ — derive $\partial L/\partial w$ with the
    chain rule (it is the scalar $\partial L/\partial z$ times the vector $x$) and compare with JAX:
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np):
    _w = np.array([0.5, -0.4, 0.2])
    _x = np.array([1.0, 2.0, -1.5])
    _a = lj.attempt(ex.neuron_grad, _w, -0.1, _x, 0.8)
    _jw, _jb = jax.grad(lambda w, b: (jnp.tanh(w @ _x + b) - 0.8) ** 2, argnums=(0, 1))(jnp.asarray(_w), -0.1)
    if _a.ok:
        _out = mo.md(f"yours: dw = `{np.round(np.asarray(_a.value[0]), 5)}`, db = `{_a.value[1]:.5f}`  \n"
                     f"JAX: dw = `{np.round(np.asarray(_jw), 5)}`, db = `{float(_jb):.5f}`")
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Forward mode vs reverse mode

    The chain rule is a product of Jacobians, and you can multiply it from either end:

    * **Forward mode** (JVP, Jacobian-vector product) pushes one *input* direction through the program
      alongside the values. One pass gives the derivative of every output with respect to **one input
      direction**. Full Jacobian of $f:\mathbb{R}^n\to\mathbb{R}^m$: **n passes**.
    * **Reverse mode** (VJP, vector-Jacobian product) runs forward, then pulls one *output* direction
      back — what your `Value` does. One pass gives the derivative of **one output** with respect to
      every input. Full Jacobian: **m passes**.

    Training has one scalar loss ($m = 1$) and millions of parameters ($n$ huge), so reverse mode gets the
    whole gradient in one backward pass — about 2–3× the cost of the forward pass, *whatever n is*. The
    price: it must store intermediate values from the forward pass (activation memory, Module 11).

    Measured below on the GPU for $f(x) = \sum \tanh(Wx)$ with $x \in \mathbb{R}^n$: `jax.grad` (reverse)
    vs [`jax.jacfwd`](https://docs.jax.dev/en/latest/_autosummary/jax.jacfwd.html#jax.jacfwd) (forward, n JVPs batched together).
    """)
    return


@app.cell
def _(mo):
    n_dim = mo.ui.slider(steps=[64, 256, 1024, 2048], value=1024, label="number of inputs n", show_value=True)
    n_dim
    return (n_dim,)


@app.cell
def _(jax, jnp, lj, mo, n_dim):
    _n = n_dim.value
    _W = jax.random.normal(jax.random.key(0), (_n, _n)) / jnp.sqrt(_n)
    _x = jnp.ones(_n)

    def _f(x):
        return jnp.sum(jnp.tanh(_W @ x))

    _t_fwd_pass = lj.timeit(jax.jit(_f), _x)["median"]
    _t_rev = lj.timeit(jax.jit(jax.grad(_f)), _x)["median"]
    _t_jacfwd = lj.timeit(jax.jit(jax.jacfwd(_f)), _x, iters=5)["median"]
    _same = bool(jnp.allclose(jax.grad(_f)(_x), jax.jacfwd(_f)(_x), atol=1e-4))
    mo.md(f"""
    | n = {_n} | time | vs. one forward pass |
    |---|---|---|
    | forward pass `f(x)` | {_t_fwd_pass * 1e3:.3f} ms | 1× |
    | reverse mode `jax.grad(f)` | {_t_rev * 1e3:.3f} ms | {_t_rev / _t_fwd_pass:.1f}× |
    | forward mode `jax.jacfwd(f)` | {_t_jacfwd * 1e3:.3f} ms | {_t_jacfwd / _t_fwd_pass:.1f}× |

    Same answer either way: {'✅' if _same else '❌'}. Forward mode's cost grows with n (it batches n JVPs,
    turning a matrix-vector product into a matrix-matrix one); reverse mode's doesn't. At small n,
    everything is dominated by launch overhead (Module 00), so the gap only shows as n grows.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Reverse mode: one backward pass gives d(loss)/d(every parameter)": "a",
            "Forward mode: one forward pass per parameter is cheap on a GPU": "b",
            "Finite differences: they are exact if h is small enough": "c",
        },
        label="**Check yourself:** a model has 36M parameters and one scalar loss. What computes the full "
              "gradient most efficiently?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _feedback = {
        "a": mo.callout(mo.md("Right — one output, many inputs: reverse mode. It costs a small constant "
                              "multiple of the forward pass regardless of parameter count."), kind="success"),
        "b": mo.callout(mo.md("That would be 36 million forward passes per training step. Forward mode wins "
                              "when there are few inputs and many outputs."), kind="warn"),
        "c": mo.callout(mo.md("No — section 2 showed round-off destroys accuracy as h shrinks, and you'd "
                              "need 72 million evaluations per step."), kind="warn"),
    }
    _feedback.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · The `jax.grad` toolbox

    `jax.grad` is a *function transformation*: it takes a Python function and returns a new function that
    computes the gradient. Because it transforms functions rather than recording a tape at runtime, it
    composes with everything else in JAX ([`jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit), `vmap`, even `grad` of `grad`).
    """)
    return


@app.cell
def _(jax, jnp, mo):
    def _loss(params, x, y):
        pred = jnp.tanh(params["w"] @ x + params["b"])
        return jnp.mean((pred - y) ** 2)

    _params = {"w": jnp.array([[0.5, -0.4], [0.1, 0.3]]), "b": jnp.array([0.0, -0.1])}
    _x, _y = jnp.array([1.0, 2.0]), jnp.array([0.5, -0.5])
    _val, _grads = jax.value_and_grad(_loss)(_params, _x, _y)
    _gx = jax.grad(_loss, argnums=1)(_params, _x, _y)

    def _loss_aux(params, x, y):
        pred = jnp.tanh(params["w"] @ x + params["b"])
        return jnp.mean((pred - y) ** 2), pred

    (_v2, _pred), _ = jax.value_and_grad(_loss_aux, has_aux=True)(_params, _x, _y)
    _d2 = jax.grad(jax.grad(jnp.sin))(1.0)
    mo.md(f"""
    | call | result |
    |---|---|
    | `jax.value_and_grad(loss)(params, x, y)` | loss = {float(_val):.4f}, plus grads in one pass |
    | gradient of a **dict** of params (a *pytree*) | same structure back: `w` grad `{[[round(v, 4) for v in r] for r in _grads['w'].tolist()]}`, `b` grad `{[round(v, 4) for v in _grads['b'].tolist()]}` |
    | `jax.grad(loss, argnums=1)` — w.r.t. the input `x` | `{[round(v, 4) for v in _gx.tolist()]}` |
    | `has_aux=True` — return extras alongside the loss | predictions `{[round(v, 4) for v in _pred.tolist()]}` |
    | [`jax.grad(jax.grad(jnp.sin))(1.0)`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sin.html#jax.numpy.sin) — second derivative | {float(_d2):.4f} (= −sin 1) |

    The dict case matters most: a model's parameters are a nested structure of arrays, and `jax.grad`
    returns gradients with the *same* structure, ready to subtract (Module 07 generalizes this).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · `jvp`, `vjp`, `jacobian`, `hessian`

    Under `jax.grad` are the two primitives from section 6, available directly:

    * `jax.jvp(f, (x,), (v,))` → $(f(x),\ J v)$ — forward mode, one input direction $v$.
    * `y, vjp_fn = jax.vjp(f, x)`, then `vjp_fn(u)` → $u^\top J$ — reverse mode, one output direction $u$.
    * `jax.jacobian` (= `jacrev`) and `jax.jacfwd` build the full matrix from many VJPs / JVPs.
    * `jax.hessian(f)` is the matrix of second derivatives — the curvature that Adam approximates
      cheaply (Module 05).
    """)
    return


@app.cell
def _(jax, jnp, mo):
    def g(x):
        return jnp.stack([x[0] * x[1], jnp.sin(x[2]), jnp.sum(x**2), x[1] - x[0]])

    _x = jnp.array([0.5, -1.0, 2.0])
    _J = jax.jacobian(g)(_x)
    _v = jnp.array([1.0, 0.0, 0.0])
    _, _Jv = jax.jvp(g, (_x,), (_v,))
    _y, _vjp = jax.vjp(g, _x)
    _u = jnp.array([0.0, 0.0, 1.0, 0.0])
    (_uJ,) = _vjp(_u)
    _H = jax.hessian(lambda x: jnp.sum(x**2) + x[0] * x[1])(_x)

    def _fmt(a):
        return "<br>".join(str([round(float(v), 3) for v in row]) for row in jnp.atleast_2d(a))

    mo.md(f"""
    For $g:\\mathbb{{R}}^3 \\to \\mathbb{{R}}^4$, $g(x) = (x_0x_1,\\ \\sin x_2,\\ \\sum x_i^2,\\ x_1 - x_0)$ at
    $x = (0.5, -1, 2)$:

    | | value |
    |---|---|
    | `jax.jacobian(g)(x)` — 4×3 | {_fmt(_J)} |
    | `jvp` with $v = e_0$ → first **column** of J | {_fmt(_Jv)} |
    | `vjp` with $u = e_2$ → third **row** of J | {_fmt(_uJ)} |
    | `jax.hessian` of $\\sum x_i^2 + x_0x_1$ | {_fmt(_H)} |
    """)
    return (g,)


@app.cell
def _(ex, g, jax, jnp, lj, mo):
    _x = jnp.array([0.5, -1.0, 2.0])
    _a = lj.attempt(ex.jacobian_via_vjp, g, _x)
    if _a.ok:
        _match = bool(jnp.allclose(_a.value, jax.jacobian(g)(_x), atol=1e-5))
        _out = mo.md(f"**Stretch — `jacobian_via_vjp(g, x)`:** shape `{tuple(_a.value.shape)}`, matches "
                     f"`jax.jacobian`: {'✅' if _match else '❌'}. You just built `jacrev`: one VJP per output "
                     "row, all at once with `vmap`.")
    else:
        _out = mo.vstack([mo.md("**Stretch — `jacobian_via_vjp`:** build the Jacobian from VJPs."),
                          _a.view(mo)])
    _out
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(3))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Andrej Karpathy, [The spelled-out intro to neural networks and backpropagation: building
      micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — the video version of section 5.
    * JAX docs, [The Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
      — jvp/vjp, Hessians, and more.
    * Baydin et al., [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
      — forward vs reverse mode, properly.
    * 3Blue1Brown, [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) — the chain rule
      picture for a neural network.

    **Next:** Module 04 — probability, softmax and cross-entropy: the loss function that trains every
    language model.
    """)
    return


if __name__ == "__main__":
    app.run()
