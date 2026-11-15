import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="08 · Control flow & numerics")


@app.cell
def _():
    import marimo as mo
    import time
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from jax import lax

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    return ex, jax, jnp, lax, lj, mo, np, plt, time


@app.cell
def _(mo):
    mo.md(r"""
    # 08 · Control flow, scan, and mixed precision

    **Goal:** write loops and branches that survive [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit), and understand the number formats your GPU
    computes in well enough to know when bfloat16 is safe and when it quietly lies.

    **You will:** see why a Python `if` on a traced value fails · use [`lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html#jax.lax.cond), [`lax.switch`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html#jax.lax.switch),
    `fori_loop`, `while_loop` and [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan) · measure why `scan` beats an unrolled Python loop · read the
    bits of float32, bfloat16 and float16 · watch bfloat16 accumulation stall · catch XLA deleting your
    rounding · reproduce, in miniature, the bf16 gradient study from your deepseek-v3-jax run.

    Module 06 showed that `jit` traces your function once with abstract values. This module is about the
    consequence: control flow has to be expressed *as data the compiler can see*.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Why Python `if` breaks under `jit`

    Inside `jit`, arguments are **tracers**: placeholders with a shape and dtype but no value. Python's `if`
    needs a real `True`/`False` *now*, while tracing — and there isn't one yet. The cell below tries it.
    """)
    return


@app.cell
def _(jax, jnp, mo):
    def _relu_python_if(x):
        if x > 0:  # needs a concrete bool while tracing
            return x
        return jnp.zeros_like(x)

    try:
        jax.jit(_relu_python_if)(jnp.float32(3.0))
        _msg = "no error?!"
    except Exception as _e:  # noqa: BLE001 - show whatever JAX raises
        _msg = f"{type(_e).__name__}: {str(_e).splitlines()[0]}"
    mo.md(f"""
    `jax.jit(relu_with_python_if)(3.0)` raises:

    > `{_msg}`

    The fixes, from simplest to most general:

    * **[`jnp.where(x > 0, x, 0)`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html#jax.numpy.where)** — compute both sides, select elementwise. Best for cheap, elementwise choices.
    * **`lax.cond(pred, true_fn, false_fn, x)`** — a real branch; only one side runs on the device.
    * **`lax.switch(index, [f0, f1, f2], x)`** — the n-way version.
    * Or mark the argument **static** (`static_argnums`, Module 06) when it's a configuration value, not data.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Loops: `fori_loop`, `while_loop`, `scan`

    | Primitive | Use when | Shape of the computation |
    |---|---|---|
    | [`lax.fori_loop(lo, hi, body, init)`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html#jax.lax.fori_loop) | fixed number of steps, only the final state matters | `state = body(i, state)` |
    | [`lax.while_loop(cond, body, init)`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html#jax.lax.while_loop) | stop on a condition (convergence) | `while cond(state): state = body(state)` |
    | `lax.scan(f, carry, xs)` | walk over a sequence and **keep every step's output** | `carry, y_t = f(carry, x_t)` |

    `scan` is the workhorse of deep learning: RNNs, cumulative statistics, the layer loop in some
    transformer implementations, and diffusion samplers are all scans. It's also the only one of the three
    that [`jax.grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html#jax.grad) can differentiate in reverse mode without restrictions (`while_loop` has a data-dependent
    trip count, so reverse-mode can't know how many steps to store).

    Below: Newton's method for $\sqrt{a}$ written with `while_loop` — iterate $x \leftarrow (x + a/x)/2$
    until it stops moving. Each step roughly *doubles* the number of correct digits.
    """)
    return


@app.cell
def _(mo):
    import math

    _a = 2.0
    _x = _a
    _rows = []
    for _i in range(6):
        _x = (_x + _a / _x) / 2
        _rows.append(f"| {_i + 1} | {_x:.15f} | {abs(_x - 2**0.5):.1e} |")
    _rows = "\n".join(_rows)
    mo.md(
        "Newton's method for √2, step by step (plain Python floats, so you can see the digits):\n\n"
        "| step | x | error |\n|---|---|---|\n" + _rows
        + f"\n\nTrue value: {math.sqrt(2.0):.15f}. Five steps from a terrible start (x = 2) to "
        "machine precision — that's *quadratic convergence*."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · `lax.scan`, piece by piece

    ```python
    def step(carry, x_t):
        new_carry = ...        # state that flows to the next step
        y_t = ...              # output kept for this step
        return new_carry, y_t

    final_carry, ys = lax.scan(step, init_carry, xs)   # ys stacks every y_t along axis 0
    ```

    A tiny recurrent network is the classic example: a hidden state $h$ is updated from each input,
    $h_t = \tanh(h_{t-1} W + x_t)$, and every $h_t$ is kept. The compiler sees **one** step function and a
    trip count, not $N$ copies of the step. The next experiment shows why that matters.
    """)
    return


@app.cell
def _(jax, jnp, lax, lj, mo, time):
    _d = 32
    _W = jax.random.normal(jax.random.key(0), (_d, _d)) / jnp.sqrt(_d)

    def _rnn_scan(xs):
        def _step(h, x):
            h = jnp.tanh(h @ _W + x)
            return h, h

        return lax.scan(_step, jnp.zeros(_d), xs)[1]

    def _rnn_unrolled(xs):
        h, out = jnp.zeros(_d), []
        for t in range(xs.shape[0]):  # Python loop: traced N times → N copies in the graph
            h = jnp.tanh(h @ _W + xs[t])
            out.append(h)
        return jnp.stack(out)

    _rows = []
    scan_results = {}
    for _n in (10, 100, 1000):
        _xs = jax.random.normal(jax.random.key(1), (_n, _d))
        _line = [f"| {_n} "]
        for _name, _f in (("scan", _rnn_scan), ("unrolled", _rnn_unrolled)):
            _t0 = time.perf_counter()
            _compiled = jax.jit(_f).lower(_xs).compile()
            _compile = time.perf_counter() - _t0
            _run = lj.timeit(_compiled, _xs, warmup=1, iters=5)["median"]
            scan_results[(_name, _n)] = (_compile, _run)
            _line.append(f"| {_compile:.2f} s | {_run * 1e3:.2f} ms ")
        _rows.append("".join(_line) + "|")
    _rows = "\n".join(_rows)
    mo.md(
        "### Scan vs an unrolled Python loop (32-unit RNN, on the GPU)\n\n"
        "| N steps | scan compile | scan run | unrolled compile | unrolled run |\n|---|---|---|---|---|\n"
        + _rows
        + "\n\nThe unrolled loop's **compile time grows with N** — XLA has to optimize N copies of the step. "
        "The scan compiles once, whatever N is. The unrolled version often *runs* faster — no loop "
        "bookkeeping, and XLA can fuse work across steps — which is why `scan` has an `unroll=` argument to "
        "trade one for the other (unroll a few steps per iteration). For long sequences or deep layer stacks, `scan` is the only sane choice."
    )
    return (scan_results,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · What a float actually is

    A floating-point number is $(-1)^{\text{sign}} \times 1.\text{mantissa} \times 2^{\text{exponent} - \text{bias}}$.
    The **exponent bits** set the *range* (how big and how small); the **mantissa bits** set the
    *precision* (how many significant digits). Same 32-, 16- or 16-bit budget, split differently:
    """)
    return


@app.cell
def _(jnp, mo):
    _rows = []
    for _name, _dt, _bits in (("float32", jnp.float32, (1, 8, 23)), ("bfloat16", jnp.bfloat16, (1, 8, 7)),
                              ("float16", jnp.float16, (1, 5, 10))):
        _fi = jnp.finfo(_dt)
        _rows.append(f"| **{_name}** | {_bits[0]} · {_bits[1]} · {_bits[2]} | {float(_fi.max):.3g} | "
                     f"{float(_fi.tiny):.3g} | {float(_fi.eps):.3g} | ~{-jnp.log10(_fi.eps):.1f} |")
    _rows = "\n".join(_rows)
    mo.md(
        "| format | sign · exponent · mantissa bits | largest | smallest normal | epsilon (gap above 1.0) | decimal digits |\n"
        "|---|---|---|---|---|---|\n" + _rows + "\n\n"
        "**bfloat16** (\"brain float\") is float32 with the mantissa chopped to 7 bits: **the same range as "
        "float32**, but only 2–3 significant digits. **float16** spends more bits on precision and fewer on "
        "range: it overflows at 65,504. For training, range matters more than the last digit — gradients and "
        "activations span many orders of magnitude — which is why bf16 won and why your GPU's matrix units "
        "support it."
    )
    return


@app.cell
def _(mo):
    bits_input = mo.ui.number(value=3.14159265, step=0.001, label="a number to inspect")
    bits_input
    return (bits_input,)


@app.cell
def _(bits_input, jnp, mo, np):
    def _bits(value, dtype, uint, sizes):
        raw = np.asarray(jnp.asarray(value, dtype)).view(uint).item()
        s = format(raw, f"0{sum(sizes)}b")
        sign, exp, man = s[: sizes[0]], s[sizes[0]: sizes[0] + sizes[1]], s[sizes[0] + sizes[1]:]
        stored = float(jnp.asarray(value, dtype).astype(jnp.float32))
        return f"| {dtype.__name__} | `{sign} {exp} {man}` | {stored!r} |"

    _v = bits_input.value
    mo.md(
        f"Bits of **{_v}** (sign · exponent · mantissa) and the value each format actually stores:\n\n"
        "| format | bits | stored value |\n|---|---|---|\n"
        + _bits(_v, jnp.float32, np.uint32, (1, 8, 23)) + "\n"
        + _bits(_v, jnp.bfloat16, np.uint16, (1, 8, 7)) + "\n"
        + _bits(_v, jnp.float16, np.uint16, (1, 5, 10))
        + "\n\nNotice bfloat16's bits are exactly the **first 16 bits of float32** — converting is just "
        "rounding off the bottom half. Try 70000 (float16 overflows to `inf`) or 1e-6 (float16 loses most digits)."
    )
    return


@app.cell
def _(jnp, mo):
    _x = jnp.float32(12.0)
    _e16 = jnp.exp(_x.astype(jnp.float16))
    _ebf = jnp.exp(_x.astype(jnp.bfloat16))
    _g = jnp.float32(1e-8)
    mo.md(f"""
    ### Range in practice

    * $e^{{12}}$ = {float(jnp.exp(_x)):,.0f}. In float16: **{float(_e16)}**. In bfloat16: **{float(_ebf):,.0f}**
      (rounded, but finite). One overflow inside a softmax or a loss and training is dead.
    * A tiny gradient of 1e-8: float16 stores **{float(_g.astype(jnp.float16))}** (underflow to zero),
      bfloat16 stores **{float(_g.astype(jnp.bfloat16)):.3g}**.

    float16 training needs *loss scaling* (multiply the loss by a big constant so gradients stay above the
    underflow limit, divide afterwards). bfloat16 doesn't — it has float32's range.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Where bfloat16 bites: accumulation

    With 8 significant bits, the gap between neighbouring bfloat16 numbers between 4 and 8 is
    $2^{-5} \approx 0.031$. Add 0.01 to a running total of 4 and the exact answer, 4.01, is closer to 4 than
    to the next representable number, 4.031 — it rounds back to 4. The sum **stalls**. Below, a running
    total of 0.01s is kept in each format with `lax.scan` (so the carry really stays in that dtype):
    """)
    return


@app.cell
def _(jnp, lax, np, plt):
    def _running_sum(dtype, n):
        xs = jnp.full((n,), 0.01, dtype)
        return lax.scan(lambda c, x: (c + x, c + x), jnp.zeros((), dtype), xs)[1]

    _n = 4000
    _steps = np.arange(1, _n + 1)
    accum_final = {}
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _ax.plot(_steps, _steps * 0.01, "k--", lw=1.2, label="exact", zorder=5)
    for _dt, _c in ((jnp.float32, "C0"), (jnp.bfloat16, "C3"), (jnp.float16, "C2")):
        _vals = np.asarray(_running_sum(_dt, _n).astype(jnp.float32))
        accum_final[jnp.dtype(_dt).name] = float(_vals[-1])
        _ax.plot(_steps, _vals, color=_c, label=jnp.dtype(_dt).name)
    _ax.set(xlabel="number of 0.01s added", ylabel="running total",
            title="Accumulating 0.01 in each format")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return (accum_final,)


@app.cell
def _(accum_final, mo):
    mo.md(rf"""
    After 4,000 additions (exact total 40): bfloat16 is stuck at **{accum_final['bfloat16']:g}**, float16 at
    **{accum_final['float16']:g}** (its gap reaches 2⁻⁵ at 32, so the same thing happens there), float32 gives
    **{accum_final['float32']:.4f}**. Early on bf16 actually *overshoots* — between 2 and 4 each 0.01 rounds
    *up* to the gap of 0.0156 — then stalls for good. This is why every serious mixed-precision recipe does **reductions —
    sums, means, softmax denominators, the loss — in float32**, even when the inputs are bf16. Your
    `stable_sum` exercise is exactly that.

    ### XLA can delete your rounding

    One trap for anyone *measuring* precision. Here are two ways to round float32 to bf16 and back:
    """)
    return


@app.cell
def _(jax, jnp, lax, mo):
    _x = jnp.float32(3.14159265)

    def _astype_trip(x):
        return x.astype(jnp.bfloat16).astype(jnp.float32) - x

    def _reduce_precision(x):
        return lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7) - x

    _hlo = jax.jit(_astype_trip).lower(_x).compile().as_text()
    _line = next((ln.strip() for ln in _hlo.splitlines() if "subtract(" in ln), "")
    mo.md(f"""
    | rounding error of π via… | eager | under `jit` |
    |---|---|---|
    | `.astype(bf16).astype(f32)` | {float(_astype_trip(_x)):.3e} | **{float(jax.jit(_astype_trip)(_x)):.3e}** |
    | [`lax.reduce_precision(x, 8, 7)`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_precision.html#jax.lax.reduce_precision) | {float(_reduce_precision(_x)):.3e} | {float(jax.jit(_reduce_precision)(_x)):.3e} |

    Under `jit` the round trip vanished. The compiled program computes:

    `{_line[:160]}`

    — `x − x`. XLA's default is to *allow excess precision*: it may keep a value in a wider format than you
    asked for, because that is almost always what you want for accuracy. But if you are studying rounding,
    use `lax.reduce_precision`, which XLA must honour. (Your `bf16_relative_error` exercise needs this.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Matrix multiply: bf16 in, float32 out

    The GPU's matrix units multiply bf16 numbers but can **accumulate the dot products in float32**. In JAX
    you ask for that with [`preferred_element_type=jnp.float32`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.float32.html#jax.numpy.float32). Each product $a_{ik}b_{kj}$ is exact-ish;
    what matters is that the long sum over $k$ doesn't stall the way Section 5's did.
    """)
    return


@app.cell
def _(jax, jnp, mo):
    _rows = []
    for _k in (64, 1024, 8192):
        _ka, _kb = jax.random.split(jax.random.key(_k))
        _a = jax.random.uniform(_ka, (256, _k))
        _b = jax.random.uniform(_kb, (_k, 256))
        _ref = _a @ _b
        _bf_out = (_a.astype(jnp.bfloat16) @ _b.astype(jnp.bfloat16)).astype(jnp.float32)
        _f32_out = jnp.matmul(_a.astype(jnp.bfloat16), _b.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
        _rel = lambda y: float(jnp.linalg.norm(y - _ref) / jnp.linalg.norm(_ref))  # noqa: E731
        _rows.append(f"| {_k} | {_rel(_bf_out):.2e} | {_rel(_f32_out):.2e} |")
    _rows = "\n".join(_rows)
    mo.md(
        "Relative error vs a float32 matmul (256×K @ K×256, uniform positive inputs so the exact sums are large, never near zero):\n\n"
        "| K (length of each dot product) | bf16 output | bf16 inputs, **f32 output** |\n|---|---|---|\n"
        + _rows
        + "\n\nA **bf16 output** can be no better than bf16 itself: rounding each result to 8 bits costs "
        "~2e-3 whatever K is. With a **float32 output**, the only error left is from rounding the inputs, and "
        "those small, random errors partly cancel across a long dot product — the error *falls* as K grows. "
        "Asking for float32 output costs almost nothing and is what matrix units are built for."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · The mixed-precision recipe — straight from deepseek-v3-jax

    The model you trained keeps **master weights in float32** and computes in bfloat16. From
    `deepseek_v3.py`:

    ```python
    cast = lambda p: jax.tree.map(lambda a: a.astype(cfg.dtype), p)   # mixed precision: f32 master weights
    loss, grads = jax.value_and_grad(lambda p: loss_fn(cast(p), ...))(params)   # grads come back in f32
    ```

    The `cast` happens **inside** the function being differentiated, so gradients flow back through it to
    the float32 parameters and AdamW updates float32 numbers — tiny updates aren't rounded away. Then a few
    places deliberately go back to float32:

    | Where | Why |
    |---|---|
    | router scores `sigmoid(x.astype(f32) @ w.astype(f32))` | expert choice is sensitive to small differences |
    | attention softmax `.astype(jnp.float32)` | exponentials and a sum (Section 5!) |
    | cross-entropy `log_softmax(logits.astype(f32))` | the loss is a reduction over 8,192 classes |

    Matmuls — 95%+ of the FLOPs — stay in bf16, where the ~3× speedup from Module 00 lives.
    """)
    return


@app.cell
def _(jax, jnp, mo, np):
    def _init(key, sizes):
        keys = jax.random.split(key, len(sizes) - 1)
        return [{"w": jax.random.normal(k, (m, n)) / np.sqrt(m), "b": jnp.zeros(n)}
                for k, m, n in zip(keys, sizes[:-1], sizes[1:])]

    def _loss(params, x, y, dtype):
        h = x.astype(dtype)
        for i, layer in enumerate(params):
            h = h @ layer["w"].astype(dtype) + layer["b"].astype(dtype)
            if i < len(params) - 1:
                h = jax.nn.gelu(h)
        logp = jax.nn.log_softmax(h.astype(jnp.float32))  # loss in f32, as in the recipe
        return -jnp.mean(jnp.take_along_axis(logp, y[:, None], 1))

    _params = _init(jax.random.key(0), [256, 512, 512, 512, 64])
    _x = jax.random.normal(jax.random.key(1), (512, 256))
    _y = jax.random.randint(jax.random.key(2), (512,), 0, 64)

    def _grads(device, dtype):
        put = lambda t: jax.device_put(t, device)  # noqa: E731
        g = jax.jit(jax.grad(_loss), static_argnums=3)(put(_params), put(_x), put(_y), dtype)
        return [np.asarray(leaf, np.float64) for leaf in jax.tree.leaves(g)]

    def _rel(a, b):
        return float(np.median([np.linalg.norm(x - y) / np.linalg.norm(y) for x, y in zip(a, b)]))

    _cpu, _gpu = jax.devices("cpu")[0], jax.devices()[0]
    _ref = _grads(_cpu, jnp.float32)
    mixed_case = {
        "gpu f32 vs cpu f32": _rel(_grads(_gpu, jnp.float32), _ref),
        "cpu bf16 vs exact": _rel(_grads(_cpu, jnp.bfloat16), _ref),
        "gpu bf16 vs exact": _rel(_grads(_gpu, jnp.bfloat16), _ref),
    }
    mo.md(f"""
    ### Reproduce the case study in miniature

    A 4-layer MLP, one batch, gradients computed four ways. "Exact" means float32 on the CPU. Median
    relative error over the parameter arrays:

    | | median gradient error |
    |---|---|
    | GPU float32 vs CPU float32 | {mixed_case['gpu f32 vs cpu f32']:.1e} |
    | CPU **bf16 compute** vs exact | {mixed_case['cpu bf16 vs exact']:.1e} |
    | GPU **bf16 compute** vs exact | {mixed_case['gpu bf16 vs exact']:.1e} |

    That's the same picture as the full-size study you ran on deepseek-v3-jax: in float32 the GPU matches
    the CPU to ~1e-6 (it *is* doing the same math); in bf16 **both** backends land measurably away from the
    exact gradient (on the real model the GPU was slightly closer), and they land in
    *different* places because they round in different orders. (This small, shallow model lands under 1%
    off; the real 6-layer MoE was 3–12% per parameter group.) Training still converged: the ROCm run ended
    at val loss 3.527 vs the M5's 3.507. bf16 gradients are noisy, not wrong — and SGD is built to average
    out noise (Module 05's minibatch section made the same point).
    """)
    return (mixed_case,)


@app.cell
def _(mo):
    mp_quiz = mo.ui.radio(
        options={
            "Matrix multiplies in the attention and MLP layers": "matmul",
            "The softmax and the cross-entropy loss": "softmax",
            "Neither — everything runs in bf16 for speed": "neither",
        },
        label="**Check yourself:** in the deepseek-v3-jax recipe, which of these is computed in float32?",
    )
    mp_quiz
    return (mp_quiz,)


@app.cell
def _(mo, mp_quiz):
    _fb = {
        "softmax": mo.callout(mo.md("Right — exponentials and long sums are where bf16's 8 bits hurt, and they're "
                                    "cheap, so they go back to f32. The matmuls stay bf16."), kind="success"),
        "matmul": mo.callout(mo.md("Matmuls are the expensive part and the reason to use bf16 at all; they stay "
                                   "bf16 (ideally with f32 accumulation). Look again at Section 7's table."),
                             kind="warn"),
        "neither": mo.callout(mo.md("Section 5 is the counterexample: a bf16 running sum stalls at 2. Reductions "
                                    "and the loss need f32."), kind="warn"),
    }
    _fb.get(mp_quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Your turn

    `modules/08-control-flow-numerics/exercises.py` has seven functions. The tests run every one under
    `jax.jit`, so Python `if`/`for` on values won't pass — that's the point.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo):
    _cs = lj.attempt(ex.cumulative_sum, jnp.arange(1.0, 9.0))
    _ns = lj.attempt(ex.newton_sqrt, 2.0)
    _pw = [lj.attempt(ex.piecewise, _i, jnp.float32(0.5)) for _i in range(3)]
    _items = [mo.md("**`cumulative_sum([1..8])`** — expect 1, 3, 6, 10, 15, 21, 28, 36:"), _cs.view(mo),
              mo.md("**`newton_sqrt(2.0)`** — expect 1.4142135:"), _ns.view(mo),
              mo.md("**`piecewise(i, 0.5)` for i = 0, 1, 2** — expect 0.25, sin 0.5 ≈ 0.479, −0.5:")]
    _items += [_p.view(mo) for _p in _pw]
    mo.vstack(_items)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np, plt):
    _t = np.arange(300)
    _noisy = jnp.sin(_t / 25.0) + 0.4 * jax.random.normal(jax.random.key(3), (300,))
    _e = lj.attempt(ex.ema, _noisy, 0.08)
    if _e.ok:
        _fig, _ax = plt.subplots(figsize=(7, 2.8))
        _ax.plot(_t, _noisy, color="0.75", lw=1, label="noisy signal")
        _ax.plot(_t, np.asarray(_e.value), "C3", lw=2, label="your ema, α = 0.08")
        _ax.plot(_t, np.sin(_t / 25.0), "k--", lw=1, label="true signal")
        _ax.legend(fontsize=8)
        _ax.set(title="Exponential moving average via lax.scan")
        _fig.tight_layout()
        _out = _fig
    else:
        _out = _e.view(mo)
    mo.vstack([mo.md("**`ema`** — smoothing a noisy sine. Training dashboards smooth loss curves exactly like this:"),
               _out])
    return


@app.cell
def _(ex, jnp, lj, mo, np, plt):
    _xs = jnp.logspace(-3, 3, 2000, dtype=jnp.float32)
    _err = lj.attempt(ex.bf16_relative_error, _xs)
    if _err.ok:
        _fig, _ax = plt.subplots(figsize=(7, 2.8))
        _ax.semilogx(np.asarray(_xs), np.asarray(_err.value), lw=0.8)
        _ax.axhline(2.0**-8, color="C3", ls="--", label="2⁻⁸ (half the gap at 1.0)")
        _ax.set(xlabel="x", ylabel="relative error", title="Rounding float32 → bfloat16: your bf16_relative_error")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _out = _fig
        _note = mo.md(f"Worst case {float(jnp.max(_err.value)):.2e} ≤ 2⁻⁸ ≈ 3.9e-3, at every scale — "
                      "*relative* precision is the same from 0.001 to 1000. That's what floating point buys.")
    else:
        _out, _note = _err.view(mo), mo.md("")
    mo.vstack([_out, _note])
    return


@app.cell
def _(ex, jax, jnp, lax, lj, mo):
    _ka, _kb = jax.random.split(jax.random.key(4))
    _a = jax.random.normal(_ka, (512, 1024))
    _b = jax.random.normal(_kb, (1024, 512))
    _mm = lj.attempt(ex.mixed_precision_matmul, _a, _b)
    _data = jnp.full((100_000,), 0.01, jnp.bfloat16)
    _ss = lj.attempt(ex.stable_sum, _data)
    _naive = lax.scan(lambda c, x: (c + x, None), jnp.zeros((), jnp.bfloat16), _data)[0]
    _items = [mo.md("**`mixed_precision_matmul`** — dtype and error vs float32:")]
    if _mm.ok:
        _rel = float(jnp.linalg.norm(_mm.value - _a @ _b) / jnp.linalg.norm(_a @ _b))
        _items.append(mo.md(f"output dtype `{_mm.value.dtype}`, relative error {_rel:.2e}"))
    else:
        _items.append(_mm.view(mo))
    _items.append(mo.md(f"**`stable_sum`** of 100,000 × 0.01 stored in bf16 (exact ≈ 1000; a bf16 running "
                        f"sum gives {float(_naive):g}):"))
    _items.append(_ss.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(8))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * JAX docs: [Control flow and logical operators with JIT](https://docs.jax.dev/en/latest/control-flow.html)
      and the [`lax.scan` reference](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
    * Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (2017) — master
      weights and loss scaling, written for float16
    * Kalamkar et al., [A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322) (2019)
    * David Goldberg, [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)

    **Next:** Module 09 — linear and logistic regression from scratch: your first models trained on real data.
    """)
    return


if __name__ == "__main__":
    app.run()
