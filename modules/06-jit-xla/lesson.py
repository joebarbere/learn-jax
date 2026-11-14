import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="06 · jit and XLA")


@app.cell
def _():
    import re
    import time

    import marimo as mo
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    return ex, jax, jnp, lj, mo, plt, re, time


@app.cell
def _(mo):
    mo.md(r"""
    # 06 · `jit` and XLA: tracing, compilation, timing

    **Goal:** understand what `jax.jit` actually does to your function, so that you can predict when it
    will be fast, when it will compile again, and when it will refuse your code outright.

    **You will:** watch a function get *traced* · read the program JAX records (a *jaxpr*) and the
    compiler's output (HLO) · hit the classic "traced value in an `if`" error and fix it · count
    recompilations · time GPU code correctly (and see how wrong naive timing is) · measure what fusion
    buys on your card.

    Every `train_step` you've run — including the deepseek-v3-jax one that did 12,000 steps at 35k tok/s —
    was a jitted function. This module is how that works.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Tracing: your function runs once, with stand-ins

    `jax.jit(f)` doesn't compile Python. On the first call it runs `f` with **tracers** — placeholder
    objects that carry a shape and dtype but no values — and records every array operation they pass
    through. That recording is compiled by **XLA** into GPU kernels. Later calls with the same input
    shapes and dtypes skip Python entirely and run the compiled program.

    The cell below proves it: the function appends to a Python list (a *side effect*), and we call it
    three times.
    """)
    return


@app.cell
def _(jax, jnp, mo):
    trace_log = []

    @jax.jit
    def scaled(x):
        trace_log.append(repr(x))  # a Python side effect — only happens while tracing
        return x * 2.0 + 1.0

    _r1 = scaled(jnp.arange(3.0))
    _r2 = scaled(jnp.arange(3.0) + 10)
    _r3 = scaled(jnp.ones(3))
    mo.md(f"""
    Called **3 times**, the side effect ran **{len(trace_log)} time(s)**. What `x` looked like inside:

    `{trace_log[0]}`

    Not numbers — a *tracer* with a shape (`f32[3]`). Results were still right: `{_r2.tolist()}`.
    **Rule:** Python code in a jitted function runs at trace time; only JAX operations end up compiled.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · The recording: jaxprs

    `jax.make_jaxpr(f)(args)` shows what tracing recorded — JAX's own small intermediate language.
    Each line is one *primitive* (`sin`, `mul`, `dot_general`, …) on typed values like `f32[3]`.
    Python is gone: loops are unrolled, `if`s on static values already decided.
    """)
    return


@app.cell
def _(jax, jnp, mo):
    def gelu_ish(x):
        return 0.5 * x * (1.0 + jnp.tanh(0.79788456 * (x + 0.044715 * x**3)))

    gelu_jaxpr = jax.make_jaxpr(gelu_ish)(jnp.ones(4))
    mo.md(f"`jax.make_jaxpr(gelu_ish)(jnp.ones(4))` — the tanh approximation of GELU:\n\n```\n{gelu_jaxpr}\n```")
    return (gelu_ish,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · What XLA does with it: lowering and fusion

    The jaxpr is *lowered* to **StableHLO** (the input language of the XLA compiler), and XLA
    *compiles* that for your GPU. The big optimization is **fusion**: the eight elementwise
    operations of `gelu_ish` become *one* GPU kernel that reads `x` once and writes the result once,
    instead of eight kernels each reading and writing the whole array.
    """)
    return


@app.cell
def _(gelu_ish, jax, jnp, mo, re):
    _lowered = jax.jit(gelu_ish).lower(jnp.ones(1024))
    _stablehlo = _lowered.as_text()
    _compiled = _lowered.compile().as_text()
    _ops = [_m.group(1) for _ln in _stablehlo.splitlines()
            if (_m := re.search(r"(stablehlo\.\w+)", _ln))]
    _fusions = [ln.strip().split(", calls=")[0] for ln in _compiled.splitlines()
                if " fusion(" in ln and "ROOT" in ln]
    mo.md(f"""
    **StableHLO** (what JAX hands the compiler): {len(_ops)} operations —
    `{", ".join(sorted(set(_ops)))}`

    **Compiled HLO** for the GPU, the entry computation's result:
    ```
    {_fusions[0] if _fusions else "(no fusion found)"}
    ```
    One `fusion` instruction: the whole function became a single kernel. (`kind=kLoop` means an
    elementwise loop; matmuls show up as calls into hipBLASLt instead — the library Module 00's
    `--xla_gpu_enable_triton_gemm=false` setting tells XLA to prefer.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Measuring fusion on your card

    Without `jit`, JAX runs *eagerly*: every operation is dispatched on its own, so a chain of six
    elementwise ops on a 16M-element array makes six round trips through GPU memory. With `jit`,
    one fused kernel makes one. Elementwise ops are limited by memory bandwidth, not arithmetic,
    so this matters a lot.
    """)
    return


@app.cell
def _(jax, jnp, lj, plt):
    def chain(x):
        y = jnp.sin(x)
        y = y * 2.0
        y = y + 1.0
        y = jnp.tanh(y)
        y = y * x
        return y - 0.5

    _x = jax.random.normal(jax.random.key(0), (1 << 24,))
    _eager = lj.timeit(chain, _x, iters=10)["median"]
    _jit = lj.timeit(jax.jit(chain), _x, iters=10)["median"]
    _gb = 4 * _x.size / 1e9
    fusion_speedup = _eager / _jit
    _fig, _ax = plt.subplots(figsize=(5, 2.2))
    _ax.barh(["eager (6 kernels)", "jit (1 fused kernel)"], [_eager * 1e3, _jit * 1e3], color=["#999", "C0"])
    _ax.set_xlabel("milliseconds (lower is better)")
    _ax.set_title(f"6 elementwise ops on 16M floats: {fusion_speedup:.1f}× faster fused")
    _fig.tight_layout()
    fused_note = (f"Fused: one read + one write of {_gb:.2f} GB in {_jit * 1e3:.2f} ms ≈ "
                  f"**{2 * _gb / _jit:.0f} GB/s**, a good fraction of the card's ~288 GB/s memory bandwidth. "
                  f"Eager mode moves the same array six times over, so it can't get close.")
    _fig
    return fused_note, fusion_speedup


@app.cell
def _(fused_note, mo):
    mo.md(fused_note)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Pure functions only

    Because tracing happens once, a jitted function must be **pure**: output depends only on inputs,
    no side effects. Anything else is frozen at trace time. Here a function reads a global
    "learning rate"; we change the global afterwards…
    """)
    return


@app.cell
def _(jax, mo):
    lr_global = {"value": 0.1}

    @jax.jit
    def step(w):
        return w - lr_global["value"] * 1.0  # reads a Python global: baked in at trace time

    _before = float(step(1.0))
    lr_global["value"] = 0.5
    _after = float(step(1.0))
    mo.md(f"""
    With lr = 0.1: `step(1.0) = {_before:.2f}`. After setting the global to 0.5: `step(1.0) = {_after:.2f}`.

    **Still using 0.1** — the value was captured as a constant when the function was traced. No error,
    just a silently wrong result. The fix is always the same: pass it in as an argument.
    `print` inside a jitted function has the same trap (it prints tracers, once); use
    `jax.debug.print` to print runtime values.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Python `if` on a traced value

    A tracer has no value, so `if x > 0:` can't decide which branch to record:
    """)
    return


@app.cell
def _(jax, jnp, mo):
    def abs_with_if(x):
        if x > 0:
            return x
        return -x

    try:
        jax.jit(abs_with_if)(jnp.float32(-2.0))
        _msg = "(no error?)"
    except Exception as _e:  # noqa: BLE001 - we want to show whatever JAX raises
        _msg = f"{type(_e).__name__}: {str(_e).splitlines()[0]}"

    @jax.jit
    def abs_with_where(x):
        return jnp.where(x > 0, x, -x)  # both branches computed, then selected elementwise

    abs_static = jax.jit(abs_with_if, static_argnames="x")  # x becomes a compile-time constant
    mo.md(f"""
    ```
    {_msg}
    ```

    Two fixes, for two situations:

    | Fix | When | Here |
    |---|---|---|
    | `jnp.where(cond, a, b)` | the condition depends on *data* (per element) | `{float(abs_with_where(jnp.float32(-2.0)))}` |
    | `static_argnames=` | the value is a *setting* (sizes, flags, modes) | `{abs_static(x=-2.0)}` |

    A static argument must be hashable (an int, a string, a tuple — not an array), and **every new
    value compiles a new version**. `lax.cond`/`lax.switch` (Module 08) are the third option when you
    really need only one branch to run.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · When does it compile again?

    JAX caches compiled versions keyed by: input **shapes**, **dtypes**, and the values of **static**
    arguments. Change any of them → trace and compile again. Same ones → reuse. Watch the counter:
    """)
    return


@app.cell
def _(jax, jnp, mo):
    compiles = []

    def _norm(x, *, power):
        compiles.append((x.shape, str(x.dtype), power))
        return jnp.sum(jnp.abs(x) ** power) ** (1.0 / power)

    norm = jax.jit(_norm, static_argnames="power")
    _calls = [
        ("float32[100], power=2", jnp.ones(100), 2),
        ("float32[100], power=2 (different values)", jnp.arange(100.0), 2),
        ("float32[101], power=2", jnp.ones(101), 2),
        ("bfloat16[100], power=2", jnp.ones(100, jnp.bfloat16), 2),
        ("float32[100], power=3", jnp.ones(100), 3),
        ("float32[100], power=2 (again)", jnp.zeros(100), 2),
    ]
    _rows = []
    for _label, _x, _p in _calls:
        _n = len(compiles)
        norm(_x, power=_p)
        _rows.append(f"| `{_label}` | {'**compiled**' if len(compiles) > _n else 'cached'} |")
    _table = "\n".join(_rows)
    mo.md("| call | result |\n|---|---|\n" + _table + f"\n\n{len(compiles)} compilations for "
          f"{len(_calls)} calls. Variable-length inputs (sentences!) would compile once per length — "
          "the fix is **padding to a few bucket sizes**, which you'll implement below.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Timing GPU code: the async-dispatch trap

    JAX **dispatches asynchronously**: calling a jitted function queues work on the GPU and returns
    immediately with a *future* array. Python keeps going while the GPU computes. Great for speed —
    terrible for naive timing, which measures only the queueing.
    """)
    return


@app.cell
def _(jax, jnp, mo, time):
    _a = jax.random.normal(jax.random.key(1), (4096, 4096), jnp.bfloat16)
    _mm = jax.jit(lambda m: m @ m)
    jax.block_until_ready(_mm(_a))  # compile + warm up

    _t0 = time.perf_counter()
    for _ in range(10):
        _out = _mm(_a)
    _naive = (time.perf_counter() - _t0) / 10
    jax.block_until_ready(_out)

    _t0 = time.perf_counter()
    for _ in range(10):
        _out = jax.block_until_ready(_mm(_a))
    _real = (time.perf_counter() - _t0) / 10
    mo.md(f"""
    A 4096² bf16 matmul, timed two ways:

    | method | ms per call | implied TFLOP/s |
    |---|---|---|
    | naive `perf_counter()` around the call | {_naive * 1e3:.3f} | {2 * 4096**3 / _naive / 1e12:,.0f} (impossible) |
    | with `jax.block_until_ready` | {_real * 1e3:.3f} | {2 * 4096**3 / _real / 1e12:.1f} |

    The naive number is a fantasy (the card peaks around 28 TFLOP/s). Always block before stopping
    the clock — `learnjax.timeit` does, and so will your `time_jitted`. (Printing or converting a
    result with `float()` also blocks, which is why training loops that log every step run a bit
    slower.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Compile time vs run time

    Compiling isn't free — XLA optimizes the whole program. It pays off when the function runs many
    times: deepseek-v3-jax compiled its train step once and ran it 12,000 times. A jitted function
    called once on new shapes every time would be *slower* than eager mode.
    """)
    return


@app.cell
def _(jax, jnp, mo, time):
    def mlp_like(w1, w2, x):
        return jnp.tanh(jnp.tanh(x @ w1) @ w2).sum()

    _rows = []
    for _n in (256, 1024, 4096):
        _k1, _k2 = jax.random.split(jax.random.key(_n))
        _w = jax.random.normal(_k1, (_n, _n), jnp.bfloat16)
        _x = jax.random.normal(_k2, (256, _n), jnp.bfloat16)
        _t0 = time.perf_counter()
        _c = jax.jit(mlp_like).lower(_w, _w, _x).compile()
        _tc = time.perf_counter() - _t0
        jax.block_until_ready(_c(_w, _w, _x))
        _t0 = time.perf_counter()
        for _ in range(20):
            jax.block_until_ready(_c(_w, _w, _x))
        _tr = (time.perf_counter() - _t0) / 20
        _rows.append(f"| {_n} | {_tc * 1e3:.0f} | {_tr * 1e3:.3f} | {_tc / _tr:,.0f} |")
    mo.md("| width | compile (ms) | one run (ms) | runs to pay back the compile |\n|---|---|---|---|\n"
          + "\n".join(_rows)
          + "\n\nCompile time grows slowly with array size; run time grows fast. That's why JAX caches "
          "compiled programs, and why recompiling every step (Section 7) quietly destroys throughput.")
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "It compiles once, then reuses the compiled version": "a",
            "It compiles again, because the values changed": "b",
            "It compiles again, because the shape changed": "c",
        },
        label="**Check yourself:** a jitted `f` was called on a `float32[32, 128]` batch. You call it "
              "with a *different* `float32[32, 128]` batch. What happens?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _feedback = {
        "a": mo.callout(mo.md("Right. Compiled versions are keyed by shapes, dtypes and static "
                              "arguments — not by array values."), kind="success"),
        "b": mo.callout(mo.md("No — values never trigger recompilation; that's the whole point of "
                              "tracing with shape-only placeholders."), kind="warn"),
        "c": mo.callout(mo.md("Look again: both batches are `float32[32, 128]`. Same shape, same "
                              "dtype → cached."), kind="warn"),
    }
    _feedback.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 · Your turn

    Open `modules/06-jit-xla/exercises.py`. Seven functions, each one a tool you'll use for the rest
    of the course: inspecting a jaxpr, writing jit-safe branches, static arguments, counting
    compilations, bucketing variable-length inputs, and timing compile and run separately.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo):
    _counts = lj.attempt(ex.count_primitives, lambda x: jnp.tanh(x @ x.T) + 1.0, jnp.ones((4, 3)))
    _clip = lj.attempt(ex.jit_safe_clip, jnp.array([-2.0, 0.3, 4.0]), -1.0, 1.0)
    _pow = lj.attempt(ex.power_series_jit, jnp.array([0.5, 2.0]), n=4)
    mo.vstack([
        mo.md("**`count_primitives`** on `tanh(x @ x.T) + 1` — expect one each of "
              "`transpose`, `dot_general`, `tanh`, `add`:"),
        _counts.view(mo),
        mo.md("**`jit_safe_clip([-2, 0.3, 4], -1, 1)`** — expect `[-1, 0.3, 1]`:"),
        _clip.view(mo),
        mo.md("**`power_series_jit([0.5, 2], n=4)`** — expect `[1.9375, 31]`:"),
        _pow.view(mo),
    ])
    return


@app.cell
def _(ex, jnp, lj, mo):
    _items = []
    _counter_attempt = lj.attempt(ex.counting_jit, lambda x: x + 1)
    if _counter_attempt.ok:
        _f, _counter = _counter_attempt.value
        for _shape in [(3,), (3,), (5,), (3,), (5, 2)]:
            _f(jnp.ones(_shape))
        _items.append(mo.md(f"**`counting_jit`**: 5 calls with shapes (3,), (3,), (5,), (3,), (5,2) → "
                            f"**{_counter['traces']} traces** (expect 3)."))
    else:
        _items.append(_counter_attempt.view(mo))

    _pad = lj.attempt(ex.pad_to_bucket, jnp.arange(1.0, 12.0))
    _items.append(mo.md("**`pad_to_bucket(arange(1, 12))`** — 11 values pad to the 16 bucket:"))
    _items.append(_pad.view(mo))
    if _pad.ok:
        _mean = lj.attempt(ex.masked_mean, *_pad.value)
        _items.append(mo.md("**`masked_mean`** of it — expect 6.0, the mean of 1…11 (not of the padding):"))
        _items.append(_mean.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _w = jax.random.normal(jax.random.key(3), (1024, 1024), jnp.bfloat16)
    _t = lj.attempt(ex.time_jitted, lambda m: jnp.tanh(m @ m).sum(), _w)
    _items = [mo.md("**`time_jitted`** on a 1024² bf16 matmul + tanh:"), _t.view(mo)]
    if _t.ok:
        _items.append(mo.md(f"Compiling cost about **{_t.value['compile_seconds'] / _t.value['run_seconds']:,.0f} "
                            f"runs'** worth of time."))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(6))
    return


@app.cell
def _(fusion_speedup, mo):
    mo.md(f"""
    ## Takeaways

    * `jit` traces once with shape-only placeholders, then runs compiled XLA code. Python side effects
      happen at trace time only.
    * Branch on data with `jnp.where` (or `lax.cond`); make settings `static_argnames`.
    * New shapes, dtypes or static values → recompile. Bucket variable lengths.
    * Fusion turned six kernels into one, **{fusion_speedup:.1f}× faster** on your card.
    * Always `block_until_ready` before stopping a timer.

    ## Further reading
    * JAX docs: [Just-in-time compilation](https://docs.jax.dev/en/latest/jit-compilation.html) and
      [JAX sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
    * JAX docs: [Understanding jaxprs](https://docs.jax.dev/en/latest/jaxpr.html)
    * OpenXLA: [StableHLO specification](https://openxla.org/stablehlo/spec) and
      [XLA architecture](https://openxla.org/xla/architecture)
    * Horace He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) — why fusion wins
      for memory-bound ops.

    **Next:** Module 07 — `vmap`, pytrees and functional state: batching for free and models as data.
    """)
    return


if __name__ == "__main__":
    app.run()
