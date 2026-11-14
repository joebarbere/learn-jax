import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="01 · Arrays")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    return ex, jax, jnp, lj, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 01 · Arrays: NumPy → jax.numpy, shapes, broadcasting, dtypes

    **Goal:** read and write array code fluently. Every neural network in this course is a handful of
    array operations; if you can predict the **shape** of every intermediate result, you can debug
    almost anything.

    **You will:** inspect shapes and axes · index, reshape and reduce along an axis · learn the three
    broadcasting rules (and predict shapes before running code) · see how JAX picks dtypes · meet JAX's
    immutable arrays and `.at[]` · move data between NumPy, the CPU and the GPU · write `einsum`
    · time a Python loop against a vectorized version on the GPU.

    If you know NumPy, `jax.numpy` will feel familiar: same function names, same semantics, with three
    differences you'll meet below (immutability, out-of-bounds indexing, default dtypes).
    """)
    return


@app.cell
def _(jnp, mo):
    _examples = {
        "scalar `jnp.array(3.0)`": jnp.array(3.0),
        "vector `jnp.arange(5)`": jnp.arange(5),
        "matrix `jnp.ones((3, 4))`": jnp.ones((3, 4)),
        "image batch `jnp.zeros((32, 28, 28, 1))`": jnp.zeros((32, 28, 28, 1)),
        "token batch `jnp.zeros((32, 256), jnp.int32)`": jnp.zeros((32, 256), jnp.int32),
    }
    _rows = "\n    ".join(
        f"| {k} | `{v.shape}` | {v.ndim} | {v.size:,} | `{v.dtype}` |" for k, v in _examples.items()
    )
    mo.md(f"""
    ## 1 · Shape, ndim, size, dtype

    An array is a block of numbers plus a **shape** (the length of each axis). `ndim` is the number of
    axes, `size` the total count of numbers, `dtype` the number type.

    | array | shape | ndim | size | dtype |
    |---|---|---|---|---|
    {_rows}

    By convention the **first axis is the batch**: 32 images, or 32 sequences of 256 tokens. Most bugs
    in ML code are an axis in the wrong place, so get used to printing `.shape`.
    """)
    return


@app.cell
def _(jnp, mo):
    grid = jnp.arange(20).reshape(4, 5)
    _g = str(grid).replace("\n", "\n    ")  # keep the indentation mo.md dedents
    mo.md(f"""
    ## 2 · Indexing and slicing

    `grid = jnp.arange(20).reshape(4, 5)`:

    ```
    {_g}
    ```

    | expression | result | shape |
    |---|---|---|
    | `grid[1, 2]` | `{grid[1, 2]}` | `{grid[1, 2].shape}` |
    | `grid[1]` (a row) | `{grid[1]}` | `{grid[1].shape}` |
    | `grid[:, 2]` (a column) | `{grid[:, 2]}` | `{grid[:, 2].shape}` |
    | `grid[1:3, ::2]` | `{grid[1:3, ::2].tolist()}` | `{grid[1:3, ::2].shape}` |
    | `grid[-1]` (last row) | `{grid[-1]}` | `{grid[-1].shape}` |
    | `grid[grid % 2 == 0]` (boolean mask) | `{grid[grid % 2 == 0]}` | `{grid[grid % 2 == 0].shape}` |
    | `grid[jnp.array([0, 3])]` (fancy index) | `{grid[jnp.array([0, 3])].tolist()}` | `{grid[jnp.array([0, 3])].shape}` |

    **JAX difference #1:** out-of-bounds indices don't raise. `grid[10, 0]` gives
    `{grid[10, 0]}` — the index is *clamped* to the last row. On a GPU there's no cheap way to raise an
    error mid-kernel, so JAX picks a defined behaviour instead. Be careful with your indices.
    """)
    return (grid,)


@app.cell
def _(jnp, mo):
    _imgs = jnp.zeros((32, 28, 28, 3))
    mo.md(f"""
    ## 3 · Reshape, transpose, and moving axes

    Reshaping reinterprets the same numbers with a new shape (sizes must multiply to the same total);
    transposing reorders axes.

    | operation on a batch of images `x.shape = {_imgs.shape}` (batch, height, width, channels) | new shape |
    |---|---|
    | `x.reshape(32, -1)` — flatten each image to a vector (`-1` = "work it out") | `{_imgs.reshape(32, -1).shape}` |
    | `x.transpose(0, 3, 1, 2)` — channels-first, as PyTorch likes | `{_imgs.transpose(0, 3, 1, 2).shape}` |
    | `jnp.moveaxis(x, -1, 1)` — same thing, by naming one axis | `{jnp.moveaxis(_imgs, -1, 1).shape}` |
    | `x[None]` or `x[jnp.newaxis]` — add a leading axis of size 1 | `{_imgs[None].shape}` |
    | `x[..., 0]` — `...` means "all the leading axes" | `{_imgs[..., 0].shape}` |
    | `x.reshape(4, 8, 28, 28, 3)` — split the batch into 4 groups of 8 | `{_imgs.reshape(4, 8, 28, 28, 3).shape}` |
    """)
    return


@app.cell
def _(mo):
    axis_pick = mo.ui.radio(options=["axis=None", "axis=0", "axis=1", "axis=1, keepdims=True"],
                            value="axis=0", label="**Reduce `grid.sum(...)` along:**", inline=True)
    axis_pick
    return (axis_pick,)


@app.cell
def _(axis_pick, grid, mo):
    _results = {
        "axis=None": grid.sum(),
        "axis=0": grid.sum(axis=0),
        "axis=1": grid.sum(axis=1),
        "axis=1, keepdims=True": grid.sum(axis=1, keepdims=True),
    }
    _r = _results[axis_pick.value]
    _g = str(grid).replace("\n", "\n    ")
    _rs = str(_r).replace("\n", "\n    ")
    _explain = {
        "axis=None": "sums everything into one number.",
        "axis=0": "collapses axis 0 (the rows): you get one sum **per column**, shape (5,).",
        "axis=1": "collapses axis 1 (the columns): one sum **per row**, shape (4,).",
        "axis=1, keepdims=True": "same sums, but keeps the collapsed axis as size 1: shape (4, 1). That "
                                 "shape broadcasts back against `grid`, e.g. `grid / grid.sum(1, keepdims=True)`.",
    }
    mo.md(f"""
    ## 4 · Reductions along an axis

    The `axis` you pass is the axis that **disappears**. `grid` is 4×5:

    ```
    {_g}
    ```

    `grid.sum({axis_pick.value})` {_explain[axis_pick.value]}

    ```
    {_rs}
    ```
    shape `{_r.shape}`. The same rule holds for `mean`, `max`, `argmax`, `std`, `logsumexp` …
    """)
    return


@app.cell
def _(jnp, mo):
    _a = jnp.ones((8, 1, 3))
    _b = jnp.ones((5, 1))
    mo.md(f"""
    ## 5 · Broadcasting

    Broadcasting lets arrays of different shapes combine *without copying*: `x - x.mean(0)` subtracts a
    row of means from every row. The rules, applied to shapes **right-aligned**:

    1. Pad the shorter shape on the left with 1s.
    2. Per axis, sizes match if they're **equal**, or if one of them is **1** (it stretches).
    3. Anything else is an error.

    | | axis 0 | axis 1 | axis 2 |
    |---|---|---|---|
    | `a.shape = (8, 1, 3)` | 8 | 1 | 3 |
    | `b.shape = (5, 1)` → padded | *1* | 5 | 1 |
    | result | 8 | 5 | 3 |

    JAX agrees: `(a + b).shape` = `{(_a + _b).shape}`.

    Common patterns: `x[:, None] - y[None, :]` makes an (n, m) table of all pairs;
    `x / x.sum(-1, keepdims=True)` normalizes rows; `image * mask[..., None]` applies a 2-D mask to every
    channel.
    """)
    return


@app.cell
def _(mo):
    shape_quiz = mo.ui.radio(
        options=["(4, 3)", "(3, 4)", "(4, 4, 3)", "error"],
        label="**Predict the shape:** `jnp.ones((4, 1)) * jnp.ones((3,))`",
    )
    shape_quiz
    return (shape_quiz,)


@app.cell
def _(mo, shape_quiz):
    _ok = mo.callout(mo.md("Right: (4, 1) and (3,) → pad to (1, 3) → axis 0: 4 vs 1 → 4; axis 1: 1 vs 3 → 3. "
                           "Result (4, 3) — an outer product."), kind="success")
    _no = mo.callout(mo.md("Not quite. Right-align: `(4, 1)` over `(1, 3)` (padded). Axis by axis, a 1 "
                           "stretches to match the other size."), kind="warn")
    {None: mo.md("")}.get(shape_quiz.value, _ok if shape_quiz.value == "(4, 3)" else _no)
    return


@app.cell
def _(jnp, mo):
    _bf = jnp.ones(3, jnp.bfloat16)
    _f32 = jnp.ones(3, jnp.float32)
    _i32 = jnp.ones(3, jnp.int32)
    _i8 = jnp.ones(3, jnp.int8)
    mo.md(f"""
    ## 6 · dtypes and type promotion

    | expression | result dtype | why |
    |---|---|---|
    | `jnp.array([1.0, 2.0])` | `{jnp.array([1.0, 2.0]).dtype}` | **JAX difference #2:** default float is 32-bit (NumPy's is 64-bit) |
    | `jnp.array([1, 2])` | `{jnp.array([1, 2]).dtype}` | default int is 32-bit too |
    | `int32 + float32` | `{(_i32 + _f32).dtype}` | mixing ints and floats gives a float |
    | `bfloat16 + float32` | `{(_bf + _f32).dtype}` | wider float wins |
    | `bfloat16 * 2.0` (Python float) | `{(_bf * 2.0).dtype}` | Python scalars are **weakly typed**: they adopt the array's dtype |
    | `int8 + 1000` | `{(_i8 + 1000).dtype}` | weak too — and 1000 doesn't fit in int8, so watch for overflow |
    | `bfloat16 * jnp.float32(2.0)` | `{(_bf * jnp.float32(2.0)).dtype}` | a *typed* scalar promotes the result |

    Weak types matter in mixed-precision training: `x_bf16 * 0.5` stays bf16, which is what you want.
    64-bit floats are off unless you set `JAX_ENABLE_X64=1`; GPUs are much slower at them anyway.
    """)
    return


@app.cell
def _(jnp, mo):
    _v = jnp.zeros(5)
    try:
        _v[0] = 1.0
        _err = "no error?"
    except TypeError as e:
        _err = str(e).split("\n")[0][:120]
    _set = _v.at[0].set(1.0)
    _add = _v.at[jnp.array([1, 1, 3])].add(1.0)
    mo.md(f"""
    ## 7 · Immutability and `.at[]`

    **JAX difference #3:** arrays can't be modified in place. `v[0] = 1.0` raises:

    > `TypeError: {_err}`

    Instead, `.at[...]` returns a **new** array:

    | expression | result |
    |---|---|
    | `v` | `{_v}` |
    | `v.at[0].set(1.0)` | `{_set}` |
    | `v.at[jnp.array([1, 1, 3])].add(1.0)` | `{_add}` — repeated indices accumulate |
    | `v` afterwards | `{_v}` — unchanged |

    Why? JAX transforms functions (`grad`, `jit`, `vmap`), and that's only safe for *pure* functions
    that don't mutate their inputs. Inside `jit`, XLA turns `.at[].set` into an in-place update when it
    can, so you don't pay for the copy.
    """)
    return


@app.cell
def _(jax, jnp, mo, np):
    _host = np.arange(4.0)
    _dev = jnp.asarray(_host)
    _back = np.asarray(_dev)
    _cpu = jax.device_put(_dev, jax.devices("cpu")[0])
    mo.md(f"""
    ## 8 · Where the data lives: NumPy, CPU, GPU

    | object | type | lives on |
    |---|---|---|
    | `np.arange(4.0)` | `{type(_host).__name__}` | host RAM, `{_host.dtype}` |
    | `jnp.asarray(that)` | `{type(_dev).__name__}` | `{_dev.devices()}` (copied to the GPU, now `{_dev.dtype}`) |
    | `np.asarray(that)` | `{type(_back).__name__}` | host RAM (copied back) |
    | `jax.device_put(x, jax.devices('cpu')[0])` | `{type(_cpu).__name__}` | `{_cpu.devices()}` |

    JAX runs an operation where its inputs live. Copies between host and GPU go over PCIe (your card
    has a 4.0 ×8 link, ~16 GB/s), which is slow compared to the GPU's own memory (~288 GB/s): load data
    once, keep it on the device, and pull back only small results like the loss.
    """)
    return


@app.cell
def _(jnp, mo):
    _a = jnp.ones((2, 3))
    _b = jnp.ones((3, 4))
    _x = jnp.ones((8, 16, 64))
    mo.md(f"""
    ## 9 · `einsum`: shapes as a sentence

    `jnp.einsum` names each axis with a letter. Letters that appear in the inputs but not the output are
    summed over. It's often the clearest way to write tensor code — attention in Module 15 is two einsums.

    | einsum | means | result shape |
    |---|---|---|
    | `"ij,jk->ik"` | matrix multiply | `{jnp.einsum("ij,jk->ik", _a, _b).shape}` |
    | `"ij->ji"` | transpose | `{jnp.einsum("ij->ji", _a).shape}` |
    | `"ij->i"` | row sums | `{jnp.einsum("ij->i", _a).shape}` |
    | `"i,i->"` | dot product | `{jnp.einsum("i,i->", _a[0], _a[0]).shape}` |
    | `"btd,bsd->bts"` on `(8,16,64)` twice | every pair of positions' similarity, per batch | `{jnp.einsum("btd,bsd->bts", _x, _x).shape}` |
    """)
    return


@app.cell
def _(mo):
    n_points = mo.ui.slider(steps=[64, 128, 256, 512], value=128, show_value=True,
                            label="number of points n (distances to n points each)")
    mo.vstack([mo.md(r"""
    ## 10 · GPU experiment: loops vs vectorized

    Compute all pairwise squared distances between n points in 64 dimensions. The *loop* version
    issues one small GPU operation per row (n launches); the *vectorized* version is one broadcasted
    expression compiled by `jit`. Same math, very different speed.
    """), n_points])
    return (n_points,)


@app.cell
def _(jax, jnp, lj, mo, n_points):
    _n = n_points.value
    _pts = jax.random.normal(jax.random.key(0), (_n, 64))

    def _loop(p):
        return jnp.stack([jnp.sum((p - p[i]) ** 2, axis=-1) for i in range(p.shape[0])])

    _vec = jax.jit(lambda p: jnp.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
    _t_loop = lj.timeit(_loop, _pts, warmup=1, iters=3)["median"]
    _t_vec = lj.timeit(_vec, _pts, iters=10)["median"]
    mo.md(f"""
    | n = {_n} | time |
    |---|---|
    | Python loop over rows | **{_t_loop * 1e3:.1f} ms** |
    | vectorized + `jit` | **{_t_vec * 1e3:.3f} ms** |
    | speed-up | **{_t_loop / _t_vec:,.0f}×** |

    The loop's cost is almost all *overhead* — Python and kernel launches — not arithmetic. On a GPU,
    "no Python loops over data" isn't style advice; it's the difference between usable and not.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11 · Your turn

    Open `modules/01-arrays/exercises.py`: six functions, each a pattern you'll reuse in every later
    module. The rule: **no Python loops over array elements** — express it with broadcasting, reductions
    and `einsum`. Cells below call your code; `make test N=01` grades it.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo):
    _cases = [((8, 1, 3), (5, 1)), ((3,), (4, 3)), ((1, 4), (3, 1)), ((2, 3), (3, 2))]
    _rows = []
    for _a, _b in _cases:
        _mine = lj.attempt(ex.broadcast_shape, _a, _b)
        try:
            _ref = str(jnp.broadcast_shapes(_a, _b))
        except ValueError:
            _ref = "error"
        if _mine.todo:
            _got = "*not implemented*"
        elif _mine.ok:
            _got = str(_mine.value)
        else:
            _got = "error" if "ValueError" in _mine.error else "raised " + _mine.error.splitlines()[-1]
        _rows.append(f"| `{_a}` | `{_b}` | {_got} | {_ref} | {'✅' if _got == _ref else '·'} |")
    mo.md("**`broadcast_shape`** vs JAX's own `jnp.broadcast_shapes`:\n\n"
          "| a | b | yours | JAX | |\n|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell
def _(ex, jnp, lj, mo):
    _x = jnp.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
    _n = lj.attempt(ex.normalize_rows, _x)
    _o = lj.attempt(ex.one_hot, jnp.array([0, 2, 1]), 4)
    _items = [mo.md("**`normalize_rows([[3, 4], [0, 0], [1, 1]])`** — expect `[[0.6, 0.8], [0, 0], [0.707, 0.707]]`:"),
              _n.view(mo),
              mo.md("**`one_hot([0, 2, 1], 4)`**:"), _o.view(mo)]
    mo.vstack(_items)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _k1, _k2 = jax.random.split(jax.random.key(1))
    _a = jax.random.normal(_k1, (6, 2, 3))
    _b = jax.random.normal(_k2, (6, 3, 4))
    _bm = lj.attempt(ex.batched_matmul, _a, _b)
    _pts = jax.random.normal(_k1, (5, 3))
    _pd = lj.attempt(ex.pairwise_sq_dists, _pts, _pts)
    _items = [mo.md("**`batched_matmul`** on (6, 2, 3) × (6, 3, 4):"), _bm.view(mo, show_value=False)]
    if _bm.ok:
        _err = float(jnp.abs(_bm.value - _a @ _b).max())
        _items.append(mo.md(f"shape `{_bm.value.shape}`, max difference from `a @ b`: `{_err:.1e}`"))
    _items += [mo.md("**`pairwise_sq_dists`** of 5 points with themselves (the diagonal should be 0):"),
               _pd.view(mo)]
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    window = mo.ui.slider(1, 50, value=15, show_value=True, label="moving-average window")
    window
    return (window,)


@app.cell
def _(ex, jax, jnp, lj, mo, plt, window):
    _t = jnp.linspace(0, 4 * jnp.pi, 300)
    _signal = jnp.sin(_t) + 0.4 * jax.random.normal(jax.random.key(2), _t.shape)
    _ma = lj.attempt(ex.moving_average, _signal, window.value)
    if _ma.ok:
        _fig, _ax = plt.subplots(figsize=(7, 2.6))
        _ax.plot(_signal, lw=0.8, alpha=0.5, label="noisy signal")
        _w = window.value
        _ax.plot(jnp.arange(_ma.value.shape[0]) + (_w - 1) / 2, _ma.value, lw=2,
                 label=f"moving average, window {_w}")
        _ax.legend(loc="upper right", fontsize=8)
        _ax.set_title("Smoothing with your moving_average")
        _fig.tight_layout()
        _out = _fig
    else:
        _out = _ma.view(mo)
    _out
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(1))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * JAX docs: [JAX — The Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
      (immutability, out-of-bounds indexing, random numbers) and
      [Type promotion semantics](https://docs.jax.dev/en/latest/type_promotion.html)
    * NumPy: [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) — the rules with pictures
    * Tim Rocktäschel, [einsum is all you need](https://rockt.github.io/2018/04/30/einsum)
    * Jake VanderPlas, [Python Data Science Handbook, ch. 2](https://jakevdp.github.io/PythonDataScienceHandbook/)
      — NumPy fundamentals, all of which carry over to `jax.numpy`

    **Next:** Module 02 — linear algebra as geometry: what a matrix *does* to space.
    """)
    return


if __name__ == "__main__":
    app.run()
