import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="24 · Parallelism")


@app.cell
def _():
    import jax

    # Sharding needs many devices; your machine has one GPU. JAX can split the CPU into 8 simulated
    # devices — set before any JAX backend starts (the mechanism the JAX 201 docs use).
    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_num_cpu_devices", 8)

    import contextlib
    import importlib.util
    import io
    from functools import partial

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from jax.sharding import AxisType, NamedSharding

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    _spec = importlib.util.spec_from_file_location("m24_parallel", mo.notebook_dir() / "parallel.py")
    par = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(par)
    cpus = jax.devices()
    return (AxisType, NamedSharding, contextlib, cpus, ex, io, jax, jnp, lj, mo, np, par, partial, plt)


@app.cell
def _(cpus, mo):
    mo.md(f"""
    # 24 · Parallelism and sharding

    **Goal:** split one computation across many devices — the data, the weights, or both — and
    understand exactly what communication that costs. This is how every large model is trained.

    **You will:** build meshes of named axes and lay arrays out on them · read shardings in array
    types · see which **collectives** the compiler inserts, by reading compiled HLO · write collectives
    yourself with [`shard_map`](https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html#jax.shard_map) (including an all-reduce built from neighbour-to-neighbour passes) · run
    the same training step as **data parallel, FSDP, tensor parallel and FSDP+TP** by swapping one dict
    of specs, measuring memory and communication · sketch **pipeline** and **expert** parallelism ·
    create parameters directly in their shards.

    **The devices:** your RX 7600 XT is one device, so this lesson runs on **{len(cpus)} simulated CPU
    devices** (`{cpus[0]}` … `{cpus[-1]}`). Everything here — layouts, collectives, correctness, memory
    per device — behaves the same as on real accelerators. What it *can't* show is speed: CPU
    "devices" share one memory, so communication is nearly free. Module 25 runs this code on real
    multi-GPU hardware.

    Primary sources: the JAX docs' [201 section](https://docs.jax.dev/en/latest/201/index.html)
    ([placement](https://docs.jax.dev/en/latest/201/placement.html),
    [distributed arrays](https://docs.jax.dev/en/latest/201/sharding.html),
    [shard_map](https://docs.jax.dev/en/latest/201/shard-map.html)) and the JAX docs'
    [Training Cookbook](https://docs.jax.dev/en/latest/the-training-cookbook.html).
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Training across many machines started with data parallelism, as in Google's DistBelief (Dean et al.,
    2012). As models outgrew one device, Megatron-LM (Shoeybi et al., 2019) split layers across GPUs (tensor
    parallelism), GPipe (Huang et al., 2018) split them into pipeline stages, and ZeRO (Rajbhandari et al.,
    2019) sharded optimizer state, the idea behind FSDP. GShard (2020) and GSPMD (2021) let a compiler
    insert the communication from sharding annotations, which is how JAX's sharding API works.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(AxisType, cpus, jax, mo):
    mesh = jax.make_mesh((4, 2), ("X", "Y"), devices=cpus, axis_types=(AxisType.Explicit,) * 2)
    _grid = "\n".join(f"    | X={_r} | " + " | ".join(f"CPU {d.id}" for d in _row) + " |"
                      for _r, _row in enumerate(mesh.devices))
    mo.md(f"""
    ## 1 · Meshes and named axes

    A **mesh** arranges devices in a grid and names its axes. [`jax.make_mesh((4, 2), ("X", "Y"))`](https://docs.jax.dev/en/latest/_autosummary/jax.make_mesh.html#jax.make_mesh) gives
    {mesh.devices.size} devices as 4 rows (axis `X`) × 2 columns (axis `Y`):

    | | Y=0 | Y=1 |
    |---|---|---|
{_grid}

    Names are the point: a layout says "split this dimension across `X`", never "across devices 0, 2,
    4, 6". On real hardware you choose axes so the chattiest communication runs over the fastest links.

    Axis types: **Explicit** (the default for `jax.make_mesh`) puts shardings into array *types*, so
    you see them and the compiler won't guess; **Auto** lets the compiler decide inside [`jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit).
    This mesh is `{mesh}`.
    """)
    return (mesh,)


@app.cell
def _(mo):
    spec_choice = mo.ui.dropdown(
        options=['P("X", "Y")', 'P("X", None)', 'P(None, "Y")', 'P("Y", "X")', 'P(("X", "Y"), None)', "P()"],
        value='P("X", "Y")', label="PartitionSpec for an 8×4 array")
    spec_choice
    return (spec_choice,)


@app.cell
def _(jax):
    SPEC_OPTIONS = {'P("X", "Y")': jax.P("X", "Y"), 'P("X", None)': jax.P("X", None),
                    'P(None, "Y")': jax.P(None, "Y"), 'P("Y", "X")': jax.P("Y", "X"),
                    'P(("X", "Y"), None)': jax.P(("X", "Y"), None), "P()": jax.P()}
    return (SPEC_OPTIONS,)


@app.cell
def _(NamedSharding, SPEC_OPTIONS, contextlib, io, jax, jnp, mesh, mo, spec_choice):
    _spec = SPEC_OPTIONS[spec_choice.value]
    _a = jax.device_put(jnp.arange(32.0).reshape(8, 4), NamedSharding(mesh, _spec))
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        jax.debug.visualize_array_sharding(_a)
    _shards = sorted({(s.device.id, s.data.shape, s.index) for s in _a.addressable_shards})
    _rows = "\n".join(f"    | CPU {d} | {shape} | `{idx}` |" for d, shape, idx in _shards[:8])
    mo.md(f"""
    `jax.device_put(x, NamedSharding(mesh, {spec_choice.value}))` → type **`{jax.typeof(_a)}`**

    ```text
    {_buf.getvalue().rstrip().replace(chr(10), chr(10) + "    ")}
    ```

    | device | block shape | which slice of the global array |
    |---|---|---|
{_rows}

    A [`PartitionSpec[`](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec) (`](https://docs.jax.dev/en/latest/jax.sharding.html#module-jax.sharding)jax.P`) has one entry per dimension: an axis name splits that dimension across
    the axis, `None` keeps it whole, a tuple of names splits across several axes. Devices that share a
    block hold **copies** — `P()` means fully replicated.
    """)
    return


@app.cell
def _(ex, jax, lj, mo):
    _cases = [((8, 4), jax.P("X", "Y")), ((8, 4), jax.P("X", None)), ((8, 4), jax.P(("X", "Y"), None)),
              ((16, 6, 3), jax.P(None, "Y"))]
    _items = [mo.md("**Your turn — `partition_shape`:** one device's block shape, from the spec alone.")]
    for _shape, _spec in _cases:
        _a = lj.attempt(ex.partition_shape, _shape, _spec, {"X": 4, "Y": 2})
        if not _a.ok:
            _items.append(_a.view(mo))
            break
        _items.append(mo.md(f"`{_shape}` with `{_spec}` → `{_a.value}`"))
    mo.vstack(_items)
    return


@app.cell
def _(NamedSharding, jax, jnp, mesh, mo):
    _x = jax.device_put(jnp.ones((8, 4)), NamedSharding(mesh, jax.P("X", "Y")))
    _w = jax.device_put(jnp.ones((4, 6)), NamedSharding(mesh, jax.P("Y", None)))
    with jax.set_mesh(mesh):
        try:
            _ = _x @ _w
            _ambig = "no error"
        except Exception as _e:  # noqa: BLE001
            _ambig = f"{type(_e).__name__}: {str(_e).splitlines()[0][:260]}"
        _z = jnp.matmul(_x, _w, out_sharding=jax.P("X", None))
        _r = jax.reshard(_x, jax.P(None, "Y"))
        _s = _x.sum(axis=1)
    _unc = jnp.ones(3)
    mo.md(f"""
    ## 2 · Sharding in types

    In Explicit mode the sharding is part of the type, like the shape. `x` is **`{jax.typeof(_x)}`**:
    8 rows split 4 ways over `X`, 4 columns split 2 ways over `Y`. `w` is **`{jax.typeof(_w)}`**.

    **Propagation** follows rules you can predict: `x.sum(axis=1)` sums away the `Y`-split dimension, so
    the result is **`{jax.typeof(_s)}`** (JAX inserts the cross-device sum for you).

    **Ambiguity is an error, not a guess.** `x @ w` contracts over a dimension split on `Y` in *both*
    operands; the result could be kept split, or summed and replicated. JAX refuses to pick:

    > `{_ambig}`

    You say what you want with `out_sharding=`:
    [`jnp.matmul(x, w, out_sharding=jax.P("X", None))`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matmul.html#jax.numpy.matmul) → **`{jax.typeof(_z)}`**. To change a layout on
    purpose, [`jax.reshard(x, jax.P(None, "Y"))`](https://docs.jax.dev/en/latest/201/sharding.html) → **`{jax.typeof(_r)}`** (that's data movement — a
    collective). In Auto mode, [`jax.lax.with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html#jax.lax.with_sharding_constraint) plays the same role as a hint.

    **Committed vs uncommitted:** [`jnp.ones(3)`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ones.html#jax.numpy.ones) with no placement is *uncommitted*
    (`committed={_unc.committed}`) and may move to wherever it's used; anything you [`device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html#jax.device_put) with a
    sharding is committed (`{_x.committed}`). Mixing committed arrays from *different* meshes inside one
    `jit` is an error — JAX won't silently copy across meshes.
    """)
    return


@app.cell
def _(NamedSharding, ex, jax, jnp, lj, mesh, mo, par):
    _a = jax.device_put(jnp.ones((8, 64)), NamedSharding(mesh, jax.P("X", "Y")))
    _b = jax.device_put(jnp.ones((64, 16)), NamedSharding(mesh, jax.P("Y", None)))
    _c = jax.device_put(jnp.ones((8, 64)), NamedSharding(mesh, jax.P("X", None)))
    _cases = [
        ("sum over the X-split rows", lambda a: a.sum(0), (_a,)),
        ("elementwise (no communication)", lambda a: jnp.tanh(a) * 2, (_a,)),
        ("matmul, contraction split on Y → replicated output",
         lambda a, b: jnp.matmul(a, b, out_sharding=jax.P("X", None)), (_a, _b)),
        ("reshard rows X → columns X", lambda c: jax.reshard(c, jax.P(None, "X")), (_c,)),
        ("reshard to fully replicated", lambda a: jax.reshard(a, jax.P()), (_a,)),
    ]
    _count = ex.count_collectives
    _probe = lj.attempt(_count, "")
    _using = "your `count_collectives`" if _probe.ok else "the reference counter (yours isn't done yet)"
    if not _probe.ok:
        _count = par.count_collectives
    _rows = []
    _first_line = ""
    with jax.set_mesh(mesh):
        for _name, _f, _args in _cases:
            _hlo = jax.jit(_f).lower(*_args).compile().as_text()
            _c2 = _count(_hlo)
            _rows.append(f"| {_name} | " + " | ".join(str(_c2[k]) for k in par.COLLECTIVES) + " |")
            if not _first_line:
                _first_line = next((ln.strip()[:150] for ln in _hlo.splitlines() if "all-reduce(" in ln), "")
    _table = "\n".join("    " + r for r in _rows)
    mo.vstack([
        mo.md(f"""
    ## 3 · What the compiler inserts

    Sharding is a promise about *layout*; communication is the compiler's job. **XLA's SPMD partitioner**
    (the GSPMD paper, below) rewrites your single-device program into one per-device program plus
    **collectives** — the communication patterns that move data between devices:

    * **all-reduce** — every device gets the sum (or max…) of everyone's values
    * **all-gather** — every device gets everyone's blocks, concatenated
    * **reduce-scatter** — sum, but each device keeps only its slice of the result
    * **all-to-all** — every device sends a different block to every other device
    * **collective-permute** — each device sends to one specific other device

    Counted in the compiled HLO of each operation (using {_using}):

    | operation | all-reduce | all-gather | reduce-scatter | all-to-all | collective-permute |
    |---|---|---|---|---|---|
{_table}

    What the first one looks like in HLO:

    ```text
    {_first_line}
    ```

    Elementwise work never communicates. Reductions over a split dimension need an all-reduce.
    Changing which dimension is split is an all-to-all, and making a split array whole is an all-gather.
    Learning to *predict* this table is most of the skill of parallel ML.
    """),
    ])
    return


@app.cell
def _(NamedSharding, cpus, jax, jnp, mo, np, partial):
    m8 = jax.make_mesh((8,), ("i",), devices=cpus)
    _x = jax.device_put(jnp.arange(8.0), NamedSharding(m8, jax.P("i")))

    def _show(f, x, out_spec=jax.P("i")):
        g = partial(jax.shard_map, mesh=m8, in_specs=jax.P("i"), out_specs=out_spec)(f)
        return np.asarray(g(x))

    _ops = {
        "input (device i holds i)": np.asarray(_x),
        "psum": _show(lambda b: jax.lax.psum(b, "i"), _x),
        "axis_index": _show(lambda b: jax.lax.axis_index("i").astype(b.dtype)[None], _x),
        "ppermute → right neighbour": _show(lambda b: jax.lax.ppermute(b, "i", [(k, (k + 1) % 8) for k in range(8)]), _x),
        "all_gather (tiled)": _show(lambda b: jax.lax.all_gather(b, "i", tiled=True, to="invarying"), _x, jax.P()),
        "psum_scatter of 8 ones": np.asarray(partial(jax.shard_map, mesh=m8, in_specs=jax.P(), out_specs=jax.P("i"))(
            lambda b: jax.lax.psum_scatter(b, "i", tiled=True))(jax.device_put(jnp.ones(8), NamedSharding(m8, jax.P())))),
    }
    _rows = "\n".join(f"    | {k} | `{np.array2string(v, precision=0, separator=' ')}` |" for k, v in _ops.items())
    mo.md(f"""
    ## 4 · `shard_map`: writing the per-device program yourself

    [`jax.shard_map`](https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html) flips the view:
    instead of one global program the compiler splits, you write the program **one device runs**, on its
    block, and call collectives by axis name. `in_specs` says how to cut the inputs into blocks;
    `out_specs` how to stitch the per-device results back together.

    Unlike [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html), which removes the
    mapped axis, a shard_map body sees a block **with the same rank**: an `[8]` array split over 8
    devices arrives as a `[1]` array on each.

    On an 8-device axis `"i"`, with device *k* holding the value *k*:

    | collective | result (one value per device, in device order) |
    |---|---|
{_rows}

    `psum` gives everyone the total (28); `ppermute` rotates values one device to the right;
    `all_gather` hands every device the whole array; for `psum_scatter`, every device contributes the same
    eight ones, so each element of the sum is 8, and each device keeps just one element of it.
    """)
    return (m8,)


@app.cell
def _(NamedSharding, jax, jnp, m8, mo, np, partial):
    def _body(fix):
        def f(b):
            c0 = jnp.zeros(())
            if fix:
                c0 = jax.lax.pcast(c0, "i", to="varying")
            c, _ = jax.lax.scan(lambda c, v: (c + v, None), c0, b)
            return c[None]
        return partial(jax.shard_map, mesh=m8, in_specs=jax.P("i"), out_specs=jax.P("i"))(f)

    _x = jax.device_put(jnp.arange(16.0), NamedSharding(m8, jax.P("i")))
    try:
        _body(False)(_x)
        _err = "(no error)"
    except Exception as _e:  # noqa: BLE001
        _err = str(_e).split("\n\n")[1][:300] if "\n\n" in str(_e) else str(_e)[:300]
    _ok = np.asarray(_body(True)(_x))
    mo.md(f"""
    **Types track what varies by device.** Inside a shard_map, JAX knows which values differ per device
    (*varying* over `i`, shown as `{{V:i}}`) and which are the same everywhere. A [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan) whose carry
    starts as a constant but picks up per-device data breaks that bookkeeping:

    > {_err}

    The fix the 201 docs give: declare the starting carry as varying with
    [`jax.lax.pcast(c0, "i", to="varying")`](https://docs.jax.dev/en/latest/jax.lax.html#module-jax.lax). Then each device sums its own two elements:
    `{_ok.tolist()}`. (This check, `check_vma`, catches real bugs, like forgetting a `psum` and
    silently returning different answers on different devices.)
    """)
    return


@app.cell
def _(NamedSharding, ex, jax, lj, m8, mo, np, par):
    _k1, _k2, _k3 = jax.random.split(jax.random.key(0), 3)
    _x = jax.random.normal(_k1, (6, 64))
    _w = jax.random.normal(_k2, (64, 32))
    _want = np.asarray(_x @ _w)
    _items = [mo.md("""
    **Your turn — the two halves of a Megatron MLP, and a ring.** Tensor parallelism (Shoeybi et al.,
    Megatron-LM, below) splits a layer's weight across devices in one of two ways:

    * **row-parallel** `row_parallel_matmul`: split the contraction dimension; each device computes a
      *partial sum* of the whole output, and an all-reduce adds them up.
    * **column-parallel** `column_parallel_matmul`: split the output columns; each device computes its
      own columns, and an all-gather assembles them.

    Megatron's trick is to pair them (column-parallel then row-parallel) so the expensive middle never
    needs gathering. Then `ring_allreduce`: build `psum` out of `ppermute` alone.
    """)]
    _cases = [
        ("row_parallel_matmul", lambda: ex.row_parallel_matmul(
            jax.device_put(_x, NamedSharding(m8, jax.P(None, "i"))),
            jax.device_put(_w, NamedSharding(m8, jax.P("i", None))), m8, "i")),
        ("column_parallel_matmul", lambda: ex.column_parallel_matmul(
            jax.device_put(_x, NamedSharding(m8, jax.P())),
            jax.device_put(_w, NamedSharding(m8, jax.P(None, "i"))), m8, "i")),
    ]
    for _name, _f in _cases:
        _a = lj.attempt(_f)
        if _a.ok:
            _err = float(np.abs(np.asarray(_a.value) - _want).max())
            _items.append(mo.md(f"`{_name}`: max error vs `x @ w` = **{_err:.1e}** {'✅' if _err < 1e-3 else '❌'}"))
        else:
            _items.append(_a.view(mo))
    _blocks = jax.device_put(jax.random.normal(_k3, (8 * 4,)), NamedSharding(m8, jax.P("i")))
    _r = lj.attempt(ex.ring_allreduce, _blocks, m8, "i")
    if _r.ok:
        _total = np.asarray(_blocks).reshape(8, 4).sum(0)
        _err = float(np.abs(np.asarray(_r.value).reshape(8, 4) - _total).max())
        _hlo = jax.jit(lambda b: ex.ring_allreduce(b, m8, "i")).lower(_blocks).compile().as_text()
        _cc = par.count_collectives(_hlo)
        _items.append(mo.md(
            f"`ring_allreduce`: every device holds the total (max error **{_err:.1e}**). Its HLO has "
            f"**{_cc['collective-permute']} collective-permutes and {_cc['all-reduce']} all-reduces** — "
            f"8 devices, 7 hops around the ring. A real ring all-reduce is a reduce-scatter then an "
            f"all-gather, so each device sends only 2·(n−1)/n of the data instead of (n−1)× — that's why "
            f"it's bandwidth-optimal, and why NCCL/RCCL use it."))
    else:
        _items.append(_r.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(cpus, jax, par):
    params = par.init_params(jax.random.key(0))
    batch = par.make_batch(jax.random.key(1))
    ref = par.reference_step(params, batch, cpus[0])
    rows = [par.run_strategy(_n, params, batch, ref, cpus) for _n in par.STRATEGIES]
    return batch, params, ref, rows


@app.cell
def _(mo, par, rows):
    def _mb(b):
        return f"{b / 2**20:.2f}"

    _spec_rows = "\n".join(
        f"    | {n} | {'×'.join(map(str, s))} {a} | " + ", ".join(f"{k} `{v}`" for k, v in specs.items())
        + f" | `{bs}` |" for n, (s, a, specs, bs) in par.STRATEGIES.items())
    _res = "\n".join(
        f"    | {r['strategy']} | {r['mesh']} | {_mb(r['per_device_bytes'])} | {_mb(r['total_bytes'])} | "
        f"{r['max_err']:.1e} | {r['all-reduce']} | {r['all-gather']} | {r['reduce-scatter']} |" for r in rows)
    mo.md(f"""
    ## 5 · One model, four parallelism strategies

    The JAX docs' [Training Cookbook](https://docs.jax.dev/en/latest/the-training-cookbook.html) keeps the
    model code fixed and makes the parallelism strategy **a config: a dict of PartitionSpecs**. Here's a
    tiny GPT-like model (embedding → residual MLP block → output head; {par.V} vocab, width {par.D}, MLP
    {par.F}) trained with Adam, under four configs. Only the dicts change:

    | strategy | mesh | parameter specs | batch spec |
    |---|---|---|---|
{_spec_rows}

    * **Data parallel (DP):** every device has the whole model and a slice of the batch; gradients are
      averaged with an all-reduce.
    * **FSDP** (fully sharded data parallel, the ZeRO idea): also split the *weights and optimizer
      state*; gather each weight just before use, so memory per device drops ~N-fold.
    * **Tensor parallel (TP):** split each layer's matrices (Megatron-style) and keep the batch whole.
    * **FSDP+TP:** both, on a 4×2 mesh: one axis for data/FSDP, the other for tensor parallelism.

    One Adam step under each, against the same step on a single device:

    | strategy | mesh | params + Adam state per device (MiB) | logical size (MiB) | max error vs 1 device | all-reduce | all-gather | reduce-scatter |
    |---|---|---|---|---|---|---|---|
{_res}

    All four compute the **same numbers** (differences are float rounding from summing in a different
    order). DP keeps a full copy of everything on every device (8 × 3 MiB stored across the mesh); the
    sharded strategies store each byte once, an eighth per device.
    The communication differs: DP's single all-reduce averages gradients; FSDP adds all-gathers to fetch
    weights; TP all-reduces activations inside the layers. (XLA *combines* and rewrites collectives per
    backend, so counts measure communication *events*, not bytes. On this CPU backend no reduce-scatter
    appears at all, even for FSDP, whose gradient reduction is the textbook reduce-scatter: XLA chose
    all-reduces here. Module 25 looks at what it picks on real GPUs.)
    """)
    return


@app.cell
def _(plt, rows):
    _fig, _ax = plt.subplots(figsize=(6.5, 2.6))
    _names = [r["strategy"] for r in rows]
    _ax.barh(_names, [r["per_device_bytes"] / 2**20 for r in rows], color=["#888", "#1f77b4", "#2ca02c", "#9467bd"])
    _ax.axvline(rows[0]["total_bytes"] / 2**20 / 8, ls="--", c="k", lw=0.8, label="total ÷ 8 devices")
    _ax.set_xlabel("params + Adam state per device (MiB)")
    _ax.invert_yaxis()
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(batch, cpus, ex, jax, lj, mo, par, params, ref, rows):
    _shapes = {k: v.shape for k, v in params.items()}
    _items = [mo.md("**Your turn — `bytes_per_device` predicts the table; `fsdp_specs` writes FSDP's dict for you.**")]
    _b = lj.attempt(ex.bytes_per_device, _shapes, par.STRATEGIES["FSDP"][2], {"data": 8})
    if _b.ok:
        _meas = next(r for r in rows if r["strategy"] == "FSDP")["per_device_bytes"]
        _items.append(mo.md(
            f"Predicted FSDP: **{_b.value:,} bytes**; measured: **{_meas:,}**. The difference "
            f"({_meas - _b.value} bytes) is Adam's step counter, one scalar replicated everywhere."))
    else:
        _items.append(_b.view(mo))
    _s = lj.attempt(ex.fsdp_specs, params, 8, "data")
    if _s.ok:
        _mine = par.run_strategy("FSDP (yours)", params, batch, ref, cpus,
                                 config=((8,), ("data",), _s.value, jax.P("data")))
        _items.append(mo.md("Your specs: " + ", ".join(f"{k} `{v}`" for k, v in _s.value.items())
                            + f" → one Adam step with them: max error vs 1 device **{_mine['max_err']:.1e}**, "
                            f"**{_mine['per_device_bytes'] / 2**20:.2f} MiB** per device."))
        _items.append(mo.md(
            "These are the specs [`jax.tree_util.tree_map_with_path`](https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.tree_map_with_path.html) "
            "produces from the parameter *names and shapes*: the same tool the Cookbook suggests for "
            "per-parameter rules, and how real codebases shard models with thousands of parameters."))
    else:
        _items.append(_s.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(NamedSharding, cpus, jax, mo, np, par):
    _me = jax.make_mesh((8,), ("expert",), devices=cpus)
    _k = jax.random.split(jax.random.key(4), 3)
    _x = jax.random.normal(_k[0], (64, 32))
    _gw = jax.random.normal(_k[1], (32, 8))
    _ew = jax.random.normal(_k[2], (8, 32, 32)) / 6
    _args = (jax.device_put(_x, NamedSharding(_me, jax.P("expert"))),
             jax.device_put(_gw, NamedSharding(_me, jax.P())),
             jax.device_put(_ew, NamedSharding(_me, jax.P("expert"))))
    _got = par.moe_expert_parallel(*_args, _me)
    _err = float(np.abs(np.asarray(_got) - np.asarray(par.moe_reference(_x, _gw, _ew))).max())
    _cc = par.count_collectives(jax.jit(lambda a, b, c: par.moe_expert_parallel(a, b, c, _me)).lower(*_args).compile().as_text())
    _load = np.bincount(np.asarray(np.argmax(_x @ _gw, -1)), minlength=8)
    mo.md(f"""
    ### Expert parallelism: where MoE goes

    Module 17's mixture-of-experts ran every expert on every device. At scale, **each device owns some
    experts** (GShard's design, below) and tokens travel to them. With 8 experts on 8 devices and top-1
    routing: each device sorts its tokens by destination, one **all-to-all** delivers them, each device
    runs its expert, and a second all-to-all sends results home.

    Compiled HLO: **{_cc['all-to-all']} all-to-alls**, {_cc['all-reduce']} all-reduces. Max error vs the
    dense single-device MoE: **{_err:.1e}**. Tokens per expert this batch: `{_load.tolist()}` — the
    imbalance is why routing needs load balancing (Module 17), and why real systems cap each expert's
    *capacity* and drop overflow tokens (this sketch gives every expert room for all of a device's tokens).
    """)
    return


@app.cell
def _(mo):
    n_micro = mo.ui.slider(1, 16, value=4, label="microbatches", show_value=True)
    n_micro
    return (n_micro,)


@app.cell
def _(NamedSharding, cpus, jax, mo, n_micro, np, par, plt):
    _S, _M = 2, n_micro.value
    _mp = jax.make_mesh((_S,), ("stage",), devices=cpus[:_S])
    _ws = jax.random.normal(jax.random.key(2), (_S, 64, 64)) / 8
    _mbs = jax.random.normal(jax.random.key(3), (_M, 8, 64))
    _out = par.pipeline_forward(jax.device_put(_ws, NamedSharding(_mp, jax.P("stage"))),
                                jax.device_put(_mbs, NamedSharding(_mp, jax.P())), _mp)
    _err = float(np.abs(_out - np.asarray(par.pipeline_reference(_ws, _mbs))).max())
    _fig, _ax = plt.subplots(figsize=(7, 1.6))
    for _s, _m, _t in par.gpipe_schedule(_S, _M):
        _ax.barh(_s, 1, left=_t, color=plt.cm.tab20(_m % 20), edgecolor="k")
        _ax.text(_t + 0.5, _s, str(_m), ha="center", va="center", fontsize=8, color="#1f2430")
    _ax.set_yticks(range(_S), [f"stage {s}" for s in range(_S)])
    _ax.invert_yaxis()
    _ax.set_xlabel("clock tick")
    _ax.set_xlim(0, _M + _S - 1)
    _fig.tight_layout()
    _bubble = (_S - 1) / (_M + _S - 1)
    mo.vstack([mo.md(f"""
    ### Pipeline parallelism: split the layers

    Put the first half of the layers on one device and the second half on another, and stream
    **microbatches** through, passing activations to the next stage with `ppermute` each clock tick
    (GPipe-style; the JAX 201 shard_map page builds the same). Two stages, **{_M} microbatches**: output
    max error vs running the stages in sequence on one device = **{_err:.1e}**.

    The empty cells are the **bubble**: stages waiting for work. Its share of the schedule is
    (S−1)/(M+S−1) = **{_bubble:.0%}** here; more microbatches shrink it. (This is the forward pass only;
    a training schedule interleaves backward passes too.)
    """), _fig])
    return


@app.cell
def _(NamedSharding, jax, m8, mo, np):
    with jax.set_mesh(m8):
        _init = jax.jit(lambda k: jax.random.normal(k, (4096, 1024), out_sharding=jax.P("i", None)) * 0.02)
        _w = _init(jax.random.key(0))
        _mem = _init.lower(jax.random.key(0)).compile().memory_analysis()
    _local = np.arange(64.0).reshape(64, 1)
    _batch = jax.make_array_from_process_local_data(NamedSharding(m8, jax.P("i")), _local)
    _out_mb = getattr(_mem, "output_size_in_bytes", 0) / 2**20
    mo.md(f"""
    ## 6 · Creating arrays where they live

    Make a big weight on one device and then `device_put` it, and that device needs room for **all** of
    it, for a moment. For models bigger than one device that moment never comes. The Cookbook's fix:
    run the initializer under `jit` with an `out_sharding`, so each device generates only its own block.

    [`jax.random.normal(key, (4096, 1024), out_sharding=jax.P("i", None))`](https://docs.jax.dev/en/latest/_autosummary/jax.random.normal.html#jax.random.normal) under `jit` → **`{jax.typeof(_w)}`**;
    the compiled program's output per device is **{_out_mb:.1f} MiB** (of 16 MiB in total). The full
    matrix never exists anywhere.

    The same idea for data: each *process* loads only its own rows, and
    [`jax.make_array_from_process_local_data`](https://docs.jax.dev/en/latest/_autosummary/jax.make_array_from_process_local_data.html)
    assembles them into one global array: **`{jax.typeof(_batch)}`**. In one process it just shards the
    rows you pass; across several hosts it's how every host feeds its own slice of the batch.
    """)
    return


@app.cell
def _(jax, jnp, mo):
    _counter = jax.new_ref(jnp.zeros(3))

    @jax.jit
    def _bump(r, x):
        r[...] += x

    _bump(_counter, jnp.ones(3))
    _bump(_counter, jnp.ones(3))
    mo.md(f"""
    ### Sidebar: mutable references

    Everything in this course has been functional: a step takes state and *returns* new state. JAX also
    has **refs**: [`jax.new_ref[`](https://docs.jax.dev/en/latest/jax.ref.html) (the `](https://docs.jax.dev/en/latest/jax.ref.html#module-jax.ref)jax.ref` module) makes a
    mutable box a jitted function can update in place (`r[...] += x`). After two calls of a jitted
    `bump(r, ones)`, the ref holds `{_counter[...].tolist()}` and the function returned nothing.

    The Training Cookbook uses refs for the whole train state (parameters and optimizer state updated
    in place, the step returns only metrics), which avoids threading state through every call and works
    naturally with donation. The trade-off: you lose "same inputs, same outputs", which is what makes
    functional code easy to test, cache and reason about. Know both; this course stays functional.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Data parallel": "dp",
            "FSDP": "fsdp",
            "Tensor parallel": "tp",
            "Expert parallel": "ep",
        },
        label="**Check yourself:** which strategy keeps a complete copy of every weight on every device?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "dp": mo.callout(mo.md("Right. DP splits only the batch; every device holds the full model and "
                               "optimizer state, which is why it stops working once the model doesn't fit "
                               "on one device, and FSDP takes over."), kind="success"),
        "fsdp": mo.callout(mo.md("No: FSDP is the one that *removes* the copies, sharding weights and "
                                 "optimizer state and gathering each weight only when it's used."), kind="warn"),
        "tp": mo.callout(mo.md("No: TP splits each weight matrix across devices."), kind="warn"),
        "ep": mo.callout(mo.md("No: each device holds only its own experts."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · What simulation can't tell you

    Every number above is about **correctness and memory**, and those carry over exactly to real
    hardware. What 8 CPU "devices" sharing one RAM can't show:

    * **Communication cost.** Here a collective is a memory copy. On GPUs it crosses PCIe, NVLink or a
      network at very different speeds, and whether DP, FSDP or TP is fastest depends on it.
    * **Overlap.** Good parallel programs hide communication behind computation (fetch the next
      layer's weights while computing this one). You can only see that in a real multi-GPU profile.
    * **The bubble's real price**, and how much more microbatches help.
    * **Multiple hosts:** [`jax.distributed.initialize`](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html#jax.distributed.initialize), one process per machine, a mesh spanning
      machines.

    **Next:** Module 25 runs this module's training step on real NVIDIA GPUs in AWS, *unchanged*, to
    measure exactly those.
    """)
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(24))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    * JAX docs, 201 section: [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/201/sharding.html),
      [Manual parallelism with shard_map](https://docs.jax.dev/en/latest/201/shard-map.html) and
      [Data placement](https://docs.jax.dev/en/latest/201/placement.html). This lesson follows them.
    * JAX docs: [The Training Cookbook](https://docs.jax.dev/en/latest/the-training-cookbook.html), for
      config-driven DP/FSDP/TP and ref-based train state.
    * Austin, Douglas et al. (Google DeepMind), [How to Scale Your Model](https://jax-ml.github.io/scaling-book/).
      A free book on sharding Transformers in JAX, with the communication arithmetic this lesson skips.
    * Hugging Face (nanotron team), [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook).
      DP, ZeRO, TP, pipeline and expert parallelism explained from the PyTorch side, with measurements.
    """)
    return


if __name__ == "__main__":
    app.run()
