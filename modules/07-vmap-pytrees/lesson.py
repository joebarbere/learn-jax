import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="07 · vmap and pytrees")


@app.cell
def _():
    from dataclasses import dataclass
    from typing import NamedTuple

    import marimo as mo
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    return NamedTuple, dataclass, ex, jax, jnp, lj, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 07 · `vmap`, pytrees and functional state

    **Goal:** two habits that make JAX code short and fast. Write every function for **one example**
    and let `jax.vmap` add the batch dimension. Keep every piece of state — parameters, optimizer
    moments, random keys — in **pytrees** you pass in and return, so `jit`, `grad` and `vmap` all compose.

    **You will:** batch a function without touching it · build an all-pairs distance matrix from a
    two-vector function · race a Python loop against `vmap` · get per-example gradients (and spot an
    outlier with them) · take pytrees apart and put them back together · train **eight models at once**
    with one extra `vmap`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · `vmap`: write it for one, run it for many

    Here's a tiny "model" written for a *single* input vector — no batch axis anywhere:

    ```python
    def predict(w, x):          # w: (3, 2), x: (2,)  ->  (3,)
        return jnp.tanh(w @ x)
    ```

    `jax.vmap(predict, in_axes=(None, 0))` maps it over axis 0 of `x` while sharing `w` (`None` means
    "don't map this argument"). It is **not** a Python loop: `vmap` rewrites the computation so the
    matrix-vector product becomes one matrix-matrix product.
    """)
    return


@app.cell
def _(jax, jnp, mo):
    def predict(w, x):
        return jnp.tanh(w @ x)

    _w = jnp.ones((3, 2)) * 0.1
    _xs = jnp.arange(10.0).reshape(5, 2)
    _batched = jax.vmap(predict, in_axes=(None, 0))
    _out = _batched(_w, _xs)
    _single = jax.make_jaxpr(predict)(_w, _xs[0])
    _vjaxpr = jax.make_jaxpr(_batched)(_w, _xs)
    _dots = [ln.strip() for ln in str(_vjaxpr).splitlines() if "dot_general" in ln]
    mo.md(f"""
    `predict(w, xs[0])` → shape `{predict(_w, _xs[0]).shape}`; `vmap(predict)(w, xs)` with 5 inputs → shape
    `{_out.shape}`.

    The batched jaxpr contains **{len(_dots)}** `dot_general` producing `{_dots[0].split(" = ")[0] if _dots else "?"}` —
    one matrix product for all five inputs, not five matrix-vector products (the unbatched jaxpr's
    `dot_general` produces `{[ln.strip() for ln in str(_single).splitlines() if "dot_general" in ln][0].split(" = ")[0]}`). `in_axes` says which axis of each argument to map over; `out_axes`
    (default 0) where the batch axis goes in the result.
    """)
    return (predict,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Nested `vmap`: all pairs from a function of two

    A distance between *two vectors* is easy to write. vmap it over the first argument, then over the
    second, and you get the full $N \times M$ matrix — the pattern behind attention scores (Module 15),
    where every token is compared with every other token.
    """)
    return


@app.cell
def _(jax, jnp, mo, plt):
    def cosine(u, v):
        return u @ v / (jnp.linalg.norm(u) * jnp.linalg.norm(v))

    _all_pairs = jax.vmap(jax.vmap(cosine, in_axes=(None, 0)), in_axes=(0, None))
    _angles = jnp.linspace(0, jnp.pi, 12)
    _vecs = jnp.stack([jnp.cos(_angles), jnp.sin(_angles)], axis=1)
    sim = _all_pairs(_vecs, _vecs)
    _fig, _ax = plt.subplots(figsize=(4, 3.3))
    _im = _ax.imshow(sim, cmap="RdBu", vmin=-1, vmax=1)
    _fig.colorbar(_im, ax=_ax)
    _ax.set_title("cosine(u_i, v_j) for 12 unit vectors\nfrom 0° to 180°", fontsize=10)
    _ax.set_xlabel("j")
    _ax.set_ylabel("i")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Shape `{sim.shape}` from a function that only knows about two vectors. "
                            "Neighbours are similar (red), opposite directions are −1 (blue).")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · How fast? Python loop vs `vmap` vs hand-batched

    Three ways to apply `predict` to 1,024 inputs with a 512×512 weight matrix on the GPU:
    a Python loop over a jitted single-example function, `jit(vmap(predict))`, and the version you
    would write by hand (`tanh(xs @ w.T)`).
    """)
    return


@app.cell
def _(jax, jnp, lj, plt, predict):
    _k1, _k2 = jax.random.split(jax.random.key(0))
    _w = jax.random.normal(_k1, (512, 512)) / 512**0.5
    _xs = jax.random.normal(_k2, (1024, 512))
    _one = jax.jit(predict)

    def _loop(w, xs):
        return jnp.stack([_one(w, x) for x in xs])

    _t_loop = lj.timeit(_loop, _w, _xs, warmup=1, iters=3)["median"]
    _t_vmap = lj.timeit(jax.jit(jax.vmap(predict, in_axes=(None, 0))), _w, _xs)["median"]
    _t_hand = lj.timeit(jax.jit(lambda w, xs: jnp.tanh(xs @ w.T)), _w, _xs)["median"]
    vmap_times = {"Python loop": _t_loop, "jit(vmap)": _t_vmap, "hand-batched": _t_hand}
    _fig, _ax = plt.subplots(figsize=(5.5, 2.2))
    _ax.barh(list(vmap_times), [t * 1e3 for t in vmap_times.values()], color=["#999", "C0", "C1"])
    _ax.set_xscale("log")
    _ax.set_xlim(left=0.05)
    _ax.set_xlabel("milliseconds, log scale (lower is better)")
    _ax.set_title(f"1,024 examples: vmap is {_t_loop / _t_vmap:,.0f}× faster than the loop", fontsize=10)
    _fig.tight_layout()
    _fig
    return (vmap_times,)


@app.cell
def _(mo, vmap_times):
    mo.md(f"""
    `vmap` matches the hand-batched version ({vmap_times['jit(vmap)'] * 1e3:.3f} ms vs
    {vmap_times['hand-batched'] * 1e3:.3f} ms) — they compile to the same program. The loop pays
    Python and kernel-launch overhead 1,024 times ({vmap_times['Python loop'] * 1e3:.0f} ms). So you get
    to write the clear single-example version *and* run at full speed.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Per-example gradients

    `jax.grad(loss)` of a *mean* loss gives one gradient for the whole batch. Sometimes you want one
    gradient **per example**: to find examples that dominate training, to clip each example's
    influence (the core of differentially-private training), or to measure gradient noise (Module 05).
    In most frameworks that's awkward. In JAX it's `vmap(grad(loss_one_example))`.

    Below: 64 points on a line, plus one corrupted label. Per-example gradient norms find it instantly.
    """)
    return


@app.cell
def _(jax, jnp, plt):
    def sq_loss(params, x, y):
        return (params["w"] * x + params["b"] - y) ** 2

    _kx, _kn = jax.random.split(jax.random.key(7))
    _xs = jax.random.uniform(_kx, (64,), minval=-1, maxval=1)
    _ys = 2.0 * _xs + 0.5 + 0.05 * jax.random.normal(_kn, (64,))
    _ys = _ys.at[17].set(8.0)  # one bad label
    _params = {"w": jnp.array(1.5), "b": jnp.array(0.0)}
    _g = jax.vmap(jax.grad(sq_loss), in_axes=(None, 0, 0))(_params, _xs, _ys)
    grad_norms = jnp.sqrt(_g["w"] ** 2 + _g["b"] ** 2)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(8, 2.6))
    _a1.scatter(_xs, _ys, s=10, c=["C3" if i == 17 else "C0" for i in range(64)])
    _a1.set_title("data (red: corrupted label)", fontsize=10)
    _a2.bar(range(64), grad_norms, color=["C3" if i == 17 else "C0" for i in range(64)])
    _a2.set_title("per-example gradient norm", fontsize=10)
    _a2.set_xlabel("example")
    _fig.tight_layout()
    _fig
    return (grad_norms,)


@app.cell
def _(grad_norms, jnp, mo):
    mo.md(f"""
    Example **{int(jnp.argmax(grad_norms))}** has gradient norm {float(grad_norms.max()):.1f}, versus a
    median of {float(jnp.median(grad_norms)):.2f}: one bad label is pulling harder than dozens of good
    points. Clipping each example's gradient to a fixed norm *before* averaging stops any single
    example from dominating.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Pytrees: models are just nested containers

    A **pytree** is any nesting of lists, tuples, dicts (and `None`, NamedTuples, registered classes)
    whose leaves are arrays. JAX's transformations accept and return pytrees, which is how
    `jax.grad(loss)(params)` hands back gradients *shaped exactly like* `params`, and how optax
    updated a whole model in Module 05.

    The core tools: `jax.tree.leaves` (the arrays, in a fixed order), `jax.tree.structure` (the
    shape of the nesting), `jax.tree.flatten`/`unflatten` (take apart, put back), and
    `jax.tree.map(f, *trees)` (apply `f` leaf by leaf).
    """)
    return


@app.cell
def _(NamedTuple, jax, jnp, mo):
    class AdamState(NamedTuple):
        count: jax.Array
        mu: dict

    tree = {
        "embed": jnp.ones((8, 4)),
        "layers": [{"w": jnp.ones((4, 4)), "b": jnp.zeros(4)}, {"w": jnp.ones((4, 2)), "b": None}],
        "opt": AdamState(count=jnp.array(0), mu={"embed": jnp.zeros((8, 4))}),
    }
    _leaves, _treedef = jax.tree.flatten(tree)
    _rebuilt = jax.tree.unflatten(_treedef, [leaf * 0 + 1 for leaf in _leaves])
    _paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]
    _rows = "\n".join(f"| `{p}` | `{leaf.shape}` |" for p, leaf in zip(_paths, _leaves, strict=True))
    mo.md(
        f"A model-ish pytree has **{len(_leaves)} leaves** (note `\"b\": None` is an empty subtree, "
        "not a leaf):\n\n| path | shape |\n|---|---|\n" + _rows
        + f"\n\n`jax.tree.map(jnp.shape, tree)` keeps the structure: "
        f"`{jax.tree.map(jnp.shape, tree)['layers'][0]}` for the first layer. Flatten → change every leaf "
        f"→ unflatten gives back a `{type(_rebuilt['opt']).__name__}` in the right place."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your own classes as pytrees

    JAX only looks inside containers it knows. Register a dataclass and it becomes a pytree too —
    this is roughly what Flax (Module 10) does with its modules.
    """)
    return


@app.cell
def _(dataclass, jax, jnp, mo):
    @dataclass
    class Linear:
        w: jax.Array
        b: jax.Array
        name: str  # metadata: part of the structure, not a leaf

    jax.tree_util.register_dataclass(Linear, data_fields=["w", "b"], meta_fields=["name"])

    _layer = Linear(w=jnp.full((2, 2), 3.0), b=jnp.ones(2), name="proj")
    _halved = jax.tree.map(lambda a: a / 2, _layer)
    _grads = jax.grad(lambda lyr: jnp.sum(jnp.array([1.0, 2.0]) @ lyr.w + lyr.b))(_layer)
    mo.md(f"""
    `jax.tree.map(lambda a: a / 2, layer)` → a `{type(_halved).__name__}` named `{_halved.name!r}` with
    `w = {_halved.w.tolist()}`.

    `jax.grad` of a loss *of the layer* returns a `Linear` too: `grads.w = {_grads.w.tolist()}`,
    `grads.b = {_grads.b.tolist()}`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Functional state: everything in, everything out

    PyTorch models *hold* their parameters and optimizers *hold* their moments, mutating them in place.
    JAX functions can't do that (Module 06: jit freezes side effects). So a JAX training step takes
    **all** state in and returns **all** new state:

    ```python
    def step(params, opt_state, key, batch):  ->  params, opt_state, key, metrics
    ```

    It looks verbose, but it's what lets transformations stack. Because `step` is a pure function of
    pytrees, `jax.jit(step)` compiles it, `jax.grad` differentiates through it, and —
    the party trick — `jax.vmap(step)` trains **several independent models at once**, one per random
    seed, in a single compiled program. Below: eight small MLPs (1 → 32 → 1) fit a sine wave
    simultaneously, each from its own initialization.
    """)
    return


@app.cell
def _(jax, jnp, np, plt):
    def init(key):
        _k1, _k2 = jax.random.split(key)
        return [
            {"w": jax.random.normal(_k1, (1, 32)) * 2.0, "b": jnp.zeros(32)},
            {"w": jax.random.normal(_k2, (32, 1)) * (1 / 32) ** 0.5, "b": jnp.zeros(1)},
        ]

    def forward(params, x):
        h = jnp.tanh(x @ params[0]["w"] + params[0]["b"])
        return h @ params[1]["w"] + params[1]["b"]

    def loss(params, x, y):
        return jnp.mean((forward(params, x) - y) ** 2)

    def train_step(params, x, y):
        _l, g = jax.value_and_grad(loss)(params, x, y)
        return jax.tree.map(lambda p, gi: p - 0.05 * gi, params, g), _l

    _x = jnp.linspace(-3, 3, 128)[:, None]
    _y = jnp.sin(_x)
    ensemble = jax.vmap(init)(jax.random.split(jax.random.key(0), 8))  # 8 models, stacked
    _step8 = jax.jit(jax.vmap(train_step, in_axes=(0, None, None)))  # params batched, data shared
    for _ in range(1500):
        ensemble, ensemble_losses = _step8(ensemble, _x, _y)
    _preds = jax.vmap(forward, in_axes=(0, None))(ensemble, _x)
    _fig, _ax = plt.subplots(figsize=(6, 2.8))
    _ax.plot(np.asarray(_x[:, 0]), np.asarray(_y[:, 0]), "k", lw=3, alpha=0.3, label="sin(x)")
    for _i in range(8):
        _ax.plot(np.asarray(_x[:, 0]), np.asarray(_preds[_i, :, 0]), lw=1)
    _ax.set_title("8 MLPs trained in one vmapped, jitted step", fontsize=10)
    _ax.legend(loc="upper left")
    _fig.tight_layout()
    _fig
    return ensemble, ensemble_losses


@app.cell
def _(ensemble, ensemble_losses, jax, mo):
    _shapes = jax.tree.map(lambda a: a.shape, ensemble)
    mo.md(f"""
    Each leaf of the ensemble has an extra leading axis of 8: the first layer's weights are
    `{_shapes[0]['w']}`. Final losses per model: `{[round(float(v), 3) for v in ensemble_losses]}`.

    **{int((ensemble_losses < 0.02).sum())} of 8** fit the curve well (loss < 0.02). The others got the
    same data, learning rate and 1,500 steps but a different random initialization, and are still well
    off the curve — a preview of why initialization gets its own section in Module 11.

    One compiled step updated all eight — the same trick runs hyperparameter sweeps (vmap over learning
    rates) or ensembles for uncertainty estimates, for the cost of a slightly bigger batch.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "in_axes=(None, 0, 0)": "a",
            "in_axes=(0, 0, 0)": "b",
            "in_axes=(0, None, None)": "c",
        },
        label="**Check yourself:** `loss(params, x, y)` handles one example. To get per-example "
              "gradients for a batch `xs, ys` with *shared* params, you write "
              "`jax.vmap(jax.grad(loss), in_axes=...)`. Which `in_axes`?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _feedback = {
        "a": mo.callout(mo.md("Right: don't map the params (`None`), map axis 0 of the data."),
                        kind="success"),
        "b": mo.callout(mo.md("That would expect a *different* params pytree per example, stacked along "
                              "axis 0 — like the 8-model ensemble, not a shared model."), kind="warn"),
        "c": mo.callout(mo.md("That's the ensemble pattern: many models, one shared dataset. Here the "
                              "model is shared and the data is batched."), kind="warn"),
    }
    _feedback.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Your turn

    Open `modules/07-vmap-pytrees/exercises.py`. You'll vmap single-example functions, get per-example
    gradients, write two pytree utilities you'll reuse constantly, and build an MLP whose parameters
    are a plain list of dicts — the from-scratch model Module 10 trains on Fashion-MNIST.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo):
    _a = jnp.arange(6.0).reshape(3, 2)
    _dot = lj.attempt(ex.batched_dot, _a, _a)
    _pts = jnp.array([[0.0, 0.0], [3.0, 4.0]])
    _dist = lj.attempt(ex.pairwise_distances, _pts, _pts)
    _norm = lj.attempt(ex.tree_l2_norm, {"a": jnp.array([3.0]), "b": [jnp.array([4.0])]})
    _axpy = lj.attempt(ex.tree_axpy, 10.0, {"a": jnp.array(1.0)}, {"a": jnp.array(2.0)})
    mo.vstack([
        mo.md("**`batched_dot(a, a)`** for rows [0,1], [2,3], [4,5] — expect `[1, 13, 41]`:"), _dot.view(mo),
        mo.md("**`pairwise_distances`** of (0,0) and (3,4) — expect `[[0, 5], [5, 0]]`:"), _dist.view(mo),
        mo.md("**`tree_l2_norm`** of leaves 3 and 4 — expect 5:"), _norm.view(mo),
        mo.md("**`tree_axpy(10, {a: 1}, {a: 2})`** — expect `{a: 12}`:"), _axpy.view(mo),
    ])
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    def _loss(p, x, y):
        return (x @ p["w"] - y) ** 2

    _peg = lj.attempt(ex.per_example_grads, _loss, {"w": jnp.array([1.0, 1.0])},
                      jnp.array([[1.0, 0.0], [0.0, 2.0]]), jnp.array([0.0, 0.0]))
    _items = [mo.md("**`per_example_grads`** for 2 examples — expect `w` grads `[[2, 0], [0, 8]]`:"),
              _peg.view(mo)]
    _init = lj.attempt(ex.init_mlp, jax.random.key(0), [784, 128, 10])
    if _init.ok:
        _params = _init.value
        _shapes = jax.tree.map(lambda a: a.shape, _params)
        _n = lj.attempt(ex.count_params, _params)
        _fwd = lj.attempt(ex.mlp_forward, _params, jnp.ones((32, 784)))
        _items += [mo.md(f"**`init_mlp([784, 128, 10])`** shapes: `{_shapes}`"),
                   mo.md("**`count_params`** — expect 101,770:"), _n.view(mo)]
        if _fwd.ok:
            _items.append(mo.md(f"**`mlp_forward`** on a batch of 32 → shape `{_fwd.value.shape}` "
                                f"(expect (32, 10)). That's a Fashion-MNIST classifier, untrained."))
        else:
            _items.append(_fwd.view(mo))
    else:
        _items.append(_init.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(7))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaways

    * Write functions for one example; `vmap` batches them, with `in_axes` saying what's shared.
    * Nested `vmap` gives all-pairs computations — the shape of attention.
    * `vmap(grad(...))` gives per-example gradients in one line.
    * Pytrees let `grad`, `jit`, optax and your own code treat a whole model as one value.
    * Pure step functions with explicit state compose — even into training eight models at once.

    ## Further reading
    * JAX docs: [Automatic vectorization](https://docs.jax.dev/en/latest/automatic-vectorization.html)
    * JAX docs: [Working with pytrees](https://docs.jax.dev/en/latest/working-with-pytrees.html)
    * JAX docs: [Stateful computations](https://docs.jax.dev/en/latest/stateful-computations.html)
    * Abadi et al., [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) —
      where per-example gradient clipping comes from.

    **Next:** Module 08 — control flow (`lax.scan`, `cond`, loops) and the numerics of mixed precision.
    """)
    return


if __name__ == "__main__":
    app.run()
