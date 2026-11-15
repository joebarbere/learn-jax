import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="10 · MLPs in PyTorch")


@app.cell
def _():
    import marimo as mo
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from learnjax.data import fashion_mnist  # NumPy-only part of the course package (no JAX here)

    return fashion_mnist, mo, np, plt, time, torch


@app.cell
def _(mo):
    mo.md(r"""
    # 10 · The same MLP in PyTorch (side-by-side)

    This notebook runs in the **PyTorch** image (`make torch-lesson N=10`), which has no JAX at all. It
    trains the exact network from the lesson — 784 → 256 → 256 → 10, ReLU, He-normal init, AdamW at
    1e-3 with weight decay 1e-4, batches of 128, 10 epochs, the same evaluation after every epoch — so you
    can compare results, speed, and above all **idioms**.

    Most ML code you'll read is PyTorch. After this notebook you should be able to translate between the two.
    """)
    return


@app.cell
def _(mo, torch):
    _props = torch.cuda.get_device_properties(0)
    mo.md(f"""
    ## 1 · The device

    | | |
    |---|---|
    | PyTorch | `{torch.__version__}` (HIP `{torch.version.hip}`) |
    | GPU available | `{torch.cuda.is_available()}` |
    | device | `{torch.cuda.get_device_name(0)}`, {_props.total_memory / 2**30:.1f} GiB |
    | architecture | `{_props.gcnArchName}` |

    Two things worth noticing:
    * On ROCm builds, PyTorch keeps the name **`torch.cuda`** for "the GPU" — so `device="cuda"` means your
      Radeon here. Code written for NVIDIA mostly runs unchanged.
    * The architecture reads **gfx1102**, your card's real ISA: this image sets no `HSA_OVERRIDE_GFX_VERSION`.
      PyTorch's ROCm build supports the card natively; only JAX's plugin needs the override (Module 00).
    """)
    return


@app.cell
def _(fashion_mnist, np, torch):
    DEVICE = "cuda"
    _d = fashion_mnist()
    _mean, _std = 0.2860, 0.3530

    def _prep(images):
        x = (images.reshape(len(images), -1).astype(np.float32) / 255.0 - _mean) / _std
        return torch.tensor(x, device=DEVICE)

    def _labels(y):
        return torch.tensor(y, device=DEVICE, dtype=torch.long)  # cross_entropy wants int64 class indices

    x_train, y_train = _prep(_d["train_images"][:55000]), _labels(_d["train_labels"][:55000])
    x_val, y_val = _prep(_d["train_images"][55000:]), _labels(_d["train_labels"][55000:])
    x_test, y_test = _prep(_d["test_images"]), _labels(_d["test_labels"])
    return DEVICE, x_test, x_train, x_val, y_test, y_train, y_val


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · The model

    ```python
    class MLP(torch.nn.Module):
        def __init__(self, sizes):
            super().__init__()
            self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
            for layer in self.layers:                     # match the JAX lesson's He-normal init
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                torch.nn.init.zeros_(layer.bias)

        def forward(self, x):
            for layer in self.layers[:-1]:
                x = torch.relu(layer(x))
            return self.layers[-1](x)
    ```

    Almost line for line the Flax NNX class. Two quiet differences: `nn.Linear` stores its weight as
    `(out, in)` and computes `x @ W.T + b` (JAX's `(in, out)` with `x @ w + b`), and its *default* init is
    not He-normal — we set it explicitly so the two runs start from comparable weights.
    """)
    return


@app.cell
def _(DEVICE, time, torch):
    class MLP(torch.nn.Module):
        def __init__(self, sizes):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:], strict=True))
            for layer in self.layers:
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                torch.nn.init.zeros_(layer.bias)

        def forward(self, x):
            for layer in self.layers[:-1]:
                x = torch.relu(layer(x))
            return self.layers[-1](x)

    SIZES = [784, 256, 256, 10]
    EPOCHS = 10

    @torch.no_grad()
    def accuracy(model, x, y, bs=1000):
        model.eval()
        correct = sum(int((model(x[i:i + bs]).argmax(-1) == y[i:i + bs]).sum()) for i in range(0, len(x), bs))
        model.train()
        return correct / len(x)

    def train(x_train, y_train, x_val, y_val, compile_model=False):
        torch.manual_seed(0)
        model = MLP(SIZES).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # PyTorch's default wd is 1e-2
        fwd = torch.compile(model) if compile_model else model
        gen = torch.Generator(device=DEVICE).manual_seed(1)
        history, train_time = [], 0.0
        for epoch in range(EPOCHS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            perm = torch.randperm(len(x_train), device=DEVICE, generator=gen)
            losses = []
            for i in range(len(x_train) // 128):
                idx = perm[i * 128:(i + 1) * 128]
                loss = torch.nn.functional.cross_entropy(fwd(x_train[idx]), y_train[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()           # autograd walks the graph recorded during the forward pass
                opt.step()                # updates model.parameters() in place
                losses.append(loss.detach())
            torch.cuda.synchronize()      # GPU work is asynchronous here too: wait before reading the clock
            train_time += time.perf_counter() - t0
            history.append({"epoch": epoch + 1, "train_loss": float(torch.stack(losses).mean()),
                            "train_acc": accuracy(model, x_train[:10000], y_train[:10000]),
                            "val_acc": accuracy(model, x_val, y_val), "train_seconds": train_time})
        return model, history

    return EPOCHS, MLP, accuracy, train


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Train it (eager mode)

    PyTorch runs each operation **as Python reaches it** ("eager"), launching a GPU kernel per op and
    recording what happened on an *autograd tape*; `loss.backward()` replays the tape backwards. No
    compilation step, no tracing — which is why it's so pleasant to debug with `print`.
    """)
    return


@app.cell
def _(accuracy, mo, time, torch, train, x_test, x_train, x_val, y_test, y_train, y_val):
    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    eager_model, eager_hist = train(x_train, y_train, x_val, y_val)
    torch.cuda.synchronize()
    eager_total = time.perf_counter() - _t0
    eager_test = accuracy(eager_model, x_test, y_test)
    mo.md(f"""
    10 epochs: **{eager_total:.1f} s** total, of which **{eager_hist[-1]['train_seconds']:.1f} s** is the training
    steps themselves (the rest is the per-epoch accuracy evaluation). Validation **{eager_hist[-1]['val_acc']:.1%}**,
    test **{eager_test:.1%}**.
    """)
    return eager_hist, eager_test, eager_total


@app.cell
def _(eager_hist, plt):
    _ep = [r["epoch"] for r in eager_hist]
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.2))
    _a.plot(_ep, [r["train_loss"] for r in eager_hist], "o-")
    _a.set(title="training loss (PyTorch, eager)", xlabel="epoch")
    _b.plot(_ep, [r["train_acc"] for r in eager_hist], "o-", label="train (first 10k)")
    _b.plot(_ep, [r["val_acc"] for r in eager_hist], "o-", label="validation")
    _b.set(title="accuracy", xlabel="epoch", ylim=(0.8, 1.0))
    _b.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · `torch.compile`

    PyTorch 2's answer to `jax.jit`: `torch.compile(model)` captures the forward (and, via AOTAutograd, the
    backward) as a graph and generates fused kernels — on ROCm through Triton. Unlike `jax.jit` it's
    optional and falls back to eager for anything it can't capture. Let's see if it helps here.
    """)
    return


@app.cell
def _(accuracy, mo, time, torch, train, x_test, x_train, x_val, y_test, y_train, y_val):
    try:
        torch.cuda.synchronize()
        _t0 = time.perf_counter()
        _m, compiled_hist = train(x_train, y_train, x_val, y_val, compile_model=True)
        torch.cuda.synchronize()
        compiled_total = time.perf_counter() - _t0
        compiled_test = accuracy(_m, x_test, y_test)
        _view = mo.md(f"""
    `torch.compile` **works on this card**. 10 epochs: **{compiled_total:.1f} s** total, training steps
    **{compiled_hist[-1]['train_seconds']:.1f} s** (epoch 1 alone: {compiled_hist[0]['train_seconds']:.1f} s,
    most of it compilation). Test accuracy **{compiled_test:.1%}**.
    """)
    except Exception as _e:  # noqa: BLE001 - report honestly whatever goes wrong
        compiled_hist, compiled_total, compiled_test = None, None, None
        _view = mo.callout(mo.md(f"`torch.compile` failed on this setup: `{type(_e).__name__}: {str(_e)[:300]}`. "
                                 "Eager mode above is unaffected."), kind="warn")
    _view
    return compiled_hist, compiled_test, compiled_total


@app.cell
def _(compiled_hist, compiled_test, compiled_total, eager_hist, eager_test, eager_total, mo):
    _comp = (f"| PyTorch `torch.compile` | {compiled_total:.1f} s | {compiled_hist[-1]['train_seconds']:.1f} s | "
             f"{(compiled_hist[-1]['train_seconds'] - compiled_hist[0]['train_seconds']) / 9:.2f} s | "
             f"{compiled_test:.1%} |" if compiled_hist else "| PyTorch `torch.compile` | failed | | | |")
    _eager_epoch = (eager_hist[-1]['train_seconds'] - eager_hist[0]['train_seconds']) / 9
    mo.md(f"""
    ## 5 · Comparing

    | 10 epochs, same model & hyperparameters | total | training steps | training s per epoch (after the 1st) | test acc |
    |---|---|---|---|---|
    | PyTorch eager | {eager_total:.1f} s | {eager_hist[-1]['train_seconds']:.1f} s | {_eager_epoch:.2f} s | {eager_test:.1%} |
    {_comp}
    | JAX lesson: Python loop over jitted `train_step` | | | ~0.95 s (measured when written) | ~88% |
    | JAX lesson: whole epoch as one jitted `lax.scan` | | | **~0.11 s** (measured when written) | ~88% |

    The JAX rows are training steps only (per-epoch evaluation excluded), copied from the lesson's
    Section 6b because this image has no JAX. Accuracy is the same everywhere: it's the same math.

    **Speed** needs care. This network is tiny — each step is ~0.2 GFLOP (forward + backward), about
    10 µs of matrix math at this card's speed — so every number above is dominated by **overhead**, not
    the GPU. The model is *overhead-bound*:

    * PyTorch eager launches about a dozen small kernels per step from C++, with little Python in between:
      ~0.7 ms per step.
    * JAX's Python loop pays per call for slicing indices in Python and flattening/checking the jitted
      function's pytree arguments: ~2.2 ms per step.
    * JAX's `lax.scan` epoch makes **one** Python call per epoch; XLA runs all 429 steps from its own
      runtime: ~0.25 ms per step — about 3× faster than PyTorch eager here.
    * `torch.compile` fuses kernels, but there's little left to fuse in three matmuls, and it still runs
      the Python loop, so it doesn't help at this size. (PyTorch's own answer to per-step overhead is
      CUDA/HIP graphs via `torch.compile(mode="reduce-overhead")`.)

    So "which is faster?" has no answer for a model this small: it depends on how much per-step Python you
    leave in the loop. For big models each step is milliseconds of real GPU work, overhead becomes a
    rounding error, and both frameworks end up calling similar matrix kernels (**compute-bound**).
    Module 16 compares the two on a transformer in that regime.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Idioms, line by line

    | Task | JAX (functional, explicit state) | PyTorch (objects, eager, autograd tape) |
    |---|---|---|
    | parameters | a pytree you own: `params = init_mlp(key, sizes)` | live inside the module: `model.parameters()` |
    | randomness | explicit keys: `jax.random.split(key)` | global generator: `torch.manual_seed(0)` |
    | forward | `logits = mlp_apply(params, x)` | `logits = model(x)` |
    | gradients | `loss, grads = jax.value_and_grad(loss_fn)(params, x, y)` — a function of params | `loss.backward()` — fills each `p.grad` from the tape |
    | zeroing grads | not needed: grads are return values | `opt.zero_grad()` — or grads accumulate |
    | optimizer | `updates, state = tx.update(grads, state, params)` — returns new state | `opt.step()` — mutates params in place |
    | apply update | `params = optax.apply_updates(params, updates)` (new arrays) | already done by `opt.step()` |
    | compile | `jax.jit(train_step)` — required for speed, needs pure functions | `torch.compile(model)` — optional |
    | device | follows the data: `jax.device_put(x, dev)` | explicit: `x.to("cuda")`, `model.to("cuda")` |
    | async GPU | `jax.block_until_ready(x)` before timing | `torch.cuda.synchronize()` before timing |
    | eval mode | pass a flag (e.g. `train=False`) to your function | `model.eval()` / `model.train()` switch dropout etc. |
    | no-grad eval | nothing to turn off: you only get grads you ask for | `with torch.no_grad():` to skip recording the tape |
    | per-example grads | `jax.vmap(jax.grad(loss))` (Module 07) | `torch.func.vmap(torch.func.grad(...))` |

    The deep difference is in the gradient row. In JAX, the gradient is a **function transformation**:
    `jax.grad(f)` is a new function you can `jit`, `vmap` or `grad` again. In PyTorch, gradients are a
    **side effect** of running code with tracking on. Both are fine — but the JAX style is why `vmap`,
    `pmap`/sharding and higher-order derivatives compose so cleanly, and the PyTorch style is why its
    ecosystem feels so natural to Python programmers. `torch.func` brings the JAX style to PyTorch.

    ## Further reading
    * PyTorch: [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) and
      [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).
    * PyTorch docs: [torch.func](https://pytorch.org/docs/stable/func.html) — JAX-like function transforms in PyTorch.
    * JAX docs, [JAX for PyTorch users / "Thinking in JAX"](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html).
    """)
    return


if __name__ == "__main__":
    app.run()
