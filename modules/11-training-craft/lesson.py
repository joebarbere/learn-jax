import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="11 · Training craft")


@app.cell
def _():
    import marimo as mo
    import shutil
    import time

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax
    import orbax.checkpoint as ocp
    from flax import nnx

    import learnjax as lj
    from learnjax import rosetta
    from learnjax.data import fashion_mnist

    ex = lj.load_exercises(mo.notebook_dir())
    runs_dir = mo.notebook_dir().parent.parent / "runs" / "11-training-craft"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return ex, fashion_mnist, jax, jnp, lj, mo, nnx, np, ocp, optax, plt, rosetta, runs_dir, shutil, time


@app.cell
def _(mo):
    mo.md(r"""
    # 11 · Training craft: init, norms, schedules, checkpoints

    **Goal:** learn the handful of techniques that separate "the network trains" from "the loss is NaN" or
    "the loss is flat" — and the debugging habits that tell you which one you're looking at.

    **You will:** watch signals vanish or explode through a 20-layer network and fix it with the right
    initialization · fix it again with LayerNorm and RMSNorm (the one in your deepseek-v3-jax model) and see why BatchNorm fell out of favor for language models · see
    why residual connections let gradients reach the first layer · make a small model overfit on purpose and
    rein it in with dropout and weight decay · learn a debugging playbook (overfit one batch, watch the
    gradient norm, catch NaNs at the source) · checkpoint with Orbax and resume *bit-identically* · write a backward pass by hand and match [`jax.grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html#jax.grad) ("Backprop Ninja").

    Module 10 got an MLP to ~88% on Fashion-MNIST with defaults that quietly did a lot of work. This module
    is about those defaults — and what to do when they aren't enough.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    In the 2000s, deep networks were notoriously hard to train. Glorot & Bengio (2010) traced the problem to
    activation and gradient statistics drifting layer by layer, giving Xavier initialization; He et al.
    (2015) adjusted it for ReLU. Dropout (Srivastava et al., 2014) fought overfitting, BatchNorm (Ioffe &
    Szegedy, 2015) steadied training, and residual connections (He et al., 2015) let networks reach 100+
    layers. LayerNorm (2016) and RMSNorm (2019) became the transformer-era normalizations.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Initialization: keep the signal alive

    A dense layer computes $y_j = \sum_{i=1}^{n} w_{ji} x_i$ over $n$ = *fan-in* inputs. If the weights are
    independent with variance $\sigma^2$ and the inputs have mean-square $\mathbb{E}[x^2]$, the variances add:

    $$\operatorname{Var}(y_j) = n\,\sigma^2\,\mathbb{E}[x^2].$$

    Stack 20 layers and that factor $n\sigma^2$ gets **multiplied 20 times**. If it's 0.8, the signal shrinks
    to $0.8^{20} \approx 1\%$; if it's 1.25, it grows 87×. Only a factor of ~1 keeps activations at a sane
    scale — and the same argument applies to gradients flowing backwards.

    ReLU zeroes half its inputs, which halves $\mathbb{E}[x^2]$, so to keep the factor at 1 you need
    $\sigma^2 = 2/n$: **He initialization**. For tanh or linear layers there's no halving, and Glorot/Xavier
    split the difference between forward (fan-in) and backward (fan-out): $\sigma^2 = 2/(n_\text{in} + n_\text{out})$.

    Below: random data pushed through a 20-layer, width-256 ReLU network. The *gain* multiplies the He
    standard deviation. Gain 1 is He.
    """)
    return


@app.cell
def _(jax, jnp):
    def _rms(h):
        return h * jax.lax.rsqrt(jnp.mean(h * h, -1, keepdims=True) + 1e-6)

    def _ln(h):
        _m = h.mean(-1, keepdims=True)
        return (h - _m) * jax.lax.rsqrt(((h - _m) ** 2).mean(-1, keepdims=True) + 1e-5)

    def act_stds(gain, norm=None, depth=20, width=256, seed=0):
        """Std of the activations after each layer of a deep ReLU MLP (He std × gain)."""
        _key = jax.random.key(seed)
        _key, _kx = jax.random.split(_key)
        _h = jax.random.normal(_kx, (1024, width))
        _out = []
        for _ in range(depth):
            _key, _kw = jax.random.split(_key)
            _w = jax.random.normal(_kw, (width, width)) * gain * (2 / width) ** 0.5
            _h = jax.nn.relu(_h @ _w)
            if norm == "rms":
                _h = _rms(_h)
            elif norm == "layer":
                _h = _ln(_h)
            elif norm == "batch":  # over the batch axis (train-mode statistics)
                _bm = _h.mean(0, keepdims=True)
                _h = (_h - _bm) * jax.lax.rsqrt(((_h - _bm) ** 2).mean(0, keepdims=True) + 1e-5)
            _out.append(float(_h.std()))
        return _out

    return (act_stds,)


@app.cell
def _(mo):
    gain = mo.ui.slider(0.5, 1.5, step=0.05, value=1.2, label="init gain (× He std)", show_value=True)
    gain
    return (gain,)


@app.cell
def _(act_stds, gain, plt):
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    for _g, _c in [(0.7, "tab:blue"), (1.0, "tab:green"), (1.3, "tab:red")]:
        _ax.semilogy(range(1, 21), act_stds(_g), color=_c, alpha=0.5, label=f"gain {_g}")
    _ax.semilogy(range(1, 21), act_stds(gain.value), "k-o", ms=3, label=f"your slider: {gain.value:.2f}")
    _ax.set(xlabel="layer", ylabel="activation std (log scale)",
            title="20-layer ReLU MLP: only gain ≈ 1 keeps the signal steady")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ex, lj, mo):
    _he = lj.attempt(ex.init_scale, 784, 512, "he")
    _xa = lj.attempt(ex.init_scale, 784, 512, "xavier")
    _items = [mo.md("### Your turn: `init_scale`\n\nStd for a 784 → 512 layer (Module 10's first layer):"),
              mo.md("He:"), _he.view(mo), mo.md("Xavier:"), _xa.view(mo)]
    if _he.ok and _xa.ok:
        _items.append(mo.md(
            f"He ≈ {_he.value:.4f}, Xavier ≈ {_xa.value:.4f}. Flax's `nnx.Linear` defaults to a variance-"
            f"scaling init of this family (LeCun normal, σ² = 1/fan_in) — close enough to He that Module 10's "
            f"3-layer MLP trained fine. At 20+ layers the difference compounds, which is what you just saw."))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Normalization: stop depending on luck

    Good initialization only fixes the *starting* scale; training moves the weights and the scale drifts
    again. **Normalization layers** re-standardize activations at every layer, every step:

    * **LayerNorm:** $\displaystyle \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma + \beta$, with mean $\mu$ and
      variance $\sigma^2$ taken over the features of *each example* (the last axis).
    * **RMSNorm:** $\displaystyle \frac{x}{\sqrt{\operatorname{mean}(x^2) + \epsilon}}\,w$ — skip the mean
      subtraction and the bias. Cheaper and works as well in transformers, so modern LLMs (Llama, DeepSeek)
      use it. Your deepseek-v3-jax model's `rms_norm` is literally
      [`x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * w`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.rsqrt.html#jax.lax.rsqrt).

    Here the same 20-layer network with a **bad** init (gain 0.7 or 1.3), with and without a norm after
    each layer:
    """)
    return


@app.cell
def _(act_stds, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for _ax, _g in zip(_axes, (0.7, 1.3), strict=True):
        _ax.semilogy(range(1, 21), act_stds(_g), label="no norm")
        _ax.semilogy(range(1, 21), act_stds(_g, "layer"), "--", label="LayerNorm")
        _ax.semilogy(range(1, 21), act_stds(_g, "rms"), ":", lw=2.5, label="RMSNorm")
        _ax.set(title=f"gain {_g}", xlabel="layer")
    _axes[0].set_ylabel("activation std")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ex, jax, jnp, lj, mo, nnx):
    _x = jax.random.normal(jax.random.key(7), (4, 64)) * 3 + 1
    _ln = lj.attempt(ex.layer_norm, _x, jnp.ones(64), jnp.zeros(64))
    _rn = lj.attempt(ex.rms_norm, _x, jnp.ones(64))
    _items = [mo.md("### Your turn: `layer_norm` and `rms_norm`\n\nCompared against Flax's own layers on "
                    "the same input (max absolute difference; ~1e-6 or less means you match):")]
    if _ln.ok:
        _ref = nnx.LayerNorm(64, epsilon=1e-5, rngs=nnx.Rngs(0))(_x)
        _items.append(mo.md(f"`layer_norm` vs `nnx.LayerNorm`: **{float(jnp.abs(_ln.value - _ref).max()):.2e}** · "
                            f"row means ≈ {float(jnp.abs(_ln.value.mean(-1)).max()):.1e}, "
                            f"row stds ≈ {float(_ln.value.std(-1).mean()):.4f}"))
    else:
        _items.append(_ln.view(mo))
    if _rn.ok:
        _ref = nnx.RMSNorm(64, epsilon=1e-6, rngs=nnx.Rngs(0))(_x)
        _items.append(mo.md(f"`rms_norm` vs `nnx.RMSNorm`: **{float(jnp.abs(_rn.value - _ref).max()):.2e}** · "
                            f"row RMS ≈ {float(jnp.sqrt((_rn.value**2).mean(-1)).mean()):.4f} — and notice the "
                            f"row means are *not* zero ({float(_rn.value.mean()):.3f}): RMSNorm only fixes scale."))
    else:
        _items.append(_rn.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo, rosetta):
    mo.vstack([
        rosetta.tabs(mo, "LayerNorm", solves="layer_norm"),
        rosetta.tabs(mo, "RMSNorm", solves="rms_norm"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2b · BatchNorm: normalizing over the *batch* instead

    The original normalization layer (Ioffe & Szegedy, 2015) took the statistics over a different axis:
    for each **feature**, the mean and variance across the **examples in the batch**. It's what made deep
    CNNs like ResNet easy to train, and it's still common in vision. But it has two awkward properties:

    1. **An example's output depends on its batch-mates.** Change the other 31 images in the batch and
       image #1's activations change. With small batches the statistics are noisy.
    2. **Train and inference behave differently.** At inference you may have one example — no batch to take
       statistics over. So during training BatchNorm keeps **running averages** of the mean and variance,
       and at inference uses those instead. That's *state that changes during the forward pass*.

    In PyTorch that state hides inside the module (`model.train()` / `model.eval()` flip a flag). In
    functional JAX there is no hidden state, so it becomes explicit: the function takes the running stats
    and returns updated ones — exactly the signature of your exercise below. Flax NNX wraps this in a
    `BatchStat` variable and a `use_running_average` flag.

    Transformers use LayerNorm/RMSNorm instead: per-example statistics, no batch dependence, no running
    state, and identical behavior when generating one token at a time.
    """)
    return


@app.cell
def _(act_stds, jax, jnp, plt):
    # (left) BatchNorm rescues the bad-init deep stack too; (right) but one example's output depends on its batch
    def _deep_out(x, norm, depth=10, width=128):
        _h = x
        for _i in range(depth):
            _w = jax.random.normal(jax.random.key(1000 + _i), (width, width)) * (2 / width) ** 0.5
            _h = jax.nn.relu(_h @ _w)
            if norm == "batch":
                _m = _h.mean(0, keepdims=True)
                _h = (_h - _m) * jax.lax.rsqrt(((_h - _m) ** 2).mean(0, keepdims=True) + 1e-5)
            else:
                _m = _h.mean(-1, keepdims=True)
                _h = (_h - _m) * jax.lax.rsqrt(((_h - _m) ** 2).mean(-1, keepdims=True) + 1e-5)
        return _h

    _x0 = jax.random.normal(jax.random.key(1), (1, 128))
    _sizes = [2, 4, 8, 16, 64, 256]
    _runs = {"batch1": ("batch", 1), "batch10": ("batch", 10), "layer": ("layer", 10)}
    _wobble = {_k: [] for _k in _runs}
    for _B in _sizes:
        for _k, (_norm, _depth) in _runs.items():
            _outs = []
            for _t in range(16):  # the same example, 16 different sets of batch-mates
                _mates = jax.random.normal(jax.random.key(10_000 * _B + _t), (_B - 1, 128))
                _outs.append(_deep_out(jnp.concatenate([_x0, _mates]), _norm, depth=_depth)[0])
            _outs = jnp.stack(_outs)
            _wobble[_k].append(float(_outs.std(0).mean() / (jnp.abs(_outs).mean() + 1e-9)))
    bn_wobble = dict(sizes=_sizes, **_wobble)

    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    _axes[0].semilogy(range(1, 21), act_stds(1.3), label="no norm")
    _axes[0].semilogy(range(1, 21), act_stds(1.3, "batch"), "--", label="BatchNorm")
    _axes[0].semilogy(range(1, 21), act_stds(1.3, "layer"), ":", lw=2.5, label="LayerNorm")
    _axes[0].set(title="gain 1.3: BatchNorm fixes the scale too", xlabel="layer", ylabel="activation std")
    _axes[0].legend(fontsize=8)
    _axes[1].loglog(_sizes, _wobble["batch1"], "o-", label="BatchNorm, 1 layer")
    _axes[1].loglog(_sizes, _wobble["batch10"], "o--", label="BatchNorm, 10 layers")
    _axes[1].loglog(_sizes, [_wobble["batch1"][0] * (2 / _b) ** 0.5 for _b in _sizes], ":", color="gray",
                    label="∝ 1/√B")
    _axes[1].set(title="same example, different batch-mates", xlabel="batch size",
                 ylabel="relative change in its output")
    _axes[1].set_xticks(_sizes, labels=[str(_b) for _b in _sizes])
    _axes[1].minorticks_off()
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return (bn_wobble,)


@app.cell
def _(bn_wobble, mo):
    _b1, _b10 = bn_wobble["batch1"], bn_wobble["batch10"]
    mo.md(f"""
    After **one** BatchNorm layer, a fixed example's output moves by **{_b1[0]:.0%}** of its size when its
    batch-mates change at batch size 2, **{_b1[2]:.0%}** at 8 and **{_b1[-1]:.0%}** at 256 — batch-statistics
    noise shrinking roughly like 1/√B (dotted line). Through **ten** layers the noise compounds: still
    **{_b10[-1]:.0%}** at batch 256. With LayerNorm the change is exactly **{max(bn_wobble['layer']):.0f}** at every
    batch size — each example is normalized on its own, so it isn't even on the log plot. That batch dependence is
    why BatchNorm needs large batches and running statistics, and why language models (trained and served one
    sequence at a time) don't use it.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, nnx):
    _x = jax.random.normal(jax.random.key(8), (32, 6)) * 2 + 3
    _a = lj.attempt(ex.batch_norm, _x, jnp.ones(6), jnp.zeros(6), jnp.zeros(6), jnp.ones(6), 0.1, True)
    _items = [mo.md("### Your turn: `batch_norm`\n\nOne training step on a batch of 32, compared with "
                    "`nnx.BatchNorm` (Flax's `momentum=0.9` means *keep* 90% — the same as your 0.1 "
                    "weight on the new batch), then 300 steps to see the running stats converge:")]
    if _a.ok:
        _y, _m, _v = _a.value
        _bn = nnx.BatchNorm(6, momentum=0.9, epsilon=1e-5, rngs=nnx.Rngs(0))
        _ref = _bn(_x, use_running_average=False)
        _rows = [f"output vs `nnx.BatchNorm`: **{float(jnp.abs(_y - _ref).max()):.1e}** · running mean vs Flax: "
                 f"**{float(jnp.abs(_m - _bn.mean[...]).max()):.1e}** · running var vs Flax: "
                 f"**{float(jnp.abs(_v - _bn.var[...]).max()):.1e}**"]
        _m, _v = jnp.zeros(6), jnp.ones(6)
        for _i in range(300):
            _xb = jax.random.normal(jax.random.key(500 + _i), (64, 6)) * 2 + 3
            _, _m, _v = ex.batch_norm(_xb, jnp.ones(6), jnp.zeros(6), _m, _v, 0.1, True)
        _ye, _, _ = ex.batch_norm(_x[:1], jnp.ones(6), jnp.zeros(6), _m, _v, 0.1, False)
        _rows.append(f"after 300 steps on data with mean 3 and variance 4: running mean ≈ "
                     f"**{float(_m.mean()):.2f}**, running var ≈ **{float(_v.mean()):.2f}**. In eval mode a "
                     f"*single* example (batch of 1) normalizes fine with those: output mean "
                     f"{float(_ye.mean()):.2f} — in train mode a batch of 1 would be all zeros.")
        _items += [mo.md(_r) for _r in _rows]
    else:
        _items.append(_a.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "BatchNorm (training mode)", solves="batch_norm")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Residual connections: a highway for gradients

    Even with a decent init, the gradient reaching layer 1 of a plain deep network is a product of one
    Jacobian per layer above it — the same compounding problem, backwards. A **residual** block adds its
    input back: $x_{\ell+1} = x_\ell + f(x_\ell)$. Now $\partial x_{\ell+1}/\partial x_\ell = I + \partial f/\partial x_\ell$:
    there's always an identity path, so the gradient can't simply multiply away.

    Every block in your deepseek-v3-jax model is `x = x + attention(rms_norm(x))` then
    `x = x + moe(rms_norm(x))` — the **pre-norm residual** pattern you'll implement below.

    The plot measures the gradient norm of the **first** layer's weights as depth grows, for a plain ReLU MLP
    and a pre-norm residual one, both initialized with the common $\sigma^2 = 1/\text{fan-in}$ (no He factor).
    """)
    return


@app.cell
def _(jax, jnp, plt):
    def _first_layer_grad(depth, residual, width=128):
        _keys = jax.random.split(jax.random.key(depth), depth + 2)
        _ws = [jax.random.normal(k, (width, width)) / width**0.5 for k in _keys[:depth]]
        _x = jax.random.normal(_keys[-2], (256, width))
        _t = jax.random.normal(_keys[-1], (256, width))

        def _loss(ws):
            _h = _x
            for _w in ws:
                if residual:
                    _hn = _h * jax.lax.rsqrt(jnp.mean(_h * _h, -1, keepdims=True) + 1e-6)
                    _h = _h + jax.nn.relu(_hn @ _w) / depth**0.5
                else:
                    _h = jax.nn.relu(_h @ _w)
            return jnp.mean((_h - _t) ** 2)

        return float(jnp.linalg.norm(jax.grad(_loss)(_ws)[0]))

    _depths = [2, 4, 8, 16, 32, 64]
    _plain = [_first_layer_grad(_d, False) for _d in _depths]
    _res = [_first_layer_grad(_d, True) for _d in _depths]
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _ax.loglog(_depths, _plain, "o-", label="plain ReLU MLP")
    _ax.loglog(_depths, _res, "s-", label="pre-norm residual")
    _ax.set(xlabel="depth (layers)", ylabel="‖∂loss/∂W₁‖", title="gradient reaching the first layer")
    _ax.set_xticks(_depths, labels=[str(_d) for _d in _depths])
    _ax.minorticks_off()
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _k1, _k2 = jax.random.split(jax.random.key(11))
    _p = {"norm": jnp.ones(16), "w1": jax.random.normal(_k1, (16, 64)) * 0.2, "w2": jnp.zeros((64, 16))}
    _x = jax.random.normal(_k2, (3, 16))
    _a = lj.attempt(ex.residual_block, _p, _x)
    _items = [mo.md("### Your turn: `residual_block`\n\nWith `w2` all zeros, a residual block should be "
                    "*exactly* the identity:")]
    if _a.ok:
        _same = bool(jnp.array_equal(_a.value, _x))
        _items.append(mo.md(f"block(x) == x: **{_same}**. That's why residual networks are easy to start: "
                            f"a freshly initialized deep stack behaves like a shallow one, and each block only "
                            f"has to learn a *correction*. (Some models initialize `w2` to zero on purpose.)"))
    else:
        _items.append(_a.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Learning-rate warmup (recap from Module 05)

    At step 0 the weights are random and Adam's moment estimates are built from one or two gradients, so
    the first updates are the least trustworthy — exactly when a full-size learning rate would do the most
    damage. **Warmup** ramps the rate from 0 over the first few hundred steps (deepseek-v3-jax: 200 of 12,000)
    and then decays it. Norms, residuals, good init and warmup are complementary: together they let you use a
    learning rate large enough to train fast without the early blow-up. Module 05's `warmup_cosine` is the
    schedule; nothing new to build here.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Overfitting, on purpose

    With 1,000 training images and a 670k-parameter MLP, the model can simply **memorize** the training set.
    Training loss goes to ~0 while the loss on held-out images bottoms out and then *rises*: the model gets
    more confident about answers that don't generalize. Two classic counter-measures:

    * **Dropout:** during training, zero each hidden activation with probability $p$ (and scale the rest by
      $1/(1-p)$ so the expected value is unchanged). The network can't rely on any single unit, so it learns
      redundant, more general features. At evaluation time dropout is **off** — `model.eval()` in Flax NNX.
      It needs randomness, so in JAX it needs a PRNG key (NNX keeps one in `nnx.Rngs(dropout=...)`).
    * **Weight decay:** shrink every weight a little each step (AdamW: $\theta \leftarrow \theta - \eta\lambda\theta$),
      a pull toward simpler functions with small weights.

    Below, the same MLP trained three ways for 100 epochs on 1,000 images, validated on 5,000 others.
    """)
    return


@app.cell
def _(fashion_mnist, np):
    _d = fashion_mnist()
    _x = _d["train_images"].reshape(-1, 784).astype(np.float32) / 255.0
    _y = _d["train_labels"]
    train_x, train_y = _x[:1000], _y[:1000]
    val_x, val_y = _x[1000:6000], _y[1000:6000]
    return train_x, train_y, val_x, val_y


@app.cell
def _(jnp, nnx, np, optax):
    class MLP(nnx.Module):
        """784 → 512 → 512 → 10 with optional dropout after each hidden layer."""

        def __init__(self, rate: float, rngs: nnx.Rngs):
            self.l1 = nnx.Linear(784, 512, rngs=rngs)
            self.l2 = nnx.Linear(512, 512, rngs=rngs)
            self.l3 = nnx.Linear(512, 10, rngs=rngs)
            self.drop = nnx.Dropout(rate, rngs=rngs)

        def __call__(self, x):
            x = self.drop(nnx.relu(self.l1(x)))
            x = self.drop(nnx.relu(self.l2(x)))
            return self.l3(x)

    def xent(model, x, y):
        return optax.softmax_cross_entropy_with_integer_labels(model(x), y).mean()

    @nnx.jit
    def train_step(model, opt, x, y):
        _loss, _grads = nnx.value_and_grad(xent)(model, x, y)
        opt.update(model, _grads)
        return _loss

    @nnx.jit
    def eval_metrics(model, x, y):
        _logits = model(x)
        return (optax.softmax_cross_entropy_with_integer_labels(_logits, y).mean(),
                jnp.mean(_logits.argmax(-1) == y))

    def make_model(rate=0.0, wd=0.0, lr=1e-3, seed=0):
        _m = MLP(rate, nnx.Rngs(seed, dropout=seed + 1))
        return _m, nnx.Optimizer(_m, optax.adamw(lr, weight_decay=wd), wrt=nnx.Param)

    def batch_at(x, y, step, size=100):
        """Deterministic minibatch for a given step (so training can be resumed exactly)."""
        _i = np.random.default_rng(step).choice(len(x), size, replace=False)
        return x[_i], y[_i]

    return MLP, batch_at, eval_metrics, make_model, train_step, xent


@app.cell
def _(batch_at, eval_metrics, make_model, time, train_step, train_x, train_y, val_x, val_y):
    def run_variant(rate, wd, epochs=100, steps_per_epoch=10):
        _m, _o = make_model(rate, wd)
        _hist = {"epoch": [], "train": [], "val": [], "acc": []}
        for _e in range(1, epochs + 1):
            _m.train()
            _losses = [float(train_step(_m, _o, *batch_at(train_x, train_y, _e * 1000 + _s)))
                       for _s in range(steps_per_epoch)]
            if _e % 5 == 0:
                _m.eval()
                _vl, _va = eval_metrics(_m, val_x, val_y)
                _hist["epoch"].append(_e)
                _hist["train"].append(sum(_losses) / len(_losses))
                _hist["val"].append(float(_vl))
                _hist["acc"].append(float(_va))
        return _hist

    _t0 = time.perf_counter()
    results = {
        "no regularization": run_variant(0.0, 0.0),
        "dropout 0.5": run_variant(0.5, 0.0),
        "weight decay 1.0": run_variant(0.0, 1.0),
    }
    reg_seconds = time.perf_counter() - _t0
    return reg_seconds, results


@app.cell
def _(mo, plt, reg_seconds, results):
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.6))
    for _name, _h in results.items():
        _a.plot(_h["epoch"], _h["train"], label=_name)
        _b.plot(_h["epoch"], _h["val"], label=_name)
    _a.set(title="training loss (1,000 images)", xlabel="epoch", yscale="log")
    _b.set(title="validation loss (5,000 other images)", xlabel="epoch")
    _b.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {_n} | {min(_h['val']):.3f} | {_h['val'][-1]:.3f} | {_h['acc'][-1]:.1%} |"
                      for _n, _h in results.items())
    mo.vstack([_fig, mo.md(f"""
| variant | best val loss | final val loss | final val accuracy |
|---|---|---|---|
{_rows}

All three runs took {reg_seconds:.0f} s together on the GPU. Without regularization the training loss
collapses toward zero while validation loss climbs well past its minimum — classic overfitting. Dropout and
weight decay keep the *final* validation loss much lower. Notice what they *don't* do here: accuracy barely
moves. The unregularized model still picks the right class about as often; it has just become wildly
overconfident on the ones it gets wrong, and the loss punishes confident mistakes. On 1,000 images no trick
beats more data — Module 10 used all 60,000 and reached ~88%.
""")])
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _x = jnp.ones((100, 1000))
    _a = lj.attempt(ex.dropout, jax.random.key(0), _x, 0.5, True)
    _b = lj.attempt(ex.dropout, jax.random.key(0), _x, 0.5, False)
    _items = [mo.md("### Your turn: `dropout`\n\nRate 0.5 on a block of ones:")]
    if _a.ok and _b.ok:
        _items.append(mo.md(
            f"train mode: {float(jnp.mean(_a.value == 0)):.1%} zeroed, survivors = {float(_a.value.max()):.1f}, "
            f"mean = {float(_a.value.mean()):.3f} (should stay ≈ 1) · eval mode unchanged: "
            f"**{bool(jnp.array_equal(_b.value, _x))}**"))
    else:
        _items.append((_a if not _a.ok else _b).view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo, rosetta):
    mo.vstack([
        rosetta.tabs(mo, "dropout (inverted)", solves="dropout"),
        rosetta.tabs(mo, "Bernoulli samples", solves="dropout"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · A debugging playbook

    | Symptom | First suspects | First check |
    |---|---|---|
    | Loss doesn't go down at all | bug in the loss or labels; learning rate far too low; frozen params | **overfit one batch** |
    | Loss goes down, then NaN | learning rate too high; log/exp/divide on bad values; bf16 overflow | grad norm; `jax_debug_nans` |
    | Sudden loss spikes | a bad batch; LR too high late in training; no clipping | grad norm at the spike; clip |
    | Train ≪ val loss | overfitting | more data, dropout, weight decay, smaller model |
    | Train ≈ val, both high | underfitting | bigger model, train longer, higher LR |

    **Overfit a single batch first.** Before any long run, train on *one* small batch. A healthy model and
    loss drive it to ~0 within a few hundred steps. If they can't, the bug is in your code, not your data
    or hyperparameters — and you found out in seconds instead of hours.
    """)
    return


@app.cell
def _(make_model, plt, train_step, train_x, train_y):
    _m, _o = make_model(lr=1e-3)
    _m.train()
    _xb, _yb = train_x[:32], train_y[:32]
    _hist = [float(train_step(_m, _o, _xb, _yb)) for _ in range(200)]
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.semilogy(_hist)
    _ax.set(title=f"overfitting one batch of 32: loss {_hist[0]:.2f} → {_hist[-1]:.1e}", xlabel="step",
            ylabel="loss")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(batch_at, ex, lj, make_model, mo, nnx, plt, train_x, train_y, xent):
    @nnx.jit
    def _loss_and_grads(model, x, y):
        return nnx.value_and_grad(xent)(model, x, y)

    _m, _o = make_model(lr=3e-3)
    _m.train()
    _l0, _g0 = _loss_and_grads(_m, *batch_at(train_x, train_y, 0))
    _first = lj.attempt(ex.global_grad_norm, _g0)
    _items = [mo.md("""**Watch the gradient norm.** It is one number per step summarizing how hard the
    optimizer is being pushed — the first thing to plot when a run misbehaves. A slow rise often precedes a
    blow-up; a spike pinpoints the bad step. Gradient clipping (Module 05) caps it.

    ### Your turn: `global_grad_norm`
    """)]
    if _first.ok:
        _norms, _losses = [], []
        for _s in range(300):
            _l, _g = _loss_and_grads(_m, *batch_at(train_x, train_y, _s))
            _norms.append(float(ex.global_grad_norm(_g)))
            _losses.append(float(_l))
            _o.update(_m, _g)
        _fig, _ax = plt.subplots(figsize=(7, 3))
        _ax.plot(_norms, label="global grad norm")
        _ax.plot(_losses, label="loss", alpha=0.7)
        _ax.set(xlabel="step", title="gradient norm and loss, lr 3e-3")
        _ax.legend()
        _fig.tight_layout()
        _items += [mo.md(f"Step-0 gradient norm: **{_first.value:.3f}**. Over training the norm settles as the "
                         f"loss falls, with bumps on hard batches:"), _fig]
    else:
        _items.append(_first.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(jax, jnp, mo):
    def _naive_xent(logits, y):
        _p = jnp.exp(logits) / jnp.exp(logits).sum(-1, keepdims=True)  # overflows for big logits
        return -jnp.mean(jnp.log(_p[jnp.arange(len(y)), y]))

    _logits = jnp.array([[120.0, 3.0, -1.0], [2.0, 95.0, 0.5]])
    _y = jnp.array([0, 1])
    _silent = float(jax.jit(_naive_xent)(_logits, _y))

    try:
        with jax.debug_nans(True):  # scoped: switched back off when the block ends
            jax.jit(_naive_xent)(_logits, _y)
        _caught = "(no error raised)"
    except FloatingPointError as _err:
        _caught = str(_err).splitlines()[0][:220]

    _fixed = float(jnp.mean(-jax.nn.log_softmax(_logits)[jnp.arange(2), _y]))
    mo.md(f"""
    **Catch NaNs at the source.** A naive softmax cross-entropy on large logits: `exp(120)` overflows float32
    to `inf`, and `inf / inf` is NaN. Run normally, it **silently** returns `{_silent}` — and in a real run
    that NaN would flow into the gradients and wipe out every weight.

    With NaN debugging on, JAX checks every jitted function's output and, when it finds a NaN, re-runs the
    function op by op and raises at the *first* operation that produced one:

    > `{_caught}`

    It's slow, so turn it on only while hunting: either globally with
    [`jax.config.update("jax_debug_nans", True)`](https://docs.jax.dev/en/latest/_autosummary/jax.config.html#jax.config) at the top of a script, or — as this cell does — scoped with
    [`with jax.debug_nans(True):`](https://docs.jax.dev/en/latest/_autosummary/jax.debug_nans.html#jax.debug_nans), which switches itself off when the block ends. The fix is the log-sum-exp form
    from Module 04 — [`jax.nn.log_softmax`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.log_softmax.html#jax.nn.log_softmax) gives the finite loss **{_fixed:.4f}**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Loss-spike triage, in order:** (1) find the step — log loss *and* grad norm every step, cheaply;
    (2) replay that exact batch (deterministic data order makes this possible — see `batch_at` above) and check
    its inputs; (3) check the learning rate at that step; (4) add or tighten gradient clipping; (5) only then
    suspect precision (bf16 range) and enable `jax_debug_nans`. Your deepseek-v3-jax `train.py` checks
    `isfinite(loss)` every step and refuses to save a non-finite checkpoint — the cheap insurance version.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Checkpointing with Orbax

    Long runs crash, get preempted, or need a second look. A checkpoint must hold **everything** that
    determines the future: parameters, optimizer state (Adam's moments and step count), the RNG state
    (dropout masks!), and the data position. Get all four right and a resumed run is *bit-identical* to one
    that never stopped — the gold standard, and testable.

    In Flax NNX, `nnx.state(model)` and `nnx.state(optimizer)` are pytrees of arrays (including the dropout
    RNG counter). Orbax's `StandardCheckpointer` saves any pytree; restoring into an abstract copy from a
    fresh model and calling `nnx.update` puts it back. Data position is handled by `batch_at(step)`.

    The experiment: train 300 steps straight through; separately, train 150, **save**, build a brand-new model
    and optimizer, **restore**, and train 150 more.
    """)
    return


@app.cell
def _(batch_at, make_model, mo, nnx, np, ocp, plt, runs_dir, shutil, train_step, train_x, train_y):
    def _steps(m, o, start, stop):
        m.train()
        return [float(train_step(m, o, *batch_at(train_x, train_y, s))) for s in range(start, stop)]

    _ma, _oa = make_model(rate=0.3)
    _straight = _steps(_ma, _oa, 0, 300)

    _mb, _ob = make_model(rate=0.3)
    _first = _steps(_mb, _ob, 0, 150)
    _ckpt = runs_dir / "resume_demo"
    shutil.rmtree(_ckpt, ignore_errors=True)
    _ck = ocp.StandardCheckpointer()
    _ck.save(_ckpt, {"model": nnx.state(_mb), "opt": nnx.state(_ob)})
    _ck.wait_until_finished()

    _mc, _oc = make_model(rate=0.3, seed=0)  # fresh objects; same structure
    _restored = _ck.restore(_ckpt, {"model": nnx.state(_mc), "opt": nnx.state(_oc)})
    nnx.update(_mc, _restored["model"])
    nnx.update(_oc, _restored["opt"])
    _resumed = _first + _steps(_mc, _oc, 150, 300)

    _diff = float(np.max(np.abs(np.array(_straight) - np.array(_resumed))))
    _files = sorted(p.name for p in _ckpt.iterdir())
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_straight, lw=3, alpha=0.4, label="straight through")
    _ax.plot(_resumed, "k--", lw=1, label="save @150 → restore → continue")
    _ax.axvline(150, color="gray", ls=":")
    _ax.set(xlabel="step", ylabel="loss", title="resumed run vs uninterrupted run")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Largest difference between the two curves over 300 steps: **{_diff}** "
                           f"(dropout on, so this also proves the RNG state was saved). Checkpoint directory "
                           f"`runs/11-training-craft/resume_demo/` contains: `{_files}`.")])
    return


@app.cell
def _(ex, jnp, lj, mo, runs_dir, shutil):
    _path = runs_dir / "your_turn_ckpt"
    shutil.rmtree(_path, ignore_errors=True)
    _state = {"params": {"w": jnp.arange(6.0).reshape(2, 3)}, "step": jnp.array(7)}
    _a = lj.attempt(ex.save_and_restore, _state, _path)
    _items = [mo.md("### Your turn: `save_and_restore`")]
    if _a.ok:
        _items.append(mo.md(f"Restored `w` = `{_a.value['params']['w'].tolist()}`, step "
                            f"`{int(_a.value['step'])}`. You now have everything needed to make any training "
                            f"loop in this course resumable."))
    else:
        _items.append(_a.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Backprop Ninja: the backward pass by hand

    `jax.grad` has done every backward pass in this course. Once, do it yourself — in **tensor** form, not
    one scalar at a time like Module 03's `Value` class — for a 2-layer MLP on Fashion-MNIST:

    $$h_\text{pre} = xW_1 + b_1,\quad h = \tanh(h_\text{pre}),\quad \text{logits} = hW_2 + b_2,\quad
    L = -\tfrac{1}{N}\textstyle\sum_n \log \operatorname{softmax}(\text{logits}_n)_{y_n}.$$

    Walk backwards, keeping every gradient the same shape as the thing it's the gradient *of*:

    * **Softmax + cross-entropy** collapse into one beautiful line: $\partial L/\partial\,\text{logits} =
      (\text{probs} - \text{onehot}(y))/N$. Push the right class's probability up, everything else down.
    * **A linear layer** $\text{out} = xW + b$ gives three rules you'll use forever: $dW = x^\top\,d\text{out}$,
      $db = \sum_\text{rows} d\text{out}$ (the bias was broadcast over rows, so its gradient sums over them),
      $dx = d\text{out}\,W^\top$. Check the shapes: they're the only ones that fit.
    * **tanh** is elementwise: $\tanh'(z) = 1 - \tanh(z)^2$, and you already have $\tanh(z) = h$ in the cache.

    `forward_with_cache` (provided) keeps the intermediates; `manual_backward` is yours. The table compares
    every parameter's gradient with `jax.grad` — once those numbers are ~1e-7, autodiff holds no mystery:
    it's this, done systematically.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, train_x, train_y):
    _k = jax.random.split(jax.random.key(42), 2)
    _params = {"w1": jax.random.normal(_k[0], (784, 64)) * (1 / 784) ** 0.5, "b1": jnp.zeros(64),
               "w2": jax.random.normal(_k[1], (64, 10)) * (1 / 64) ** 0.5, "b2": jnp.zeros(10)}
    _x, _y = jnp.asarray(train_x[:256]), jnp.asarray(train_y[:256])
    _fwd = lj.attempt(ex.forward_with_cache, _params, _x, _y)
    _items = [mo.md("### Your turn: `manual_backward`")]
    if _fwd.ok:
        _loss, _cache = _fwd.value
        _mine = lj.attempt(ex.manual_backward, _params, _cache, _y)
        if _mine.ok:
            with jax.default_matmul_precision("highest"):
                _ref = jax.grad(lambda p: ex.forward_with_cache(p, _x, _y)[0])(_params)
            _rows = ["| parameter | shape | max abs error vs `jax.grad` | relative error |", "|---|---|---|---|"]
            for _n in ("w1", "b1", "w2", "b2"):
                _err = float(jnp.abs(_mine.value[_n] - _ref[_n]).max())
                _rel = _err / (float(jnp.abs(_ref[_n]).max()) + 1e-30)
                _rows.append(f"| `{_n}` | {tuple(_ref[_n].shape)} | {_err:.1e} | {_rel:.1e} |")
            _items.append(mo.md(f"Loss on 256 Fashion-MNIST images: {float(_loss):.4f}\n\n" + "\n".join(_rows)))
        else:
            _items.append(_mine.view(mo))
    else:
        _items.append(_fwd.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Overfitting — add data, dropout or weight decay": "a",
            "Underfitting — make the model bigger": "b",
            "The learning rate is too low": "c",
        },
        label="**Check yourself:** training loss keeps falling while validation loss has started rising. What's happening?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right — the model is memorizing the training set. Regularize, get more data, "
                              "or stop at the validation minimum (early stopping)."), kind="success"),
        "b": mo.callout(mo.md("Underfitting looks different: *both* losses stay high. Here training loss is "
                              "fine — it's generalization that's failing."), kind="warn"),
        "c": mo.callout(mo.md("A too-low learning rate makes *both* curves fall slowly; it doesn't make "
                              "validation loss rise."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(11))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
      — the debugging playbook, at length. Read it twice.
    * He et al., [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (He init) and
      He et al., [Deep Residual Learning](https://arxiv.org/abs/1512.03385) (ResNets).
    * Zhang & Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).
    * Google Research, [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook).
    * Orbax docs: [Checkpointing](https://orbax.readthedocs.io/en/latest/guides/checkpoint/checkpointing_pytrees.html).

    **Next:** Module 12 — convolutional networks: building an assumption about images into the architecture.
    """)
    return


if __name__ == "__main__":
    app.run()
