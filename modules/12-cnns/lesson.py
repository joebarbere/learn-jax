import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="12 · CNNs")


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
    # 12 · Convolutional networks

    **Goal:** understand convolution — the operation that made deep learning work on images — well enough
    to build it yourself, and see how one *assumption about the data*, baked into the architecture, beats a
    bigger generic model.

    **You will:** slide a kernel along a 1-D signal, then over a Fashion-MNIST image with hand-made edge
    detectors · learn `jax.lax.conv_general_dilated`, its NHWC/HWIO layouts, padding, stride and the output-size
    formula · pool · count parameters and see weight sharing pay off · *prove* translation equivariance by
    shifting an image · train a CNN that beats Module 10's MLP with fewer parameters · look inside at its
    learned filters · show that convolution is secretly a matrix multiply.

    Module 10's MLP flattened each image into 784 unrelated numbers. Shift a shirt two pixels left and, to the
    MLP, it's a completely different input. This module fixes that.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Convolution is a sliding dot product

    Take a short vector of weights — the **kernel**, say $w = [w_0, w_1, w_2]$ — and slide it along a
    signal $x$. At each position, take the dot product of the kernel with the piece of signal under it:

    $$y_i = \sum_{j=0}^{k-1} x_{i+j}\, w_j$$

    That's it. (Mathematicians flip the kernel and call this *cross-correlation*; every deep-learning library
    computes this unflipped version and calls it convolution. Since the kernel is learned, the flip doesn't
    matter.) The kernel decides **what pattern to look for**; sliding it means **looking for it everywhere**.
    An averaging kernel smooths; a kernel that subtracts the left half from the right half responds to
    *changes* — edges. (Making it wider averages away noise, so only real changes stand out.)
    """)
    return


@app.cell
def _(jnp, np, plt):
    _rng = np.random.default_rng(0)
    _t = np.arange(200)
    signal = jnp.array(np.where(_t > 100, 1.0, 0.0) + 0.5 * np.sin(_t / 9) + 0.15 * _rng.normal(size=200),
                       dtype=jnp.float32)

    def slide(x, w):
        """Reference sliding dot product (a plain loop — fine for 200 samples)."""
        _k = len(w)
        return jnp.array([float(x[_i:_i + _k] @ w) for _i in range(len(x) - _k + 1)])

    _smooth = slide(signal, jnp.ones(9) / 9)
    _edge = slide(signal, jnp.concatenate([-jnp.ones(5), jnp.ones(5)]) / 5)
    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    _ax.plot(signal, color="0.7", label="signal (step at t=100 + wave + noise)")
    _ax.plot(np.arange(4, 196), _smooth, label="kernel [1/9 × 9]: smoothing")
    _ax.plot(np.arange(5, 196), _edge, label="kernel [-1×5, +1×5] / 5: change detector")
    _ax.legend(fontsize=8, loc="upper left")
    _ax.set(xlabel="t")
    _fig.tight_layout()
    _fig
    return signal, slide


@app.cell
def _(ex, jnp, lj, mo, signal, slide):
    _w = jnp.array([-1.0, 0.0, 0.0, 0.0, 1.0])
    _a = lj.attempt(ex.conv1d, signal, _w)
    _items = [mo.md("### Your turn: `conv1d` — no loops\n\nThe plot above used a Python loop. Yours builds "
                    "every window at once with an index grid and does **one** matrix-vector product:")]
    if _a.ok:
        _ref = slide(signal, _w)
        _items.append(mo.md(f"Output shape `{_a.value.shape}` · max difference from the loop version: "
                            f"**{float(jnp.abs(_a.value - _ref).max()):.1e}**. The index-grid trick — "
                            f"`arange(n-k+1)[:, None] + arange(k)` — is the seed of *im2col* in Section 8."))
    else:
        _items.append(_a.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Convolution in 2-D

    For images the kernel is a small 2-D patch (3×3 is typical) slid over both axes. Real images have
    **channels** (RGB; or, deeper in a network, 32 or 64 feature channels), so a kernel spans all input
    channels, and a layer has one kernel per **output** channel. JAX's general convolution needs to be told
    which axis is which:

    * images **NHWC** — (batch, height, width, channels);
    * kernels **HWIO** — (kernel height, kernel width, in-channels, out-channels).

    ```python
    jax.lax.conv_general_dilated(images, kernels, window_strides=(1, 1), padding="SAME",
                                 dimension_numbers=("NHWC", "HWIO", "NHWC"))
    ```

    Below: four hand-made 3×3 kernels applied to a Fashion-MNIST image as one 4-output-channel convolution.
    Edge detectors light up where brightness *changes* in their direction.
    """)
    return


@app.cell
def _(FASHION_CLASSES, fashion_mnist, jnp, np):
    _d = fashion_mnist()
    train_x = _d["train_images"][..., None].astype(np.float32) / 255.0   # (60000, 28, 28, 1) NHWC
    train_y = _d["train_labels"]
    test_x = _d["test_images"][..., None].astype(np.float32) / 255.0
    test_y = _d["test_labels"]
    sample = jnp.array(train_x[0:1])  # an ankle boot
    sample_name = FASHION_CLASSES[int(train_y[0])]
    hand_kernels = {
        "vertical edges (Sobel x)": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "horizontal edges (Sobel y)": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "blur": [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    }
    _k = np.stack([np.array(v, np.float32) for v in hand_kernels.values()], -1)  # (3, 3, 4)
    _k[..., 2] /= 16.0
    hand_bank = jnp.array(_k[:, :, None, :])  # HWIO: (3, 3, 1, 4)
    return hand_bank, hand_kernels, sample, sample_name, test_x, test_y, train_x, train_y


@app.cell
def _(hand_bank, hand_kernels, jax, plt, sample, sample_name):
    hand_maps = jax.lax.conv_general_dilated(sample, hand_bank, (1, 1), "SAME",
                                             dimension_numbers=("NHWC", "HWIO", "NHWC"))
    _fig, _axes = plt.subplots(1, 5, figsize=(11, 2.6))
    _axes[0].imshow(sample[0, :, :, 0], cmap="gray")
    _axes[0].set_title(f"input: {sample_name}", fontsize=9)
    for _i, _name in enumerate(hand_kernels):
        _axes[_i + 1].imshow(hand_maps[0, :, :, _i], cmap="RdBu_r" if "edge" in _name else "gray")
        _axes[_i + 1].set_title(_name, fontsize=9)
    for _a in _axes:
        _a.axis("off")
    _fig.tight_layout()
    _fig
    return (hand_maps,)


@app.cell
def _(ex, hand_bank, hand_maps, jnp, lj, mo, sample):
    _a = lj.attempt(ex.conv2d, sample, hand_bank)
    _items = [mo.md("### Your turn: `conv2d`\n\nWrap `conv_general_dilated` with the right layouts, "
                    "then check against the maps above:")]
    if _a.ok:
        _items.append(mo.md(f"shape `{_a.value.shape}` · max difference: "
                            f"**{float(jnp.abs(_a.value - hand_maps).max()):.1e}**"))
    else:
        _items.append(_a.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Padding, stride and output size

    A $k$-wide kernel fits $n - k + 1$ times along an $n$-pixel axis, so each "valid" convolution shrinks
    the image. Two knobs control the output size:

    * **padding** $p$ — add $p$ zeros on each side ("SAME" picks $p$ to keep the size, for stride 1);
    * **stride** $s$ — jump $s$ pixels between positions (stride 2 halves the size).

    $$n_\text{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

    (**Dilation** spaces the kernel's taps apart, so a 3×3 kernel with dilation 2 covers a 5×5 area — a
    cheap way to see wider context. Its effective size is $k + (k-1)(d-1)$.)
    """)
    return


@app.cell
def _(mo):
    n_in = mo.ui.slider(4, 32, value=28, label="input size n", show_value=True)
    k_in = mo.ui.slider(1, 7, value=3, label="kernel k", show_value=True)
    s_in = mo.ui.slider(1, 4, value=1, label="stride s", show_value=True)
    p_in = mo.ui.slider(0, 3, value=0, label="padding p", show_value=True)
    mo.hstack([n_in, k_in, s_in, p_in])
    return k_in, n_in, p_in, s_in


@app.cell
def _(ex, jax, jnp, k_in, lj, mo, n_in, p_in, s_in):
    _n, _k, _s, _p = n_in.value, k_in.value, s_in.value, p_in.value
    _items = []
    if _k > _n + 2 * _p:
        _items.append(mo.callout(mo.md("The kernel doesn't fit — make it smaller or pad more."), kind="warn"))
    else:
        _real = jax.lax.conv_general_dilated(jnp.ones((1, _n, _n, 1)), jnp.ones((_k, _k, 1, 1)), (_s, _s),
                                             [(_p, _p), (_p, _p)],
                                             dimension_numbers=("NHWC", "HWIO", "NHWC")).shape[1]
        _items.append(mo.md(f"JAX's actual output: **{_real} × {_real}**"))
        _a = lj.attempt(ex.conv_output_size, _n, _k, _s, _p)
        _items.append(mo.md("Your `conv_output_size`:"))
        _items.append(_a.view(mo) if not _a.ok else mo.md(
            f"**{_a.value}** {'✅ matches' if _a.value == _real else '❌ differs from JAX'}"))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Pooling

    After detecting features we often don't need their *exact* position — "there's an edge around here" is
    enough. **Max pooling** keeps the largest value in each 2×2 block, halving height and width: 4× less data
    for the next layer, and a little tolerance to small shifts. In JAX it's a `jax.lax.reduce_window`.
    """)
    return


@app.cell
def _(ex, hand_maps, lj, mo, plt):
    _a = lj.attempt(ex.max_pool2d, hand_maps)
    _items = [mo.md("### Your turn: `max_pool2d`")]
    if _a.ok:
        _fig, _axes = plt.subplots(1, 2, figsize=(5.5, 2.6))
        _axes[0].imshow(hand_maps[0, :, :, 0], cmap="RdBu_r")
        _axes[0].set_title(f"vertical edges {hand_maps.shape[1]}×{hand_maps.shape[2]}", fontsize=9)
        _axes[1].imshow(_a.value[0, :, :, 0], cmap="RdBu_r")
        _axes[1].set_title(f"after 2×2 max pool {_a.value.shape[1]}×{_a.value.shape[2]}", fontsize=9)
        for _x in _axes:
            _x.axis("off")
        _fig.tight_layout()
        _items.append(_fig)
    else:
        _items.append(_a.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Weight sharing: why convolutions are cheap

    A 3×3 conv from 32 to 64 channels has one 3×3×32 kernel per output channel: $3\cdot3\cdot32\cdot64 + 64$
    = 18,496 parameters — **whatever the image size**, because the same kernel is reused at every position.
    A dense layer mapping a 28×28×32 feature map to 28×28×64 would need a weight for every input-output pair.
    """)
    return


@app.cell
def _(ex, lj, mo):
    _conv = lj.attempt(ex.count_conv_params, 3, 3, 32, 64)
    _items = [mo.md("### Your turn: `count_conv_params`")]
    if _conv.ok:
        _dense = (28 * 28 * 32) * (28 * 28 * 64) + 28 * 28 * 64
        _items.append(mo.md(f"""
| layer (28×28×32 → 28×28×64) | parameters |
|---|---|
| 3×3 convolution | **{_conv.value:,}** |
| dense, every pixel to every pixel | **{_dense:,}** ({_dense / _conv.value:,.0f}× more) |

That dense layer's weights alone are {_dense * 4 / 2**30:.1f} GiB in float32; training it with Adam (16 bytes
per parameter, Module 00) needs {_dense * 16 / 2**30:.0f} GiB — more than your card has, for one layer.
"""))
    else:
        _items.append(_conv.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Translation equivariance, demonstrated

    Because the same kernel is applied everywhere, **shifting the input shifts the output by the same amount**
    — the feature map moves with the object. A dense layer has no such property. Below, the boot is shifted
    3 pixels right. For the convolution we shift the *new* feature map back and compare; for a dense layer
    (random weights, 784 → 784) we compare its outputs directly.
    """)
    return


@app.cell
def _(hand_bank, jax, jnp, mo, sample):
    def _conv(x):
        return jax.lax.conv_general_dilated(x, hand_bank, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))

    _shifted = jnp.roll(sample, 3, axis=2)
    _a, _b = _conv(sample), jnp.roll(_conv(_shifted), -3, axis=2)
    _inner = (slice(None), slice(2, -2), slice(5, -5), slice(None))  # ignore wrapped-around borders
    _conv_rel = float(jnp.linalg.norm(_a[_inner] - _b[_inner]) / jnp.linalg.norm(_a[_inner]))

    _w = jax.random.normal(jax.random.key(0), (784, 784)) / 28.0
    _d1, _d2 = sample.reshape(1, -1) @ _w, _shifted.reshape(1, -1) @ _w
    _dense_rel = float(jnp.linalg.norm(_d1 - _d2) / jnp.linalg.norm(_d1))
    mo.md(f"""
    | layer | relative change of the output after shifting the input 3 px |
    |---|---|
    | convolution (feature map shifted back by 3 px) | **{_conv_rel:.1e}** — identical, just moved |
    | dense layer (784 → 784, random weights) | **{_dense_rel:.0%}** — no way to "shift it back" |

    This is the assumption a CNN builds in: *a pattern means the same thing wherever it appears.* An MLP has to
    learn that separately for every position, from data.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Train a CNN — and race Module 10's MLP

    A small, classic CNN in Flax NNX (`nnx.Conv` defaults to "SAME" padding and NHWC):

    `conv 3×3 → 32` → ReLU → max-pool → `conv 3×3 → 64` → ReLU → max-pool → flatten (7·7·64) → `dense 128` →
    ReLU → `dense 10`

    against the 784 → 512 → 512 → 10 MLP from Module 10. Same optimizer (Adam, 1e-3), same batches,
    5 epochs each over all 60,000 images, scored on the 10,000 test images.
    """)
    return


@app.cell
def _(jax, jnp, nnx):
    def pool2(x):
        return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

    class CNN(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
            self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
            self.fc1 = nnx.Linear(7 * 7 * 64, 128, rngs=rngs)
            self.fc2 = nnx.Linear(128, 10, rngs=rngs)

        def features(self, x):
            return nnx.relu(self.conv1(x))  # first-layer feature maps, for visualization

        def __call__(self, x):
            x = pool2(self.features(x))
            x = pool2(nnx.relu(self.conv2(x)))
            return self.fc2(nnx.relu(self.fc1(x.reshape(x.shape[0], -1))))

    class MLP(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.l1 = nnx.Linear(784, 512, rngs=rngs)
            self.l2 = nnx.Linear(512, 512, rngs=rngs)
            self.l3 = nnx.Linear(512, 10, rngs=rngs)

        def __call__(self, x):
            x = x.reshape(x.shape[0], -1)
            return self.l3(nnx.relu(self.l2(nnx.relu(self.l1(x)))))

    return CNN, MLP


@app.cell
def _(CNN, MLP, jax, jnp, nnx, np, optax, test_x, test_y, time, train_x, train_y):
    @nnx.jit
    def _step(model, opt, x, y):
        def _loss(m):
            return optax.softmax_cross_entropy_with_integer_labels(m(x), y).mean()

        _l, _g = nnx.value_and_grad(_loss)(model)
        opt.update(model, _g)
        return _l

    @nnx.jit
    def _correct(model, x, y):
        return jnp.sum(model(x).argmax(-1) == y)

    def _train(cls, epochs=5, batch=128):
        _m = cls(nnx.Rngs(0))
        _o = nnx.Optimizer(_m, optax.adam(1e-3), wrt=nnx.Param)
        _rng = np.random.default_rng(0)
        _accs = []
        _t0 = time.perf_counter()
        for _ in range(epochs):
            _perm = _rng.permutation(len(train_x))
            for _i in range(0, len(train_x) - batch + 1, batch):
                _b = _perm[_i:_i + batch]
                _l = _step(_m, _o, train_x[_b], train_y[_b])
            jax.block_until_ready(_l)
            _accs.append(sum(int(_correct(_m, test_x[_j:_j + 1000], test_y[_j:_j + 1000]))
                             for _j in range(0, 10000, 1000)) / 10000)
        _n = sum(_p.size for _p in jax.tree.leaves(nnx.state(_m, nnx.Param)))
        return _m, {"acc": _accs, "seconds": time.perf_counter() - _t0, "params": _n}

    cnn, cnn_stats = _train(CNN)
    _, mlp_stats = _train(MLP)
    return cnn, cnn_stats, mlp_stats


@app.cell
def _(cnn_stats, mlp_stats, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ep = range(1, len(cnn_stats["acc"]) + 1)
    _ax.plot(_ep, [100 * a for a in cnn_stats["acc"]], "o-", label="CNN")
    _ax.plot(_ep, [100 * a for a in mlp_stats["acc"]], "s-", label="MLP (Module 10)")
    _ax.set(xlabel="epoch", ylabel="test accuracy (%)", title="Fashion-MNIST test accuracy")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"""
| model | parameters | test accuracy after 5 epochs | training time |
|---|---|---|---|
| CNN | {cnn_stats['params']:,} | **{cnn_stats['acc'][-1]:.1%}** | {cnn_stats['seconds']:.0f} s |
| MLP | {mlp_stats['params']:,} | **{mlp_stats['acc'][-1]:.1%}** | {mlp_stats['seconds']:.0f} s |

The CNN wins with {1 - cnn_stats['params'] / mlp_stats['params']:.0%} fewer parameters. It's slower per epoch — each conv weight is reused at every one of
784 positions, so it does far more arithmetic per parameter (a good trade on a GPU). Most of its parameters
are actually in `fc1`; the convolutions themselves hold only {320 + 18_496:,}.
""")])
    return


@app.cell
def _(cnn, plt, sample):
    _k = cnn.conv1.kernel[...]  # (3, 3, 1, 32)
    _maps = cnn.features(sample)
    _fig, _axes = plt.subplots(2, 8, figsize=(11, 3.2))
    for _i in range(8):
        _axes[0, _i].imshow(_k[:, :, 0, _i], cmap="RdBu_r")
        _axes[0, _i].set_title(f"filter {_i}", fontsize=8)
        _axes[1, _i].imshow(_maps[0, :, :, _i], cmap="magma")
    for _a in _axes.flat:
        _a.axis("off")
    _fig.suptitle("learned first-layer 3×3 filters (top) and what each sees in the boot (bottom)", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Nobody told the network to find edges. Several learned filters look like the hand-made ones in Section 2
    — light-to-dark transitions in some direction — because detecting edges is simply useful for telling a
    sandal from a sneaker. Deeper layers combine these into larger, more specific patterns.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Convolution is a matrix multiply (im2col)

    GPUs are built for matrix multiplies (Module 00's TFLOP/s). Libraries often compute convolutions by
    **unfolding** every kernel-sized patch of the image into a row (*im2col*), then multiplying that patch
    matrix by the kernels reshaped into a matrix:
    (positions × patch-size) @ (patch-size × out-channels). One big matmul does the whole layer.
    """)
    return


@app.cell
def _(ex, hand_bank, jax, jnp, lj, mo, sample):
    _a = lj.attempt(ex.im2col, sample[0], 3, 3)
    _items = [mo.md("### Your turn (stretch): `im2col`")]
    if _a.ok:
        _as_matmul = (_a.value @ hand_bank.reshape(9, 4)).reshape(26, 26, 4)
        _ref = jax.lax.conv_general_dilated(sample, hand_bank, (1, 1), "VALID",
                                            dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]
        _items.append(mo.md(
            f"Patch matrix `{_a.value.shape}` @ kernel matrix `(9, 4)` → reshaped `{_as_matmul.shape}` · max "
            f"difference from `conv_general_dilated`: **{float(jnp.abs(_as_matmul - _ref).max()):.1e}**. "
            f"The price: each pixel is copied into up to 9 rows — memory for speed."))
    else:
        _items.append(_a.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Architecture encodes assumptions

    A CNN assumes **locality** (nearby pixels matter together) and **translation equivariance** (a pattern
    means the same thing anywhere). For images both are true, so the CNN needs less data and fewer
    parameters than an MLP that has to learn them.

    Language is different. In *"The detective who had followed the suspect for three days finally **saw**
    him"*, the word that matters for "saw" is "detective", eleven words back — not its neighbours. A fixed
    3-wide window is the wrong assumption. Part 4 builds **attention**: instead of a fixed kernel saying
    *where* to look, the model computes, for each word, *which* other positions to look at — from the data
    itself.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "The same kernel weights are reused at every position": "a",
            "Convolutions use smaller numbers": "b",
            "Pooling removes most of the parameters": "c",
        },
        label="**Check yourself:** why does a 3×3 conv layer have so few parameters compared with a dense layer on the same feature map?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right — weight sharing. The parameter count depends on kernel size and "
                              "channels, not on the image size."), kind="success"),
        "b": mo.callout(mo.md("The values aren't the point — it's how many there are. Each kernel is reused "
                              "at every position (weight sharing)."), kind="warn"),
        "c": mo.callout(mo.md("Pooling has no parameters at all. The saving comes from reusing each kernel at "
                              "every position."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Vincent Dumoulin & Francesco Visin, [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
      — the padding/stride/dilation pictures everyone links to.
    * Chris Olah, [Conv Nets: A Modular Perspective](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/).
    * Stanford CS231n, [Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) notes.
    * Distill, [Feature Visualization](https://distill.pub/2017/feature-visualization/) — what deeper layers learn.
    * JAX docs: [`jax.lax.conv_general_dilated`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv_general_dilated.html).

    **Next:** Module 13 — tokenization: turning text into the integers a language model reads.
    """)
    return


if __name__ == "__main__":
    app.run()
