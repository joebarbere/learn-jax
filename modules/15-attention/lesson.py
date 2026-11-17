import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="15 · Attention")


@app.cell
def _():
    import marimo as mo
    import time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax

    import learnjax as lj
    from learnjax import rosetta

    ex = lj.load_exercises(mo.notebook_dir())
    return Path, ex, jax, jnp, lj, mo, np, optax, plt, rosetta, time


@app.cell
def _(mo):
    mo.md(r"""
    # 15 · Attention from scratch

    **Goal:** understand the one idea that turned language models from "predict from the last few words"
    into "predict from *whichever* earlier words matter" — and be able to read the attention code in
    deepseek-v3-jax line by line.

    **You will:** build attention as a soft dictionary lookup on hand-picked vectors · see why scores are
    divided by $\sqrt{d}$ · add the causal mask a language model needs · go from one head to many with the
    exact `einsum` strings deepseek-v3-jax uses · discover that attention ignores word order (and fix it
    with positions) · measure the $O(T^2)$ cost on your GPU · train a tiny attention layer and watch it
    learn *where to look*.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    By 2014, neural translation squeezed a whole sentence into one fixed vector (seq2seq). Bahdanau, Cho &
    Bengio (2014) let the decoder look back at every input position and weight them, which was the birth of
    attention. Vaswani et al. (2017) removed recurrence entirely, "Attention Is All You Need", and the
    Transformer took over. FlashAttention (Dao et al., 2022) later reorganized attention around memory
    traffic to make long sequences affordable.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The problem with a fixed window

    Module 14's MLP language model reads a fixed window — say the last 3 tokens — concatenates their
    embeddings, and predicts the next one. Two problems:

    * **The window is the same for every prediction.** In *"The keys to the old cabinet **are** …"* the verb
      agrees with *keys*, four tokens back. In *"The cabinet **is** …"* it's one token back. A fixed window
      either misses the far case or wastes capacity on irrelevant tokens.
    * **Position is baked into the weights.** "Token 3-back" and "token 2-back" use different weights, so
      a pattern learned at one distance doesn't transfer to another.

    What we want is a layer that, for each position, **computes which earlier positions are relevant from
    their content**, then takes a weighted average of them. That layer is attention.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · A soft dictionary lookup

    A Python dict does a *hard* lookup: the query must equal one key exactly, and you get that key's value.
    Attention does a *soft* one:

    1. Compare the **query** $q$ with every **key** $k_i$ by a dot product: $s_i = q \cdot k_i$
       (big when they point the same way — Module 02's cosine similarity, without the normalization).
    2. Turn scores into weights that are positive and sum to 1: $w = \mathrm{softmax}(s)$ (Module 04).
    3. Return the weighted average of the **values**: $\text{out} = \sum_i w_i v_i$.

    Below, five keys are unit arrows in the plane, each with a scalar value. Rotate the query and change
    its length: a long query makes the softmax sharp (close to a hard lookup), a short one blurs it.
    """)
    return


@app.cell
def _(mo):
    q_angle = mo.ui.slider(0, 360, value=40, step=5, label="query angle (°)", show_value=True)
    q_scale = mo.ui.slider(0.5, 8.0, value=3.0, step=0.5, label="query length", show_value=True)
    mo.hstack([q_angle, q_scale])
    return q_angle, q_scale


@app.cell
def _(jax, jnp, mo, np, plt, q_angle, q_scale):
    toy_names = ["apple", "banana", "cherry", "date", "elder"]
    _angles = np.deg2rad([0, 72, 144, 216, 288])
    toy_keys = jnp.array(np.stack([np.cos(_angles), np.sin(_angles)], 1), jnp.float32)
    toy_values = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    _a = np.deg2rad(q_angle.value)
    _q = q_scale.value * jnp.array([np.cos(_a), np.sin(_a)], jnp.float32)
    _scores = toy_keys @ _q
    _w = jax.nn.softmax(_scores)
    _out = float(_w @ toy_values)
    _rows = "\n".join(
        f"| {n} | ({float(k[0]):+.2f}, {float(k[1]):+.2f}) | {float(s):+.2f} | **{float(w):.3f}** | {float(v):.0f} |"
        for n, k, s, w, v in zip(toy_names, toy_keys, _scores, _w, toy_values))
    _fig, _ax = plt.subplots(figsize=(3.4, 3.4))
    for _k, _n, _wi in zip(np.asarray(toy_keys), toy_names, np.asarray(_w)):
        _ax.arrow(0, 0, _k[0], _k[1], width=0.02 + 0.08 * _wi, color="C0", alpha=0.3 + 0.7 * _wi,
                  length_includes_head=True)
        _ax.text(1.15 * _k[0], 1.15 * _k[1], _n, ha="center", va="center", fontsize=9)
    _qn = np.asarray(_q) / max(float(jnp.linalg.norm(_q)), 1e-6)
    _ax.arrow(0, 0, _qn[0], _qn[1], width=0.03, color="C3", length_includes_head=True)
    _ax.text(0.55 * _qn[0], 0.55 * _qn[1] + 0.12, "q", color="C3", fontsize=11)
    _ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal", xticks=[], yticks=[],
            title="keys (thicker = more weight)")
    mo.hstack([
        mo.md(f"""
    | key | key vector | score $q\\cdot k$ | weight | value |
    |---|---|---|---|---|
    {_rows}

    **Output = Σ weight × value = {_out:.2f}**

    A hard lookup would return exactly one value. Here every value contributes in proportion to its
    weight — and because it's all smooth arithmetic, gradients can flow into the query and keys.
    """),
        _fig,
    ], justify="start", gap=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Self-attention: every token is a query

    In **self**-attention the queries, keys and values all come from the same sequence: each token asks
    "which tokens are like me?" Here are six tokens with hand-picked 3-d embeddings (the two *the*s are
    identical; *cat* and *mat* are close). With no learned projections yet — query = key = value =
    embedding — the score matrix is just $XX^\top / \sqrt{d}$:
    """)
    return


@app.cell
def _(jax, jnp, mo, np, plt):
    sent_words = ["the", "cat", "sat", "on", "the", "mat"]
    sent_x = jnp.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.5], [0.0, 0.0, 2.0],
                        [1.0, 0.0, 1.0], [2.0, 0.0, 0.0], [0.0, 2.0, -0.5]], jnp.float32)

    def attend_scratch(q, k, v):
        """Single-head attention, no mask — the whole idea in three lines."""
        scores = q @ k.T / jnp.sqrt(q.shape[-1])
        weights = jax.nn.softmax(scores, axis=-1)
        return weights @ v, weights

    _, _w = attend_scratch(sent_x, sent_x, sent_x)
    _fig, _ax = plt.subplots(figsize=(4.2, 3.6))
    _im = _ax.imshow(np.asarray(_w), cmap="Blues", vmin=0, vmax=1)
    _ax.set(xticks=range(6), xticklabels=sent_words, yticks=range(6), yticklabels=sent_words,
            xlabel="key (attended to)", ylabel="query (attending)", title="softmax(XXᵀ/√d)")
    for (_i, _j), _v in np.ndenumerate(np.asarray(_w)):
        _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center", fontsize=7,
                 color="white" if _v > 0.5 else "black")
    mo.hstack([_fig, mo.md(r"""
    Read a **row**: where that token looks. Each *the* splits its attention between itself and the
    other *the*; *cat* and *mat* look at each other. Every row sums to 1.

    Real models don't use raw embeddings: they learn three matrices and use
    $Q = XW_q$, $K = XW_k$, $V = XW_v$. That lets "what I'm looking for" (query), "what I contain"
    (key) and "what I pass along" (value) differ — *are* can look for a plural noun without itself
    being one.
    """)], justify="start", gap=2)
    return attend_scratch, sent_words, sent_x


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Why divide by $\sqrt{d}$?

    If the entries of $q$ and $k$ are independent with variance 1, then $q\cdot k$ is a sum of $d$ such
    products, so its **variance is $d$** and its typical size is $\sqrt{d}$. Real heads have
    $d = 64$–$128$: unscaled scores of ±10 or more push softmax into *saturation* — one weight ≈ 1,
    the rest ≈ 0. A saturated softmax has almost zero gradient, so the model can't learn to change where
    it looks. Dividing by $\sqrt{d}$ brings the scores back to variance 1 at every width.

    Move the slider: each point averages 200 random queries against 32 random keys.
    """)
    return


@app.cell
def _(mo):
    head_dim = mo.ui.slider(steps=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], value=64,
                            label="head width d", show_value=True)
    head_dim
    return (head_dim,)


@app.cell
def _(head_dim, jax, jnp, mo, np, plt):
    _ds = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    def _stats(d, key):
        kq, kk = jax.random.split(key)
        q = jax.random.normal(kq, (200, d))
        k = jax.random.normal(kk, (32, d))
        s = q @ k.T
        raw = jax.nn.softmax(s, -1).max(-1).mean()
        scaled = jax.nn.softmax(s / jnp.sqrt(d), -1).max(-1).mean()
        return float(s.std()), float(raw), float(scaled)

    _res = np.array([_stats(d, jax.random.key(i)) for i, d in enumerate(_ds)])
    _fig, _ax = plt.subplots(figsize=(6.4, 3.0))
    _ax.semilogx(_ds, _res[:, 1], "o-", label="unscaled: softmax(q·k)", base=2)
    _ax.semilogx(_ds, _res[:, 2], "s-", label="scaled: softmax(q·k / √d)", base=2)
    _ax.axhline(1 / 32, color="gray", lw=0.8, ls=":")
    _ax.text(2.2, 1 / 32 + 0.03, "uniform (1/32)", fontsize=8, color="gray")
    _ax.axvline(head_dim.value, color="C3", lw=1)
    _ax.set(xlabel="head width d", ylabel="largest weight (avg)", ylim=(0, 1.05),
            title="softmax saturation vs head width")
    _ax.legend(fontsize=8)
    _i = _ds.index(head_dim.value)
    mo.vstack([_fig, mo.md(
        f"At d = {head_dim.value}: unscaled scores have std **{_res[_i, 0]:.1f}** (≈ √d = "
        f"{np.sqrt(head_dim.value):.1f}); the biggest unscaled weight averages **{_res[_i, 1]:.2f}**, the "
        f"scaled one **{_res[_i, 2]:.2f}**. The scaled curve stays flat — that's the point.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · The causal mask

    A language model is trained to predict token $t+1$ from tokens $\le t$. If position 3 could attend to
    position 4, it would just copy the answer — perfect training loss, useless model. So we **mask** the
    future: set those scores to $-\infty$ *before* the softmax. $e^{-\infty} = 0$, so masked positions get
    exactly zero weight and the remaining weights still sum to 1.

    The mask is lower-triangular: row $t$ may see columns $0..t$. This is why one forward pass over a
    sequence of length $T$ trains $T$ predictions at once — each position only ever sees its past.

    **Your turn:** implement `causal_mask` and `scaled_dot_product` (with the optional mask) in
    `exercises.py`.
    """)
    return


@app.cell
def _(ex, lj, mo, np, plt, sent_words, sent_x):
    _m = lj.attempt(ex.causal_mask, 6)
    _sd = lj.attempt(ex.scaled_dot_product, sent_x, sent_x, sent_x, _m.value) if _m.ok else None
    _items = [mo.md("**`causal_mask(6)`:**"), _m.view(mo, show_value=False) if not _m.ok else
              mo.md("```\n" + "\n".join(" ".join("■" if b else "·" for b in row)
                                         for row in np.asarray(_m.value)) + "\n```")]
    if _sd is not None:
        if _sd.ok:
            _w = np.asarray(_sd.value[1])
            _fig, _ax = plt.subplots(figsize=(4.2, 3.6))
            _ax.imshow(_w, cmap="Blues", vmin=0, vmax=1)
            _ax.set(xticks=range(6), xticklabels=sent_words, yticks=range(6), yticklabels=sent_words,
                    title="causal self-attention weights")
            for (_i, _j), _v in np.ndenumerate(_w):
                _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if _v > 0.5 else "black")
            _items += [_fig, mo.md(
                f"The upper triangle is exactly zero (max = {_w[np.triu_indices(6, 1)].max():.1f}). The first "
                f"*the* can only see itself (weight {_w[0, 0]:.2f}); the second *the* (row 4) now splits between "
                f"the two *the*s and the tokens in between.")]
        else:
            _items.append(_sd.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "scaled dot-product attention (causal)", solves=("causal_mask", "scaled_dot_product"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Many heads

    One head computes one attention pattern. But a token may need several kinds of context at once —
    its subject, the previous word, the matching quote mark. **Multi-head attention** runs $H$ smaller
    heads in parallel, each with width $d = D/H$, and concatenates their outputs. Same parameter count as
    one wide head, but $H$ independent patterns.

    In code, "splitting into heads" is just a reshape — no data moves:

    | step | shape | code |
    |---|---|---|
    | input | (B, T, D) | `x` |
    | project | (B, T, D) | `x @ wq`, `x @ wk`, `x @ wv` |
    | split heads | (B, T, H, d) | `.reshape(B, T, H, D // H)` |
    | scores | (B, H, T, T) | `einsum("bthd,bshd->bhts", q, k) / √d` |
    | mask + softmax | (B, H, T, T) | `where(mask, scores, -inf)`, `softmax(-1)` |
    | mix values | (B, T, H, d) | `einsum("bhts,bshd->bthd", w, v)` |
    | merge heads + project | (B, T, D) | `.reshape(B, T, D) @ wo` |

    Those two `einsum` strings are **character for character** the ones in deepseek-v3-jax's `mla`
    function. Read `"bthd,bshd->bhts"` as: for each batch b and head h, dot query position t with key
    position s over the head feature d. The final $W_o$ lets heads talk to each other.

    **Your turn:** `split_heads`, `merge_heads`, `attention_weights`, `multi_head_attention`.
    """)
    return


@app.cell
def _(jax, np):
    def make_mha_params(key, D):
        ks = jax.random.split(key, 4)
        return {n: jax.random.normal(k, (D, D)) / np.sqrt(D) for n, k in zip(("wq", "wk", "wv", "wo"), ks)}

    mh_D, mh_H = 64, 4
    mh_params = make_mha_params(jax.random.key(15), mh_D)
    mh_x = jax.random.normal(jax.random.key(16), (2, 16, mh_D))
    return make_mha_params, mh_D, mh_H, mh_params, mh_x


@app.cell
def _(jax, mh_D, mh_H, mh_params, mh_x):
    def reference_mha(params, x, n_heads, causal=True):
        """The lesson's own reference, built on jax.nn.dot_product_attention (layout B, T, H, d)."""
        B, T, D = x.shape
        q, k, v = ((x @ params[n]).reshape(B, T, n_heads, D // n_heads) for n in ("wq", "wk", "wv"))
        out = jax.nn.dot_product_attention(q, k, v, is_causal=causal)
        return out.reshape(B, T, D) @ params["wo"]

    mh_reference = reference_mha(mh_params, mh_x, mh_H)
    assert mh_reference.shape == (2, 16, mh_D)
    return mh_reference, reference_mha


@app.cell
def _(ex, jnp, lj, mh_H, mh_params, mh_reference, mh_x, mo):
    _split = lj.attempt(ex.split_heads, mh_x, mh_H)
    _mha = lj.attempt(ex.multi_head_attention, mh_params, mh_x, mh_H, True)
    _items = [mo.md("**`split_heads(x, 4)`** on a (2, 16, 64) input:"),
              mo.md(f"shape `{tuple(_split.value.shape)}` ✅") if _split.ok else _split.view(mo)]
    if _mha.ok:
        _err = float(jnp.abs(_mha.value - mh_reference).max())
        _items.append(mo.md(
            f"**Your `multi_head_attention`** vs [`jax.nn.dot_product_attention`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html#jax.nn.dot_product_attention) (JAX's built-in, which takes "
            f"the same B, T, H, d layout): max difference **{_err:.1e}** "
            f"{'✅' if _err < 1e-4 else '— should be below 1e-4'}."))
    else:
        _items.append(_mha.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(Path, jax, make_mha_params, mo, np, reference_mha):
    # Test vectors for the PyTorch notebook (it can't import JAX), from the lesson's reference — not your code.
    _out_dir = Path("/course/runs/15-attention")
    _p = make_mha_params(jax.random.key(150), 64)
    _x = jax.random.normal(jax.random.key(151), (2, 32, 64))
    _arrays = {k: np.asarray(v) for k, v in _p.items()}
    _arrays.update(x=np.asarray(_x), n_heads=np.array(4),
                   out_causal=np.asarray(reference_mha(_p, _x, 4, True)),
                   out_full=np.asarray(reference_mha(_p, _x, 4, False)))
    try:
        _out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(_out_dir / "mha_reference.npz", **_arrays)
        _msg = f"Saved test vectors to `{_out_dir}/mha_reference.npz` for the PyTorch notebook (`make torch-lesson N=15`)."
    except OSError as _e:
        _msg = f"(Couldn't save test vectors for the PyTorch notebook: {_e})"
    mo.md(_msg)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Attention doesn't know word order

    Look back at the formula: scores depend on *what* the tokens are, never on *where* they are. Shuffle
    the input tokens and (without a mask) the outputs come out shuffled the same way — the layer is
    **permutation-equivariant**. "dog bites man" and "man bites dog" would look alike.

    The fix is to put position into the input. The original Transformer *adds* a fixed pattern of sines
    and cosines at different frequencies to each embedding; GPT-2 learns a position-embedding table
    instead. Modern models (Llama, DeepSeek) use **RoPE**, which rotates queries and keys by an angle
    proportional to position — Module 17. (The causal mask also leaks order, since position $t$ sees
    exactly $t+1$ tokens, but that's a weak signal to rely on.)

    **Your turn:** `sinusoidal_positions` and `is_permutation_equivariant`.
    """)
    return


@app.cell
def _(ex, jnp, lj, mh_H, mh_params, mh_x, mo, plt, reference_mha):
    _perm = [5, 2, 9, 0, 14, 1, 7, 3, 12, 8, 15, 4, 11, 6, 13, 10]
    _f = lambda z: reference_mha(mh_params, z, mh_H, causal=False)  # noqa: E731
    _direct = float(jnp.abs(_f(mh_x[:, jnp.array(_perm)]) - _f(mh_x)[:, jnp.array(_perm)]).max())
    _items = [mo.md(f"Shuffle 16 tokens, run full (unmasked) attention, compare with shuffling the output: "
                    f"max difference **{_direct:.1e}** — the same numbers, just reordered.")]
    _pe = lj.attempt(ex.sinusoidal_positions, 16, mh_x.shape[-1])
    if _pe.ok:
        _fig, _ax = plt.subplots(figsize=(6.2, 2.4))
        _ax.imshow(_pe.value, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
        _ax.set(xlabel="feature", ylabel="position", title="your sinusoidal_positions(16, 64)")
        _items.append(_fig)
        _eq = lj.attempt(ex.is_permutation_equivariant, _f, mh_x, _perm)
        _eq_pos = lj.attempt(ex.is_permutation_equivariant,
                             lambda z: reference_mha(mh_params, z + _pe.value, mh_H, causal=False), mh_x, _perm)
        if _eq.ok and _eq_pos.ok:
            _items.append(mo.md(
                f"`is_permutation_equivariant`: plain attention → **{_eq.value}**; with positions added → "
                f"**{_eq_pos.value}**. Low features (left) oscillate fast and tell neighbours apart; high "
                f"features change slowly and encode coarse position."))
        else:
            _items.append((_eq if not _eq.ok else _eq_pos).view(mo))
    else:
        _items.append(_pe.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · The price: $O(T^2)$

    The score matrix has one entry per (query, key) pair: $B \cdot H \cdot T^2$ numbers. Double the context
    and attention costs **4×** the compute and, if the scores are stored, 4× the memory — while the
    projections and the MLP only cost 2×. This is why context length is expensive, and why the KV cache
    (Module 18), MLA's compressed keys (Module 17), and fused "flash" kernels exist.

    Measured on your GPU: one head-batch of B = 1, H = 8, d = 64, float32, comparing a hand-written
    jitted version with `jax.nn.dot_product_attention`.
    """)
    return


@app.cell
def _(jax, jnp, lj, mo, np, plt):
    @jax.jit
    def _naive(q, k, v):
        s = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(q.shape[-1])
        s = jnp.where(jnp.tril(jnp.ones((q.shape[1], q.shape[1]), bool)), s, -jnp.inf)
        return jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(s, -1), v)

    _builtin = jax.jit(lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True))
    _Ts = [256, 512, 1024, 2048, 4096]
    _rows, _tn, _tb = [], [], []
    for _T in _Ts:
        _q, _k, _v = (jax.random.normal(jax.random.key(i), (1, _T, 8, 64)) for i in range(3))
        _a = lj.timeit(_naive, _q, _k, _v, iters=5)["median"]
        _b = lj.timeit(_builtin, _q, _k, _v, iters=5)["median"]
        _tn.append(_a)
        _tb.append(_b)
        _rows.append(f"| {_T:,} | {8 * _T * _T * 4 / 2**20:,.0f} MiB | {_a * 1e3:.2f} ms | {_b * 1e3:.2f} ms |")
    _fig, _ax = plt.subplots(figsize=(5.4, 3.2))
    _ax.loglog(_Ts, np.array(_tn) * 1e3, "o-", label="hand-written (jit)", base=2)
    _ax.loglog(_Ts, np.array(_tb) * 1e3, "s-", label="jax.nn.dot_product_attention", base=2)
    _ref = np.array(_Ts, float) ** 2
    _ax.loglog(_Ts, _ref / _ref[-1] * _tn[-1] * 1e3, "k:", label="∝ T²", base=2)
    _ax.set(xlabel="sequence length T", ylabel="ms", title="causal attention time (B=1, H=8, d=64)")
    _ax.legend(fontsize=8)
    _ratio = _tn[-1] / _tn[-2]
    mo.hstack([_fig, mo.md(f"""
    | T | score matrix | hand-written | built-in |
    |---|---|---|---|
    {chr(10).join(_rows)}

    From 2,048 to 4,096 tokens the time grew **{_ratio:.1f}×** (quadratic would be 4×). At short lengths
    fixed overheads hide the curve; at long lengths the $T^2$ term wins. On this ROCm build the built-in
    uses XLA (`implementation="xla"`; its fused `cudnn` path is NVIDIA-only), so the two are close. The
    PyTorch notebook (`make torch-lesson N=15`) shows what a fused FlashAttention kernel does on this same
    card: about 40× faster at T = 4,096, using a few percent of the memory.
    """)], justify="start", gap=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Watch attention learn where to look

    A task a fixed window can't do: **associative recall**. Each example shows 16 (key, value) pairs in
    random order, then a query key; the answer is the value paired with that key. The right pair can be
    anywhere, so the model must *find* it by content.

    The model is one attention head and nothing else: each pair is a token (one-hot key ⊕ one-hot value),
    the query token attends over the 16 pair tokens, and a linear layer reads the answer off the
    attention output. Chance is 1/16. Training takes a few seconds:
    """)
    return


@app.cell
def _(jax, jnp, optax, time):
    _NK, _NV, _NP, _D = 32, 16, 16, 64

    def recall_batch(key, B):
        k1, k2, k3 = jax.random.split(key, 3)
        keys = jax.vmap(lambda kk: jax.random.permutation(kk, _NK)[:_NP])(jax.random.split(k1, B))
        vals = jax.random.randint(k2, (B, _NP), 0, _NV)
        which = jax.random.randint(k3, (B,), 0, _NP)
        pairs = jnp.concatenate([jax.nn.one_hot(keys, _NK), jax.nn.one_hot(vals, _NV)], -1)
        query = jnp.concatenate([jax.nn.one_hot(keys[jnp.arange(B), which], _NK), jnp.zeros((B, _NV))], -1)
        return pairs, query, vals[jnp.arange(B), which], which

    def recall_init(key):
        ks = jax.random.split(key, 5)
        s = 1 / jnp.sqrt(_D)
        return {"emb": jax.random.normal(ks[0], (_NK + _NV, _D)) * 0.2,
                "wq": jax.random.normal(ks[1], (_D, _D)) * s, "wk": jax.random.normal(ks[2], (_D, _D)) * s,
                "wv": jax.random.normal(ks[3], (_D, _D)) * s, "out": jax.random.normal(ks[4], (_D, _NV)) * s}

    def recall_apply(p, pairs, query):
        hp, hq = pairs @ p["emb"], query @ p["emb"]
        q, k, v = hq @ p["wq"], hp @ p["wk"], hp @ p["wv"]
        w = jax.nn.softmax(jnp.einsum("bd,bsd->bs", q, k) / jnp.sqrt(_D), -1)
        return jnp.einsum("bs,bsd->bd", w, v) @ p["out"], w

    _opt = optax.adam(1e-3)

    def _loss(p, pairs, query, y):
        logits, _ = recall_apply(p, pairs, query)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def _train_chunk(p, s, key):
        def body(carry, k):
            p, s = carry
            pairs, query, y, _ = recall_batch(k, 256)
            l, g = jax.value_and_grad(_loss)(p, pairs, query, y)
            u, s = _opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), l
        (p, s), ls = jax.lax.scan(body, (p, s), jax.random.split(key, 20))
        return p, s, ls.mean()

    _p = recall_init(jax.random.key(0))
    _s = _opt.init(_p)
    _eval = recall_batch(jax.random.key(999), 2000)
    _logits0, _ = recall_apply(_p, _eval[0], _eval[1])
    recall_hist = [(0, float("nan"), float((_logits0.argmax(-1) == _eval[2]).mean()))]
    _t0 = time.perf_counter()
    for _i, _k in enumerate(jax.random.split(jax.random.key(1), 50)):
        _p, _s, _l = _train_chunk(_p, _s, _k)
        _logits, _ = recall_apply(_p, _eval[0], _eval[1])
        recall_hist.append(((_i + 1) * 20, float(_l), float((_logits.argmax(-1) == _eval[2]).mean())))
    recall_seconds = time.perf_counter() - _t0
    recall_params = _p
    recall_eval = _eval
    return recall_apply, recall_eval, recall_hist, recall_params, recall_seconds


@app.cell
def _(mo, np, plt, recall_apply, recall_eval, recall_hist, recall_params, recall_seconds):
    _h = np.array(recall_hist)
    _pairs, _query, _y, _which = recall_eval
    _, _w = recall_apply(recall_params, _pairs, _query)
    _w = np.asarray(_w)
    _hit = float((_w.argmax(-1) == np.asarray(_which)).mean())
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3.0), gridspec_kw={"width_ratios": [1, 1.3]})
    _a1.plot(_h[:, 0], _h[:, 2], "o-", ms=3)
    _a1.axhline(1 / 16, color="gray", ls=":", lw=0.8)
    _a1.set(xlabel="training step", ylabel="held-out accuracy", ylim=(0, 1.02), title="recall accuracy")
    _a2.imshow(_w[:8], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    _a2.scatter(np.asarray(_which)[:8], range(8), marker="o", facecolors="none", edgecolors="C3", s=160,
                linewidths=1.5, label="matching pair")
    _a2.set(xlabel="pair position (0–15)", ylabel="example", title="where the query looks", yticks=range(8))
    _a2.legend(fontsize=7, loc="upper right")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"After {int(_h[-1, 0]):,} steps ({recall_seconds:.1f} s): **{_h[-1, 2]:.1%}** held-out accuracy (chance "
        f"6.25%). The query's biggest attention weight lands on the matching pair in **{_hit:.1%}** of "
        f"examples — the red circles sit on the dark cells. Nobody told the model where to look: it learned "
        f"$W_q$ and $W_k$ so that a key's query and the pair holding that key point the same way.")])
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "To keep softmax from saturating, so gradients still flow": "a",
            "To make the attention weights sum to 1": "b",
            "To make attention cheaper to compute": "c",
        },
        label="**Check yourself:** why are attention scores divided by √d?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right. q·k grows like √d; without scaling the softmax becomes nearly one-hot "
                              "and its gradient vanishes."), kind="success"),
        "b": mo.callout(mo.md("Softmax already sums to 1 whatever you divide by. The scaling controls how "
                              "*sharp* the weights are."), kind="warn"),
        "c": mo.callout(mo.md("It's one extra multiply — it doesn't change the cost. It changes the scale of "
                              "the scores so softmax stays in a trainable range."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(15))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — §3.2 is this
      module in four paragraphs.
    * Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
    * Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
      — the "self-attention" segment builds the same masked weighted average.
    * Anthropic, [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
      — what attention heads actually learn (induction heads are a two-layer version of §9).
    * Tri Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) — computing attention without
      ever storing the $T^2$ score matrix.

    **Next:** Module 16 — stack attention with an MLP, residuals and norms into a Transformer block, and
    train a GPT on your GPU.
    """)
    return


if __name__ == "__main__":
    app.run()
