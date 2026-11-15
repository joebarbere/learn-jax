import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="17 · Modern LLM architecture")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import time

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax

    import learnjax as lj
    from learnjax.data import tiny_shakespeare

    ex = lj.load_exercises(mo.notebook_dir())
    _spec = importlib.util.spec_from_file_location("m17_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    return M, ex, jax, jnp, lj, mo, np, optax, plt, time, tiny_shakespeare


@app.cell
def _(mo):
    mo.md(r"""
    # 17 · Modern LLM architecture: RoPE, RMSNorm, SwiGLU, GQA, KV cache, MLA, MoE

    **Goal:** know every part of a 2025-era LLM block well enough to build it, test it, and spot it in
    real code — by building two reference designs from the *same* parts and racing them.

    | | **Llama-3-style** | **DeepSeek-V3-style** (the model you trained) |
    |---|---|---|
    | normalization | RMSNorm, pre-norm | RMSNorm, pre-norm |
    | positions | RoPE | RoPE (on a small slice of each head) |
    | attention | grouped-query attention (GQA) | Multi-head Latent Attention (MLA) |
    | feed-forward | SwiGLU | DeepSeekMoE: shared + routed SwiGLU experts |
    | extra | — | aux-loss-free load balancing, multi-token prediction |

    **You will:** implement RMSNorm · RoPE twice (complex numbers, and the rotate-half form your
    deepseek-v3-jax code uses) and prove they're the same thing · SwiGLU · GQA with a KV cache, and test
    cached decoding against the full forward pass · fix two real KV-cache bugs from a published tutorial ·
    derive why MLA caches 144 numbers per token instead of 480 · route tokens to experts and balance them
    with DeepSeek's bias trick · then assemble a tiny Llama and a tiny DeepSeek from your parts and train both ·
    (stretch) test how much RoPE a model needs: partial, hybrid and none.

    Module 15 built attention and Module 16 a GPT. Everything here is a refinement of that GPT block —
    each one solving a specific problem that showed up at scale. Code excerpts quoted below are from
    Vaibhaav Tiwari's [deepseek-v3-jax](https://github.com/Vaibhaav-Tiwari/deepseek-v3-jax) (MIT), which you replicated.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    After ChatGPT, open-weights models took off. Meta's LLaMA (Touvron et al., 2023) combined RMSNorm, RoPE
    (Su et al., 2021) and SwiGLU (Shazeer, 2020) into the recipe most open models copy, and Llama 2 (2023)
    used grouped-query attention (Ainslie et al., 2023) in its largest model. Mixture-of-experts goes back
    to Shazeer et al. (2017) and Switch Transformers (2021); DeepSeek-V2 (2024) introduced multi-head latent
    attention, and DeepSeek-V3 (December 2024) added auxiliary-loss-free balancing and multi-token
    prediction.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · RMSNorm: LayerNorm minus the parts that don't matter

    LayerNorm (Module 11) does two things per token: **re-centre** (subtract the mean) and **re-scale**
    (divide by the standard deviation), then applies a learned scale $\gamma$ and shift $\beta$.
    Zhang & Sennrich (2019) found the re-centring and $\beta$ contribute almost nothing in transformers —
    the re-scaling is what stabilizes training. RMSNorm keeps only that:

    $$\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}}\; w$$

    One reduction instead of two, no bias, and in a pre-norm residual network the activations' mean is
    small anyway, so the two are nearly identical in practice. In deepseek_v3.py it's one line:

    ```python
    def rms_norm(x, w, eps=1e-6):
        return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * w
    ```
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np):
    _x = jax.random.normal(jax.random.key(0), (4, 256)) * 3.0 + 0.05      # near-zero-mean activations
    _rn = lj.attempt(ex.rms_norm, _x, jnp.ones(256))
    _items = [mo.md("### Your turn: `rms_norm`"), _rn.view(mo, show_value=False)]
    if _rn.ok:
        _ln = (_x - _x.mean(-1, keepdims=True)) / jnp.sqrt(_x.var(-1, keepdims=True) + 1e-6)
        _diff = float(jnp.abs(_rn.value - _ln).max())
        _rms = np.asarray(jnp.sqrt(jnp.mean(_rn.value**2, -1)))
        _items.append(mo.md(
            f"Output RMS per row: {', '.join(f'{r:.4f}' for r in _rms)} (should be 1). On activations with a "
            f"small mean, RMSNorm and LayerNorm differ by at most **{_diff:.3f}** here — the mean was "
            f"doing almost nothing."))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · RoPE: position as rotation

    Attention scores are dot products $q \cdot k$, and a dot product has no idea *where* the two tokens
    are. GPT-2 (and Module 16) added a learned position vector to each token. **Rotary position
    embedding** (Su et al., 2021) instead *rotates* $q$ and $k$ by an angle proportional to their
    position, using pairs of features as 2-D planes:

    * Split the $D$ features of a head into $D/2$ pairs. Pair $i$ rotates at frequency
      $\theta_i = \text{base}^{-2i/D}$: pair 0 spins fast (1 radian per token), the last pair barely moves.
    * A token at position $p$ has pair $i$ rotated by angle $p\,\theta_i$ — like clock hands, each ticking
      at a different speed.
    * Rotating $q$ by $m\theta$ and $k$ by $n\theta$ leaves their dot product depending only on
      $(m-n)\theta$: **attention sees relative position**, for free, with no extra parameters.

    **Two ways to write it.** Treat each pair $(a, b)$ as a complex number $a + ib$; rotating by $\phi$ is
    multiplying by $e^{i\phi}$. That's the style of Saurabh Alone's ["Implementing LLaMA3 in 100 Lines of Pure Jax"](https://saurabhalone.com/blog/llama3-jax) (2025), and it pairs
    **neighbouring** features $(x_0,x_1), (x_2,x_3), \dots$ — the *interleaved* convention. deepseek_v3.py
    instead pairs feature $i$ with feature $i + D/2$ — the *halves* ("rotate-half") convention — and uses
    only cos and sin:

    ```python
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    ```

    Same rotations, different pairing. Implement both.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _x = jax.random.normal(jax.random.key(1), (2, 16, 4, 64))
    _pos = jnp.arange(16)
    rope_c = lj.attempt(ex.rope_complex, _x, _pos)
    rope_h = lj.attempt(ex.rope_rotate_half, _x, _pos)
    _items = [mo.md("### Your turn: `rope_complex` and `rope_rotate_half`"),
              mo.md("`rope_complex`:"), rope_c.view(mo, show_value=False),
              mo.md("`rope_rotate_half`:"), rope_h.view(mo, show_value=False)]
    if rope_c.ok and rope_h.ok:
        _to_halves = lambda a: jnp.concatenate([a[..., 0::2], a[..., 1::2]], -1)  # noqa: E731
        _naive = float(jnp.abs(rope_c.value - rope_h.value).max())
        _perm = float(jnp.abs(ex.rope_rotate_half(_to_halves(_x), _pos) - _to_halves(rope_c.value)).max())
        _norm = float(jnp.abs(jnp.linalg.norm(rope_h.value, axis=-1) - jnp.linalg.norm(_x, axis=-1)).max())
        _items.append(mo.md(f"""
| check | max abs difference |
|---|---|
| complex vs rotate-half, compared directly | {_naive:.3f} — *not* equal |
| after reordering features (evens, then odds) | **{_perm:.1e}** — identical |
| vector length before vs after rotation | {_norm:.1e} — rotations preserve length |

The two conventions are the same rotation applied to a different pairing of features. That matters in
practice: a checkpoint trained with one convention needs its `wq`/`wk` columns permuted to run with the
other — exactly what Hugging Face's Llama conversion script does.
"""))
    mo.vstack(_items)
    return (rope_h,)


@app.cell
def _(ex, jax, jnp, plt, rope_h):
    _fig = None
    if rope_h.ok:
        _q = jax.random.normal(jax.random.key(2), (1, 1, 1, 64))
        _k = jax.random.normal(jax.random.key(3), (1, 1, 1, 64))
        _P = 64
        _pos = jnp.arange(_P)
        _qs = ex.rope_rotate_half(jnp.broadcast_to(_q, (1, _P, 1, 64)), _pos)[0, :, 0]   # (P, 64)
        _ks = ex.rope_rotate_half(jnp.broadcast_to(_k, (1, _P, 1, 64)), _pos)[0, :, 0]
        _S = _qs @ _ks.T                                                                   # (m, n)
        _fig, _ax = plt.subplots(figsize=(4.6, 3.8))
        _im = _ax.imshow(_S, cmap="RdBu_r")
        _ax.set(title="q·k after RoPE, query pos m vs key pos n", xlabel="key position n",
                ylabel="query position m")
        _fig.colorbar(_im, ax=_ax, shrink=0.8)
        _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The **same** query and key vectors, placed at every pair of positions. Every diagonal (constant
    $m - n$) has a single colour: the score depends only on the offset, never on absolute position.
    That's the relative-position property — and it's why a model trained on short sequences has a
    fighting chance on longer ones.

    ### The base: 10,000 vs 500,000

    The original RoPE (and deepseek-v3-jax) uses base 10,000; Llama 3 uses **500,000**. The slowest pair's
    *wavelength* — tokens until it completes a full turn — sets the longest distance the model can tell
    apart without the angle wrapping around. Below, wavelength $2\pi/\theta_i$ for each pair of a
    128-wide head:
    """)
    return


@app.cell
def _(np, plt):
    _D = 128
    _i = np.arange(_D // 2)
    _fig, _ax = plt.subplots(figsize=(6.4, 3.2))
    for _base in (10_000, 500_000):
        _ax.plot(_i, 2 * np.pi * _base ** (2 * _i / _D), label=f"base {_base:,}")
    for _ctx, _ls in ((8192, ":"), (131072, "--")):
        _ax.axhline(_ctx, color="gray", ls=_ls, lw=1)
        _ax.text(1, _ctx * 1.3, f"{_ctx:,}-token context", fontsize=8, color="gray")
    _ax.set(yscale="log", xlabel="feature pair i", ylabel="wavelength (tokens)",
            title="RoPE wavelengths, head dim 128")
    _ax.legend(loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    With base 10,000 only the last handful of pairs have wavelengths beyond an 8K context, and none come
    close to 128K. Raising the base slows every pair down, so more of them stay unambiguous over long
    distances — that (plus training on long documents) is how Llama 3 stretched to 128K tokens.
    Your 256-token deepseek-v3-jax model never needed it.

    ### Beyond RoPE: all of it, some of it, or none

    RoPE in every layer is still the most common choice, but it isn't the only one, and newer models
    increasingly use less of it. Here is every scheme in one place, including the two you already built:

    | scheme | how position gets in | who uses it |
    |---|---|---|
    | learned absolute | a trained vector per position, added to the token embedding; the table caps the context | GPT-2 (Module 16), BERT (Module 32) |
    | sinusoidal | fixed sin/cos vectors, added to the token embedding | the original Transformer (Module 15) |
    | relative bias | a learned number per head and distance bucket, added to the attention scores | T5 |
    | ALiBi | a fixed penalty proportional to distance, added to the attention scores | BLOOM, MPT |
    | RoPE in every layer | rotate $q$ and $k$ (this section) | Llama 2 and 3, Qwen2.5, Mistral |
    | partial RoPE | rotate only a slice of each head; the rest carries no position | GPT-NeoX/Pythia (25%), Phi-2 (40%), DeepSeek-V3's MLA (64 of 192 q/k features, section 7) |
    | hybrid | RoPE in some layers, nothing in others | SmolLM3 and Llama 4 (no RoPE in every 4th layer), Cohere Command A (no positions in its global-attention layers) |
    | NoPE | nothing at all: only the causal mask | research models; the attention layers of some recurrent hybrids |

    **How can NoPE work at all?** The causal mask leaks position. A query at position $p$ averages over
    exactly $p + 1$ values, so if it attends evenly to everything, a feature that is 1 only on the first
    token comes out as $1/(p+1)$, and the model has found its own position (Haviv et al., 2022). What the
    mask can't do directly is tell a key 1 token back from one 10 tokens back. Something has to learn that.

    **Why use less RoPE?**

    * **Learned tables don't scale.** In GPT-2 Medium (width 1,024) the position table is 0.30% of the
      weights at 1,024 positions. At a 100K-token context it would be over 22%, as Sebastian Raschka works out
      in [Understanding Positional Embeddings in Modern LLMs](https://magazine.sebastianraschka.com/p/understanding-positional-embeddings) (2026).
    * **Partial RoPE** leaves some features free to match on *content*, whatever the distance. Raschka cites
      a 2026 study in which rotating about 10% of each head often matched full RoPE at 1–8B parameters,
      though full RoPE won some evaluations. In his view, partial RoPE is "hardly worth it". In his survey of
      open-weight models, partial RoPE nonetheless went from a minority of RoPE models in 2025 to a majority
      in 2026.
    * **Hybrids** give some layers free rein over long distances. Hugging Face reports that SmolLM3's
      3-in-4 pattern improved long-context performance. SmolLM3 is the bigger sibling of the SmolLM2 models
      you'll run in Modules 18–21.

    You'll test partial, hybrid and NoPE yourself at the end of section 10 (stretch exercise
    `rope_partial`).

    **Aside: why rotations, of all things?** We want each position $p$ to transform vectors by some
    $R(p)$ so that $\langle R(m)q,\, R(n)k\rangle$ depends only on $n - m$. That holds when the $R(p)$ are
    orthogonal and compose like positions add, $R(a + b) = R(a)\,R(b)$, because then
    $q^\top R(m)^\top R(n)\,k = q^\top R(n - m)\,k$. In group-theory terms, that's an orthogonal
    representation of the translation group. Every such continuous family splits into independent 2-D
    rotation blocks, each spinning at its own frequency, plus coordinates that never move. So RoPE's
    "clock hands" aren't one design among many. They're essentially the *only* way to get relative position
    out of a dot product, and the coordinates that never move are partial RoPE. (This view came up in the
    [Hacker News discussion](https://news.ycombinator.com/item?id=50506376) of Raschka's article.)

    ## 3 · SwiGLU: a feed-forward block with a gate

    The original transformer FFN is `relu(x @ W1) @ W2`. SwiGLU (Shazeer, 2020) adds a third matrix and
    multiplies elementwise by a *gate*:

    $$\text{SwiGLU}(x) = W_2\big(\operatorname{silu}(x W_1) \odot x W_3\big)$$

    The SiLU branch decides, per hidden unit, how much of the linear branch passes through. Three
    matrices instead of two, so to keep the parameter count equal the hidden size shrinks from $4d$ to
    $\tfrac{2}{3}\cdot 4d$: a standard FFN has $2 \cdot d \cdot 4d = 8d^2$ weights, SwiGLU has
    $3 \cdot d \cdot h$, equal when $h = 8d/3$. (Llama 3 8B uses 14,336 for $d$ = 4,096 — a bit more.)

    One trap: Meta's code calls the gate `w1` and the up-projection `w3`; some tutorials swap the names.
    The math is the same, but loading real weights with the names swapped silently breaks the model.
    deepseek_v3.py follows Meta:

    ```python
    def swiglu(p, x):
        return (jax.nn.silu(x @ p["w1"]) * (x @ p["w3"])) @ p["w2"]
    ```
    """)
    return


@app.cell
def _(ex, jax, lj, mo):
    _d, _h = 128, 352
    _k = jax.random.split(jax.random.key(4), 4)
    _p = {"w1": jax.random.normal(_k[0], (_d, _h)) * 0.05, "w3": jax.random.normal(_k[1], (_d, _h)) * 0.05,
          "w2": jax.random.normal(_k[2], (_h, _d)) * 0.05}
    _sw = lj.attempt(ex.swiglu, _p, jax.random.normal(_k[3], (8, _d)))
    _items = [mo.md("### Your turn: `swiglu`"), _sw.view(mo, show_value=False)]
    if _sw.ok:
        _items.append(mo.md(
            f"Output shape {tuple(_sw.value.shape)}. With $d$ = {_d}, a 4d ReLU FFN has "
            f"{2 * _d * 4 * _d:,} weights; SwiGLU with $h$ = {_h} has {3 * _d * _h:,} — matched to within 3%."))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Grouped-query attention: share keys and values between heads

    When an LLM generates text it keeps every previous token's **keys and values** in a *KV cache*
    (Section 5), so each new token only computes its own. That cache, not the weights, is what fills GPU
    memory on long contexts. Multi-head attention stores a key and a value **per head**.

    **GQA** (Ainslie et al., 2023) gives the model fewer key/value heads than query heads: with 32 query
    heads and 8 KV heads, each KV head is shared by a group of 4 query heads. The cache shrinks 4×; quality
    barely moves. At the extremes: 1 KV head is **MQA** (multi-query), as many as query heads is plain MHA.

    Implementation is one extra step — repeat each KV head so the shapes line up again:
    `repeat_kv(k, H // Hkv)`, then ordinary attention. Implement `repeat_kv`, then `gqa_attention`
    with RoPE and an optional cache (the docstring walks through the steps).
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    gqa_cfg = {"n_heads": 4, "n_kv_heads": 2, "head_dim": 16, "rope_theta": 10000.0}
    _k = jax.random.split(jax.random.key(5), 5)
    gqa_params = {"wq": jax.random.normal(_k[0], (64, 64)) * 0.1, "wk": jax.random.normal(_k[1], (64, 32)) * 0.1,
                  "wv": jax.random.normal(_k[2], (64, 32)) * 0.1, "wo": jax.random.normal(_k[3], (64, 64)) * 0.1}
    gqa_x = jax.random.normal(_k[4], (2, 12, 64))
    _rep = lj.attempt(ex.repeat_kv, jnp.arange(2 * 3 * 2 * 4.0).reshape(2, 3, 2, 4), 2)
    gqa_full = lj.attempt(ex.gqa_attention, gqa_params, gqa_x, jnp.arange(12), gqa_cfg)
    _items = [mo.md("### Your turn: `repeat_kv` and `gqa_attention`"),
              mo.md("`repeat_kv` on a (2, 3, 2, 4) array with n_rep=2:"), _rep.view(mo, show_value=False),
              mo.md("`gqa_attention` on (2, 12, 64) with 4 query heads, 2 KV heads:"),
              gqa_full.view(mo, show_value=False)]
    if _rep.ok:
        _items.append(mo.md(f"`repeat_kv` → shape {tuple(_rep.value.shape)}; heads 0,1 are copies of KV head 0: "
                            f"{bool(jnp.array_equal(_rep.value[:, :, 0], _rep.value[:, :, 1]))}"))
    if gqa_full.ok:
        _out, _cache = gqa_full.value
        _items.append(mo.md(f"Output {tuple(_out.shape)}; cache keys {tuple(_cache['k'].shape)} — the cache "
                            f"holds **2** heads, not 4."))
    mo.vstack(_items)
    return gqa_cfg, gqa_full, gqa_params, gqa_x


@app.cell
def _(ex, mo):
    _rows = [
        # name, layers, cached KV heads (MHA/GQA/MQA) or None, k_dim, v_dim, mla cfg or None
        ("Llama-3-8B as plain MHA (32 KV heads)", 32, 32, 128, 128, None),
        ("Llama-3-8B, actual: GQA (8 KV heads)", 32, 8, 128, 128, None),
        ("Llama-3-8B as MQA (1 KV head)", 32, 1, 128, 128, None),
        ("DeepSeek-V3 as plain MHA (128 heads)", 61, 128, 128 + 64, 128, None),
        ("DeepSeek-V3, actual: MLA", 61, None, 0, 0, {"kv_rank": 512, "d_rope": 64}),
        ("your deepseek-v3-jax as MHA (6 heads)", 6, 6, 32 + 16, 32, None),
        ("your deepseek-v3-jax, actual: MLA", 6, None, 0, 0, {"kv_rank": 128, "d_rope": 16}),
    ]
    try:
        _lines = ["| model / attention | numbers per token per layer | KV cache per token (bf16) | tokens in 8 GiB |",
                  "|---|---|---|---|"]
        for _name, _L, _hkv, _kd, _vd, _mla in _rows:
            _n = ex.mla_cache_floats(_mla) if _mla else ex.mha_cache_floats(_hkv, _kd, _vd)
            _bytes = _n * _L * 2
            _size = f"{_bytes / 2**20:.2f} MiB" if _bytes >= 2**20 else f"{_bytes / 2**10:.1f} KiB"
            _lines.append(f"| {_name} | {_n:,} | {_size} | {8 * 2**30 // _bytes:,} |")
        _out = mo.vstack([mo.md("### KV-cache memory, real models"), mo.md("\n".join(_lines)), mo.md(
            "GQA buys Llama 3 a 4× smaller cache than MHA. MLA (Section 7) buys DeepSeek-V3 about **70×** — "
            "at 128 heads, plain MHA would need almost 5 MiB per token, so 8 GiB of cache would hold fewer "
            "than 2,000 tokens of context for a single user.")])
    except NotImplementedError:
        _out = mo.callout(mo.md("**Your turn:** implement `mha_cache_floats` and `mla_cache_floats` in "
                                "`exercises.py` to fill in this table."), kind="info")
    except Exception as _e:  # noqa: BLE001 - show learner errors instead of crashing the notebook
        _out = mo.callout(mo.md(f"Your cache functions raised `{type(_e).__name__}: {_e}`"), kind="danger")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · The KV cache: generate without recomputing the past

    Generating token $t+1$ needs attention from the newest query to **all** previous keys and values.
    Without a cache you'd re-run the whole prefix through every layer for each new token — the work for a
    500-token reply grows with the *square* of its length. With a cache you compute $q, k, v$ for the one
    new token, append $k, v$, and attend. Each step still reads the whole cache (attention is linear in
    context length per token), but nothing is recomputed.

    The rule every implementation must satisfy: **cached decoding gives the same outputs as one full
    forward pass.** Below, your `gqa_attention` runs the 12-token sequence all at once, then as a 5-token
    prefill followed by 7 one-token decode steps:
    """)
    return


@app.cell
def _(ex, gqa_cfg, gqa_full, gqa_params, gqa_x, jnp, lj, mo):
    def _decode():
        _out, _cache = ex.gqa_attention(gqa_params, gqa_x[:, :5], jnp.arange(5), gqa_cfg)
        _outs = [_out]
        for _t in range(5, 12):
            _o, _cache = ex.gqa_attention(gqa_params, gqa_x[:, _t:_t + 1], jnp.array([_t]), gqa_cfg, _cache)
            _outs.append(_o)
        return jnp.concatenate(_outs, axis=1), _cache

    _dec = lj.attempt(_decode) if gqa_full.ok else None
    if _dec is None:
        _view = mo.md("*(Finish `gqa_attention` above first.)*")
    elif _dec.ok:
        _err = float(jnp.abs(_dec.value[0] - gqa_full.value[0]).max())
        _view = mo.md(f"Max difference, cached vs full: **{_err:.1e}** (float32 rounding) — final cache holds "
                      f"{_dec.value[1]['k'].shape[1]} positions. "
                      + ("✅ cached decoding matches." if _err < 1e-4 else "❌ these should match — check the "
                         "mask uses absolute positions and the cache is appended along the sequence axis."))
    else:
        _view = _dec.view(mo)
    _view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Find the bugs

    `buggy_cached_attention` in your `exercises.py` is adapted from
    Saurabh Alone's ["Implementing LLaMA3 in 100 Lines of Pure Jax"](https://saurabhalone.com/blog/llama3-jax) (2025), whose RoPE explanation is excellent. It looks reasonable and contains **two real bugs**. Here it is, run as a 5-token
    prefill then decode steps, against the correct all-at-once answer:

    ```python
    k = jnp.concatenate([cache_k, k_new], axis=-1)
    v = jnp.concatenate([cache_v, v_new], axis=-1)
    T, S = q.shape[1], k.shape[1]
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(q.shape[-1])
    mask = jnp.tril(jnp.ones((max_len, max_len), dtype=bool))[:T, :S]
    scores = jnp.where(mask, scores, -jnp.inf)
    ```
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _B, _L, _H, _D = 1, 9, 2, 8
    _q, _k, _v = (jax.random.normal(jax.random.key(10 + _i), (_B, _L, _H, _D)) for _i in range(3))
    _allowed = jnp.tril(jnp.ones((_L, _L), bool))
    _s = jnp.where(_allowed, jnp.einsum("bthd,bshd->bhts", _q, _k) / jnp.sqrt(_D), -jnp.inf)
    _truth = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(_s, -1), _v)

    def _run(fn):
        _ck, _cv = _k[:, :5], _v[:, :5]
        _outs = []
        for _t in range(5, _L):
            _o, (_ck, _cv) = fn(_q[:, _t:_t + 1], _k[:, _t:_t + 1], _v[:, _t:_t + 1], _ck, _cv)
            _outs.append(_o)
        return float(jnp.abs(jnp.concatenate(_outs, 1) - _truth[:, 5:]).max())

    _buggy = lj.attempt(_run, ex.buggy_cached_attention)
    _fixed = lj.attempt(_run, ex.fix_buggy_cached_attention)
    _items = [mo.md("**The buggy version:**")]
    _items.append(mo.md(f"error vs truth: {_buggy.value:.3f}") if _buggy.ok else mo.callout(
        mo.md(f"It crashes:\n\n```\n{_buggy.error.strip().splitlines()[-1][:300]}\n```"), kind="danger"))
    _items.append(mo.md("**Your `fix_buggy_cached_attention`:**"))
    if _fixed.ok:
        _items.append(mo.md(f"max error vs truth: **{_fixed.value:.1e}** " + (
            "✅ both bugs fixed." if _fixed.value < 1e-4 else
            "— it runs, so bug 1 is fixed, but the numbers are wrong. That's bug 2: a silent one.")))
    else:
        _items.append(_fixed.view(mo))
    _items.append(mo.accordion({"Hint 1": mo.md(
        "Keys are shaped (B, S, H, D). Which axis grows as tokens are added?"),
        "Hint 2": mo.md(
        "During decode T = 1, so `[:T, :S]` takes row 0 of the causal mask — the row for the *first* token "
        "of the sequence. Which row belongs to a query at absolute position P + t?"),
        "Why tests catch this": mo.md(
        "Bug 1 crashes, so you'd find it. Bug 2 produces plausible-looking text from a model that, during "
        "decode, can only attend to the very first token. The cached-vs-full comparison is the test that "
        "catches it — write it for any attention code you touch.")}))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Multi-head Latent Attention: cache a compressed latent

    GQA shrinks the cache by sharing heads. DeepSeek's **MLA** shrinks it by *compressing*: each token's
    keys and values for **all** heads are generated from one small latent vector $c$ (width `kv_rank`),
    and only $c$ is cached. At attention time, `wkv_b` decompresses $c$ back into per-head keys and values.

    One complication: RoPE rotates keys by position, and a rotation can't be pushed through the
    decompression matrix. So MLA splits each head's query/key into a **"nope" part** (no position, derived
    from $c$) and a small **RoPE part**, and the RoPE key $k_r$ is a single vector **shared by all heads**,
    cached alongside $c$. From deepseek_v3.py:

    ```python
    kv_a = x @ p["wkv_a"]                                          # [B,T,kv_rank+dr]
    c, kr = rms_norm(kv_a[..., :cfg.kv_rank], p["kv_norm"]), apply_rope(kv_a[..., cfg.kv_rank:], cos, sin)
    ...
    kv = (c @ p["wkv_b"]).reshape(B, S, H, dn + dv)                # decompress latent into per-head K and V
    ```

    For your trained model's config (6 heads, `d_nope` 32, `d_rope` 16, `d_v` 32, `kv_rank` 128):

    * plain MHA would cache, per token per layer, a key of $32 + 16$ and a value of $32$ for each of 6 heads:
      $6 \times (48 + 32) = 480$ numbers;
    * MLA caches $c$ and $k_r$: $128 + 16 = 144$ numbers — 3.3× less. That 144-vs-480 is checked in
      deepseek-v3-jax's own test suite. Your `mha_cache_floats`/`mla_cache_floats` filled in the table in
      Section 4.

    (Production MLA also "absorbs" `wkv_b` into the query and output projections so the latent is never
    decompressed at decode time. deepseek-v3-jax skips that optimization for clarity.)
    """)
    return


@app.cell
def _(ex, lj, mo):
    _a = lj.attempt(ex.mla_cache_floats, {"kv_rank": 128, "d_rope": 16})
    _b = lj.attempt(ex.mha_cache_floats, 6, 48, 32)
    mo.vstack([mo.md("### Your turn: `mla_cache_floats` and `mha_cache_floats` (deepseek-v3-jax config)"),
               mo.md("MLA:"), _a.view(mo), mo.md("MHA:"), _b.view(mo)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Mixture of experts: many FFNs, few used per token

    A dense FFN applies all its weights to every token. **DeepSeekMoE** replaces it with:

    * **shared experts** (here 1) that every token uses, plus
    * **routed experts** (16 in your model) of which each token uses only the top $k$ (4) — chosen by a
      **router**: a linear layer + sigmoid giving each token an affinity score per expert.

    The model gets the *capacity* of all experts with the *per-token compute* of $k$ of them — your 36.2M
    model only uses 13.6M parameters per token. The gate weights are the picked experts' scores,
    normalized to sum to 1.

    **Dense vs sparse compute.** deepseek_v3.py runs *every* expert on *every* token and zeroes the unpicked
    ones (`einsum("nd,edh->neh", ...)`) — simple, jit-friendly, and on a small model fast enough. It wastes
    $(E-k)/E$ of the FFN FLOPs. Real systems **dispatch**: sort tokens by expert, run each expert only on its
    tokens ([`jax.lax.ragged_dot`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html#jax.lax.ragged_dot)), and with expert parallelism put different experts on different GPUs
    (Module 24).

    **Load balancing.** Left alone, routing collapses: an expert that's slightly better gets picked more,
    trains more, gets better, gets picked more. The classic fix is an **auxiliary loss** penalizing
    imbalance — but it pulls gradients away from the language-modeling objective. DeepSeek-V3's fix
    touches no gradients: a per-expert **bias** added to the scores *only when choosing* the top $k$, nudged
    after every step:

    $$b_e \leftarrow b_e + \gamma \cdot \operatorname{sign}(\overline{\text{load}} - \text{load}_e)$$

    ```python
    _, idx = jax.lax.top_k(scores + jax.lax.stop_gradient(bias), cfg.top_k)   # bias decides WHO is picked...
    picked = jax.nn.one_hot(idx, cfg.n_routed).sum(-2) > 0
    gated = jnp.where(picked, scores, 0.0)                                     # ...but gate values use raw scores
    ```
    """)
    return


@app.cell
def _(ex, jnp, lj, mo):
    _scores = jnp.array([[0.91, 0.12, 0.85, 0.30], [0.40, 0.65, 0.55, 0.20]])
    _r0 = lj.attempt(ex.moe_route, _scores, jnp.zeros(4), 2)
    _r1 = lj.attempt(ex.moe_route, _scores, jnp.array([0.0, 0.8, 0.0, 0.0]), 2)
    _ub = lj.attempt(ex.update_bias, jnp.zeros(4), jnp.array([0.40, 0.10, 0.30, 0.20]), 0.01)
    _items = [mo.md("### Your turn: `moe_route` and `update_bias`"),
              mo.md("Scores for 2 tokens × 4 experts: `[[0.91, 0.12, 0.85, 0.30], [0.40, 0.65, 0.55, 0.20]]`, top-2.")]
    if _r0.ok and _r1.ok:
        _fmt = lambda a: "<br>".join("[" + ", ".join(f"{v:.3f}" for v in row) + "]" for row in a.tolist())  # noqa: E731
        _items.append(mo.md(f"""
| bias | gates |
|---|---|
| none | {_fmt(_r0.value)} |
| +0.8 on expert 1 | {_fmt(_r1.value)} |

With the bias, token 0 now picks expert 1 — yet its gate is still its raw score 0.12, normalized
(0.12 / (0.91 + 0.12) ≈ {0.12 / 1.03:.3f}). The bias changes routing, never the output weights.
"""))
    else:
        _items.append(_r0.view(mo))
    if _ub.ok:
        _items.append(mo.md(f"`update_bias` with loads [0.40, 0.10, 0.30, 0.20], γ = 0.01 → "
                            f"{[round(float(v), 3) for v in _ub.value]}: the busiest expert is pushed down."))
    else:
        _items.append(_ub.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Multi-token prediction (briefly)

    DeepSeek-V3 also trains a small extra block to predict the token **after next**: it takes the main
    model's hidden state at position $t$, mixes in the embedding of token $t+1$, runs one more transformer
    block, and predicts $t+2$ with the shared output head. Its loss is added with a small weight (0.3 in
    deepseek-v3-jax). It gives the model a denser training signal, and at inference the extra head can
    propose a draft token for **speculative decoding** (Module 18). From deepseek_v3.py's `loss_fn`:

    ```python
    e = params["embed"][tokens[:, k:T + k]]                                  # shared embedding
    h = jnp.concatenate([rms_norm(h, m["norm_h"]), rms_norm(e, m["norm_e"])], -1) @ m["proj"]
    h, _, _ = block(m["block"], None, h, cos, sin, cfg)
    ```

    The two tiny models below skip MTP to stay short.

    ## 10 · Assemble both and race them

    `models.py` (next to this notebook — read it) builds both architectures from **your** functions:
    `rms_norm`, `rope_rotate_half`, `swiglu`, `gqa_attention` for Llama; `rms_norm`, `rope_rotate_half`,
    `swiglu`, `moe_route`, `update_bias` plus an MLA layer adapted from deepseek_v3.py for DeepSeek. Both
    are 4 layers of width 128, trained on tiny-shakespeare characters for 600 steps.
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lj, mo, np, tiny_shakespeare):
    _text = tiny_shakespeare()
    chars = sorted(set(_text))
    _ids = np.array([chars.index(c) for c in _text], dtype=np.int32)
    _n = int(0.9 * len(_ids))
    train_ids, val_ids = jnp.asarray(_ids[:_n]), jnp.asarray(_ids[_n:])
    vocab = len(chars)

    _x = jnp.ones((1, 4, 2, 8))
    _checks = {
        "rms_norm": lj.attempt(ex.rms_norm, _x, jnp.ones(8)),
        "rope_rotate_half": lj.attempt(ex.rope_rotate_half, _x, jnp.arange(4)),
        "swiglu": lj.attempt(ex.swiglu, {"w1": jnp.ones((8, 4)), "w3": jnp.ones((8, 4)), "w2": jnp.ones((4, 8))},
                             _x),
        "repeat_kv": lj.attempt(ex.repeat_kv, _x, 2),
        "gqa_attention": lj.attempt(ex.gqa_attention, {"wq": jnp.ones((8, 8)), "wk": jnp.ones((8, 4)),
                                                       "wv": jnp.ones((8, 4)), "wo": jnp.ones((8, 8))},
                                    jnp.ones((1, 4, 8)), jnp.arange(4),
                                    {"n_heads": 2, "n_kv_heads": 1, "head_dim": 4, "rope_theta": 1e4}),
        "moe_route": lj.attempt(ex.moe_route, jnp.full((3, 4), 0.5), jnp.zeros(4), 2),
        "update_bias": lj.attempt(ex.update_bias, jnp.zeros(4), jnp.full(4, 0.25), 0.01),
        "mla_cache_floats": lj.attempt(ex.mla_cache_floats, {"kv_rank": 1, "d_rope": 1}),
        "mha_cache_floats": lj.attempt(ex.mha_cache_floats, 1, 1, 1),
    }
    parts_ready = all(a.ok for a in _checks.values())
    _missing = [k for k, a in _checks.items() if not a.ok]

    llama_params = M.init_llama(jax.random.key(0), vocab, M.LLAMA)
    ds_params = M.init_deepseek(jax.random.key(0), vocab, M.DEEPSEEK)
    if parts_ready:
        _L, _D = M.LLAMA, M.DEEPSEEK
        _kv_l = ex.mha_cache_floats(_L["n_kv_heads"], _L["head_dim"], _L["head_dim"])
        _kv_d = ex.mla_cache_floats(_D)
        _kv_mha = ex.mha_cache_floats(_D["n_heads"], _D["d_nope"] + _D["d_rope"], _D["d_v"])
        _view = mo.md(f"""
| tiny model | total params | active per token | KV cache, numbers per token per layer |
|---|---|---|---|
| Llama-style (4 query heads, 1 KV head) | {M.count_params(llama_params):,} | {M.active_params(llama_params, _L):,} | {_kv_l} (MHA would be {ex.mha_cache_floats(4, 32, 32)}) |
| DeepSeek-style (MLA, 8 routed experts, top-2 + 1 shared) | {M.count_params(ds_params):,} | {M.active_params(ds_params, _D):,} | {_kv_d} (MHA would be {_kv_mha}) |

The DeepSeek model has nearly **twice** the parameters but touches about the same number per token as the
Llama one — that's the MoE trade: more capacity at the same compute (ignoring the dense-compute shortcut).
Vocabulary: {vocab} characters; {len(train_ids):,} training tokens.
""")
    else:
        _view = mo.callout(mo.md("**Finish the exercises to assemble the models.** Still to do: "
                                 + ", ".join(f"`{m}`" for m in _missing)), kind="info")
    _view
    return chars, ds_params, llama_params, parts_ready, train_ids, val_ids, vocab


@app.cell
def _(M, ds_params, ex, jax, jnp, llama_params, optax, parts_ready, time, train_ids, val_ids):
    STEPS = 600

    def _optimizer():
        _sched = optax.warmup_cosine_decay_schedule(0.0, 3e-3, 60, STEPS, 3e-4)
        return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(_sched, weight_decay=0.1))

    def train(params, forward, biases, gamma):
        _opt = _optimizer()
        _train_chunk, _eval = M.make_trainer(forward, _opt, gamma=gamma, update_bias=ex.update_bias)
        _state, _key, _t0 = _opt.init(params), jax.random.key(1), time.perf_counter()
        _losses, _loads = [], []
        for _ in range(STEPS // 100):
            _key, _k = jax.random.split(_key)
            params, _state, biases, _l, _ld = _train_chunk(params, _state, biases, train_ids, _k, 100)
            _losses.append(_l)
            _loads.append(_ld)
        _val = float(_eval(params, biases, val_ids, jax.random.key(9)))
        return {"params": params, "biases": biases, "losses": jnp.concatenate(_losses), "val": _val,
                "loads": jnp.concatenate(_loads), "seconds": time.perf_counter() - _t0}

    n_moe = M.DEEPSEEK["n_layers"] - M.DEEPSEEK["n_dense"]
    ds_forward = lambda p, x, b: M.deepseek_forward(ex, p, x, M.DEEPSEEK, b)  # noqa: E731
    llama_forward = lambda p, x, b: M.llama_forward(ex, p, x, M.LLAMA)  # noqa: E731
    race = None
    if parts_ready:
        race = {"llama": train(llama_params, llama_forward, jnp.zeros(()), 0.0),
                "deepseek": train(ds_params, ds_forward, jnp.zeros((n_moe, M.DEEPSEEK["n_routed"])), 1e-2)}
    return ds_forward, llama_forward, n_moe, race, train


@app.cell
def _(mo, np, plt, race, vocab):
    _out = mo.md("*(The race runs once your parts are done.)*")
    if race is not None:
        _fig, _ax = plt.subplots(figsize=(6.4, 3.2))
        for _name, _r in race.items():
            _ax.plot(_r["losses"], lw=0.8, label=f"{_name}: val {_r['val']:.3f}, {_r['seconds']:.0f} s")
        _ax.set(xlabel="step", ylabel="train loss (nats/char)", title="Tiny Llama vs tiny DeepSeek",
                ylim=(1.2, 3.6))
        _ax.legend()
        _fig.tight_layout()
        _out = mo.vstack([_fig, mo.md(
            f"Both learn: validation loss {race['llama']['val']:.3f} (Llama-style) and "
            f"{race['deepseek']['val']:.3f} (DeepSeek-style) nats per character, down from {np.log(vocab):.2f} "
            f"(a uniform guess over {vocab} characters). At this size and step count they're a tie — the DeepSeek design "
            "pays off when the cache and per-token compute dominate, at billions of parameters and long "
            "contexts. The DeepSeek run is slower here because of the dense expert compute and an extra "
            "compile.")])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Does the bias actually balance the experts?

    The DeepSeek run above used your `update_bias` with γ = 0.01. Same model, same data, same seed —
    trained again with γ = 0 (no balancing). Each bar is one expert's share of routed slots over the last
    100 steps, for each of the 3 MoE layers; perfectly even is 1/8 = 12.5%.
    """)
    return


@app.cell
def _(M, ds_forward, ds_params, jnp, n_moe, np, plt, race, train):
    _out = None
    if race is not None:
        _nobias = train(ds_params, ds_forward, jnp.zeros((n_moe, M.DEEPSEEK["n_routed"])), 0.0)
        _fig, _axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
        _E = M.DEEPSEEK["n_routed"]
        for _ax, (_title, _r) in zip(_axes, (("with bias (γ = 0.01)", race["deepseek"]),
                                             ("no balancing (γ = 0)", _nobias)), strict=True):
            _load = np.asarray(_r["loads"][-100:].mean(0)) * 100                      # (layers, E)
            for _l in range(_load.shape[0]):
                _ax.bar(np.arange(_E) + (_l - 1) * 0.27, _load[_l], width=0.27, label=f"MoE layer {_l + 1}")
            _ax.axhline(100 / _E, color="gray", ls="--", lw=1)
            _ax.set(title=f"{_title}\nval {_r['val']:.3f}, load {_load.min():.1f}–{_load.max():.1f}%",
                    xlabel="expert")
        _axes[0].set_ylabel("share of routed slots (%)")
        _axes[0].legend(fontsize=7)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    With the bias, every expert in every layer stays within about half a percent of 12.5%. Without it,
    some experts take two or three times their share while others sit almost idle — the same effect you
    measured at full scale in deepseek-v3-jax (5.7–6.9% vs 0.09–18.6% over 16 experts). Validation loss
    barely differs here (the unbalanced run can even come out a hair ahead); what balance buys is
    **efficiency**: with sparse dispatch across GPUs, an
    overloaded expert is a straggler every other GPU waits for, and idle experts are wasted memory.

    ### Samples

    Each model continues the same prompt, 200 characters at temperature 0.8. (No KV cache here — each step
    re-runs a fixed 128-character window, which keeps one compiled shape. Module 18 builds proper cached
    generation.)
    """)
    return


@app.cell
def _(chars, ds_forward, jax, jnp, llama_forward, mo, race, val_ids):
    _out = mo.md("*(Samples appear after the race.)*")
    if race is not None:
        _W = 128

        def _sample(forward, params, biases, key, n=200):
            _step = jax.jit(lambda p, b, ctx, k: jax.random.categorical(k, forward(p, ctx[None], b)[0][0, -1] / 0.8))
            _ctx = val_ids[1000:1000 + _W]
            _gen = []
            for _k in jax.random.split(key, n):
                _nxt = _step(params, biases, _ctx, _k)
                _gen.append(int(_nxt))
                _ctx = jnp.concatenate([_ctx[1:], _nxt[None]])
            return "".join(chars[i] for i in _gen)

        _prompt = "".join(chars[int(i)] for i in val_ids[1000:1000 + _W])
        _lines = [f"**Prompt (last 128 characters):**\n\n```\n{_prompt}\n```"]
        for _name, _fwd in (("Llama-style", llama_forward), ("DeepSeek-style", ds_forward)):
            _r = race["llama" if _name.startswith("Llama") else "deepseek"]
            _txt = _sample(_fwd, _r["params"], _r["biases"], jax.random.key(3))
            _lines.append(f"**{_name}:**\n\n```\n{_txt}\n```")
        _out = mo.md("\n\n".join(_lines))
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Stretch: how much RoPE does a model need?

    Section 2 listed four ways to place position: RoPE in every layer, **partial** RoPE, a **hybrid**
    with no RoPE in some layers, and **NoPE**. Here are all four on the tiny Llama, with the same
    initialization, data, seed and 600 steps. Each trains on 128-character windows and is then
    evaluated on **512**-character windows. The loss at each position shows how the model copes in the
    range it trained on (0–127) and past it, at positions it never saw.

    * RoPE every layer: the race's Llama, reused.
    * Partial: 8 of each head's 32 features rotated (25%, GPT-NeoX's choice).
    * Hybrid 3:1: full RoPE in layers 1–3 and none in layer 4 (SmolLM3's and Llama 4's pattern).
    * NoPE: no rotation anywhere.

    `models.py` has the variant attention (`pos_attention`) and `loss_by_position`. It needs your
    **`rope_partial`** exercise.
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lj, llama_params, mo, race, train, val_ids):
    _a = lj.attempt(ex.rope_partial, jnp.ones((1, 4, 2, 8)), jnp.arange(4), 4)
    _items = [mo.md("### Your turn (stretch): `rope_partial`"), _a.view(mo, show_value=False)]
    pos_runs = None
    if _a.ok and race is not None:
        _D = M.LLAMA["head_dim"]
        _cfgs = {"RoPE every layer": None, "partial (8 of 32)": (8,) * 4,
                 "hybrid 3:1": (_D, _D, _D, 0), "NoPE": (0,) * 4}
        pos_runs = {}
        for _name, _rd in _cfgs.items():
            if _rd is None:
                _params, _secs = race["llama"]["params"], race["llama"]["seconds"]
                _fwd = lambda p, x: M.llama_forward(ex, p, x, M.LLAMA)  # noqa: E731
            else:
                _fwd = lambda p, x, rd=_rd: M.llama_pos_forward(ex, p, x, M.LLAMA, rd)  # noqa: E731
                _r = train(llama_params, lambda p, x, b, f=_fwd: f(p, x), jnp.zeros(()), 0.0)
                _params, _secs = _r["params"], _r["seconds"]
            _lp = jax.jit(lambda p, k, f=_fwd: M.loss_by_position(f, p, val_ids, k, 512))(_params, jax.random.key(9))
            pos_runs[_name] = {"loss": _lp, "seconds": _secs}
    elif _a.ok:
        _items.append(mo.md("*(The comparison runs once the race above has run.)*"))
    mo.vstack(_items)
    return (pos_runs,)


@app.cell
def _(mo, np, plt, pos_runs):
    _out = None
    if pos_runs is not None:
        _fig, _ax = plt.subplots(figsize=(6.4, 3.2))
        _k = np.ones(16) / 16
        for _name, _r in pos_runs.items():
            _l = np.asarray(_r["loss"])
            _ax.plot(np.arange(len(_l) - 15) + 8, np.convolve(_l, _k, "valid"), lw=1, label=_name)
        _ax.axvline(128, color="gray", ls="--", lw=1)
        _ax.text(132, 1.25, "training length", fontsize=8, color="gray")
        _ax.set(xlabel="position in the 512-character window", ylabel="loss (nats/char)",
                title="Loss by position, 16-position average", ylim=(1.2, None))
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _rows = "\n".join(
            f"| {_n} | {float(_r['loss'][:128].mean()):.3f} | {float(_r['loss'][128:256].mean()):.3f} | "
            f"{float(_r['loss'][256:].mean()):.3f} |" for _n, _r in pos_runs.items())
        _out = mo.vstack([_fig, mo.md(f"""
| positions | 0–127 (trained) | 128–255 (2×) | 256–511 (4×) |
|---|---|---|---|
{_rows}
""")])
    _out
    return


@app.cell
def _(mo, pos_runs):
    _out = None
    if pos_runs is not None:
        _m = {n: (float(r["loss"][:128].mean()), float(r["loss"][128:256].mean()), float(r["loss"][256:].mean()))
              for n, r in pos_runs.items()}
        _full, _part, _hyb, _nope = (_m[n] for n in ("RoPE every layer", "partial (8 of 32)", "hybrid 3:1", "NoPE"))
        _out = mo.md(f"""
    **Inside the training length**, full, partial and hybrid RoPE tie ({_full[0]:.2f}, {_part[0]:.2f},
    {_hyb[0]:.2f}). Rotating only a quarter of each head, or dropping RoPE from one layer in four, costs
    nothing. **NoPE** lags far behind ({_nope[0]:.2f}). A 4-layer model given 600 steps hasn't learned to
    pull distances out of the causal mask, and for characters, knowing the previous few positions exactly
    is most of the job.

    **Past the training length**, everything gets worse, and the schemes fail differently:

    * Full RoPE breaks right away ({_full[1]:.2f} at 2×). Its fast pairs have turned full circle many
      times in training, but its slow pairs now reach angles they never reached in 128 positions.
    * Partial RoPE holds up best at 2× ({_part[1]:.2f}). Its 4 pairs run at wavelengths of about 6, 63, 630
      and 6,300 characters, and the 24 unrotated features don't care where they are. By 4× it falls apart
      too ({_part[2]:.2f}).
    * The hybrid edges out full RoPE at 4× ({_hyb[2]:.2f} vs {_full[2]:.2f}). Its RoPE-free layer gives a
      small amount of slack.
    * NoPE degrades least ({_nope[2]:.2f} at 4×) only because it never used precise positions to begin with.

    Two seeds gave the same ordering when this lesson was written. None of these tricks makes a model work
    at 4× its training length on its own. Real long-context models also *train* on long documents and raise
    the RoPE base (section 2). The small-model penalty for NoPE is one reason production models mix it with
    RoPE rather than going without.
    """)
    _out
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "It caches a small compressed latent per token instead of per-head keys and values": "a",
            "It uses fewer query heads than key/value heads": "b",
            "It drops the value vectors entirely": "c",
        },
        label="**Check yourself:** how does MLA shrink the KV cache?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right — `kv_rank + d_rope` numbers per token (144 in your model) instead of a key "
                              "and value for every head (480). Keys and values are re-derived from the latent."),
                        kind="success"),
        "b": mo.callout(mo.md("That's backwards, and it describes GQA's idea (fewer *KV* heads than query heads), "
                              "not MLA's. MLA compresses all heads' K/V into one latent."), kind="warn"),
        "c": mo.callout(mo.md("No — values are still used, they're just decompressed from the cached latent "
                              "on demand."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.lineage as _lin

    _m = _lin.by_id()
    _llama = ("mqa", "pre_norm", "rmsnorm", "rope", "swiglu", "residual", "next_token")
    _ds = ("mla", "rope", "partial_rope", "moe_topk", "shared_experts", "aux_free", "pre_norm", "rmsnorm", "swiglu",
           "residual", "next_token")
    _table = _lin.compare_md(
        [("**your Llama**", _llama), ("Llama 3", _m["llama3"].uses), ("**your DeepSeek**", _ds),
         ("DeepSeek-V3", _m["deepseek_v3"].uses), ("SmolLM3", _m["smollm3"].uses)],
        ["position", "normalization", "sequence mixing", "feed-forward", "objective"])
    mo.md(f"""
## You are here

Your two tiny models next to the real ones in the course's family tree (Module 00b, Section 12):

{_table}

Two differences are deliberate. Your Llama has 4 query heads and **one** KV head, which is
grouped-query attention with a single group, also called multi-query attention. Llama 3 uses 8 KV
heads. Your DeepSeek skips multi-token prediction (section 9). SmolLM3 is the "hybrid" from section
2's Beyond RoPE table: Llama's blocks, with no RoPE in every 4th layer.
    """)
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(17))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Su et al., [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021), and
      Christopher Fleetwood's [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding)
    * Sebastian Raschka, [Understanding Positional Embeddings in Modern LLMs](https://magazine.sebastianraschka.com/p/understanding-positional-embeddings)
      (2026) — learned → sinusoidal → RoPE → partial, NoPE and hybrids, with a survey of 2026 open-weight models
    * Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (2023);
      Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (2020)
    * Meta, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024) — Section 3 is the architecture
    * DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (2024) — MLA, DeepSeekMoE,
      aux-loss-free balancing, MTP
    * Vaibhaav Tiwari, [deepseek-v3-jax](https://github.com/Vaibhaav-Tiwari/deepseek-v3-jax) — the 190-line model you replicated;
      reread it now, every line should be familiar

    **Next:** Module 18 — inference: KV-cache decoding for real, sampling strategies, speculative decoding
    (where MTP comes back) and quantization.
    """)
    return


if __name__ == "__main__":
    app.run()
