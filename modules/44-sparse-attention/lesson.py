import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="44 · Efficient and sparse attention")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import json

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import rosetta

    ex = lj.load_exercises(mo.notebook_dir())
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")  # re-runs skip XLA compilation
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    _spec = importlib.util.spec_from_file_location("m44_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    RUNS = M.RUNS

    def read(name):
        try:
            return json.loads((RUNS / name).read_text())
        except OSError:
            return None
    return M, RUNS, ex, jax, jnp, lj, mo, np, plt, read, rosetta


@app.cell
def _(mo):
    mo.md(r"""
    # 44 · Efficient and sparse attention: reading only part of the past (2019–2026)

    **Take first:** Module 15 (attention), Module 18 (the KV cache and SmolLM2), Module 23 (FlashAttention)
    and Module 41, which replaced attention with a fixed-size memory. This module keeps attention and makes
    each query read **fewer keys**.

    **Goal:** full attention compares every token with every earlier token: $T^2/2$ scores to fill a
    prompt, and a KV cache that grows with the context and must be read in full for every new token.
    Production models cut this in three ways: a **fixed window** of recent tokens (Mistral, Gemma), a few
    **sink** tokens that the model can always see (StreamingLLM), and **learned selection**, where the model
    decides which blocks or tokens to read (MoBA, NSA, DeepSeek Sparse Attention). You'll build each one
    and race them on the same model and data, at 2,048 tokens of context.

    **You will:** size a KV cache with local and global layers · write sliding-window and sink masks ·
    see SmolLM2 fall apart without its sink token and recover with four · write a sliding window that
    really costs $O(T \cdot w)$ and time it · write MoBA's block selection and DSA's indexer · compare six
    attention patterns on quality and on how much of the past they read.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    OpenAI's **Sparse Transformer** (Child et al., 2019) read only strided and local positions to model
    long images and audio. **Longformer** and **BigBird** (2020) brought sliding windows plus a few global
    tokens to long documents. Then FlashAttention (2022) made full attention cheap enough that sparsity
    went quiet for a while, until context lengths grew past 100K. **Mistral 7B** (2023) shipped a 4,096-token
    sliding window; **Gemma 2** (2024) alternated local and global layers 1 : 1 and **Gemma 3** (2025) went
    to 5 : 1. **StreamingLLM** (Xiao et al., 2023) explained why plain windows break pretrained models: the
    first token is a "sink" that soaks up unneeded attention. In 2025 selection became learned: Moonshot's
    **MoBA**, DeepSeek's **Native Sparse Attention** (an ACL 2025 best paper) and DeepSeek V3.2's
    **DeepSeek Sparse Attention**, which picks 2,048 tokens per query with a small "lightning indexer".

    *Timeline: Module 00b.*
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Where attention's cost goes

    Two costs grow with context length $T$:

    * **Prefill** (reading the prompt): every query scores every earlier key, $\approx T^2/2$ dot products
      per head per layer. Double the prompt, four times the work.
    * **Decode** (writing): each new token reads the whole KV cache. The cache holds a key and a value per
      token, per layer, per KV head, so its size grows linearly with $T$, and at long contexts reading it,
      not the matrix multiplies, sets the speed (Module 18).

    A sliding window of $w$ tokens fixes both for the layers that use it: $T \cdot w$ scores, and a cache
    that stops growing at $w$. Gemma 3 keeps one **global** layer for every five **local** ones, so long-range
    information still has a path.

    **Your turn:** `kv_cache_bytes(context, n_layers, n_global, window, n_kv_heads, d_head, bytes_per)`.
    """)
    return


@app.cell
def _(ex, lj, mo):
    _shape = dict(n_layers=32, n_kv_heads=8, d_head=128)                  # Llama 3 8B's attention shape
    _rows = []
    _ok = True
    for _ctx in (8_192, 32_768, 131_072):
        _full = lj.attempt(ex.kv_cache_bytes, _ctx, n_global=32, window=1024, **_shape)
        _mix = lj.attempt(ex.kv_cache_bytes, _ctx, n_global=6, window=1024, **_shape)   # ~5 : 1 local : global
        if not (_full.ok and _mix.ok):
            _out = (_full if not _full.ok else _mix).view(mo)
            _ok = False
            break
        _rows.append(f"| {_ctx:,} | {_full.value / 2**30:.2f} GB | {_mix.value / 2**30:.2f} GB | "
                     f"{_full.value / _mix.value:.1f}× |")
    if _ok:
        _out = mo.md(f"""
    KV cache for **one** sequence, bf16, with Llama 3 8B's attention shape (32 layers, 8 KV heads of 128):

    | context | all 32 layers global | 26 local (window 1,024) + 6 global | saved |
    |---|---|---|---|
    {chr(10).join(_rows)}

    At 128K tokens the all-global cache alone would fill your whole 16 GB GPU, before any weights; the
    5 : 1 mix needs a fifth of that.
    The saving approaches 32 / 6 ≈ 5.3× as the context grows, because only the global layers keep growing.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Sliding windows and sinks, as masks

    Every pattern in this module is a boolean mask over (query, key) pairs: True means "may read". A
    **sliding window** of $w$ lets query $i$ read keys $i-w+1 \dots i$. **Sinks** add the first few tokens
    of the sequence to every query's window, whatever its position.

    **Your turn:** `window_mask(T, window, n_sink)`. It must also work when `window` and `n_sink` are traced
    numbers inside [`jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit): the race's model passes each layer its own window through [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan).
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lj, mo, np, plt):
    _a = lj.attempt(ex.window_mask, 24, 6)
    _b = lj.attempt(ex.window_mask, 24, 6, 2)
    _k = jax.random.split(jax.random.key(0), 3)
    _q, _kk = jax.random.normal(_k[0], (1, 1, 24, 8)), jax.random.normal(_k[1], (1, 1, 24, 8))
    _c = lj.attempt(ex.moba_mask, _q, _kk, 4, 2)
    _s = jax.random.normal(_k[2], (1, 24, 24))
    _d = lj.attempt(ex.topk_causal_mask, _s, 6)
    if not (_a.ok and _b.ok):
        _out = (_a if not _a.ok else _b).view(mo)
    else:
        _panels = [("causal (full)", np.tril(np.ones((24, 24), bool))), ("window 6", np.asarray(_a.value)),
                   ("window 6 + 2 sinks", np.asarray(_b.value))]
        if _c.ok:
            _panels.append(("MoBA: own block + 2 chosen\n(blocks of 4)", np.asarray(_c.value)[0, 0]))
        if _d.ok:
            _panels.append(("top-6 by an indexer", np.asarray(_d.value)[0]))
        _fig, _axs = plt.subplots(1, len(_panels), figsize=(2.3 * len(_panels), 2.6))
        for _ax, (_t, _m) in zip(_axs, _panels):
            _ax.imshow(_m, cmap="Blues", vmin=0, vmax=1.3)
            _ax.set_title(_t, fontsize=8)
            _ax.set_xticks([])
            _ax.set_yticks([])
        _axs[0].set_ylabel("query →", fontsize=8)
        _axs[0].set_xlabel("key →", fontsize=8)
        _fig.tight_layout()
        _n = [int(m.sum()) for _t, m in _panels]
        _out = mo.vstack([_fig, mo.md(
            f"24 tokens. Keys read in total: full {_n[0]}, window {_n[1]}, window + sinks {_n[2]}"
            + (f", MoBA {_n[3]}" if _c.ok else "") + (f", top-6 {_n[-1]}" if _d.ok else "")
            + ". The last two panels come from sections 5 and 6: there, *which* keys are read depends on the "
              "data, so the pattern changes from row to row.")])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Attention sinks: why a plain window breaks a pretrained model

    Take SmolLM2-135M, trained with full attention, and run it on 2,048-token detective passages with
    different masks, no training at all. A window of 256 tokens looks like plenty for predicting the next
    word. StreamingLLM found that it is not: once the first token leaves the window, the loss jumps.

    The reason is in the softmax. Every query's attention weights must sum to 1, even when nothing in the
    context is relevant. Models learn to dump that leftover weight on the first token, which every query
    can see; it becomes a **sink**. Remove it and the leftover weight lands on real tokens instead, and
    every layer's output drifts. Keeping just the first few tokens fixes it.
    """)
    return


@app.cell
def _(mo, np, plt, read):
    _r = read("sinks.json")
    if _r is None:
        _out = mo.md("*Run `train.py --sinks` (about 3 minutes) to see SmolLM2's sinks here.*")
    else:
        _pl = _r["pos_loss"]
        _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 2]})
        _first = _r["attention_to_first"]
        _a1.bar(range(len(_first)), _first)
        _a1.set_xlabel("layer")
        _a1.set_ylabel("attention on token 0")
        _a1.set_title("share of attention on the first token", fontsize=9)
        for _name, _v in _pl.items():
            _v = np.asarray(_v)
            _sm = np.convolve(_v, np.ones(64) / 64, "valid")
            _a2.plot(np.arange(len(_sm)) + 32, _sm, label=_name, lw=1.2)
        _a2.set_xlabel("position in the passage")
        _a2.set_ylabel("loss (64-token average)")
        _a2.set_title("SmolLM2-135M, no retraining", fontsize=9)
        _a2.legend(fontsize=7)
        _fig.tight_layout()
        _mean = {k: float(np.mean(v)) for k, v in _pl.items()}
        _late = {k: float(np.mean(v[512:])) for k, v in _pl.items()}
        _rows = "\n".join(f"| {k} | {_mean[k]:.3f} | {_late[k]:.3f} |" for k in _pl)
        _keys = list(_pl)
        _out = mo.vstack([_fig, mo.md(f"""
    | mask | loss, all positions | loss, positions 512+ |
    |---|---|---|
    {_rows}

    On average over layers, queries put **{np.mean(_first):.0%}** of their attention on token 0 (the left
    plot; the token itself is just the passage's first word). The plain window costs
    **{_late[_keys[1]] - _late[_keys[0]]:.2f} nats** per token once the first token is out of reach; adding back
    **four** tokens recovers all but {_late[_keys[2]] - _late[_keys[0]]:.2f}. The control row matters: widening
    the window by the same four tokens instead does nothing, so it is *which* tokens, not how many.
    """)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · A window that really is cheaper

    A mask only hides scores; computing all $T^2$ of them and then throwing most away saves nothing.
    To *gain* speed, compute only the scores you keep. For a window of $w$: cut the sequence into blocks
    of $w$ queries. Every key a query in block $n$ can read sits in block $n$ or block $n-1$, so each
    block attends to $2w$ keys and the total is $T \cdot 2w$ scores instead of $T^2$. FlashAttention's
    local-window option does the same inside its tiles.

    **Your turn:** `banded_attention(q, k, v, window)`, checked against dense attention with your mask.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np):
    _k = jax.random.split(jax.random.key(1), 3)
    _q, _kk, _v = (jax.random.normal(k, (1, 2, 64, 16)) for k in _k)
    _a = lj.attempt(ex.banded_attention, _q, _kk, _v, 16)
    _m = lj.attempt(ex.window_mask, 64, 16)
    banded_ready = _a.ok and _m.ok
    if not banded_ready:
        _out = (_a if not _a.ok else _m).view(mo)
    else:
        _s = jnp.einsum("bhtd,bhsd->bhts", _q, _kk) / 4.0
        _ref = jnp.einsum("bhts,bhsd->bhtd", jax.nn.softmax(jnp.where(_m.value, _s, -jnp.inf), -1), _v)
        _err = float(jnp.max(jnp.abs(_a.value - _ref)))
        _out = mo.md(f"Your banded attention vs dense attention under your window mask (64 tokens, window 16): "
                     f"max difference **{_err:.1e}** {'✅' if _err < 1e-4 else '❌'}.")
    _out
    return (banded_ready,)


@app.cell
def _(mo, plt, read):
    _b = read("bench.json")
    if _b is None:
        _out = mo.md("*Run `train.py --bench` (about a minute) to time it.*")
    else:
        _T = sorted(int(t) for t in _b)
        _fig, _ax = plt.subplots(figsize=(6.5, 3))
        _ax.plot(_T, [_b[str(t)]["banded_ms"] for t in _T], "o-", label="banded window (w = 512)")
        _dT = [t for t in _T if "dense_ms" in _b[str(t)]]
        _ax.plot(_dT, [_b[str(t)]["dense_ms"] for t in _dT], "s-", label="dense causal")
        _ax.set_xscale("log", base=2)
        _ax.set_yscale("log")
        _ax.set_xlabel("sequence length")
        _ax.set_ylabel("ms, one layer, batch 1")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _big = max(_dT)
        _out = mo.vstack([_fig, mo.md(f"""
    One attention layer's prefill on your GPU (6 heads of 64, bf16). Dense time grows about 4× per doubling
    once it is past launch overheads; the banded window grows 2×. At {_big:,} tokens the window is
    **{_b[str(_big)]["dense_ms"] / _b[str(_big)]["banded_ms"]:.1f}×** faster, and dense attention stops at
    {_big:,} because its {_big:,}² scores no longer fit in memory, while the band runs to {max(_T):,}.
    (Both are plain `jnp` code; a fused kernel such as FlashAttention is faster for both.)
    """)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Learned selection: let the query choose

    A window bets that the useful past is the recent past. In a detective story the useful token might be
    a name introduced 1,500 tokens ago. **Learned selection** keeps a small budget of keys per query but
    lets the query choose *which*:

    * **MoBA** (Moonshot, 2025): split the keys into blocks; score each block by the query's dot product
      with the block's **mean key**; read your own block plus the `top` best earlier ones. No new
      parameters; the attention's own keys do the routing, like a mixture of experts over blocks (Module 17).
    * **NSA** (DeepSeek, 2025): three branches combined by a learned gate: compressed block summaries, a
      few selected full blocks, and a sliding window. Designed with the GPU kernel in mind (whole blocks).
    * **DSA** (DeepSeek V3.2, 2025): a tiny separate attention, the **lightning indexer** (few heads, ReLU
      instead of softmax, FP8 in production), scores every past token; the main attention reads only the
      top $k$ (2,048 in V3.2). The indexer is trained to agree with the main attention's own weights.

    **Your turn:** `moba_mask`, `indexer_scores` and `topk_causal_mask`.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np):
    _T, _d = 32, 4
    _k = jnp.zeros((1, 1, _T, _d)).at[0, 0, 8:16, 0].set(5.0)
    _q = jnp.zeros((1, 1, _T, _d)).at[0, 0, :, 0].set(1.0)
    _a = lj.attempt(ex.moba_mask, _q, _k, 8, 1)
    _ks = jax.random.split(jax.random.key(2), 3)
    _qi, _ki, _w = jax.random.normal(_ks[0], (1, 6, 2, 4)), jax.random.normal(_ks[1], (1, 6, 4)), jnp.ones((1, 6, 2))
    _b = lj.attempt(ex.indexer_scores, _qi, _ki, _w)
    _c = lj.attempt(ex.topk_causal_mask, jnp.arange(36.0).reshape(1, 6, 6), 2)
    if not (_a.ok and _b.ok and _c.ok):
        _out = next(x for x in (_a, _b, _c) if not x.ok).view(mo)
    else:
        _picked = [b for b in range(3) if bool(_a.value[0, 0, 31, b * 8])]
        _ref = jnp.einsum("bth,bths->bts", _w, jax.nn.relu(jnp.einsum("bthd,bsd->bths", _qi, _ki)))
        _out = mo.md(f"""
    * MoBA: keys in block 1 all point along the query; the last query picks earlier block(s) **{_picked}**
      {"✅" if _picked == [1] else "❌ expected [1]"}.
    * Indexer: max difference from the formula **{float(jnp.max(jnp.abs(_b.value - _ref))):.1e}**.
    * Top-2 of increasing scores picks the two latest allowed keys: row 5 →
      `{np.flatnonzero(np.asarray(_c.value[0, 5])).tolist()}`.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### How DSA's indexer learns

    Top-$k$ selection has no gradient, so the language-model loss can't teach the indexer which tokens
    were worth reading. DeepSeek trains it with its own loss instead: sum the main attention's weights over
    heads, renormalize, and make the indexer's softmax match it (a KL divergence), with the main model's
    gradients stopped. In V3.2 this starts with a short **dense warm-up** (full attention, indexer learning
    to imitate it) before switching to sparse attention. `models.py` does the same: dense for the first 500
    steps, then the top 128 keys.

    One trap we fell into: the indexer needs its **own optimizer**, not just its own loss. Our first run
    clipped all gradients together to a global norm of 1. The indexer's KL gradients were large, so they set
    the norm and every update to the language model got scaled down; DSA trailed full attention even during
    the dense warm-up, when the two models are identical. `make_optimizer` now clips and steps the
    indexer's weights separately ([`optax.multi_transform`](https://optax.readthedocs.io/en/latest/api/combining_optimizers.html#optax.multi_transform), as in Module 20).

    $$\mathcal L_I = \sum_t \mathrm{KL}\Big(\tfrac{1}{Z_t}\textstyle\sum_h a^{(h)}_{t,\cdot} \ \Big\Vert\ \mathrm{softmax}(I_{t,\cdot})\Big).$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · The race

    One model (Module 41's attention model: 8 layers, width 384, 18.5M parameters), six masks, the same
    49M tokens of the detective corpus in 2,048-token windows (`train.py --variant …`, 35–65 minutes each):

    | variant | each query reads |
    |---|---|
    | full | every earlier token (on average 1,024) |
    | swa | the last 128 tokens, in every layer |
    | swa-sink | the last 124 + the first 4 |
    | local-global | the last 128 in six layers; everything in two (LLLGLLLG) |
    | moba | its own 64-token block + the 2 best earlier blocks (≤ 192) |
    | dsa | the 128 tokens its indexer scores highest |

    All of them use masked *dense* attention, so this measures quality, not speed (section 4 did speed).
    "Keys read" is the share of the causal $T^2/2$ pairs each pattern allows.
    """)
    return


@app.cell
def _(M, RUNS, mo, np, plt):
    _names = [n for n in M.VARIANTS if (RUNS / n / "pos_loss.json").exists()]
    if not _names:
        _out = mo.md("*No race runs yet: run `train.py --variant full` (and the others).*")
    else:
        import json as _json

        _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 3.3))
        _rows = []
        _pos = {}
        for _n in _names:
            _m = M.read_metrics(RUNS / _n / "metrics.jsonl")
            _a1.plot([r["step"] for r in _m], [r["val"] for r in _m], label=_n, lw=1.2)
            _p = np.asarray(_json.loads((RUNS / _n / "pos_loss.json").read_text()))
            _pos[_n] = _p
            _sm = np.convolve(_p, np.ones(64) / 64, "valid")
            _a2.plot(np.arange(len(_sm)) + 32, _sm, label=_n, lw=1.0)
            _rows.append((_n, _m[-1]["val"], _m[-1]["read"], _p[:128].mean(), _p[128:512].mean(), _p[512:].mean(),
                          _m[-1]["tok_per_s"]))
        _a1.set_ylim(min(r[1] for r in _rows) - 0.05, min(r[1] for r in _rows) + 0.8)
        _a1.set_xlabel("step")
        _a1.set_ylabel("validation loss")
        _a1.legend(fontsize=7)
        _a2.set_xlabel("position in the 2,048-token window")
        _a2.set_ylabel("loss (64-token average)")
        _a2.set_ylim(min(_sm.min() for _sm in [np.convolve(p, np.ones(64) / 64, "valid") for p in _pos.values()]) - 0.05,
                     None)
        _fig.tight_layout()
        _tab = "\n".join(f"| {n} | **{v:.3f}** | {r:.1%} | {a:.3f} | {b:.3f} | {c:.3f} |"
                         for n, v, r, a, b, c, _t in sorted(_rows, key=lambda x: x[1]))
        _out = mo.vstack([_fig, mo.md(f"""
    | variant | val loss | keys read | loss, positions 0–127 | 128–511 | 512–2,047 |
    |---|---|---|---|---|---|
    {_tab}
    """)])
    _out
    return


@app.cell
def _(M, RUNS, mo, np):
    import json as _json2

    _have = {n: _json2.loads((RUNS / n / "pos_loss.json").read_text()) for n in M.VARIANTS
             if (RUNS / n / "pos_loss.json").exists()}
    if len(_have) < len(M.VARIANTS):
        _out = mo.md("*The reading of the race appears once all six runs are done.*")
    else:
        _v = {n: (M.read_metrics(RUNS / n / "metrics.jsonl")[-1]["val"], float(np.mean(p[512:])))
              for n, p in _have.items()}
        _full = _v["full"][0]
        _best = min((n for n in _v if n != "full"), key=lambda n: _v[n][0])
        _gap = {n: _v[n][0] - _full for n in _v}
        _beat = [n for n in _v if _gap[n] < 0]
        _txt = [f"* **Full attention: {_full:.3f}.** "
                + ("It wins, as expected; the question is how much each pattern gives up."
                   if not _beat else
                   f"It does not win: {', '.join(_beat)} beat{'s' if len(_beat) == 1 else ''} it. That is "
                   "common in short runs. A window builds in a bias toward recent tokens, which most "
                   "next-word prediction needs; full attention has to learn that from data, and 49M tokens "
                   "isn't enough to finish."
                   + (" Its last column (positions 512 and later) shows it isn't yet turning the extra past "
                      "into lower loss." if _v["full"][1] > min(x[1] for x in _v.values()) + 0.01 else ""))]
        _txt.append(f"* **Best sparse pattern: {_best}**, {_gap[_best]:+.3f} nats against full, reading "
                    f"{M.read_metrics(RUNS / _best / 'metrics.jsonl')[-1]['read']:.0%} of the keys.")
        _txt.append(f"* **Plain window vs window + sinks:** {_gap['swa']:+.3f} vs {_gap['swa-sink']:+.3f}. "
                    + ("Trained with sinks from the start, the window does better: the model can park unused "
                       "attention on them." if _gap["swa-sink"] < _gap["swa"] - 0.005 else
                       "Trained from scratch, the sinks make little difference: a model that never had them "
                       "learns to manage without, unlike pretrained SmolLM2 in section 3."))
        _lg = _v["local-global"][0] - _v["swa"][0]
        if _gap["swa"] <= 0:
            _txt.append(f"* **Two global layers** (local-global) change the window's score by {_lg:+.3f}: "
                        + ("the window's bias plus a little long range is the best mix here."
                           if _lg < -0.005 else "no clear gain at this context length."))
        else:
            _txt.append(f"* **Two global layers** (local-global, {_gap['local-global']:+.3f}) "
                        + ("recover most of what the window lost." if _gap["local-global"] < 0.5 * _gap["swa"]
                           else "recover only part of what the window lost."))
        _txt.append(f"* **Learned selection:** MoBA {_gap['moba']:+.3f}, DSA {_gap['dsa']:+.3f}, against the "
                    f"window's {_gap['swa']:+.3f} with a similar budget of keys."
                    + (" DSA comes last, and that is a fair result for how we used it, not for DSA. DeepSeek "
                       "added the indexer to an already trained dense model (V3.1-Terminus): it learned to "
                       "imitate mature attention, then sparse training continued. Ours had 500 steps of a "
                       "newborn model to imitate, and its 128 picks are free to skip the nearby tokens a "
                       "window keeps by construction. It also ran slower here (masked dense attention plus "
                       "a full T² indexer); the speed only shows up with a kernel that skips the unread keys."
                       if _gap["dsa"] > max(_gap["swa"], _gap["moba"]) else ""))
        _txt.append("* **Where the loss goes:** compare the last column of the table. Early positions have "
                    "little past to lose, so every variant is close there; differences show up late in the "
                    "window, where the useful past is further away.")
        _out = mo.md("\n".join(_txt) + """

    Caveat: 49M tokens of one corpus at 2,048 tokens is a small test. Most of what a detective story needs
    is local, which flatters windows; tasks that must find one fact 100K tokens back (Module 41's recall
    task, at scale) are where selection is supposed to pay.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · What shipped

    | model | attention | source |
    |---|---|---|
    | Mistral 7B (2023) | sliding window of 4,096 in every layer (rolling KV cache) | arXiv 2310.06825 |
    | Gemma 2 (2024) | local (window 4,096) and global layers alternating 1 : 1 | arXiv 2408.00118 |
    | Gemma 3 (2025) | 5 local (window 1,024) : 1 global | arXiv 2503.19786 |
    | Kimi (Moonshot, 2025) | MoBA: block selection by mean key | arXiv 2502.13189 |
    | NSA (DeepSeek, 2025) | compressed + selected + sliding branches, gated | arXiv 2502.11089 |
    | DeepSeek V3.2 (2025) | DSA: lightning indexer, top 2,048 tokens per query; DeepSeek cut API prices by half on launch | arXiv 2512.02556 |

    StreamingLLM's sinks have also been built into training: some models add a learned "sink" logit or
    token to every attention softmax so the leftover weight has somewhere harmless to go (Module 41's
    gated attention is another fix for the same problem).
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"The first token carries the most information about the passage": "a",
                 "Softmax weights must sum to 1, and the model learned to park unneeded weight on the first "
                 "token, which every query can see": "b",
                 "RoPE gives the first position a special encoding": "c"},
        label="**Check yourself:** why does dropping the first token break a pretrained model's window?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("The sink works with any first token, even a meaningless one; and widening the "
                              "window by four ordinary tokens didn't help."), kind="warn"),
        "b": mo.callout(mo.md("Right: it's a place to put attention, not a source of information."),
                        kind="success"),
        "c": mo.callout(mo.md("RoPE is relative: no position is special to it. StreamingLLM found the effect in "
                              "models with several kinds of position encoding."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(44))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Guangxuan Xiao, [StreamingLLM](https://github.com/mit-han-lab/streaming-llm) — the code and demos for
      attention sinks.
    * Evan Miller, [Attention Is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html) —
      the "softmax must sum to one" argument, and a proposed fix.

    **Next:** Module 45, diffusion language models: writing a whole passage at once.
    """)
    return


if __name__ == "__main__":
    app.run()
