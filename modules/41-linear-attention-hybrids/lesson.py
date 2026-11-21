import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="41 · Linear attention and hybrids")


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
    from learnjax import lineage, rosetta

    ex = lj.load_exercises(mo.notebook_dir())
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")  # re-runs skip XLA compilation
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    _spec = importlib.util.spec_from_file_location("m41_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    RUNS = mo.notebook_dir().parent.parent / "runs" / "41-linear-attention-hybrids"
    return M, RUNS, ex, jax, jnp, json, lineage, lj, mo, np, plt, rosetta


@app.cell
def _(mo):
    mo.md(r"""
    # 41 · Linear attention and hybrids: from fast weights to Gated DeltaNet (2020–2026)

    **Take first:** Module 15 (attention), Module 18 (the KV cache) and Module 36 (state-space models, Mamba,
    associative recall). Module 40 used the same detective corpus and training budget.

    **Goal:** attention remembers everything by keeping every key and value, so its memory grows with the
    context. The other way is a **fixed-size memory**: one matrix per head, written to as tokens go by. Since
    2020 that idea has been rebuilt four times — add to the memory, add and forget, overwrite, overwrite and
    forget — and the last version, **Gated DeltaNet**, is now inside Qwen3-Next, Qwen3.5 and (as Kimi Delta
    Attention) Kimi Linear, three of every four layers, with ordinary attention in the fourth.

    **You will:** count what a model must keep in memory while it generates · write linear attention two ways
    and see they are one function · add a decay and run it chunk by chunk, the way Mamba-2 does · write the
    delta rule and watch it overwrite where linear attention piles up · test recall on a synthetic task ·
    race attention, linear attention, Mamba-2, Gated DeltaNet and a 3 : 1 hybrid on the same data · time all
    of them on long prompts and long generations.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    In 1992 Jürgen Schmidhuber proposed **fast weights**: one network writes, token by token, into the weight
    matrix of another. Katharopoulos et al. (2020) noticed that attention *without the softmax* is exactly
    that — a matrix memory written with $k v^\top$ and read with $q$ — and called it linear attention. Schlag,
    Irie and Schmidhuber (2021) made the link explicit and proposed a better write, the **delta rule** from
    1960s adaptive filters (Widrow and Hoff). RetNet (2023) and GLA (2023) added forgetting; Mamba-2 (2024)
    showed that its selective SSM is the same recurrence. DeltaNet (2024) made the delta rule fast to train, and
    Gated DeltaNet (NVIDIA, ICLR 2025) combined the two. In 2025 the big labs shipped hybrids: MiniMax-01
    (7 linear layers per attention layer), Qwen3-Next and Kimi Linear (3 : 1), NVIDIA's Nemotron-H (Mamba-2 with
    about 8% attention). Then MiniMax-M2 went back to full attention. Section 8 says why.

    *Timeline: Module 00b. Family tree: Module 00b, section 12.*
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Attention is a memory that grows

    When a Transformer generates token $t$, each attention layer compares the new query with **every** earlier
    key and mixes **every** earlier value. So it keeps them all: the KV cache of Module 18, $2 \times$ heads
    $\times$ head width numbers per token per layer. At 4,096 tokens that's fine. At a million, the cache is
    bigger than the model.

    A **linear-attention layer** keeps one matrix per head instead, $S \in \mathbb{R}^{d_k \times d_v}$, whatever
    the context. It can't recall everything exactly — it has to *compress* — but its memory and its cost per
    generated token are constant. A **hybrid** keeps a few attention layers for exact look-ups and makes the
    rest linear.

    **Your turn:** `memory_costs(n_layers, attn_every, n_heads, d_head, context)` in `exercises.py`.
    """)
    return


@app.cell
def _(mo):
    ctx_pick = mo.ui.dropdown({"4,096": 4096, "32,768": 32768, "262,144": 262144, "1,048,576": 1048576},
                              value="262,144", label="context (tokens)")
    ctx_pick
    return (ctx_pick,)


@app.cell
def _(ctx_pick, ex, lj, mo):
    _C = ctx_pick.value
    _configs = [
        ("this module's model, all attention", 8, 1, 6, 64),
        ("this module's model, 3 : 1 hybrid", 8, 4, 6, 64),
        ("this module's model, all Gated DeltaNet", 8, 0, 6, 64),
        ("Qwen3-Next-shaped, all attention", 48, 1, 2, 256),
        ("Qwen3-Next-shaped, 3 : 1 hybrid", 48, 4, 2, 256),
    ]
    _att = [(_n, lj.attempt(ex.memory_costs, _L, _r, _h, _d, _C)) for _n, _L, _r, _h, _d in _configs]
    costs_ready = all(a.ok for _n, a in _att)
    if not costs_ready:
        _out = next(a for _n, a in _att if not a.ok).view(mo)
    else:
        _mb = lambda b: f"{b / 2**20:,.1f} MB" if b < 2**30 else f"{b / 2**30:,.2f} GB"
        _rows = "\n".join(f"| {n} | {a.value['attn_layers']} / {a.value['linear_layers']} | "
                          f"{_mb(a.value['kv_cache_bytes'])} | {_mb(a.value['state_bytes'])} | "
                          f"**{_mb(a.value['total_bytes'])}** |" for n, a in _att)
        _q_all, _q_hyb = _att[3][1].value["total_bytes"], _att[4][1].value["total_bytes"]
        _out = mo.md(f"""
    Memory per sequence after {_C:,} tokens (bf16 cache, f32 state):

    | model | attention / linear layers | KV cache | linear states | total |
    |---|---|---|---|---|
    {_rows}

    The hybrid needs **{_q_all / _q_hyb:.1f}×** less memory than the all-attention version of the same shape.
    That's close to 4× because the attention layers dominate: the states are a few megabytes and don't grow.
    (The "Qwen3-Next-shaped" rows use its 48 layers and attention heads, 2 KV heads of width 256, and give
    the linear layers the same head shape. The real model's Gated DeltaNet layers have 32 value heads of
    128 × 128 — four times more state, still constant.)
    """)
    _out
    return (costs_ready,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Linear attention: drop the softmax

    Causal attention for one head, writing the softmax as a normalized similarity:

    $$o_t = \frac{\sum_{s \le t} \mathrm{sim}(q_t, k_s)\, v_s}{\sum_{s \le t} \mathrm{sim}(q_t, k_s)},
    \qquad \mathrm{sim}(q, k) = e^{q \cdot k / \sqrt d}.$$

    Replace $\mathrm{sim}(q, k)$ with a plain dot product $q \cdot k$ (Katharopoulos et al. used
    $\phi(q) \cdot \phi(k)$ with $\phi = \mathrm{elu} + 1$; modern layers normalize $q$ and $k$ and drop the
    denominator, putting a norm after the layer instead). Then the sum can be **regrouped**:

    $$o_t = \sum_{s \le t} (q_t^\top k_s)\, v_s = q_t^\top \underbrace{\Big(\sum_{s \le t} k_s v_s^\top\Big)}_{S_t},
    \qquad S_t = S_{t-1} + k_t v_t^\top.$$

    Two ways to compute one function. The **parallel form** builds the $T \times T$ score matrix, masks it,
    and multiplies by $V$: matmuls, good for training. The **recurrent form** carries $S_t$, a $d_k \times d_v$
    matrix, and updates it once per token: constant memory, good for generating. $S$ is a **fast weight**
    matrix: a linear map from keys to values, written by the input as it streams by.

    **Your turn:** `causal_linear_attention(q, k, v)` (parallel) and `linear_attention_scan(q, k, v, log_a)`
    (recurrent; leave the decay for section 3, but accept it).
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lj, mo):
    _k = jax.random.split(jax.random.key(0), 3)
    _q, _kk, _v = (M.l2_norm(jax.random.normal(k, (2, 4, 256, 32))) for k in _k)
    _a = lj.attempt(ex.causal_linear_attention, _q, _kk, _v)
    _b = lj.attempt(ex.linear_attention_scan, _q, _kk, _v)
    lin_ready = _a.ok and _b.ok
    if not lin_ready:
        _out = (_a if not _a.ok else _b).view(mo)
    else:
        _err = float(jnp.abs(_a.value - _b.value).max() / jnp.abs(_b.value).max())
        _T, _d = 256, 32
        _out = mo.md(f"""
    Parallel and recurrent forms agree to **{_err:.1e}** ✅ (batch 2, 4 heads, {_T} steps, width {_d}).

    | form | work | memory while generating |
    |---|---|---|
    | parallel: $(QK^\\top \\odot M)V$ | $T^2 d$ multiply-adds ({_T**2 * _d:,} here) | every $k$, $v$ so far |
    | recurrent: $S_t = S_{{t-1}} + k_t v_t^\\top$ | $T d^2$ ({_T * _d**2:,} here) | one {_d} × {_d} matrix |
    """)
    _out
    return (lin_ready,)


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "decayed linear attention (recurrent form)",
                 solves=("causal_linear_attention", "linear_attention_scan"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What's wrong with plain linear attention?** It never forgets. Every $k v^\top$ is added to $S$ forever, so
    after thousands of tokens the memory is a blur of everything, and a key that was written twice returns the
    *sum* of its two values. The section 6 race shows the cost. The next three sections are the fixes.

    ## 3 · Forgetting: a decay, and Mamba-2 hiding inside

    Fade the memory a little every step, by a factor $a_t \in (0, 1]$ the layer computes from the token:

    $$S_t = a_t S_{t-1} + k_t v_t^\top, \qquad o_t = S_t^\top q_t.$$

    RetNet (2023) used a fixed $a$ per head; GLA (2023) and Mamba-2 (2024) let it depend on the input. Put
    Mamba-2's selective SSM next to it — state $h_t = a_t h_{t-1} + B_t x_t^\top$, output $y_t = h_t^\top C_t$,
    with $A$ a scalar times the identity — and rename $C \to q$, $B \to k$, $x \to v$: **the same recurrence**.
    That's Mamba-2's "state space duality": the SSM *is* a kind of linear attention.

    The parallel form still exists. Unroll the recurrence: $o_t = \sum_{s \le t} \big(a_{s+1} \cdots a_t\big)
    (q_t^\top k_s)\, v_s$. So it's masked attention with a **decay mask** $L_{ts} = a_{s+1} \cdots a_t$:

    $$O = (Q K^\top \odot L)\, V.$$

    Products of many numbers below 1 underflow, so work in logs: with $c_t = \sum_{m \le t} \log a_m$,
    $\log L_{ts} = c_t - c_s$, the **segment sum**.

    **Chunks.** The parallel form costs $T^2$; the recurrence is a long sequential loop that can't use matrix
    cores. Mamba-2's algorithm (and every linear layer since) does both: split the sequence into chunks of,
    say, 64. *Inside* a chunk, use the masked-attention form (a 64 × 64 matmul). *Between* chunks, pass the
    state along: one $d_k \times d_v$ matrix per chunk, through a scan that is $T/64$ steps long.

    **Your turn:** `segsum(log_a)`, then `chunked_linear_attention(q, k, v, log_a, chunk)` (the docstring
    lists the four steps).
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lin_ready, lj, mo, np, plt):
    _T = 24
    _la = jnp.full((_T,), np.log(0.9)).at[12].set(np.log(0.02))       # a sentence boundary at step 12
    _s = lj.attempt(ex.segsum, _la)
    _k = jax.random.split(jax.random.key(1), 4)
    _q, _kk, _v = (M.l2_norm(jax.random.normal(k, (2, 3, 256, 16))) for k in _k[:3])
    _lg = -jax.nn.softplus(jax.random.normal(_k[3], (2, 3, 256))) * 0.5
    _c = lj.attempt(ex.chunked_linear_attention, _q, _kk, _v, _lg, 64)
    chunk_ready = _s.ok and _c.ok and lin_ready
    if not (_s.ok and _c.ok):
        _out = (_s if not _s.ok else _c).view(mo)
    elif not lin_ready:
        _out = mo.md("*(Finish section 2's `linear_attention_scan` first: it's the reference for this check.)*")
    else:
        _ref = ex.linear_attention_scan(_q, _kk, _v, _lg)
        _errs = {c: float(jnp.abs(ex.chunked_linear_attention(_q, _kk, _v, _lg, c) - _ref).max()
                          / jnp.abs(_ref).max()) for c in (16, 64, 256)}
        _fig, _ax = plt.subplots(figsize=(3.6, 3.2))
        _ax.imshow(np.asarray(jnp.exp(_s.value)), cmap="viridis", vmin=0, vmax=1)
        _ax.set_xlabel("written at step s")
        _ax.set_ylabel("read at step t")
        _ax.set_title("decay mask exp(segsum)", fontsize=9)
        _fig.tight_layout()
        _out = mo.hstack([_fig, mo.md(f"""
    Your decay mask for $a_t = 0.9$, except $a_{{12}} = 0.02$: the layer "decided" to forget at step 12, so
    nothing written before it survives into the bottom-right block. Light = remembered, dark = gone.

    Chunked vs step by step (decay on, 256 steps): max relative difference
    **{_errs[16]:.1e}** with chunks of 16, **{_errs[64]:.1e}** with 64, **{_errs[256]:.1e}** with one chunk
    of 256 ✅. The chunk size changes the cost, not the answer.
    """)], widths=[1, 1.4])
    _out
    return (chunk_ready,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · The delta rule: overwrite instead of add

    Decay helps the memory make room, but it forgets *everything* at the same rate. The **delta rule** fixes
    writing itself. Before writing $(k_t, v_t)$, ask the memory what it already says for this key,
    $\hat v_t = S^\top k_t$, and write only the difference, scaled by a strength $\beta_t \in (0, 1)$:

    $$S_t = S_{t-1} + \beta_t\, k_t \big(v_t - S_{t-1}^\top k_t\big)^\top
          = (I - \beta_t k_t k_t^\top)\, S_{t-1} + \beta_t k_t v_t^\top.$$

    It's one step of gradient descent on $\tfrac12 \lVert S^\top k_t - v_t \rVert^2$ with learning rate
    $\beta_t$: the layer *learns* the key→value map while it reads (the "fast weights" view). With
    $\beta = 1$ and a unit-length key, the old value at $k_t$ is erased and replaced. **Gated DeltaNet**
    adds Mamba-2's decay in front:

    $$S_t = a_t (I - \beta_t k_t k_t^\top)\, S_{t-1} + \beta_t k_t v_t^\top.$$

    **Your turn:** `gated_delta_scan(q, k, v, beta, log_a)`, step by step.
    """)
    return


@app.cell
def _(M, chunk_ready, ex, jax, jnp, lj, mo, np, plt):
    _k = jax.random.split(jax.random.key(2), 5)
    _q, _kk, _v = (M.l2_norm(jax.random.normal(k, (2, 3, 256, 16))) for k in _k[:3])
    _lg = -jax.nn.softplus(jax.random.normal(_k[3], (2, 3, 256))) * 0.3
    _bt = jax.nn.sigmoid(jax.random.normal(_k[4], (2, 3, 256)))
    _a = lj.attempt(ex.gated_delta_scan, _q, _kk, _v, _bt, _lg)
    delta_ready = _a.ok
    overwrite = None
    if not _a.ok:
        _out = _a.view(mo)
    else:
        # The overwrite test: n keys, each written `w` times with a fresh value; then read every key back.
        _d, _res = 64, {}
        for _nk in (8, 32, 64):
            for _w in (1, 2, 4, 8):
                _r = np.random.default_rng(0)
                _K = _r.normal(size=(_nk, _d))
                _K /= np.linalg.norm(_K, axis=-1, keepdims=True)
                _W = _nk * _w
                _which = np.concatenate([_r.permutation(_nk) for _ in range(_w)])
                _V = _r.normal(size=(_W, _d))
                _V /= np.linalg.norm(_V, axis=-1, keepdims=True)
                _last = {int(i): t for t, i in enumerate(_which)}
                _tgt = np.stack([_V[_last[i]] for i in range(_nk)])
                _z = np.zeros((_nk, _d))
                _qq = jnp.asarray(np.concatenate([np.zeros((_W, _d)), _K]), jnp.float32)
                _vv = jnp.asarray(np.concatenate([_V, _z]), jnp.float32)
                _lin = ex.causal_linear_attention(_qq, jnp.asarray(np.concatenate([_K[_which], _z]), jnp.float32), _vv)
                _dl = ex.gated_delta_scan(_qq, jnp.asarray(np.concatenate([_K[_which], _K]), jnp.float32), _vv,
                                          jnp.asarray(np.concatenate([np.ones(_W), np.zeros(_nk)]), jnp.float32))
                _cos = lambda o, t=_tgt, n=_W: float(np.mean(
                    np.sum(np.asarray(o)[n:] * t, -1) / (np.linalg.norm(np.asarray(o)[n:], axis=-1) + 1e-9)))
                _res[(_nk, _w)] = (_cos(_lin), _cos(_dl))
        overwrite = _res
        _fig, _ax = plt.subplots(figsize=(5.2, 2.8))
        for _i, _nk in enumerate((8, 32, 64)):
            _ws = (1, 2, 4, 8)
            _ax.plot(_ws, [_res[(_nk, w)][1] for w in _ws], "-o", color=f"C{_i}", label=f"delta rule, {_nk} keys")
            _ax.plot(_ws, [_res[(_nk, w)][0] for w in _ws], "--", color=f"C{_i}", alpha=0.7,
                     label=f"linear attention, {_nk} keys")
        _ax.set_xscale("log", base=2)
        _ax.set_xlabel("times each key was written")
        _ax.set_ylabel("cosine(read, latest value)")
        _ax.set_ylim(0, 1.02)
        _ax.legend(fontsize=7, ncol=2)
        _fig.tight_layout()
        _txt = ""
        if chunk_ready:
            _ch = M.chunked_gated_delta(ex, _q, _kk, _v, _bt, _lg, 64)
            _txt = (f" The chunked version in `models.py` (below) matches your scan to "
                    f"**{float(jnp.abs(_ch - _a.value).max() / jnp.abs(_a.value).max()):.1e}**.")
        _out = mo.vstack([mo.md(f"""
    Your gated delta rule runs ✅.{_txt}

    **Overwriting, measured.** A 64 × 64 memory; 8, 32 or 64 unit-length random keys, each written 1–8 times
    with a fresh random value; then every key is read back and compared with its *latest* value (cosine, 1 =
    perfect). Linear attention returns the sum of everything ever written to the key; the delta rule (β = 1)
    returns the last write:
    """), _fig])
    _out
    return delta_ready, overwrite


@app.cell
def _(mo, overwrite):
    if overwrite is None:
        _out = mo.md("")
    else:
        _l8, _d8 = overwrite[(8, 8)]
        _l1, _d1 = overwrite[(64, 1)]
        _d64 = overwrite[(64, 8)][1]
        _out = mo.md(f"""
    With 8 keys written 8 times each, linear attention reads back **{_l8:.2f}** of the latest value and the
    delta rule **{_d8:.2f}**. Written once, they're about equal ({_l1:.2f} vs {_d1:.2f} with 64 keys): the delta
    rule's gain is in *updating* what's stored. Both get worse with more keys at any number of writes
    ({_d64:.2f} for the delta rule at 64 keys): a 64 × 64 matrix can only hold about 64 independent
    associations exactly. That limit is why hybrids keep some attention.
    """)
    _out
    return


@app.cell
def _(mo, rosetta):
    mo.vstack([rosetta.tabs(mo, "gated delta rule", solves="gated_delta_scan"),
               rosetta.tabs(mo, "triangular solve (forward substitution)")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Training it fast: the WY trick** (`models.py`, `chunked_gated_delta`; DeltaNet's 2024 paper). The delta
    rule's update multiplies $S$ by $(I - \beta k k^\top)$, which looks hopelessly sequential. The way out: name
    what each step *writes*, $u_t = \beta_t \big(v_t - (a_t S_{t-1})^\top k_t\big)$. Then
    $S_t = a_t S_{t-1} + k_t u_t^\top$ is plain decayed linear attention with values $u$, and section 3's
    chunked algorithm applies. Inside a chunk the $u$'s depend on each other, but only on *earlier* ones in the
    same chunk, so they satisfy a lower-triangular system:

    $$(I + A)\, U = \operatorname{diag}(\beta)\, V - \operatorname{diag}(\beta \gamma)\, K S_{\text{in}},
    \qquad A_{ij} = \beta_i \tfrac{\gamma_i}{\gamma_j}\, k_i^\top k_j \ (j < i),$$

    with $\gamma$ the decay since the chunk began. One triangular solve per chunk gives $T = (I + A)^{-1}$; then
    $U = T \operatorname{diag}(\beta) V - \big(T \operatorname{diag}(\beta\gamma) K\big) S_{\text{in}}$. The
    first term and the matrix $W = T \operatorname{diag}(\beta \gamma) K$ are computed for every chunk at once;
    only $W S_{\text{in}}$ waits for the state. Everything is matmuls on 64 × 64 blocks.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Recall: what a fixed memory can't do

    Module 36's **multi-query associative recall** (MQAR, Arora et al. 2023): show $n$ key→value pairs, then
    ask for every key again in a new order. Attention looks back at the pair. A linear layer must have stored
    all $n$ pairs in its state as they went by.

    `train.py --mqar` trains 2-layer models of width 128 for 3,000 steps at 32 to 256 pairs (keys drawn from
    512 tokens). Each linear layer has 2 heads of 32, so its state is 2 × 32 × 32 = 2,048 numbers. The
    contenders: each kind of linear layer; Mamba-2 and Gated DeltaNet again with a 10× slower *starting*
    decay; a hybrid whose second layer is attention; and pure attention twice, with and without the short
    causal convolution that every linear layer here has in front of it.
    """)
    return


@app.cell
def _(RUNS, json, mo, plt):
    _path = RUNS / "mqar.json"
    mqar = json.loads(_path.read_text()) if _path.exists() else {}
    _names = ["linear", "mamba2", "mamba2-slow", "deltanet", "gdn", "gdn-slow", "hybrid", "attention+conv",
              "attention"]
    _labels = {"linear": "linear attention", "mamba2": "+ decay (Mamba-2)", "mamba2-slow":
               "+ decay, slower at the start", "deltanet": "delta rule", "gdn": "gated delta rule",
               "gdn-slow": "gated delta, slower decay at the start", "hybrid": "gated delta + attention",
               "attention+conv": "attention × 2, with the short conv", "attention": "attention × 2, no conv"}
    _pairs = sorted({int(k.split("/")[1]) for k in mqar})
    if not mqar:
        _out = mo.callout(mo.md("""
    **No recall results yet** (about 40 minutes on an RX 7600 XT):
    ```
    containers/run.sh python modules/41-linear-attention-hybrids/train.py --mqar
    ```"""), kind="info")
    else:
        _rows = "\n".join(f"| {_labels[n]} | " + " | ".join(
            f"{mqar[f'{n}/{p}']['test_acc']:.0%}" if f"{n}/{p}" in mqar else "–" for p in _pairs) + " |"
            for n in _names)
        _fig, _ax = plt.subplots(figsize=(5.2, 2.8))
        for _i, _n in enumerate(_names):
            _ps = [p for p in _pairs if f"{_n}/{p}" in mqar]
            _ax.plot(_ps, [mqar[f"{_n}/{p}"]["test_acc"] for p in _ps], "-o", label=_labels[_n], color=f"C{_i}")
        _ax.set_xscale("log", base=2)
        _ax.set_xticks(_pairs, [str(p) for p in _pairs])
        _ax.set_xlabel("key-value pairs to remember")
        _ax.set_ylabel("recall accuracy")
        _ax.set_ylim(0, 1.02)
        _ax.legend(fontsize=6, ncol=2)
        _fig.tight_layout()
        _out = mo.vstack([mo.md(f"""
    Test accuracy on 1,024 fresh sequences (chance: under 1%). "–": not run (the hybrid at 256 pairs hung the
    GPU; docs/ROCM_GOTCHAS.md #20d).

    | model | {' | '.join(f'{p} pairs' for p in _pairs)} |
    |---|{'---|' * len(_pairs)}
    {_rows}
    """), _fig])
    _out
    return (mqar,)


@app.cell
def _(mo, mqar):
    def _acc(n, p):
        return mqar.get(f"{n}/{p}", {}).get("test_acc")

    def _pct(n, p):
        a = _acc(n, p)
        return "–" if a is None else f"{a:.0%}"

    _pairs = sorted({int(k.split("/")[1]) for k in mqar})
    _need = ("linear", "mamba2", "gdn", "attention", "attention+conv")
    if len(_pairs) >= 3 and all(_acc(n, _pairs[0]) is not None for n in _need):
        _lo, _hi = _pairs[0], _pairs[-1]
        _mid = 64 if 64 in _pairs else _pairs[1]
        _out = mo.md(f"""
    Four things to read off the table — not all of them the story the papers tell.

    **1 · Attention needs to see the previous token.** To answer, a model must find the earlier copy of the
    queried key and output the token *after* it, so every value first has to learn which key came before
    it. With the short convolution, two attention layers get **{_pct('attention+conv', _hi)}** even at {_hi}
    pairs; without it, {_pct('attention', _lo)} at {_lo}. A width-4 causal convolution hands each position its
    neighbours for free; plain attention must discover a previous-token head and an induction head
    (Module 37), and in 3,000 steps here it didn't. That's why H3, Based, Mamba and every linear layer in
    this module put a short convolution first.

    **2 · A fixed state runs out.** Plain linear attention, which never forgets, holds up longest:
    {_pct('linear', _lo)} at {_lo} pairs, {_pct('linear', _hi)} at {_hi}. Its 2,048-number state can't keep
    {_hi} pairs apart; attention keeps every key and doesn't care.

    **3 · Forgetting costs recall — until the gates learn to stay open.** Mamba-2's decay, with its standard
    starting values, forgets within tens of steps: {_pct('mamba2', _lo)} at {_lo} pairs but
    {_pct('mamba2', _mid)} at {_mid}. Start the same layer with a 10× slower decay and it gets
    {_pct('mamba2-slow', _mid)} at {_mid}. In a language model, forgetting old text is mostly right (section
    6); in a recall task it is exactly wrong, and 3,000 steps aren't enough to unlearn it.

    **4 · The delta rule learned this task slowly here.** Gated DeltaNet got {_pct('gdn', _lo)} at {_lo} pairs;
    with the slower starting decay, {_pct('gdn-slow', _lo)} at {_lo} but {_pct('gdn-slow', _mid)} at {_mid}
    (the plain delta rule: {_pct('deltanet', _lo)} at {_lo}). Learning rates of 3 × 10⁻³ and 3 × 10⁻⁴ didn't
    rescue it at {_mid} pairs. The Gated DeltaNet and Kimi Linear papers report the opposite at 1B+
    parameters and 100B tokens, so read this as "at width 128 and 3,000 steps", not as a verdict; the delta
    rule's write is a regression step whose strength β must also be learned, and that takes time. The
    hybrid (Gated DeltaNet first, plain attention second) got {_pct('hybrid', _lo)} at {_lo} pairs and
    {_pct('hybrid', _pairs[2])} at {_pairs[2]}: in 3,000 steps its attention layer didn't take over the
    recall the way the conv-equipped attention model did.

    The lesson the field drew from Zoology still holds: a fixed state compresses, and something with exact
    look-up has to sit somewhere in the stack — which is why the production models below keep some
    attention.
    """)
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · The race

    Same data (the detective corpus), same number of tokens (12,000 steps of 16 × 512 = 98M, as in Module 40),
    same optimizer, same size: 8 layers of width 384 with SwiGLU feed-forwards, 18.5M parameters each (±0.5%).
    Only the sequence mixer changes. Every linear layer gets the same surroundings as Gated DeltaNet's:
    a short causal convolution on $q, k, v$, unit-length $q$ and $k$, a norm on the output and a SiLU output
    gate. The attention layers use RoPE and a sigmoid output gate (Qwen's "gated attention").

    | variant | layers | mixer |
    |---|---|---|
    | attention | AAAAAAAA | softmax attention |
    | linear | LLLLLLLL | $S \mathrel{+}= k v^\top$ |
    | mamba2 | MMMMMMMM | + decay |
    | gdn | GGGGGGGG | + delta rule |
    | hybrid | GGGAGGGA | 3 : 1, like Qwen3-Next |

    The model stores a layout like `GGGAGGGA` as a repeating **block** (`GGGA`, twice) and runs the repeats
    with [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan), so XLA compiles one block instead of eight layers. That's how the model cards write it
    too: Qwen3-Next is "12 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))".
    """)
    return


@app.cell
def _(M, RUNS, mo, plt):
    _names = ["attention", "linear", "mamba2", "gdn", "hybrid"]
    full = {n: M.read_metrics(RUNS / n / "metrics.jsonl") for n in _names}
    full = {k: v for k, v in full.items() if v}
    if not full:
        _out = mo.callout(mo.md("""
    **No full runs yet.** Each takes 35–60 minutes on an RX 7600 XT; run them one at a time:
    ```
    T="containers/run.sh python modules/41-linear-attention-hybrids/train.py"
    for v in attention linear mamba2 gdn hybrid; do $T --variant $v; done
    ```
    then re-run this cell."""), kind="info")
    else:
        _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
        for _i, (_k, _m) in enumerate(full.items()):
            _ax.plot([r["step"] for r in _m], [r["val"] for r in _m], label=_k, color=f"C{_i}")
        _ax.set_ylim(top=min(5.0, max(r["val"] for m in full.values() for r in m)))
        _ax.set_xlabel("step")
        _ax.set_ylabel("validation loss")
        _ax.legend()
        _fig.tight_layout()
        _rows = "\n".join(
            f"| {k} | {m[-1]['step']:,} | {m[-1]['val']:.3f} | {min(r['val'] for r in m):.3f} | "
            f"{sum(r['tok_per_s'] for r in m[1:]) / max(len(m) - 1, 1):,.0f} | {m[-1]['elapsed_s'] / 60:.0f} min |"
            for k, m in full.items())
        _out = mo.vstack([_fig, mo.md(f"""
    | variant | steps | final val loss | best | tok/s (training) | time |
    |---|---|---|---|---|---|
    {_rows}
    """)])
    _out
    return (full,)


@app.cell
def _(full, mo):
    _f = {k: m[-1]["val"] for k, m in full.items() if m[-1]["step"] == 12000}
    if "attention" in _f and len(_f) > 1:
        _diffs = ", ".join(f"**{k}** {v - _f['attention']:+.3f}" for k, v in _f.items() if k != "attention")
        _order = " < ".join(sorted(_f, key=_f.get))

        def _step(a, b, what):
            if a not in _f or b not in _f:
                return ""
            d = _f[a] - _f[b]
            return f"{what}: **{d:+.3f}**" + (" (a tie)" if abs(d) < 0.01 else "") + "; "

        _ladder = ("Climbing the ladder — " + _step("mamba2", "linear", "adding forgetting (linear → mamba2)")
                   + _step("gdn", "mamba2", "adding the delta rule (mamba2 → gdn)")
                   + _step("hybrid", "gdn", "putting attention in every 4th layer (gdn → hybrid)")).rstrip("; ") + "."
        _out = mo.md(f"""
    **Reading the race.** Final validation loss relative to attention: {_diffs} (negative is better).
    From best to worst: {_order}. One run each, one seed: read gaps under ~0.01 as ties. {_ladder}
    The papers report Gated DeltaNet ahead of Mamba-2 at 1.3B parameters and 100B tokens, and hybrids ahead
    of pure attention; at 18M parameters and 512-token windows, attention's exact look-up over a short
    context is hard to beat, and the hybrid gets most of the way there with three quarters of its layers
    keeping a fixed-size state.
    """)
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(RUNS, full, json, mo, np, plt):
    _pos = {n: json.loads((RUNS / n / "pos_loss.json").read_text()) for n in full
            if (RUNS / n / "pos_loss.json").exists()}
    if len(_pos) >= 2:
        _fig, _ax = plt.subplots(figsize=(6.5, 3.0))
        _w = 16
        for _i, (_n, _p) in enumerate(_pos.items()):
            _sm = np.convolve(_p, np.ones(_w) / _w, mode="valid")
            _ax.plot(np.arange(len(_sm)) + _w // 2, _sm, label=_n, color=f"C{list(full).index(_n)}")
        _ax.set_xscale("log", base=2)
        _ax.set_xlabel("position in the 512-token window")
        _ax.set_ylabel("validation loss (smoothed)")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _late = {n: float(np.mean(p[256:])) for n, p in _pos.items()}
        _early = {n: float(np.mean(p[8:32])) for n, p in _pos.items()}
        _gain = {n: _early[n] - _late[n] for n in _pos}
        _best = max(_gain, key=_gain.get)
        _worst = min(_gain, key=_gain.get)
        _out = mo.vstack([_fig, mo.md(f"""
    **Who uses the context?** Loss at each position of the window, averaged over the validation batches. Every
    model starts high (position 0 has no context) and improves as it reads. From positions 8–32 to positions
    256–511, **{_best}** improves most ({_gain[_best]:.3f}) and **{_worst}** least ({_gain[_worst]:.3f}).
    A model that can't use distant tokens flattens out early; one that can keeps going down.
    """)])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Long prompts, long generations

    The point of all this is cost at long context. Two measurements on one layer (6 heads of 64, batch 1):

    * **Prefill:** process a whole prompt of $T$ tokens. Attention is $O(T^2)$; the chunked linear layers are
      $O(T)$; the step-by-step recurrence is $O(T)$ too, but one small step at a time.
    * **Decode:** generate one token after a context of $C$. Attention reads its whole KV cache; Gated
      DeltaNet updates a 6 × 64 × 64 state.

    (This module's attention is plain XLA: it builds the full $T \times T$ score matrix. Module 23's
    FlashAttention algorithm avoids storing it, which moves attention's curve down but doesn't change its
    slope: the work is still $T^2$.)
    """)
    return


@app.cell
def _(M, chunk_ready, delta_ready, ex, mo, plt):
    if not (chunk_ready and delta_ready):
        bench = None
        _out = mo.md("*(Needs sections 3 and 4's exercises.)*")
    else:
        bench = (M.bench_prefill(ex, lengths=(1024, 4096, 16384), cap=8192),
                 M.bench_decode(contexts=(1024, 8192, 65536)))
        _pre, _dec = bench
        _cols = ["attention", "mamba2 (chunked)", "gdn (chunked)", "gdn (step by step)"]
        _rows = "\n".join(f"| {T:,} | " + " | ".join(f"{r[c] * 1e3:,.1f} ms" if c in r else "–" for c in _cols)
                          + " |" for T, r in _pre.items())
        _drows = "\n".join(f"| {C:,} | {r['attention_ms']:.3f} ms | {r['gdn_ms']:.3f} ms | "
                           f"{r['kv_bytes'] / 2**20:,.1f} MB | {r['state_bytes'] / 2**10:,.0f} KB |"
                           for C, r in _dec.items())
        _fig, _ax = plt.subplots(1, 2, figsize=(8, 2.8))
        for _c in _cols:
            _Ts = [T for T, r in _pre.items() if _c in r]
            _ax[0].plot(_Ts, [_pre[T][_c] * 1e3 for T in _Ts], "-o", label=_c)
        _ax[0].set_xscale("log", base=2)
        _ax[0].set_yscale("log")
        _ax[0].set_xlabel("prompt length T")
        _ax[0].set_ylabel("ms (one layer)")
        _ax[0].legend(fontsize=7)
        _Cs = list(_dec)
        _ax[1].plot(_Cs, [_dec[C]["attention_ms"] for C in _Cs], "-o", label="attention (KV cache)")
        _ax[1].plot(_Cs, [_dec[C]["gdn_ms"] for C in _Cs], "-o", label="gated delta (state)")
        _ax[1].set_xscale("log", base=2)
        _ax[1].set_xlabel("context C")
        _ax[1].set_ylabel("ms per generated token")
        _ax[1].legend(fontsize=7)
        _fig.tight_layout()
        _out = mo.vstack([_fig, mo.md(f"""
    | prompt | {' | '.join(_cols)} |
    |---|---|---|---|---|
    {_rows}

    | context | attention, per token | gated delta, per token | KV cache | state |
    |---|---|---|---|---|
    {_drows}
    """)])
    _out
    return (bench,)


@app.cell
def _(bench, mo):
    if bench is None:
        _out = mo.md("")
    else:
        _pre, _dec = bench
        _Ts = [T for T in _pre if "attention" in _pre[T]]
        _t0, _t1 = _Ts[0], _Ts[-1]
        _ratio_att = _pre[_t1]["attention"] / _pre[_t0]["attention"]
        _ratio_gdn = _pre[_t1]["gdn (chunked)"] / _pre[_t0]["gdn (chunked)"]
        _tl = _Ts[-1]                                                     # longest length with attention measured
        _ga, _aa, _ma = _pre[_tl]["gdn (chunked)"], _pre[_tl]["attention"], _pre[_tl]["mamba2 (chunked)"]
        _c0, _c1 = list(_dec)[0], list(_dec)[-1]
        _da = _dec[_c1]["attention_ms"] / _dec[_c0]["attention_ms"]
        _dg = _dec[_c1]["gdn_ms"] / _dec[_c0]["gdn_ms"]
        _cross_txt = (f"At {_tl:,} tokens chunked Gated DeltaNet is already faster than attention "
                      f"({_ga * 1e3:.1f} vs {_aa * 1e3:.1f} ms)." if _ga < _aa else
                      f"At {_tl:,} tokens attention is still faster ({_aa * 1e3:.1f} ms vs {_ga * 1e3:.1f}): its "
                      f"matmuls are big and efficient, while the chunked delta rule runs a scan of small matmuls "
                      f"and a triangular solve per chunk. Mamba-2's chunked form, with no solve, takes "
                      f"{_ma * 1e3:.1f} ms. (Short prompts are dominated by launching kernels, so the first "
                      f"row says little.)")
        _cross_txt += (" Attention's column stops at 8,192 tokens: at 16,384 its 6 × 16,384 × 16,384 float32 "
                       "score matrix alone would need 6 GB, which is the wall FlashAttention (Module 23) removes.")
        _out = mo.md(f"""
    From {_t0:,} to {_t1:,} tokens ({_t1 // _t0}× longer), attention's prefill took **{_ratio_att:.0f}×** longer
    and chunked Gated DeltaNet's **{_ratio_gdn:.1f}×**. {_cross_txt} The step-by-step scan is the slowest way
    to run the same math: that's why the chunked algorithm matters for training and for prompts.

    Generating, attention's cost per token grew **{_da:.1f}×** from a {_c0:,}-token to a {_c1:,}-token context;
    Gated DeltaNet's **{_dg:.1f}×** (it doesn't depend on the context at all). At batch 1 both are dominated
    by launching small kernels; the memory column is the part that limits a real server, where many
    sequences share the GPU: Kimi Linear reports up to 6× faster decoding at a million tokens.
    """)
    _out
    return


@app.cell
def _(lineage, mo):
    _cols = [("Jamba", "jamba"), ("Nemotron-H", "nemotron_h"), ("MiniMax-01", "minimax01"),
             ("Qwen3-Next", "qwen3_next"), ("Kimi Linear", "kimi_linear")]
    _by = lineage.by_id()
    _table = lineage.compare_md([(h, _by[i].uses) for h, i in _cols], slots=["sequence mixing", "feed-forward"])
    mo.md(f"""
    ## 8 · The design space in 2025–2026

    | model | linear layer | ratio (linear : attention) | notes |
    |---|---|---|---|
    | Jamba (AI21, 2024) | Mamba | 7 : 1 | plus MoE; Module 36 |
    | MiniMax-01 (Jan 2025) | lightning attention (linear) | 7 : 1 | 456B, 45.9B active |
    | Nemotron-H (NVIDIA, 2025) | Mamba-2 | ~92% : 8% (4 of 52 layers in the 8B) | up to 3× faster inference |
    | Qwen3-Next (Alibaba, Sep 2025) | Gated DeltaNet | 3 : 1, gated attention | 80B, 3B active, 512 experts |
    | Kimi Linear (Moonshot, Oct 2025) | Kimi Delta Attention | 3 : 1, MLA | per-channel decay; 48B, 3B active |
    | Qwen3.5 (Feb 2026) | Gated DeltaNet | 3 : 1 | kept Qwen3-Next's layout at 397B |
    | Nemotron 3 Nano / Super (2025–26) | Mamba-2 | mostly Mamba-2 | plus LatentMoE in Super (Module 40) |

    **The case against.** MiniMax shipped a 7 : 1 hybrid in MiniMax-01 and M1, then made **M2** (Oct 2025) a
    full-attention model. Their post ("Why did M2 end up as a full attention model?") gives the reasons: at
    small scale the hybrid matched full attention on the usual benchmarks, but at larger scale it showed clear
    deficits in complex, multi-hop reasoning; measuring that reliably was expensive; the serving stack for
    linear attention (memory-bound kernels, low-precision state, prefix caching, speculative decoding) was
    still immature; and a sliding-window hybrid they tried got worse as the context grew, which they found
    unacceptable for agents. Others went another way entirely: DeepSeek-V3.2 kept full attention but made it
    **sparse** (each query reads only the top-k tokens a small indexer picks).

    So the honest summary: attention's exact look-up is still worth keeping; how *little* of it you can keep,
    and what to use for the rest, is an open question that the 3 : 1 Gated-DeltaNet layout currently leads.

    Blocks each model uses (from the course's family tree, `learnjax/lineage.py`):

    {_table}
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"Every layer's memory grows with the context, just more slowly": "a",
                 "Only the attention layers' KV cache grows; the Gated DeltaNet states stay the same size": "b",
                 "Nothing grows: the hybrid has a fixed-size memory": "c"},
        label="**Check yourself:** a 3 : 1 hybrid generates its millionth token. What has grown with the context?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("The linear layers' state is $d_k \\times d_v$ per head no matter how long the "
                              "context is."), kind="warn"),
        "b": mo.callout(mo.md("Right — a quarter of the layers keep a KV cache (so about 4× less memory than "
                              "all-attention), and the rest keep a constant state."), kind="success"),
        "c": mo.callout(mo.md("The attention layers still cache every key and value."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(41))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Songlin Yang, [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/) — a three-part
      blog on the delta rule and its chunkwise algorithm, by the first author of DeltaNet and Gated DeltaNet.
    * [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) — the Triton kernels behind
      most of these models, with reference implementations of every layer in this module.
    * Tri Dao and Albert Gu, [State Space Duality (Mamba-2)](https://tridao.me/blog/2024/mamba2-part1-model/) —
      the four-part blog on SSD and its chunked algorithm.
    * MiniMax, [Why did M2 end up as a full attention model?](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model)

    **Next:** Module 42, small models done well — distillation, pruning and borrowing from big models: what
    a single GPU can do best.
    """)
    return


if __name__ == "__main__":
    app.run()
