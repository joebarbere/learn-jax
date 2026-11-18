import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="40 · LatentMoE and Engram")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import json
    import os
    import subprocess
    import sys
    import time

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from tokenizers import Tokenizer

    import learnjax as lj
    from learnjax import rosetta
    from learnjax.data import detective_corpus

    ex = lj.load_exercises(mo.notebook_dir())
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")  # re-runs skip XLA compilation
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    _spec = importlib.util.spec_from_file_location("m40_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    RUNS = mo.notebook_dir().parent.parent / "runs" / "40-latentmoe-engram"
    return (M, RUNS, Tokenizer, detective_corpus, ex, jax, jnp, json, lj, mo, np, os, plt, rosetta, subprocess,
            sys, time)


@app.cell
def _(mo):
    mo.md(r"""
    # 40 · Latent experts and conditional memory: LatentMoE and Engram (2025–2026)

    **Take first:** Module 17 (MLA, DeepSeekMoE, aux-loss-free balancing) and Module 24 (expert parallelism,
    all-to-all). Module 16 introduced the detective corpus this lesson trains on.

    **Goal:** two 2026 ideas that add new kinds of *sparsity* to the DeepSeek-style model you replicated. NVIDIA's
    **LatentMoE** asks what experts cost to move and store, and shrinks the part that travels. DeepSeek's
    **Engram** asks what shouldn't need an expert at all — "Sherlock" is followed by "Holmes" — and answers it
    with a table lookup instead of computation.

    **You will:** count an MoE layer's parameters, FLOPs and network bytes, with and without a latent
    projection · build a LatentMoE layer · run it across 8 simulated devices and read the bytes off the
    compiled program · compress a tokenizer's vocabulary · hash n-grams into tables the way DeepSeek's code does
    (bit for bit) · write Engram's gate and its short causal convolution · race all of it against your
    deepseek-v3-jax baseline on the same data · switch Engram off to see what it memorized · prefetch its
    lookups from host memory.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Mixture-of-experts goes back to Jacobs, Jordan, Nowlan and Hinton (1991); Shazeer et al. (2017) made it
    sparse and huge, and DeepSeek (2024) made the experts small and many. By 2025 the limits were less about
    FLOPs than about bytes: moving tokens between GPUs and loading expert weights. In January 2026 two papers
    came at this from different sides. NVIDIA's LatentMoE (Elango et al.), shipped in Nemotron 3 Super and
    Ultra, runs the routed experts in a compressed latent space. DeepSeek's Engram revived a much older idea —
    the n-gram language model of the 1980s–2000s, and the hashed embeddings of the 2010s — as a lookup memory
    beside the experts, and found that the best models split their sparse parameters between the two.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(M, jax, mo):
    _base = M.param_report(jax.eval_shape(lambda: M.init_model(jax.random.key(0), 8192, M.BASE)), M.BASE)
    mo.md(rf"""
    ## 1 · Two walls an MoE runs into

    Your deepseek-v3-jax model, rebuilt here without its multi-token-prediction head, has
    **{_base['total']:.1f}M** parameters (36.2M with it), but each token uses only **{_base['active']:.1f}M** of
    them: of its 16 routed experts per layer, a token visits 4. That's the deal
    mixture-of-experts makes — the capacity of every expert for the compute of a few. Scale it up and two
    costs that FLOPs don't count take over.

    **Wall 1: bytes.** In Module 24 each device owned some experts, and an **all-to-all** carried every token's
    $d$-wide vector to the devices of its $k$ experts and the results back: $2kd$ numbers per token per MoE
    layer, over the slowest wires in the machine. And when generating one token at a time, the GPU spends
    its time *loading* each picked expert's weights, not multiplying them. LatentMoE shrinks both by making
    the experts narrower — and spends the savings on more of them.

    **Wall 2: lookups done by computation.** "Sherlock" → "Holmes", "Scotland" → "Yard", "said the" →
    "inspector": a lot of what a language model knows is local, fixed phrases. A Transformer has no lookup
    instruction, so it rebuilds them with attention and FFN layers, token by token, in its first few layers.
    Engram adds the missing primitive: a big table indexed by the last 2–3 tokens, read in $O(1)$.

    Both are **sparse**: most parameters sit idle for any given token. They're sparse along different axes, and
    DeepSeek's question is how to split a parameter budget between them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · LatentMoE: shrink what travels

    Elango et al. (2026) write one MoE layer as

    $$\mathrm{MoE}(x) = W_\uparrow\Big(\sum_{i \in \mathrm{TopK'}} p_i\, E_i(W_\downarrow x)\Big)
      + \sum_{j \in \text{shared}} E_j(x),\qquad
      W_\downarrow \in \mathbb{R}^{d\times \ell},\; W_\uparrow \in \mathbb{R}^{\ell \times d},\; \ell = d/\alpha.$$

    Read it right to left. The **router** still looks at the full $x$ (the gates $p_i$ use all $d$ features).
    The token is **projected down** to $\ell$ numbers *before* it's sent anywhere; every routed expert reads and
    writes width $\ell$ (its hidden width $m$ is unchanged); the weighted sum is **projected up** once. Shared
    experts stay at full width — they never travel.

    That makes each expert $\alpha$ times smaller, so the same parameter budget buys $N' = \alpha N$ experts.
    Two ways to spend it:

    * **ℓ-MoE_eff**: keep $k$ picks per token. A quarter of the bytes and expert FLOPs per token (at α = 4).
    * **ℓ-MoE_acc** (the one NVIDIA recommends and ships): $k' = \alpha k$ picks. Bytes per token come back to
      exactly what they were — $k'\ell = \alpha k \cdot d/\alpha = kd$ — but the layer has far more ways to
      combine experts. That's where the accuracy comes from.

    **Your turn:** `latent_moe_costs` — the arithmetic for one layer. Slide α and switch scales.
    """)
    return


@app.cell
def _(mo):
    alpha_pick = mo.ui.slider(steps=[1, 2, 4, 8], value=4, label="α = d / ℓ", show_value=True)
    scale_pick = mo.ui.radio(
        options={"your deepseek-v3-jax layer (d 384, m 256, 16 experts, top-4)": "small",
                 "a large baseline (d 4096, m 2688, 128 experts, top-6)": "large"},
        value="your deepseek-v3-jax layer (d 384, m 256, 16 experts, top-4)", label="MoE layer")
    mo.hstack([alpha_pick, scale_pick], justify="start", gap=2)
    return alpha_pick, scale_pick


@app.cell
def _(alpha_pick, ex, lj, mo, plt, scale_pick):
    _d, _m, _n, _k = (384, 256, 16, 4) if scale_pick.value == "small" else (4096, 2688, 128, 6)
    _att = {v: lj.attempt(ex.latent_moe_costs, _d, _m, _n, _k, alpha_pick.value, v)
            for v in ("standard", "eff", "acc")}
    if not all(a.ok for a in _att.values()):
        _out = next(a for a in _att.values() if not a.ok).view(mo)
    else:
        _c = {v: a.value for v, a in _att.items()}
        _rows = "\n".join(
            f"| {v} | {c['width']} | {c['experts']} | {c['active']} | {c['routed_params'] / 1e6:.2f}M "
            f"| {c['active_params'] / 1e6:.2f}M | {c['a2a_bytes_per_token']:,} |" for v, c in _c.items())
        _fig, _ax = plt.subplots(1, 3, figsize=(9, 2.4))
        for _a, _key, _title in zip(_ax, ("routed_params", "flops_per_token", "a2a_bytes_per_token"),
                                    ("parameters (routed)", "FLOPs per token", "bytes sent per token"),
                                    strict=True):
            _a.bar(list(_c), [_c[v][_key] / _c["standard"][_key] for v in _c], color=["C0", "C1", "C2"])
            _a.set_title(_title, fontsize=9)
            _a.axhline(1, color="grey", lw=0.6)
        _ax[0].set_ylabel("× standard")
        _fig.tight_layout()
        _out = mo.vstack([mo.md(f"""
    | variant | width | experts | picked | routed params | active params | bytes / token (bf16) |
    |---|---|---|---|---|---|---|
    {_rows}
    """), _fig])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    At α = 4 the "acc" layer has the same parameters (plus two small projections), the same FLOPs and the same
    bytes per token as the standard one — and four times as many experts, each picked token choosing 4× more of
    them. "eff" pays a quarter of the FLOPs and bytes. At α = 1 all three agree: there's nothing to compress.

    **Two things the paper is honest about.** The projections cost something: NVIDIA reports up to ~9% overhead
    at trillion-parameter scale. And the shipped Nemotron 3 Super uses 512 experts, top-22 with $\ell$ = 1024,
    $d$ = 4096 — the rule says top-24; the papers don't explain the difference.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Build a LatentMoE layer

    **Your turn:** `latent_moe(p, x, gates)`. The router has already run on the full $x$ — that's why the gates
    come in as an argument. The check below gives it identity projections and $\ell = d$: then LatentMoE must
    be *exactly* Module 17's DeepSeekMoE layer (models.py's `dense_moe`).
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lj, mo):
    _cfg = M.variant(M.SMALL, "moe")
    _p = M.init_model(jax.random.key(1), 64, _cfg)["layers"][1]["moe"]
    _x = jax.random.normal(jax.random.key(2), (24, _cfg["d_model"]))
    _scores = jax.nn.sigmoid(_x @ _p["router"])
    _gates = M.moe_route(_scores, jnp.zeros(_cfg["n_routed"]), _cfg["top_k"])
    _a = lj.attempt(ex.latent_moe, {**_p, "down": jnp.eye(_cfg["d_model"]), "up": jnp.eye(_cfg["d_model"])},
                    _x, _gates)
    _lat = M.variant(M.SMALL, "latent-acc")
    _pl = M.init_model(jax.random.key(1), 64, _lat)["layers"][1]["moe"]
    _b = lj.attempt(ex.latent_moe, _pl, _x, M.moe_route(jax.nn.sigmoid(_x @ _pl["router"]),
                                                        jnp.zeros(_lat["n_routed"]), _lat["top_k"]))
    if _a.ok and _b.ok:
        _err = float(jnp.abs(_a.value - M.dense_moe(_p, _x, _gates)).max())
        latent_ready = True
        _out = mo.md(f"""
    With identity projections your layer matches DeepSeekMoE to **{_err:.1e}** ✅. A real latent layer
    (α = 4: width {_lat['latent']}, {_lat['n_routed']} experts, top-{_lat['top_k']}) maps
    {tuple(_x.shape)} → {tuple(_b.value.shape)}.
    """)
    else:
        latent_ready = False
        _out = (_a if not _a.ok else _b).view(mo)
    _out
    return (latent_ready,)


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "LatentMoE layer", solves="latent_moe")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Count the bytes on 8 simulated devices

    Formulas are claims; a compiled program is evidence. `ep_latent.py` (next to this notebook) runs the layer
    with **expert parallelism** on 8 simulated CPU devices, as Module 24 did, but closer to real systems:
    top-$k$ routing, several experts per device, and a fixed **capacity** of slots per pair of devices
    (capacity factor 1.25 — a token past it would be dropped). Each device projects its tokens down *before*
    the all-to-all and up *after* the return trip. It checks the result against a single-device layer, then
    reads the size of every all-to-all out of the lowered program. It runs in a subprocess, because the CPU must
    be split into 8 devices before JAX starts, and this notebook is already using your GPU.
    """)
    return


@app.cell
def _(json, mo, os, subprocess, sys):
    _env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    _r = subprocess.run([sys.executable, str(mo.notebook_dir() / "ep_latent.py")], capture_output=True, text=True,
                        env=_env, timeout=600)
    ep = json.loads(_r.stdout.strip().splitlines()[-1]) if _r.returncode == 0 else None
    if ep is None:
        _out = mo.callout(mo.md(f"ep_latent.py failed:\n```\n{_r.stderr[-1500:]}\n```"), kind="danger")
    else:
        _s = ep["standard"]["act_bytes_per_device"]
        _rows = "\n".join(
            f"| {k} | {v['width']} | {v['experts']} | {v['top_k']} | {v['capacity']} | {v['n_all_to_all']} "
            f"| {v['act_bytes_per_device']:,} ({v['act_bytes_per_device'] / _s:.2f}×) | {v['id_bytes_per_device']:,} "
            f"| {v['rel_err']:.1e} |" for k, v in ep.items())
        _out = mo.md(f"""
    | layer | width | experts | top-k | slots per device pair | all-to-alls | activation bytes per device | expert-id bytes | error vs 1 device |
    |---|---|---|---|---|---|---|---|---|
    {_rows}

    The compiled program agrees with the paper: **eff** moves a quarter of the activation bytes, **acc** exactly
    as many as the standard layer. Two details the formula hides: acc has 4× more *slots*, so the small
    integer messages saying which expert each slot is for grow 4× (still tiny next to the activations), and the
    errors of ~{max(v['rel_err'] for v in ep.values()):.0e} are bf16 rounding on the wire — activations travel
    as 2-byte numbers. (Nemotron goes further: FP4 for the dispatch, bf16 for the combine.)
    """)
    _out
    return (ep,)


@app.cell
def _(mo):
    mo.md(r"""
    **What simulation can't tell you:** time. On a real cluster the all-to-all runs over NVLink or InfiniBand
    while other experts compute; NVIDIA measured the trade on H100s and on GB200 racks. On your single GPU the
    model below runs every expert densely (as Module 17 did), so "eff" is *not* faster here — its savings are
    in bytes and in a sparse kernel you'd write with
    [`jax.lax.ragged_dot`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html). What the race
    in section 8 *can* measure is the paper's actual claim: quality per parameter.
    """)
    return


@app.cell
def _(Tokenizer, detective_corpus, M, mo):
    corpus = detective_corpus()
    tok = Tokenizer.from_file(str(corpus["tokenizer_path"]))
    texts = M.token_texts(tok)
    mo.md(rf"""
    ## 5 · Engram, part 1: tokens that mean the same thing

    Engram's table is indexed by the last few tokens. A byte-level BPE vocabulary spends ids on surface
    variations: `the`, ` the`, `The`, ` The`, `THE` and ` THE` are six different tokens in the detective
    tokenizer, so "in the" and "In The" would land in different rows, each seen less often. Engram first maps every token id to a
    **canonical** id (NFKC, accents stripped, lowercase, whitespace collapsed) — DeepSeek's paper reports 23%
    fewer ids for its 128k vocabulary. The model itself still sees the original tokens; only the table index
    uses the compressed ones.

    **Your turn:** `compress_vocab(texts)`. The detective tokenizer has **{len(texts):,}** tokens;
    {sum(chr(0xFFFD) in t for t in texts)} of them are fragments of multi-byte UTF-8 characters (on their own
    they decode to the replacement character U+FFFD) and must never be merged.
    """)
    return corpus, texts, tok


@app.cell
def _(ex, lj, mo, np, texts):
    _a = lj.attempt(ex.compress_vocab, texts)
    if _a.ok:
        _lookup, _n = _a.value
        _lookup = np.asarray(_lookup)
        _groups = {}
        for _i, _g in enumerate(_lookup.tolist()):
            _groups.setdefault(_g, []).append(_i)
        _big = sorted(_groups.values(), key=len, reverse=True)
        _show = [g for g in _big if any(texts[i].strip().isalpha() for i in g)][:6]
        _fmt = lambda s: "`" + repr(s)[1:-1].replace("|", "\\|").replace("`", "'") + "`"
        _rows = "\n".join("| " + ", ".join(_fmt(texts[i]) for i in g) + " |" for g in _show)
        compressed = (_lookup, _n)
        _out = mo.md(f"""
    {len(texts):,} → **{_n:,}** canonical ids: **{1 - _n / len(texts):.1%}** fewer (paper: 23.4% on a 128k
    vocabulary — a small vocabulary has fewer variants to merge). Longer words mostly exist in one form
    only: ` Holmes` has no `HOLMES` twin in an 8K vocabulary.

    | some of the biggest groups |
    |---|
    {_rows}

    These ids match DeepSeek's `engram_demo_v1.py` exactly: the course checked your reference implementation
    against the demo's own normalizer on this tokenizer (6,890 groups both ways).
    """)
    else:
        compressed = None
        _out = _a.view(mo)
    _out
    return (compressed,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Engram, part 2: hash n-grams into tables

    There are 6,890² ≈ 47 million possible bigrams and 6,890³ trigrams — far too many rows to store. Engram
    **hashes** each n-gram into a table of a few tens of thousands of rows and accepts **collisions**:

    $$\text{mix}_t = (x_t \cdot m_0) \oplus (x_{t-1} \cdot m_1) \oplus \dots \oplus (x_{t-n+1} \cdot m_{n-1}),
      \qquad \text{row}_{n,j} = \text{mix}_t \bmod P_{n,j}$$

    ($\oplus$ = XOR, $m_i$ random odd 64-bit multipliers, $P_{n,j}$ distinct primes.) Two n-grams that collide in
    one head almost never collide in another, because each head reduces modulo a different prime; Engram
    concatenates the rows of **all** heads ($K$ per order), so the model can tell colliding n-grams apart by the
    heads where they differ. That's the hashing trick of the 2010s (Weinberger et al.; Svenstrup et al.'s hash
    embeddings), applied to an LLM's n-grams.

    This runs on the **host**, in NumPy, with 64-bit integers: the rows depend only on the token ids, so they can
    be computed in the data pipeline before the model runs (JAX uses 32-bit integers unless you enable x64).
    The multipliers are chosen so that id × multiplier never overflows 64 bits.

    **Your turn:** `ngram_hashes(ids, multipliers, primes, pad_id)`.
    """)
    return


@app.cell
def _(M, compressed, ex, lj, mo, np, tok):
    _cfg = M.variant(M.BASE, "engram", rho=0.75)
    _e = _cfg["engram"]
    _sent = "Sherlock Holmes said to the inspector from Scotland Yard"
    _ids = np.array(tok.encode(_sent).ids, np.int64)
    if compressed is None:
        hash_ready = False
        _out = mo.md("*(Finish `compress_vocab` first.)*")
    else:
        _lookup, _n = compressed
        _mult = M.multipliers(_n, max(_e["orders"]), _e["seed"], _e["layer"])
        _primes = M.head_primes(_e["rows"], _e["orders"], _e["heads"])
        _a = lj.attempt(ex.ngram_hashes, _lookup[_ids][None], _mult, _primes, int(_lookup[0]))
        hash_ready = _a.ok
        if _a.ok:
            _h = np.asarray(_a.value)[0]
            _rows = "\n".join(f"| `{tok.decode([int(i)])}` | " + " | ".join(str(v) for v in _h[t]) + " |"
                              for t, i in enumerate(_ids))
            _hdr = " | ".join([f"2-gram h{j}" for j in range(_e['heads'])] + [f"3-gram h{j}" for j in range(_e['heads'])])
            _out = mo.md(f"""
    Table sizes (primes): {_primes}. Your rows for one sentence — every position reads
    {2 * _e['heads']} rows, one per (order, head):

    | token | {_hdr} |
    |---|{'---|' * 2 * _e['heads']}
    {_rows}
    """)
        else:
            _out = _a.view(mo)
    _out
    return (hash_ready,)


@app.cell
def _(M, compressed, corpus, hash_ready, mo, np, plt):
    if not hash_ready:
        _out = mo.md("")
    else:
        _lookup, _n = compressed
        _x = _lookup[corpus["train"][:5_000_000].astype(np.int64)]
        _bi = np.unique(_x[:-1] * _n + _x[1:])
        _tri = np.unique((_x[:-2] * _n + _x[1:-1]) * _n + _x[2:])
        _sizes = np.array([1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000])
        _rng = np.random.default_rng(0)

        def _collide(n_keys, size, heads):
            """Fraction of distinct n-grams that share ALL `heads` rows with some other n-gram."""
            _rows = np.stack([_rng.integers(0, size, n_keys) for _ in range(heads)], 1)
            _, _inv, _cnt = np.unique(_rows, axis=0, return_inverse=True, return_counts=True)
            return float((_cnt[_inv] > 1).mean())

        _at = {(_k, _h): _collide(_k, 30_000, _h) for _k in (len(_bi), len(_tri)) for _h in (1, 4)}
        _fig, _ax = plt.subplots(figsize=(6, 3))
        for _keys, _name, _c in ((len(_bi), "bigrams", "C0"), (len(_tri), "trigrams", "C1")):
            _ax.plot(_sizes, [_collide(_keys, s, 1) for s in _sizes], "-o", color=_c, ms=3, label=f"{_name}, 1 head")
            _ax.plot(_sizes, [_collide(_keys, s, 4) for s in _sizes], "--o", color=_c, ms=3, label=f"{_name}, all 4 heads")
        _ax.axvline(30_000, color="grey", lw=0.6)
        _ax.set_xscale("log")
        _ax.set_xlabel("rows per head")
        _ax.set_ylabel("share of n-grams colliding")
        _ax.legend(fontsize=7)
        _fig.tight_layout()
        _out = mo.vstack([mo.md(f"""
    The first 5M training tokens hold **{len(_bi):,}** distinct bigrams and **{len(_tri):,}** distinct trigrams.
    With ~30k rows per head (the grey line: this module's iso-parameter Engram), every row of a single head is
    shared: **{_at[(len(_bi), 1)]:.0%}** of bigrams and **{_at[(len(_tri), 1)]:.0%}** of trigrams collide
    with another. Across *all four* heads at once, only **{_at[(len(_bi), 4)]:.2%}** of bigrams and
    **{_at[(len(_tri), 4)]:.2%}** of trigrams have a twin — the concatenated rows still tell them apart.
    (Collisions here are simulated with uniform random rows, which is what a good hash gives.)
    """), _fig])
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "n-gram hash (multiplicative XOR)", solves="ngram_hashes")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Engram, part 3: a gate, and a short convolution

    The concatenated rows $e_t$ are static memory: the same for "Holmes" whatever the story says. A **gate**
    decides how much of it to use, by comparing the memory with the hidden state $h_t$ (paper eq. 3–4):

    $$k_t = W_K e_t,\quad v_t = W_V e_t,\quad
      \alpha_t = \sigma\!\left(\frac{\mathrm{RMSNorm}(h_t)^\top \mathrm{RMSNorm}(k_t)}{\sqrt d}\right),\quad
      \tilde v_t = \alpha_t v_t$$

    If the memory agrees with the context, $\alpha_t$ opens; if a collision brought back nonsense, it closes.
    (DeepSeek's demo code adds a signed square root before the sigmoid; the paper doesn't, and this course
    follows the paper.) Then a **short causal convolution** lets neighbouring positions' memories mix — kernel
    4, dilation 3 (the largest n-gram order), one filter per channel (eq. 5):

    $$Y = \tilde V + \mathrm{SiLU}\big(\mathrm{Conv1D}(\mathrm{RMSNorm}(\tilde V))\big), \qquad
      H \leftarrow H + Y \;\text{ before the layer's attention.}$$

    The conv's weights start at **zero**, so at step 0 the block adds exactly $\tilde V$ and training grows the
    convolution in. The paper uses Engram at layers 2 and 15 of 30 (layer 2 alone was its best single
    placement); this model, with 6 layers, reads it before layer 1 (0-based).

    **Your turn:** `engram_gate(h, e, p)` and `causal_dwconv(x, w, dilation)`.
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lj, mo, np):
    _cfg = M.variant(M.SMALL, "engram", rho=0.75, rows=4093)
    _p = M.init_engram(jax.random.key(3), _cfg)
    _h = jax.random.normal(jax.random.key(4), (2, 16, _cfg["d_model"]))
    _ids = jax.random.randint(jax.random.key(5), (2, 16, 8), 0, 4093)
    _g = lj.attempt(ex.engram_gate, _h, M.gather_rows(_p["table"], _ids, _cfg), _p)
    _w = jax.random.normal(jax.random.key(6), (4, 3))
    _x = jnp.zeros((1, 12, 3)).at[0, 2].set(1.0)                               # an impulse at t = 2
    _c = lj.attempt(ex.causal_dwconv, _x, _w, 3)
    engram_ready = _g.ok and _c.ok
    if engram_ready:
        _y, _ = M.engram_block(ex, _p, _h, _ids, _cfg)
        _resp = np.asarray(_c.value)[0, :, 0]
        _out = mo.md(f"""
    Gate values on random inputs: mean **{float(_g.value[1].mean()):.2f}**, range
    {float(_g.value[1].min()):.2f}–{float(_g.value[1].max()):.2f} (a fresh gate sits near ½). Your conv's response
    to an impulse at t = 2 on channel 0, dilation 3: nonzero at t = {[int(t) for t in np.nonzero(np.abs(_resp) > 1e-7)[0]]}
    — the impulse reappears 0, 3, 6 and 9 steps later, never earlier. And with the zero-initialized conv, the
    whole block returns exactly α·v: max difference **{float(jnp.abs(_y - _g.value[0]).max()):.1e}**.
    """)
    else:
        _out = (_g if not _g.ok else _c).view(mo)
    _out
    return (engram_ready,)


@app.cell
def _(mo, rosetta):
    mo.vstack([rosetta.tabs(mo, "Engram context gate", solves="engram_gate"),
               rosetta.tabs(mo, "causal depthwise dilated convolution", solves="causal_dwconv")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · The race

    Same data (the deepseek-v3-jax detective corpus), same tokens per step, same optimizer, same total
    parameters (±2%). Four contenders:

    * **moe** — your deepseek-v3-jax architecture: MLA, 1 shared + 16 routed experts, top-4 (no multi-token
      prediction here).
    * **latent-acc** — LatentMoE, α = 4: experts of width 96, 64 of them, top-16.
    * **latent-eff** — the same, top-4.
    * **engram** — 12 of the 16 experts kept; the other 4 experts' parameters (×5 layers) moved into an Engram
      table of 8 heads × ~30k rows before layer 1. Same active compute.

    First a live sprint at Module 17's tiny width (not iso-parameter: the Engram table here is a fixed
    8 × 8k rows), then the full runs. The first time, each model takes about a minute to compile (mostly
    XLA autotuning its matmuls); the compiled programs go into the persistent cache in `/cache`, so
    re-runs take seconds.
    """)
    return


@app.cell
def _(M, compressed, corpus, engram_ready, ex, hash_ready, jax, jnp, latent_ready, mo, np, texts, time):
    _train = corpus["train"][:3_000_000]
    _data = jnp.asarray(_train.astype(np.int32))
    _plan = [("moe", M.variant(M.SMALL, "moe"))]
    if latent_ready:
        _plan.append(("latent-acc", M.variant(M.SMALL, "latent-acc")))
    if engram_ready and hash_ready and compressed is not None:
        _plan.append(("engram", M.variant(M.SMALL, "engram", rho=0.75, rows=8191)))
    live = {}
    for _name, _cfg in _plan:
        _params = M.init_model(jax.random.key(0), 8192, _cfg)
        _hasher = M.Hasher(ex, texts, _cfg) if "engram" in _cfg else None
        _hashes = None if _hasher is None else jnp.asarray(_hasher.corpus(_train))
        _vh = None if _hasher is None else _hasher.corpus(corpus["val"])
        _opt = M.make_optimizer(_params, 2e-3, 400, warmup=40)
        _state, _bias = _opt.init(_params), M.init_biases(_cfg)
        _chunk, _eval = M.make_trainer(ex, _cfg, _opt, batch=16, seq_len=128)
        _vals = M.val_batches(corpus["val"], _vh, 16, 128, n=4)
        _t0, _curve = time.perf_counter(), []
        for _i in range(16):  # 25 steps per jit call, an eval every 100
            _params, _state, _bias, _l, _ = _chunk(_params, _state, _bias, _data, _hashes, jax.random.key(_i), 25)
            if _i % 4 == 3:
                _curve.append(float(np.mean([_eval(_params, _bias, x, y, hw) for x, y, hw in _vals])))
        live[_name] = {"val": _curve, "s": time.perf_counter() - _t0, "params": M.param_report(_params, _cfg),
                       "model": (_params, _bias, _cfg, _hasher)}
    mo.md("| variant | params | val loss after 100 / 200 / 300 / 400 steps | time |\n|---|---|---|---|\n" + "\n".join(
        f"| {k} | {v['params']['total']:.2f}M | " + " / ".join(f"{x:.3f}" for x in v["val"]) + f" | {v['s']:.0f} s |"
        for k, v in live.items()) + ("" if len(live) == 3 else "\n\n*(More rows appear as you finish the exercises.)*"))
    return (live,)


@app.cell
def _(M, RUNS, mo, plt):
    _names = {"moe": "moe", "latent-acc": "latent-acc", "latent-eff": "latent-eff", "engram-r75-l1": "engram"}
    full = {lbl: M.read_metrics(RUNS / d / "metrics.jsonl") for d, lbl in _names.items()}
    full = {k: v for k, v in full.items() if v}
    if not full:
        _out = mo.callout(mo.md("""
    **No full runs yet.** Each takes 30–45 minutes on an RX 7600 XT; run them one at a time:
    ```
    containers/run.sh python modules/40-latentmoe-engram/train.py --variant moe
    containers/run.sh python modules/40-latentmoe-engram/train.py --variant engram
    containers/run.sh python modules/40-latentmoe-engram/train.py --variant latent-acc
    containers/run.sh python modules/40-latentmoe-engram/train.py --variant latent-eff
    ```
    then re-run this cell."""), kind="info")
    else:
        _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
        for _k, _m in full.items():
            _ax.plot([r["step"] for r in _m], [r["val"] for r in _m], label=_k)
        _ax.set_ylim(top=min(4.6, max(r["val"] for m in full.values() for r in m)))
        _ax.set_xlabel("step")
        _ax.set_ylabel("validation loss")
        _ax.legend()
        _fig.tight_layout()
        _rows = "\n".join(
            f"| {k} | {m[-1]['step']:,} | {m[-1]['val']:.3f} | {min(r['val'] for r in m):.3f} | "
            f"{sum(r['tok_per_s'] for r in m[1:]) / max(len(m) - 1, 1):,.0f} | {m[-1]['max_load']:.2f}× | "
            f"{m[-1]['elapsed_s'] / 60:.0f} min |" for k, m in full.items())
        _out = mo.vstack([_fig, mo.md(f"""
    | variant | steps | final val loss | best | tok/s | busiest expert / mean | time |
    |---|---|---|---|---|---|---|
    {_rows}
    """)])
    _out
    return (full,)


@app.cell
def _(full, mo):
    _f = {k: m[-1]["val"] for k, m in full.items() if m[-1]["step"] == 12000}
    if "moe" in _f and len(_f) > 1:
        _diffs = ", ".join(f"**{k}** {v - _f['moe']:+.3f}" for k, v in _f.items() if k != "moe")
        _best = min(_f, key=_f.get)
        _out = mo.md(f"""
    **Reading the race.** Final validation loss relative to the moe baseline: {_diffs} (negative is better).
    The best of these runs is **{_best}**. One run per variant, one seed: differences under ~0.01 are inside
    seed noise at this scale, so read them as "no difference". For scale: DeepSeek's Engram-27B beat its MoE
    baseline by 0.012 in validation loss (1.634 → 1.622) after 262B tokens; NVIDIA's acc variant gained 4.6
    MMLU-Pro points at 73B. Your runs are ~1,000× smaller and see 98M tokens, so what matters is the direction
    and the mechanism, not matching the size of the effect.
    """)
        _e, _b = full.get("engram"), full.get("moe")
        if _e and _b:
            _bv = {r["step"]: r["val"] for r in _b}
            _lead = [r["step"] for r in _e if r["step"] in _bv and r["val"] < _bv[r["step"]]]
            _gap = {k: m[-1]["val"] - m[-1]["loss"] for k, m in full.items() if m[-1]["step"] == 12000}
            if _lead and _f.get("engram", 0) > _f["moe"]:
                _out = mo.vstack([_out, mo.md(f"""
    **Why Engram leads and then falls behind.** Engram's validation loss was below the baseline's at
    {len(_lead)} of {len(_e)} evaluations, the last at step {max(_lead):,}; after that the baseline caught up.
    Now compare the gap between validation and training loss at the end: **engram {_gap['engram']:.2f}**
    against **moe {_gap['moe']:.2f}**. A larger gap is the signature of memorizing. The detective corpus has
    21.7 million training tokens, so 12,000 steps of 8,192 tokens pass over it about 4.5 times. A hash
    table with hundreds of thousands of rows, looked up by exactly the n-grams in the training text, is the
    perfect place to store those n-grams' continuations verbatim: each row gets a gradient whenever its
    n-gram appears, with no sharing to keep it general, and on the second pass it has seen them all. DeepSeek trained on 262B tokens and never repeated data, so their table had to learn
    what *recurs*, not what *occurred*. The lesson for a small model: conditional memory pays when the
    data is much bigger than the memory. Here it isn't, so the table gives you a fast start and then
    overfits. Try `--steps 3000` runs (below), dropout on the retrieved rows, or a smaller table.
    """)])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(M, RUNS, mo, plt):
    _sweep = {1.0: "moe-3000", 0.75: "engram-r75-l1-3000", 0.5: "engram-r50-l1-3000", 0.25: "engram-r25-l1-3000"}
    _place = {0: "engram-r75-l0-3000", 1: "engram-r75-l1-3000", 2: "engram-r75-l2-3000", 4: "engram-r75-l4-3000"}
    _sv = {r: M.read_metrics(RUNS / d / "metrics.jsonl") for r, d in _sweep.items()}
    _pv = {layer: M.read_metrics(RUNS / d / "metrics.jsonl") for layer, d in _place.items()}
    _sv = {r: m[-1]["val"] for r, m in _sv.items() if m and m[-1]["step"] == 3000}
    _pv = {layer: m[-1]["val"] for layer, m in _pv.items() if m and m[-1]["step"] == 3000}
    sweeps = (_sv, _pv)
    if len(_sv) < 2:
        _out = mo.callout(mo.md("""
    **Sweeps (optional, ~7 minutes each):** how much of the budget to give Engram, and where to put it.
    ```
    T="containers/run.sh python modules/40-latentmoe-engram/train.py --steps 3000"
    $T --variant moe
    for r in 0.75 0.5 0.25; do $T --variant engram --rho $r; done
    for l in 0 2 4; do $T --variant engram --engram-layer $l; done
    ```"""), kind="info")
    else:
        _fig, _ax = plt.subplots(1, 2, figsize=(8, 2.8))
        _r = sorted(_sv)
        _ax[0].plot([100 * x for x in _r], [_sv[x] for x in _r], "-o")
        _ax[0].set_xlabel("% of the sparse budget kept in experts")
        _ax[0].set_ylabel("val loss at 3,000 steps")
        _ax[0].set_title("budget split (paper's best: 75–80%)", fontsize=9)
        if _pv:
            _ax[1].bar([str(x) for x in sorted(_pv)], [_pv[x] for x in sorted(_pv)], color="C1")
            _ax[1].set_ylim(min(_pv.values()) - 0.03, max(_pv.values()) + 0.02)
            _ax[1].set_xlabel("Engram before layer (0-based)")
            _ax[1].set_title("placement (paper: early is best)", fontsize=9)
        _fig.tight_layout()
        _out = _fig
    _out
    return (sweeps,)


@app.cell
def _(mo, sweeps):
    _sv, _pv = sweeps
    if len(_sv) >= 3:
        _best = min(_sv, key=_sv.get)
        _txt = (f"The lowest loss at 3,000 steps came from keeping **{_best:.0%}** of the budget in experts "
                f"(val {_sv[_best]:.3f}; all-experts {_sv.get(1.0, float('nan')):.3f}).")
        if _pv:
            _bl = min(_pv, key=_pv.get)
            _txt += f" Among placements, Engram before layer **{_bl}** did best (val {_pv[_bl]:.3f})."
        _won = [r for r in _sv if r < 1.0 and _sv[r] < _sv.get(1.0, -1)]
        if 1.0 in _sv and len(_won) == len(_sv) - 1:
            _txt += (" At this length, before the table has had time to overfit, **every** split beat the "
                     "all-experts baseline: the early lead from the race, measured at one point.")
        _out = mo.md(_txt + """ These are short runs with one seed each, so treat small gaps as noise; the
    paper's U-shaped curve came from models 20× larger trained on 100B tokens. The question it answers
    is still the right one to ask of any sparse model: which kind of capacity is the bottleneck?""")
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · What did Engram memorize?

    DeepSeek's **suppression test**: take a trained Engram model and switch the module off at inference
    ($Y = 0$). If Engram holds local, phrase-like knowledge, the loss should jump most on tokens that a short
    context predicts — the second half of a split word ("Hol" → "mes"), the next word of a stock phrase — and
    least on tokens that need the whole story. Below: per-token loss with Engram on and off, on validation text,
    grouped by whether the token **continues a word** (no leading space) or **starts one**.
    """)
    return


@app.cell
def _(M, RUNS, compressed, corpus, engram_ready, ex, jax, jnp, live, mo, np, texts, tok):
    _src = RUNS / "engram-r75-l1"
    if (_src / "params.npz").exists() and engram_ready and compressed is not None:
        _params, _bias, _cfg, _ = M.load(_src)
        _hasher, _from = M.Hasher(ex, texts, _cfg), "the full engram run"
    elif "engram" in live:
        _params, _bias, _cfg, _hasher = live["engram"]["model"]
        _from = "the live sprint's tiny model (run train.py for the full one)"
    else:
        _params = None
    if _params is None:
        suppress = None
        _out = mo.md("*(Needs the Engram exercises and a trained Engram model.)*")
    else:
        _val = corpus["val"][:16 * 256 + 1].astype(np.int32)
        _x, _y = _val[:-1].reshape(16, 256), _val[1:].reshape(16, 256)
        _hw = jnp.asarray(_hasher.corpus(_val[:-1]).reshape(16, 256, -1))

        @jax.jit
        def _nll(params, bias, x, y, hw, scale):  # arrays as arguments: closing over params would bake them in
            _lg = M.forward(ex, params, x, hw, bias, _cfg, engram_scale=scale)[0]
            return -jnp.take_along_axis(jax.nn.log_softmax(_lg.astype(jnp.float32)), y[..., None], -1)[..., 0]

        _args = (_params, _bias, jnp.asarray(_x), jnp.asarray(_y), _hw)
        _on, _off = np.asarray(_nll(*_args, 1.0)), np.asarray(_nll(*_args, 0.0))
        _cont = np.array([[texts[int(t)][:1].isalnum() for t in row] for row in _y])
        _d = _off - _on
        suppress = {"from": _from, "on": float(_on.mean()), "off": float(_off.mean()),
                    "cont": float(_d[_cont].mean()), "start": float(_d[~_cont].mean()), "n_cont": int(_cont.sum())}
        _flat = np.argsort(_d.ravel())[::-1]
        _seen, _ex = set(), []
        for _i in _flat:
            _b, _t = divmod(int(_i), 256)
            _key = int(_x[_b, _t])                                     # one row per last token: show variety
            if _key in _seen:
                continue
            _w = tok.decode([int(v) for v in _x[_b, max(_t - 4, 0):_t + 1]]) + " ⟶ " + tok.decode([int(_y[_b, _t])])
            _w = repr(_w)[1:-1].replace("|", "\\|").replace("`", "'")
            _seen.add(_key)
            _ex.append(f"| `{_w}` | +{_d.ravel()[_i]:.2f} |")
            if len(_ex) == 8:
                break
        _rows = "\n".join(_ex)
        _out = mo.md(f"""
    From {_from}, on {_x.size:,} validation tokens: mean loss **{suppress['on']:.3f}** with Engram,
    **{suppress['off']:.3f}** without (+{suppress['off'] - suppress['on']:.3f}). The jump on tokens that
    **continue a word** ({suppress['n_cont']:,} of them): **+{suppress['cont']:.3f}**; on tokens that
    **start a word**: **+{suppress['start']:.3f}**. The biggest single jumps are huge: +{_d.max():.0f} nats
    means the "off" model gives the right token about e^−{_d.max():.0f} of the probability the "on" model did.

    | where switching Engram off hurt most, one per context token (context ⟶ target) | loss increase |
    |---|---|
    {_rows}
    """)
    _out
    return (suppress,)


@app.cell
def _(mo, suppress):
    if suppress is None:
        _out = mo.md("")
    else:
        _ratio = suppress["cont"] / max(suppress["start"], 1e-9)
        _lead = (f"Word continuations lose {_ratio:.1f}× as much as word starts when Engram is switched off — the "
                 "model has moved some of its \"finish this word / this phrase\" work into the table, as the paper "
                 "found for named entities and stock phrases" if _ratio > 1.2 else
                 f"Word continuations and word starts lose about the same ({_ratio:.1f}×) when Engram is switched "
                 "off: at this scale the table didn't specialize in finishing words, unlike the paper's named "
                 "entities and stock phrases")
        _out = mo.md(f"""
    {_lead} (switching Engram off cost the paper's 27B model most of its trivia knowledge but little reading
    comprehension). Note what the test can't separate: the rest of the network trained *with* Engram
    and never learned to work without it, so the "off" model is a damaged model, not a model without Engram.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 · Lookups you can prefetch

    An expert is chosen by the router *during* the forward pass: you can't know which weights a token needs
    until you've computed its hidden state. Engram's rows depend only on the token ids — known before the
    forward pass starts. So the table doesn't have to live in GPU memory at all: the host can gather the next
    batch's rows from a table in ordinary RAM and copy them over while the GPU computes the current batch. DeepSeek
    kept a **100-billion-parameter** table in host memory this way and lost under 3% throughput (H800s,
    PCIe). For a single 16 GB card that's the interesting part: memory capacity in RAM, not VRAM.

    Below, the same Engram model three ways, forward passes only: the table on the GPU; the table in a NumPy
    array on the host, gathered and copied **before** each batch (no overlap); and the same with the next batch
    gathered and sent **while** the current one runs — JAX dispatches work asynchronously, so [`device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html#jax.device_put)
    of batch $i+1$ overlaps the GPU's work on batch $i$.
    """)
    return


@app.cell
def _(M, compressed, engram_ready, ex, jax, jnp, mo, np, texts, time):
    if not (engram_ready and compressed is not None):
        prefetch = None
        _out = mo.md("*(Needs the Engram exercises.)*")
    else:
        _cfg = M.variant(M.BASE, "engram", rho=0.75)
        _params = M.init_model(jax.random.key(0), 8192, _cfg)
        _hasher = M.Hasher(ex, texts, _cfg)
        _table_host = np.asarray(_params["engram"]["table"])
        _rng = np.random.default_rng(0)
        _batches = [_rng.integers(0, 8192, (32, 256)).astype(np.int32) for _ in range(12)]
        _hashes = [_hasher(b) for b in _batches]
        _bias = M.init_biases(_cfg)

        @jax.jit
        def _fwd_ids(params, x, hw):
            return M.forward(ex, params, x, hw, _bias, _cfg)[0][:, -1].sum()

        @jax.jit
        def _fwd_rows(params, x, rows):
            return M.forward(ex, params, x, None, _bias, _cfg, engram_rows=rows)[0][:, -1].sum()

        _dev = [(jnp.asarray(b), jnp.asarray(h)) for b, h in zip(_batches, _hashes, strict=True)]
        _fwd_ids(_params, *_dev[0]).block_until_ready()
        _fwd_rows(_params, _dev[0][0], jnp.asarray(M.gather_rows(_table_host, _hashes[0], _cfg))).block_until_ready()

        def _timed(fn):
            _t0 = time.perf_counter()
            fn()
            return (time.perf_counter() - _t0) / len(_batches) * 1e3

        def _on_gpu():
            for x, h in _dev:
                _r = _fwd_ids(_params, x, h)
            _r.block_until_ready()

        def _host_serial():
            for (x, _), h in zip(_dev, _hashes, strict=True):
                _rows = jax.device_put(M.gather_rows(_table_host, h, _cfg))
                _fwd_rows(_params, x, _rows).block_until_ready()

        def _host_prefetch():
            _nxt = jax.device_put(M.gather_rows(_table_host, _hashes[0], _cfg))
            for i, (x, _) in enumerate(_dev):
                _cur = _nxt
                _r = _fwd_rows(_params, x, _cur)                               # returns at once (async dispatch)
                if i + 1 < len(_dev):
                    _nxt = jax.device_put(M.gather_rows(_table_host, _hashes[i + 1], _cfg))  # overlaps the GPU
            _r.block_until_ready()

        prefetch = {k: min(_timed(f) for _ in range(3)) for k, f in
                    (("table on the GPU", _on_gpu), ("host table, no overlap", _host_serial),
                     ("host table, prefetched", _host_prefetch))}
        _mem = {}
        try:
            _mem["kinds"] = [m.kind for m in jax.devices()[0].addressable_memories()]
            _pin = jax.device_put(_table_host[:1024], jax.sharding.SingleDeviceSharding(jax.devices()[0],
                                                                                         memory_kind="pinned_host"))
            _mem["pinned_host"] = f"works ({_pin.sharding.memory_kind})"
        except Exception as _e:  # noqa: BLE001 — report whatever the backend says
            _mem["pinned_host"] = f"not available here: {type(_e).__name__}: {str(_e)[:120]}"
        _base = prefetch["table on the GPU"]
        _rows = "\n".join(f"| {k} | {v:.1f} ms | {v / _base:.2f}× |" for k, v in prefetch.items())
        _out = mo.md(f"""
    Forward pass, batch 32 × 256, {_params['engram']['table'].size / 1e6:.1f}M-parameter table
    ({_params['engram']['table'].nbytes / 1e6:.0f} MB):

    | where the table lives | per batch | vs GPU table |
    |---|---|---|
    {_rows}

    Memory kinds this GPU exposes to JAX: `{_mem.get('kinds')}`; placing an array in pinned host memory
    (`memory_kind="pinned_host"`, which lets the GPU copy from RAM without a staging copy):
    {_mem['pinned_host']}.
    """)
    _out
    return (prefetch,)


@app.cell
def _(mo, prefetch):
    if prefetch is None:
        _out = mo.md("")
    else:
        _gain = prefetch["host table, no overlap"] - prefetch["host table, prefetched"]
        _hid = (f"Prefetching hid **{_gain:.1f} ms** of the host's gather-and-copy per batch behind the GPU's work."
                if _gain > 0.5 else "Prefetching made no measurable difference here.")
        _out = mo.md(f"""
    {_hid} Whatever
    remains above the GPU-table time is host work that didn't fit in the shadow — NumPy gathering ~65k random
    rows on one CPU core, and the copy over PCIe. The paper's numbers come from a production inference engine
    with the gather on many CPU threads and the first transformer block as the shadow; the principle is the
    same: deterministic addresses can be fetched early.
    """)
    _out
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "a": "LatentMoE-acc moves fewer bytes per token than a standard MoE layer",
            "b": "LatentMoE-acc moves the same bytes per token but picks 4× more, 4× smaller experts",
            "c": "LatentMoE runs the router on the latent, so routing is 4× cheaper",
        },
        label="**Check yourself:** at α = 4, which is true?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("That's the *eff* variant. *acc* spends the savings on k' = 4k picks, and "
                              "k'ℓ = kd: the same bytes."), kind="warn"),
        "b": mo.callout(mo.md("Right — same bytes, same FLOPs, many more combinations of experts."),
                        kind="success"),
        "c": mo.callout(mo.md("The router reads the full $x$ (and shared experts stay at width $d$); only the "
                              "routed experts work in the latent."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(40))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * DeepSeek-AI, [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram) — the paper and
      `engram_demo_v1.py`, whose hashing and tokenizer compression this module reproduces bit for bit.
    * NVIDIA, [Nemotron 3: Efficient and Open Intelligence](https://arxiv.org/abs/2512.20856) — LatentMoE in a
      hybrid Mamba/attention model family, with the comparison figure.
    * Kilian Weinberger et al., [Feature Hashing for Large Scale Multitask Learning](https://arxiv.org/abs/0902.2206)
      — the hashing trick Engram's tables rest on.
    * JAX docs: [`jax.lax.ragged_dot`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html) — the
      grouped matmul a sparse (dispatching) MoE layer needs on one device.

    **Next:** Part 12 continues with linear attention and hybrid models (planned) — the Mamba/attention mixes
    that Nemotron 3 pairs with LatentMoE.
    """)
    return


if __name__ == "__main__":
    app.run()
