import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="48 · Tracing a circuit")


@app.cell
def _():
    import marimo as mo
    import html
    import importlib.util
    import textwrap
    import time

    import jax
    import jax.numpy as jnp
    import jaxprobe as jp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import llama, rosetta

    ex = lj.load_exercises(mo.notebook_dir())
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")  # re-runs skip XLA compilation
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    _spec = importlib.util.spec_from_file_location("m48_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    params, cfg, tok = llama.load(llama.SMOLLM_135M, dtype="float32")  # float32: gradients and graphs are exact
    return M, cfg, ex, html, jax, jnp, jp, lj, mo, np, params, plt, rosetta, textwrap, time, tok


@app.cell
def _(M, np):
    # results written by train.py (None until that stage has run)
    patch_r = M.read_json(M.RUNS / "patch.json")
    _f = M.RUNS / "patch.npz"
    patch_a = dict(np.load(_f)) if _f.exists() else None
    tc_r = M.read_json(M.RUNS / "transcoders.json")
    tc_hist = M.read_json(M.RUNS / "transcoders_hist.json")
    graph_r = M.read_json(M.RUNS / "graph.json")
    return graph_r, patch_a, patch_r, tc_hist, tc_r


@app.cell
def _(mo):
    mo.md(r"""
    # 48 · Tracing a circuit: patching, transcoders and attribution graphs (2022–2025)

    **Take first:** Module 47 (the residual stream of SmolLM2-135M, sparse autoencoders, `jaxprobe`), and
    Module 37 (attention heads as readable parts).

    **Goal:** Module 47 read *what* a model represents. This module asks *how* it gets an answer: which parts,
    wired in what order, turn a prompt into a prediction. You'll find the parts of SmolLM2-135M that solve a
    small reasoning task by breaking and repairing its activations, find the same parts again from one backward
    pass, and then build an **attribution graph**: a map, for one prompt, of which features drive which others
    all the way to the output, with what the map can't explain kept in plain view.

    **You will:** measure a behaviour with a logit difference · locate it with activation patching, layer by
    layer and head by head · get the same answer 10× faster with attribution patching, and see where that
    shortcut breaks · test whether the heads you found are enough · replace all 30 MLPs with sparse transcoders
    · build, prune and read an attribution graph · check its claims by intervening on the real model.
    """)
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **Where this came from**

    **Causal tracing** (Meng et al., 2022; Module 46) corrupted a prompt and restored one layer at a time to
    find where facts live. **Wang et al.** (2022) used the same move, **activation patching**, head by head on
    GPT-2 small and mapped a 26-head circuit for the **indirect-object** task you'll study here. **Conmy et
    al.** (2023) automated the search (ACDC); **Nanda** (2023) and **Kramár et al.** (2024) replaced thousands
    of patching runs with one gradient (**attribution patching**). Heads and neurons are not always the right
    parts, so **Marks et al.** (2024) traced circuits of sparse-autoencoder features with error terms, and
    **Dunefsky et al.** (2024) introduced **transcoders**, sparse stand-ins for MLPs. In 2025 Anthropic
    (**Ameisen et al.**, **Lindsey et al.**) combined these into **attribution graphs** on Claude 3.5 Haiku and
    open-sourced the method as *circuit-tracer*.

    *Timeline: Module 00b.*
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · A behaviour to explain

    > When Mary and John went to the store, John gave a drink to → **Mary**

    To finish this sentence the model has to notice which name appeared twice (John, the *subject*, **S**) and
    produce the other one (Mary, the *indirect object*, **IO**). Small models do this well, it needs
    information to move between positions, and Wang et al. mapped how GPT-2 small does it, so there is
    something to compare against. It's called **indirect object identification** (IOI).

    The prompts below are 96 sentences from one template with random single-token names, places and objects,
    half in the order *A B … B* and half *B A … B*. Each comes with a **flipped** twin: same names, same
    order, but the *other* name repeats, so the right answer flips. Everything else stays fixed, which is the
    point: any difference between the two runs is about who repeated.

    We score a run by the **logit difference**, logit(IO) − logit(S) at the last position, averaged over the
    prompts. It is linear in the final residual stream (probabilities are not), so contributions from
    different parts add up.

    **Your turn:** `logit_diff(logits, io, s)`.
    """)
    return


@app.cell
def _(M, cfg, ex, jax, jnp, lj, mo, params, tok):
    ioi = M.ioi(tok, 96)
    fwd = jax.jit(M.forward(params, cfg))
    logits_clean, logits_flip = fwd(ioi["clean"]), fwd(ioi["flip"])
    _a = lj.attempt(ex.logit_diff, logits_clean, ioi["io"], ioi["s"])
    _b = lj.attempt(ex.logit_diff, logits_flip, ioi["io"], ioi["s"])
    _last = logits_clean[:, -1]
    _right = float((jnp.take_along_axis(_last, ioi["io"][:, None], 1) >
                    jnp.take_along_axis(_last, ioi["s"][:, None], 1)).mean())
    _ex = "\n".join(f"* {t}" for t in ioi["text"][:3])
    if _a.ok and _b.ok:
        ld_clean, ld_flip = float(_a.value), float(_b.value)
        _out = mo.md(f"""
    {_ex}

    **`logit_diff`** ✅ clean **{ld_clean:+.2f}**, flipped **{ld_flip:+.2f}** (measured against the *clean*
    answer, so the flipped prompts prefer the other name by about as much). The model ranks IO above S on
    **{_right:.0%}** of the clean prompts.
    """)
    else:
        ld_clean = ld_flip = None
        _out = mo.vstack([_a.view(mo), _b.view(mo)])
    _out
    return fwd, ioi, ld_clean, ld_flip, logits_clean, logits_flip


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "logit difference", solves="logit_diff")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Activation patching: where does the answer live?

    Run the model on the flipped prompt, but at one place (one layer, one position) paste in the activation
    from the clean run. If the logit difference swings back toward the clean value, that activation carried
    the information that decides the answer. Normalise so the numbers are comparable:

    $$\text{effect} = \frac{m_{\text{patched}} - m_{\text{flipped}}}{m_{\text{clean}} - m_{\text{flipped}}}
    \qquad 0 = \text{nothing restored},\ 1 = \text{fully restored}$$

    The tool is `jaxprobe` from Module 47: `learnjax/llama.py` marks the residual stream with taps, and a
    *patch* replaces a tap's value during the run:

    ```python
    put = lambda x: x.at[:, pos].set(clean_resid[layer][:, pos])
    logits, _ = jp.run(forward, flipped, patch={"layers.resid": jp.at(layer - 1, put)})
    ```

    **Your turn:** `patching_effect(patched, clean, corrupt)`.
    """)
    return


@app.cell
def _(M, cfg, ex, ioi, jax, jnp, jp, ld_clean, ld_flip, lj, mo, params):
    _f = M.forward(params, cfg)
    _, _cap = jax.jit(lambda t: jp.run(_f, t, capture="layers.resid"))(ioi["clean"])
    clean_resid = _cap["layers.resid"]

    @jax.jit
    def _patched(layer, pos):
        _put = lambda x: x.at[:, pos].set(clean_resid[layer - 1][:, pos])
        lg, _ = jp.run(_f, ioi["flip"], patch={"layers.resid": lambda x: jnp.where(jp.index() + 1 == layer,
                                                                                 _put(x), x)})
        return lg

    _rows = []
    _ok = ld_clean is not None
    for _layer, _pos, _name in [(10, M.POS["S2"], "S2 (the repeated name)"), (10, M.POS["END"], "END (last token)"),
                                (26, M.POS["S2"], "S2"), (26, M.POS["END"], "END")]:
        _m = lj.attempt(ex.logit_diff, _patched(_layer, _pos), ioi["io"], ioi["s"])
        if _ok and _m.ok:
            _e = lj.attempt(ex.patching_effect, float(_m.value), ld_clean, ld_flip)
            if _e.ok:
                _rows.append(f"| {_layer} | {_name} | {float(_m.value):+.2f} | **{float(_e.value):+.2f}** |")
            else:
                _ok = False
    if _ok and _rows:
        _out = mo.md("**`patching_effect`** ✅\n\n| layer | position patched | logit diff | effect |\n|---|---|---|---|\n"
                     + "\n".join(_rows))
    else:
        _out = mo.md("Finish `logit_diff` and `patching_effect` to see four patches here.")
    _out
    return


@app.cell
def _(mo, np, patch_a, patch_r, plt):
    if patch_r is None:
        _out = mo.md("*The full sweep (every layer × every position, and every head) comes from "
                     "`containers/run.sh python modules/48-tracing-circuits/train.py --patch` (~3 min).*")
    else:
        _R = patch_a["resid"]
        _fig, _ax = plt.subplots(figsize=(8.5, 4.6))
        _im = _ax.imshow(_R.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1, origin="upper")
        _ax.set_yticks(range(len(patch_r["tokens"])), [repr(t) for t in patch_r["tokens"]], fontsize=8)
        _ax.set_xlabel("residual stream after layer (0 = embedding)")
        _ax.set_title("Patching one clean residual state into the flipped run")
        _fig.colorbar(_im, ax=_ax, label="effect")
        _fig.tight_layout()
        # where is the information? how long S2 holds all of it, and when END takes over
        _s2, _end = _R[:, 9], _R[:, -1]
        _full = int(np.max(np.flatnonzero(_s2 > 0.9)))
        _half = int(np.min(np.flatnonzero(_end > 0.5)))
        _done = int(np.min(np.flatnonzero(_end > 0.8)))
        _out = mo.vstack([_fig, mo.md(f"""
    Only two positions matter. Through layer {_full}, patching the second mention of the subject (**S2**) alone
    restores the whole answer (effect ≥ 0.9), and patching any other token does nothing. Then it moves: by layer
    {_half} more than half of it sits at the last token (END {_end[_half]:.2f}, S2 {_s2[_half]:.2f}), and by layer
    {_done} END holds {_end[_done]:.2f}, where the prediction is made. Something in layers {_full + 1}–{_done} reads S2
    from the last position: attention heads, the only parts that move information between positions.
    """)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Which heads?

    Patch one head at a time: replace head *h* of layer *ℓ*'s output (its `z`, before the output projection,
    at every position) with the clean run's. 30 layers × 9 heads = 270 runs. Heads are named *layer.head* and
    counted from 0, so head 20.0 sits in the 21st block and writes into the stream "after layer 21" of the plot
    above.
    """)
    return


@app.cell
def _(mo, np, patch_a, patch_r, plt):
    if patch_r is None:
        _out = mo.md("*Needs `train.py --patch`.*")
    else:
        _Hp = patch_a["heads"]
        _fig, _ax = plt.subplots(figsize=(8.5, 3.2))
        _lim = float(np.abs(_Hp).max())
        _im = _ax.imshow(_Hp.T, aspect="auto", cmap="RdBu", vmin=-_lim, vmax=_lim)
        _ax.set_xlabel("layer")
        _ax.set_ylabel("head")
        _ax.set_title("Patching one head's clean output into the flipped run")
        _fig.colorbar(_im, ax=_ax, label="effect")
        _fig.tight_layout()
        _rows = []
        for _r in patch_r["roles"][:8]:
            _att = {"IO": _r["attn_io"], "S1": _r["attn_s1"], "S2": _r["attn_s2"], "first": _r["attn_first"]}
            _top = max(_att, key=_att.get)
            _rows.append(f"| {_r['layer']}.{_r['head']} | {_r['patch']:+.2f} | {_r['attn_io']:.2f} | {_r['attn_s1']:.2f} "
                         f"| {_r['attn_s2']:.2f} | {_r['attn_first']:.2f} | {_r['dla']:+.2f} |")
        _out = mo.vstack([_fig, mo.md(
            "The strongest heads, with where they look *from the last token* (attention weight on IO, on the first "
            "mention of S, on S2, and on the first token, which works as a resting place) and what they write "
            "straight toward the answer (**direct logit attribution**: the head's output at the last position, "
            "through the final norm, dotted with the unembedding of IO minus that of S):\n\n"
            "| head | patch effect | → IO | → S1 | → S2 | → first | direct IO − S |\n|---|---|---|---|---|---|---|\n"
            + "\n".join(_rows))])
    _out
    return


@app.cell
def _(mo, patch_r):
    if patch_r is None:
        _out = mo.md("")
    else:
        _R = patch_r["roles"]
        _s2 = [r for r in _R if r["attn_s2"] > 0.4 and r["patch"] > 0]
        _mov = [r for r in _R if r["attn_io"] > 0.4 and r["dla"] > 0]
        _neg = [r for r in _R if r["attn_io"] > 0.4 and r["dla"] < 0]
        _h = lambda rs: ", ".join(f"{r['layer']}.{r['head']}" for r in rs) or "none"
        _out = mo.md(f"""
    Three kinds of head show up, the same roles Wang et al. found in GPT-2 small:

    * **S2 readers** ({_h(_s2)}): from the last token they look at the repeated name. They carry "this name was
      the subject" to where the answer is made. (In GPT-2 small these are the *S-inhibition* heads.)
    * **Name movers** ({_h(_mov)}): they look at the other name, IO, and write it toward the output (positive
      direct effect on IO − S).
    * **Negative name movers** ({_h(_neg)}): they also look at IO but push it *down*. Restoring them hurts the
      clean answer, hence the negative patch effect. McDougall et al. (2023) explain them as *copy
      suppression*: a check against over-confidently repeating a name from the context.

    The rest of the top heads write to the answer without attending to either name; they act on what earlier
    heads put at the last position.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Attribution patching: every patch from one backward pass

    Each patching run costs a forward pass, and a real search needs thousands. **Attribution patching** asks a
    cheaper question: if the metric were linear near the flipped run, what would each patch do? Then the answer
    is a dot product with the gradient,

    $$\Delta m \approx \nabla_a m\,\big\rvert_{\text{flipped}} \cdot (a_{\text{clean}} - a_{\text{flipped}})$$

    and one backward pass gives the gradient for *every* activation at once. With jaxprobe, add a zero offset at
    the tap and differentiate with respect to the offset:

    ```python
    def metric(dz):
        lg, _ = jp.run(forward, flipped, patch={"layers.z": lambda z: z + dz[jp.index()]})
        return logit_diff(lg, io, s)
    grad = jax.grad(metric)(jnp.zeros_like(z_flipped))       # (layers, batch, pos, heads, head_dim)
    ```

    **Your turn:** `attribution_patch(grad, clean, corrupt)`.
    """)
    return


@app.cell
def _(M, cfg, ex, ioi, jax, jnp, jp, ld_clean, ld_flip, lj, mo, np, params, patch_a, patch_r, plt, time):
    _f = M.forward(params, cfg)
    _run = jax.jit(lambda t: jp.run(_f, t, capture="layers.z")[1]["layers.z"])
    _zc, _zf = _run(ioi["clean"]), _run(ioi["flip"])

    def _metric(dz):
        lg, _ = jp.run(_f, ioi["flip"], patch={"layers.z": lambda z: z + dz[jp.index()]})
        last = lg[:, -1]
        return (jnp.take_along_axis(last, ioi["io"][:, None], 1) - jnp.take_along_axis(last, ioi["s"][:, None], 1)).mean()

    _g = jax.jit(jax.grad(_metric))
    jax.block_until_ready(_g(jnp.zeros_like(_zf)))
    _t0 = time.time()
    _grad = jax.block_until_ready(_g(jnp.zeros_like(_zf)))
    t_grad = time.time() - _t0
    _a = lj.attempt(ex.attribution_patch, _grad, _zc, _zf)
    atp_heads = None
    if _a.ok and ld_clean is not None:
        atp_heads = np.asarray(jnp.asarray(_a.value).sum((1, 2))) / (ld_clean - ld_flip)  # (layers, heads)
    if atp_heads is None:
        _out = _a.view(mo) if not _a.ok else mo.md("Finish `logit_diff` first.")
    elif patch_a is None:
        _out = mo.md(f"**`attribution_patch`** ✅ all {atp_heads.size} heads in {t_grad * 1e3:.0f} ms. Run "
                     "`train.py --patch` to compare with real patching.")
    else:
        _Hp = patch_a["heads"]
        _r = float(np.corrcoef(_Hp.ravel(), atp_heads.ravel())[0, 1])
        _fig, _ax = plt.subplots(figsize=(4.6, 4.2))
        _ax.scatter(_Hp.ravel(), atp_heads.ravel(), s=10, alpha=0.6)
        _lim = [min(_Hp.min(), atp_heads.min()) - 0.05, max(_Hp.max(), atp_heads.max()) + 0.05]
        _ax.plot(_lim, _lim, "k:", lw=1)
        _worst = int(np.argmax(np.abs(_Hp - atp_heads).ravel()))
        _wl, _wh = divmod(_worst, _Hp.shape[1])
        _ax.annotate(f"{_wl}.{_wh}", (_Hp[_wl, _wh], atp_heads[_wl, _wh]), fontsize=9)
        _ax.set_xlabel("activation patching (270 forward passes)")
        _ax.set_ylabel("attribution patching (1 backward pass)")
        _fig.tight_layout()
        _out = mo.hstack([_fig, mo.md(f"""
    **`attribution_patch`** ✅ one gradient: **{t_grad * 1e3:.0f} ms** for all {atp_heads.size} heads (the
    one-by-one sweep of {patch_r['n_heads']} patches took {patch_r['t_heads']:.0f} s, about
    {patch_r['t_heads'] / t_grad:,.0f}× longer).

    Correlation with real patching: **{_r:.2f}**. The linear estimate gets the ranking mostly right, but look
    at head **{_wl}.{_wh}**: patching it restores **{_Hp[_wl, _wh]:+.2f}**, the gradient says
    **{atp_heads[_wl, _wh]:+.2f}**. A gradient is only exact for small changes, and swapping which name is
    repeated is not small. That head's effect runs through later heads, whose softmax attention and norms bend
    the response far from a straight line. Kramár et al. (2024) trace most such misses to saturated attention.
    Use attribution patching to rank candidates, then confirm the top ones with real patches.
    """)])
    _out
    return (atp_heads,)


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "attribution patching", solves="attribution_patch")
    return


@app.cell
def _(mo, patch_r, plt):
    if patch_r is None:
        _out = mo.md("## 5 · Are these heads the circuit?\n\n*Needs `train.py --patch`.*")
    else:
        _c = patch_r["curve"]
        _fig, _ax = plt.subplots(figsize=(6, 3.2))
        _ax.plot([r["k"] for r in _c], [r["top_pos"] for r in _c], "o-", label="top-k heads that help")
        _ax.plot([r["k"] for r in _c], [r["top_abs"] for r in _c], "s--", label="top-k by |effect| (incl. negative)")
        _ax.set_xscale("log")
        _ax.axhline(1, color="k", lw=0.8, ls=":")
        _ax.set_xlabel("heads given their clean output (k)")
        _ax.set_ylabel("effect on the flipped run")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _by = {r["k"]: r for r in _c}
        _k5, _k10 = _by.get(5), _by.get(10)
        _all = _c[-1]
        _out = mo.vstack([mo.md(f"""
    ## 5 · Are these heads the circuit?

    A list of important heads isn't yet a mechanism. A sharper test is **sufficiency**: start from the flipped
    run and give *only* the top *k* heads their clean output, everything else stays flipped. If those heads are
    how the model decides, they alone should bring the answer back.
    """), _fig, mo.md(f"""
    The top two heads alone restore **{_by[2]['top_pos']:.0%}** of the clean logit difference, five restore
    {_k5['top_pos']:.0%}; all {_all['k']} heads together give {_all['top_pos']:.0%}. Ranking by size instead
    (dashed) brings in the negative heads, and the total overshoots to {_k10['top_abs']:.0%} at ten heads: given
    their clean outputs in an otherwise flipped run, the negative heads suppress the *flipped* answer's name, which
    helps the clean one. So the decision runs through a handful of late heads reading what earlier layers stored
    at S2, the same shape as the circuit in GPT-2 small. What this doesn't say is *how* S2's position came to hold "I am the repeated
    name", or what the MLPs contribute; patching heads can't see inside them. That needs different parts.
    """)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Transcoders: MLPs you can read

    Heads have obvious inputs and outputs; MLP neurons don't mean much one at a time (superposition, Module
    37). Module 47's sparse autoencoders split the residual stream into readable features; a **transcoder** does
    the same for an MLP, but instead of reconstructing its input it *predicts the MLP's output from its input*:

    $$z = \mathrm{TopK}\big(W_{\text{enc}}\, x_{\text{in}} + b_{\text{enc}}\big), \qquad
    \hat y = W_{\text{dec}}\, z + b_{\text{dec}} \approx \mathrm{MLP}(x_{\text{in}})$$

    So each MLP becomes a layer of sparse features, each with a readable decoder direction. Unlike an SAE on
    the stream, everything a transcoder feature writes goes straight into the stream at its layer.

    `train.py --transcoders` trains one for each of SmolLM2-135M's 30 MLPs, 8,192 features each, 32 active per
    token, together in one pass over 15M tokens of Wikipedia (WikiText-103). It runs the matrix multiplies in
    bfloat16, because on this GPU that's much faster for these shapes:
    """)
    return


@app.cell
def _(jax, jnp, mo, time):
    def _bench(dt):
        x = jax.random.normal(jax.random.key(0), (2048, 576)).astype(dt)
        W = jax.random.normal(jax.random.key(1), (576, 8192)).astype(dt)
        f = jax.jit(lambda x, W: jax.nn.relu(x @ W) @ W.T)
        jax.block_until_ready(f(x, W))
        t0 = time.time()
        for _ in range(20):
            r = f(x, W)
        jax.block_until_ready(r)
        return (time.time() - t0) / 20

    _t32, _t16 = _bench(jnp.float32), _bench(jnp.bfloat16)
    _fl = 2 * 2 * 2048 * 576 * 8192
    mo.md(f"One transcoder's encode + decode for 2,048 tokens: float32 **{_t32 * 1e3:.1f} ms** "
          f"({_fl / _t32 / 1e12:.1f} TFLOP/s), bfloat16 **{_t16 * 1e3:.1f} ms** ({_fl / _t16 / 1e12:.1f} TFLOP/s), "
          f"**{_t32 / _t16:.1f}×** faster. The weights and the optimiser stay in float32; only the products are "
          "rounded.")
    return


@app.cell
def _(mo, np, plt, tc_r):
    if tc_r is None:
        _out = mo.md("*Train them with `containers/run.sh python modules/48-tracing-circuits/train.py "
                     "--transcoders` (~70 min); the graphs below need them.*")
    else:
        _fvu = np.array(tc_r["fvu"])
        _one = np.array(tc_r["ce_one"]) - tc_r["ce_clean"]
        _zero = np.array(tc_r["ce_zero_one"]) - tc_r["ce_clean"]
        _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.2))
        _a1.bar(range(len(_fvu)), _fvu)
        _a1.set_xlabel("layer")
        _a1.set_ylabel("FVU of the MLP output")
        _a1.set_title("What each transcoder misses")
        _a2.plot(_zero, "o-", label="MLP output set to 0")
        _a2.plot(_one, "s-", label="MLP replaced by its transcoder")
        _a2.set_xlabel("layer replaced (one at a time)")
        _a2.set_ylabel("extra loss (nats)")
        _a2.set_yscale("log")
        _a2.legend(fontsize=8)
        _fig.tight_layout()
        _rec = 1 - _one / np.maximum(_zero, 1e-9)
        _out = mo.vstack([_fig, mo.md(f"""
    Trained on {tc_r['tokens'] / 1e6:.0f}M tokens in {tc_r['seconds'] / 60:.0f} minutes. On held-out Wikipedia
    they leave **{np.median(_fvu):.0%}** of an MLP's output variance unexplained (median over layers; best layer
    {int(np.argmin(_fvu))} at {_fvu.min():.0%}, worst {int(np.argmax(_fvu))} at {_fvu.max():.0%}). Swapping in one
    transcoder at a time costs on average **{_one.mean():.3f}** nats of next-token loss, against
    {_zero.mean():.3f} for deleting that MLP: they keep a median **{np.median(_rec):.0%}** of what each MLP does.
    Swapping all 30 at once compounds the errors: loss goes from {tc_r['ce_clean']:.2f} to
    **{tc_r['ce_all']:.2f}**. That gap is why the graph below keeps an **error node** at every layer and
    position: the transcoders explain much of each MLP, not all of it, and the graph has to say so.
    """)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Attribution graphs

    Now build the map for one prompt (Ameisen et al. 2025, with per-layer transcoders):

    1. **Replace** every MLP by its transcoder, and add back what it gets wrong at each layer and position as a
       fixed **error** term. Nothing changes yet: the output is identical.
    2. **Freeze** the attention patterns and the norms' scale factors at their values on this prompt. What's
       left between the token embeddings, the feature activations, the errors and the logits is *linear*:
       attention just mixes positions with fixed weights, and every MLP has become "decoder × activations".
    3. In a linear model, a source's effect on a target is exact and additive: the derivative of the target's
       pre-activation (or logit) with respect to the source, times the source's value. Those are the
       **edges**. One backward pass per target node; `jaxprobe.trace` batches them with [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap) over one
       [`jax.vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html#jax.vjp).
    4. **Prune.** Normalise each node's incoming edges to sum to 1, then compute every node's **total
       influence** on the output, through all paths: $x = w\hat A + w\hat A^2 + \dots$ for logit weights *w*.
       Keep the features carrying 80% of it.

    **Your turn:** `edge_weights(jac, values)`, `normalize_rows(A)` and `total_influence(A_norm, w)`.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo, np):
    # a 4-node chain with a shortcut: 3 → 2 → 1 → 0 (the output), plus 3 → 0 directly
    _J = jnp.array([[0, 2.0, 0, 1.0], [0, 0, 3.0, 0], [0, 0, 0, -1.0], [0, 0, 0, 0]])
    _v = jnp.array([0.0, 1.0, 0.5, 2.0])
    _e = lj.attempt(ex.edge_weights, _J, _v)
    _n = lj.attempt(ex.normalize_rows, _e.value) if _e.ok else _e
    _t = lj.attempt(ex.total_influence, _n.value, jnp.array([1.0, 0, 0, 0])) if _n.ok else _n
    if _t.ok:
        _x = np.asarray(_t.value)
        _out = mo.md(f"""
    **`edge_weights` · `normalize_rows` · `total_influence`** ✅ on a toy graph where node 3 reaches the output
    directly and through 2 → 1: influences on node 0 = {np.round(_x, 3).tolist()}. Node 3 gets {_x[3]:.2f}:
    {float(np.asarray(_n.value)[0, 3]):.2f} through its direct edge and the rest through 2 → 1, because every path
    back from the output ends at it.
    """)
    else:
        _out = mo.vstack([a.view(mo) for a in (_e, _n, _t) if not a.ok][:1])
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "total influence in a graph", solves="total_influence")
    return


@app.cell
def _(M, cfg, ex, jnp, lj, mo, np, params, tc_r, time, tok):
    # a live graph for a short prompt (needs the transcoders)
    if tc_r is None:
        live = None
        _out = mo.md("*Needs `train.py --transcoders`.*")
    else:
        from jaxprobe import trace as _trace
        tcs, tc_meta = M.load_transcoders()
        _t0 = time.time()
        _g = M.attribute(params, cfg, tok, tcs, tc_meta, M.PROMPTS["opposite"][0])
        _secs = time.time() - _t0
        _gp = _trace.prune(_g, max_features=M.VIEW_FEATURES, max_edges=M.VIEW_EDGES)
        _w = np.zeros(len(_g.adj), np.float32)
        _w[_g.slices()["logit"]] = _g.logit_probs / _g.logit_probs.sum()
        _A = lj.attempt(ex.normalize_rows, jnp.asarray(_g.adj))
        _x = lj.attempt(ex.total_influence, _A.value, jnp.asarray(_w)) if _A.ok else _A
        _agree = float(np.abs(np.asarray(_x.value) - _gp.influence).max()) if _x.ok else None
        _rows = np.r_[np.arange(_g.T, _g.T + _g.N), np.arange(len(_g.adj) - len(_g.logit_ids), len(_g.adj))]
        _sum_err = float(np.abs(_g.adj[_rows].sum(1) + _g.bias - _g.values).max())
        live = {"graph": _gp, "raw": _g}
        _out = mo.md(f"""
    **{M.PROMPTS['opposite'][0]!r}** → {', '.join(f"{tok.decode([int(t)])!r} {p:.2f}" for t, p in
                                             zip(_g.logit_ids[:3], _g.logit_probs[:3], strict=False))}

    {_g.N:,} active features ({cfg.n_layers} layers × {_g.T - 1} positions × 32, the sink at position 0 left
    out), {len(_g.adj):,} nodes in all, built in {_secs:.0f} s. Two checks that the replacement is exact:
    the frozen linear model reproduces the real logits to **{_g.scores['logit_check']:.1e}**, and every node's
    incoming edges plus its constant part add up to its value to **{_sum_err:.1e}**.
    {'' if _agree is None else f'Your `total_influence` matches the library to **{_agree:.1e}**.'}

    Influence is spread thin: carrying 80% of it takes **{_gp.scores['n_features_threshold']:,}** features.
    To draw it, keep the {_gp.scores['n_features']} most influential and the {_gp.scores['n_edges']:,} edges
    that matter most among them.
    """)
    _out
    return (live,)


@app.cell
def _(np, plt):
    def draw_graph(nodes, edges, tokens, n_layers, labels=None, title="", max_edges=150, top_labels=8):
        """Kept nodes at (position, layer); edges coloured by sign, width by weight."""
        byid = {n["id"]: n for n in nodes}
        fig, ax = plt.subplots(figsize=(10, 6.2))
        y = lambda n: -1 if n["kind"] == "embed" else n_layers if n["kind"] == "logit" else n["layer"]
        rng = np.random.default_rng(0)
        jit = {n["id"]: (rng.uniform(-0.3, 0.3) if n["kind"] in ("feature", "error") else 0.0) for n in nodes}
        xy = {n["id"]: (n["pos"] + jit[n["id"]], y(n)) for n in nodes}
        es = sorted(edges, key=lambda e: -abs(e["w"]))[:max_edges]
        wmax = max((abs(e["w"]) for e in es), default=1.0)
        for e in es:
            if e["s"] in xy and e["t"] in xy:
                (x0, y0), (x1, y1) = xy[e["s"]], xy[e["t"]]
                ax.plot([x0, x1], [y0, y1], color="tab:blue" if e["w"] > 0 else "tab:red",
                        lw=0.3 + 2.5 * abs(e["w"]) / wmax, alpha=0.35, zorder=1)
        style = {"feature": ("o", "tab:orange"), "error": ("s", "0.6"), "embed": ("^", "tab:green"),
                 "logit": ("D", "tab:purple")}
        infl = np.array([n.get("influence", 0.0) for n in nodes])
        imax = max(infl.max(), 1e-9)
        for kind, (mk, c) in style.items():
            ns = [n for n in nodes if n["kind"] == kind]
            if ns:
                ax.scatter([xy[n["id"]][0] for n in ns], [xy[n["id"]][1] for n in ns], marker=mk, c=c,
                           s=[20 + 220 * (n.get("influence", 0) / imax) ** 0.5 if kind != "logit" else 80 for n in ns],
                           edgecolor="k", lw=0.4, zorder=2, label=kind)
        feats = sorted([n for n in nodes if n["kind"] == "feature"], key=lambda n: -n.get("influence", 0))
        for n in feats[:top_labels]:
            lab = (labels or {}).get(f"{n['layer']}/{n['index']}")
            txt = f"{n['layer']}/{n['index']}" + (f": {'/'.join(lab['promotes'][:2])}" if lab else "")
            ax.annotate(txt, xy[n["id"]], fontsize=7, xytext=(4, 3), textcoords="offset points")
        for n in nodes:
            if n["kind"] == "logit":
                ax.annotate(f"{n['tok']!r} {n['prob']:.2f}", xy[n["id"]], fontsize=8, xytext=(6, 0),
                            textcoords="offset points")
        ax.set_xticks(range(len(tokens)), [repr(t) for t in tokens], rotation=35, fontsize=8, ha="right")
        ax.set_ylabel("layer (embeddings at −1, output at the top)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        return fig
    return (draw_graph,)


@app.cell
def _(cfg, draw_graph, live, mo, np, tok):
    if live is None:
        _out = mo.md("")
    else:
        _g = live["graph"]
        _keep = np.flatnonzero(_g.keep & ((_g.adj != 0).any(0) | (_g.adj != 0).any(1)))
        _nodes = []
        for _i in _keep:
            _n = dict(_g.node_info(int(_i)), id=int(_i), influence=float(_g.influence[_i]))
            if _n["kind"] == "logit":
                _n["tok"] = tok.decode([_n["index"]])
            _nodes.append(_n)
        _sub = _g.adj[np.ix_(_keep, _keep)]
        _t, _s = np.nonzero(_sub)
        _edges = [{"s": int(_keep[a]), "t": int(_keep[b]), "w": float(_sub[b, a])} for b, a in zip(_t, _s, strict=True)]
        _toks = [tok.decode([int(t)]) for t in _g.tokens]
        _out = mo.vstack([draw_graph(_nodes, _edges, _toks, cfg.n_layers, title="'The opposite of hot is' (live)"),
                          mo.md(f"""
    Replacement score **{_g.scores['replacement']:.2f}**: of the influence that reaches the output, that share
    starts at token embeddings and flows through features; the other **{_g.scores['error_share']:.2f}** starts
    at error nodes (grey squares), the part the transcoders don't explain. Completeness, the share of each
    node's inputs that come from features or embeddings rather than errors (weighted by influence):
    **{_g.scores['completeness']:.2f}**. Ameisen et al. found that *cross-layer* transcoders (one feature
    writing to every later layer) leave less unexplained than per-layer ones like these. Read the graph as *a*
    partial explanation with its gaps marked, not as the model.
    """)])
    _out
    return


@app.cell
def _(draw_graph, graph_r, mo):
    if graph_r is None:
        _out = mo.md("*The graphs for all three prompts, with feature labels and interventions, come from "
                     "`train.py --graph` (~5 min).*")
    else:
        _g = graph_r["graphs"]["dallas"]
        _out = mo.vstack([mo.md(f"### {_g['text']!r} → {_g['target']!r}"),
                          draw_graph(_g["nodes"], _g["edges"], _g["tokens"], 30, graph_r["labels"],
                                     title=f"{_g['text']!r}")])
    _out
    return


@app.cell
def _(graph_r, html, mo, textwrap):
    if graph_r is None:
        _out = mo.md("")
    else:
        _g = graph_r["graphs"]["dallas"]
        _sh = _g["layer_share"]
        _rows = []
        for _f in _g["mid"][:12]:
            _lab = graph_r["labels"].get(f"{_f['layer']}/{_f['index']}")
            _ctx = ""
            if _lab and _lab["contexts"]:
                _c = _lab["contexts"][0]
                _ctx = html.escape(f"…{_c['left'][-40:]}[[{_c['tok']}]]{_c['right']}").replace("|", "¦").replace("\n", "⏎")
            _tok = html.escape(repr(_g["tokens"][_f["pos"]]))
            _ctx2 = ""
            if _lab and len(_lab["contexts"]) > 1:
                _c = _lab["contexts"][1]
                _ctx2 = html.escape(f"…{_c['left'][-30:]}[[{_c['tok']}]]{_c['right']}").replace("|", "¦").replace("\n", "⏎")
            _rows.append(f"| {_f['layer']}/{_f['index']} | {_tok} | {_f['influence']:.3f} | {_ctx} | {_ctx2} |")
        _first = [f for f in _g["top"] if f["layer"] == 0][:4]
        _first_s = ", ".join(f"{f['layer']}/{f['index']} on {_g['tokens'][f['pos']]!r}" for f in _first)
        _out = mo.md(textwrap.dedent(f"""
    Where is the influence? **{_sh[0]:.0%}** of the features' total influence sits in layer 0 and
    {_sh[-1]:.0%} in layer {len(_sh) - 1}; every middle layer holds {min(_sh[1:-1]):.0%}–{max(_sh[1:-1]):.0%}. The
    top of the ranking is layer-0 features that recognise the input tokens ({_first_s}): everything downstream
    depends on them. The more interesting ones are in between. The most influential features in layers 1–28,
    with where they fire in the prompt and their two strongest activations on held-out Wikipedia ([[ ]] marks
    the token). In the run this lesson was written from, most of them sit at the last token in layers 25–28 and
    fire hardest on the comma in "Scranton, Pennsylvania" or "Portland, Oregon": *a US state name comes next*.
    Yours were trained from the same seed, but check the table before you trust that sentence:

    | feature | at | influence | top example | second |
    |---|---|---|---|---|
    """) + "\n".join(_rows))
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Testing the graph on the real model

    A graph built from a frozen, linearised model makes claims about the real one, and they can be wrong:
    attention patterns aren't really fixed, and transcoder features aren't really the MLP. So test them. Take
    the graph's most influential features and remove their contributions from the *real* model's MLP outputs
    (`jaxprobe.trace.ablate_features`), with nothing frozen, and watch the probability of the answer. Compare
    with removing the same number of active features picked at random.
    """)
    return


@app.cell
def _(graph_r, mo, plt, textwrap):
    if graph_r is None:
        _out = mo.md("*Needs `train.py --graph`.*")
    else:
        _fig, _axes = plt.subplots(1, len(graph_r["graphs"]), figsize=(11, 3.1), sharey=False)
        _lines = []
        for _ax, (_name, _g) in zip(_axes, graph_r["graphs"].items(), strict=True):
            _r = _g["intervene"]
            _n = [x["n"] for x in _r["ablate"]]
            _ax.plot(_n, [x["p_top"] for x in _r["ablate"]], "o-", label="top by influence")
            _ax.plot(_n, [x["p_mid"] for x in _r["ablate"]], "^-", label="top in layers 1–28")
            _ax.plot(_n, [x["p_random"] for x in _r["ablate"]], "s--", label="random active")
            _ax.axhline(_r["p"], color="k", lw=0.8, ls=":")
            _ax.set_xscale("log")
            _ax.set_title(f"P({_g['target']!r})", fontsize=9)
            _ax.set_xlabel("features removed")
            _last = _r["ablate"][-1]
            _lines.append(f"| {_g['text']!r} | {_r['p']:.2f} | **{_last['p_top']:.3f}** | {_last['p_mid']:.2f} | "
                          f"{_last['p_random']:.2f} |")
        _axes[0].legend(fontsize=7)
        _fig.tight_layout()
        _out = mo.vstack([_fig, mo.md(
            "Removing 30 features (probability of the answer):\n\n| prompt | before | top 30 | top 30 in layers "
            "1–28 | 30 random |\n|---|---|---|---|---|\n" + "\n".join(_lines) + textwrap.dedent("""

    The graph's ranking is causal: its top features matter far more than random active ones, which change
    nothing. But most of that comes from the input-token features of layer 0; take them away and you have
    mostly corrupted the prompt. The middle-layer features the graph highlights do matter, and less: removing
    thirty of them lowers the answer's probability, and doesn't remove it. The model computes the answer along
    many parallel routes, and some of the route runs through attention and through error nodes, which removing
    features can't touch. That is the honest state of the method at this scale: graphs point at real causes, and
    no small set of features is the whole story.
    """))])
    _out
    return


@app.cell
def _(graph_r, mo):
    if graph_r is None:
        _out = mo.md("")
    else:
        _g = graph_r["graphs"]["ioi"]
        _s = _g["scores"]
        _bypos = {}
        for _f in _g["top"][:20]:
            _bypos[_g["tokens"][_f["pos"]]] = _bypos.get(_g["tokens"][_f["pos"]], 0) + _f["influence"]
        _top = sorted(_bypos.items(), key=lambda kv: -kv[1])[:4]
        _out = mo.md(f"""
    And the indirect-object prompt from section 1, **{_g['text']!r}**: its graph keeps {_s['n_features']}
    features with a replacement score of {_s['replacement']:.2f}. Of its 20 most influential features, the
    influence sits at {', '.join(f"{t!r} ({v:.2f})" for t, v in _top)}. Sections 2–5 found the decision made
    by attention from the last token to the names; in the graph, attention is frozen, so those heads show up
    only as edges between positions, not as nodes. The two methods look at different parts and complement
    each other: patching says *which heads*, the graph says *which features they carry*.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Browse the graphs

    `train.py --graph` also writes every graph in the JSON format of Anthropic's open-source **circuit-tracer**
    viewer, the one behind Neuronpedia's attribution-graph pages. To click through them (hover a node to see
    its edges, pin nodes, group them into supernodes):

    ```bash
    PORT=8032 containers/run.sh python tools/graph_viewer.py     # then open http://localhost:8032
    ```

    It downloads the viewer's HTML and JavaScript (MIT) once, at a pinned commit, and serves them with the
    graphs from `runs/48-tracing-circuits/graphs/`.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"Attribution patching gives the same numbers as activation patching, only faster": "a",
                 "An attribution graph's edges are exact for the frozen replacement model on that prompt, and the "
                 "error nodes measure how much of the real model it leaves unexplained": "b",
                 "If the top heads restore most of the logit difference, the MLPs play no part": "c"},
        label="**Check yourself:** which statement do this lesson's measurements support?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    if quiz.value is None:
        _out = mo.md("")
    elif quiz.value == "b":
        _out = mo.callout(mo.md("Right. The replacement plus errors reproduces the logits exactly and each node's "
                                "edges add up to its value; how much flows through error nodes is the honest "
                                "measure of what the transcoders miss."), kind="success")
    elif quiz.value == "a":
        _out = mo.callout(mo.md("Not quite: it's a first-order estimate. It correlated well overall but misjudged a "
                                "head whose attention moves a lot between the two runs (section 4)."), kind="warn")
    else:
        _out = mo.callout(mo.md("Not quite: sufficiency of the heads (section 5) says they carry the decision, "
                                "not that nothing else contributes; the graphs show MLP features on the path, "
                                "e.g. the ones that build what S2's position holds."), kind="warn")
    _out
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(48))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    * [circuit-tracer](https://github.com/safety-research/circuit-tracer) and
      [Neuronpedia's graph pages](https://www.neuronpedia.org/graph/info) — the open-source PyTorch version of
      section 7, with pretrained transcoders for Gemma 2 and Llama 3.2, and a browser to explore graphs.
    * Lindsey et al., [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
      — attribution graphs on Claude 3.5 Haiku: two-hop facts, planning rhymes, arithmetic, and where the method
      fails.
    * Heimersheim & Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
      — noising vs denoising, choosing corruptions, and what a patching result does and doesn't show.
    * [jaxprobe](https://github.com/joebarbere/jaxprobe) — the library under this module: `jaxprobe.trace` is
      sections 6–8 in about 400 lines of JAX.

    **Next:** Module 49, continual learning: forgetting, consolidation and sleep.
    """)
    return


if __name__ == "__main__":
    app.run()
