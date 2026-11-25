import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="47 · Reading activations")


@app.cell
def _():
    import marimo as mo
    import html
    import importlib.util
    import textwrap

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import interp, llama, rosetta

    ex = lj.load_exercises(mo.notebook_dir())
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")  # re-runs skip XLA compilation
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    _spec = importlib.util.spec_from_file_location("m47_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    params, cfg, tok = llama.load(llama.SMOLLM_135M)
    return M, cfg, ex, html, interp, jax, jnp, lj, mo, np, params, plt, rosetta, textwrap, tok


@app.cell
def _(M, np):
    # results written by train.py (None until that stage has run)
    lens_r = M.read_json(M.RUNS / "lens.json")
    sae_r = M.read_json(M.RUNS / "sae.json")
    dash_r = M.read_json(M.RUNS / "dash.json")
    steer_r = M.read_json(M.RUNS / "steer.json")
    ablate_r = M.read_json(M.RUNS / "ablate.json")
    map_r = M.read_json(M.RUNS / "map.json")
    auto_r = M.read_json(M.RUNS / "autointerp.json")
    _f = M.RUNS / "lens.npz"
    lens_params = dict(np.load(_f)) if _f.exists() else None
    return ablate_r, auto_r, dash_r, lens_params, lens_r, map_r, sae_r, steer_r


@app.cell
def _(mo):
    mo.md(r"""
    # 47 · Reading activations: lenses, sparse autoencoders and steering (2020–2025)

    **Take first:** Module 16 or 17 (the residual stream), Module 37 (probes and superposition) and Module 46
    (editing weights; its section 1 readout comes back here).

    **Goal:** Module 37 opened a model you trained yourself. This module opens a real pretrained one,
    SmolLM2-135M, and reads what it computes in between its input and its output. First by decoding every
    layer with the model's own output head, then by splitting its activations into thousands of sparse,
    readable features, and finally by writing to them: adding one direction to make the model cheerful or
    gloomy, and baking the same change into the weights.

    **You will:** watch a prediction form layer by layer with the logit and tuned lenses · train sparse
    autoencoders on the residual stream and score them three ways · browse the features they find and
    classify text with them using Module 46's readout · steer generation with a mean-difference direction and
    measure what it costs · prove that a weight edit and an activation edit are the same thing · compare two
    models' representations and carry a steering direction from one to the other.
    """)
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **Where this came from**

    In 2020 a blogger writing as **nostalgebraist** decoded GPT-2's middle layers with its output matrix,
    the **logit lens**, and watched predictions sharpen layer by layer; **Belrose et al.** (2023) made it
    reliable with a learned map per layer, the **tuned lens**. Reading single neurons kept failing because
    of **superposition** (Module 37). In 2023 Anthropic (**Bricken et al.**) and **Cunningham et al.**
    trained **sparse autoencoders** on activations and found features a person could name; in 2024 they
    were scaled to production models (Anthropic's Scaling Monosemanticity, OpenAI's TopK SAEs by **Gao et
    al.**, DeepMind's open **Gemma Scope**). Writing to activations grew alongside: **ActAdd** (Turner et
    al., 2023) and **CAA** (Rimsky et al., 2023) steer with one added vector, and **Arditi et al.** (2024)
    showed that refusal runs through a single direction that can be cut out of the weights.

    *Timeline: Module 00b.*
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The logit lens: decoding every layer

    Each token carries one vector through the model, the **residual stream** (Module 16). Every attention and
    MLP block reads it and *adds* its output back. Only at the very end does the model turn the stream into a
    prediction: a final RMSNorm, then the output matrix (for SmolLM2, the transposed embedding matrix).

    So ask a blunt question: what if we did that to the stream after layer 5, or 15? The **logit lens** applies
    the final norm and output head to a middle layer's state and reads off the top tokens. If the layers
    refine a running guess in a shared vocabulary space, the guess should come into focus as you go up.

    $$\text{lens}(h_\ell) = \mathrm{RMSNorm}(h_\ell;\, g_{\text{final}})\, W_U$$

    **Your turn:** `logit_lens(h, norm_w, W_U)`.
    """)
    return


@app.cell
def _(cfg, ex, interp, jnp, lj, mo, np, params, tok):
    _ids = jnp.asarray([tok.encode("The capital of France is").ids])
    _h = interp.forward(params, cfg, _ids, collect=True)["resid"][15, 0]
    _a = lj.attempt(ex.logit_lens, _h, params["final_norm"], params["embed"].T)
    if _a.ok:
        _ref = interp.unembed(params, cfg, _h)
        _err = float(jnp.abs(jnp.asarray(_a.value) - _ref).max() / jnp.abs(_ref).max())
        _top = tok.decode([int(np.asarray(_a.value)[-1].argmax())])
        _out = mo.md(f"**`logit_lens`** {'✅' if _err < 2e-2 else '❌'}: layer 15 of 30 on *The capital of France is*, "
                     f"max difference from the course's version {_err:.1e} (relative; bf16 weights). Its top guess "
                     f"for the next token there: `{_top!r}`.")
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo):
    lens_prompt = mo.ui.text(value="The Eiffel Tower is located in the city of", label="prompt", full_width=True)
    lens_prompt
    return (lens_prompt,)


@app.cell
def _(cfg, html, interp, jax, jnp, lens_params, lens_prompt, mo, np, params, tok):
    _ids = jnp.asarray([tok.encode(lens_prompt.value or "Hello").ids])
    _out = interp.forward(params, cfg, _ids, collect=True)
    _R = _out["resid"][:, 0, -1]                                    # (L+1, d): the last token at every layer
    _final = int(_out["logits"][0, -1].argmax())

    def _top(lg, n=3):
        p = jax.nn.softmax(lg)
        i = np.argsort(-np.asarray(lg))[:n]
        return " ".join(f"<code>{html.escape(repr(tok.decode([int(j)])))}</code>&nbsp;{float(p[j]):.2f}" for j in i)

    _rows = []
    for _l in range(0, cfg.n_layers + 1, 2):
        _lg = interp.unembed(params, cfg, _R[_l])
        _tl = "—"
        if lens_params is not None and _l < cfg.n_layers:
            _lp = {k: jnp.asarray(v) for k, v in lens_params.items()}
            _tl = _top(interp.tuned_lens_logits(params, cfg, _lp, _l, _R[_l]))
        _hit = "✓" if int(_lg.argmax()) == _final else ""
        _rows.append(f"<tr><td>{_l}</td><td>{_top(_lg)}</td><td>{_hit}</td><td>{_tl}</td></tr>")
    mo.Html("<table><tr><th>layer</th><th>logit lens: top 3 (probability)</th><th>= final?</th>"
            "<th>tuned lens: top 3</th></tr>" + "".join(_rows) + "</table>"
            + f"<p>Final prediction: <code>{html.escape(repr(tok.decode([_final])))}</code>. Layer 0 is the "
            "token embedding itself; layer 30 is the model's real output.</p>")
    return


@app.cell
def _(lens_r, mo, np, plt):
    if lens_r is None:
        _out = mo.md("*Run `train.py --lens` (about 17 minutes) to train the tuned lens and fill in this plot.*")
    else:
        _fig, _axs = plt.subplots(1, 2, figsize=(9, 3.2))
        for _name, _c in zip(lens_r["domains"], ("C0", "C1", "C2")):
            _d = lens_r["domains"][_name]
            _axs[0].plot(_d["logit_kl"], color=_c, ls="--", label=f"{_name}: logit lens")
            _axs[0].plot(_d["tuned_kl"], color=_c, label=f"{_name}: tuned lens")
            _axs[1].plot(_d["logit_top1"], color=_c, ls="--")
            _axs[1].plot(_d["tuned_top1"], color=_c)
        _axs[0].set(xlabel="layer", ylabel="KL(final ‖ lens), nats", yscale="log", title="how far from the output")
        _axs[1].set(xlabel="layer", ylabel="top-1 agrees with final", title="same top token as the model")
        _axs[0].legend(fontsize=7)
        _fig.tight_layout()
        _d = lens_r["domains"]["detective"]
        _l10 = 10
        _first = lambda xs: next((i for i, x in enumerate(xs) if x >= 0.5), None)  # noqa: E731
        _lt, _tt = _first(_d["logit_top1"]), _first(_d["tuned_top1"])
        _out = mo.vstack([_fig, mo.md(
            f"On held-out detective text, at layer {_l10} the logit lens is {_d['logit_kl'][_l10]:.1f} nats from the "
            f"model's final distribution and the tuned lens {_d['tuned_kl'][_l10]:.1f}. The logit lens first agrees "
            f"with the model's top token half the time at layer {_lt}; the tuned lens at layer {_tt}. "
            f"(Tuned lens: one affine map $h \\mapsto h + hA_\\ell + b_\\ell$ per layer, {lens_r['steps']} Adam steps "
            "on KL to the final output; the model itself is frozen.)")])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What the lens shows.** Early layers decoded directly look like noise, or like the *current* token
    (layer 0 is the embedding, so its best guess is the input word itself). The right answer appears late
    and then firms up. That is the logit lens's known weakness: early layers don't yet write in the output
    head's coordinates, so their real content is hidden. The tuned lens learns a small change of basis per
    layer and reads earlier predictions, which is why its curve drops sooner. Neither proves *how* the model
    gets there; Module 48 traces that.

    **A strange token.** Look at the stream's size: one token in each sequence (the first, and some others
    like it) grows to norms about a hundred times the rest from layer 3 onward. These are **massive
    activations** (Sun et al. 2024), the same attention sinks Module 44 kept in its sliding window: the model
    parks unneeded attention there, and the huge values act like a fixed bias. They matter for what comes
    next, because a few such tokens would dominate any average we take.
    """)
    return


@app.cell
def _(cfg, interp, jnp, mo, np, params, plt, tok):
    _ids = jnp.asarray([tok.encode("The detective looked at the muddy footprints and said nothing. Then he "
                                   "turned to the window").ids])
    _R = np.asarray(interp.forward(params, cfg, _ids, collect=True)["resid"][:, 0].astype(jnp.float32))
    _n = np.linalg.norm(_R, axis=-1)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(_n[:, 0], label="position 0 (the first token)")
    _ax.plot(np.median(_n[:, 1:], 1), label="median of the other positions")
    _ax.plot(_n[:, 1:].max(1), ls=":", label="largest of the other positions")
    _ax.set(yscale="log", xlabel="layer", ylabel="‖h‖", title="residual-stream norm by layer")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"At layer 15: first token {_n[15, 0]:,.0f}, median of the rest {np.median(_n[15, 1:]):,.0f}.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Sparse autoencoders: splitting the stream into features

    Module 37's toy model squeezed 5 sparse features into 2 dimensions by giving each a direction and letting
    them overlap. A language model does this at scale: many more concepts than its 576 dimensions, each a
    direction, few active at once. Reading one *neuron* mixes several of them. A **sparse autoencoder** (SAE)
    tries to undo the squeeze: learn a dictionary of many more directions (8,192 here, 14× the width) such
    that every activation is a sum of just a few.

    $$z = \operatorname{TopK}_k\big(\mathrm{ReLU}((x - b_{dec})W_{enc} + b_{enc})\big), \qquad \hat x = z\,W_{dec} + b_{dec}$$

    Train it to reconstruct $x$ with each decoder row kept at unit length. Two ways to force sparsity:

    * **ReLU + L1** (Bricken et al. 2023): a plain ReLU code, plus a penalty $\lambda\sum_i z_i$. Easy, but
      the penalty also shrinks the codes that *should* be large, and the sparsity you get depends on λ.
    * **TopK** (Gao et al. 2024): keep only the $k$ largest codes per token. Sparsity is exactly $k$ and
      nothing shrinks. Units that stop firing ("dead latents") get no gradient; an auxiliary loss, **AuxK**,
      asks the dead ones to explain what's left of the error, which revives most of them.

    We train on the residual stream after layer 15 (the middle) of SmolLM2-135M, over a mix of detective
    novels (70%), radio-astronomy abstracts (25%) and Shakespeare (5%), leaving out the massive-norm tokens
    (norm over 10× the median: the sinks above).

    **Your turn:** `topk_activation(pre, k)` and `fvu(x, x_hat)`, the fraction of variance unexplained:
    0 is a perfect reconstruction, 1 is no better than predicting the average.
    """)
    return


@app.cell
def _(ex, interp, jax, jnp, lj, mo, np):
    _pre = jax.random.normal(jax.random.key(0), (64, 512))
    _a = lj.attempt(ex.topk_activation, _pre, 8)
    _x = jax.random.normal(jax.random.key(1), (200, 16)) * 3 + 1
    _xh = _x + 0.5 * jax.random.normal(jax.random.key(2), _x.shape)
    _b = lj.attempt(ex.fvu, _x, _xh)
    _parts = []
    if _a.ok:
        _z = np.asarray(_a.value)
        _ok = np.allclose(_z, np.asarray(interp.topk_relu(_pre, 8)), atol=1e-6)
        _parts.append(mo.md(f"**`topk_activation`** {'✅' if _ok else '❌'}: nonzeros per row "
                            f"{(_z > 0).sum(1).min()}–{(_z > 0).sum(1).max()} (at most k = 8; fewer when a kept "
                            "value is negative)."))
    else:
        _parts.append(_a.view(mo))
    if _b.ok:
        _ref = float(interp.fvu(_x, _xh))
        _parts.append(mo.md(f"**`fvu`** {'✅' if abs(float(_b.value) - _ref) < 1e-4 else '❌'}: {float(_b.value):.4f} "
                            f"for noise of std 0.5 on data of std 3 (expected about 0.5²/3² = {0.25 / 9:.4f})."))
    else:
        _parts.append(_b.view(mo))
    mo.vstack(_parts)
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "TopK sparse code", solves="topk_activation")
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "fraction of variance unexplained", solves="fvu")
    return


@app.cell
def _(mo, sae_r, textwrap):
    if sae_r is None:
        _out = mo.md("*Run `train.py --sae` (about 45 minutes: six SAEs on the same 20M tokens) to fill this in.*")
    else:
        _rows = ["| SAE | L0 (codes per token) | FVU | loss recovered | dead in training | fire on > 10% of tokens |",
                 "|---|---|---|---|---|---|"]
        for _n, _s in sae_r["saes"].items():
            _kind = f"TopK, k = {_s['k']}" if _s["kind"] == "topk" else f"ReLU + L1, λ = {_s['l1']:g}"
            _rows.append(f"| {_kind} | {_s['l0']:.1f} | {_s['fvu']:.3f} | {_s['ce_recovered_zero']:.1%} | "
                         f"{_s['dead_train']:,} | {_s['dense_eval']:,} |")
        _out = mo.md("\n".join(_rows) + "\n\n" + textwrap.dedent(f"""
        Measured on {sae_r['n_latents']:,}-latent SAEs after {sae_r['tokens'] / 1e6:.0f}M training tokens, on held-out text.
        **Loss recovered** splices the SAE's reconstruction into the model at layer {sae_r['layer']} and asks how much of
        the damage from zeroing that layer it undoes: the model's next-token loss is {sae_r['ce_clean']:.2f} nats
        clean and {sae_r['ce_zero']:.2f} with the layer zeroed; 100% would mean the SAE loses nothing the model uses.
        {sae_r['outlier_frac']:.2%} of tokens (the sinks) are left untouched.
        """))
    _out
    return


@app.cell
def _(mo, plt, sae_r):
    if sae_r is None:
        _out = mo.md("")
    else:
        _fig, _axs = plt.subplots(1, 2, figsize=(9, 3.2))
        for _kind, _mk, _lab in (("topk", "o-", "TopK"), ("relu", "s--", "ReLU + L1")):
            _ss = sorted([s for s in sae_r["saes"].values() if s["kind"] == _kind], key=lambda s: s["l0"])
            _axs[0].plot([s["l0"] for s in _ss], [s["fvu"] for s in _ss], _mk, label=_lab)
            _axs[1].plot([s["l0"] for s in _ss], [s["ce_spliced"] - sae_r["ce_clean"] for s in _ss], _mk, label=_lab)
        _axs[0].set(xscale="log", xlabel="L0 (active codes per token)", ylabel="FVU", title="reconstruction")
        _axs[1].set(xscale="log", xlabel="L0", ylabel="extra loss when spliced in (nats)", title="what the model loses")
        _axs[0].legend()
        _fig.tight_layout()
        _top = {s["k"]: s for s in sae_r["saes"].values() if s["kind"] == "topk"}
        _relu = [s for s in sae_r["saes"].values() if s["kind"] == "relu"]
        _pairs = [(abs(r["l0"] - t["l0"]) / t["l0"], t, r) for t in _top.values() for r in _relu]
        _cmp = ""
        if _pairs:
            _, _t, _r = min(_pairs, key=lambda x: x[0])
            _better = _t["fvu"] < _r["fvu"]
            _cmp = (f" At similar sparsity, TopK-{_t['k']} (L0 {_t['l0']:.0f}, FVU {_t['fvu']:.3f}, loss recovered "
                    f"{_t['ce_recovered_zero']:.1%}) {'beats' if _better else 'does not beat'} ReLU + L1 at "
                    f"λ = {_r['l1']:g} (L0 {_r['l0']:.0f}, FVU {_r['fvu']:.3f}, {_r['ce_recovered_zero']:.1%}).")
        _dom = _top.get(32, next(iter(_top.values())))["fvu_by_domain"]
        _worst = max(_dom, key=_dom.get)
        _out = mo.vstack([_fig, mo.md(
            "Lower-left is better: fewer active codes, less error. More codes per token always reconstruct better; "
            "the question is how much sparsity costs." + _cmp + " By domain, TopK-32's FVU is "
            + ", ".join(f"{k} {v:.3f}" for k, v in _dom.items())
            + f": {_worst} is hardest, the domain it saw least of or the one most unlike the rest.")])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reading the scores honestly.** Every SAE here leaves a real share of the variance unexplained, and
    splicing any of them in costs the model some loss: an SAE is a *lossy* description of the model, not the
    model. Bigger SAEs trained on billions of tokens do better (Gemma Scope's are trained on 4–16 billion);
    ours saw 20 million. FVU also needs care: a first trial run of this module that kept the sink tokens
    scored FVU 0.05 after 3M tokens, because explaining a few giant norms dominated the variance, while its
    ordinary tokens were reconstructed worse than the numbers above.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Reading features

    A feature is only interesting if a person can say what it does. The usual **dashboard**: where it fires
    most strongly (its top contexts), what it pushes the output toward (its decoder direction sent through the
    final norm and unembedding), and how often it fires. Pick a group below. The domain groups are the
    features whose firing rate in that domain most exceeds every other domain; *random* is an honest sample of
    living features, which tend to be messier than the hand-picked ones papers show. `train.py --dash` collects
    all of this with `jaxprobe.dash` and also writes `runs/47-reading-activations/features.html`: every one of
    the 8,192 features, with a card each, in one page you can open in a browser.
    """)
    return


@app.cell
def _(dash_r, mo):
    if dash_r is None:
        feat_group = None
        _out = mo.md("*Run `train.py --sae` then `train.py --dash` to fill this in.*")
    else:
        feat_group = mo.ui.dropdown(options=list(dash_r["show"]), value="detective", label="features")
        _out = feat_group
    _out
    return (feat_group,)


@app.cell
def _(dash_r, feat_group, mo):
    if dash_r is None or feat_group is None:
        feat_pick = None
        _out = mo.md("")
    else:
        _ids = [str(i) for i in dash_r["show"][feat_group.value]]
        feat_pick = mo.ui.dropdown(options=_ids, value=_ids[0], label="feature id")
        _out = feat_pick
    _out
    return (feat_pick,)


@app.cell
def _(M, auto_r, dash_r, feat_pick, html, mo):
    if dash_r is None or feat_pick is None:
        _out = mo.md("")
    else:
        _f = dash_r["features"][feat_pick.value]
        _a = (auto_r or {}).get("features", {}).get(feat_pick.value)
        _expl = ""
        if _a is not None:
            _pct = {k: "n/a" if _a[k] is None else f"{_a[k]:.0%}" for k in ("detection", "fuzzing")}
            _expl = (f"<p><b>{html.escape(auto_r['model'])} says:</b> {html.escape(_a['explanation'])} <small>"
                     f"(detection {_pct['detection']}, fuzzing {_pct['fuzzing']}: see auto-interpretation below)"
                     "</small></p>")

        def _ctx(c):
            top = max(c["acts"]) or 1.0
            spans = []
            for t, a in zip(c["tokens"], c["acts"]):
                t = html.escape(t).replace("\n", "⏎")
                alpha = 0.85 * max(a, 0) / top
                spans.append(f'<span style="background: rgba(255,140,0,{alpha:.2f})">{t}</span>')
            return f"<li><small>[{c['domain']}]</small> {''.join(spans)}</li>"

        _rates = ", ".join(f"{d} {r:.2%}" for d, r in zip(M.DOMAINS, _f["rate"]))
        _out = mo.Html(
            f"<p><b>Feature {feat_pick.value}</b>: fires on {_f['density']:.3%} of tokens ({_rates}).</p>" + _expl +
            f"<p>Promotes: {' '.join('<code>' + html.escape(repr(t)) + '</code>' for t in _f['promotes'])}<br>"
            f"Suppresses: {' '.join('<code>' + html.escape(repr(t)) + '</code>' for t in _f['suppresses'])}</p>"
            "<p>Top contexts (orange = how strongly it fires on each token):</p><ul>"
            + "".join(_ctx(c) for c in _f["contexts"]) + "</ul>")
    _out
    return


@app.cell
def _(dash_r, mo, np, plt):
    if dash_r is None:
        _out = mo.md("")
    else:
        _c, _e = dash_r["log10_density_hist"]
        _fig, _ax = plt.subplots(figsize=(5.5, 2.8))
        _ax.bar(_e[:-1], _c, width=np.diff(_e), align="edge")
        _ax.set(xlabel="log10(fraction of tokens it fires on)", ylabel="features", title="feature density")
        _fig.tight_layout()
        _acc = dash_r["readout_acc"]
        _ds = dash_r["domain_specific_counts"]
        _out = mo.vstack([_fig, mo.md(f"""
        {dash_r['alive']:,} of 8,192 features fired at least once on {sum(dash_r['n_tokens']):,.0f} held-out tokens.

        **Module 46's readout, again.** In 46 §1 you tabulated how often each hand-built detector fired per digit
        and turned the table into log-odds weights. Do the same with SAE features and *domains*: a feature
        "fires" on a 128-token window if it is active anywhere in it; count per domain on half the windows,
        build the naive Bayes readout, and classify the other half ({dash_r['readout_n_test']:,} windows).
        Accuracy: {', '.join(f'{k} {v:.1%}' for k, v in _acc.items())} (overall {dash_r['readout_overall']:.1%}).
        Features firing over 10× more often in one domain than in either other: {', '.join(f'{k} {v:,}' for k, v in _ds.items())}.
        No feature was trained to know about domains; they fall out of reconstructing the stream.
        """)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Explaining features automatically.** Reading thousands of dashboards by hand doesn't scale, so the field
    hands them to an LLM (Bills et al. 2023; Paulo et al. 2024). Show a chat model half of a feature's top
    contexts with the firing tokens marked and ask for a one-sentence explanation. Then test the explanation
    on the other half, which the model hasn't seen:

    * **Detection:** mix those contexts with random windows where the feature does *not* fire, and ask, for
      each, whether the feature fires anywhere in it. This tests whether the explanation says *which texts*.
    * **Fuzzing:** mark tokens in those contexts, either the ones where it fires or as many others, and ask
      whether the marks are right. This tests whether it says *where*.

    Both are balanced yes/no questions, so 50% is chance. They're scored beside a **control**: every feature
    gets another feature's explanation. If the control scores well too, the test is too easy. `train.py
    --autointerp` does this with `jaxprobe.autointerp`, for the 30 features above plus 170 picked at random.
    """)
    return


@app.cell
def _(auto_r, dash_r, mo, np, textwrap):
    if auto_r is None:
        _out = mo.md(r"""
        This step is optional because it needs a local chat model (LM Studio, Module 27, or llama.cpp, Module
        19). Start the server, load a model, then run `NETWORK=host containers/run.sh python
        modules/47-reading-activations/train.py --autointerp http://localhost:1234/v1 MODEL_NAME` (the course
        ran Qwen3.5-9B: about 2,600 requests in 37 minutes). `NETWORK=host` lets the container reach a server
        that only listens on the host's localhost. Turn off "thinking" for a thinking model: `train.py` sends
        `reasoning_effort: "none"`, without which Qwen3.5 spent over 4,000 tokens on one question.
        """)
    else:
        _S, _F = auto_r["summary"], auto_r["features"]
        _rows = ["| | real explanations | control (another feature's) |", "|---|---|---|"]
        for _m in ("detection", "fuzzing"):
            _rows.append(f"| {_m} | {_S[_m]['accuracy']:.1%} ± {_S[_m]['stderr']:.1%} | "
                         f"{_S['control_' + _m]['accuracy']:.1%} ± {_S['control_' + _m]['stderr']:.1%} |")
        _shown = {str(f) for g in (dash_r or {"show": {}})["show"].values() for f in g}
        _rest = [v for k, v in _F.items() if k not in _shown and v["detection"] is not None]
        _det = np.array([v["detection"] for v in _rest])
        _grp = ["| group | detection | fuzzing |", "|---|---|---|"]
        for _g, _ids in (dash_r or {"show": {}})["show"].items():
            _v = [_F[str(f)] for f in _ids if str(f) in _F]
            _cells = []
            for _k in ("detection", "fuzzing"):
                _d = [x[_k] for x in _v if x[_k] is not None]
                _cells.append(f"{np.mean(_d):.0%}" if _d else "n/a")
            _grp.append(f"| {_g} | {_cells[0]} | {_cells[1]} |")
        _out = mo.md(textwrap.dedent(f"""
        **{auto_r['n_features']} features explained by `{auto_r['model']}`**, mean accuracy over features (±
        standard error):
        """) + "\n\n" + "\n".join(_rows) + textwrap.dedent(f"""

        The explanations carry real information: about 30 points above their own control on detection, and
        about 15 on fuzzing. Of the {len(_rest)} randomly picked features, {(_det >= 0.875).mean():.0%} were
        detected at 14 of 16 or better and {(_det <= 0.5).mean():.0%} at chance or worse. Saying *where*
        (fuzzing) is harder than saying *which texts*: an explanation that names the right kind of text picks out
        the right excerpts without saying which token carries the feature. The feature card above shows each
        explanation. By group:
        """) + "\n\n" + "\n".join(_grp) + textwrap.dedent("""

        The *common* features fire on nearly every token, so there are no windows without them and no tokens to
        mark wrongly: neither test applies. The model still wrote an explanation for each (see their cards),
        and nothing here can tell whether it means anything.

        Two caveats. The held-out contexts are the feature's *top* ones, the easiest; Paulo et al. sample across
        the whole activation range, which scores lower. And negatives are random windows, so an explanation that
        only names the domain ("Shakespeare") gets credit on detection; the control catches that only when the
        borrowed explanation comes from another domain.
        `runs/47-reading-activations/features_explained.html` has all of them, one card per feature.
        """))
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Steering: writing to the stream

    If a concept is a direction, adding that direction should turn the concept up. The recipe (**CAA**, Rimsky
    et al. 2023): run the model on examples with and without a property, take the mean activation of each set
    at one layer, and subtract. Here the property is sentiment: 100 sentences like *"The movie was
    wonderful."* against the same subjects with *terrible*, *awful*, … read at the adjective. **ActAdd**
    (Turner et al. 2023) is the one-pair version: *"I love it."* minus *"I hate it."*

    To steer, add $c \cdot u$ to the residual stream at that layer, on every token, while generating. Two
    numbers keep it honest:

    * **behaviour**: on 12 neutral prompts (*"I went to the restaurant last night and the food was"*), the log
      probability that the next word is positive (*good, nice, lovely, …*) minus negative (*bad, poor, boring,
      …*). These words are **not** the ones the direction was built from, so the model can't score by copying.
    * **damage**: the KL divergence between the steered and unsteered model's next-token predictions on
      detective-novel text that has nothing to do with sentiment.

    $c$ is measured in units of the typical residual norm at that layer, so layers compare fairly.

    **Your turn:** `mean_difference(pos, neg)`.
    """)
    return


@app.cell
def _(M, cfg, ex, interp, jnp, lj, mo, np, params, steer_r, tok):
    _layer = steer_r["best"] if steer_r else 12

    def _acts(adjs):
        t, last = M.pad_batch([tok.encode(f"{s} was {a}.").ids for s in M.SUBJECTS for a in adjs])
        R = interp.forward(params, cfg, t, collect=True, upto=_layer)["resid"][-1].astype(jnp.float32)
        return R[jnp.arange(t.shape[0]), last - 1]

    _a = lj.attempt(ex.mean_difference, _acts(M.BUILD_POS), _acts(M.BUILD_NEG))
    if _a.ok:
        _u = np.asarray(_a.value)
        _f = M.RUNS / "caa.npy"
        _cmp = (f"; cosine with train.py's direction {float(_u @ np.load(_f)[_layer]):.4f}" if _f.exists() else "")
        _out = mo.md(f"**`mean_difference`** {'✅' if abs(np.linalg.norm(_u) - 1) < 1e-4 else '❌ (not unit length)'}: "
                     f"direction at layer {_layer} from {len(M.SUBJECTS) * len(M.BUILD_POS)} pairs{_cmp}.")
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo, plt, steer_r):
    if steer_r is None:
        _out = mo.md("*Run `train.py --steer` (about 5 minutes) to fill this in.*")
    else:
        _L, _m = steer_r["layers"], steer_r["mults"]
        _ip, _in = _m.index(0.25), _m.index(-0.25)
        _fig, _axs = plt.subplots(1, 2, figsize=(9, 3.2))
        for _name, _st in (("CAA (100 pairs)", "-"), ("ActAdd (1 pair)", "--")):
            _sw = steer_r["sweep"]["caa" if _name.startswith("CAA") else "actadd"]
            _axs[0].plot(_L, [_sw[str(layer)][_ip][0] for layer in _L], "C2" + _st, label=f"{_name}, +0.25")
            _axs[0].plot(_L, [_sw[str(layer)][_in][0] for layer in _L], "C3" + _st, label=f"{_name}, −0.25")
            _axs[1].plot(_L, [max(_sw[str(layer)][_ip][1], _sw[str(layer)][_in][1]) for layer in _L], "k" + _st,
                         label=_name)
        _axs[0].axhline(steer_r["base_score"], color="gray", lw=0.8, label="unsteered")
        _axs[0].set(xlabel="layer steered", ylabel="log P(pos) − log P(neg)", title="behaviour at c = ±0.25")
        _axs[1].set(xlabel="layer steered", ylabel="KL on unrelated text (nats)", title="damage at c = ±0.25")
        _axs[0].legend(fontsize=6)
        _axs[1].legend(fontsize=7)
        _fig.tight_layout()
        _b = str(steer_r["best"])
        _c = steer_r["sweep"]["caa"][_b]
        _cos = steer_r["cos_caa_actadd"][int(_b)]
        _out = mo.vstack([_fig, mo.md(
            f"Unsteered, the neutral prompts already lean positive (score {steer_r['base_score']:+.2f}). The best layer "
            f"(largest swing with damage under 0.5 nats) is **{_b}**: c = +0.25 moves the score to {_c[_ip][0]:+.2f} "
            f"and c = −0.25 to {_c[_in][0]:+.2f}, for {max(_c[_ip][1], _c[_in][1]):.2f} nats of KL on detective text. "
            f"Doubling to c = ±0.5 costs {max(_c[_m.index(0.5)][1], _c[_m.index(-0.5)][1]):.2f} nats. CAA and ActAdd "
            f"directions have cosine {_cos:.2f} at that layer: "
            + ("one pair already points roughly the same way as a hundred." if _cos > 0.7 else
               "one pair finds only part of the direction a hundred pairs average out, yet steers nearly as hard "
               "(dashed lines): sentiment is not the only thing *love* minus *hate* changes."))])
    _out
    return


@app.cell
def _(mo, steer_r):
    if steer_r is None:
        _out = mo.md("")
    else:
        _g = steer_r["generations"]
        _n = len(steer_r["prompts"])
        _k = _g["0.0"]["n"] // _n
        _rows = ["| prompt | c = −0.25 | unsteered | c = +0.25 |", "|---|---|---|---|"]
        for _i in (0, 1, 2, 5):
            _cells = [_g[m]["texts"][_i * _k].replace("\n", " ⏎ ").replace("|", "/").strip() for m in ("-0.25", "0.0", "0.25")]
            _rows.append(f"| *{steer_r['prompts'][_i]}* | " + " | ".join(_cells) + " |")
        _sum = ", ".join(f"c = {m}: {_g[m]['pos']} positive-leaning and {_g[m]['neg']} negative-leaning"
                         for m in ("-0.25", "0.0", "0.25"))
        _out = mo.md(f"**Generations at layer {steer_r['best']}** (sampled at temperature 0.7, same random seed in each "
                     f"column):\n\n" + "\n".join(_rows)
                     + f"\n\nOver all {_g['0.0']['n']} samples per setting, counted with a small word list: {_sum}.")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What steering is and isn't.** One vector, found from a hundred sentences and added at one layer, flips
    the tone of what the model writes about restaurants, flights and phones it never saw in those sentences.
    That says sentiment is close to a single direction in this model. But every unit of steering costs damage
    elsewhere, the best layer has to be found by sweeping, and a direction built for one property usually
    drags others along. The same method finds directions for refusal, sycophancy or personas (Arditi et al.
    2024; Chen et al. 2025): it is a research tool for asking what a model represents, and safety work uses
    it to *find* such directions. This course steers only benign properties.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · A weight edit is an activation edit

    Steering adds a direction at runtime. **Directional ablation** removes one: project $u$ out of the
    stream, $h \leftarrow h - (h\cdot u)\,u$, after every block writes to it. Because the stream is a *sum*
    of writes (embeddings, then each attention output, then each MLP output), removing $u$ from every write
    is the same as removing it from the stream, and that can be baked into the weights once:

    $$W' = W - \alpha\,(W u)\,u^\top \quad\text{for every matrix that writes to the stream.}$$

    This is the edit in Module 46 §5: Takehi et al.'s model removed a "looping" direction from MLP outputs,
    and Fan et al.'s model edited itself with $W - \alpha\,u u^\top W$ (the column-vector form of the same
    thing), with $u$ a mean difference like section 4's. Arditi et al. used it to delete a model's refusal
    direction, which is why it matters for safety: a one-line weight change can remove a behaviour. Here we
    use the sentiment direction.

    **Your turn:** `ablate_direction(W, u, alpha)`. The cell below checks the claim itself: it edits every
    writer of a float32 copy of SmolLM2 with *your* function, untying the output head first, and compares
    against projecting the activations at runtime.
    """)
    return


@app.cell
def _(M, cfg, ex, interp, jax, jnp, lj, mo, np, params, steer_r, tok):
    from dataclasses import replace as _replace

    _f = M.RUNS / "caa.npy"
    if _f.exists() and steer_r:
        _u = jnp.asarray(np.load(_f)[steer_r["best"]])
        _what = f"the sentiment direction from layer {steer_r['best']}"
    else:
        _u = jax.random.normal(jax.random.key(3), (cfg.d_model,))
        _u = _u / jnp.linalg.norm(_u)
        _what = "a random direction (run train.py --steer for the sentiment one)"
    _a = lj.attempt(ex.ablate_direction, jnp.eye(3), jnp.array([1.0, 0.0, 0.0]), 1.0)
    if _a.ok:
        _p32 = jax.tree.map(lambda a: a.astype(jnp.float32), params)
        _c32 = _replace(cfg, dtype="float32")
        _t, _last = M.pad_batch([tok.encode(p).ids for p in M.NEUTRAL[:6]])
        _w = interp.forward(M.orthogonalize(_p32, ex, _u), _replace(_c32, tied=False), _t)["logits"]
        _act = interp.forward(_p32, _c32, _t, edit=M.project_out(_u), edit_mid=M.project_out(_u))["logits"]
        _base = interp.forward(_p32, _c32, _t)["logits"]
        _real = (jnp.arange(_t.shape[1])[None] <= _last[:, None])[..., None]
        _d = float(jnp.abs(jnp.where(_real, _w - _act, 0)).max())
        _mv = float(jnp.abs(jnp.where(_real, _base - _act, 0)).max())
        _out = mo.md(f"**`ablate_direction`** {'✅' if _d < 1e-2 * max(_mv, 1e-3) else '❌'}: with {_what}, weights "
                     f"edited by your function and activations projected at runtime give logits within **{_d:.1e}** "
                     f"of each other (float32 rounding), while the edit itself moves them by up to {_mv:.2f}.")
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "directional ablation of a weight matrix", solves="ablate_direction")
    return


@app.cell
def _(ablate_r, mo, textwrap):
    if ablate_r is None:
        _out = mo.md("*Run `train.py --steer --ablate` to fill this in.*")
    else:
        _rows = ["| direction removed | α | neutral prompts | positive contexts | negative contexts | gap | extra loss on detective text |",
                 "|---|---|---|---|---|---|---|"]
        for _r in ablate_r["rows"]:
            _rows.append(f"| {_r['direction']} | {_r['alpha']:g} | {_r['neutral']:+.2f} | {_r['pos_ctx']:+.2f} | "
                         f"{_r['neg_ctx']:+.2f} | {_r['gap']:.2f} | {_r['ce_delta']:+.3f} |")
        _r0 = ablate_r["rows"][0]
        _r1 = next(r for r in ablate_r["rows"] if r["direction"] == "sentiment" and r["alpha"] == 1.0)
        _r2 = next(r for r in ablate_r["rows"] if r["direction"] == "sentiment" and r["alpha"] == 2.0)
        _rr = next(r for r in ablate_r["rows"] if r["direction"] == "random" and r["alpha"] == 1.0)
        if _r2["gap"] < 0:
            _refl = "flips the sign of the gap"
        else:
            _refl = (f"does not flip the model's reading (gap {_r2['gap']:.2f}) and costs {_r2['ce_delta']:+.2f} nats "
                     "on unrelated text")
        _out = mo.md("\n".join(_rows) + "\n\n" + textwrap.dedent(f"""
        Scores are log P(positive word) − log P(negative word). The **contexts** state the sentiment first
        (*"The food was cold and the staff were rude. In one word, the meal was"*), so the **gap** between
        positive and negative contexts measures whether the model can still *use* sentiment it has read.
        Removing the direction (α = 1) shrinks the gap from {_r0['gap']:.2f} to {_r1['gap']:.2f} for
        {_r1['ce_delta']:+.3f} nats on unrelated text; a random direction of the same size leaves it at
        {_rr['gap']:.2f}. Reflecting it (α = 2) {_refl}: the direction carries much of the sentiment signal, but
        not all of it, and not only sentiment.
        """))
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Two models, one space?

    SmolLM2-135M (576 wide, 30 layers) and SmolLM2-360M (960 wide, 32 layers) were trained separately. Do
    they organize text the same way? Their vectors live in different spaces, so compare *neighbourhoods*: for
    each token, which other tokens does each model put nearest? The **Platonic representation hypothesis**
    (Huh et al. 2024) says capable models converge on similar answers. The two models share a tokenizer, so
    token *i* means the same thing in both.

    **Your turn:** `mutual_knn(X, Y, k)`.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _X = jax.random.normal(jax.random.key(4), (300, 32))
    _Q = jnp.linalg.svd(jax.random.normal(jax.random.key(5), (32, 32)))[0]           # a random rotation
    _a = lj.attempt(ex.mutual_knn, _X, 2.0 * _X @ _Q, 10)
    _b = lj.attempt(ex.mutual_knn, _X, jax.random.normal(jax.random.key(6), (300, 32)), 10)
    if _a.ok and _b.ok:
        _ok = abs(float(_a.value) - 1) < 1e-6 and float(_b.value) < 0.2
        _out = mo.md(f"**`mutual_knn`** {'✅' if _ok else '❌'}: a rotated, scaled copy scores {float(_a.value):.3f}; an "
                     f"unrelated cloud scores {float(_b.value):.3f} (chance is 10/299 = {10 / 299:.3f}).")
    else:
        _out = (_a if not _a.ok else _b).view(mo)
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "mutual nearest-neighbour alignment", solves="mutual_knn")
    return


@app.cell
def _(map_r, mo, np, plt):
    if map_r is None:
        _out = mo.md("*Run `train.py --map` (about 5 minutes; downloads SmolLM2-360M once) to fill this in.*")
    else:
        _K = np.asarray(map_r["knn"])
        _fig, _ax = plt.subplots(figsize=(5.2, 3.8))
        _im = _ax.imshow(_K, origin="lower", cmap="viridis", aspect="auto")
        _ax.set_xticks(range(len(map_r["layers_360m"])), map_r["layers_360m"])
        _ax.set_yticks(range(len(map_r["layers_135m"])), map_r["layers_135m"])
        _ax.set(xlabel="SmolLM2-360M layer", ylabel="SmolLM2-135M layer",
                title=f"mutual {map_r['k']}-NN agreement ({map_r['n_knn']:,} tokens)")
        _fig.colorbar(_im)
        _fig.tight_layout()
        _i, _j = np.unravel_index(_K.argmax(), _K.shape)
        _m = map_r["map"]
        _out = mo.vstack([_fig, mo.md(
            f"Best match: 135M layer {map_r['layers_135m'][_i]} with 360M layer {map_r['layers_360m'][_j]}, "
            f"{_K[_i, _j]:.2f} of neighbours shared, against {map_r['knn_shuffled']:.3f} when the token pairing is "
            f"shuffled: depth {map_r['layers_135m'][_i] / 30:.2f} of the way up one model matches depth "
            f"{map_r['layers_360m'][_j] / 32:.2f} of the other. A linear map from 135M layer {_m['from']} "
            f"to 360M layer {_m['to']}, fit by least squares on {_m['n_train']:,} tokens, explains R² = {_m['r2']:.2f} "
            f"of the 360M activations on {_m['n_test']:,} held-out tokens and picks out the right token among them "
            f"{_m['top1']:.1%} of the time from its mapped vector alone.")])
    _out
    return


@app.cell
def _(map_r, mo, textwrap):
    if map_r is None:
        _out = mo.md("")
    else:
        _t = map_r["transfer"]
        _rows = ["| steering 360M with | c = −0.25 | c = +0.25 |", "|---|---|---|"]
        for _k in ("360M's own direction", "mapped 135M direction", "random direction"):
            _rows.append(f"| {_k} | {_t[_k]['-0.25']:+.2f} | {_t[_k]['0.25']:+.2f} |")
        _own = abs(_t["360M's own direction"]["0.25"] - _t["base"])
        _mapped = abs(_t["mapped 135M direction"]["0.25"] - _t["base"])
        _how = "almost as well as" if _mapped > 0.6 * _own else "less well than"
        _out = mo.md(textwrap.dedent(f"""
        **Carrying a direction across.** Send section 4's sentiment direction from the small model through the
        linear map and use it to steer the big one (unsteered score {_t['base']:+.2f}):
        """) + "\n" + "\n".join(_rows) + "\n\n" + textwrap.dedent(f"""
        The mapped direction has cosine {map_r['cos_mapped_own']:.2f} with the 360M model's own sentiment
        direction, and steers it {_how} that one; a random direction does nothing. The map needed paired tokens
        to fit; **vec2vec** (Jha et al. 2025) learns such translations between embedding models with *no* pairs
        at all, from the shared geometry alone: the strongest evidence yet for the Platonic view, and a privacy
        problem for anyone who thought embeddings were safe to share.
        """))
    _out
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"Every SAE feature corresponds to one concept the model uses, so the SAE is an exact "
                 "description of the model": "a",
                 "Adding the direction at runtime and removing it from every matrix that writes to the residual "
                 "stream give the same outputs, because the stream is a sum of those writes": "b",
                 "The logit lens shows that early layers don't know the answer": "c"},
        label="**Check yourself:** which statement do this lesson's measurements support?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("No: every SAE here leaves variance unexplained and costs loss when spliced in, and "
                              "plenty of features are hard to name. It is a lossy, useful description."), kind="warn"),
        "b": mo.callout(mo.md("Right (for *removing* a direction): section 5 checks it to float32 rounding. Adding a "
                              "fixed vector at one layer is a bias change instead, also a weight edit."),
                        kind="success"),
        "c": mo.callout(mo.md("Not quite: early layers don't write in the output head's coordinates yet. The tuned "
                              "lens, with one learned map per layer, reads predictions earlier."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(47))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    * [Neuronpedia](https://www.neuronpedia.org/) — browse dashboards for published SAEs on GPT-2, Gemma and
      Llama, the same views as section 3 at full scale.
    * Templeton et al., [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
      — SAEs on Claude 3 Sonnet, with feature steering (the Golden Gate Bridge) and safety-relevant features.
    * [SAELens](https://github.com/decoderesearch/SAELens) — the standard PyTorch library for training and
      loading SAEs; its file layout is what Gemma Scope and others publish in.
    * Arditi et al., [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)
      — section 5's projection applied to refusal, and what it means for open-weights safety.
    * [jaxprobe](https://github.com/joebarbere/jaxprobe) — this course's own interpretability library, grown out of
      this module. `learnjax/llama.py` marks its forward pass with jaxprobe taps, and `interp.forward` is a thin
      layer over `jaxprobe.run`. Use the library directly to capture or patch any tap, e.g. each layer's
      attention output. It also has BatchTopK and JumpReLU SAEs.

    **Next:** Module 48, tracing a circuit: patching, transcoders and attribution graphs.
    """)
    return


if __name__ == "__main__":
    app.run()
