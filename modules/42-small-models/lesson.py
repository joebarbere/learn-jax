import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="42 · Small models done well")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import json

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from tokenizers import Tokenizer

    import learnjax as lj
    from learnjax import rosetta

    ex = lj.load_exercises(mo.notebook_dir())
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")  # re-runs skip XLA compilation
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    _spec = importlib.util.spec_from_file_location("m42_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    RUNS = mo.notebook_dir().parent.parent / "runs" / "42-small-models"
    return M, RUNS, Tokenizer, ex, jax, jnp, json, lj, mo, np, plt, rosetta


@app.cell
def _(mo):
    mo.md(r"""
    # 42 · Small models done well: distillation, pruning and borrowing from big models (2015–2026)

    **Take first:** Module 16 (training a GPT), Module 18 (`learnjax/llama.py` and SmolLM2), Module 05
    (Muon). Modules 40 and 41 trained on the same detective corpus.

    **Goal:** you have one 16 GB GPU and a few hours. What gives the best small language model? Not a new
    architecture. The tricks that matter are about where the knowledge comes from: a **teacher** that shows
    the student a whole distribution instead of one right answer, a **pretrained model** to start from, a
    big model **cut down** to size, **data** worth learning from, and an optimizer that learns faster per
    step. Every 2025 small open model (Gemma 3, Qwen3, Llama 3.2, SmolLM2) was made with some of these.

    **You will:** price a training run in GPU-hours · write the distillation loss · train a 31M-parameter
    student from scratch, by distillation, and with Muon · continue SmolLM2-135M's pretraining · prune
    SmolLM2-360M Minitron-style with your importance scores and heal it · filter the data with the teacher
    · compare all of them on the same validation tokens.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Buciluă, Caruana and Niculescu-Mizil compressed ensembles into one network in 2006; Hinton, Vinyals and
    Dean (2015) named it **distillation** and showed the "dark knowledge" in a teacher's wrong answers.
    DistilBERT (2019) made it standard for language models. Pruning is older still (LeCun's Optimal Brain
    Damage, 1989), but for LLMs it took NVIDIA's Minitron (2024) — prune by measured importance, then heal
    by distillation — to make it pay. Meanwhile data got the credit it deserved: TinyStories and phi-1 (2023)
    showed tiny models learn far more from clean, simple text. Chinchilla (2022) said how many tokens a model
    of a given size *should* see; Llama 3, SmolLM2 and Gemma (2024–25) trained small models on 100–1,000×
    more, because a small model is cheap to *run*. By 2025 Gemma 2 and 3, Llama 3.2 and Qwen3's small models
    were all distilled from bigger siblings.

    *Timeline: Module 00b.*
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · What a training run costs

    A dense model does about **6 FLOPs per parameter per token** in training (2 forward, 4 backward). Your
    RX 7600 XT measured about 27 TFLOP/s on large bf16 matmuls (Module 00); real training gets a fraction of
    that, the **MFU** (Module 22). Chinchilla's rule of thumb is ~20 tokens per parameter for the best model
    *per training FLOP*. Small open models ignore it on purpose.

    **Your turn:** `training_cost(n_params, n_tokens, tflops, mfu)`.
    """)
    return


@app.cell
def _(ex, lj, mo):
    _rows_in = [("this module's student", 31.5e6, 49e6), ("SmolLM2-135M", 135e6, 2e12),
                ("SmolLM2-360M (the teacher)", 360e6, 4e12), ("SmolLM2-1.7B", 1.7e9, 11e12),
                ("Llama 3 8B", 8e9, 15e12)]
    _att = [(n, lj.attempt(ex.training_cost, p, t, 27.0, 0.4)) for n, p, t in _rows_in]
    if not all(a.ok for _n, a in _att):
        _out = next(a for _n, a in _att if not a.ok).view(mo)
    else:
        def _dur(h):
            return f"{h * 60:.0f} min" if h < 1 else f"{h:,.0f} h" if h < 24 * 60 else f"{h / 24 / 365:,.1f} years"
        _rows = "\n".join(f"| {n} | {a.value['tokens_per_param']:,.0f} | {a.value['flops']:.1e} | "
                          f"**{_dur(a.value['hours'])}** |" for n, a in _att)
        _out = mo.md(f"""
    On your GPU at 27 TFLOP/s and 40% MFU:

    | model | tokens per parameter (Chinchilla: 20) | training FLOPs | time on one RX 7600 XT |
    |---|---|---|---|
    {_rows}

    The student fits in an afternoon; even the smallest real SmolLM2 would take your card years. Every
    trick below is a way to **borrow** some of those years from someone else's GPUs: their model's
    predictions, their weights, or their judgment about data.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Distillation: learn from the teacher's whole distribution

    Ordinary training says "the next token was *gun*". A teacher says "*gun* 41%, *revolver* 22%, *pistol*
    9%, *letter* 3%…": it knows *revolver* is nearly right and *letter* isn't. That ranking of wrong answers
    is information the one-hot label throws away. Hinton et al. train the student to match it:

    $$\mathcal L_{\mathrm{KD}} = T^2\, \mathrm{KL}\big(\mathrm{softmax}(\ell/T) \,\Vert\, \mathrm{softmax}(z/T)\big),$$

    with $\ell$ the teacher's logits, $z$ the student's, and a temperature $T \ge 1$ that flattens both so the
    small probabilities count. Storing all 49,152 probabilities for every token of the corpus would take
    ~4 TB, so the teacher pass (`train.py --run teacher`) keeps each position's **top 16** (Gemma 3 keeps
    256 sampled ones). The teacher is **SmolLM2-360M**; the student is a 31M-parameter SmolLM2-shaped model
    with the same tokenizer (distillation needs the two to share a vocabulary).

    **Your turn:** `kd_loss(student_logits, top_ids, top_logp, T)`.
    """)
    return


@app.cell
def _(M, RUNS, Tokenizer, ex, jax, jnp, lj, mo, np, plt):
    _k = jax.random.split(jax.random.key(0), 2)
    _t = jax.nn.log_softmax(jax.random.normal(_k[0], (4, 64)) * 3)
    _v, _i = jax.lax.top_k(_t, 64)
    _a = lj.attempt(ex.kd_loss, jax.random.normal(_k[1], (4, 64)), _i, _v)
    _b = lj.attempt(ex.kd_loss, _t, _i, _v)
    kd_ready = _a.ok and _b.ok
    if not kd_ready:
        _out = (_a if not _a.ok else _b).view(mo)
    else:
        _kl = float(jnp.mean(jnp.sum(jnp.exp(_t) * (_t - jax.nn.log_softmax(jax.random.normal(_k[1], (4, 64)))), -1)))
        _txt = (f"With all 64 tokens kept, your loss is the KL divergence: **{float(_a.value):.4f}** vs "
                f"{_kl:.4f} ✅, and a student equal to the teacher scores **{float(_b.value):.1e}**.")
        _tdir = RUNS / "teacher"
        if (_tdir / "top_logp.npy").exists():
            _lp = np.load(_tdir / "top_logp.npy", mmap_mode="r")
            _ids = np.load(_tdir / "top_ids.npy", mmap_mode="r")
            _row = np.asarray(_lp[3, 200], np.float32)
            _tok = Tokenizer.from_file(str(M.tokenizer_path()))
            _labels = [repr(_tok.decode([int(i)]))[1:-1] for i in _ids[3, 200]]
            _fig, _ax = plt.subplots(figsize=(7, 2.6))
            for _T, _c in ((1.0, "C0"), (2.0, "C1")):
                _p = np.exp(_row / _T - np.max(_row / _T))
                _ax.bar(np.arange(16) + (0.2 if _T > 1 else -0.2), _p / _p.sum(), 0.4, color=_c, label=f"T = {_T:g}")
            _ax.set_xticks(range(16), _labels, rotation=60, fontsize=7)
            _ax.set_ylabel("teacher probability")
            _ax.legend(fontsize=8)
            _fig.tight_layout()
            _out = mo.vstack([mo.md(_txt + " Below: the teacher's top 16 next tokens at one position of the "
                                    "training text, as stored by the teacher pass, at T = 1 and T = 2."), _fig])
        else:
            _out = mo.md(_txt + " (Run `train.py --run teacher` to see the teacher's real predictions here.)")
    _out
    return (kd_ready,)


@app.cell
def _(mo, rosetta):
    mo.vstack([rosetta.tabs(mo, "distillation loss (teacher's top-k, temperature T)", solves="kd_loss"),
               rosetta.tabs(mo, "KL divergence")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · The student race

    Four 31M students, same architecture, same 6,000 steps of 16 × 512 tokens (49M tokens, ~2.4 passes over
    the corpus), all scored on the same 64 validation windows:

    * **scratch** — cross-entropy, AdamW.
    * **kd** — the distillation loss only (α = 1, T = 1, as Gemma 2 does), against the teacher's top 16.
    * **kd-mix** — half distillation, half cross-entropy on the true token (α = 0.5: Hinton's recipe).
    * **muon** — cross-entropy, Muon for the hidden matrices (section 6), AdamW for the rest.
    * **data-best / data-random** — cross-entropy on half the corpus (section 5).

    The dashed lines are models nobody here trained: SmolLM2-360M (the teacher) and SmolLM2-135M, scored
    as they come.
    """)
    return


@app.cell
def _(M, RUNS, json, mo, plt):
    _names = ["scratch", "kd", "kd-mix", "muon", "data-best", "data-random"]
    runs = {n: M.read_metrics(RUNS / n / "metrics.jsonl") for n in _names}
    runs = {k: v for k, v in runs.items() if v}
    _bp = RUNS / "teacher" / "baselines.json"
    baselines = json.loads(_bp.read_text()) if _bp.exists() else {}
    if not runs:
        _out = mo.callout(mo.md("""
    **No runs yet.** The teacher pass first (~30 min), then each student (~20 min):
    ```
    T="containers/run.sh python modules/42-small-models/train.py"
    $T --run teacher
    for r in scratch kd muon data-best data-random cpt135 pruned pruned-ce; do $T --run $r; done
    ```"""), kind="info")
    else:
        _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
        for _i, (_k, _m) in enumerate(runs.items()):
            _ax.plot([r["step"] for r in _m], [r["val"] for r in _m], label=_k, color=f"C{_i}")
        for _j, (_n, _b) in enumerate(baselines.items()):
            _ax.axhline(_b["val"], ls="--", lw=0.8, color=f"C{6 + _j}", label=_n)
        _ax.set_ylim(top=min(6.0, max(r["val"] for m in runs.values() for r in m)))
        _ax.set_xlabel("step")
        _ax.set_ylabel("validation loss (nats / token)")
        _ax.legend(fontsize=7)
        _fig.tight_layout()
        _rows = "\n".join(f"| {k} | {m[-1]['step']:,} | {m[-1]['val']:.3f} | "
                          f"{sum(r['tok_per_s'] for r in m[1:]) / max(len(m) - 1, 1):,.0f} | "
                          f"{m[-1]['elapsed_s'] / 60:.0f} min |" for k, m in runs.items())
        _brows = "\n".join(f"| {n} ({b['params'] / 1e6:.0f}M, not trained here) | – | {b['val']:.3f} | – | – |"
                           for n, b in baselines.items())
        _out = mo.vstack([_fig, mo.md(f"""
    | model | steps | val loss | tok/s | time |
    |---|---|---|---|---|
    {_rows}
    {_brows}
    """)])
    _out
    return baselines, runs


@app.cell
def _(RUNS, baselines, mo, runs):
    _f = {k: m[-1]["val"] for k, m in runs.items() if m[-1]["step"] == 6000}
    _parts = []
    if "scratch" in _f and "kd" in _f:
        _d = _f["kd"] - _f["scratch"]
        _parts.append(f"Distillation alone moved the student's validation loss by **{_d:+.3f}** nats per token "
                      f"({'better' if _d < 0 else 'worse'} than training on the true tokens).")
        if _d > 0:
            _parts.append("Why worse? The stored targets are the teacher's top 16, renormalized: they say every "
                          "other token has probability zero. But the teacher itself scores "
                          f"{baselines.get('SmolLM2-360M (teacher)', {}).get('val', float('nan')):.2f} nats, so the "
                          "true next token is often outside its top 16, and a student trained only on those "
                          "targets learns to be confidently wrong there. Full-vocabulary distillation doesn't "
                          "have this problem; Gemma 3 keeps 256 sampled logits per token, not 16.")
    if "scratch" in _f and "kd-mix" in _f:
        _dm = _f["kd-mix"] - _f["scratch"]
        _parts.append(f"Mixing the true token back in (kd-mix, half and half): **{_dm:+.3f}** against scratch"
                      + (" — the soft targets now help, at no extra cost per step: the teacher ran once, in "
                         "advance." if _dm < 0 else ". At this budget the teacher's top 16 didn't add enough "
                         "to beat training on the true tokens."))
    if "scratch" in _f and "muon" in _f:
        _parts.append(f"Muon: **{_f['muon'] - _f['scratch']:+.3f}** against AdamW at the same step count.")
    _t = baselines.get("SmolLM2-360M (teacher)", {}).get("val")
    if _t is not None and _f:
        _parts.append(f"The teacher itself scores {_t:.3f}: an 11× smaller student trained for 25 minutes is far "
                      f"behind, whatever the recipe.")
    _samples = {n: (RUNS / n / "sample.txt").read_text() for n in ("scratch", "kd-mix")
                if (RUNS / n / "sample.txt").exists()}
    _stxt = "\n\n".join(f"**{n}:** {s.strip()[:400]}" for n, s in _samples.items())
    mo.md(" ".join(_parts) + (f"\n\nSamples (temperature 0.7, same prompt):\n\n{_stxt}" if _stxt else "")
          if _parts else "")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Start from a pretrained model

    The cheapest way to borrow compute: don't start from random weights. **Continued pretraining** keeps
    training SmolLM2-135M — which has already seen 2 trillion tokens of web text, code and math — on the
    detective corpus, with the same next-token loss, for 4,000 steps of 8 × 512 tokens (16M tokens, a third
    of what the students see). Before any training it already knows
    English; the steps here teach it the corpus's style and names. (One caveat: these are public-domain
    novels, and pretraining sets often include such books, so SmolLM2 may have seen some of them.)
    """)
    return


@app.cell
def _(M, RUNS, baselines, mo, runs):
    _c = M.read_metrics(RUNS / "cpt135" / "metrics.jsonl")
    if not _c:
        _out = mo.md("*(Run `train.py --run cpt135` for this section's numbers.)*")
    else:
        _b0 = baselines.get("SmolLM2-135M", {}).get("val")
        _sc = runs.get("scratch", [{}])[-1].get("val")
        _first = _c[0]
        _txt = f"SmolLM2-135M continued for {_c[-1]['step']:,} steps reaches **{_c[-1]['val']:.3f}**"
        if _b0 is not None:
            _txt += f" (it started at {_b0:.3f} before any training here; after {_first['step']} steps: {_first['val']:.3f})"
        if _sc is not None and _b0 is not None and _b0 < _sc:
            _txt += (f". The 31M student trained from scratch on 49M tokens: {_sc:.3f} — the pretrained model "
                     f"beat it before a single step here, and then improved further on a third of the tokens. "
                     f"(It's also 4× bigger, and it saw 2T tokens on someone else's GPUs.)")
        elif _sc is not None and _c[-1]["val"] < _sc:
            _txt += (f". The 31M student trained from scratch on 49M tokens: {_sc:.3f}; the pretrained model got "
                     f"there on a third of the tokens.")
        elif _sc is not None:
            _txt += f". The 31M student trained from scratch on 49M tokens: {_sc:.3f}."
        _out = mo.md(_txt + " When a pretrained model of the right size exists, starting from it beats any trick "
                            "for training from scratch.")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Prune a big model, then heal it (Minitron)

    What if the pretrained model you have is the wrong size — too big? NVIDIA's **Minitron** recipe: run
    the big model on a small calibration set, score how much each piece is used, cut the least important
    pieces, then **heal** the damage with a short distillation from the original. They made Nemotron-4 8B and
    4B from 15B with up to 40× fewer training tokens than training those sizes from scratch.

    Two scores, both measured on activations, no gradients needed:

    * **Width:** a feed-forward neuron's importance is its mean absolute activation, $\mathbb E\,|a_j|$, where
      $a = \mathrm{silu}(h W_{\text{gate}}) \odot h W_{\text{up}}$. Neurons that are always near zero can go.
    * **Depth:** a layer's **block influence** (ShortGPT, 2024) is $1 - \cos(h_{\text{in}}, h_{\text{out}})$
      averaged over tokens: how much the layer turns the hidden state. Layers that barely turn it can go.

    **Your turn:** `ffn_importance`, `prune_ffn` and `layer_importance`. `train.py --run pruned` then cuts
    SmolLM2-360M from 32 layers to 24 and from 2,560 to 1,280 feed-forward neurons per layer (362M → 195M
    parameters), and heals it for 4,000 steps of 8 × 512 tokens with section 3's kd-mix loss against the
    uncut model's top 16 (`pruned-ce`: with cross-entropy alone).
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _k = jax.random.split(jax.random.key(3), 4)
    _acts = jax.random.normal(_k[0], (16, 128, 10)) * jnp.arange(10)   # (32, 128, 10) hangs XLA: ROCM_GOTCHAS #20e
    _a = lj.attempt(ex.ffn_importance, _acts)
    _lay = {"w_gate": jax.random.normal(_k[1], (2, 4, 10)), "w_up": jax.random.normal(_k[2], (2, 4, 10)),
            "w_down": jax.random.normal(_k[3], (2, 10, 4)), "attn_norm": jnp.ones((2, 4))}
    _b = lj.attempt(ex.prune_ffn, _lay, jnp.array([[9, 8, 7], [0, 1, 2]]))
    _h = jax.random.normal(_k[0], (3, 5, 8))
    _c = lj.attempt(ex.layer_importance, _h, _h + 0.1 * jax.random.normal(_k[1], (3, 5, 8)))
    prune_ready = _a.ok and _b.ok and _c.ok
    if not prune_ready:
        _out = next(x for x in (_a, _b, _c) if not x.ok).view(mo)
    else:
        _imp = [round(float(x), 2) for x in _a.value]
        _out = mo.md(f"""
    Neurons scaled by 0…9: your importances {_imp} — they rank in order ✅. Pruning keeps shapes
    {tuple(_b.value['w_gate'].shape)} / {tuple(_b.value['w_down'].shape)} ✅. A layer that adds only a little
    noise to its input has block influence **{float(_c.value):.4f}** (near 0: a candidate to drop) ✅.
    """)
    _out
    return (prune_ready,)


@app.cell
def _(M, RUNS, baselines, json, mo, np, plt):
    _p = M.read_metrics(RUNS / "pruned" / "metrics.jsonl")
    _pc = M.read_metrics(RUNS / "pruned-ce" / "metrics.jsonl")
    _cfgp = RUNS / "pruned" / "config.json"
    if not _p or not _cfgp.exists():
        _out = mo.md("*(Run `train.py --run pruned` and `--run pruned-ce` for this section's numbers.)*")
    else:
        _meta = json.loads(_cfgp.read_text())
        _bi = np.array(_meta["block_influence"])
        _kept = set(_meta["layers_kept"])
        _fig, _ax = plt.subplots(figsize=(6.5, 2.4))
        _ax.bar(range(len(_bi)), _bi, color=["C0" if i in _kept else "C3" for i in range(len(_bi))])
        _ax.set_yscale("log")
        _ax.set_xlabel("SmolLM2-360M layer (red: dropped)")
        _ax.set_ylabel("block influence")
        _fig.tight_layout()
        _drop = sorted(set(range(len(_bi))) - _kept)
        _t = baselines.get("SmolLM2-360M (teacher)", {}).get("val")
        _r = baselines.get("SmolLM2-135M", {}).get("val")
        _c135 = M.read_metrics(RUNS / "cpt135" / "metrics.jsonl")
        _rows = [("SmolLM2-360M, uncut (the teacher)", "362M", _t),
                 ("pruned, before healing", f"{_meta['params'] / 1e6:.0f}M", _meta["val_before_healing"]),
                 ("pruned + 16M tokens of kd-mix", f"{_meta['params'] / 1e6:.0f}M", _p[-1]["val"])]
        if _pc:
            _rows.append(("pruned + 16M tokens of cross-entropy", f"{_meta['params'] / 1e6:.0f}M", _pc[-1]["val"]))
        _rows.append(("SmolLM2-135M as released", "135M", _r))
        if _c135:
            _rows.append(("SmolLM2-135M + 16M tokens (section 4)", "135M", _c135[-1]["val"]))
        _table = "\n".join(f"| {a} | {b} | {c:.3f} |" for a, b, c in _rows if c is not None)
        _out = mo.vstack([_fig, mo.md(f"""
    Block influence of each of the teacher's 32 layers on 64 calibration windows. The first layer changes
    its input completely (it turns token embeddings into something else) and the last turns the state
    toward the output vocabulary; the middle layers each turn it only a little, and the pass dropped
    layers {_drop}.

    | model | parameters | val loss |
    |---|---|---|
    {_table}
    """)])
    _out
    return


@app.cell
def _(M, RUNS, mo):
    _p = M.read_metrics(RUNS / "pruned" / "metrics.jsonl")
    _pc = M.read_metrics(RUNS / "pruned-ce" / "metrics.jsonl")
    if _p and _pc:
        _d = _p[-1]["val"] - _pc[-1]["val"]
        _why = ("That is the Minitron result: after pruning, the uncut model is the best teacher for putting back "
                "what was lost." if _d < 0 else
                "Here plain cross-entropy healed as well or better: with only 16M tokens, the soft targets added "
                "little.")
        _out = mo.md(f"""
    Healing with the teacher's help (kd-mix) ended **{abs(_d):.3f}** {"below" if _d < 0 else "above"}
    healing with cross-entropy alone. {_why} Either way, almost half the network was removed in seconds and
    most of the damage repaired in under an hour on one GPU.
    """)
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Better data, and a better optimizer

    **Data.** TinyStories showed that models under 10M parameters can write fluent, consistent stories if
    the data uses a 3–4-year-old's vocabulary; phi-1 trained a 1.3B code model on ~7B tokens of
    "textbook-quality" text and reached 50.6% on HumanEval; FineWeb-Edu kept the web pages an LLM-trained
    classifier rated educational and gained 4 points on MMLU. The cheapest version here: the teacher pass
    already measured the teacher's loss on every 512-token window. **data-best** trains on the half the
    teacher finds most predictable (**perplexity filtering**), **data-random** on a random half, for the
    same number of steps.

    **Muon** (Module 05) replaces each hidden matrix's update with its momentum, **orthogonalized**: every
    singular value pushed to about 1, so no direction of the update dominates. Five Newton–Schulz steps do it
    with matmuls alone. It took the nanoGPT speedrun's record from 45 minutes (llm.c's baseline) toward the
    current 40 seconds on 8 H100s, with many other tricks; Moonshot reports about 2× AdamW's compute
    efficiency at scale.

    **Your turn:** `newton_schulz(G, steps)`.
    """)
    return


@app.cell
def _(ex, jax, lj, mo, np, plt):
    _G = jax.random.normal(jax.random.key(5), (96, 48)) @ jax.numpy.diag(jax.numpy.linspace(0.05, 3, 48))
    _a = lj.attempt(ex.newton_schulz, _G)
    if not _a.ok:
        _out = _a.view(mo)
    else:
        _s0 = np.linalg.svd(np.asarray(_G), compute_uv=False)
        _s1 = np.linalg.svd(np.asarray(_a.value), compute_uv=False)
        _fig, _ax = plt.subplots(figsize=(5.5, 2.4))
        _ax.plot(_s0 / _s0.max(), "o-", ms=3, label="G (scaled to max 1)")
        _ax.plot(_s1, "o-", ms=3, label="newton_schulz(G)")
        _ax.set_xlabel("singular value index")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _out = mo.vstack([mo.md(f"""
    Singular values before (spread over {_s0.min() / _s0.max():.2f}–1 of the largest) and after: all within
    **{_s1.min():.2f}–{_s1.max():.2f}** ✅. Not exactly 1: Muon's coefficients trade accuracy for speed, and
    Keller Jordan found a band around 1 trains as well as exact orthogonalization.
    """), _fig])
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "Muon: orthogonalized update (Newton–Schulz)", solves="newton_schulz")
    return


@app.cell
def _(M, RUNS, Tokenizer, mo, np):
    _b = M.read_metrics(RUNS / "data-best" / "metrics.jsonl")
    _r = M.read_metrics(RUNS / "data-random" / "metrics.jsonl")
    _nll_p = RUNS / "teacher" / "nll.npy"
    if not (_b and _r and _nll_p.exists()):
        _out = mo.md("")
    else:
        _nll = np.load(_nll_p)
        _tok = Tokenizer.from_file(str(M.tokenizer_path()))
        _w = M.windows(M.smol_corpus()["train"])
        _lo, _hi = int(np.argmin(_nll)), int(np.argmax(_nll))
        _show = lambda i: repr(_tok.decode(_w[i, :60].tolist()))[1:-1].replace("|", "\\|")[:300]
        _d = _b[-1]["val"] - _r[-1]["val"]
        _out = mo.md(f"""
    On the full validation set, the model trained on the teacher's easiest half ended **{_d:+.3f}** against
    the random half ({_b[-1]['val']:.3f} vs {_r[-1]['val']:.3f}). {'Filtering helped.' if _d < -0.01 else 'Filtering did not help here.' if _d > 0.01 else 'No real difference.'}
    Look at what the filter likes and dislikes. The window the teacher found easiest (loss {_nll[_lo]:.2f}):

    > {_show(_lo)}

    and the hardest ({_nll[_hi]:.2f}):

    > {_show(_hi)}

    "Predictable" means clean prose, but also repetition, tables of contents and boilerplate. Perplexity
    filtering is a blunt tool; FineWeb-Edu's classifier asks a better question ("is this educational?"),
    and phi and TinyStories wrote the data they wanted instead of filtering for it.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · What the 2025 small models did

    | model | how it got good | source |
    |---|---|---|
    | SmolLM2 135M / 360M / 1.7B (Hugging Face) | data: 2, 4 and 11T carefully mixed tokens, no distillation | arXiv 2502.02737 |
    | Gemma 2 2B / 9B (Google, 2024) | distillation instead of next-token prediction, >50× compute-optimal tokens | arXiv 2408.00118 |
    | Gemma 3 1B / 4B (2025) | distillation, 256 sampled teacher logits per token; 2T / 4T tokens | arXiv 2503.19786 |
    | Llama 3.2 1B / 3B (Meta, 2024) | pruned from Llama 3.1 8B, then trained on 8B and 70B logits | Meta model card |
    | Qwen3 0.6B–4B (Alibaba, 2025) | "strong-to-weak" distillation from the big Qwen3 models | arXiv 2505.09388 |
    | Llama-3.1-Minitron 4B (NVIDIA, 2024) | Minitron prune + distill from Llama 3.1 8B, 94B tokens | arXiv 2408.11796 |

    Two more ideas, not trained here:

    * **µP** (Yang et al., 2022): parametrize the model so the best learning rate doesn't change with width;
      tune on a 10M model, train the 1B with the same settings.
    * **Sparse upcycling** (Komatsuzaki et al., 2022): turn a trained dense model into a mixture of experts
      by copying its feed-forward block into every expert (Modules 17 and 40), then keep training.

    The lesson of this module: for a single GPU, the question isn't "which architecture?" but "whose
    compute can I reuse?" — a teacher's predictions, a pretrained checkpoint, or a bigger model cut down.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"a": "Distillation needs the teacher at every training step, so it doubles the cost",
                 "b": "The teacher and student must use the same tokenizer for token-level distillation",
                 "c": "A pruned model needs no further training"},
        label="**Check yourself:** which is true of this module's recipes?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Here the teacher ran once, in advance; its top-16 predictions were stored and reused "
                              "every epoch."), kind="warn"),
        "b": mo.callout(mo.md("Right — the loss compares probabilities token by token, so both must predict over "
                              "the same vocabulary. That's why the student here uses SmolLM2's tokenizer, not the "
                              "8K one from Modules 16 and 40."), kind="success"),
        "c": mo.callout(mo.md("Look at the 'before healing' row: cutting half the network hurts a lot until a "
                              "short retraining repairs it."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(42))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Keller Jordan, [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — the GPT-2 speedrun:
      every record's tricks, from Muon to value embeddings, with the code.
    * Hugging Face, [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
      — how the SmolLM team plans, runs and debugs small-model training.
    * Andrej Karpathy, [nanochat](https://github.com/karpathy/nanochat) — a full ChatGPT-style pipeline
      (tokenizer, pretraining, fine-tuning, RL) sized for one node and a few hundred dollars.

    **Next:** Part 12 continues (planned): vision transformers to vision-language models, efficient and
    sparse attention, and diffusion language models.
    """)
    return


if __name__ == "__main__":
    app.run()
