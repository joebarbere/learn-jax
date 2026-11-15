import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="21 · Evaluation")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import re
    import time

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from tokenizers import Tokenizer

    import learnjax as lj
    from learnjax import llama
    from learnjax.data import detective_corpus, tiny_shakespeare

    ex = lj.load_exercises(mo.notebook_dir())
    _spec = importlib.util.spec_from_file_location("m21_scoring", mo.notebook_dir() / "scoring.py")
    S = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(S)
    return (S, Tokenizer, detective_corpus, ex, jax, jnp, lj, llama, mo, np, plt, re, time,
            tiny_shakespeare)


@app.cell
def _(mo):
    mo.md(r"""
    # 21 · Evaluating language models

    **Goal:** produce evaluation numbers you'd bet on — and recognize the ones you shouldn't.

    **You will:** compute perplexity properly with sliding windows · learn why perplexities from different
    tokenizers can't be compared, and fix it with **bits per byte** · score a real benchmark (ARC-Easy) by
    log-likelihood, three ways · put **confidence intervals** on accuracies and test whether one model
    really beats another · watch a prompt template move accuracy by ~8 points · detect benchmark
    **contamination** with n-gram overlap · see exact-match scoring give a misleading 0% · catch an
    **LLM judge** choosing by position instead of content.

    **Models:** SmolLM2-135M and SmolLM2-360M (base) plus SmolLM2-360M-Instruct, all running in
    `learnjax.llama` on your GPU — and, if you did the long run in Module 16, *your* detective-fiction GPT.
    The model plumbing (batched, fixed-shape log-prob scoring) is in `scoring.py` next to this notebook.
    Your exercises turn those raw log-probs into numbers.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Benchmarks have steered the field: ImageNet (2009) for vision, and perplexity for language since the
    n-gram era. As models grew, single scores gave way to suites: GLUE (2018), MMLU (Hendrycks et al., 2020)
    and HELM (Liang et al., 2022). ARC (Clark et al., 2018) is the science-question benchmark used here.
    Zheng et al. (2023) studied using a strong model as a judge, alongside the crowd-voted Chatbot Arena.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · What a single number hides

    "Model X scores 71.0%" needs four more facts before it means anything:

    | question | why it matters (all measured below) |
    |---|---|
    | on what data, scored how? | sum vs length-normalized log-probs change rankings |
    | ± what? | 300 items gives roughly ±5 points of pure sampling noise |
    | with which prompt? | three phrasings of the same questions: an 8-point spread |
    | had the model seen the test? | contamination inflates scores silently |

    Everything in this module is one of those four questions, made concrete.
    """)
    return


@app.cell
def _(S, llama, mo):
    _t = mo.status.spinner(title="Loading three SmolLM2 models (first run downloads ~1.7 GB)…")
    with _t:
        models = {
            "SmolLM2-135M": S.load_model(llama.SMOLLM_135M),
            "SmolLM2-360M": S.load_model("HuggingFaceTB/SmolLM2-360M"),
            "SmolLM2-360M-Instruct": S.load_model(llama.SMOLLM_360M),
        }
    tok = models["SmolLM2-135M"][2]  # all three share one 49,152-token tokenizer
    _rows = "\n".join(f"| {k} | {llama.count_params(p) / 1e6:.0f}M | {c.n_layers} | {c.n_heads} / {c.n_kv_heads} |"
                      for k, (p, c, _) in models.items())
    mo.md(f"""
| model | params | layers | query / KV heads |
|---|---|---|---|
{_rows}

The two 360M models share an architecture and pre-training; the Instruct one was then fine-tuned to
chat. Keeping the base model lets us separate the effect of **size** from the effect of **tuning**.
    """)
    return models, tok


@app.cell
def _(S, mo):
    gpt16 = S.load_gpt16()
    if gpt16 is None:
        _msg = mo.callout(mo.md(
            "No Module 16 long run found (`runs/16-gpt/params.npz`). The tables below will skip *your* GPT. "
            "To include it: `containers/run.sh python modules/16-gpt/train.py` (~15 min), then re-run."),
            kind="info")
    else:
        _p, _c, _m, _src = gpt16
        _msg = mo.md(f"Found your Module 16 GPT (`runs/16-gpt`): d_model={_c.d_model}, {_c.n_layers} layers, "
                     f"8,192-token detective tokenizer — loaded with {_src}.")
    _msg
    return (gpt16,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Perplexity done right: sliding windows

    Perplexity is $\exp(\text{mean negative log-likelihood per token})$ — the effective number of tokens
    the model is choosing between at each step (Module 04). On a long text, the model has a maximum
    context, so you score it in windows. The naive way chops the text into non-overlapping chunks: the
    first tokens of every chunk are predicted from almost nothing, which makes the model look worse than
    it is. The honest way **slides** the window by a smaller stride and scores only the *new* tokens of
    each window, so every token gets plenty of context.

    ```
    stride = window:  [=========][=========][=========]      each chunk starts cold
    stride < window:  [=========]
                           [====|====]                       score only the part after "|"
                                [====|====]
    ```

    **Your turn:** `sliding_window_nll` in `exercises.py`. Below it drives SmolLM2-360M over 12,000 tokens
    of held-out detective fiction (the Module 16 validation split, decoded back to text).
    """)
    return


@app.cell
def _(Tokenizer, detective_corpus, tiny_shakespeare, tok):
    _d = detective_corpus()
    det_tok = Tokenizer.from_file(str(_d["tokenizer_path"]))
    det_ids16 = _d["val"][:12000]
    texts = {
        "detective (held out)": det_tok.decode(det_ids16.tolist()),
        "Shakespeare": tiny_shakespeare()[:40000],
    }
    smol_ids = {k: tok.encode(v).ids for k, v in texts.items()}
    return det_ids16, det_tok, smol_ids, texts


@app.cell
def _(S, ex, lj, mo, models, np, smol_ids, time):
    _p, _c, _ = models["SmolLM2-360M"]
    _ids = smol_ids["detective (held out)"]
    _rows, _first = [], None
    for _stride in (512, 256, 128):
        _t0 = time.perf_counter()
        _a = lj.attempt(ex.sliding_window_nll, S.window_logprob_fn(_p, _c, 512), _ids, 512, _stride)
        if not _a.ok:
            _first = _a
            break
        _nll, _n = _a.value
        _rows.append(f"| 512 | {_stride} | {_n:,} | {np.exp(_nll / _n):.2f} | {time.perf_counter() - _t0:.1f} s |")
    if _first is not None:
        _out = _first.view(mo)
    else:
        _out = mo.md(f"""
| window | stride | tokens scored | perplexity / token | time |
|---|---|---|---|---|
{chr(10).join(_rows)}

Same model, same text: perplexity drops as the stride shrinks, purely because tokens see more context.
Stride = window also skips the first token of every chunk (it has no context at all), so it even scores
fewer tokens. Halving the stride doubles the compute; past a point the gain flattens. **Always report
the window and stride** with a perplexity.
        """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Different tokenizers, incomparable perplexities → bits per byte

    Perplexity is *per token*, and a token means different things to different models. Your Module 16 GPT
    splits text into 8,192 possible tokens; SmolLM2 into 49,152. Bigger vocabulary → fewer, rarer tokens
    per sentence → each prediction is harder, so per-token perplexity rises even if the model is better.
    Comparing them is comparing prices in yen and dollars.

    The fix: divide the *total* surprise by something every model agrees on — the text's length in
    **bytes** — and measure it in bits:

    $$\text{BPB} = \frac{\text{total NLL in nats}}{\text{bytes} \times \ln 2}$$

    1 bit per byte means the model squeezes the text to 1/8 of its UTF-8 size. Below, every model scores the
    *same two strings*; window 512 / stride 256 for SmolLM2, window 256 / stride 128 for your GPT (its
    context length). Total NLL is scaled up to the whole text from the tokens scored (all but the first).
    """)
    return


@app.cell
def _(S, det_ids16, det_tok, ex, gpt16, lj, mo, models, np, plt, smol_ids, texts, tok):
    _rows, _bars, _todo = [], {}, None
    _texts = list(texts)
    for _name, (_p, _c, _) in models.items():
        _vals = []
        for _tx in _texts:
            _ids = smol_ids[_tx]
            _a = lj.attempt(ex.sliding_window_nll, S.window_logprob_fn(_p, _c, 512), _ids, 512, 256)
            _b = lj.attempt(ex.bits_per_byte, (_a.value[0] * (len(_ids) - 1) / _a.value[1]) if _a.ok else 0.0,
                            len(texts[_tx].encode("utf-8")))
            if not (_a.ok and _b.ok):
                _todo = _a if not _a.ok else _b
                break
            _vals.append((len(_ids), np.exp(_a.value[0] / _a.value[1]), _b.value))
        if _todo is not None:
            break
        _bars[_name] = [v[2] for v in _vals]
        _rows.append(f"| {_name} | {tok.get_vocab_size():,} | " + " | ".join(
            f"{n:,} tok · ppl {ppl:.1f} · **{bpb:.3f}**" for n, ppl, bpb in _vals) + " |")
    if _todo is None and gpt16 is not None:
        _gp, _gc, _gm, _ = gpt16
        _fn = S.gpt16_window_logprob_fn(_gp, _gc, _gm)
        _vals = []
        for _tx in _texts:
            _ids = det_ids16 if _tx.startswith("detective") else np.array(det_tok.encode(texts[_tx]).ids)
            _nll, _n = ex.sliding_window_nll(_fn, _ids, 256, 128)
            _bpb = ex.bits_per_byte(_nll * (len(_ids) - 1) / _n, len(texts[_tx].encode("utf-8")))
            _vals.append((len(_ids), np.exp(_nll / _n), _bpb))
        _bars["your Module 16 GPT"] = [v[2] for v in _vals]
        _rows.append("| your Module 16 GPT | 8,192 | " + " | ".join(
            f"{n:,} tok · ppl {ppl:.1f} · **{bpb:.3f}**" for n, ppl, bpb in _vals) + " |")
    if _todo is not None:
        _out = _todo.view(mo)
    else:
        _fig, _ax = plt.subplots(figsize=(7, 3))
        _w = 0.8 / len(_bars)
        for _i, (_k, _v) in enumerate(_bars.items()):
            _ax.bar(np.arange(len(_texts)) + _i * _w, _v, _w, label=_k)
        _ax.set_xticks(np.arange(len(_texts)) + 0.4 - _w / 2, _texts)
        _ax.set_ylabel("bits per byte (lower = better)")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _out = mo.vstack([mo.md(
            "| model | vocab | " + " | ".join(_texts) + " |\n|---|---|" + "---|" * len(_texts) + "\n"
            + "\n".join(_rows)), _fig])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reading the table.** Per-token perplexities can't be lined up across tokenizers (look how different
    the token counts are for the same string); bits per byte can. Things to notice:

    * **Size helps everywhere:** 360M beats 135M on both texts.
    * **Domain shift:** your GPT only ever saw detective novels. On held-out detective text it is within
      ~0.3 bits/byte of models 10–26× its size trained on trillions of tokens; on Shakespeare it falls apart
      (more than double its in-domain BPB), while SmolLM2 gets worse by only ~0.3. In-domain specialists can
      be cheap; generalists are robust.
    * **Per-token numbers use different units:** your GPT cuts the detective text into 12,000 tokens,
      SmolLM2 into ~11,000 — so their "ppl" columns measure surprise per *different* unit. Only the BPB
      column lines up.
    * **A caveat you'll meet again in Section 7:** these novels are public-domain Project Gutenberg books.
      SmolLM2's web-scale training data may well contain them, so its "held-out" score may not be held out
      at all.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Multiple-choice benchmarks, scored by log-likelihood

    **ARC-Easy** (AI2 Reasoning Challenge, grade-school science; CC BY-SA 4.0) asks questions with four
    answers. A base model can't "pick B", but it *can* tell you how likely each answer is as a continuation:

    ```
    (illustrative numbers)
    Question: Which form of energy is produced when a rubber band vibrates?
    Answer: chemical      → log p = −9.8
    Answer: light         → log p = −8.1
    Answer: electrical    → log p = −9.6
    Answer: sound         → log p = −3.2   ← highest: the model's choice
    ```

    The catch is **length**: a sum of log-probs over more tokens is more negative, so long answers are
    penalized. Common fixes divide by the number of tokens, or by the number of bytes (tokenizer-free, as in
    `lm-evaluation-harness`'s `acc_norm`). Which is "right" depends on the benchmark — so measure.

    **Your turn:** `choice_logprob` and `mc_accuracy`. Below: 300 random ARC-Easy test questions (1,200
    scored continuations per model), all three models, all three scoring rules.
    """)
    return


@app.cell
def _(S, mo, np):
    arc = S.arc_items(300, seed=0)
    arc_labels = np.array([it["label"] for it in arc])
    _it = arc[3]
    mo.md("**Example item:** " + _it["question"] + "\n\n" + "\n".join(
        f"* {'**' if i == _it['label'] else ''}{c}{' ✓**' if i == _it['label'] else ''}"
        for i, c in enumerate(_it["choices"])))
    return arc, arc_labels


@app.cell
def _(S, arc, mo, models, tok):
    # Model compute only (no exercises): per-token log-probs for every (item, choice), per model.
    with mo.status.spinner(title="Scoring 1,200 continuations × 3 models…"):
        mc = {}
        for _name, (_p, _c, _) in models.items():
            _ctx, _cont, _nb = S.build_mc(tok, arc, "Question/Answer")
            _lp, _mask = S.score_continuations(_p, _c, _ctx, _cont, length=192, batch=16)
            mc[_name] = (_lp, _mask, _nb)
    return (mc,)


@app.cell
def _(arc, ex, lj, mc, mo, np):
    _lp, _mask, _nb = mc["SmolLM2-360M"]
    _sum = lj.attempt(ex.choice_logprob, _lp, _mask)
    _norm = lj.attempt(ex.choice_logprob, _lp, _mask, True)
    if not (_sum.ok and _norm.ok):
        _out = (_sum if not _sum.ok else _norm).view(mo)
    else:
        _s, _n = _sum.value.reshape(-1, 4), _norm.value.reshape(-1, 4)
        _ntok = _mask.sum(-1).reshape(-1, 4)
        _disagree = np.nonzero(_s.argmax(-1) != _n.argmax(-1))[0]
        _i = int(_disagree[0]) if len(_disagree) else 0
        _it = arc[_i]
        _rows = "\n".join(
            f"| {'✓ ' if j == _it['label'] else ''}{c} | {_ntok[_i, j]} | {_s[_i, j]:.2f}"
            f"{' ◀' if j == _s[_i].argmax() else ''} | {_n[_i, j]:.2f}{' ◀' if j == _n[_i].argmax() else ''} |"
            for j, c in enumerate(_it["choices"]))
        _out = mo.md(f"""
**Where the rules disagree** — SmolLM2-360M on: *{_it['question']}*
({len(_disagree)} of {len(arc)} items flip between the two rules)

| choice | tokens | sum log p | mean log p / token |
|---|---|---|---|
{_rows}
        """)
    _out
    return


@app.cell
def _(arc_labels, ex, lj, mc, mo):
    _rows, _todo = [], None
    mc_correct = {}
    for _name, (_lp, _mask, _nb) in mc.items():
        _s = lj.attempt(ex.choice_logprob, _lp, _mask)
        _n = lj.attempt(ex.choice_logprob, _lp, _mask, True)
        if not (_s.ok and _n.ok):
            _todo = _s if not _s.ok else _n
            break
        _scores = {"sum": _s.value, "per token": _n.value, "per byte": _s.value / _nb}
        _acc = {k: lj.attempt(ex.mc_accuracy, v.reshape(-1, 4), arc_labels) for k, v in _scores.items()}
        if not all(a.ok for a in _acc.values()):
            _todo = next(a for a in _acc.values() if not a.ok)
            break
        mc_correct[_name] = _scores["sum"].reshape(-1, 4).argmax(-1) == arc_labels
        _rows.append(f"| {_name} | " + " | ".join(f"{100 * a.value:.1f}%" for a in _acc.values()) + " |")
    _out = _todo.view(mo) if _todo is not None else mo.md(f"""
| model | sum | per token | per byte |
|---|---|---|---|
{chr(10).join(_rows)}

Chance is 25%. Two surprises worth a minute each: on ARC-Easy the plain **sum** wins (normalizing helps
on benchmarks like HellaSwag whose wrong answers are systematically longer — it depends on the data), and
the **Instruct** model scores *below* its own base model. Is that real, or noise? Next section.
    """)
    _out
    return (mc_correct,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Is the difference real? Bootstrap confidence intervals

    300 questions is a sample. Had AI2 written a different 300, the accuracy would differ. The
    **bootstrap** estimates how much, with no formulas: resample the 300 items *with replacement* a few
    thousand times, recompute accuracy each time, and read off the middle 95% of those accuracies.

    To compare two models, don't compare their two intervals — they're evaluated on the **same** items, so
    resample *paired*: draw item indices once, score both models on them, look at the difference. Shared
    easy and hard items cancel out, giving a much tighter answer.

    **Your turn:** `bootstrap_ci` and `paired_bootstrap_diff`.
    """)
    return


@app.cell
def _(ex, lj, mc_correct, mo, np, plt):
    _ci = {k: lj.attempt(ex.bootstrap_ci, v, 4000, 0) for k, v in mc_correct.items()}
    _pairs = [("SmolLM2-135M", "SmolLM2-360M"), ("SmolLM2-360M", "SmolLM2-360M-Instruct")]
    _d = {p: lj.attempt(ex.paired_bootstrap_diff, mc_correct[p[0]], mc_correct[p[1]], 4000, 0)
          for p in _pairs if p[0] in mc_correct and p[1] in mc_correct}
    if not mc_correct:
        _out = mo.md("*(Finish `choice_logprob` and `mc_accuracy` first.)*")
    elif not all(a.ok for a in _ci.values()):
        _out = next(a for a in _ci.values() if not a.ok).view(mo)
    else:
        _fig, _ax = plt.subplots(figsize=(6, 2.4))
        for _i, (_k, _a) in enumerate(_ci.items()):
            _m, _lo, _hi = _a.value
            _ax.errorbar(100 * _m, _i, xerr=[[100 * (_m - _lo)], [100 * (_hi - _m)]], fmt="o", capsize=4)
        _ax.set_yticks(range(len(_ci)), list(_ci))
        _ax.set_xlabel("ARC-Easy accuracy, % (95% bootstrap CI)")
        _ax.axvline(25, color="gray", lw=0.8, ls=":")
        _fig.tight_layout()
        _rows = []
        for (_a_name, _b_name), _r in _d.items():
            if not _r.ok:
                _rows = None
                _todo = _r
                break
            _diff, _lo, _hi, _p = _r.value
            _verdict = "real" if (_lo > 0 or _hi < 0) else "could be noise"
            _rows.append(f"| {_a_name} → {_b_name} | {100 * _diff:+.1f} | [{100 * _lo:+.1f}, {100 * _hi:+.1f}] | "
                         f"{_p:.3f} | {_verdict} |")
        _tbl = (_todo.view(mo) if _rows is None else mo.md(
            "| change | Δ accuracy (pts) | 95% CI | P(Δ ≤ 0) | verdict |\n|---|---|---|---|---|\n" + "\n".join(_rows)))
        _w = np.mean([100 * (a.value[2] - a.value[1]) / 2 for a in _ci.values()])
        _out = mo.vstack([_fig, mo.md(f"Each interval is about ±{_w:.0f} points wide — the noise floor of a "
                                      "300-item benchmark."), _tbl])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Both changes are real: **2.7× more parameters buys ~10 points**, and **chat-tuning costs ~15 points —
    under this scoring method**. The Instruct model wasn't made dumber; it was trained to answer inside a
    chat template, so raw `Question: … Answer: …` text continuations are no longer what it puts
    probability on. That's why leaderboards score base models by log-likelihood and chat models by
    *generating* answers in their own format — and why "model A beats model B" claims need the scoring
    method attached.
    """)
    return


@app.cell
def _(S, arc, arc_labels, ex, lj, mo, models, np, plt, tok):
    _p, _c, _ = models["SmolLM2-360M"]
    _res, _todo = {}, None
    with mo.status.spinner(title="Re-scoring with two more templates…"):
        for _t in S.TEMPLATES:
            _ctx, _cont, _nb = S.build_mc(tok, arc, _t)
            _lp, _mask = S.score_continuations(_p, _c, _ctx, _cont, length=192, batch=16)
            _s = lj.attempt(ex.choice_logprob, _lp, _mask)
            _n = lj.attempt(ex.choice_logprob, _lp, _mask, True)
            if not (_s.ok and _n.ok):
                _todo = _s if not _s.ok else _n
                break
            _res[_t] = (np.mean(_s.value.reshape(-1, 4).argmax(-1) == arc_labels),
                        np.mean(_n.value.reshape(-1, 4).argmax(-1) == arc_labels))
    if _todo is not None:
        _out = _todo.view(mo)
    else:
        _fig, _ax = plt.subplots(figsize=(6.5, 2.6))
        _x = np.arange(len(_res))
        _ax.bar(_x - 0.2, [100 * v[0] for v in _res.values()], 0.4, label="sum")
        _ax.bar(_x + 0.2, [100 * v[1] for v in _res.values()], 0.4, label="per token")
        _ax.set_xticks(_x, list(_res))
        _ax.set_ylim(50, 80)
        _ax.set_ylabel("accuracy %")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _rows = "\n".join(f"| {k} | `{S.TEMPLATES[k][0]!r}` | {100 * a:.1f}% | {100 * b:.1f}% |"
                          for k, (a, b) in _res.items())
        _spread = 100 * (max(v[0] for v in _res.values()) - min(v[0] for v in _res.values()))
        _out = mo.vstack([mo.md(f"""
## 6 · Prompt format is part of the result

Same model (SmolLM2-360M base), same 300 questions, same answers — only the text around them changes.

| template | context | sum | per token |
|---|---|---|---|
{_rows}

An **{_spread:.0f}-point** spread from wording alone — larger than the ±5-point sampling noise, and close to
the gain from 2.7× more parameters. Reported benchmark numbers bake in a specific template; reproduce a
number only with the same one.
        """), _fig])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Contamination: did the model see the test?

    Web-scale training sets contain copies of popular benchmarks, and a model that memorized a test item
    isn't reasoning about it. You can't usually inspect a model's training data, but when you *can* — your
    own models, open datasets — the standard check is **n-gram overlap**: what fraction of a test item's
    word sequences (here 8 words long) appear verbatim in the training text? Long n-grams rarely match by
    chance, so high overlap means a copy.

    **Your turn:** `ngram_overlap`. The demo below builds three "training corpora" — detective fiction,
    ARC-Easy's own *training* split, and detective fiction with 15 test items quietly pasted in — and scans
    all 300 test items against each. (The scan uses a precomputed set of the corpus's n-grams — the same
    idea as your function, built once instead of 300 times.)
    """)
    return


@app.cell
def _(S, Tokenizer, arc, detective_corpus, ex, lj, mo, np, re):
    def _words(s):
        return re.findall(r"[a-z0-9]+", s.lower())

    _items = [_words(it["question"] + " " + it["choices"][it["label"]]) for it in arc]
    _d = detective_corpus()
    _det = _words(Tokenizer.from_file(str(_d["tokenizer_path"])).decode(_d["train"][:400_000].tolist()))
    _train = _words(" ".join(x["question"] + " " + " ".join(x["choices"]) for x in S.arc_split("train")))
    _leak_idx = set(np.random.default_rng(7).choice(len(arc), 15, replace=False).tolist())
    _leaky = list(_det)
    for _i in sorted(_leak_idx):
        _pos = (_i * 9973) % len(_leaky)
        _leaky[_pos:_pos] = _items[_i]
    _n = 8

    def _overlaps(corpus):
        grams = {tuple(corpus[i:i + _n]) for i in range(len(corpus) - _n + 1)}
        out = []
        for w in _items:
            g = [tuple(w[i:i + _n]) for i in range(len(w) - _n + 1)]
            out.append(sum(x in grams for x in g) / len(g) if g else 0.0)
        return np.array(out)

    _check = lj.attempt(ex.ngram_overlap, _items[0], _items[0] + ["x"], _n)
    if not _check.ok:
        _out = _check.view(mo)
    else:
        _fast = _overlaps(_train)
        _agree = all(abs(ex.ngram_overlap(_items[i], _train, _n) - _fast[i]) < 1e-9 for i in range(5))
        _rows = []
        for _name, _corp in [("detective fiction (400K tokens)", _det), ("ARC-Easy train split", _train),
                             ("detective + 15 pasted test items", _leaky)]:
            _o = _overlaps(_corp)
            _flag = np.nonzero(_o >= 0.5)[0]
            _hit = len(set(_flag.tolist()) & _leak_idx) if "pasted" in _name else None
            _rows.append(f"| {_name} | {len(_corp):,} | {np.mean(_o > 0):.1%} | {len(_flag)}"
                         + (f" (all {_hit} planted items found)" if _hit == len(_leak_idx) else
                            (f" ({_hit} of 15 planted)" if _hit is not None else "")) + " |")
        _out = mo.md(f"""
| training corpus | words | items with any 8-gram match | items flagged (≥ 50% overlap) |
|---|---|---|---|
{chr(10).join(_rows)}

Your `ngram_overlap` agrees with the fast scan: **{_agree}**. Novels share no 8-word runs with science
questions; ARC's own training split shares stock phrases (*"which of the following best …"*) with a few
test items, and even one near-duplicate — splits of the same benchmark aren't perfectly disjoint; the
planted items light up at ~100% overlap. Real audits (GPT-3's, Llama's)
use exactly this with 8–13-grams, plus fuzzier variants that survive reformatting — which this exact-match
check would miss.
        """)
    _out
    return


@app.cell
def _(S, mo, models, re, tok):
    _problems = [(12, 7), (25, 17), (8, 9), (31, 44), (56, 23), (19, 19), (70, 15), (3, 98), (41, 41), (66, 27),
                 (14, 28), (90, 5), (33, 33), (47, 16), (11, 89), (62, 29), (5, 7), (38, 45), (24, 76), (58, 13)]
    _p, _c, _ = models["SmolLM2-360M-Instruct"]
    _prompts = [S.llama.chat_prompt(tok, f"What is {a} + {b}? Answer with just the number.") for a, b in _problems]
    _end = tok.token_to_id("<|im_end|>")
    _outs = [tok.decode(o[: o.index(_end)] if _end in o else o).strip()
             for o in S.greedy_generate(_p, _c, _prompts, n_new=10)]
    _strict = [o == str(a + b) for o, (a, b) in zip(_outs, _problems, strict=True)]
    _nums = [re.findall(r"-?\d+", o) for o in _outs]
    _lenient = [bool(n) and n[-1] == str(a + b) for n, (a, b) in zip(_nums, _problems, strict=True)]
    _ex_rows = "\n".join(f"| {a} + {b} | `{o}` | {'✓' if s else '✗'} | {'✓' if l else '✗'} |"
                         for (a, b), o, s, l in list(zip(_problems, _outs, _strict, _lenient, strict=True))[:6])
    mo.md(f"""
## 8 · Generative evaluation: the grader is part of the model

Chat models are evaluated by letting them *answer* and checking the answer. Twenty two-digit additions,
SmolLM2-360M-Instruct, greedy decoding (`scoring.greedy_generate`, a fixed-shape loop with no KV cache —
Module 18 makes this fast):

| problem | model output | exact match | last number |
|---|---|---|---|
{_ex_rows}
| … | | | |
| **all 20** | | **{sum(_strict)}/20** | **{sum(_lenient)}/20** |

The model mostly gets the sum right, but restates the problem ("12 + 7 = 19") despite being told to
answer with just the number. **Exact match** scores that as wrong; extracting the **last number** scores it
right. Neither is wrong — they measure different things (format-following vs arithmetic) — but a paper
reporting one without saying which is reporting the grader as much as the model.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · LLM-as-judge, and position bias

    For open-ended answers there is often no string to match, so people ask a (stronger) model to judge:
    *"Here are answers A and B — which is better?"* It's cheap and scales, but judges have biases. The
    simplest test: show every pair **twice, with the order swapped**. A judge reasoning about content picks
    the same *answer* both times, so its *letter* flips. A judge that picks the same letter both times is
    choosing by position.

    Below, SmolLM2-360M-Instruct judges 20 factual questions, each with one right and one wrong answer.
    We read its verdict from the next-token logits of "A" vs "B" (no sampling, so it's deterministic).

    **Your turn:** `position_bias_rate`.
    """)
    return


@app.cell
def _(S, ex, lj, mo, models, np, tok):
    _pairs = [
        ("What is the capital of France?", "Paris", "Lyon"),
        ("How many legs does a spider have?", "Eight", "Six"),
        ("What gas do plants absorb from the air for photosynthesis?", "Carbon dioxide", "Oxygen"),
        ("What is 9 times 7?", "63", "56"),
        ("Which planet is closest to the Sun?", "Mercury", "Venus"),
        ("What is the boiling point of water at sea level in Celsius?", "100 degrees", "90 degrees"),
        ("Who wrote 'Romeo and Juliet'?", "William Shakespeare", "Charles Dickens"),
        ("What is the largest ocean on Earth?", "The Pacific Ocean", "The Atlantic Ocean"),
        ("How many days are in a leap year?", "366", "365"),
        ("What is the chemical symbol for gold?", "Au", "Ag"),
        ("What is the square root of 81?", "9", "8"),
        ("Which organ pumps blood through the body?", "The heart", "The liver"),
        ("What is the freezing point of water in Fahrenheit?", "32 degrees", "0 degrees"),
        ("How many continents are there?", "Seven", "Five"),
        ("What is the main language spoken in Brazil?", "Portuguese", "Spanish"),
        ("Which is the largest planet in our solar system?", "Jupiter", "Saturn"),
        ("What do bees make?", "Honey", "Milk"),
        ("How many sides does a hexagon have?", "Six", "Eight"),
        ("What is H2O commonly called?", "Water", "Salt"),
        ("In which year did World War II end?", "1945", "1918"),
    ]
    _p, _c, _ = models["SmolLM2-360M-Instruct"]

    def _prompt(q, a, b):
        return S.llama.chat_prompt(tok, f"Question: {q}\n\nAnswer A: {a}\nAnswer B: {b}\n\n"
                                        "Which answer is correct? Reply with just the letter A or B.")

    _A, _B = tok.encode("A").ids[0], tok.encode("B").ids[0]
    _logits = S.next_token_logits(_p, _c, [_prompt(q, g, w) for q, g, w in _pairs]
                                  + [_prompt(q, w, g) for q, g, w in _pairs])
    _v = np.where(_logits[:, _A] >= _logits[:, _B], "A", "B")
    _ab, _ba = _v[: len(_pairs)].tolist(), _v[len(_pairs):].tolist()
    _rate = lj.attempt(ex.position_bias_rate, _ab, _ba)
    if S.server_reachable():
        _server = ("A llama.cpp server from Module 19 is reachable at `http://llama:8080` — a much stronger "
                   "judge. Re-run this same both-orders test against it.")
    else:
        _server = ("A stronger local judge — the llama.cpp server from Module 19 (reachable at `http://llama:8080` "
                   "when the course container joins `learn-jax-net`) — is worth re-running this test on.")
    _acc_ab = np.mean(np.array(_ab) == "A")   # right answer shown first
    _acc_ba = np.mean(np.array(_ba) == "B")   # right answer shown second
    if not _rate.ok:
        _out = _rate.view(mo)
    else:
        _out = mo.md(f"""
| | right answer shown as A | right answer shown as B |
|---|---|---|
| judge picks the right answer | {_acc_ab:.0%} | {_acc_ba:.0%} |
| judge says "A" | {np.mean(np.array(_ab) == 'A'):.0%} | {np.mean(np.array(_ba) == 'A'):.0%} |

**Position-bias rate: {_rate.value:.0%}** of items got the same letter both times. This small judge's
verdict is decided almost entirely by order: it looks perfect when the right answer happens to be first
and useless when it's second — averaged over one ordering, you'd have concluded anything you liked.
Bigger judges are far better but not immune (MT-Bench measured position bias in GPT-4 too). Defenses:
always evaluate both orders and count only consistent verdicts, randomize order, and spot-check by hand.
{_server}
        """)
    _out
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Model A, because it scores 2 points higher": "a",
            "Can't tell yet: compute a paired bootstrap CI of the difference on the shared items": "b",
            "Model B, because instruct models are always better": "c",
        },
        label="**Check yourself:** on the same 300 questions, model A scores 66% and model B 64%. Which is better?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Not yet — on 300 items the sampling noise alone is about ±5 points. A 2-point gap "
                              "is well inside it unless a *paired* test says otherwise."), kind="warn"),
        "b": mo.callout(mo.md("Right. Paired, because both models answered the same items — the shared easy and "
                              "hard items cancel and the test gets much sharper than comparing two separate "
                              "intervals."), kind="success"),
        "c": mo.callout(mo.md("Section 5 showed the opposite under log-likelihood scoring: the Instruct model "
                              "lost 15 points to its own base. 'Better' depends on how you score."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(21))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * EleutherAI, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the
      standard open implementation of everything in Sections 4–6 (`acc` vs `acc_norm` is our sum vs per-byte)
    * Hugging Face, [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)
      — the sliding-window recipe from Section 2
    * Biderman et al., [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782)
      (2024) — prompt sensitivity, scoring choices and reporting, from the harness's authors
    * Clark et al., [Think you have Solved Question Answering? Try ARC](https://arxiv.org/abs/1803.05457)
      (2018) — the benchmark used here

    **Next:** Module 22 — profiling JAX on ROCm: where the time in all those forward passes actually goes.
    """)
    return


if __name__ == "__main__":
    app.run()
