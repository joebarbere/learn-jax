import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="49 · Continual learning")


@app.cell
def _():
    import marimo as mo
    import sys

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import rosetta

    ex = lj.load_exercises(mo.notebook_dir())
    sys.path.insert(0, str(mo.notebook_dir()))
    import facts as F
    import models as M
    return F, M, ex, jax, jnp, lj, mo, np, plt, rosetta


@app.cell
def _(M):
    # results written by train.py (None until that arm has run)
    ARMS = ("icl", "sft", "replay", "ewc", "expert_one", "expert_new", "expert_new_retain", "distill", "distill_retain",
            "bio_text", "dream", "dream_select")
    res = {a: M.read_json(M.RUNS / a / "results.json") for a in ARMS}
    base = M.read_json(M.RUNS / "base.json")
    have = [a for a in ARMS if res[a] is not None]
    return ARMS, base, have, res


@app.cell
def _(mo):
    mo.md(r"""
    # 49 · Continual learning: forgetting, consolidation and sleep (1989–2026)

    **Take first:** Module 20 (fine-tuning and LoRA), Module 40 (mixture of experts) and Module 42 (distillation).

    **Goal:** a language model's knowledge stops at its training cutoff. You can tell it new things in the prompt,
    and it uses them well, until the prompt is gone. You can fine-tune the new things in, and it learns them,
    and quietly loses some of what it knew. This module measures that trade on a real model, then tries the fixes
    the field has proposed, from 1989's diagnosis to 2026's "language models need sleep", and checks which ones
    actually help at this scale.

    **You will:** invent 200 people the model can't know about and teach them over four "days" · measure
    forgetting with an accuracy matrix · protect old knowledge with replay and elastic weight consolidation ·
    add new low-rank experts instead of overwriting old weights · let the model teach itself with its notes in
    the prompt (self-distillation, the core of the Sleep paper) · let it write its own practice questions
    ("dreams") · and score everything on the same board.
    """)
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **Where this came from**

    In **1989** McCloskey and Cohen trained a network on one set of arithmetic facts and then a second, and watched
    it lose the first set almost completely; they called it **catastrophic interference**. In **1995**
    McClelland, McNaughton and O'Reilly explained why brains avoid it: a fast learner (the hippocampus) stores new
    experiences, and during sleep it **replays** them to a slow learner (the neocortex), which absorbs them
    gradually between older memories. Deep learning met the problem again at scale: **Kirkpatrick et al. (2017)**
    anchored the weights an old task needs (elastic weight consolidation), and Lopez-Paz and Ranzato (2017) gave
    the accuracy matrix used to score it. For language models, **SEAL** (Zweiger et al., 2025) had the model write
    its own training data, **SDFT** (Shenfeld et al., 2026) had it teach itself with a demonstration in its prompt,
    and **Behrouz et al. (2026)** combined self-distillation, new experts and self-generated "dreams" into a
    **sleep** phase between periods of use.

    *Timeline: Module 00b.*
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Two memories: the prompt and the weights

    A chat model has two places to keep what it knows. Its **weights** hold everything from training, and they
    don't change when you talk to it. Its **context** (the prompt so far) holds what you just told it: reading it is
    *in-context learning* (Module 20), and it is very good at that, but it's gone when the conversation ends.
    The Sleep paper's analogy is a patient with anterograde amnesia: old memories intact, the present held only for
    a moment, nothing new ever moving from one to the other.

    *Continual learning* is the problem of moving knowledge from the first kind of memory to the second without
    breaking what is already there. Below, the model meets one invented person, first with nothing in its prompt
    and then with the person's short biography in front of it.
    """)
    return


@app.cell
def _(F, M):
    params, cfg, tok = M.load()
    world = F.make_world(n_days=4, per_day=50, seed=0)
    return cfg, params, tok, world


@app.cell
def _(F, M, cfg, mo, params, tok, world):
    _p = world.days[0][0]
    _qa = [(F.TEMPLATES[a][3].format(name=_p.name), _p.get(a), _p, a) for a in F.ATTRIBUTES]
    _stop = M.stop_ids(tok)
    _closed = M.decode(cfg, params, [M.prompt_ids(tok, q) for q, *_ in _qa], stop=_stop, batch=8)
    _open = M.decode(cfg, params, [M.prompt_ids(tok, q, _p.bio()) for q, *_ in _qa], P=M.P_CTX, stop=_stop, batch=8)
    _rows = "\n".join(f"| {q} | {a} | {tok.decode(M.strip_stop(c, _stop))} | {tok.decode(M.strip_stop(o, _stop))} |"
                      for (q, a, _, _), c, o in zip(_qa, _closed, _open))
    mo.md(f"""
    **The bio:** *{_p.bio()}*

    | question (worded differently from the bio) | truth | no bio in the prompt | bio in the prompt |
    |---|---|---|---|
    {_rows}
    """)
    return


@app.cell
def _(base, mo, np, res):
    if base is None or res["icl"] is None:
        _out = mo.md("*Run `train.py` for the measurements over all 200 people.*")
    else:
        _icl = 100 * np.mean([r["acc_by_day"][d] for d, r in enumerate(res["icl"]["rows"])])
        _out = mo.md(f"""
    Over all 200 people (one held-out question per fact, 1,000 questions), the untouched model gets
    **{100 * sum(base['acc_by_day']) / len(base['acc_by_day']):.1f}%** right with nothing in its prompt: lucky guesses
    of a year or a city. With each person's bio in the prompt it gets **{_icl:.1f}%**. It also knows **{len(base['world_known'])} of {base['world_total']}** capitals
    of the world, and that is the old knowledge we'll watch for damage, together with its loss on Wikipedia text
    ({base['text_nll']:.3f} nats per token).
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Facts nobody could know

    To measure learning cleanly, the facts must be new to the model and checkable by string match. So the module
    *invents* them, the way Allen-Zhu and Li ("Physics of Language Models", 2023) test knowledge storage: 200
    people with made-up names, each with five attributes drawn at random (birth year, birth city, university,
    employer, profession). The cities and universities are real, so the model knows how to *say* them; only the
    link between the person and the attribute is new.

    The people arrive in four **days** of 50. Every attribute has five question templates: three are used for
    teaching, and two are held back for testing. So a test asks about a fact the model was taught, in words it
    never saw: rote recall of the training sentence doesn't count.
    """)
    return


@app.cell
def _(F, mo, world):
    _ppl = "\n".join(f"| {p.name} | {p.birth_year} | {p.birth_city} | {p.university} | {p.employer} | {p.profession} |"
                     for p in world.days[0][:5])
    _t = "\n".join(f"| {a} | {' · '.join(F.TEMPLATES[a][:3])} | {' · '.join(F.TEMPLATES[a][3:])} |"
                   for a in F.ATTRIBUTES)
    mo.md(f"""
    | name | born | in | studied at | works for | as |
    |---|---|---|---|---|---|
    {_ppl}

    | attribute | taught with (`{{name}}` is filled in) | tested with |
    |---|---|---|
    {_t}

    The teaching data for one day is 50 people × 5 facts × 3 phrasings = 750 question/answer pairs, where the answer
    is the bio's sentence for that fact ("Varin Oskeld was born in 1971.").
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Catastrophic forgetting, measured

    The plain way to teach a model is **supervised fine-tuning** (SFT, Module 50): train every weight on the
    question/answer pairs, one day after another. Afterwards, ask about every day's people.

    Continual-learning papers summarize such a run with an **accuracy matrix**: `R[i, j]` is the accuracy on day
    `j`'s facts after training through day `i`. The diagonal is "just learned"; below it is "learned earlier, still
    remembered?"; above it is "not taught yet". Three numbers come out of it (Lopez-Paz & Ranzato, 2017; Chaudhry et
    al., 2018): the **average** of the last row (what you know at the end), **backward transfer** (how later
    training changed earlier tasks; negative = forgetting), and **forgetting** (how far each task fell from its
    best).

    **Your turn:** `forgetting_metrics(R)`.
    """)
    return


@app.cell
def _(ex, lj, mo):
    _R = [[0.9, 0.0, 0.0], [0.6, 0.8, 0.0], [0.3, 0.7, 0.9]]
    _a = lj.attempt(ex.forgetting_metrics, _R)
    mo.vstack([mo.md("A made-up run where each new day pushes out part of the last:"), _a.view(mo)])
    return


@app.cell
def _(M, np, plt, res):
    def acc_matrix(arm, key="acc_by_day"):
        r = res.get(arm)
        return None if r is None else np.array([row[key] for row in r["rows"]])

    def heat(ax, R, title):
        ax.imshow(R, vmin=0, vmax=1, cmap="viridis")
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                ax.text(j, i, f"{100 * R[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if R[i, j] < 0.6 else "black")
        ax.set_xticks(range(R.shape[1]))
        ax.set_yticks(range(R.shape[0]))
        ax.set_xlabel("facts of day j")
        ax.set_ylabel("after training day i")
        ax.set_title(title, fontsize=9)

    def heat_row(arms, titles):
        mats = [(acc_matrix(a), t) for a, t in zip(arms, titles) if acc_matrix(a) is not None]
        if not mats:
            return None
        fig, axs = plt.subplots(1, len(mats), figsize=(3.1 * len(mats), 3.0))
        for ax, (R, t) in zip(np.atleast_1d(axs), mats):
            heat(ax, R, t)
        fig.tight_layout()
        return fig

    def metrics(arm):
        R = acc_matrix(arm)
        return None if R is None else M.forgetting_metrics(R)
    return acc_matrix, heat_row, metrics


@app.cell
def _(heat_row, mo):
    _f = heat_row(["sft"], ["full fine-tuning (SFT), held-out questions"])
    _f if _f is not None else mo.md("*Run `train.py --arms sft` to see the accuracy matrix.*")
    return


@app.cell
def _(acc_matrix, base, mo, res):
    _R, _S = acc_matrix("sft"), acc_matrix("sft", "seen_by_day")
    if _R is None or base is None:
        _out = mo.md("")
    else:
        def _span(v):
            lo, hi = 100 * min(v), 100 * max(v)
            return f"{lo:.0f}%" if round(lo) == round(hi) else f"{lo:.0f}–{hi:.0f}%"

        _w = [r["world"] for r in res["sft"]["rows"]]
        _n = [r["text_nll"] for r in res["sft"]["rows"]]
        _out = mo.md(f"""
    Each new day is learned well (diagonal {_span(_R.diagonal())} on questions worded differently from the
    training ones; {_span(_S.diagonal())} in the taught wording), and earlier days fade as later ones arrive: day 0 goes from {100 * _R[0, 0]:.0f}% to
    {100 * _R[-1, 0]:.0f}%. The damage isn't limited to the new facts. The capitals the model knew fall to
    {', '.join(f'{100 * w:.0f}%' for w in _w)} after each day, and its loss on Wikipedia text rises from
    {base['text_nll']:.3f} to {', '.join(f'{n:.3f}' for n in _n)} nats. Nothing in the training data mentions capitals;
    the same weights simply store both.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · The classic fixes: replay and elastic weights

    **Replay.** McClelland's answer from 1995: keep practising the old material. Here a quarter of each batch is
    drawn from earlier days' question/answer pairs. It costs storage (you must keep old data) and a little
    compute, and it's still the baseline most methods fail to beat.

    **Elastic weight consolidation** (EWC, Kirkpatrick et al., 2017) needs no old data. After the old task, measure
    how much each weight matters to it, the diagonal of the **Fisher information** $F_i$ (in practice, the mean
    squared gradient of the old task's loss), and add a spring that pulls each weight back toward its old value
    $\theta^*_i$, as stiff as $F_i$:

    $$\mathcal L(\theta) = \mathcal L_{\text{new}}(\theta) + \lambda \sum_i F_i\,(\theta_i - \theta^*_i)^2$$

    Weights the old task barely used can move freely; important ones are held. Here the "old task" is general
    knowledge, so $F$ comes from Wikipedia text and $\theta^*$ is the pretrained model. The picture below is the
    idea in two dimensions.

    **Your turn:** `ewc_penalty(params, anchor, fisher)`, which works on whole pytrees.
    """)
    return


@app.cell
def _(np, plt):
    # two weights; the old task cares a lot about w0 and little about w1; the new task wants both to move
    _old, _new = np.array([0.0, 0.0]), np.array([2.0, 2.0])
    _H_old, _H_new = np.diag([8.0, 0.3]), np.diag([1.0, 1.0])

    def _path(lam, steps=200, lr=0.05):
        w, out = _old.copy(), [_old.copy()]
        for _ in range(steps):
            g = 2 * _H_new @ (w - _new) + lam * 2 * np.diag(_H_old) * (w - _old)
            w = w - lr * g
            out.append(w.copy())
        return np.array(out)

    _g = np.linspace(-0.7, 2.5, 120)
    _X, _Y = np.meshgrid(_g, _g)
    _Lold = _H_old[0, 0] * _X**2 + _H_old[1, 1] * _Y**2
    fig_ewc, _ax = plt.subplots(figsize=(4.6, 4.0))
    _ax.contour(_X, _Y, _Lold, levels=[0.5, 2, 6, 12], colors="tab:blue", linewidths=0.8)
    _ax.contour(_X, _Y, (_X - 2) ** 2 + (_Y - 2) ** 2, levels=[0.1, 0.5, 1.5, 3], colors="tab:orange",
                linewidths=0.8, linestyles="--")
    for _lam, _c, _lab in ((0.0, "black", "plain training on the new task"), (1.0, "tab:green", "with EWC, λ = 1")):
        _p = _path(_lam)
        _ax.plot(_p[:, 0], _p[:, 1], color=_c, label=f"{_lab}: ends at ({_p[-1, 0]:.2f}, {_p[-1, 1]:.2f})")
    _ax.plot(*_old, "o", color="tab:blue")
    _ax.plot(*_new, "o", color="tab:orange")
    _ax.set_xlabel("w0 (the old task needs it)")
    _ax.set_ylabel("w1 (the old task barely uses it)")
    _ax.set_title("blue: old task's loss · orange: new task's", fontsize=9)
    _ax.legend(fontsize=7, loc="lower right")
    fig_ewc.tight_layout()
    fig_ewc
    return


@app.cell
def _(ex, jnp, lj, mo):
    _params = {"w": jnp.array([1.0, 2.0, 3.0]), "head": {"b": jnp.array([0.5])}}
    _anchor = {"w": jnp.array([1.0, 0.0, 1.0]), "head": {"b": jnp.array([0.0])}}
    _fisher = {"w": jnp.array([10.0, 1.0, 0.5]), "head": {"b": jnp.array([4.0])}}
    _a = lj.attempt(ex.ewc_penalty, _params, _anchor, _fisher)
    _a.view(mo)
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "elastic weight consolidation penalty", solves="ewc_penalty")
    return


@app.cell
def _(base, mo, np, res):
    def board(arms):
        rows = []
        for a in arms:
            r = res.get(a)
            if r is None:
                continue
            R = np.array([x["acc_by_day"] for x in r["rows"]])
            m = r["metrics"]
            w = r["rows"][-1]["world"]
            n = r["rows"][-1]["text_nll"]
            rows.append(f"| {a} | {100 * m['avg']:.1f}% | {100 * R.diagonal().mean():.1f}% | {100 * m['bwt']:+.1f} | "
                        f"{100 * m['forgetting']:.1f} | " + (f"{100 * w:.0f}% | {n - base['text_nll']:+.3f} | "
                                                             if w is not None else "— | — | ")
                        + f"{r.get('trainable', 0) / 1e6:.1f}M | {r['secs_total'] / 60:.1f} |")
        if not rows:
            return mo.md("*Run `train.py` to fill this table.*")
        return mo.md("""
    | arm | final avg (all days) | just learned (diagonal) | backward transfer | forgetting | capitals kept | Δ Wikipedia loss | trained params | minutes |
    |---|---|---|---|---|---|---|---|---|
    """ + "\n".join(rows))

    def stat(arm):
        """Headline numbers of one arm, in percent / nats, for the prose."""
        r = res[arm]
        R = np.array([x["acc_by_day"] for x in r["rows"]])
        return {"avg": 100 * r["metrics"]["avg"], "bwt": 100 * r["metrics"]["bwt"], "diag": 100 * R.diagonal(),
                "world": 100 * r["rows"][-1]["world"], "dnll": r["rows"][-1]["text_nll"] - base["text_nll"],
                "R": 100 * R}
    return board, stat


@app.cell
def _(board):
    board(["sft", "replay", "ewc"])
    return


@app.cell
def _(mo, res, stat):
    if any(res[a] is None for a in ("sft", "replay", "ewc")):
        _out = mo.md("")
    else:
        _s, _r, _e = stat("sft"), stat("replay"), stat("ewc")
        _out = mo.md(f"""
    The two fixes protect **different things**, and each protects only what it can see.

    **Replay** keeps the old days: {_r['avg']:.0f}% at the end against {_s['avg']:.0f}% for plain fine-tuning, with
    almost no backward transfer ({_r['bwt']:+.1f} points). It does nothing for the capitals ({_r['world']:.0f}% at
    the end, {_s['world']:.0f}% without replay), because no capital is in the replay buffer.

    **EWC** is the mirror image. Its Fisher came from Wikipedia, so it holds the capitals ({_e['world']:.0f}%) and
    the Wikipedia loss ({_e['dnll']:+.3f} nats) almost exactly where they started, and it learns each new day at least
    as well as SFT (diagonal {_e['diag'].mean():.0f}% against {_s['diag'].mean():.0f}%). But nothing anchors the
    *earlier days*, and they go faster than under plain SFT ({_e['bwt']:+.0f} against {_s['bwt']:+.0f} points of
    backward transfer): with most weights held near the pretrained model, each day's facts seem to be written into
    the same few free directions as the last.

    The textbook EWC adds each finished day's Fisher to the penalty, with that day's weights as a new anchor. One
    trap if you try it (a good stretch exercise): the empirical Fisher is a mean *squared gradient*, and on facts the
    model has just fitted to near-zero loss the gradients are near zero, so λ has to be retuned for it.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Don't overwrite: add capacity

    Full fine-tuning lets every weight move. The other extreme: **freeze the model** and put the new knowledge in
    new parameters. LoRA (Module 20) does that with a low-rank update to existing matrices. The Sleep paper adds
    small **experts** beside each MLP, as in a mixture of experts (Module 40). Each expert is a little two-layer
    MLP of rank $r$ that reads the same normalized input as the MLP and adds to its output:

    $$\text{mlp\_out} = \text{SwiGLU}(h) + \sum_e g_e\,\operatorname{silu}(h A_e)\,B_e,\qquad A_e \in \mathbb R^{d\times r},\ B_e \in \mathbb R^{r\times d}$$

    $B_e$ starts at zero, so a new expert changes nothing until it is trained. Growing a tensor would change
    shapes and recompile, so all experts are allocated up front and switched on by the gate $g$, as the paper
    suggests (it compares this with the brain's fixed number of neurons and its new connections).

    Two arms test **expansion** itself, with rank 64 (2.8M parameters per expert, against the model's 494M):
    `expert_one` trains one expert on every day (capacity is shared), and `expert_new` trains a new expert each
    day and freezes the old ones. There is no router: every expert is on for every token, as in the Sleep paper's
    simplest variant. A third arm, `expert_new_retain`, adds a term that the Sleep paper doesn't have but much of
    the continual-learning literature does (it's "learning without forgetting", Li & Hoiem 2016, applied to text):
    on a batch of Wikipedia text, the KL divergence from the frozen model's next-token distributions to the current
    model's, so the new expert is pushed to change nothing where it isn't needed.

    **Your turn:** `expert_out(h, A, B, gate)`.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _k = jax.random.split(jax.random.key(0), 3)
    _h = jax.random.normal(_k[0], (1, 3, 8))
    _A = jax.random.normal(_k[1], (2, 8, 4))
    _B = jnp.zeros((2, 4, 8)).at[0].set(jax.random.normal(_k[2], (4, 8)))
    _a = lj.attempt(ex.expert_out, _h, _A, _B, jnp.array([1.0, 1.0]))
    _b = lj.attempt(ex.expert_out, _h, _A, _B, jnp.array([0.0, 1.0]))
    _msg = mo.md("")
    if _a.ok and _b.ok:
        _msg = mo.md(f"Expert 0 trained, expert 1 brand new (B = 0). Both on: output norm "
                     f"{float(jnp.linalg.norm(_a.value)):.3f}. Only the new one on: "
                     f"{float(jnp.linalg.norm(_b.value)):.3f} (it adds nothing yet).")
    mo.vstack([_a.view(mo), _msg])
    return


@app.cell
def _(heat_row, mo):
    _f = heat_row(["expert_one", "expert_new"], ["one shared expert", "a new expert per day"])
    _f if _f is not None else mo.md("*Run `train.py --arms expert_one expert_new`.*")
    return


@app.cell
def _(board):
    board(["sft", "expert_one", "expert_new", "expert_new_retain"])
    return


@app.cell
def _(mo, res, stat):
    if any(res[a] is None for a in ("sft", "expert_one", "expert_new", "expert_new_retain")):
        _out = mo.md("")
    else:
        _s, _o, _n, _k = stat("sft"), stat("expert_one"), stat("expert_new"), stat("expert_new_retain")
        _out = mo.md(f"""
    Freezing the model did **not** protect anything by itself. One shared expert overwrites itself day after day
    ({_o['bwt']:+.0f} points of backward transfer), keeps about as few capitals as full fine-tuning
    ({_o['world']:.0f}% against {_s['world']:.0f}%), and damages the Wikipedia loss far more ({_o['dnll']:+.2f}
    nats against {_s['dnll']:+.2f}): a rank-64 expert trained at a learning rate of 10⁻³ is a big, always-on
    change. A new expert per day helps only a little ({_n['avg']:.0f}% at the end against
    {_o['avg']:.0f}%), although every earlier expert is frozen. The likely reason is the missing router: the newest
    expert adds its output to *every* token, including the questions about earlier days' people, and nothing in
    its training tells it to stay quiet there. Frozen weights can't be overwritten, but what they compute can still
    be drowned out.

    The retain term is what makes expansion work here: {_k['avg']:.0f}% at the end, a Wikipedia loss within
    {_k['dnll']:.3f} nats of the original, and capitals {_k['world']:.0f}%. Asking each expert to do nothing on
    ordinary text acts as a soft, learned version of the router this setup lacks. Day 0's people still fade
    (from {_k['R'][0, 0]:.0f}% to {_k['R'][-1, 0]:.0f}%): the retain batch is Wikipedia, not earlier days' questions.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Teach yourself: self-distillation

    The model already *can* answer about a person when the bio is in its prompt (Section 1). That version of the
    model is a ready-made teacher. **Self-distillation** trains the model without the bio to produce the same
    next-token distributions as the model with it:

    * **teacher**: the model as it was before tonight's update, with the day's bio in the prompt;
    * **student**: the same model plus the day's new expert, *without* the bio;
    * the student answers a question in its own words (a sample), and at every token of that answer it is pushed
      toward the teacher's whole distribution over the next token.

    This is **SDFT** (Shenfeld et al., 2026), and it is what the Sleep paper calls **knowledge seeding**, an
    "upward" distillation from the model-with-notes into a bigger model (the one with the new expert). It differs
    from SFT in two ways. The targets are distributions, not one demonstration's tokens. And the student trains on
    its **own** samples (*on-policy*), so it is corrected where it actually goes wrong, and its outputs stay close
    to what it would have said anyway, which is SDFT's explanation for why it forgets less.

    Below, one question at the first answer token where the fact matters: what the teacher (with the bio) and the
    untrained student (without) put their probability on.
    """)
    return


@app.cell
def _(F, M, cfg, jnp, mo, np, params, tok, world):
    _p = world.days[0][0]
    _q = F.TEMPLATES["birth_city"][3].format(name=_p.name)
    _ans = M.answer_ids(tok, F.BIO_SENTENCES["birth_city"].format(name=_p.name, v=_p.birth_city))
    _k = len(tok.encode(f"{_p.name} was born in the city of").ids)      # answer tokens before the city
    _t_tok, _t_mask, _ = M.layout([M.prompt_ids(tok, _q, _p.bio())], [_ans], M.P_CTX, M.N_ANS)
    _s_tok, _s_mask, _ = M.layout([M.prompt_ids(tok, _q)], [_ans], M.P_LEN, M.N_ANS)
    t_logits = M.answer_logits(params, None, None, jnp.asarray(_t_tok), jnp.asarray(_t_mask), cfg, M.P_CTX,
                               M.N_ANS)[0, _k]
    s_logits = M.answer_logits(params, None, None, jnp.asarray(_s_tok), jnp.asarray(_s_mask), cfg, M.P_LEN,
                               M.N_ANS)[0, _k]

    def _top(lg, n=5):
        pr = np.asarray(jnp.exp(lg - jnp.max(lg)) / jnp.sum(jnp.exp(lg - jnp.max(lg))))
        idx = np.argsort(-pr)[:n]
        return [(tok.decode([int(i)]), float(pr[i])) for i in idx]

    _tt, _st = _top(t_logits), _top(s_logits)
    _rows = "\n".join(f"| `{a!r}` {100 * pa:.1f}% | `{b!r}` {100 * pb:.1f}% |" for (a, pa), (b, pb) in zip(_tt, _st))
    mo.md(f"""
    *{_q}* (truth: **{_p.birth_city}**). Next token after "{tok.decode(_ans[:_k])}":

    | teacher (bio in prompt) | student (no bio) |
    |---|---|
    {_rows}
    """)
    return s_logits, t_logits


@app.cell
def _(mo):
    mo.md(r"""
    How far apart are those two distributions? Distillation needs a divergence, and the choice matters:

    * **forward KL**, $\mathrm{KL}(p_{\text{teacher}} \Vert q_{\text{student}})$, is large wherever the teacher
      puts mass and the student doesn't: it makes the student *cover* everything the teacher considers;
    * **reverse KL**, $\mathrm{KL}(q \Vert p)$, is large wherever the student puts mass the teacher doesn't: it
      makes the student *avoid* what the teacher rules out, and settle on one of the teacher's modes;
    * **Jensen-Shannon** compares both with their average, is symmetric, and never exceeds $\log 2$.

    SDFT's theory points to reverse KL (it is the on-policy RL objective with the teacher's log-ratio as reward),
    but the authors found forward KL worked best in practice; this module's runs use forward KL.

    **Your turn:** `token_kl(teacher_logits, student_logits, kind)`.
    """)
    return


@app.cell
def _(ex, lj, mo, s_logits, t_logits):
    _r = {k: lj.attempt(ex.token_kl, t_logits, s_logits, k) for k in ("forward", "reverse", "jsd")}
    _msg = mo.md("")
    if all(a.ok for a in _r.values()):
        _msg = mo.md(" · ".join(f"**{k}** {float(a.value):.3f} nats" for k, a in _r.items()))
    mo.vstack([_r["forward"].view(mo), _msg])
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "per-token KL: forward, reverse, Jensen-Shannon", solves="token_kl")
    return


@app.cell
def _(heat_row, mo):
    _f = heat_row(["expert_new", "distill"], ["new expert, SFT on demonstrations", "new expert, self-distillation"])
    _f if _f is not None else mo.md("*Run `train.py --arms expert_new distill`.*")
    return


@app.cell
def _(board):
    board(["sft", "expert_new", "distill", "expert_new_retain", "distill_retain"])
    return


@app.cell
def _(mo, res, stat):
    if any(res[a] is None for a in ("expert_new", "distill", "expert_new_retain", "distill_retain")):
        _out = mo.md("")
    else:
        _n, _d, _k, _dk = stat("expert_new"), stat("distill"), stat("expert_new_retain"), stat("distill_retain")
        _out = mo.md(f"""
    Self-distillation learned the facts without ever seeing a demonstration answer, and kept more of them than SFT
    into a new expert ({_d['avg']:.0f}% at the end against {_n['avg']:.0f}%). It did **not** forget less in the
    sense that matters most here: backward transfer was about the same ({_d['bwt']:+.0f} against {_n['bwt']:+.0f}
    points), and general knowledge suffered too (capitals {_d['world']:.0f}% against {_n['world']:.0f}%, Wikipedia
    loss {_d['dnll']:+.2f} nats against {_n['dnll']:+.2f}: less damage to text, a little more to the capitals). SDFT measured its "forgets less" with every weight trainable; here the facts go into
    an always-on expert, and that expert's interference seems to dominate whichever loss trained it.

    Adding the retain term fixed the Wikipedia loss ({_dk['dnll']:+.3f} nats) but cost the facts: the last day
    reached only {_dk['diag'][-1]:.0f}% and the final average fell to {_dk['avg']:.0f}%, below `expert_new_retain`
    ({_k['avg']:.0f}%). The distillation signal (a soft distribution, on the model's own samples) is weaker per step
    than a demonstration, and here it lost the tug-of-war with the retain term. It also cost
    {res['distill']['secs_total'] / res['expert_new']['secs_total']:.1f} times the time of `expert_new`, for sampling
    the student's answers and running the teacher.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Learning to imitate (described, not run)

    The Sleep paper adds a second term to knowledge seeding. The teacher writes a sample, the student sees its
    start and must continue it, and a reward scores the continuation: partly a frozen judge's "same meaning?" (1 or
    0) and partly a token-level similarity built on the **Levenshtein distance** $z$, the fewest insertions,
    deletions and substitutions that turn one sequence into the other:

    $$r_{\text{abs}} = \begin{cases} 1 - \dfrac{z}{\max(\lvert\hat d\rvert, \lvert d\rvert)} & z \le z_0 \\ 0 & \text{otherwise}\end{cases}$$

    It's optimized with reinforcement learning (Module 26's tools) and mixed with the distillation loss. The paper's
    ablation credits it with 0.2–2.4 points on its maths benchmarks. This module doesn't train it: for one-line
    factual answers the distillation targets already pin the answer down, and the judge would need a second model.

    **Your turn:** `levenshtein_reward(pred, target, z0)`.
    """)
    return


@app.cell
def _(ex, lj, mo):
    _a = lj.attempt(ex.levenshtein_reward, "born in Lisbon", "born in Lisbon.", 3)
    _b = lj.attempt(ex.levenshtein_reward, "born in Lima", "born in Lisbon.", 3)
    _msg = mo.md("")
    if _a.ok and _b.ok:
        _msg = mo.md(f"One character missing: reward {_a.value:.3f}. A different city (5 edits, over the threshold "
                     f"z0 = 3): reward {_b.value:.3f}.")
    mo.vstack([_a.view(mo), _msg])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Dreaming: the model writes its own practice

    So far the teaching data came with questions. Often you only have a passage. Training on the passage itself is
    the obvious move, and it's known to work badly: knowing a text is not the same as being able to answer
    questions about it (Allen-Zhu & Li, 2023; SEAL measured 33.5% against 32.7% for doing nothing). **SEAL**
    (Zweiger et al., 2025) had the model rewrite each passage into training data (implications, question/answer
    pairs) and trained on that instead. The Sleep paper calls such self-written data **dreams**, keeps the most
    useful ones by **gradient size** (a big gradient means the model has much to learn from that example) plus a
    few at random for variety, and adds a twist this module doesn't reproduce: while dreaming, each MoE router also
    picks a random expert, to mix in unrelated knowledge.

    Three arms, each a new expert per day, none given the teaching questions: `bio_text` trains on the bios as
    plain text; `dream` has the model read each bio and write five question/answer pairs about it, twice, and
    trains on what it wrote; `dream_select` keeps the dreams with the largest gradient (top half) plus a random
    tenth of the rest. All three get as many optimizer steps per day as SFT on the 750 teaching questions.

    **Your turn:** `select_dreams(scores, k, b, seed)`.
    """)
    return


@app.cell
def _(ex, lj, mo, np):
    _scores = np.round(np.random.default_rng(3).gamma(2.0, 1.0, 12), 2)
    _a = lj.attempt(ex.select_dreams, _scores, 4, 2, 0)
    mo.vstack([mo.md(f"Twelve dreams' gradient norms: `{_scores.tolist()}`; keep the top 4 plus 2 at random."),
               _a.view(mo)])
    return


@app.cell
def _(mo, res):
    _r = res.get("dream")
    if _r is None:
        _out = mo.md("*Run `train.py --arms dream` to see the model's dreams.*")
    else:
        _ex = "\n\n".join("> " + t.replace("\n", "  \n> ") for t in _r["dream_examples"][:2])
        _out = mo.md(f"""
    Two of the model's dreams (it wrote {sum(_r['n_dreams'])} usable question/answer pairs over the four days,
    about {sum(_r['n_dreams']) / 200:.1f} per person; text it didn't format as `Q: … A: …` is dropped):

    {_ex}
    """)
    _out
    return


@app.cell
def _(board):
    board(["bio_text", "dream", "dream_select", "expert_new"])
    return


@app.cell
def _(mo, res, stat):
    if any(res[a] is None for a in ("bio_text", "dream", "dream_select", "expert_new")):
        _out = mo.md("")
    else:
        _b, _d, _x, _n = stat("bio_text"), stat("dream"), stat("dream_select"), stat("expert_new")
        _q = res["dream_select"]["score_quantiles"][0]
        _out = mo.md(f"""
    Without the teaching questions, every arm learns far less than `expert_new` (each day just after training:
    {_b['diag'].mean():.0f}% for the plain bios, {_d['diag'].mean():.0f}% for dreams, {_x['diag'].mean():.0f}% for
    selected dreams, against {_n['diag'].mean():.0f}% with the teaching questions). And here dreaming did **not**
    beat plain text ({_d['avg']:.0f}% at the end against {_b['avg']:.0f}%), unlike in SEAL. Two plausible reasons,
    neither tested: these bios are already five short facts, one per sentence, so there is little to "imply" that
    the text doesn't say; and a 0.5B model writes noisy dreams (look at the examples above: links between facts that
    the bio never makes, stated as if they were facts). Dreams did one thing better: the Wikipedia loss rose
    {_d['dnll']:.2f} nats against {_b['dnll']:.2f} for the bios, since short answers are less like training a
    language model on new text.

    **Gradient selection made it worse** ({_x['avg']:.0f}% at the end). The scores are heavy-tailed: on day 0 the
    median dream's gradient norm was {_q[2]:.0f} and the largest {_q[4]:.0f}. Selecting by gradient norm assumes that a
    big gradient means much to learn; with noisy self-written data, a big gradient can as easily mean the dream is
    wrong, or about something the bio never said, so the filter keeps the worst dreams. The Sleep paper's dreams
    come from 1.7B and 8B models, with a dreaming prompt tuned by reinforcement learning in SEAL's setup; the
    selection step may need dreams that are mostly right before it helps.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · The scoreboard

    Every arm, on the same 200 people and the same old knowledge. "In context" is the ceiling of the other memory:
    each day's bios in the prompt while that day is being asked about, nothing afterwards. The plot puts what was
    learned (final average over all four days) against what was kept (capitals still known at the end); up and to
    the right is better.
    """)
    return


@app.cell
def _(ARMS, board):
    board([a for a in ARMS if a != "icl"])
    return


@app.cell
def _(base, mo, plt, res):
    _pts = [(a, r["metrics"]["avg"], r["rows"][-1]["world"]) for a, r in res.items()
            if r is not None and r["rows"][-1]["world"] is not None]
    if not _pts or base is None:
        _out = mo.md("")
    else:
        _fig, _ax = plt.subplots(figsize=(5.4, 3.8))
        for _a, _x, _y in _pts:
            _ax.plot(100 * _x, 100 * _y, "o")
            _ax.annotate(_a, (100 * _x, 100 * _y), fontsize=8, xytext=(4, 3), textcoords="offset points")
        _ax.axhline(100, color="grey", lw=0.6, ls=":")
        _ax.set_xlabel("new facts known at the end, all 4 days (%)")
        _ax.set_ylabel("capitals still known (%)")
        _ax.set_xlim(0, 100)
        _ax.set_ylim(0, 105)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(ARMS, mo, res, stat):
    if any(res[a] is None for a in ARMS if a != "icl"):
        _out = mo.md("")
    else:
        _st = {a: stat(a) for a in ARMS if a != "icl"}
        _facts = max(_st, key=lambda a: _st[a]["avg"])
        _keep = max(_st, key=lambda a: _st[a]["world"])
        _both = max(_st, key=lambda a: min(_st[a]["avg"], _st[a]["world"]))
        _out = mo.md(f"""
    What the runs say, at this scale:

    * **Most new facts kept:** `{_facts}` ({_st[_facts]['avg']:.0f}%), the oldest method on the board, at the cost
      of storing the old data. **Most old knowledge kept:** `{_keep}` (capitals {_st[_keep]['world']:.0f}%), which
      in exchange forgets earlier days. No arm reaches the top right corner; the best compromise is `{_both}`
      ({_st[_both]['avg']:.0f}% of the facts, {_st[_both]['world']:.0f}% of the capitals).
    * Each protection works only on what it is shown: replay on its buffer, EWC on the text its Fisher came from,
      the retain term on its text batch. The obvious next experiment combines them (replay plus a retain term).
    * Adding capacity didn't avoid interference without a router or a retain term, and the Sleep paper's
      additions (self-distillation, dreams, gradient selection) did not beat simpler baselines here. That's
      a result about a 0.5B model and short synthetic facts, not a refutation of a paper that used 1.7B–8B models
      and its own architecture, but it's the honest starting point for trying them on your own data.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 · What the Sleep paper claims, and what we could test

    Behrouz et al. build their method on their own **Hope** architecture, a chain of MLPs that update at different
    rates. Consolidation happens between levels of that chain, and its results on class-incremental learning,
    long context (BABILong to 10M tokens) and two unseen languages are results for Hope plus sleep. That
    architecture isn't released, and neither is the code, so none of it can be rerun here. The parts that transfer
    to an ordinary transformer are the ones this module ran: new experts per consolidation, self-distillation from
    the model-before-update with its notes in context, and self-written dreams kept by gradient size. Its own
    numbers for those parts (maths benchmarks with Qwen3-1.7B and 8B, SQuAD with SEAL's setup) are gains of one to
    a few points, without error bars.

    Two warnings from the wider literature. Shenfeld et al. found that self-distillation needs a model whose
    in-context learning is good enough to be a teacher: at 3B parameters it fell behind SFT on their science task.
    Here the task is reading a fact from a short bio, which even 0.5B does almost perfectly (Section 1), so the
    teacher is good; harder knowledge would need a bigger model. And Kim et al. (2026) show self-distillation
    can quietly erode a model's out-of-distribution reasoning, so a continual learner needs the "old knowledge"
    checks this module uses, and more of them.

    **Caveats of this module's own runs:** one seed, one model size, synthetic facts with short answers, four days.
    Hyperparameters were chosen on day 0 for each family (learning rate and epochs), not tuned per arm across all
    four days.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "a": "Fine-tuning on facts about invented people can't harm unrelated knowledge, because the training data "
                 "never mentions it.",
            "b": "Replaying old examples kept the earlier days' facts but not the capitals, so protecting what you "
                 "taught and protecting what the model already knew are separate problems.",
            "c": "Freezing the model and training a new expert per day protects old knowledge, because the old weights "
                 "and the earlier experts never change.",
        },
        label="Which statement does this module support?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    if quiz.value is None:
        _out = mo.md("")
    elif quiz.value == "b":
        _out = mo.callout(mo.md("Right. Replay protects what is in its buffer, EWC protects what its Fisher was "
                                "measured on, and the retain term protects the text it is computed on. Each method "
                                "has to be told what \"old knowledge\" means."), kind="success")
    elif quiz.value == "a":
        _out = mo.callout(mo.md("Not quite: Section 3 shows the capitals and the Wikipedia loss getting worse under "
                                "full fine-tuning on invented people. The same weights store both."), kind="warn")
    else:
        _out = mo.callout(mo.md("Not quite: Section 5 shows `expert_new` forgetting earlier days and damaging the "
                                "Wikipedia loss. Frozen weights can't be overwritten, but without a router the new "
                                "expert adds its output to every token and drowns out what the old ones compute."),
                          kind="warn")
    _out
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(49))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    * Behrouz, Hashemi, Javanmard & Mirrokni, [Language Models Need Sleep](https://arxiv.org/abs/2606.03979) — the
      paper this module is anchored on; Sections 3.2–3.4 are consolidation, knowledge seeding and dreaming.
    * Shenfeld, Damani, Hübotter & Agrawal, [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)
      — the simplest version of the self-distillation idea, with its failure modes discussed honestly.
    * Zweiger et al., [Self-Adapting Language Models (SEAL)](https://arxiv.org/abs/2506.10943) — the dreaming step's
      origin, and a measurement of forgetting under repeated self-edits.
    * Kirkpatrick et al., [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
      — EWC, with its derivation from a Bayesian posterior.
    * Kim et al., [Why does self-distillation (sometimes) degrade the reasoning capability of LLMs?](https://arxiv.org/abs/2603.24472)
      — the counterweight: what self-distillation can cost.

    **Next:** Module 50, uncensored models: removing refusals from the data and from the weights.
    """)
    return


if __name__ == "__main__":
    app.run()
