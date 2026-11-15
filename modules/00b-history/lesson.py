import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="00b · A short history of AI")


@app.cell
def _():
    import importlib.util

    import marimo as mo
    import matplotlib.pyplot as plt

    import learnjax as lj
    from learnjax.papers import PAPERS
    from learnjax.syllabus import MODULES, label

    ex = lj.load_exercises(mo.notebook_dir())
    _spec = importlib.util.spec_from_file_location("history_events", mo.notebook_dir() / "events.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    EVENTS = _mod.EVENTS
    return EVENTS, MODULES, PAPERS, ex, label, lj, mo, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 00b · A short history of AI, and where each idea lives in this course

    **Goal:** see the whole story before you build it. Almost every idea in this course was invented
    to fix a problem someone was stuck on, and many were invented, forgotten and rediscovered.
    Knowing the story makes each lesson feel less like a list of tricks and more like a chapter.

    **You will:** explore an interactive timeline of about 50 milestones and every seminal paper the
    course cites · read the story era by era, each with pointers to the modules that rebuild it ·
    meet the two big traditions, symbolic and statistical AI · see how GPUs, CUDA, ROCm, XLA and JAX
    made it all practical · write a few small timeline tools.

    No GPU needed here. Read it now, and come back whenever a later module mentions a name or a year.
    """)
    return


@app.cell
def _(EVENTS, MODULES, PAPERS):
    _slug_of = {n: s for n, s, _t, _p in MODULES}

    def era_name(year):
        # the lesson's own copy, so the timeline works before you finish the exercises
        for _start, _end, _name in [(1940, 1969, "Foundations"), (1970, 1992, "Winters & expert systems"),
                                    (1993, 2011, "Statistical ML & the deep-learning revival"),
                                    (2012, 2016, "Deep learning"), (2017, 2021, "Transformers & scale"),
                                    (2022, 2100, "The LLM era")]:
            if _start <= year <= _end:
                return _name
        return "?"

    ERA_ORDER = ["Foundations", "Winters & expert systems", "Statistical ML & the deep-learning revival",
                 "Deep learning", "Transformers & scale", "The LLM era"]
    ERA_COLOR = dict(zip(ERA_ORDER, ["#4c72b0", "#8c8c8c", "#55a868", "#dd8452", "#c44e52", "#8172b3"]))

    _seen = set()
    papers_flat = []
    for _n in sorted(PAPERS):
        for _p in PAPERS[_n]:
            if _p.url in _seen:
                continue
            _seen.add(_p.url)
            papers_flat.append({"year": _p.year, "title": _p.title, "who": _p.authors, "kind": "paper",
                                "module": _slug_of.get(_n), "url": _p.url})
    items = [{**e, "kind": "event"} for e in EVENTS] + papers_flat
    for _it in items:
        _it["era"] = era_name(_it["year"])
    items.sort(key=lambda d: (d["year"], d["kind"] != "event"))
    return ERA_COLOR, ERA_ORDER, era_name, items, papers_flat


@app.cell
def _(ERA_ORDER, mo):
    decades = mo.ui.range_slider(start=1940, stop=2030, step=10, value=[1940, 2030], label="years",
                                 show_value=True)
    era_pick = mo.ui.dropdown(options=["All eras"] + ERA_ORDER, value="All eras", label="era")
    show_papers = mo.ui.checkbox(value=True, label="include papers")
    mo.vstack([mo.md("## 1 · The timeline"), mo.hstack([decades, era_pick, show_papers], justify="start")])
    return decades, era_pick, show_papers


@app.cell
def _(ERA_COLOR, decades, era_pick, items, plt, show_papers):
    _lo, _hi = decades.value
    shown = [it for it in items if _lo <= it["year"] <= _hi
             and (era_pick.value == "All eras" or it["era"] == era_pick.value)
             and (show_papers.value or it["kind"] == "event")]
    _ev = [it for it in shown if it["kind"] == "event"]
    _pp = [it for it in shown if it["kind"] == "paper"]
    _h = max(2.5, 0.22 * len(_ev) + 1.6)
    _fig, (_ax, _bx) = plt.subplots(2, 1, figsize=(11, _h), sharex=True,
                                    gridspec_kw={"height_ratios": [max(len(_ev), 1), 6]})
    # one row per milestone, oldest at the top, so labels never collide
    for _i, _it in enumerate(_ev):
        _y = len(_ev) - _i
        _c = ERA_COLOR[_it["era"]]
        _ax.plot([_lo - 1, _it["year"]], [_y, _y], color=_c, lw=0.4, alpha=0.35)
        _ax.scatter(_it["year"], _y, marker="*", s=70, color=_c, zorder=3)
        _ax.text(_it["year"] + 0.6, _y, f"{_it['year']}  {_it['title']}", fontsize=7, va="center")
    _ax.set_yticks([])
    _ax.set_ylim(0.3, len(_ev) + 0.7)
    _ax.set_title(f"{len(_ev)} milestones (one per row) and {len(_pp)} seminal papers (counts per year, below)",
                  fontsize=10)
    for _sp in ("top", "right", "left"):
        _ax.spines[_sp].set_visible(False)
    # papers: stacked counts per year, colored by era
    _years = sorted({it["year"] for it in _pp})
    for _yr in _years:
        _n = sum(1 for it in _pp if it["year"] == _yr)
        _bx.bar(_yr, _n, width=0.8, color=ERA_COLOR[next(it["era"] for it in _pp if it["year"] == _yr)])
    _bx.set_ylabel("papers", fontsize=8)
    _bx.set_xlabel("year")
    _bx.set_xlim(_lo - 1, _hi + 22)          # room on the right for the longest labels
    _bx.set_xticks(range(_lo, min(_hi, 2030) + 1, 10))
    for _sp in ("top", "right"):
        _bx.spines[_sp].set_visible(False)
    _handles = [plt.Line2D([], [], marker="*", ls="", color=_c, markersize=9, label=_name)
                for _name, _c in ERA_COLOR.items()]
    _fig.legend(handles=_handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False,
                bbox_to_anchor=(0.5, -0.01))
    _fig.tight_layout(rect=(0, 0.06, 1, 1))
    _fig
    return (shown,)


@app.cell
def _(mo, shown):
    mo.ui.table([{"year": it["year"], "kind": it["kind"], "title": it["title"], "who": it["who"],
                  "module": it["module"] or "", "source": it["url"]} for it in shown],
                page_size=10, selection=None, label="Everything on the timeline above")
    return


@app.cell
def _(MODULES, label):
    _title = {s: (label(s), t) for _n, s, t, _p in MODULES}

    def where(*slugs):
        """Markdown bullets pointing at course modules."""
        _rows = []
        for _s in slugs:
            _lab, _t = _title[_s]
            _rows.append(f"* **Module {_lab}** — {_t}")
        # indented to match the lesson's indented markdown, so marimo's dedent keeps it as prose
        _md = "**Where it lives in this course:**\n\n" + "\n".join(_rows)
        return "\n".join("    " + _ln for _ln in _md.splitlines())
    return (where,)


@app.cell
def _(mo, where):
    mo.md(r"""
    ## 2 · Foundations (1943–1969): a neuron in logic, a machine that learns

    In 1943 Warren McCulloch, a neurophysiologist, and Walter Pitts, a self-taught logician, described
    a neuron as a tiny logic gate: add up weighted inputs and fire if the total passes a threshold.
    Seven years later Alan Turing asked whether machines could think, and proposed judging them by
    conversation. In 1956 a summer workshop at Dartmouth, organized by John McCarthy, Marvin Minsky,
    Claude Shannon and Nathaniel Rochester, gave the field its name: *artificial intelligence*.

    Two approaches grew up side by side. One wrote intelligence down as **rules and symbols**: logic,
    search, programs that prove theorems. The other tried to **learn from examples**. Frank
    Rosenblatt's perceptron (1958) learned to classify patterns by nudging its weights after each
    mistake, and he built it in hardware as the Mark I. The press promised machines that would soon
    walk and talk.

    They didn't. In 1969 Minsky and Seymour Papert's book *Perceptrons* proved that a single-layer
    perceptron cannot learn even XOR (output 1 when exactly one input is 1). Everyone knew more layers
    would help, but nobody knew how to *train* them. Neural-network research went quiet.

    """ + where("29-perceptron-hopfield", "10-mlp"))
    return


@app.cell
def _(mo, where):
    mo.md(r"""
    ## 3 · Winters and expert systems (1970–1992)

    Early AI programs looked brilliant on toy problems and fell apart on real ones. In 1973 the
    Lighthill report told the British government that AI had failed to deliver; funding collapsed
    there and in the US. This was the first **AI winter**, roughly 1974–1980.

    Symbolic AI came back in the 1980s as **expert systems**: hundreds or thousands of if-then rules
    written down by interviewing specialists. Some made real money, like DEC's XCON for configuring
    computers. Douglas Lenat's Cyc (1984) set out to write down *all* of common sense as logic. But
    rules were brittle: they broke on anything their authors hadn't imagined, and they were expensive
    to maintain. When the specialized Lisp-machine market collapsed in the late 1980s, a **second
    winter** followed.

    Meanwhile the learning side quietly solved its biggest problem. John Hopfield (1982) showed a
    network could store memories as low points of an energy landscape, which drew physicists in. And
    in 1986 David Rumelhart, Geoffrey Hinton and Ronald Williams showed that **backpropagation** — the
    chain rule, applied backwards through the network — trains networks with hidden layers. That was
    the answer to Minsky and Papert. At Bell Labs, Yann LeCun used it to train convolutional networks
    that read handwritten digits (1989–1998), eventually processing many US bank checks.

    """ + where("03-autodiff", "29-perceptron-hopfield", "12-cnns"))
    return


@app.cell
def _(mo, where):
    mo.md(r"""
    ## 4 · Statistical ML and the deep-learning revival (1993–2011)

    Through the 1990s and 2000s, most of machine learning moved on from neural networks to methods
    with cleaner math, like support vector machines and probabilistic models. The headline AI moment
    of the decade wasn't learning at all: in 1997 IBM's Deep Blue beat world chess champion Garry
    Kasparov with enormous search and a hand-tuned evaluation.

    A few groups kept going with neural networks. In 1997 Sepp Hochreiter and Jürgen Schmidhuber
    published the **LSTM**, a recurrent cell with gates that lets learning signals survive across
    long sequences. In 2006 Hinton and colleagues showed deep networks could be trained by
    pre-training one layer at a time, and the field started calling itself **deep learning**.

    Two missing ingredients arrived. **Compute:** NVIDIA's CUDA (2007) made graphics cards
    programmable for general math, and a GPU turned out to be exactly the right shape for neural
    networks. **Data:** Fei-Fei Li's ImageNet (2009) offered millions of labeled images and a yearly
    competition to measure progress.

    """ + where("31-rnn-seq2seq", "34-deep-rl-alphazero", "23-kernels"))
    return


@app.cell
def _(mo, where):
    mo.md(r"""
    ## 5 · Deep learning (2012–2016): everything works at once

    In 2012 Alex Krizhevsky, Ilya Sutskever and Hinton trained a convolutional network, **AlexNet**,
    on two gaming GPUs and won ImageNet by a huge margin. Within a couple of years nearly every
    computer-vision group had switched to deep learning.

    The same recipe spread everywhere. **word2vec** (2013) turned words into vectors where meaning
    became geometry. DeepMind's **DQN** (2013–2015) learned dozens of Atari games from pixels and the
    score alone. Ian Goodfellow's **GANs** (2014) generated realistic images by pitting two networks
    against each other. **seq2seq** models translated whole sentences, and **attention** (2014) let
    them look back at any input word. **ResNet** (2015) added shortcut connections so networks could
    be hundreds of layers deep. Software followed: TensorFlow (2015) and PyTorch (2016).

    In March 2016 DeepMind's **AlphaGo** beat Lee Sedol at Go, a game many experts thought was still a
    decade away from falling to computers. It combined neural networks with tree search, the same
    kind of search Deep Blue relied on, but guided by learned intuition.

    """ + where("12-cnns", "30-word2vec", "34-deep-rl-alphazero", "33-generative-images",
                "31-rnn-seq2seq", "11-training-craft"))
    return


@app.cell
def _(mo, where):
    mo.md(r"""
    ## 6 · Transformers and scale (2017–2021)

    In 2017 a Google team led by Ashish Vaswani published "Attention Is All You Need". The
    **Transformer** threw out recurrence and kept only attention, so a whole sequence could be
    processed in parallel on a GPU. That one change made it practical to train on far more text.

    In 2018 two recipes split the field. Google's **BERT** pre-trained a Transformer *encoder* by
    filling in masked words; OpenAI's **GPT** pre-trained a *decoder* by predicting the next word.
    OpenAI scaled GPT up: GPT-2 (2019) and GPT-3 (2020, 175 billion parameters), which could pick up a
    new task from a few examples in the prompt. Researchers found that loss falls smoothly as models,
    data and compute grow — **scaling laws** — and scale became a strategy in its own right.

    Self-play went further too: **AlphaZero** (2017) mastered chess, shogi and Go with no human games.
    **AlphaFold 2** (2020) predicted protein structures almost as accurately as experiments.
    **CLIP** and **DALL·E** (2021) connected images and text, and **Stable Diffusion** (2022) put
    image generation on ordinary GPUs. In 2018 Bengio, Hinton and LeCun received the Turing Award
    for keeping neural networks alive through the winters.

    """ + where("15-attention", "16-gpt", "32-bert-vs-gpt", "17-modern-llm", "33-generative-images",
                "34-deep-rl-alphazero"))
    return


@app.cell
def _(mo, where):
    mo.md(r"""
    ## 7 · The LLM era (2022–)

    In November 2022 OpenAI released **ChatGPT**: a GPT model tuned to follow instructions using
    human feedback, wrapped in a free chat app. AI went from a research topic to something everyone
    used. In 2023 Meta's **Llama** weights started an open-weights wave, and tools like llama.cpp
    made it normal to run strong models on your own computer — exactly what you do in this course.

    Research kept moving. Mixture-of-experts models grew huge while staying cheap to run. **Mamba**
    (2023) offered a serious alternative to attention for long sequences. In 2024 the Nobel Prize in
    Physics went to Hopfield and Hinton for foundational neural-network work, and the Chemistry prize
    included Demis Hassabis and John Jumper for AlphaFold. In 2025 **DeepSeek-R1** showed that
    reinforcement learning with checkable answers can teach a model to reason step by step, and
    released the weights openly.

    The model you replicated at the very start, deepseek-v3-jax, is a miniature of this era: a
    Transformer with mixture-of-experts and multi-head latent attention.

    """ + where("17-modern-llm", "35-rlhf-dpo", "19-local-serving", "27-lmstudio", "36-mamba-ssm",
                "26-reasoning"))
    return


@app.cell
def _(mo, where):
    mo.md(r"""
    ## 8 · Two traditions: symbolic and statistical AI

    | | Symbolic AI | Statistical (learning) AI |
    |---|---|---|
    | Knowledge is… | written down as rules, logic and symbols | learned from data as numbers (weights) |
    | Strengths | exact, explainable, needs no training data | handles messy, real-world input; improves with data |
    | Weaknesses | brittle outside its rules; costly to write and maintain | needs lots of data and compute; hard to explain |
    | Classic examples | theorem provers, expert systems, Cyc, Deep Blue's search | perceptrons, backprop, CNNs, Transformers |

    For decades they were rivals, and each had its winter. Today's strongest systems mix them:
    AlphaGo and AlphaZero pair learned networks with explicit tree search; language models call
    tools, run code and check answers with verifiers. One modern bridge is **hyperdimensional
    computing** (also called vector-symbolic architectures): it represents symbols as very long
    vectors that can be combined into new concepts and pulled apart again. In 2023 IBM Research used
    this idea alongside neural perception to solve Raven's progressive matrices, a classic
    abstract-reasoning test.

    """ + where("34-deep-rl-alphazero", "26-reasoning", "37-interpretability"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · The hardware and software that made it possible

    The ideas were often old; what changed was **compute**. Training a network is mostly multiplying
    big matrices, and GPUs were built to do exactly that for video games.

    * **2007 — CUDA** (NVIDIA) made GPUs programmable for general math. AlexNet ran on it in 2012.
    * **2015–2016 — TensorFlow and PyTorch** made building networks a few lines of Python.
    * **2016 — TPUs** (Google) were chips designed only for neural networks.
    * **2016 — ROCm and HIP** (AMD) brought an open-source GPU compute stack, with HIP as a CUDA-like
      language. It's what runs this course on your RX 7600 XT.
    * **2017 — XLA**, a compiler that fuses many small operations into a few fast GPU kernels.
    * **2018 — JAX** joined NumPy-style code, automatic differentiation (from the Autograd project) and
      XLA compilation, using composable transforms like [`jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit), [`grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html#jax.grad) and [`vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap).

    Every lesson you run goes down that whole stack: your Python → JAX → XLA → the ROCm plugin → HIP →
    the amdgpu driver → a gfx1102 GPU. Module 00 walks through each layer, and Modules 06, 22 and 23
    look inside it.
    """)
    return


@app.cell
def _(mo):
    _people = [
        ("Alan Turing", "the imitation game (1950)", "00b"),
        ("Warren McCulloch & Walter Pitts", "the first artificial neuron (1943)", "29"),
        ("John McCarthy", "named the field at Dartmouth (1956); invented Lisp", "00b"),
        ("Frank Rosenblatt", "the perceptron (1958)", "29"),
        ("Marvin Minsky & Seymour Papert", "*Perceptrons* (1969)", "29"),
        ("John Hopfield", "Hopfield networks (1982); Nobel Prize 2024", "29"),
        ("Geoffrey Hinton", "backprop (1986), deep belief nets (2006), AlexNet (2012); Nobel 2024", "03, 12"),
        ("Yann LeCun", "convolutional networks (LeNet)", "12"),
        ("Yoshua Bengio", "neural language models (2003), attention (2014)", "14, 31"),
        ("Sepp Hochreiter & Jürgen Schmidhuber", "LSTM (1997)", "31"),
        ("Fei-Fei Li", "ImageNet (2009)", "12"),
        ("Ian Goodfellow", "GANs (2014)", "33"),
        ("Demis Hassabis & David Silver", "AlphaGo, AlphaZero (DeepMind)", "34"),
        ("Ashish Vaswani and co-authors", "the Transformer (2017)", "15"),
        ("John Jumper", "AlphaFold 2 (2020); Nobel 2024", "—"),
        ("Albert Gu & Tri Dao", "Mamba (2023); Dao also FlashAttention", "36, 23"),
    ]
    _rows = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in _people)
    mo.md("## 10 · People you'll meet again\n\n| Who | Known for | Modules |\n|---|---|---|\n" + _rows)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11 · Your turn: timeline tools

    Open `modules/00b-history/exercises.py`. Five small functions, plain Python, no JAX. The lesson
    uses them to answer questions about the timeline above.
    """)
    return


@app.cell
def _(ex, lj, mo):
    _cases = [1943, 1986, 2012, 2017, 2025]
    _a = [lj.attempt(ex.era_of, y) for y in _cases]
    if all(a.ok for a in _a):
        _out = mo.md("\n".join(f"* {y} → **{a.value}**" for y, a in zip(_cases, _a)))
    else:
        _out = next(a for a in _a if not a.ok).view(mo)
    mo.vstack([mo.md("**`era_of(year)`** for a few landmark years:"), _out])
    return


@app.cell
def _(EVENTS, ex, lj, mo):
    _w = lj.attempt(ex.events_between, EVENTS, 1974, 1993)
    if _w.ok:
        _out = mo.md(f"{len(_w.value)} milestones fall in the winter years 1974–1993:\n\n"
                     + "\n".join(f"* {e['year']} — {e['title']}" for e in _w.value))
    else:
        _out = _w.view(mo)
    mo.vstack([mo.md("**`events_between(EVENTS, 1974, 1993)`** — what happened during the winters?"), _out])
    return


@app.cell
def _(ex, lj, mo, papers_flat, plt):
    _c = lj.attempt(ex.papers_per_decade, papers_flat)
    if _c.ok:
        _fig, _ax = plt.subplots(figsize=(8, 3))
        _ax.bar([str(d) + "s" for d in _c.value], list(_c.value.values()), color="#4c72b0")
        _ax.set_ylabel("seminal papers cited")
        _ax.set_title("Papers this course cites, by decade")
        _fig.tight_layout()
        _out = _fig
    else:
        _out = _c.view(mo)
    mo.vstack([mo.md("**`papers_per_decade(papers)`** — the field accelerates:"), _out])
    return


@app.cell
def _(MODULES, ex, lj, mo):
    _index = {"perceptron": "29-perceptron-hopfield", "hopfield": "29-perceptron-hopfield",
              "word2vec": "30-word2vec", "lstm": "31-rnn-seq2seq", "seq2seq": "31-rnn-seq2seq",
              "bert": "32-bert-vs-gpt", "gan": "33-generative-images", "diffusion": "33-generative-images",
              "alphazero": "34-deep-rl-alphazero", "dqn": "34-deep-rl-alphazero", "rlhf": "35-rlhf-dpo",
              "dpo": "35-rlhf-dpo", "mamba": "36-mamba-ssm", "induction head": "37-interpretability",
              "backprop": "03-autodiff", "convolution": "12-cnns", "attention": "15-attention",
              "lora": "20-lora", "grpo": "26-reasoning", "mixture of experts": "17-modern-llm"}
    _titles = {s: t for _n, s, t, _p in MODULES}
    _queries = ["Multi-head attention", "LSTM cells", "diffusion models", "Mixture of Experts routing", "ELIZA"]
    _rows = []
    for _q in _queries:
        _r = lj.attempt(ex.module_for_concept, _q, _index)
        if not _r.ok:
            _rows = None
            _fail = _r
            break
        _rows.append(f"* {_q} → " + (f"**{_r.value}** ({_titles[_r.value]})" if _r.value else "not in the course"))
    mo.vstack([mo.md("**`module_for_concept`** — where do I learn about…?"),
               mo.md("\n".join(_rows)) if _rows is not None else _fail.view(mo)])
    return


@app.cell
def _(mo):
    q1 = mo.ui.radio(options=["Deep Blue beats Kasparov", "Backpropagation is popularized"],
                     label="**Check yourself 1:** which came first?")
    q2 = mo.ui.radio(options=["The Transformer", "AlphaGo beats Lee Sedol"],
                     label="**Check yourself 2:** which came first?")
    q3 = mo.ui.radio(options=["Minsky & Papert's *Perceptrons*", "The Dartmouth workshop"],
                     label="**Check yourself 3:** which came first?")
    mo.vstack([q1, q2, q3])
    return q1, q2, q3


@app.cell
def _(mo, q1, q2, q3):
    _answers = [(q1, "Backpropagation is popularized", "Backprop was popularized in 1986; Deep Blue won in 1997."),
                (q2, "AlphaGo beats Lee Sedol", "AlphaGo won in March 2016; the Transformer paper appeared in 2017."),
                (q3, "The Dartmouth workshop", "Dartmouth was 1956; *Perceptrons* came out in 1969.")]
    _fb = []
    for _q, _right, _why in _answers:
        if _q.value is None:
            continue
        _fb.append(mo.callout(mo.md(("Right. " if _q.value == _right else "Not quite. ") + _why),
                              kind="success" if _q.value == _right else "warn"))
    mo.vstack(_fb) if _fb else mo.md("")
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(28))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    * Michael Wooldridge, *A Brief History of Artificial Intelligence* (2021; UK title *The Road to
      Conscious Machines*, 2020) — a readable book-length version of this story.
    * Wikipedia, [History of artificial intelligence](https://en.wikipedia.org/wiki/History_of_artificial_intelligence)
      and [AI winter](https://en.wikipedia.org/wiki/AI_winter) — well-sourced overviews with references.
    * Jürgen Schmidhuber, [Annotated History of Modern AI and Deep Learning](https://arxiv.org/abs/2212.11279)
      (2022) — a detailed, opinionated history from one of the field's pioneers.
    * Every lesson ends with its **seminal papers**; all of them, by module, are in `docs/PAPERS.md`.

    **Next:** Module 01 — arrays, the grammar of everything that follows. Or jump to Part 10, which
    rebuilds the milestones above in JAX.
    """)
    return


if __name__ == "__main__":
    app.run()
