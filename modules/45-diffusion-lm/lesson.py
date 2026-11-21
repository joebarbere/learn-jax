import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="45 · Diffusion language models")


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
    from learnjax.data import detective_corpus

    ex = lj.load_exercises(mo.notebook_dir())
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")  # re-runs skip XLA compilation
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    _spec = importlib.util.spec_from_file_location("m45_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    RUN = M.RUNS / "mdlm"
    corpus = detective_corpus()
    tok = Tokenizer.from_file(str(corpus["tokenizer_path"]))
    return M, RUN, corpus, ex, jax, jnp, json, lj, mo, np, plt, rosetta, tok


@app.cell
def _(mo):
    mo.md(r"""
    # 45 · Diffusion language models: writing a whole passage at once (2021–2026)

    **Take first:** Module 33 (diffusion for images), Module 16 (training a GPT on the detective corpus) and
    Module 41, whose attention model is this module's opponent: same data, same size, same 98M training
    tokens.

    **Goal:** every language model so far wrote one token at a time, left to right. A **diffusion language
    model** starts from a passage of blanks and fills them in over a few rounds, many tokens per round, in any
    order, each guess seeing the words on *both* sides. That promises faster generation and text that can
    be revised. This module builds the simplest version that works, **masked diffusion** (MDLM, LLaDA),
    trains it on the detective corpus, and measures what it gives up.

    **You will:** write the noising process · write the training loss and see why it bounds the
    log-likelihood · write one denoising step and the schedules that drive it · watch the trained model fill
    in a passage · race it against Module 41's autoregressive model on quality, diversity and speed.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Diffusion took over image generation in 2020–2022 (Module 33): add Gaussian noise step by step, learn to
    remove it. Text is discrete, so there is no Gaussian to add. Austin et al.'s **D3PM** (2021) defined
    diffusion over discrete tokens; the simplest of its noise types replaces tokens by a **[MASK]** symbol,
    which makes the denoiser a BERT (2018) asked to fill in blanks at every masking rate. **SEDD** (Lou et
    al., 2024, an ICML best paper) and **MDLM** (Sahoo et al., 2024) turned this into a clean likelihood
    bound and closed much of the gap with autoregressive models. **LLaDA** (Nie et al., 2025) scaled it to
    8B parameters, comparable to Llama 3 8B on many benchmarks; **Dream 7B** started from Qwen2.5's weights.
    Then speed became the selling point: Inception's **Mercury** and Google's **Gemini Diffusion** (2025)
    reported over 1,000 tokens per second.

    *Timeline: Module 00b.*
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The forward process: hide each token with probability t

    Image diffusion blurs a picture into static. Masked diffusion does the same to text with a single
    move: pick a **masking rate** $t \in (0, 1]$ and replace each token by [MASK], independently, with
    probability $t$. At $t = 0.1$ the passage is easy to repair; at $t = 1$ it is all blanks, the "pure
    noise" every sample starts from.

    $$z_i = \begin{cases}\texttt{[MASK]} & \text{with probability } t\\ x_i & \text{otherwise}\end{cases}$$

    The [MASK] token is one extra row of the embedding table (id 8192, after the corpus's 8,192 real
    tokens). Unlike Gaussian noise it is **absorbing**: once a token is masked, it carries no trace of what
    it was.

    **Your turn:** `mask_tokens(key, x, t, mask_id)`, one $t$ per sequence.
    """)
    return


@app.cell
def _(M, corpus, ex, jax, jnp, lj, mo, np, tok):
    _x = jnp.asarray(corpus["val"][5000:5040].astype(np.int32))[None].repeat(3, 0)
    _t = jnp.array([0.2, 0.5, 0.9])
    _a = lj.attempt(ex.mask_tokens, jax.random.key(0), _x, _t, M.MASK)
    mask_ready = _a.ok
    if not _a.ok:
        _out = _a.view(mo)
    else:
        def _show(row):
            return "".join("▒" if int(i) == M.MASK else tok.decode([int(i)]) for i in row).replace("\n", " ")
        _rows = "\n".join(f"| {float(t):.1f} | {float((_a.value[i] == M.MASK).mean()):.0%} | `{_show(_a.value[i])}` |"
                          for i, t in enumerate(_t))
        _out = mo.md(f"""
    One 40-token stretch of the validation text, noised at three rates (▒ = [MASK]):

    | t | masked | z |
    |---|---|---|
    | 0 | 0% | `{_show(_x[0])}` |
    {_rows}

    At $t = 0.2$ you could fill most blanks yourself; at $t = 0.9$ almost nothing is left to go on.
    """)
    _out
    return (mask_ready,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · The loss: fill in the blanks, weighted by 1/t

    The denoiser is a Transformer with the causal mask removed (`models.py`: the same 8 layers of width 384
    as Module 41's attention model, so the race is fair). It sees $z$ and predicts every position; only the
    masked ones are scored. Sahoo et al. show that with the linear schedule (keep probability $1 - t$),
    weighting each sequence by $1/t$ turns this into an **upper bound on the negative log-likelihood** per
    token:

    $$\mathcal L = \mathbb E_{t \sim U(0,1]}\ \frac{1}{t\,T}\sum_{i:\, z_i = \texttt{[MASK]}} -\log p_\theta(x_i \mid z).$$

    Why $1/t$: at $t = 0.01$ about one token in a hundred is hidden, so each hidden token has to stand for
    the hundred-token passage; at $t = 1$ all are hidden and each counts once. Average over all rates and
    you get a number in the same units as an autoregressive model's loss, nats per token, which is what lets
    section 3 compare the two at all. (BERT's loss, which divides by the number of masks and uses one fixed
    rate, is not such a bound: it can't be used to generate from scratch.)

    `models.py` draws the $t$'s of a batch spread evenly over $(0, 1]$, not independently: MDLM's
    low-discrepancy trick, which lowers the loss's variance a lot at batch size 16.

    **Your turn:** `mdlm_loss(logits, x, z, t, mask_id)`.
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lj, mo, np):
    _k = jax.random.split(jax.random.key(1), 3)
    _x = jax.random.randint(_k[0], (2, 8), 0, 20)
    _z = jnp.where(jax.random.uniform(_k[1], (2, 8)) < 0.5, M.MASK, _x)
    _logits = jax.random.normal(_k[2], (2, 8, M.MASK + 1))
    _a = lj.attempt(ex.mdlm_loss, _logits, _x, _z, jnp.array([0.5, 0.5]), M.MASK)
    _b = lj.attempt(ex.mdlm_loss, _logits, _x, _z, jnp.array([0.25, 0.25]), M.MASK)
    loss_ready = _a.ok and _b.ok
    if not loss_ready:
        _out = (_a if not _a.ok else _b).view(mo)
    else:
        _ratio = float(_b.value) / float(_a.value)
        _out = mo.md(f"""
    On random logits over the {M.MASK + 1:,}-token vocabulary, your loss is **{float(_a.value):.3f}** at $t = 0.5$
    (a guess at uniform costs $\\ln {M.MASK + 1:,} = {np.log(M.MASK + 1):.2f}$ per masked token, and about half
    the positions are masked). Halving $t$ with the same masks doubles it: ratio **{_ratio:.2f}**
    {"✅" if abs(_ratio - 2) < 1e-3 else "❌ (expected 2)"}.
    """)
    _out
    return (loss_ready,)


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "masked diffusion loss (MDLM)", solves="mdlm_loss")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Training: the same model, the same data, a different question

    `train.py` trains the denoiser exactly like Module 41's attention model: 12,000 steps of 16 × 512 tokens
    (98M tokens, about 4.5 passes over the corpus), AdamW at 1e-3 with warmup and cosine decay. The only
    differences are the attention mask (none) and the task (fill in blanks at a random rate instead of
    predict the next token). About 30 minutes on the RX 7600 XT.

        containers/run.sh python modules/45-diffusion-lm/train.py

    Validation uses the bound from section 2, averaged over evenly spread masking rates (8 per window while
    training, 32 for the final number), against Module 41's validation loss on the same text.
    """)
    return


@app.cell
def _(M, RUN, json, mo, plt):
    _m = M.read_metrics(RUN / "metrics.jsonl")
    _ar = M.read_metrics(M.RUNS.parent / "41-linear-attention-hybrids" / "attention" / "metrics.jsonl")
    if not _m:
        _out = mo.md("*No run yet: run `train.py` (about 30 minutes) to see the curves here.*")
    else:
        _final = json.loads((RUN / "final.json").read_text())["val_bound_64x32"] if (RUN / "final.json").exists() \
            else _m[-1]["val"]
        _fig, _ax = plt.subplots(figsize=(7, 3.2))
        _ax.plot([r["step"] for r in _m], [r["val"] for r in _m], label="diffusion (MDLM): bound on NLL")
        if _ar:
            _ax.plot([r["step"] for r in _ar], [r["val"] for r in _ar], label="autoregressive (Module 41): NLL")
        _ax.set_xlabel("step")
        _ax.set_ylabel("validation nats / token")
        _ax.set_ylim(3.2, 6)
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _arv = _ar[-1]["val"] if _ar else float("nan")
        _gap = _final - _arv
        _out = mo.vstack([_fig, mo.md(f"""
    Final: the diffusion model's bound is **{_final:.3f}** nats per token (64 windows × 32 masking rates);
    the autoregressive model scores **{_arv:.3f}**. The gap is **{_gap:.2f} nats**, a perplexity of
    {2.718281828 ** _final:.0f} against {2.718281828 ** _arv:.0f}.

    One thing makes the gap look bigger than it is, one explains it, and one makes it real:

    * The diffusion number is an **upper bound**; its true log-likelihood is somewhat better, by an amount
      nobody can compute exactly.
    * The model has had a harder job per parameter: it learned to predict any token from any subset of
      the others, not just from the left. Nie et al. (2024) fit scaling laws showing masked diffusion needs about 16× the
      training compute of an autoregressive model to reach the same likelihood; at 98M tokens it is far
      from caught up.
    * But the bound is also what the training minimized, and on the same budget it is
      {"much " if _gap > 0.5 else ""}worse. For a small model on a small corpus, left to right is the
      easier way to learn language.
    """)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Sampling: reveal the surest tokens first

    Generation runs the forward process backwards. Start with the prompt followed by blanks. At each step,
    run the denoiser once over the whole passage, which proposes a token for every blank; **keep a few** of
    those proposals and leave the rest masked for the next step. With 256 blanks and 64 steps you keep 4
    per step, so 64 model calls instead of an autoregressive model's 256.

    Which ones to keep? Plain MDLM keeps a **random** few, which is the reverse process the loss was
    derived for. LLaDA keeps the ones the model is **most confident** about ("low-confidence remasking"):
    easy words first ("the", the end of a name), hard decisions later, when more context is visible.

    **Your turn:** `unmask_step(key, logits, z, n_reveal, mask_id, temperature, by_confidence)` and
    `reveal_schedule(n_masked, steps)`.
    """)
    return


@app.cell
def _(M, ex, jax, jnp, lj, mo):
    _logits = jnp.full((1, 6, M.MASK + 1), -5.0).at[0, 1, 7].set(8.0).at[0, 4, 9].set(6.0)
    _z = jnp.array([[3, M.MASK, M.MASK, 5, M.MASK, M.MASK]])
    _a = lj.attempt(ex.unmask_step, jax.random.key(0), _logits, _z, 2, M.MASK)
    _s = lj.attempt(ex.reveal_schedule, 256, 64)
    unmask_ready = _a.ok and _s.ok
    if not unmask_ready:
        _out = (_a if not _a.ok else _s).view(mo)
    else:
        _good = [int(v) for v in _a.value[0]] == [3, 7, M.MASK, 5, 9, M.MASK]
        _out = mo.md(f"""
    Four blanks, two of them near-certain (positions 1 and 4): your step with `n_reveal=2` gives
    `{[int(v) if int(v) != M.MASK else "MASK" for v in _a.value[0]]}` {"✅ the two sure ones" if _good else "❌ expected [3, 7, MASK, 5, 9, MASK]"}.
    `reveal_schedule(256, 64)` starts `{_s.value[:6]}…` and sums to {sum(_s.value)}.
    """)
    _out
    return (unmask_ready,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Watch it write

    The trained model, given the first 24 tokens of a validation passage, fills in the next 64 in 8 steps
    of 8 tokens, most confident first. Each row is the passage after one step.
    """)
    return


@app.cell
def _(M, RUN, corpus, ex, jax, jnp, mask_ready, mo, np, tok, unmask_ready):
    if not (RUN / "params.npz").exists():
        _out = mo.md("*No trained model yet: run `train.py`.*")
    elif not (mask_ready and unmask_ready):
        _out = mo.md("*Finish `mask_tokens` and `unmask_step` above to watch the model write.*")
    else:
        _params = M.load(RUN)
        _fwd = jax.jit(lambda p, z: M.forward(M._bf16(p), z))
        _prompt = jnp.asarray(corpus["val"][20000:20024].astype(np.int32))[None]
        _z = jnp.concatenate([_prompt, jnp.full((1, 64), M.MASK, jnp.int32)], 1)
        _key = jax.random.key(3)
        _lines = []
        for _step in range(8):
            _key, _sub = jax.random.split(_key)
            _z = ex.unmask_step(_sub, _fwd(_params, _z), _z, 8, M.MASK)
            _txt = "".join("▒" if int(i) == M.MASK else tok.decode([int(i)]) for i in _z[0, 24:])
            _lines.append(f"{_step + 1}. `{_txt.replace(chr(10), ' ').replace('`', chr(39))}`")
        _out = mo.md(f"**Prompt:** `{tok.decode([int(i) for i in _prompt[0]]).replace(chr(10), ' ')}`\n\n"
                     + "\n".join(_lines))
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Blocks: a little left-to-right helps

    Confidence ordering has a failure: the surest guesses are often short, generic words, and a passage
    built mostly of them, decided early and everywhere at once, can lose the thread. LLaDA's fix is
    **semi-autoregressive** sampling: cut the answer into blocks, fill each block with diffusion, and go
    through the blocks left to right. Block diffusion (BD3-LM, Arriola et al. 2025) trains for exactly
    this, and can keep a KV cache for finished blocks.

    **Your turn:** `block_order(T, prompt_len, block)`.
    """)
    return


@app.cell
def _(ex, lj, mo):
    _a = lj.attempt(ex.block_order, 32 + 256, 32, 32)
    _out = _a.view(mo) if not _a.ok else mo.md(
        f"256 tokens after a 32-token prompt, in blocks of 32: {len(_a.value)} blocks, "
        f"`{_a.value[:3]}` … `{_a.value[-1]}`.")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What left to right can't do: fill in the middle

    An autoregressive model only ever conditions on the left. The diffusion model was trained on blanks
    anywhere, so it can **infill**: hide 16 tokens in the middle of a passage and it uses the words on both
    sides. The same model can also be made to work with the left side only, by masking everything after the
    gap too. Below, 64 validation passages of 64 tokens with tokens 24–39 hidden.
    """)
    return


@app.cell
def _(M, RUN, corpus, ex, jax, jnp, mask_ready, mo, np, tok, unmask_ready):
    if not (RUN / "params.npz").exists():
        _out = mo.md("*No trained model yet: run `train.py`.*")
    elif not (mask_ready and unmask_ready):
        _out = mo.md("*Finish `unmask_step` above to see the infilling.*")
    else:
        _params = M.load(RUN)
        _fwd = jax.jit(lambda p, z: M.forward(M._bf16(p), z))
        _starts = np.random.default_rng(11).integers(0, len(corpus["val"]) - 64, 64)
        _x = jnp.asarray(np.stack([corpus["val"][s:s + 64] for s in _starts]).astype(np.int32))
        _gap = slice(24, 40)
        _both = _x.at[:, _gap].set(M.MASK)
        _left = _x.at[:, 24:].set(M.MASK)
        _nll, _fills = {}, {}
        for _name, _z in (("both sides", _both), ("left side only", _left)):
            _lp = jax.nn.log_softmax(_fwd(_params, _z)[:, _gap, :M.MASK].astype(jnp.float32), -1)
            _nll[_name] = float(-jnp.take_along_axis(_lp, _x[:, _gap, None], -1).mean())
            _zz, _key = _z[:1], jax.random.key(5)
            for _i in range(16):                                 # fill the gap one token per step, surest first
                _key, _sub = jax.random.split(_key)
                _inside = jnp.zeros((64,), bool).at[_gap].set(True)
                _new = ex.unmask_step(_sub, _fwd(_params, _zz), jnp.where(_inside, _zz, -1), 1, M.MASK,
                                      temperature=0.7)
                _zz = jnp.where(_inside, _new, _zz)
            _fills[_name] = _zz[0, _gap]

        def _fmt(row, fill):
            _s = lambda a: tok.decode([int(i) for i in a]).replace("\n", " ").replace("`", "'")
            return f"{_s(row[:24])} **[{_s(fill)}]** {_s(row[40:])}"
        _out = mo.md(f"""
    | context visible | loss on the 16 hidden tokens (nats / token, one model call) |
    |---|---|
    | left side only | {_nll["left side only"]:.2f} |
    | both sides | **{_nll["both sides"]:.2f}** |

    One passage, the hidden 16 tokens in brackets, filled one token per step, surest first:

    * **original:** {_fmt(_x[0], _x[0, _gap])}
    * **filled, both sides:** {_fmt(_x[0], _fills["both sides"])}
    * **filled, left only:** {_fmt(_x[0], _fills["left side only"])}

    Seeing the right-hand side lowers the loss on the gap by {_nll["left side only"] - _nll["both sides"]:.2f} nats
    per token.
    Code editors want exactly this ("fill in the middle"); autoregressive code models get it by training on
    passages rearranged as prefix, suffix, then middle.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · The race: quality, diversity, speed

    `train.py` ends by generating 256 tokens after each of 8 real 32-token prompts, with every sampler, and
    scoring the results three ways:

    * **AR loss** ("generative perplexity"): Module 41's autoregressive model reads the generated text and
      reports its loss. Lower means "looks like the corpus" to that model.
    * **Unigram entropy** of the generated tokens: how varied the vocabulary is.
    * **Repeated 4-grams**: the share of 4-token sequences that already occurred earlier in the same
      sample. Real prose repeats almost none.

    The first number alone is easy to game: a sampler that repeats "I was glad to" forever scores well,
    because each repetition is easy to predict. So read the columns together, against real text.
    """)
    return


@app.cell
def _(RUN, json, mo, tok):
    if not (RUN / "sampling.json").exists():
        _out = mo.md("*Run `train.py` to fill in this table.*")
    else:
        sampling = json.loads((RUN / "sampling.json").read_text())
        _real = sampling["real text"]
        _rows = "\n".join(
            f"| {k} | {v['gen_nll']:.2f} | {v['entropy']:.2f} | {v.get('repeat_4grams', float('nan')):.1%} | "
            f"{v['seconds_per_sample'] * 1000:.0f} ms |" if k != "real text" else
            f"| **real text** | **{v['gen_nll']:.2f}** | **{v['entropy']:.2f}** | "
            f"**{v.get('repeat_4grams', float('nan')):.1%}** | — |"
            for k, v in sampling.items())
        _c64, _r64 = sampling["mdlm 64 steps"], sampling["mdlm 64 steps, random order"]
        _ar = sampling["autoregressive (Module 41)"]
        _b256 = sampling.get("mdlm 256 steps, blocks of 32")

        def _txt(k, n=110):
            s = sampling[k]["sample"]
            return tok.decode(s[32:32 + n]).replace("\n", " ").replace("`", "'")

        _out = mo.md(f"""
    | sampler | AR loss | unigram entropy | repeated 4-grams | time / sample (batch 8) |
    |---|---|---|---|---|
    {_rows}

    What the table says:

    * **Confidence ordering wins the loss column by cheating.** At 64 steps it scores {_c64['gen_nll']:.2f},
      better than real text ({_real['gen_nll']:.2f}), but its vocabulary is narrow (entropy
      {_c64['entropy']:.2f} vs {_real['entropy']:.2f}) and {_c64['repeat_4grams']:.0%} of its 4-grams are
      repeats, against {_real['repeat_4grams']:.1%} in real text. It found a loop of safe words.
    * **Random order is honest but weak.** Its entropy matches real text ({_r64['entropy']:.2f}) and it
      hardly repeats itself, but the AR model finds it hard to read ({_r64['gen_nll']:.2f}): the model is
      sampling from what it actually learned, and section 3 showed that is about a nat per token worse.
      More steps don't rescue it.
    * **The autoregressive model is the only one close to real text in every column** ({_ar['gen_nll']:.2f},
      {_ar['entropy']:.2f}, {_ar['repeat_4grams']:.1%}). At this size and budget, it writes better.
    * **Speed.** Diffusion at 64 steps takes {_c64['seconds_per_sample'] * 1000:.0f} ms per sample against
      {_ar['seconds_per_sample'] * 1000:.0f} ms. Count calls rather than trust the ratio: 64 full-passage
      model calls against 256, and this AR sampler re-reads the whole prefix each step (no KV cache,
      Module 18), which flatters diffusion. With a cache, each AR step is cheap but there are still 256 of
      them in sequence; that sequential chain is what diffusion attacks.
    {"* **Blocks of 32 at 256 steps** read best of the diffusion samplers (" + f"{_b256['gen_nll']:.2f}" + ") and repeat less than full-passage confidence sampling, but still " + f"{_b256['repeat_4grams']:.0%}" + " of 4-grams." if _b256 else ""}

    The same prompt, three samplers (first 110 tokens):

    * **diffusion, 64 steps, confident first:** `{_txt("mdlm 64 steps")}`
    * **diffusion, 64 steps, random order:** `{_txt("mdlm 64 steps, random order")}`
    * **autoregressive:** `{_txt("autoregressive (Module 41)")}`
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Where it stands (late 2026)

    | model | what it showed | source |
    |---|---|---|
    | LLaDA 8B (Renmin Univ. & Ant, 2025) | masked diffusion from scratch, 2.3T tokens, comparable to Llama 3 8B on many benchmarks; better at "reversal" tasks | arXiv 2502.09992 |
    | Dream 7B (HKU & Huawei, 2025) | initialized from Qwen2.5-7B's autoregressive weights instead of from scratch | arXiv 2508.15487 |
    | BD3-LM (Arriola et al., 2025) | block diffusion: autoregressive across blocks, diffusion within, with a KV cache | arXiv 2503.09573 |
    | Mercury Coder (Inception, 2025) | commercial diffusion coder, 1,109 / 737 tokens/s (Mini / Small) on an H100 | arXiv 2506.17298 |
    | Gemini Diffusion (Google, May 2025) | experimental; Google reported 1,479 tokens/s; no paper | Google DeepMind |
    | Seed Diffusion Preview (ByteDance, 2025) | code model, 2,146 tokens/s reported | arXiv 2508.02193 |
    | LLaDA-MoE, LLaDA 2.0 (2025) | mixture-of-experts diffusion models (Module 17's experts, Module 45's decoding) | arXiv 2509.24389, 2512.15745 |

    The pattern matches this module at a much larger scale: diffusion models trade some quality per
    parameter for decoding in far fewer sequential steps, and the products that ship them sell speed,
    mostly on code, where a fixed-length edit filled in parallel is natural. Every published speed above is
    the vendor's own measurement.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"a": "Its loss on generated text, scored by an autoregressive model, is lowest",
                 "b": "Its loss is low, its vocabulary is as varied as real text, and it rarely repeats itself",
                 "c": "It needs the fewest model calls"},
        label="**Check yourself:** what should convince you a sampler writes good text?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Look at section 6: confidence sampling beat real text on that column by repeating "
                              "itself. One number is easy to game."), kind="warn"),
        "b": mo.callout(mo.md("Right. Quality, diversity and repetition together; real text sets the target for "
                              "all three."), kind="success"),
        "c": mo.callout(mo.md("That's speed, not quality: 4 steps is the fastest sampler in the table and "
                              "the hardest to read."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(45))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Sahoo et al., [MDLM project page and code](https://s-sahoo.com/mdlm/) — the training recipe used here,
      with the PyTorch reference implementation.
    * Nie et al., [LLaDA](https://ml-gsai.github.io/LLaDA-demo/) — demos of an 8B masked diffusion model,
      including the block-wise sampler of section 5.
    * Sander Dieleman, [Diffusion language models](https://sander.ai/2023/01/09/diffusion-language.html) —
      why language is hard for diffusion, from a diffusion researcher, written before most of this module's
      papers (2023).

    **Next:** Module 46 (planned): weights by hand, compiling, merging and editing networks.
    """)
    return


if __name__ == "__main__":
    app.run()
