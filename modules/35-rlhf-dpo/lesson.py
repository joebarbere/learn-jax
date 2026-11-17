import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="35 · RLHF and DPO")


@app.cell
def _():
    import marimo as mo
    import html
    import importlib.util
    import json
    import sys
    import time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import rosetta
    from learnjax import llama

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")   # recompiles hit the disk cache
    ex = lj.load_exercises(mo.notebook_dir())

    def _load(name):
        _spec = importlib.util.spec_from_file_location(f"m35_{name}", mo.notebook_dir() / f"{name}.py")
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)
        return _mod

    P = _load("prefs")
    RUNS = Path("/course/runs/35-rlhf-dpo")

    def code(text):
        """Inline code that survives markdown (marimo treats <|im_end|> as HTML even in backticks)."""
        return f"<code>{html.escape(str(text))}</code>"

    return P, RUNS, code, ex, jax, jnp, json, lj, llama, mo, np, plt, rosetta, time


@app.cell
def _(mo):
    mo.md(r"""
    # 35 · Learning from preferences: RLHF and DPO (2017–2023)

    **Goal:** teach a language model what people *prefer*, not just what text comes next — the step that
    turned GPT-3 into InstructGPT and ChatGPT. You'll rebuild the 2017–2022 recipe (preference pairs → a
    **reward model** → optimize the policy against it) and the 2023 shortcut that replaced most of it,
    **Direct Preference Optimization (DPO)**, on SmolLM2-360M-Instruct with LoRA, on your RX 7600 XT.

    **Take first:** Module 20 (LoRA) and Module 26 (GRPO: RL on a language model). This module reuses
    both — Module 20's LoRA layer and Module 26's KV-cached LoRA sampler, loaded by path.

    **You will:** build a preference dataset with a programmatic "labeler" · write the Bradley–Terry loss and
    train a reward model on SmolLM2-135M · write the DPO loss from log-probability ratios and train a policy
    with it · measure win rate and KL drift · do RLHF-style rejection sampling against your reward model and
    compare it with DPO · watch the reward model get *gamed* (Goodhart's law) and patch it.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    The core statistics are old: Bradley and Terry (1952) modelled the chance that one thing is preferred
    over another as a logistic function of a hidden "score" difference. In 2017 Christiano and colleagues at
    OpenAI and DeepMind used exactly that model to learn a reward from people comparing pairs of short video
    clips, teaching a simulated robot to do a backflip without anyone writing its reward function. Stiennon
    et al. (2020) applied it to summarization, and InstructGPT (Ouyang et al., 2022) scaled the pipeline —
    supervised fine-tuning, a reward model, then PPO — so well that ChatGPT followed that November.
    Constitutional AI (2022) replaced some human labels with AI feedback, and DPO (Rafailov et al., 2023)
    showed you could skip the separate reward model and RL loop entirely.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Why next-token prediction isn't enough

    A pretrained model imitates its training text. Ask it a question and it continues the way text on the
    internet continues — sometimes helpfully, often with an essay, sometimes not at all. InstructGPT's
    authors summarized what people actually want as **helpful, honest and harmless**, and noticed that
    none of those is "the most likely next token".

    You can't easily *write down* a loss for "helpful", but people can reliably say which of **two**
    answers is better. That's the whole trick of the 2017–2022 era:

    1. **Supervised fine-tuning (SFT):** show the model good example answers (Module 20 did this).
    2. **Reward model (RM):** collect comparisons "answer A ≻ answer B" and fit a network that scores answers
       so the preferred one scores higher.
    3. **RL against the RM:** optimize the policy to get high reward, with a KL penalty that keeps it close to
       the SFT model — InstructGPT used PPO; Module 26's GRPO is a lighter cousin.

    InstructGPT's headline: labelers preferred a **1.3B**-parameter InstructGPT over the **175B** GPT-3 —
    alignment, not just size, decided what people liked. Then in 2023 DPO folded steps 2–3 into one
    supervised-looking loss. We'll build both.
    """)
    return


@app.cell
def _(P, code, mo):
    TRAIN, TEST = P.split_items()
    _demo = [("The capital of Norway is Oslo.", "Oslo"),
             ("The capital of Norway is Oslo, a vibrant city on the Oslofjord known for its museums, its "
              "Viking ship collection, its parks and its architecture.", "Oslo"),
             ("Oslo.", "Oslo"),
             ("The capital of Norway is Bergen.", "Oslo"),
             ("Yes.", "Oslo")]
    _rows = "\n".join(f"    | {code(t)} | {P.oracle_score(t, k):.2f} |" for t, k in _demo)
    mo.md(f"""
    ## 2 · The task, and a labeler we can afford

    Real RLHF needs thousands of human comparisons. We simulate the labeler with a program — an **oracle** —
    so every number in this lesson is measurable. The prompts are {len(TRAIN) + len(TEST)} short factual
    questions (capitals, opposites, colours), split {len(TRAIN)} train / {len(TEST)} held out, asked with no
    system prompt. The oracle's taste:

    * **2 points** for containing the right answer;
    * **up to 1 point for style**: a complete, to-the-point answer of 4–15 words scores 1; longer answers lose
      credit until 40 words; curt 1–3-word answers get only 0.4.

    | answer to "What is the capital of Norway?" | oracle |
    |---|---|
{_rows}

    SmolLM2-360M-Instruct already knows these facts (it's right about 98% of the time), but it **rambles** —
    about 40% of its sampled answers are still going after 48 tokens. So the preference we're teaching is
    "answer, then stop". Keep the curt-answer rule in mind: hardly any training sample is curt, so a reward
    model can't learn that rule from the data. Section 7 is about what happens next.
    """)
    return TEST, TRAIN


@app.cell
def _(llama, mo, time):
    _t = time.perf_counter()
    params, cfg, tok = llama.load(llama.SMOLLM_360M)
    params_rm, cfg_rm, _ = llama.load(llama.SMOLLM_135M)
    mo.md(f"Loaded the policy (SmolLM2-360M-Instruct, {llama.count_params(params) / 1e6:.0f}M) and the "
          f"reward-model base (SmolLM2-135M, {llama.count_params(params_rm) / 1e6:.0f}M; same tokenizer) "
          f"in {time.perf_counter() - _t:.1f} s.")
    return cfg, cfg_rm, params, params_rm, tok


@app.cell
def _(P, TEST, TRAIN, cfg, jax, mo, np, params, time, tok):
    K = 4
    _t = time.perf_counter()
    samples = P.sample_answers(params, {}, cfg, tok, TRAIN, K, jax.random.key(1))
    pairs = P.build_pairs(TRAIN, *samples, K)
    test_samples = P.sample_answers(params, {}, cfg, tok, TEST, K, jax.random.key(2))
    test_pairs = P.build_pairs(TEST, *test_samples, K)
    _texts = samples[0]
    _words = np.asarray([len(P.WORD.findall(t)) for t in _texts])
    _sc = P.oracle_scores(_texts, TRAIN, K)
    mo.md(f"""
    ### Building the preference data

    For each training question we sampled **{K} answers** from the base model (temperature 1.0), scored them
    with the oracle, and kept every pair whose scores differ by at least 0.25 as (chosen ≻ rejected):
    **{len(pairs['gap'])} training pairs** and **{len(test_pairs['gap'])} held-out pairs**, in
    {time.perf_counter() - _t:.1f} s. The base model's answers average **{_words.mean():.0f} words**
    (oracle {_sc.mean():.2f} / 3), and {np.mean(_sc >= 2) * 100:.0f}% contain the right answer.
    """)
    return K, pairs, samples, test_pairs, test_samples


@app.cell
def _(P, TRAIN, code, mo, pairs, samples):
    _texts = samples[0]
    _K = len(_texts) // len(TRAIN)
    _seen, _rows = set(), []
    for _j in range(len(pairs["gap"])):
        _i = int(pairs["item"][_j])
        if _i in _seen:
            continue
        _seen.add(_i)
        _sc = [(P.oracle_score(_texts[_i * _K + _k], TRAIN[_i]["keyword"]), _texts[_i * _K + _k]) for _k in range(_K)]
        _hi, _lo = max(_sc), min(_sc)
        _rows.append(f"    | {TRAIN[_i]['question']} | {code(_hi[1][:110])} ({_hi[0]:.2f}) | {code(_lo[1][:110])} ({_lo[0]:.2f}) |")
        if len(_rows) == 4:
            break
    _table = "\n".join(_rows)
    mo.md(f"""
    | question | chosen (oracle) | rejected (oracle) |
    |---|---|---|
{_table}

    Almost every pair has the same *content* — what differs is how long the model keeps talking.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · The reward model: Bradley–Terry (1952)

    Give every answer a hidden score $r$. Bradley and Terry's model of a comparison is

    $$P(\text{chosen} \succ \text{rejected}) = \sigma\big(r(x, y_c) - r(x, y_r)\big)$$

    so fitting $r$ is just logistic regression on *differences*: minimize
    $-\log\sigma(r_c - r_r)$ over the labelled pairs ([`jax.nn.log_sigmoid`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.log_sigmoid.html#jax.nn.log_sigmoid) computes $\log\sigma$ without
    underflowing). Only differences matter — adding a constant to every
    score changes nothing — which is why reward-model scores have no natural zero.

    Our reward model is SmolLM2-135M with LoRA adapters on its attention, plus one new linear "head" that
    reads the hidden state at the **last token of the answer** and outputs a single number. (InstructGPT's
    reward model was a 6B GPT-3 built the same way.)
    """)
    return


@app.cell
def _(ex, jnp, lj, mo):
    _bt = lj.attempt(ex.bradley_terry_loss, jnp.array([2.0, 0.0]), jnp.array([0.0, 0.0]))
    _acc = lj.attempt(ex.pairwise_accuracy, jnp.array([1.0, 2.0, 0.0]), jnp.array([0.0, 3.0, -1.0]))
    bt_ready = _bt.ok and _acc.ok
    mo.vstack([
        mo.md("**Your turn — `bradley_terry_loss`.** Margins 2 and 0 → mean of log(1+e⁻²) and log 2 ≈ 0.410:"),
        _bt.view(mo),
        mo.md("**`pairwise_accuracy`** for scores (1 vs 0), (2 vs 3), (0 vs −1) → 2/3:"),
        _acc.view(mo),
    ])
    return (bt_ready,)


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "Bradley–Terry reward-model loss", solves="bradley_terry_loss")
    return


@app.cell
def _(P, bt_ready, cfg_rm, ex, jax, mo, pairs, params_rm, time):
    rm, rm_losses = None, []
    if bt_ready:
        _t = time.perf_counter()
        rm, rm_losses = P.train_reward_model(params_rm, cfg_rm, pairs, ex.bradley_terry_loss, 80, 8, jax.random.key(3))
        _out = mo.md(f"Trained the reward model for 80 steps (8 pairs each) with **your** Bradley–Terry loss in "
                     f"{time.perf_counter() - _t:.1f} s.")
    else:
        _out = mo.callout(mo.md("Finish `bradley_terry_loss` and `pairwise_accuracy` to train the reward model."),
                          kind="info")
    _out
    return rm, rm_losses


@app.cell
def _(P, cfg_rm, ex, mo, np, params_rm, plt, rm, rm_losses, test_pairs):
    rm_acc = None
    if rm is not None:
        _rc, _rr = P.rm_pair_scores(rm, params_rm, cfg_rm, test_pairs)
        rm_acc = ex.pairwise_accuracy(_rc, _rr)
        _fig, (_a, _b) = plt.subplots(1, 2, figsize=(9, 3))
        _a.plot(rm_losses, alpha=0.4)
        _a.plot(np.convolve(rm_losses, np.ones(10) / 10, mode="valid"), lw=2)
        _a.axhline(np.log(2), color="gray", ls=":", label="chance (log 2)")
        _a.set(title="reward-model training loss", xlabel="step", ylabel="Bradley–Terry loss")
        _a.legend()
        _b.hist(_rc - _rr, bins=25)
        _b.axvline(0, color="gray", ls=":")
        _b.set(title="held-out pairs: r(chosen) − r(rejected)", xlabel="margin")
        _fig.tight_layout()
        _out = mo.vstack([mo.md(f"**Held-out pairwise accuracy: {rm_acc * 100:.1f}%** on "
                                f"{len(_rc)} pairs from questions the reward model never saw (chance = 50%)."),
                          _fig])
    else:
        _out = mo.md("")
    _out
    return (rm_acc,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · The classic pipeline: optimize against the reward model

    With a reward model in hand, InstructGPT's step 3 maximizes

    $$\mathbb{E}_{y\sim\pi}\big[r(x,y)\big] - \beta\,\mathrm{KL}\big(\pi(\cdot|x)\,\|\,\pi_\text{ref}(\cdot|x)\big)$$

    with PPO (Schulman et al., 2017): sample answers, score them, push up the log-probability of the
    high-scoring ones, and stay close to the reference model so the policy doesn't wander off into text the
    reward model has never seen. In practice that means four models in memory (policy, reference, reward
    model, and a value network), lots of sampling, and notoriously finicky hyperparameters.

    We won't rebuild PPO here — Module 26 already did the modern, value-free version (GRPO). For this lesson
    the cheap stand-in is **rejection sampling** ("best-of-n"): sample $n$ answers, keep the one the
    reward model likes best (Section 6). It optimizes the same objective without any training, at $n\times$
    the inference cost.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · DPO: the reward model was inside the policy all along

    Rafailov et al. (2023) noticed that the KL-regularized objective above has a closed-form optimum:

    $$\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_\text{ref}(y|x)\,\exp\!\big(r(x,y)/\beta\big)$$

    Solve that for the reward: $r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta\log Z(x)$.
    Put it into the Bradley–Terry loss and the awkward $Z(x)$ cancels (it's the same for both answers to a
    prompt). What's left is a loss on the **policy itself**:

    $$\mathcal{L}_\text{DPO} = -\log\sigma\Big(\beta\log\frac{\pi(y_c|x)}{\pi_\text{ref}(y_c|x)} -
      \beta\log\frac{\pi(y_r|x)}{\pi_\text{ref}(y_r|x)}\Big)$$

    No reward model, no sampling during training, no RL — just the summed log-probabilities of the chosen
    and rejected answers under the policy and a frozen reference. The term $\beta\log\frac{\pi}{\pi_\text{ref}}$
    is the **implicit reward**. With LoRA, the reference model is free: it's the same weights with the
    adapters switched off (exactly how Module 26 got its KL reference).
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _logits = jax.random.normal(jax.random.key(0), (1, 3, 5))
    _slp = lj.attempt(ex.sequence_logprob, _logits, jnp.array([[1, 4, 2]]), jnp.array([[1.0, 1.0, 0.0]]))
    _ir = lj.attempt(ex.implicit_reward, jnp.array([-3.0]), jnp.array([-5.0]), 0.1)
    _z = jnp.array([-10.0])
    _dpo = lj.attempt(ex.dpo_loss, _z, _z - 3, _z, _z - 3, 0.1)
    dpo_ready = _slp.ok and _ir.ok and _dpo.ok
    mo.vstack([
        mo.md("**Your turn — `sequence_logprob`** (2 real tokens, 1 padding):"), _slp.view(mo),
        mo.md("**`implicit_reward`**: β = 0.1, log π = −3, log π_ref = −5 → 0.2:"), _ir.view(mo),
        mo.md("**`dpo_loss`** when the policy *is* the reference (every margin 0) → log 2 ≈ 0.693:"), _dpo.view(mo),
    ])
    return (dpo_ready,)


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "DPO loss", solves=("dpo_loss", "implicit_reward"))
    return


@app.cell
def _(P, cfg, dpo_ready, ex, jax, mo, pairs, params, time):
    dpo_adapters, dpo_hist, seq_lp = None, [], None
    BETA = 0.1
    if dpo_ready:
        _t = time.perf_counter()
        dpo_adapters, dpo_hist, seq_lp = P.train_dpo(params, cfg, pairs, ex.sequence_logprob, ex.dpo_loss, BETA,
                                                     40, 8, jax.random.key(4))
        _out = mo.md(f"Trained LoRA adapters on SmolLM2-360M-Instruct with **your** DPO loss: 40 steps × 8 pairs, "
                     f"β = {BETA}, in {time.perf_counter() - _t:.1f} s (including the one-off pass that computes "
                     f"the reference model's log-probs).")
    else:
        _out = mo.callout(mo.md("Finish `sequence_logprob`, `implicit_reward` and `dpo_loss` to train with DPO."),
                          kind="info")
    _out
    return BETA, dpo_adapters, dpo_hist, seq_lp


@app.cell
def _(dpo_hist, mo, np, plt):
    if dpo_hist:
        _fig, _ax = plt.subplots(1, 3, figsize=(10, 2.8))
        for _a, _k, _t in zip(_ax, ("loss", "margin", "acc"),
                              ("DPO loss", "implicit-reward margin (chosen − rejected)", "pairs ranked correctly"),
                              strict=True):
            _v = np.asarray([h[_k] for h in dpo_hist])
            _a.plot(_v, alpha=0.35)
            _a.plot(np.arange(4, len(_v)), np.convolve(_v, np.ones(5) / 5, mode="valid"), lw=2)
            _a.set(title=_t, xlabel="step")
        _ax[0].axhline(np.log(2), color="gray", ls=":")
        _fig.tight_layout()
        _out = mo.vstack([_fig, mo.md(
            "The loss starts at exactly log 2 (policy = reference), and the implicit reward learns to separate "
            "chosen from rejected answers — on the training pairs. The real test is new questions.")])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(P, TEST, cfg, code, dpo_adapters, ex, jax, lj, mo, np, params, seq_lp, tok):
    base_texts = P.greedy_answers(params, {}, cfg, tok, TEST)
    base_scores = P.oracle_scores(base_texts, TEST)
    dpo_scores, dpo_eval = None, {}
    _win = lj.attempt(ex.win_rate, [3, 1, 2], [1, 2, 2])
    if dpo_adapters is not None and _win.ok:
        _dpo_texts = P.greedy_answers(params, dpo_adapters, cfg, tok, TEST)
        dpo_scores = P.oracle_scores(_dpo_texts, TEST)
        _kl, _ = P.kl_to_reference(params, dpo_adapters, cfg, tok, TEST, jax.random.key(5), seq_lp)
        _wb = np.mean([len(P.WORD.findall(t)) for t in base_texts])
        _wd = np.mean([len(P.WORD.findall(t)) for t in _dpo_texts])
        dpo_eval = {"win": ex.win_rate(dpo_scores, base_scores), "kl": _kl, "words_base": _wb, "words_dpo": _wd}
        _rows = "\n".join(f"    | {it['question']} | {code(b[:90])} | {code(d[:90])} |"
                          for it, b, d in list(zip(TEST, base_texts, _dpo_texts, strict=True))[:6])
        _out = mo.md(f"""
    ### Held-out questions, greedy answers

    | question | base model | after DPO |
    |---|---|---|
{_rows}

    | | base | DPO |
    |---|---|---|
    | mean oracle score (of 3) | {base_scores.mean():.2f} | **{dpo_scores.mean():.2f}** |
    | mean words | {_wb:.0f} | {_wd:.0f} |

    **Win rate of DPO over the base model: {dpo_eval['win'] * 100:.0f}%** (ties count half) on
    {len(TEST)} questions it never trained on. **KL(π ‖ π_ref) ≈ {_kl:.1f} nats per answer** — how far the
    policy moved from the reference, estimated from its own samples. A bigger KL buys more change and costs
    more risk; β is the dial (`train.py` sweeps it).
    """)
    else:
        _out = mo.vstack([mo.md("**`win_rate`** for (3 vs 1), (1 vs 2), (2 vs 2) → 0.5:"), _win.view(mo)])
    _out
    return base_scores, base_texts, dpo_eval, dpo_scores


@app.cell
def _(ex, lj, mo, np):
    _bon = lj.attempt(ex.best_of_n, np.array([[0.1, 0.9, 0.5, 2.0]]), 3)
    bon_ready = _bon.ok
    mo.vstack([
        mo.md(r"""
    ## 6 · RLHF the cheap way: best-of-n against the reward model

    Sample $n$ answers, ask the reward model which is best, keep that one. As $n$ grows you're optimizing
    harder against the reward model — the same pressure PPO applies, just at inference time. (OpenAI's
    WebGPT and many reward-model papers used exactly this as their baseline.)

    **Your turn — `best_of_n`**: scores [0.1, 0.9, 0.5, 2.0] using the first 3 samples → index 1:
    """),
        _bon.view(mo),
    ])
    return (bon_ready,)


@app.cell
def _(P, TEST, bon_ready, cfg, cfg_rm, dpo_scores, ex, jax, mo, params, params_rm, plt, rm, time, tok):
    bon_rows = []
    if rm is not None and bon_ready:
        _t = time.perf_counter()
        _ns = [1, 2, 4, 8, 16, 32, 64]
        bon_rows, *_ = P.best_of_n_curve(rm, params_rm, cfg_rm, params, cfg, tok, TEST, ex.best_of_n, _ns,
                                         jax.random.key(6), temperature=1.0)
        _fig, _a = plt.subplots(figsize=(7, 3.4))
        _a.plot(_ns, [r["oracle"] for r in bon_rows], "o-", label="oracle (true preference)")
        _a.set_xscale("log", base=2)
        _a.set(xlabel="n (samples per question)", ylabel="oracle score of the pick", title="best-of-n")
        _b = _a.twinx()
        _b.plot(_ns, [r["rm"] for r in bon_rows], "s--", color="C1", label="reward model (proxy)")
        _b.set_ylabel("reward-model score of the pick", color="C1")
        if dpo_scores is not None:
            _a.axhline(dpo_scores.mean(), color="C2", ls=":", label="DPO policy, oracle (greedy, n = 1)")
        _h1, _l1 = _a.get_legend_handles_labels()
        _h2, _l2 = _b.get_legend_handles_labels()
        _a.legend(_h1 + _h2, _l1 + _l2, loc="lower right", fontsize=8)
        _fig.tight_layout()
        _best = max(bon_rows, key=lambda r: r["oracle"])
        _d_proxy = bon_rows[-1]["rm"] - bon_rows[0]["rm"]
        _d_true = bon_rows[-1]["oracle"] - bon_rows[0]["oracle"]
        if _best["n"] < _ns[-1]:
            _verdict = (f"The true score improves at first and then stops tracking the proxy: it peaked at "
                        f"n = {_best['n']} ({_best['oracle']:.2f}) and is {bon_rows[-1]['oracle']:.2f} at n = {_ns[-1]}, "
                        f"while the proxy rose by {_d_proxy:.1f} overall.")
        else:
            _verdict = (f"Here the true score is still rising at n = {_ns[-1]}, but far more slowly than the proxy: "
                        f"from n = 1 to {_ns[-1]} the reward model's score rose by {_d_proxy:.1f} while the oracle "
                        f"rose by {_d_true:.2f} — most of what the proxy \"gained\" isn't real.")
        _out = mo.vstack([_fig, mo.md(f"""
    {len(_ns)} values of n, {max(_ns) * len(TEST)} samples scored, in {time.perf_counter() - _t:.1f} s.

    | n | reward-model score | oracle score | words |
    |---|---|---|---|
    """ + "\n".join(f"    | {r['n']} | {r['rm']:.2f} | {r['oracle']:.2f} | {r['words']:.0f} |" for r in bon_rows) + f"""

    The **proxy keeps climbing** with n — that's what it's being maximized for. {_verdict} Gao, Schulman &
    Hilton (2022) measured this "reward-model overoptimization" curve at scale: past some amount of
    optimization, the true reward goes *down* while the proxy keeps going up (Section 8 shows it happening
    to DPO). Compare the green line: DPO gets its gain with **one** sample at inference time.
    """)])
    else:
        _out = mo.callout(mo.md("Needs the reward model (Section 3) and `best_of_n`."), kind="info")
    _out
    return (bon_rows,)


@app.cell
def _(P, TEST, TRAIN, cfg_rm, code, ex, jax, mo, np, pairs, params_rm, rm, samples, time, tok):
    def _probe(_rm):
        _cands = []
        for _j, _it in enumerate(TEST):
            _kw = _it["keyword"]
            _same = [o for o in TEST if o["family"] == _it["family"] and o["keyword"].lower() != _kw.lower()]
            _wrong = _same[_j % len(_same)]["keyword"] if _same else "unknown"
            _cands.append([_bests_test.get(_it["question"], f"The answer is {_kw}."), _kw.capitalize() + ".",
                           "Yes.", _wrong_sentence(_it, _wrong)])
        return P.score_candidates(_rm, params_rm, cfg_rm, tok, TEST, _cands)

    def _wrong_sentence(_it, _wrong):
        if _it["family"] == "capital":
            return f"The capital of {_it['question'].split('capital of ')[1].rstrip('?')} is {_wrong}."
        return f"The answer is {_wrong}."

    _bests_test = {_it["question"]: (f"The capital of {_it['question'].split('capital of ')[1].rstrip('?')} is "
                                     f"{_it['keyword']}." if _it["family"] == "capital" else
                                     f"The answer is {_it['keyword']}.") for _it in TEST}
    goodhart = {}
    if rm is not None:
        _t = time.perf_counter()
        _r, _o = _probe(rm)
        # the patch: add the comparisons a careful labeler would have made (real answer ≻ "Yes."/"No.")
        _texts = samples[0]
        _K = len(_texts) // len(TRAIN)
        _best = [max(_texts[i * _K:(i + 1) * _K], key=lambda t, i=i: P.oracle_score(t, TRAIN[i]["keyword"]))
                 for i in range(len(TRAIN))]
        _aug = P.concat_pairs(pairs, P.curt_wrong_negatives(tok, TRAIN, _best, pairs))
        _rm2, _ = P.train_reward_model(params_rm, cfg_rm, _aug, ex.bradley_terry_loss, 80, 8, jax.random.key(3))
        _r2, _ = _probe(_rm2)
        goodhart = {"yes_over_bare": float((_r[:, 2] > _r[:, 1]).mean()),
                    "yes_over_sentence": float((_r[:, 2] > _r[:, 0]).mean()),
                    "yes_over_bare_after": float((_r2[:, 2] > _r2[:, 1]).mean()),
                    "wrong_before": float((_r[:, 3] > _r[:, 0]).mean()),
                    "wrong_after": float((_r2[:, 3] > _r2[:, 0]).mean())}
        _names = ["a correct sentence", "the bare answer", code("Yes."), "a confident wrong sentence"]
        _yes_note = (f" (on {goodhart['yes_over_sentence'] * 100:.0f}% it even beats the correct sentence)"
                     if goodhart["yes_over_sentence"] > 0 else "")
        _rows = "\n".join(f"    | {n} | {_o[:, j].mean():.2f} | {_r[:, j].mean():.2f} | {_r2[:, j].mean():.2f} |"
                          for j, n in enumerate(_names))
        _out = mo.md(f"""
    ## 7 · Goodhart's law: the reward model is gameable

    "When a measure becomes a target, it ceases to be a good measure." We can see *why* by asking the reward
    model about answers that never appeared in its training data — the same {len(TEST)} held-out questions,
    four hand-written answers each ({time.perf_counter() - _t:.1f} s, including retraining):

    | candidate answer | oracle | reward model | patched reward model |
    |---|---|---|---|
{_rows}

    Read the reward-model column against the oracle column. {code("Yes.")} — not an answer at all — scores
    almost as high as the correct sentence and beats the *correct* bare answer on
    **{goodhart['yes_over_bare'] * 100:.0f}%** of questions{_yes_note}. And a **confident wrong sentence** beats the correct sentence on
    **{goodhart['wrong_before'] * 100:.0f}%** of questions. The base model is almost always right, so the reward
    model never saw a wrong answer during training: it learned what *did* vary — length and sentence shape —
    and nothing about correctness. A policy optimized hard against it would drift toward fluent, short,
    unchecked answers. That's reward hacking, the same failure Module 26 provoked on purpose.

    **The patch** is what real labeling teams do: add the comparisons the data was missing. With
    {len(TRAIN)} extra pairs of the form "real answer ≻ Yes./No.", the patched model now ranks {code("Yes.")}
    above the bare answer on only **{goodhart['yes_over_bare_after'] * 100:.0f}%** of questions. But it did nothing
    for the wrong sentence, which still beats the correct one on **{goodhart['wrong_after'] * 100:.0f}%**: nobody
    showed the reward model a wrong fact in a nice sentence. That's why human labelers are asked to check
    *facts*, why real pipelines keep probing their reward models and collecting data where they fail, why the
    KL penalty stays on as a seatbelt — and why verifiable rewards (Module 26) are so valuable where you can
    get them.
    """)
    else:
        _out = mo.callout(mo.md("Needs the reward model (Section 3)."), kind="info")
    _out
    return (goodhart,)


@app.cell
def _(RUNS, code, json, mo):
    _f = RUNS / "metrics.json"
    if _f.exists():
        _m = json.loads(_f.read_text())
        _sweep = _m["dpo"]["sweep"]
        _rows = "\n".join(f"    | {r['beta']} | {r['win_rate'] * 100:.0f}% | {r['oracle']:.2f} | "
                          f"{r.get('correct', float('nan')) * 100:.0f}% | {r['kl']:.1f} | {r['words']:.0f} |"
                          for r in _sweep)
        _bon = " · ".join(f"n={r['n']}: proxy {r['rm']:.2f} / oracle {r['oracle']:.2f}" for r in _m["best_of_n"])
        _curve = " → ".join(f"{c['heldout_acc'] * 100:.0f}% (step {c['step']})" for c in _m["rm"].get("curve", []))
        _lowest = min(_sweep, key=lambda r: r["beta"])
        _accs = [c["heldout_acc"] for c in _m["rm"].get("curve", [])]
        if _accs and max(_accs) - _accs[-1] > 0.03:
            _rm_note = "It peaks and then slides back — the classic sign of overfitting a small dataset."
        elif _accs:
            _rm_note = (f"It plateaus quickly and wobbles between {min(_accs) * 100:.0f}% and {max(_accs) * 100:.0f}% — "
                        "more steps don't buy more accuracy; more (and more varied) pairs would.")
        else:
            _rm_note = ""
        _ex = "\n".join(f"    | {e['q']} | {code(e['a'][:80])} |" for e in _lowest.get("examples", [])[:4])
        _out = mo.md(f"""
    ## 8 · The longer run (`train.py`)

    {_m['pairs']['train']} training pairs ({_m['args']['k']} samples per question). Held-out accuracy of the
    reward model by training step: {_curve or "(not recorded)"}. {_rm_note} `train.py` keeps the checkpoint
    that did best on the held-out pairs (step {_m['rm'].get('best_step', '?')},
    {_m['rm']['heldout_acc'] * 100:.1f}%). These held-out pairs include many with tiny score gaps (a few words'
    difference), which is harder than the live lesson's set — and a few hundred pairs is a tiny preference
    dataset: InstructGPT's reward model trained on tens of thousands of comparisons.

    DPO for {_m['args']['dpo_steps']} steps at three values of β (base: oracle {_m['eval']['oracle_base']:.2f},
    {_m['eval']['words_base']:.0f} words):

    | β | win rate vs base | oracle | correct | KL (nats) | words |
    |---|---|---|---|---|---|
{_rows}

    **Too little regularization breaks it.** At β = {_lowest['beta']} the policy moves furthest from the
    reference (KL {_lowest['kl']:.0f} nats), only {_lowest.get('correct', float('nan')) * 100:.0f}% of its answers
    are still correct, and it *loses* to the base model — over-optimization again, this time inside DPO: it
    learned "shorter wins" so well that it started cutting off the answer itself. Its answers:

    | question | β = {_lowest['beta']} answer |
    |---|---|
{_ex}

    Best-of-n on the same reward model: {_bon}.
    The whole run takes about {_m['seconds'] / 60:.0f} minutes (more on a busy GPU).
    """)
    else:
        _out = mo.callout(mo.md(
            "## 8 · The longer run\n\nRun `containers/run.sh python modules/35-rlhf-dpo/train.py` (about 15 "
            "minutes): more pairs, a longer-trained reward model, a DPO β sweep (KL vs win rate) and best-of-n "
            "up to 64. This section shows the results once `runs/35-rlhf-dpo/metrics.json` exists."), kind="info")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Where preference learning went next

    * **DPO variants** (2023–24): IPO, KTO, ORPO, SimPO and others change the loss — to resist overfitting
      the pairs, to learn from single thumbs-up/down labels instead of pairs, or to drop the reference model.
    * **AI feedback:** Constitutional AI (Bai et al., 2022) had a model critique and compare answers against
      a written set of principles, then trained on those preferences (RLAIF) — fewer human labels.
    * **Verifiable rewards:** when answers can be *checked* (math, code), you don't need a learned reward
      model at all — that's RLVR and GRPO from Module 26, and the reason DeepSeek-R1 could train reasoning
      with RL at scale. Learned reward models are still used for everything that can't be checked.
    * **The open problem** is the one Section 7 showed: any learned reward is a proxy, and optimization finds
      its gaps. Most of the craft of alignment is finding and fixing those gaps faster than the policy does.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "It gets the reward model from the policy's own log-probability ratio, so no separate reward model or RL loop is needed": "a",
            "It uses a bigger reward model than InstructGPT": "b",
            "It removes the need for preference data": "c",
        },
        label="**Check yourself:** what does DPO change compared with the InstructGPT pipeline?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right. The implicit reward β·log(π/π_ref) stands in for the reward model, and the "
                              "Bradley–Terry loss becomes a loss on the policy directly."), kind="success"),
        "b": mo.callout(mo.md("No — DPO has no separate reward model at all."), kind="warn"),
        "c": mo.callout(mo.md("No — DPO still learns from (chosen, rejected) pairs; it just skips the reward model "
                              "and the RL loop."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(35))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Lambert, Castricato, von Werra & Havrilla (Hugging Face),
      [Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) — the pipeline,
      with diagrams.
    * Nathan Lambert, [*Reinforcement Learning from Human Feedback*](https://rlhfbook.com/) — a free book on
      the whole field.
    * Gao, Schulman & Hilton, [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
      (2022) — Section 7's curve, measured carefully.
    * Hugging Face, [Fine-tune Llama 2 with DPO](https://huggingface.co/blog/dpo-trl) — DPO with the TRL
      library, in PyTorch.

    **Next:** Module 36 — state-space models and Mamba, the other road from RNNs.
    """)
    return


if __name__ == "__main__":
    app.run()
