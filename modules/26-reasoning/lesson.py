import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="26 · Reasoning models")


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
    from learnjax import llama

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")   # recompiles hit the disk cache
    ex = lj.load_exercises(mo.notebook_dir())

    def _load(name):
        _spec = importlib.util.spec_from_file_location(f"m26_{name}", mo.notebook_dir() / f"{name}.py")
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)
        return _mod

    R = _load("reasoning")
    RUNS = Path("/course/runs/26-reasoning")

    def code(text):
        """Inline code that survives markdown: marimo treats <answer> as an HTML tag even in backticks."""
        return f"<code>{html.escape(str(text))}</code>"

    return R, RUNS, code, ex, jax, jnp, json, lj, llama, mo, np, plt, time


@app.cell
def _(mo):
    mo.md(r"""
    # 26 · Reasoning models: inference scaling and RLVR with GRPO

    **Goal:** make a small language model better at a task whose answers can be *checked* — first by
    spending more compute at inference time, then by training it with reinforcement learning against a
    verifier, the recipe behind DeepSeek-R1 and today's "thinking" models.

    The structure follows Sebastian Raschka's
    [*Build a Reasoning Model (From Scratch)*](https://magazine.sebastianraschka.com/p/building-reasoning-models)
    video course (book and PyTorch code on Qwen3-0.6B): verifier → inference scaling → GRPO → distillation.
    Here it's JAX, on SmolLM2-360M-Instruct and your RX 7600 XT. His working definition is the one we use:
    a reasoning model is a conventional LLM with extra post-training by **reinforcement learning with
    verifiable rewards (RLVR)**, optionally plus **inference-time scaling**.

    **You will:** write a verifier that can't be gamed · measure how accuracy grows with samples
    (self-consistency) · try self-refinement and see whether a 360M model can check its own work · write
    GRPO's advantages, clipped loss and KL penalty, then train LoRA adapters with them · watch reward
    hacking happen and stop it · distill the improved behaviour back into a plain fine-tune.

    **Builds on:** Module 18 (on-device decoding, KV cache, sampling), Module 20 (LoRA), Module 21
    (confidence intervals — a 128-problem test set has about ±8 points of noise; keep that in mind).
    """)
    return


@app.cell
def _(R, code, llama, mo):
    params, cfg, tok = llama.load(llama.SMOLLM_360M)
    STOPS = R.stop_ids(tok)
    test = R.make_split(1, 128)            # held out: never used for training
    train = R.make_split(2, 4000)
    _ex = R.make_problem(__import__("random").Random(7))
    mo.md(f"""
    ## 1 · The task: arithmetic with a checkable answer

    Problems look like **{_ex['question']}** → `{_ex['target']}`: three three-digit numbers, `+` or `-`.
    Every answer is an exact integer, so a program can grade it. That's the defining property of RLVR
    tasks (math with known answers, code with unit tests).

    The model is [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) by
    Hugging Face ({llama.count_params(params) / 1e6:.0f}M parameters, the same weights as Modules 18–21).
    The prompt has a system message and **one worked example** in the chat history:

    {code(R.SHOT_Q)} → {code(R.SHOT_COT)}

    Without that example SmolLM2-360M *never* uses the answer tags (measured while writing this module:
    0 of 48), so a strict verifier would give it a score of zero — and, as we'll see, a reinforcement
    learner can't learn from rewards that are all zero. Train split: {len(train)} problems; test: {len(test)}.
    """)
    return STOPS, cfg, params, test, tok, train


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · The verifier is the reward

    In RLVR the *verifier* plays two roles: it measures accuracy, and its output is the reward the model
    is trained on. So it must be **strict**. Anything the verifier accidentally accepts, training will
    find and exploit — a model doesn't learn what you meant, it learns what you reward.

    Our format: the final answer goes in <code>&lt;answer&gt;…&lt;/answer&gt;</code>. Implement `verify_answer(text, target)` in
    `exercises.py`: exactly one tag, an integer inside, equal to the target. Hedging with two tags,
    "about 191", an unclosed tag — all wrong.
    """)
    return


@app.cell
def _(code, ex, lj, mo):
    _cases = [
        ("215 + 104 = 319. 319 - 128 = 191. <answer>191</answer>", 191, True),
        ("<answer> -42 </answer>", -42, True),
        ("The answer is 191.", 191, False),
        ("<answer>190</answer><answer>191</answer>", 191, False),
        ("<answer>about 191</answer>", 191, False),
        ("<answer>191", 191, False),
    ]
    _rows = ["| completion | target | expected | yours |", "|---|---|---|---|"]
    _all = True
    for _text, _t, _want in _cases:
        _a = lj.attempt(ex.verify_answer, _text, _t)
        _got = _a.value if _a.ok else ("TODO" if _a.todo else "error")
        _all = _all and _a.ok and bool(_got) == _want
        _rows.append(f"| {code(_text)} | {_t} | {_want} | {_got} {'✅' if _a.ok and bool(_got) == _want else '❌' if _a.ok else ''} |")
    verify_ok = _all
    _msg = (mo.callout(mo.md("Your verifier matches on every case."), kind="success") if verify_ok else
            mo.callout(mo.md("**Your turn:** implement `verify_answer` (and the other exercises) in "
                             "`exercises.py`. The rest of the lesson needs it."), kind="info"))
    mo.vstack([mo.md("\n".join(_rows)), _msg])
    return (verify_ok,)


@app.cell
def _(R, cfg, code, ex, mo, params, test, time, tok, verify_ok):
    baseline = None
    if verify_ok:
        _t0 = time.perf_counter()
        _cot, _cot_texts = R.greedy_accuracy(params, {}, cfg, tok, test, ex.verify_answer)
        _dir_prompts = R.encode_prompts(tok, test, style="direct")
        _dir, _dir_texts = R.greedy_accuracy(params, {}, cfg, tok, test, ex.verify_answer, prompts=_dir_prompts)
        _secs = time.perf_counter() - _t0
        baseline = {"cot": _cot, "direct": _dir, "cot_texts": _cot_texts}
        _right = [i for i, t in enumerate(_cot_texts) if ex.verify_answer(t, test[i]["target"])][:2]
        _wrong = [i for i, t in enumerate(_cot_texts) if not ex.verify_answer(t, test[i]["target"])][:2]
        _show = "\n".join(f"* {'✅' if i in _right else '❌'} **{test[i]['question']}** (= {test[i]['target']}) → "
                          f"{code(_cot_texts[i].strip())}" for i in _right + _wrong)
        _out = mo.md(f"""
    ### Baseline: greedy decoding, no training

    | prompt style | example answer shown to the model | greedy accuracy (128 test problems) |
    |---|---|---|
    | **direct** | {code(R.SHOT_DIRECT)} | **{_dir:.1%}** |
    | **chain of thought** | {code(R.SHOT_COT)} | **{_cot:.1%}** |

    ({_secs:.1f} s for both, batched on the GPU.) Asked for the answer alone, the model guesses three-number
    sums in its head — e.g. {code(_dir_texts[0].strip())} for {test[0]['question']} (= {test[0]['target']}). Writing the intermediate sum before the answer is
    *chain-of-thought* prompting (Wei et al., 2022): the model gets tokens to "think" with. From here on we
    always use the chain-of-thought prompt.

    {_show}
    """)
    else:
        _out = mo.md("*(Implement `verify_answer` to see the baseline.)*")
    _out
    return (baseline,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Inference-time scaling: more samples, better answers

    The first way to "make a model reason better" doesn't touch the weights: **spend more compute per
    question.** Sample several answers at temperature > 0 and combine them.

    * **Self-consistency** (Wang et al., 2022): sample $k$ chains of thought, extract each final answer,
      return the **majority vote**. Different wrong paths rarely agree on the *same* wrong number, while
      right paths all agree.
    * **pass@k**: the chance that *at least one* of $k$ samples is right. It isn't a method (you'd need the
      answer key to pick the right sample), it's the **ceiling** for any method that chooses among $k$
      samples — and exactly what RL can exploit: if a correct answer is sometimes sampled, it can be
      reinforced. The unbiased estimator from $n$ samples with $c$ correct (Chen et al., 2021) is
      $1 - \binom{n-c}{k} / \binom{n}{k}$.

    Implement `majority_vote` and `pass_at_k`, then the cell below samples 16 answers for each of 48
    test problems (768 completions in one batched, KV-cached pass on the GPU).
    """)
    return


@app.cell
def _(R, STOPS, baseline, cfg, ex, jax, lj, mo, np, params, test, time, tok, verify_ok):
    sc = None
    _mv = lj.attempt(ex.majority_vote, [3, 5, 3, None])
    _pk = lj.attempt(ex.pass_at_k, 10, 3, 2)
    if verify_ok and _mv.ok and _pk.ok:
        _N, _K = 48, 16
        _probs = test[:_N]
        _prompts = np.repeat(R.encode_prompts(tok, _probs), _K, axis=0)
        _t0 = time.perf_counter()
        _gen = R.sample_chunked(params, {}, cfg, _prompts, 48, jax.random.key(26), temperature=0.8)
        _secs = time.perf_counter() - _t0
        _mask = R.completion_mask(_gen, STOPS)
        _texts = R.decode_texts(tok, _gen, _mask, STOPS)
        _answers = np.array([R.extract_answer(t) for t in _texts], dtype=object).reshape(_N, _K)
        _targets = np.array([p["target"] for p in _probs])
        _correct = np.array([[a == y for a in row] for row, y in zip(_answers, _targets, strict=True)])
        ks = [1, 2, 4, 8, 16]
        maj = [np.mean([ex.majority_vote(list(_answers[i, :k])) == _targets[i] for i in range(_N)]) for k in ks]
        pak = [np.mean([ex.pass_at_k(_K, int(_correct[i].sum()), k) for i in range(_N)]) for k in ks]
        sc = {"ks": ks, "maj": maj, "pass": pak, "texts": _texts, "answers": _answers, "targets": _targets,
              "correct": _correct, "gen": _gen, "mask": _mask, "prompts": _prompts, "tokens": float(_mask.sum(1).mean())}
        _out = mo.md(f"""
    Sampled {_N * _K} completions in {_secs:.1f} s ({_N * _K * 48 / _secs:,.0f} tokens/s including padding).
    A single sample at temperature 0.8 is right {_correct.mean():.1%} of the time (one greedy answer on these
    48 problems: {np.mean([ex.verify_answer(t, p['target']) for t, p in zip(baseline['cot_texts'][:_N], _probs, strict=True)]):.1%}).
    Mean completion: {sc['tokens']:.0f} tokens.
    """)
    else:
        _out = mo.vstack([_mv.view(mo), _pk.view(mo)])
    _out
    return (sc,)


@app.cell
def _(baseline, mo, np, plt, sc):
    if sc is None or baseline is None:
        _o = mo.md("*(Needs `verify_answer`, `majority_vote` and `pass_at_k`.)*")
    else:
        _fig, _ax = plt.subplots(figsize=(7, 3.6))
        _ax.plot(sc["ks"], np.array(sc["pass"]) * 100, "o-", label="pass@k (ceiling: any sample right)")
        _ax.plot(sc["ks"], np.array(sc["maj"]) * 100, "s-", label="self-consistency (majority of k)")
        _ax.axhline(baseline["cot"] * 100, color="gray", ls="--", label="greedy (1 completion)")
        _ax.set(xscale="log", xticks=sc["ks"], xticklabels=[f"{k}\n{k * sc['tokens']:.0f} tok" for k in sc["ks"]],
                xlabel="samples per question (and tokens generated)", ylabel="accuracy (%)",
                title="Accuracy vs inference compute (48 test problems)")
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.3)
        _fig.tight_layout()
        _rows = " | ".join(f"k={k}: {m:.0%} / {p:.0%}" for k, m, p in zip(sc["ks"], sc["maj"], sc["pass"], strict=True))
        _o = mo.vstack([_fig, mo.md(f"""
    **majority / pass@k:** {_rows}

    Majority voting buys accuracy with compute — {sc['maj'][-1] - sc['maj'][0]:+.0%} from 1 to 16 samples
    here — but flattens out: once most samples agree, more votes change little. The pass@k curve keeps
    climbing, and the gap between the two lines is answers the model *can* produce but can't *pick*.
    Closing that gap — making the right path the likely one — is what reinforcement learning is for.
    """)])
    _o
    return


@app.cell
def _(R, cfg, ex, mo, np, params, sc):
    _o = mo.md("")
    if sc is not None and baseline is not None:
        # log-prob scoring: break majority-vote ties by the samples' mean token log-probability
        _seqs = np.concatenate([sc["prompts"], sc["gen"]], axis=1)
        _lp = R.logprobs_chunked(params, {}, cfg, _seqs, sc["prompts"].shape[1])
        _mean_lp = ((_lp * sc["mask"]).sum(1) / sc["mask"].sum(1)).reshape(sc["answers"].shape)

        def _vote(row, scores, k):
            _counts = {}
            for _a, _s in zip(row[:k], scores[:k], strict=True):
                if _a is not None:
                    _counts.setdefault(_a, []).append(_s)
            if not _counts:
                return None
            _best = max(len(v) for v in _counts.values())
            _tied = {a: v for a, v in _counts.items() if len(v) == _best}
            return max(_tied, key=lambda a: np.mean(_tied[a]))

        _gains = []
        _rows = ["| k | ties among top answers | majority (first-seen tie-break) | majority (log-prob tie-break) |",
                 "|---|---|---|---|"]
        for _k in (2, 4, 8):
            _first = np.mean([ex.majority_vote(list(r[:_k])) == y for r, y in zip(sc["answers"], sc["targets"], strict=True)])
            _lpv = np.mean([_vote(r, s, _k) == y for r, s, y in zip(sc["answers"], _mean_lp, sc["targets"], strict=True)])
            _ties = 0
            for _r in sc["answers"]:
                _c = {}
                for _a in _r[:_k]:
                    if _a is not None:
                        _c[_a] = _c.get(_a, 0) + 1
                if _c and sorted(_c.values())[-2:] == [max(_c.values())] * 2:
                    _ties += 1
            _rows.append(f"| {_k} | {_ties} of {len(sc['answers'])} | {_first:.1%} | {_lpv:.1%} |")
            _gains.append(_lpv - _first)
        _verdict = (f"Here log-prob tie-breaking changed accuracy by {', '.join(f'{g:+.1%}' for g in _gains)} at "
                    "k = 2, 4, 8 — ties are common at k = 2 and rare by k = 8, so that's where it can matter. ")
        _o = mo.md(r"""
    ### Breaking ties with the model's own confidence

    With few samples, votes often tie (2 samples, 2 different answers). One natural tie-breaker is the
    model's **log-probability** of each sample, averaged per token: how sure was it while writing? (Raschka
    covers this "log-prob scoring" in his inference-scaling video.)

    """ + "\n".join(_rows) + """

    """ + _verdict + """Confidence is a weak signal for a small model — it is often confidently wrong —
    which is a reason to prefer a real verifier.
    """)
    _o
    return


@app.cell
def _(R, STOPS, baseline, cfg, code, ex, jax, mo, np, params, test, tok):
    _o = mo.md("")
    if baseline is not None:
        _N = 48
        _crit = ("Check your work above step by step. If any step is wrong, correct it. "
                 "Finish with the result inside answer tags.")
        _ids = []
        for _p, _first in zip(test[:_N], baseline["cot_texts"][:_N], strict=True):
            _txt = R.prompt_text(_p["question"]) + _first.strip() + "<|im_end|>\n" + \
                f"<|im_start|>user\n{_crit}<|im_end|>\n<|im_start|>assistant\n"
            _ids.append(tok.encode(_txt).ids)
        _toks, _starts = R.left_pad(_ids, tok.token_to_id("<|endoftext|>"))
        _gen = R.sample_chunked(params, {}, cfg, _toks, 64, jax.random.key(0), 0.0, chunk=48,
                                starts=_starts)
        _texts = R.decode_texts(tok, _gen, R.completion_mask(_gen, STOPS), STOPS)
        _before = np.array([ex.verify_answer(t, p["target"]) for t, p in zip(baseline["cot_texts"][:_N], test[:_N], strict=True)])
        _after = np.array([ex.verify_answer(t, p["target"]) for t, p in zip(_texts, test[:_N], strict=True)])
        _fixed, _broke = int((~_before & _after).sum()), int((_before & ~_after).sum())
        _o = mo.md(f"""
    ### Self-refinement: can the model check its own work?

    Another inference-time idea: show the model its answer and ask it to check and correct it. On the
    first {_N} test problems (greedy both times, prompts left-padded into one batch):

    | | right before | right after critique | wrong → right | right → wrong |
    |---|---|---|---|---|
    | SmolLM2-360M | {_before.sum()} / {_N} | {_after.sum()} / {_N} | **{_fixed}** | **{_broke}** |

    Example — **{test[2]['question']}** (= {test[2]['target']}): first {code(baseline['cot_texts'][2].strip())},
    after critique {code(_texts[2].strip()[:160])}

    Self-refinement needs a model that can *recognize* an error more reliably than it makes one. At 360M
    parameters it can't: it re-solves the problem (often getting new arithmetic wrong), and with two
    questions in its context it sometimes answers the *worked example* again instead. Big
    reasoning models do this well partly *because* RL training rewarded catching mistakes.
    """)
    _o
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · GRPO: reinforcement learning with verifiable rewards

    **Group Relative Policy Optimization** (Shao et al., 2024, in *DeepSeekMath*; the RL algorithm behind
    DeepSeek-R1) turns "sometimes right" into "usually right". For each prompt:

    1. **Sample a group** of $G$ completions from the current model (the *policy*).
    2. **Score** each with the reward $r_i$ (our verifier: 1 if correct).
    3. **Advantage = how much better than its siblings:**
       $A_i = \dfrac{r_i - \operatorname{mean}(r_{1..G})}{\operatorname{std}(r_{1..G}) + \epsilon}$.
       No learned value network (PPO's critic) — the group *is* the baseline. A group where every sample
       scored the same gives $A = 0$: nothing to learn, which is why rewards that are all zero are useless.
    4. **Push up** the log-probability of every token in above-average completions and push down
       below-average ones, with PPO's clipped ratio $\rho = \pi_\theta / \pi_{\text{old}}$ so one batch
       can't move the policy too far:
       $\;\mathcal{L}_{\text{pg}} = -\,\mathbb{E}\big[\min(\rho A,\ \operatorname{clip}(\rho, 1-\varepsilon, 1+\varepsilon) A)\big]$.
    5. **Stay close to the original model** with a KL penalty, $\beta\,\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$,
       estimated per token as $k_3 = e^{\log\pi_{\text{ref}} - \log\pi_\theta} - (\log\pi_{\text{ref}} - \log\pi_\theta) - 1 \ge 0$.

    We train **LoRA adapters** (Module 20) on the attention projections, so the reference model is free:
    it's the same weights with the adapters switched off. With one optimizer step per batch, as in
    DeepSeekMath, $\pi_{\text{old}}$ is the policy that just sampled, so $\rho = 1$ and the loss *value*
    is ≈ 0 (advantages average to zero within each group) — but its *gradient* is not:
    $\nabla \mathcal{L} = -\mathbb{E}[A \,\nabla \log \pi_\theta]$, the policy gradient.

    Implement `combined_reward`, `group_advantages`, `clipped_pg_loss` and `kl_penalty`.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo, np):
    _adv = lj.attempt(ex.group_advantages, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], 4)
    _mask = jnp.ones((2, 3))
    _lp = jnp.log(jnp.full((2, 3), 0.5))
    _pg_same = lj.attempt(ex.clipped_pg_loss, _lp, _lp, jnp.array([1.0, -1.0]), _mask)
    _pg_up = lj.attempt(ex.clipped_pg_loss, _lp + jnp.log(2.0), _lp, jnp.array([1.0, -1.0]), _mask, 0.2)
    _kl = lj.attempt(ex.kl_penalty, _lp, _lp + 0.1, _mask)
    _rw = lj.attempt(ex.combined_reward, "1 + 1 = 2. <answer>2</answer>", 2)
    _items = [mo.md("**`group_advantages([1,0,0,1, 1,1,1,1], 4)`** — the second group is all-correct, so it teaches nothing:"),
              mo.md(f"`{np.round(_adv.value, 3).tolist()}`") if _adv.ok else _adv.view(mo),
              mo.md("**`clipped_pg_loss`** with ratio 1 (should be 0), then ratio 2 (the positive advantage clips at 1.2, the negative one doesn't — expect 0.4):"),
              mo.md(f"`{float(_pg_same.value):.4f}` and `{float(_pg_up.value):.4f}`") if _pg_same.ok and _pg_up.ok else _pg_same.view(mo),
              mo.md("**`kl_penalty`** with the reference 0.1 nats above the policy everywhere (expect e^0.1 − 1.1 ≈ 0.00517):"),
              mo.md(f"`{float(_kl.value):.5f}`") if _kl.ok else _kl.view(mo),
              mo.md("**`combined_reward`** for a correct, tidy answer (expect 1.2):"),
              mo.md(f"`{_rw.value}`") if _rw.ok else _rw.view(mo)]
    grpo_ok = all(a.ok for a in (_adv, _pg_same, _pg_up, _kl, _rw))
    mo.vstack(_items)
    return (grpo_ok,)


@app.cell
def _(R, baseline, cfg, ex, grpo_ok, jax, mo, params, test, time, tok, train, verify_ok):
    live = None
    if grpo_ok and verify_ok and baseline is not None:
        _steps = 12
        _ad = R.init_adapters(jax.random.key(0), cfg)
        _opt, _update = R.make_grpo(cfg, ex.clipped_pg_loss, ex.kl_penalty, lr=2e-4, beta=0.04)
        _state = {"adapters": _ad, "opt_state": _opt.init(_ad)}
        _key = jax.random.key(1)
        _hist = []
        _t0 = time.perf_counter()
        for _s in range(_steps):
            _key, _sub = jax.random.split(_key)
            _state, _st, _ = R.grpo_step(_state, params, cfg, tok, train[_s * 8:(_s + 1) * 8], ex.combined_reward,
                                         ex.group_advantages, _update, _sub)
            _hist.append(_st)
        _secs = time.perf_counter() - _t0
        _acc, _ = R.greedy_accuracy(params, _state["adapters"], cfg, tok, test, ex.verify_answer)
        live = {"hist": _hist, "acc": _acc, "secs": _secs, "adapters": _state["adapters"]}
        _o = mo.md(f"""
    ### A live GRPO run: {_steps} steps × 8 prompts × 8 samples

    {_steps * 64} sampled completions, {_secs:.0f} s. Greedy test accuracy **{baseline['cot']:.1%} → {_acc:.1%}**
    after training on {_steps * 8} problems (the test problems were never seen). With 128 test problems,
    differences under ~8 points are within noise (Module 21); `train.py` runs longer for a clearer answer.
    """)
    else:
        _o = mo.md("*(Implement the Section 2 and 4 exercises to run GRPO.)*")
    _o
    return (live,)


@app.cell
def _(live, mo, np, plt):
    if live is None:
        _o = mo.md("")
    else:
        _h = live["hist"]
        _fig, _axes = plt.subplots(1, 3, figsize=(10, 2.8))
        for _a, _k, _t in zip(_axes, ["reward", "kl", "length"],
                              ["mean reward (sampled, T=1)", "KL to base model", "completion length (tokens)"], strict=True):
            _a.plot(np.arange(1, len(_h) + 1), [s[_k] for s in _h], "o-", ms=3)
            _a.set(title=_t, xlabel="GRPO step")
            _a.grid(alpha=0.3)
        _fig.tight_layout()
        _r0, _r1 = np.mean([s["reward"] for s in _h[:4]]), np.mean([s["reward"] for s in _h[-4:]])
        _l0, _l1 = _h[0]["length"], _h[-1]["length"]
        _o = mo.vstack([_fig, mo.md(
            f"The sampled reward is noisy step to step — each step sees 8 new problems; mean of the first 4 steps "
            f"{_r0:.2f}, last 4 steps {_r1:.2f}. KL to the base model grew from {_h[0]['kl']:.4f} to {_h[-1]['kl']:.4f}: "
            "the adapters move the policy, and the β·KL penalty keeps the drift small. Completion length went "
            f"{_l0:.0f} → {_l1:.0f} tokens: this task doesn't pay for longer thinking, so none emerges (in "
            "DeepSeek-R1-Zero, long chains of thought *emerged* because harder problems paid for them).")])
    _o
    return


@app.cell
def _(R, RUNS, baseline, cfg, code, ex, json, mo, np, params, plt, test, tok, verify_ok):
    trained = None
    _m = RUNS / "metrics.jsonl"
    if not _m.exists():
        _o = mo.callout(mo.md(
            "**The longer run:** `containers/run.sh python modules/26-reasoning/train.py` (60 GRPO steps, "
            "~5 min on the RX 7600 XT, evaluates every 10 steps and saves `runs/26-reasoning/adapters.npz` "
            "— which Module 27 exports to LM Studio). Re-run this cell afterwards."), kind="info")
    else:
        _rows = [json.loads(line) for line in _m.read_text().splitlines()]
        _ev = [r for r in _rows if "test_acc" in r]
        _tr = [r for r in _rows if "reward" in r]
        _fig, _ax = plt.subplots(figsize=(7, 3))
        if _tr:
            _w = 5
            _rew = np.convolve([r["reward"] for r in _tr], np.ones(_w) / _w, mode="valid")
            _ax.plot([r["step"] for r in _tr][_w - 1:], _rew, color="C1", alpha=0.7, label="sampled reward (5-step avg)")
        _ax.plot([r["step"] for r in _ev], [r["test_acc"] for r in _ev], "o-", color="C0", label="greedy test accuracy")
        if _ev and "val_acc" in _ev[0]:
            _ax.plot([r["step"] for r in _ev], [r["val_acc"] for r in _ev], "s--", color="C2", ms=4,
                     label="greedy validation accuracy (picks the checkpoint)")
        _ax.set(xlabel="GRPO step", ylim=(0, 1.3), title="train.py: GRPO on SmolLM2-360M + LoRA")
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.3)
        _fig.tight_layout()
        _extra = ""
        if (RUNS / "adapters.npz").exists() and verify_ok and baseline is not None:
            _ad = R.load_adapters(RUNS / "adapters.npz")
            _acc, _texts = R.greedy_accuracy(params, _ad, cfg, tok, test, ex.verify_answer)
            trained = {"acc": _acc, "adapters": _ad}
            _best = next((r for r in _rows if "best_step" in r), None)
            _which = f"the checkpoint with the best *validation* accuracy (step {_best['best_step']})" if _best else "the saved adapters"
            _extra = (f"\n\nReloaded `adapters.npz`, {_which}: greedy test accuracy **{baseline['cot']:.1%} → {_acc:.1%}**. "
                      "Accuracy rises fast, then wanders within the ±8-point noise of a 128-problem test; picking the "
                      "checkpoint on a separate validation split (never on test) is the honest way to stop. "
                      f"Example: **{test[1]['question']}** → {code(_texts[1].strip())} "
                      "(Step-0 accuracy in `train.py` can differ by a few points from this notebook's baseline: separate "
                      "processes may pick different GPU matmul kernels, and bf16 near-ties flip — Module 22.)")
        _summary = (f"`train.py` logged {len(_tr)} steps; test accuracy {_ev[0]['test_acc']:.1%} at step 0 → "
                    f"{_ev[-1]['test_acc']:.1%} at step {_ev[-1]['step']}." if _ev else
                    "`train.py` hasn't logged an evaluation yet (still running?).")
        _o = mo.vstack([_fig, mo.md(_summary + _extra)])
    _o
    return (trained,)


@app.cell
def _(R, baseline, cfg, ex, live, mo, np, params, test, tok, trained):
    import re as _re
    _o = mo.md("")
    _tuned = trained if trained is not None else live
    if baseline is not None and _tuned is not None:
        def _number_right(text, target):            # ignores the tag closing: was the arithmetic right?
            _m = _re.findall(r"<answer>\s*([+-]?\d+)\s*(?:</answer>?)?\s*$", text.strip())
            return len(_m) == 1 and int(_m[0]) == target

        _rows = ["| model | verifier (strict) | right number, any closing | unclosed &lt;/answer tag |",
                 "|---|---|---|---|"]
        _stats = {}
        for _name, _ad in (("base", {}), ("after GRPO", _tuned["adapters"])):
            _acc, _texts = R.greedy_accuracy(params, _ad, cfg, tok, test, ex.verify_answer)
            _num = np.mean([_number_right(t, p["target"]) for t, p in zip(_texts, test, strict=True)])
            _open = np.mean([t.strip().endswith("</answer") for t in _texts])
            _stats[_name] = (_acc, _num, _open)
            _rows.append(f"| {_name} | {_acc:.1%} | {_num:.1%} | {_open:.1%} |")
        (_a0, _n0, _u0), (_a1, _n1, _u1) = _stats["base"], _stats["after GRPO"]
        _o = mo.md("""
    ### What did GRPO actually learn?

    Before believing a reward curve, look at *what* changed. Score the same greedy outputs two ways: the
    strict verifier, and a lenient check that only asks whether the number inside the tag is right
    (accepting an unclosed &lt;/answer):

    """ + "\n".join(_rows) + f"""

    The strict score moved {_a1 - _a0:+.1%}, but the arithmetic itself moved {_n1 - _n0:+.1%}: much of the
    gain is the model learning to **close its answer tag** (unclosed tags {_u0:.0%} → {_u1:.0%}). That is
    exactly what the reward paid for — a correct number in a malformed tag scored 0 — and it's a real,
    useful fix. But it isn't better arithmetic, and a 128-problem test can't resolve a few points of
    that. RL optimizes the reward you wrote; read the outputs to see which part of it moved.
    """)
    _o
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Reward hacking

    A reward is a *proxy* for what you want, and RL optimizes the proxy. If the format bonus were paid
    whether or not the answer is right, "write <code>&lt;answer&gt;</code> tags" would be the easiest way to earn reward —
    and the model would learn tidy, confident, wrong answers. That's **reward hacking** (Raschka's
    video 7 covers it, with format rewards conditioned on correctness as the fix).

    Below: the same GRPO, but rewarded by `format_reward` alone. Watch what happens to test accuracy —
    which this reward no longer measures — while the reward rises.
    """)
    return


@app.cell
def _(R, baseline, cfg, code, ex, grpo_ok, jax, lj, mo, np, params, plt, test, tok, train, verify_ok):
    _fr = lj.attempt(ex.format_reward, "<answer>999</answer>")
    if grpo_ok and verify_ok and _fr.ok and baseline is not None:
        _steps = 10
        _ad = R.init_adapters(jax.random.key(0), cfg)
        _opt, _update = R.make_grpo(cfg, ex.clipped_pg_loss, ex.kl_penalty, lr=5e-4, beta=0.0)
        _state = {"adapters": _ad, "opt_state": _opt.init(_ad)}
        _key = jax.random.key(3)
        _rew, _acc_curve = [], [(0, baseline["cot"])]

        def _hack(text, target):
            return ex.format_reward(text)

        for _s in range(_steps):
            _key, _sub = jax.random.split(_key)
            _state, _st, _texts = R.grpo_step(_state, params, cfg, tok, train[1000 + _s * 8:1000 + (_s + 1) * 8],
                                              _hack, ex.group_advantages, _update, _sub)
            _rew.append(_st["reward"])
        _acc, _out_texts = R.greedy_accuracy(params, _state["adapters"], cfg, tok, test, ex.verify_answer)
        _fmt = np.mean([ex.format_reward(t) for t in _out_texts])
        _fig, _ax = plt.subplots(figsize=(6, 2.8))
        _ax.plot(np.arange(1, _steps + 1), _rew, "o-", label="format-only reward (sampled)")
        _ax.axhline(baseline["cot"], color="gray", ls="--", label=f"test accuracy before ({baseline['cot']:.0%})")
        _ax.axhline(_acc, color="C3", ls=":", label=f"test accuracy after ({_acc:.0%})")
        _ax.set(xlabel="GRPO step", ylim=(0, 1.05), title="Optimizing the proxy")
        _ax.legend(fontsize=7)
        _ax.grid(alpha=0.3)
        _fig.tight_layout()
        _o = mo.vstack([_fig, mo.md(f"""
    Format reward climbed to **{_rew[-1]:.0%}** of samples (greedy outputs now {_fmt:.0%} cleanly formatted), while
    greedy test accuracy went **{baseline['cot']:.1%} → {_acc:.1%}**. The reward went up; the thing we care about
    {"fell — the model drifted toward tidy, confident, wrong answers" if _acc < baseline['cot'] - 0.05 else "didn't follow"},
    because nothing in the reward asks for correctness. (While writing this module, 25 steps took accuracy from
    31% to 13% with the reward pinned at 100%.) Example output:
    {code(_out_texts[2].strip()[:140])}

    The fix is in `combined_reward`: the format bonus is paid **only on top of a correct answer**, so the
    only way to earn it is to be right. More generally: every way your verifier can be satisfied without
    solving the task — hedging with two tags, "approximately", printing every candidate — will eventually
    be found, so test the verifier like an adversary (that's what `test_verify_answer` does).
    """)])
    else:
        _o = mo.md("*(Needs the Section 2 and 4 exercises and `format_reward`.)*")
    _o
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Distillation: teach the plain model what sampling found

    RL is one way to put better behaviour into the weights; **distillation** is another (DeepSeek-R1's
    small models were distilled from R1's outputs). Here, a cheap version that needs no bigger teacher:
    **rejection sampling** — sample several solutions per *training* problem, keep only the ones the
    verifier accepts, and fine-tune on them (supervised, like Module 20). The verifier turns the model's
    lucky samples into training data.
    """)
    return


@app.cell
def _(R, STOPS, baseline, cfg, ex, jax, jnp, mo, np, params, test, time, tok, train, verify_ok):
    distilled = None
    if verify_ok and baseline is not None:
        _t0 = time.perf_counter()
        _probs = train[2000:2064]
        _p = R.encode_prompts(tok, _probs)
        _rep = np.repeat(_p, 8, axis=0)
        _gen = R.sample_chunked(params, {}, cfg, _rep, 48, jax.random.key(5), temperature=0.8)
        _mask = R.completion_mask(_gen, STOPS)
        _texts = R.decode_texts(tok, _gen, _mask, STOPS)
        _tg = np.repeat([p["target"] for p in _probs], 8)
        _keep = np.array([ex.verify_answer(t, y) for t, y in zip(_texts, _tg, strict=True)])
        _seqs = np.concatenate([_rep, _gen], axis=1)[_keep]
        _m = _mask[_keep]
        _opt, _step = R.make_sft(cfg, lr=2e-4)
        _ad = R.init_adapters(jax.random.key(0), cfg)
        _os = _opt.init(_ad)
        _rng = np.random.default_rng(0)
        _losses = []
        for _s in range(40):
            _idx = _rng.choice(len(_seqs), 16, replace=len(_seqs) < 16)
            _ad, _os, _l = _step(_ad, _os, params, jnp.asarray(_seqs[_idx]), jnp.asarray(_m[_idx]), _p.shape[1])
            _losses.append(float(_l))
        _acc, _ = R.greedy_accuracy(params, _ad, cfg, tok, test, ex.verify_answer)
        distilled = {"acc": _acc, "kept": int(_keep.sum()), "total": len(_keep)}
        _o = mo.md(f"""
    Sampled {len(_keep)} solutions for 64 training problems, **kept the {_keep.sum()} the verifier accepted**
    ({_keep.mean():.0%}), and fine-tuned LoRA adapters on them for 40 steps (loss {_losses[0]:.3f} → {_losses[-1]:.3f}).
    Greedy test accuracy: **{baseline['cot']:.1%} → {_acc:.1%}** ({time.perf_counter() - _t0:.0f} s in all).

    Rejection-sampling fine-tuning trains only on successes; GRPO also pushes *down* on failures and
    keeps sampling from the improving policy. With a stronger teacher (a bigger model, or the GRPO-tuned
    one with self-consistency) the kept solutions get better — that's distillation proper.
    """)
    else:
        _o = mo.md("*(Needs `verify_answer`.)*")
    _o
    return (distilled,)


@app.cell
def _(baseline, distilled, live, mo, sc, trained):
    if baseline is None:
        _o = mo.md("")
    else:
        _rows = ["| method | what it costs | test accuracy |", "|---|---|---|",
                 f"| greedy, direct answer | 1 short completion | {baseline['direct']:.1%} |",
                 f"| greedy, chain of thought | 1 completion | {baseline['cot']:.1%} |"]
        if sc is not None:
            _rows.append(f"| self-consistency, 16 samples (48 problems) | 16× tokens at inference | {sc['maj'][-1]:.1%} |")
        if distilled is not None:
            _rows.append(f"| rejection-sampling fine-tune (LoRA) | {distilled['total']} samples + 40 SFT steps | {distilled['acc']:.1%} |")
        if live is not None:
            _rows.append(f"| GRPO, live ({len(live['hist'])} steps) | {len(live['hist']) * 64} samples + updates | {live['acc']:.1%} |")
        if trained is not None:
            _rows.append(f"| GRPO, `train.py` | longer run | {trained['acc']:.1%} |")
        _o = mo.md("## 7 · Scoreboard\n\n" + "\n".join(_rows) + """

    Inference-time scaling pays at *every* question; training pays once and then every greedy answer
    is better. Real reasoning models combine both — and the training signal is only as good as the
    verifier.""")
    _o
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Every completion in a group got the same reward, so all advantages are zero": "a",
            "The learning rate is too small": "b",
            "The KL penalty cancels the policy gradient": "c",
        },
        label="**Check yourself:** a GRPO run's loss and adapters don't change at all on a batch. Most likely cause?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right. GRPO's baseline is the group mean: if all G samples are equally right "
                              "(or equally wrong), every advantage is 0 and there's no gradient. That's why the task "
                              "must be *partially* solvable — and why the prompt had a worked example."), kind="success"),
        "b": mo.callout(mo.md("A small learning rate would make changes small, not zero. Look at what the "
                              "advantages are when every sample scores the same."), kind="warn"),
        "c": mo.callout(mo.md("The KL term pulls toward the reference but starts at 0 and only adds a gradient once "
                              "the policy has moved. Check the advantages."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(26))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Sebastian Raschka, [*Build a Reasoning Model (From Scratch)* — the video course](https://magazine.sebastianraschka.com/p/building-reasoning-models)
      and its code, [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) — this
      module's structure, in PyTorch on Qwen3-0.6B.
    * Sebastian Raschka, [Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms) —
      the four approaches (inference scaling, pure RL, SFT + RL, distillation) and DeepSeek-R1.
    * Nathan Lambert, [*Reinforcement Learning from Human Feedback*](https://rlhfbook.com/) — the book, with
      chapters on policy gradients, GRPO and RL with verifiable rewards.
    * Hugging Face TRL docs: [GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — the
      production version of what you wrote, with its loss variants.

    **Next:** Module 27 — take the adapter `train.py` saved, and every other model you built, into LM Studio.
    """)
    return


if __name__ == "__main__":
    app.run()
