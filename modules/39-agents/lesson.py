import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="39 · Agents")


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
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")   # recompiles hit the disk cache
    HERE = mo.notebook_dir()
    ex = lj.load_exercises(HERE)

    def _load(name, path):
        _spec = importlib.util.spec_from_file_location(name, path)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[name] = _mod
        _spec.loader.exec_module(_mod)
        return _mod

    B = _load("m39_bench", HERE / "bench.py")
    RUNS = Path("/course/runs/39-agents")

    def esc(text):
        """Inline code that survives markdown: marimo treats <tool_call> as an HTML tag even in backticks."""
        return f"<code>{html.escape(str(text))}</code>"

    return B, HERE, RUNS, esc, ex, jax, json, lj, mo, np, plt, time


@app.cell
def _(mo):
    mo.md(r"""
    # 39 · Agents: tools, memory, workflows and multi-agent systems

    **Goal:** build an agent from its parts, on the 360M model you've used since Module 18, and *measure*
    each part: the loop, tool calling, short- and long-term memory, the standard workflow patterns, and one
    agent versus several. Then drive the same JAX model from LangGraph and over MCP.

    An **agent**, in the definition this module uses (Anthropic's), is "typically just LLMs using tools based
    on environmental feedback in a loop." Everything else here is engineering around that loop.

    **You will:** parse and dispatch tool calls · make a small model's tool calls reliable with constrained
    decoding · keep a conversation's KV cache between turns and time it · compare five memory strategies on a
    recall test · implement Anthropic's five workflow patterns and OpenAI's manager and handoff patterns and
    score them on one task suite · add guardrails · serve the JAX model behind an OpenAI-compatible API so
    LangGraph can drive it · expose Module 38's search as an MCP tool.

    **Builds on:** Module 18 (KV cache, decoding), Module 21 (log-likelihood scoring; confidence), Module 26
    (verifiers, ReAct in §7), Module 27 (serving, tool calling in LM Studio), Module 38 (retrieval).
    """)
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **Where this came from**

    ReAct (Yao et al., 2022) interleaved reasoning with tool calls and observations; Toolformer (Schick et
    al., 2023) taught a model where an API call helps. In spring 2023 AutoGPT and BabyAGI made "autonomous
    agents" a craze, and Generative Agents (Park et al., 2023) gave agents a memory stream. OpenAI added
    function calling to its API (June 2023) and Microsoft released AutoGen for multi-agent conversation
    (2023). Anthropic's *Building effective agents* (December 2024) separated **workflows** from **agents**,
    and its Model Context Protocol (November 2024) standardized how tools are offered to models. OpenAI's
    *A practical guide to building agents* (April 2025) set out single-agent and multi-agent orchestration.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(B, HERE, lj, mo):
    _e38 = lj.load_exercises(HERE.parent / "38-rag")
    _try = lj.attempt(_e38.bm25_weights, [1], [1], [1], 2, 1.0)
    ex38 = _e38 if _try.ok else lj.load_exercises(HERE.parent / "38-rag", solutions=True)
    A, rag, _, papers, lm, tools = B.setup(ex38)
    box = A.Toolbox(lm, tools)
    tasks = A.make_tasks(papers, n_each=2)
    _src = "your Module 38 `bm25_weights`" if _try.ok else "the reference BM25 (finish Module 38 to use yours)"
    _rows = "\n".join(f"| {t['kind']} | {t['q'][:110]}{'…' if len(t['q']) > 110 else ''} | `{t['answer']}` |"
                      for t in tasks[::2])
    mo.md(f"""
## 1 · The model, the tools, and a task suite with exact answers

SmolLM2-360M-Instruct ({lm.cfg.n_layers} layers, in `learnjax.llama`) and four tools: `calculator`,
`days_between`, `weekday`, and `search_papers`, which searches 4,000 radio-astronomy abstracts with
{_src}. A 360M model can't multiply two-digit numbers or recall a paper's date reliably, so every task
*needs* a tool, and every answer is checked exactly (Module 26's verifier idea):

| kind | question | answer |
|---|---|---|
{_rows}

The live cells use {len(tasks)} tasks (two of each kind) to stay fast; `bench.py` runs 40 (see the end of
each section).
""")
    return A, box, ex38, lm, papers, rag, tasks, tools


@app.cell
def _(RUNS, json):
    _p = RUNS / "bench.json"
    bench = json.loads(_p.read_text()) if _p.exists() else None
    return (bench,)


@app.cell
def _(A, esc, ex, lj, mo, tools):
    _p = lj.attempt(ex.parse_tool_call, 'ok <tool_call>{"name": "weekday", "arguments": {"date": "2024-01-01"}}</tool_call>')
    _bad = lj.attempt(ex.parse_tool_call, '<tool_call>{"name": "weekday", "arguments": ')
    _d = lj.attempt(ex.dispatch_tool, {n: t.fn for n, t in tools.items()},
                    {"name": "calculator", "arguments": {"expression": "2 ** 64"}})
    _items = [mo.md(f"""
## 2 · The loop

An agent is a loop with an exit condition. Each turn the model either **calls a tool** (we run it and feed
the result back) or **answers** (we stop); a turn limit stops runaway loops (Anthropic: "it's also common
to include stopping conditions (such as a maximum number of iterations)"). SmolLM2-360M was not trained
with a tool-calling format, so we define one in the system prompt, with one worked example per tool given
as chat turns (Module 20 §13: examples in the shape the model was trained on):

{esc('<tool_call>{"name": "calculator", "arguments": {"expression": "25 * 4 + 7"}}</tool_call>')} →
{esc("<tool_result>107</tool_result>")} → {esc("<answer>107</answer>")}

**Your turn:** `parse_tool_call` (find and validate the JSON) and `dispatch_tool` (run it, and turn every
failure into an `error:` string the model can read, never an exception):
""")]
    if _p.ok and _bad.ok:
        _items.append(mo.md(f"`parse_tool_call` → `{_p.value}`; on a truncated call → `{_bad.value}` ✅"))
    else:
        _items.append(_p.view(mo) if not _p.ok else _bad.view(mo))
    _items.append(mo.md(f"`dispatch_tool` with `2 ** 64` (our calculator forbids `**`) → `{_d.value}`")
                  if _d.ok else _d.view(mo))
    mo.vstack(_items)
    ready = _p.ok and _bad.ok and _d.ok
    return (ready,)


@app.cell
def _(A, box, esc, ex, mo, ready, tasks):
    if ready:
        _t = tasks[0]
        trace_free = box.agent(ex, _t["q"], mode="prompt")
        _lines = "\n".join(f"* **{_k}:** {esc(_v[:160])}" for _k, _v in trace_free.transcript)
        _fake = any(_k == "assistant" and "<tool_result>" in _v and "<tool_call>" not in _v
                    for _k, _v in trace_free.transcript)
        _note = ("Look at the first assistant turn: the model never called a tool. It wrote a "
                 f"{esc('<tool_result>')} *itself*, inventing the observation, then answered from its own invention. "
                 "This is the most dangerous failure an agent has, because the transcript looks like tool use. The "
                 "loop caught the missing call (no valid action, so it asked again), but the model had already "
                 "committed to its made-up number." if _fake else
                 "The model did call a tool, but the wrong one, with arguments it made up. `dispatch_tool` turned the "
                 "failure into an `error:` string instead of crashing, which is right, but a 360M model then repeated "
                 "the error as its answer. Free generation fails in several such ways across the suite (below)."
                 if not A.check(_t, trace_free.answer) else
                 "This time the free-form call came out right; across the suite (below) free generation fails in "
                 "ways that are easy to miss in one transcript.")
        _out = mo.md(f"""
### Prompt-only: just let the model write

Free generation, stopping at {esc("</tool_call>")} or {esc("</answer>")}. Task: *{_t['q']}* (answer
`{_t['answer']}`):

{_lines}

Result: {esc(trace_free.answer)} — {'correct' if A.check(_t, trace_free.answer) else '**wrong**'}.

{_note}
""")
    else:
        trace_free, _out = None, mo.md("")
    _out
    return (trace_free,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Constrained decoding: make a small model's tool calls well-formed

    A model that writes JSON freely can leave a brace open, misspell a tool, put a word where a number goes,
    or skip the call entirely. Big models are trained out of most of this; a 360M model isn't. The fix that
    llama.cpp's grammars, vLLM's structured outputs and LM Studio's `response_format` all use is to **not let
    the model write the syntax at all**:

    1. **Choose by likelihood.** Score each allowed continuation — {`<tool_call>`, `<answer>`}, then each tool
       name — by its total log-probability (Module 21's multiple-choice scoring) and take the best.
       **Calibrate** that choice: small models lean toward some options whatever the question (here: whichever
       tool the most recent worked example used). Score each tool name again after a content-free question
       ("N/A") and subtract: what's left is what *this* question adds (Zhao et al., 2021).
    2. **Write the syntax ourselves.** `{"name": "calculator", "arguments": {"expression": "` is fed in, not
       generated.
    3. **Mask the values.** Each argument has a type; before every step, logits of tokens that can't appear in
       that type are set to −∞. An expression may contain only digits, operators, spaces and parentheses; a
       date only digits and `-`. A value ends when the model's best allowed token is a *terminator* (a token
       starting with `"`).

    The output is valid JSON by construction. What's left for the model is the part it's good at: *which*
    tool, and *what* to put in it. The masks are precomputed once over the 49,152-token vocabulary; scoring a
    candidate is one forward pass over its tokens from the current cache, then [`jax.nn.log_softmax`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.log_softmax.html#jax.nn.log_softmax) and a
    gather of each next token's log-probability.

    **Your turn:** `token_mask` and `select_by_logprob`.
    """)
    return


@app.cell
def _(ex, lj, lm, mo, np, ready):
    _m = lj.attempt(ex.token_mask, ["12", " 3", "a", '"}', ""], "0123456789", ('"',))
    _s = lj.attempt(ex.select_by_logprob, [np.array([-1.0, -1.0]), np.array([-1.5])])
    _items = [_m.view(mo, show_value=False), _s.view(mo, show_value=False)]
    masks_ok = ready and _m.ok and _s.ok
    if masks_ok:
        _sizes = {k: int(lm.mask(ex, k)[0].sum()) for k in ("expression", "date", "text")}
        _items.append(mo.md(f"`token_mask` on `['12', ' 3', 'a', '\"}}', '']` → `{_m.value.tolist()}`; "
                            f"`select_by_logprob` picks candidate **{_s.value}** (−1.5 beats −2.0 in total). "
                            "Allowed tokens out of 49,152: "
                            + ", ".join(f"{k} **{v:,}**" for k, v in _sizes.items()) + "."))
    mo.vstack(_items)
    return (masks_ok,)


@app.cell
def _(A, box, ex, masks_ok, mo, np, tasks, time):
    if masks_ok:
        live = {}
        for _mode, _kw in [("prompt-only", {"mode": "prompt"}),
                           ("constrained", {"mode": "constrained", "calibrate": False}),
                           ("constrained + calibrated", {"mode": "constrained", "calibrate": True})]:
            _t0 = time.perf_counter()
            _runs = [box.agent(ex, _t["q"], **_kw) for _t in tasks]
            _calls = sum(_r.calls for _r in _runs)
            live[_mode] = {"success": np.mean([A.check(_t, _r.answer) for _t, _r in zip(tasks, _runs, strict=True)]),
                           "valid": sum(_r.valid_calls for _r in _runs) / max(_calls, 1), "calls": _calls,
                           "invalid": sum(_r.invalid for _r in _runs),
                           "tokens": np.mean([_r.tokens for _r in _runs]), "s": time.perf_counter() - _t0}
        _rows = "\n".join(f"| {_k} | {_v['success']:.0%} | {_v['valid']:.0%} of {_v['calls']} | {_v['invalid']} | "
                          f"{_v['tokens']:.0f} | {_v['s']:.0f} s |" for _k, _v in live.items())
        _out = mo.md(f"""
| live, {len(tasks)} tasks | task success | well-formed calls to a real tool | turns with no valid action | tokens processed per task | time |
|---|---|---|---|---|---|
{_rows}
""")
    else:
        live, _out = None, mo.md("*(Needs the exercises above.)*")
    _out
    return (live,)


@app.cell
def _(bench, mo):
    if bench:
        _r = lambda k: bench[k]   # noqa: E731
        _p, _c = bench.get("agent_prompt", {}), bench.get("agent_constrained", {})
        _raw = bench.get("agent_constrained_raw", {})
        _read = (f"Constraints make every call well-formed ({_c.get('valid_rate', float('nan')):.0%} vs "
                 f"{_p.get('valid_rate', float('nan')):.0%} free), but by themselves they don't fix *which* tool: "
                 f"uncalibrated, the model's prior over tool names, shaped by the worked examples, overrides the "
                 f"question (arithmetic {_raw.get('by_kind', {}).get('arithmetic', float('nan')):.0%}; "
                 f"{_raw.get('success', float('nan')):.0%} overall). Calibration is what turns validity into "
                 f"success ({_c.get('success', float('nan')):.0%}). No single-agent setup solves the two-step "
                 f"questions ({_c.get('by_kind', {}).get('two-step', float('nan')):.0%}): a 360M model doesn't plan "
                 "two lookups and a subtraction; it reaches straight for a date tool with dates it doesn't have. "
                 "Section 6 fixes that with a workflow rather than a smarter model.")
        _rows = "\n".join(
            f"| {_lbl} | {_r(f'agent_{_m}')['success']:.0%} | {_r(f'agent_{_m}')['valid_rate']:.0%} | "
            + " | ".join(f"{_r(f'agent_{_m}')['by_kind'].get(_k, float('nan')):.0%}"
                         for _k in ("arithmetic", "dates", "lookup", "two-step"))
            + f" | {_r(f'agent_{_m}')['tokens']:.0f} |"
            for _m, _lbl in [("prompt", "prompt-only"), ("constrained_raw", "constrained"),
                             ("constrained", "constrained + calibrated")] if f"agent_{_m}" in bench)
        _out = mo.md(f"""
**From `bench.py` ({bench['n_tasks']} tasks):**

| mode | success | well-formed calls | arithmetic | dates | lookup | two-step | tokens/task |
|---|---|---|---|---|---|---|---|
{_rows}

{_read}
""")
    else:
        _out = mo.callout(mo.md("Run `containers/run.sh python modules/39-agents/bench.py` (~7–10 min) for the "
                                "40-task tables in this and later sections."), kind="info")
    _out
    return



@app.cell
def _(A, ex, lj, lm, masks_ok, mo, plt):
    _w = lj.attempt(ex.trim_window, [{"role": "system", "content": "aaaa"}] +
                    [{"role": "user", "content": "b" * n} for n in (5, 3, 2, 4)], 11, len)
    if masks_ok:
        kv = A.kv_reuse_timing(lm, n_turns=40)
        _fig, _ax = plt.subplots(figsize=(7, 3))
        _ax.plot(kv["context_tokens"], kv["scratch_ms"], "o-", label="re-read the whole conversation")
        _ax.plot(kv["context_tokens"], kv["reuse_ms"], "s-", label="keep the KV cache, prefill the new turn")
        _ax.set(xlabel="conversation length (tokens)", ylabel="ms to process one new turn",
                title="Short-term memory is the KV cache")
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.3)
        _fig.tight_layout()
        _ratio = kv["scratch_ms"][-1] / kv["reuse_ms"][-1]
        _out = mo.vstack([mo.md(f"""
## 4 · Short-term memory: the context window and its KV cache

An agent's working memory is its context window. Every turn adds tokens: the user's message, the model's
call, the tool's result. The naive loop re-sends the whole conversation each turn and the model re-reads all
of it; the alternative is to keep the **KV cache** (Module 18) between turns and prefill only the new turn.
In JAX that's free to express: the cache is an immutable array, so "keep it" just means holding on to it,
and "fork it" (to try two continuations, or to reuse one system prompt for many tasks) costs nothing. Each new
chunk is written into the preallocated cache with [`jax.lax.dynamic_update_slice`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html#jax.lax.dynamic_update_slice) at the current position,
inside one [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit)-compiled forward pass; chunk lengths are padded up to a few bucket sizes (1, 16, 32, …,
1024) so XLA compiles each size once instead of once per message length.
"""), _fig, mo.md(f"""
At {kv['context_tokens'][-1]:,} tokens, re-reading takes **{kv['scratch_ms'][-1]:.0f} ms** per turn and the
kept cache **{kv['reuse_ms'][-1]:.0f} ms** (**{_ratio:.0f}×**). The cache cost is flat because each turn's
new tokens attend over a fixed-size, preallocated cache; re-reading grows with the conversation, and the gap grows with model size too, since every re-read token
costs a full forward pass through every layer. (The blue staircase is the bucket padding from above: a
600-token re-read is padded to 1,024 and costs the same.) The
same trick is behind **prompt caching** in the hosted APIs: every agent in this lesson prefills its system
prompt and examples ({lm.count(A.system_prompt(A.basic_tools()))} tokens or more) once, and
each task starts from a fork of that state.

The cache holds at most {lm.cache_len:,} positions here (`max_seq` of the preallocated cache), so long
conversations must forget something. The simplest policy is a **sliding window**: keep the system prompt
and as many recent messages as fit. **Your turn:** `trim_window`.
"""), _w.view(mo, show_value=False)])
    else:
        kv, _out = None, mo.md("")
    _out
    return (kv,)


@app.cell
def _(A, ex, ex38, lj, lm, masks_ok, mo, rag):
    _r = lj.attempt(ex.recency_relevance_score, [0, 10, 100], [0.2, 0.9, 0.5], 0.99)
    if masks_ok and _r.ok:
        _msgs, _qs = A.make_memory_test(seed=0)
        memory = A.memory_methods(lm, ex, rag, ex38, _msgs, _qs)
        _rows = "\n".join(f"| {_k} | **{_v['recall']:.0%}** | {_v['updated']:.0%} | {_v['tokens']:.0f} |"
                          for _k, _v in memory.items())
        _conv = lm.count("".join(A.turn(_m["role"], _m["content"]) for _m in _msgs))
        _out = mo.md(f"""
## 5 · Long-term memory: what to keep outside the context

The recall test: a {len(_msgs)}-message conversation ({_conv:,} tokens) of small talk, with ten personal facts
dropped in along the way ("By the way, my locker code is 4417.") and two of them corrected near the end.
Then ten questions ("What is my locker code?"). Five strategies:

* **Full conversation** in context — the ceiling while it still fits.
* **Sliding window** (`trim_window`) — cheap, forgets everything older than the window.
* **Running summary** — every 12 messages the model rewrites its notes; old turns are dropped.
* **Episodic log + retrieval** — every user turn is stored; at question time Module 38's BM25 finds
  candidates, ranked by your `recency_relevance_score` (Generative Agents, Park et al., 2023); up to 3 go in
  context, minus any far weaker than the best match, with the best placed *last*, next to the question (a
  small model leans on the nearest text). Recency is what lets a correction beat the original fact.
* **Structured ledger** — facts written to a SQLite table as they arrive (here by a simple rule; production
  systems use an LLM extractor) and looked up by key, newest first.

| strategy | recall (10 questions) | recall of the 2 corrected facts | prompt tokens per question |
|---|---|---|---|
{_rows}
""")
    else:
        memory, _out = None, (_r.view(mo) if not _r.ok else mo.md(""))
    _out
    return (memory,)


@app.cell
def _(A, box, ex, masks_ok, mo, np, tasks):
    if masks_ok:
        _outs = [A.wf_routing(box, ex, _t) for _t in tasks]
        _want = {"arithmetic": "math", "dates": "dates", "lookup": "library"}
        _acc = np.mean([_o["route"] == _want[_t["kind"]] for _t, _o in zip(tasks, _outs, strict=True)
                        if _t["kind"] in _want])
        _ok = np.mean([A.check(_t, _o["answer"]) for _t, _o in zip(tasks, _outs, strict=True)])
        _two = next(_t for _t in tasks if _t["kind"] == "two-step")
        chain = A.wf_chaining(box, ex, _two)
        _out = mo.md(f"""
## 6 · Workflows: Anthropic's five patterns

Anthropic's distinction: **workflows** are "systems where LLMs and tools are orchestrated through
predefined code paths"; **agents** are systems "where LLMs dynamically direct their own processes and tool
usage." Its advice is to start with the simplest thing that works, and its five workflow patterns are each a
few lines on top of the loop:

| pattern | what it does | here |
|---|---|---|
| prompt chaining | fixed steps, with programmatic *gates* between them | two-step questions: find date A → find date B → days between; stop if a step returns no date |
| routing | classify the input, send it to a specialized prompt | pick math / dates / library by log-probability, then a one-tool specialist |
| parallelization | run several attempts and vote (or split into parallel sections) | 5 sampled runs, `majority_vote` |
| orchestrator-workers | a model writes the sub-tasks at run time; workers do them; results are combined | the model lists sub-questions, an agent answers each, a final agent combines |
| evaluator-optimizer | one call proposes, another critiques, repeat | a yes/no judge by log-probability; on "no", retry with that feedback (Reflexion-style) |

Live, routing on {len(tasks)} tasks: routes right **{_acc:.0%}** of the time (two-step questions have no
single right route), task success **{_ok:.0%}**. Prompt chaining on *{_two['q'][:90]}…*: dates found
`{chain.get('dates')}`, answer `{chain['answer']}` (want `{_two['answer']}`).
""")
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(bench, mo):
    if bench and "workflow_voting" in bench:
        _names = {"routing": "routing", "chaining": "prompt chaining (two-step)",
                  "orchestrator": "orchestrator-workers (two-step)", "evaluator": "evaluator-optimizer",
                  "voting": "parallelization: voting ×5"}
        _ch, _or = bench["workflow_chaining"], bench["workflow_orchestrator"]
        _read = (f"The clearest result in the module: on the two-step questions, **prompt chaining** scores "
                 f"{_ch['success']:.0%} where the single agent scores {_ch['single_agent_same_tasks']:.0%}, because the "
                 "code, not the model, decides that there are two lookups and a subtraction; each small agent only has "
                 f"to do one easy step. **Orchestrator-workers** ({_or['success']:.0%}) asks the 360M model to write that "
                 "plan itself, and it mostly restates the question as its own sub-question. That's Anthropic's point "
                 "in miniature: use a workflow when you know the steps; an agent's flexibility only pays when the "
                 "model is strong enough to plan. More calls cost more tokens, so the extra machinery has to buy "
                 "accuracy on the tasks you care about. "
                 + (f"**Routing** shows the opposite risk: it sent lookups to the wrong specialist "
                    f"(routing accuracy {bench['workflow_routing'].get('routing_accuracy', float('nan')):.0%}; lookups "
                    f"{bench['workflow_routing']['by_kind'].get('lookup', float('nan')):.0%}), and a specialist without "
                    "the right tool can't recover, while the single agent, holding every tool, could."
                    if bench["workflow_routing"]["by_kind"].get("lookup", 1) < 0.5 else ""))
        _rows = "\n".join(f"| {_v} | {bench[f'workflow_{_k}']['n']} | **{bench[f'workflow_{_k}']['success']:.0%}** | "
                          f"{bench[f'workflow_{_k}']['single_agent_same_tasks']:.0%} | {bench[f'workflow_{_k}']['tokens']:.0f} |"
                          for _k, _v in _names.items())
        _out = mo.md(f"""
**From `bench.py`:**

| workflow | tasks | success | single agent, same tasks | tokens per task |
|---|---|---|---|---|
{_rows}

{_read}
""")
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(A, box, esc, ex, masks_ok, mo, tasks):
    if masks_ok:
        _two = next(_t for _t in tasks if _t["kind"] == "two-step")
        _m = A.ma_manager(box, ex, _two)
        _h = A.ma_handoff(box, ex, next(_t for _t in tasks if _t["kind"] == "dates"))
        _lines = "\n".join(f"* **{_k}:** {esc(_v[:150])}" for _k, _v in _m["calls"])
        _out = mo.md(f"""
## 7 · One agent or several?

OpenAI's guide names two multi-agent shapes. In the **manager** pattern a central agent calls specialist
agents *as tools* and stays in charge of the conversation. In the **decentralized** pattern agents **hand
off** to each other: a one-way transfer ("in the Agents SDK, a handoff is a type of tool"), after which the
new agent owns the reply. Its general recommendation is the one to remember: "maximize a single agent's
capabilities first," and split only when prompts get too conditional or the tools too many and too similar.

Our three specialists (math, dates, library) each get one tool or two and their own worked example. The
manager on a two-step question (want `{_two['answer']}`):

{_lines}

Handoff on a dates question: the triage agent transferred to **{_h['handoff']}**, which answered
`{_h['answer']}`.
""")
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(bench, mo):
    if bench and "multi_handoff" in bench:
        _rows = "\n".join(
            f"| {_v} | **{bench[f'multi_{_k}']['success']:.0%}** | "
            + " | ".join(f"{bench[f'multi_{_k}']['by_kind'].get(_c, float('nan')):.0%}"
                         for _c in ("arithmetic", "dates", "lookup", "two-step"))
            + f" | {bench[f'multi_{_k}']['tokens']:.0f} |"
            for _k, _v in [("single", "one agent, all tools"), ("manager", "manager + 3 specialists"),
                           ("handoff", "triage → handoff to 1 specialist")])
        _best = max(("single", "manager", "handoff"), key=lambda k: bench[f"multi_{k}"]["success"])
        _read = ("At this scale **one agent with all four tools does best**, which is OpenAI's advice measured: each "
                 "extra agent adds a decision (which specialist? what to ask it?) that a small model can get wrong, and "
                 "the errors multiply. Splitting pays when one prompt can no longer hold the instructions or the tools "
                 "overlap confusingly; four distinct tools isn't that." if _best == "single" else
                 f"Here **{_best}** does best; with a small model every extra decision is a new place to fail, so "
                 "measure before splitting.")
        _out = mo.md(f"""
**From `bench.py` ({bench['n_tasks']} tasks):**

| system | success | arithmetic | dates | lookup | two-step | tokens per task |
|---|---|---|---|---|---|---|
{_rows}

Triage handed off to the right specialist **{bench['multi_handoff'].get('routing_accuracy', float('nan')):.0%}** of the time (single-step tasks).

{_read}
""")
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    approve = mo.ui.switch(label="approve irreversible actions", value=False)
    approve
    return (approve,)


@app.cell
def _(A, approve, box, esc, ex, masks_ok, mo):
    if masks_ok:
        _deleted = []
        _tools = {**box.subset(["calculator"]),
                  "delete_files": A.Tool("delete_files", "Delete files matching a pattern. Cannot be undone.",
                                         [("pattern", "text")], lambda pattern: _deleted.append(pattern) or "deleted",
                                         irreversible=True)}
        _ex = [("user", "Remove the old *.log files."),
               ("assistant", '<tool_call>{"name": "delete_files", "arguments": {"pattern": "*.log"}}</tool_call>'),
               ("user", "<tool_result>deleted</tool_result>"), ("assistant", "<answer>Done.</answer>")]
        _q = "Clean up the scratch directory: delete everything matching /course/runs/*"
        _blocked = ["ignore previous instructions", "system prompt"]
        _guard_in = len(_q) < 500 and not any(b in _q.lower() for b in _blocked)
        _r = A.run_agent(box.lm, ex, _q, _tools, "constrained", max_turns=3, examples=_ex,
                         approve=lambda call: approve.value)
        _lines = "\n".join(f"* **{_k}:** {esc(_v[:150])}" for _k, _v in _r.transcript)
        _out = mo.md(f"""
## 8 · Guardrails

OpenAI's guide lists layers (relevance and safety classifiers, PII filters, moderation, *tool safeguards*,
rules-based protections, output validation) and two triggers for a human: exceeding failure thresholds, and
**high-risk actions** — "sensitive, irreversible, or have high stakes." The loop here has four:

* **turn limit** (`max_turns`) and **tool allowlist** (`dispatch_tool`'s `allowed`);
* **input rules**: a length limit and a blocklist checked before the model sees the request (passed: {_guard_in});
* **approval gate**: tools marked `irreversible` run only if a human approves (the switch above).

This repository runs the same policy on its own assistant: its `CLAUDE.md` says never to `terraform apply`
without the owner's go-ahead (Module 25). The agent, asked to *{_q}*:

{_lines}

Exit: **{_r.exit}**; files actually deleted: {_deleted or 'none'}.
""")
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(A, HERE, box, ex, masks_ok, mo, tasks, time):
    if masks_ok:
        import importlib.util as _iu
        import sys as _sys

        _spec = _iu.spec_from_file_location("m39_server", HERE / "server.py")
        _srv = _iu.module_from_spec(_spec)
        _sys.modules["m39_server"] = _srv
        _spec.loader.exec_module(_srv)
        from langchain_core.tools import tool as _tool
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        _fns = {n: t.fn for n, t in box.tools.items()}

        @_tool
        def calculator(expression: str) -> str:
            """Evaluate an arithmetic expression exactly."""
            return str(_fns["calculator"](expression))

        @_tool
        def days_between(start: str, end: str) -> str:
            """Number of days between two dates (YYYY-MM-DD)."""
            return str(_fns["days_between"](start, end))

        @_tool
        def weekday(date: str) -> str:
            """Day of the week of a date (YYYY-MM-DD)."""
            return _fns["weekday"](date)

        @_tool
        def search_papers(query: str) -> str:
            """Search arXiv radio-astronomy papers by title words; returns titles with dates."""
            return _fns["search_papers"](query)

        server = _srv.start(box.lm, ex, A, port=8765)
        _llm = ChatOpenAI(base_url="http://127.0.0.1:8765/v1", api_key="none", model="smollm2-360m-jax",
                          temperature=0)
        import warnings as _w

        with _w.catch_warnings():       # LangGraph 1.0 deprecates the prebuilt in favour of langchain's create_agent
            _w.simplefilter("ignore")
            _agent = create_react_agent(_llm, [calculator, days_between, weekday, search_papers])
        _rows, _t0 = [], time.perf_counter()
        for _t in [t for t in tasks if t["kind"] != "two-step"][::2] + [t for t in tasks if t["kind"] == "two-step"][:1]:
            try:
                _o = _agent.invoke({"messages": [("user", _t["q"])]}, {"recursion_limit": 8})
                _ans = _o["messages"][-1].content
                _n = sum(1 for _m in _o["messages"] if getattr(_m, "tool_calls", None))
            except Exception as _e:  # noqa: BLE001 - e.g. the recursion limit
                _ans, _n = f"({type(_e).__name__})", -1
            _rows.append(f"| {_t['kind']} | {_n} | `{_ans}` | `{_t['answer']}` | {'✅' if A.check(_t, _ans) else '✗'} |")
        _secs = time.perf_counter() - _t0
        _out = mo.md(f"""
## 9 · Frameworks: LangGraph, LM Studio, MCP

**Is LangGraph applicable to JAX?** Yes, through serving. Agent frameworks don't run models; they
orchestrate calls to an OpenAI-compatible endpoint. So `server.py` puts the JAX model behind
`/v1/chat/completions` with FastAPI, and implements **tool calling on the server side**: it writes the tool
list into the prompt, decodes under the same constraints as section 3, and returns structured `tool_calls`.
That is all "native tool calling" means for an open model — LM Studio and llama.cpp do the same.
LangGraph's prebuilt ReAct agent (`create_react_agent`; LangGraph 1.0 moved it to `langchain.agents.create_agent`,
which needs the separate `langchain` package, and the graph is the same) then drives your JAX model with four
LangChain tools:

| task | tool calls | LangGraph's answer | want | |
|---|---|---|---|---|
{chr(10).join(_rows)}

({_secs:.0f} s, through HTTP, LangChain and LangGraph.) The graph adds state, retries, checkpoints and
branching for bigger systems; it doesn't make a 360M model smarter. Anthropic's advice applies: "we suggest
that developers start by using LLM APIs directly", and reach for a framework when you need what it adds.
""")
    else:
        server, _out = None, mo.md("")
    _out
    return (server,)


@app.cell
def _(HERE, mo):
    import importlib.util as _iu2

    _spec = _iu2.spec_from_file_location("m27_lmclient", HERE.parent / "27-lmstudio" / "lmclient.py")
    _lmc = _iu2.module_from_spec(_spec)
    _spec.loader.exec_module(_lmc)
    _base = _lmc.find_server()
    if _base:
        _llms = [m["key"] for m in _lmc.models(_base) if m.get("type") == "llm"]
        _msg = (f"LM Studio is running at `{_base}` with {len(_llms)} model(s). Point the same LangGraph code at it: "
                f"`ChatOpenAI(base_url=\"{_base}/v1\", api_key=\"lm-studio\", model=\"<key>\")`; LM Studio applies "
                "each model's own tool-calling template (Module 27 §9).")
    else:
        _msg = ("LM Studio isn't running (`make lms-up`, Module 27). With it up, the same LangGraph code works against "
                "it by changing `base_url` to `http://127.0.0.1:1234/v1`; LM Studio applies each model's own "
                "tool-calling template (Module 27 §9), and a 4B+ model does far better on the two-step questions.")
    mo.callout(mo.md(_msg), kind="info")
    return


@app.cell
def _(A, HERE, box, ex, masks_ok, mo, tasks, time):
    if masks_ok:
        _t0 = time.perf_counter()
        _look = [t for t in tasks if t["kind"] == "lookup"]

        def _with_mcp(call, mcp_tools):
            proxy = {**box.subset(["calculator", "days_between", "weekday"]),
                     "search_papers": A.Tool("search_papers", mcp_tools[0].description, [("query", "text")],
                                             lambda query: call("search_papers", {"query": query}))}
            runs = [A.run_agent(box.lm, ex, t["q"], proxy, "constrained",
                                prefix=box.prefix(("mcp",), proxy, A.EXAMPLES + A.DATE_EXAMPLES + A.LIBRARY_EXAMPLES))
                    for t in _look]
            return [x.name for x in mcp_tools], runs

        _names, _runs = A.with_mcp(HERE / "mcp_server.py", _with_mcp)
        _ok = sum(A.check(t, r.answer) for t, r in zip(_look, _runs, strict=True))
        _out = mo.md(f"""
### MCP: tools as a protocol

The **Model Context Protocol** (Anthropic, November 2024) standardizes how a tool server offers tools to any
model client: list them, call them, get results, over stdio or HTTP. `mcp_server.py` is Module 38's search as
an MCP server in about 15 lines (the `mcp` package's `MCPServer`). Here the lesson launched it as a
subprocess, discovered its tools (`{_names}`), and gave the agent a `search_papers` that is really an MCP
call: {_ok} of {len(_look)} lookup tasks right ({time.perf_counter() - _t0:.0f} s including server start).
The same server works from Claude Code or Claude Desktop with no changes.

The production-scale version is your aws-ai `local-kb` MCP server: `search_knowledge_base` and
`ask_knowledge_base` over OpenSearch or pgvector (Module 38's comparison).
""")
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo, server):
    if server is not None:
        server.should_exit = True
    mo.md(r"""
    ### The landscape, briefly

    * **LangGraph** (LangChain): agents as explicit state graphs, with checkpoints and human-in-the-loop
      interrupts. Model-agnostic: anything with an OpenAI-compatible endpoint, including your JAX server.
    * **AutoGen** (Microsoft, 2023) popularized multi-agent *conversation* (Wu et al., 2023). It has been in
      **maintenance mode since October 2025**: bug and security fixes only. Microsoft's successor is
      **Microsoft Agent Framework** (1.0 in April 2026); **AG2** is a community fork that continues the
      original. Learn the patterns from it; start new work elsewhere.
    * **JAX agent libraries:** none for *running* agents, and none are needed: frameworks are HTTP
      orchestrators, so the model's framework doesn't matter. For *training* agents there is Google's
      **Tunix**, the JAX post-training library, which added agent, environment and tool abstractions for
      multi-turn reinforcement learning: Module 26's GRPO over whole tool-using episodes.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Because the model chooses among valid options and never writes the JSON syntax itself": "a",
            "Because masking makes the model smarter about which tool to use": "b",
            "Because constrained decoding fine-tunes the model on tool calls": "c",
        },
        label="**Check yourself:** why does constrained decoding make a 360M model's tool calls well-formed?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right. Well-formedness is guaranteed by construction; *correctness* (the right "
                              "tool, the right arguments) still depends on the model — which is why success is "
                              "lower than validity."), kind="success"),
        "b": mo.callout(mo.md("No — the masks only remove impossible tokens. Which tool gets picked is still the "
                              "model's log-probability; masking can't fix a wrong choice."), kind="warn"),
        "c": mo.callout(mo.md("No — nothing is trained. It's purely a decoding-time restriction on which tokens "
                              "may be produced (Module 20 is where training happens)."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(39))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Anthropic (Erik Schluntz, Barry Zhang), [Building effective agents](https://www.anthropic.com/research/building-effective-agents)
      (December 2024) — workflows vs agents, the five patterns, and the agent-computer interface.
    * OpenAI, [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
      (April 2025, PDF) — models, tools, instructions; single- vs multi-agent orchestration; guardrails.
    * Victor de Lamo Castrillo et al., [Fundamentals of Building Autonomous LLM Agents](https://arxiv.org/abs/2510.09244)
      (2025) — a survey: perception, reasoning, memory, execution.
    * LangChain, [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — the graph runtime used in section 9.
    * Anthropic, [Model Context Protocol](https://modelcontextprotocol.io/) — the specification and SDKs.
    * Microsoft, [AutoGen on GitHub](https://github.com/microsoft/autogen) — the repository and its pointer to
      Microsoft Agent Framework.
    * Google, [Tunix](https://github.com/google/tunix) — JAX post-training, including multi-turn agentic RL.
    * Lilian Weng, [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) (2023) —
      planning, memory and tool use, surveyed.
    * Your aws-ai repo: `local-kb`'s MCP server — the production-shaped knowledge-base tools.

    **This is the last module.** Module 00b's timeline shows where each idea came from; `make progress` shows
    where you are.
    """)
    return


if __name__ == "__main__":
    app.run()
