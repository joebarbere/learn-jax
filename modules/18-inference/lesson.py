import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="18 · Inference")


@app.cell
def _():
    import marimo as mo
    import html
    import importlib.util
    import time
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import llama
    from learnjax import rosetta

    # Compiled programs are cached on disk (the /cache volume), so re-running this lesson skips most compiles.
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    ex = lj.load_exercises(mo.notebook_dir())
    _spec = importlib.util.spec_from_file_location("m18_infer", mo.notebook_dir() / "infer.py")
    I = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(I)
    return I, SimpleNamespace, ex, html, jax, jnp, lj, llama, mo, np, plt, rosetta, time


@app.cell
def _(mo):
    mo.md(r"""
    # 18 · Inference: KV cache, sampling, quantization

    **Goal:** understand what limits the speed of a language model *after* training, and the handful of
    tricks every inference engine (llama.cpp, vLLM, Ollama) is built from — measured on a real pretrained
    model on your RX 7600 XT.

    **The model:** [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) by Hugging Face's SmolLM team
    (Apache-2.0), loaded into `learnjax/llama.py` — a pure-JAX Llama made of exactly the parts you built
    in Module 17 (RMSNorm, RoPE, grouped-query attention, SwiGLU) and checked against Hugging Face
    `transformers` (identical greedy tokens). Its little sibling SmolLM2-135M shares the tokenizer, which
    makes it a natural *draft* model.

    **You will:** measure prefill vs decode and see why they're limited by different things · predict
    decode speed from memory bandwidth (a roofline) · speed up decoding by keeping the loop on the GPU ·
    batch requests for throughput · size the KV cache · implement top-k, top-p and repetition
    penalties and watch what they do · run speculative decoding (and see honestly why it doesn't pay off
    at this size) · quantize the model to int8 and int4 and measure what it costs in quality.

    Helpers live in `infer.py` next to this notebook — read it alongside the lesson.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Caching keys and values is as old as autoregressive Transformers. Holtzman et al. (2019) showed why
    greedy decoding repeats itself and proposed nucleus sampling. Speculative decoding (Leviathan, Kalman &
    Matias, 2022) uses a small draft model to speed up a big one. Quantization went from 8-bit inference
    (LLM.int8(), Dettmers et al., 2022) to 4-bit weights (GPTQ, Frantar et al., 2022), which made large
    models fit on ordinary GPUs.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(I, llama, mo):
    t360, c360, tok = llama.load(llama.SMOLLM_360M)
    t135, c135, _tok135 = llama.load(llama.SMOLLM_135M)
    stop_id = tok.token_to_id("<|im_end|>")
    code_ids = llama.chat_prompt(tok, "Write a Python function that checks whether a number is prime.")
    mo.md(f"""
    | model | parameters | bytes in bf16 | layers | heads (query / KV) |
    |---|---|---|---|---|
    | SmolLM2-360M-Instruct (target) | {llama.count_params(t360) / 1e6:.0f}M | {I.param_bytes(t360) / 1e6:.0f} MB | {c360.n_layers} | {c360.n_heads} / {c360.n_kv_heads} |
    | SmolLM2-135M (draft, base) | {llama.count_params(t135) / 1e6:.0f}M | {I.param_bytes(t135) / 1e6:.0f} MB | {c135.n_layers} | {c135.n_heads} / {c135.n_kv_heads} |

    Weights download once into the cache volume. The prompt used for speed tests is a chat-formatted
    request: *"Write a Python function that checks whether a number is prime."* ({len(code_ids)} tokens).
    """)
    return c135, c360, code_ids, stop_id, t135, t360, tok


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Two phases: prefill and decode

    Generating a reply happens in two very different phases:

    * **Prefill** — the whole prompt goes through the model in **one** forward pass. Hundreds of tokens
      share every weight read, so the GPU does lots of math per byte: it's **compute-bound**, like training.
    * **Decode** — then tokens come out **one at a time**: each new token needs a full forward pass, but
      only for that single position (the KV cache from Modules 16–17 holds everything earlier). Every
      weight is read from memory to do just ~2 FLOPs with it. Decode is **memory-bandwidth-bound**.

    The numbers below make the difference concrete. "Effective GB/s" = how many bytes of weights the
    model streams per second (bytes per token × tokens per second).
    """)
    return


@app.cell
def _(I, c360, code_ids, jax, jnp, llama, mo, np, t360, time):
    _N = llama.count_params(t360)
    _bytes = I.param_bytes(t360)
    _x = jnp.asarray(np.random.default_rng(0).integers(0, c360.vocab, (1, 512)), jnp.int32)
    jax.block_until_ready(I.prefill(t360, c360, _x))
    _t = time.perf_counter()
    for _ in range(3):
        jax.block_until_ready(I.prefill(t360, c360, _x))
    _dt = (time.perf_counter() - _t) / 3
    prefill_tok_s = 512 / _dt
    llama.generate(t360, c360, code_ids, 64, jax.random.key(0), temperature=0.0)      # compile (same shapes)
    _t = time.perf_counter()
    py_tokens = llama.generate(t360, c360, code_ids, 64, jax.random.key(0), temperature=0.0)
    py_tok_s = len(py_tokens) / (time.perf_counter() - _t)
    mo.md(f"""
    | phase (360M, bf16) | tokens/s | math rate (2·N FLOPs/token) | weights streamed |
    |---|---|---|---|
    | prefill, 512-token prompt | **{prefill_tok_s:,.0f}** | {2 * _N * prefill_tok_s / 1e12:.1f} TFLOP/s | — (each weight read once for all 512) |
    | decode, `llama.generate` | **{py_tok_s:.0f}** | {2 * _N * py_tok_s / 1e12:.3f} TFLOP/s | {_bytes * py_tok_s / 1e9:.0f} GB/s |

    Prefill runs at a decent fraction of the card's ~27 bf16 TFLOP/s (Module 00). Decode uses well under
    1% of the compute — the GPU spends its time *waiting for weights*, not multiplying them. (If another
    program is using the GPU, all timings here drop; re-run a cell to re-measure.)
    """)
    return prefill_tok_s, py_tok_s, py_tokens


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Decode as a roofline

    If every token must read every weight once, the fastest decode possible is

    $$\text{tokens/s} \le \frac{\text{memory bandwidth}}{\text{bytes of weights}}$$

    The RX 7600 XT's memory bandwidth is about **288 GB/s**. This "roofline" (a paper in this module's
    reading list) tells you what to expect before you run anything — and what to change: fewer bytes per
    weight (quantization, §8) raises the ceiling directly.
    """)
    return


@app.cell
def _(ex, lj, llama, mo, t360):
    _a = lj.attempt(ex.decode_roofline_tok_s, llama.count_params(t360), 2, 288)
    roof360 = _a.value if _a.ok else None
    mo.vstack([mo.md("**`decode_roofline_tok_s(360M params, 2 bytes, 288 GB/s)`** — about 398 tokens/s:"),
               _a.view(mo)])
    return (roof360,)


@app.cell
def _(ex, lj, llama, mo, t135, t360):
    _rows = []
    for _name, _p in (("SmolLM2-135M", t135), ("SmolLM2-360M", t360)):
        _n = llama.count_params(_p)
        _cells = [lj.attempt(ex.decode_roofline_tok_s, _n, _b, 288) for _b in (2, 1, 0.5)]
        if all(_c.ok for _c in _cells):
            _rows.append(f"| {_name} | " + " | ".join(f"{_c.value:,.0f}" for _c in _cells) + " |")
    if _rows:
        _out = mo.md("| decode ceiling (tok/s) | bf16 (2 B) | int8 (1 B) | int4 (0.5 B) |\n|---|---|---|---|\n"
                     + "\n".join(_rows)
                     + "\n\nSmaller *and* lower-precision models decode faster. A 7B model in bf16 (14 GB) "
                       "would top out near 20 tok/s on this card; in 4-bit, near 80.")
    else:
        _out = mo.md("*(Implement `decode_roofline_tok_s` to fill in this table.)*")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Why `generate` is slow, and the fix

    `llama.generate` is a Python loop: for every token it calls the jitted forward pass, then `int(tok)`
    to read the token back so Python can decide what to do next. That read **waits for the GPU to finish**
    (Module 06's async dispatch), then Python launches the next step — so the GPU idles between tokens,
    and each step pays launch overhead for dozens of small kernels.

    The fix: put the whole loop inside one compiled program with [`lax.fori_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html#jax.lax.fori_loop). The cache has a fixed
    size (prompt + new tokens), so the loop body has static shapes and compiles once:

    ```python
    def body(i, carry):                      # from infer.decode_on_device
        cache, tok, key, out = carry
        logits, cache = llama.forward(params, tok[:, None], cfg, cache, T + i - 1)
        tok = sample(key, logits[:, -1], ...)
        return cache, tok, key, out.at[:, i].set(tok)
    out = jax.lax.fori_loop(1, n_new, body, (cache, first_tok, key, out))
    ```

    The trade-off: you can't stop early on an end-of-reply token without a `while_loop` (or generating a
    few wasted tokens) — real engines handle that with a fixed budget plus a "done" flag per sequence.
    """)
    return


@app.cell
def _(I, c135, c360, code_ids, jax, jnp, mo, np, plt, py_tok_s, py_tokens, roof360, t135, t360, time):
    def _time_dev(params, cfg, n=64):
        _p = jnp.asarray(code_ids, jnp.int32)[None]
        jax.block_until_ready(I.decode_on_device(params, cfg, _p, n, jax.random.key(0)))   # compile
        _t = time.perf_counter()
        _o = I.decode_on_device(params, cfg, _p, n, jax.random.key(0))
        jax.block_until_ready(_o)
        return n / (time.perf_counter() - _t), np.asarray(_o[0]).tolist()

    dev_tok_s, _dev_tokens = _time_dev(t360, c360)
    dev135_tok_s, _ = _time_dev(t135, c135)
    _same = _dev_tokens[: len(py_tokens)] == py_tokens
    _sync_ms = 1e3 / py_tok_s - 1e3 / dev_tok_s
    _fig, _ax = plt.subplots(figsize=(6.5, 2.6))
    _labels = ["Python loop\n(llama.generate)", "on-device loop\n(fori_loop)"]
    _vals = [py_tok_s, dev_tok_s]
    _ax.barh(_labels, _vals, color=["#bbb", "C0"])
    if roof360:
        _ax.axvline(roof360, color="C3", ls="--", label=f"bandwidth roofline {roof360:.0f}")
        _ax.legend(loc="lower right", fontsize=8)
    _ax.set_xlabel("decode tokens/s (360M, bf16, batch 1)")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"""
    On-device decoding: **{dev_tok_s:.0f} tok/s** vs **{py_tok_s:.0f}** for the Python loop
    (**{dev_tok_s / py_tok_s:.1f}×**), and the greedy tokens are identical: **{_same}**. The Python loop
    costs about **{_sync_ms:.1f} ms extra per token** — the round trip to the host — on top of
    {1e3 / dev_tok_s:.1f} ms of GPU work. The smaller the model, the bigger that fixed cost looms: the
    135M model decodes at {dev135_tok_s:.0f} tok/s on-device.
    """),
        _fig,
        mo.md("""
    Still well under the roofline. What's left is mostly **per-layer overhead**: each token runs 32
    layers × ~15 small kernels, each reading a few MB — too little work per launch to reach full bandwidth
    at batch size 1. That's why the next trick, batching, helps so much.
    """),
    ])
    return dev135_tok_s, dev_tok_s


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "KV cache write")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Batching: same weights, more tokens

    Decode reads all the weights to produce one token per sequence. With a batch of $B$ sequences, the
    *same* weight read produces $B$ tokens — so total throughput rises almost for free until compute (or
    KV-cache memory) becomes the limit. This is why servers batch many users' requests together, and why
    a single chat feels slower than a server's advertised throughput.
    """)
    return


@app.cell
def _(I, c360, code_ids, jax, jnp, mo, plt, t360, time):
    _bs, _rates = [1, 8, 32], []
    for _b in _bs:
        _p = jnp.tile(jnp.asarray(code_ids, jnp.int32)[None], (_b, 1))
        jax.block_until_ready(I.decode_on_device(t360, c360, _p, 64, jax.random.key(0)))
        _t = time.perf_counter()
        jax.block_until_ready(I.decode_on_device(t360, c360, _p, 64, jax.random.key(0)))
        _rates.append(_b * 64 / (time.perf_counter() - _t))
    _fig, _ax = plt.subplots(figsize=(5.5, 3))
    _ax.plot(_bs, _rates, "o-", label="total tokens/s")
    _ax.plot(_bs, [_rates[0] * _b for _b in _bs], ":", color="gray", label="perfect scaling")
    _ax.set(xscale="log", yscale="log", xlabel="batch size (sequences decoded together)",
            ylabel="tokens/s", title="decode throughput, 360M bf16")
    _ax.set_xticks(_bs, [str(_b) for _b in _bs])
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"Batch 32 delivers **{_rates[-1]:,.0f} tok/s** in total — {_rates[-1] / _rates[0]:.0f}× batch 1 — "
        f"while each individual sequence gets {_rates[-1] / 32:.0f} tok/s. Throughput and per-user latency "
        f"are different numbers; inference engines trade one for the other.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · How big is the KV cache?

    Every generated or prompt token leaves a key and a value vector in **every layer** for every KV head.
    For long contexts and big batches this rivals the weights themselves — it's what limits how many users
    a server can batch, and why Module 17's grouped-query attention and MLA exist.
    """)
    return


@app.cell
def _(I, SimpleNamespace, c360, ex, lj, mo, t360):
    _a = lj.attempt(ex.kv_cache_bytes, c360, 8192, 1)
    _items = [mo.md("**`kv_cache_bytes(SmolLM2-360M, 8192 positions, batch 1)`** — about 336 MB:"), _a.view(mo)]
    if _a.ok:
        _mha = SimpleNamespace(n_layers=c360.n_layers, n_kv_heads=c360.n_heads, head_dim=c360.head_dim)
        _w = I.param_bytes(t360) / 1e9
        _rows = []
        for _s, _b in ((2048, 1), (8192, 1), (8192, 32)):
            _g = ex.kv_cache_bytes(c360, _s, _b) / 1e9
            _m = ex.kv_cache_bytes(_mha, _s, _b) / 1e9
            _rows.append(f"| {_s:,} | {_b} | {_g:.2f} GB | {_m:.2f} GB | {_g / _w:.1f}× |")
        _items.append(mo.md(
            f"| context | batch | GQA cache ({c360.n_kv_heads} KV heads) | if it were MHA ({c360.n_heads} heads) "
            f"| GQA cache ÷ weights ({_w:.2f} GB) |\n|---|---|---|---|---|\n" + "\n".join(_rows)
            + f"\n\nGrouped-query attention shares each KV head among {c360.n_heads // c360.n_kv_heads} query heads, "
              "so the cache is 3× smaller than plain multi-head attention would need. At 32 users × 8K context "
              "the cache still dwarfs the model — which is exactly the problem paged KV caches (vLLM, Module 19) "
              "and MLA's compressed cache (Module 17) attack."))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Choosing the next token

    The model outputs a probability for all 49,152 tokens. How you pick from them changes the text a lot:

    * **Greedy** — always the most likely token. Deterministic, and prone to loops: once a phrase is
      likely, repeating it makes it *more* likely.
    * **Temperature** $T$ — divide logits by $T$ before softmax. $T<1$ sharpens, $T>1$ flattens.
    * **Top-k** — sample only among the $k$ most likely tokens.
    * **Top-p (nucleus)** — sample among the smallest set of tokens whose probabilities add up to $p$.
      Adapts to the model's confidence: few candidates when it's sure, many when it isn't.
    * **Repetition penalty** — shrink the logits of tokens already in the text.

    Move the sliders: bars show the next-token distribution after the prompt below; grey bars are tokens
    your filters remove.
    """)
    return


@app.cell
def _(jnp, llama, t360, c360, tok):
    story_prompt = "Once upon a time, in a small village by the sea, there lived a"
    story_ids = tok.encode(story_prompt).ids
    _logits, _ = llama.forward_jit(t360, jnp.asarray(story_ids)[None], c360)
    story_logits = _logits[0, -1]
    return story_ids, story_logits, story_prompt


@app.cell
def _(mo):
    temp = mo.ui.slider(0.1, 2.0, step=0.1, value=1.0, label="temperature", show_value=True)
    topk = mo.ui.slider(0, 50, step=1, value=0, label="top-k (0 = off)", show_value=True)
    topp = mo.ui.slider(0.05, 1.0, step=0.05, value=1.0, label="top-p (1 = off)", show_value=True)
    mo.hstack([temp, topk, topp])
    return temp, topk, topp


@app.cell
def _(ex, jax, jnp, lj, mo, np, plt, story_logits, story_prompt, temp, tok, topk, topp):
    _x = story_logits / temp.value
    _fk = lj.attempt(ex.top_k_filter, _x, topk.value)
    _fp = lj.attempt(ex.top_p_filter, _fk.value if _fk.ok else _x, topp.value)
    _filtered = _fp.value if _fp.ok else (_fk.value if _fk.ok else _x)
    _probs = np.asarray(jax.nn.softmax(_x))
    _kept = np.isfinite(np.asarray(_filtered))
    _top = np.argsort(-_probs)[:20]
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.bar(range(20), _probs[_top], color=["C0" if _kept[i] else "#ccc" for i in _top])
    _ax.set_xticks(range(20), [repr(tok.decode([int(i)]))[1:-1] for i in _top], rotation=60, fontsize=8)
    _ax.set_ylabel("probability")
    _ax.set_title(f"next token after “…{story_prompt[-25:]}”  (T={temp.value})", fontsize=9)
    _fig.tight_layout()
    _n_kept = int(_kept.sum())
    _mass = float(jnp.sum(jnp.where(jnp.asarray(_kept), jax.nn.softmax(_x), 0.0)))
    _notes = [_fig, mo.md(f"**{_n_kept:,}** of {_x.shape[0]:,} tokens can be sampled, covering "
                          f"{_mass:.0%} of the probability before renormalizing.")]
    if not (_fk.ok and _fp.ok):
        _notes.append(mo.md("*(Implement `top_k_filter` and `top_p_filter` in `exercises.py` to see them "
                            "grey out tokens; until then only temperature applies.)*"))
    mo.vstack(_notes)
    return


@app.cell
def _(mo, rosetta):
    mo.vstack([
        rosetta.tabs(mo, "sort and argsort"),
        rosetta.tabs(mo, "temperature and top-k filter", solves="top_k_filter"),
        rosetta.tabs(mo, "top-p (nucleus) filter", solves="top_p_filter"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Side by side

    Three continuations of the same prompt from **SmolLM2-135M** (a base model — no chat tuning, so its
    habits show), 80 tokens each, with the sampler built from *your* functions. Greedy is deterministic
    and tends toward safe, repetitive phrasing (on long outputs it can loop outright); nucleus sampling
    varies; a repetition penalty pushes greedy away from words it already used — at some cost to fluency.
    """)
    return


@app.cell
def _(I, c135, ex, html, jax, jnp, lj, mo, story_ids, story_prompt, t135, tok):
    def _pick(temperature, top_p, penalty):
        def pick(logits, prev, key):
            if penalty != 1.0:
                logits = ex.apply_repetition_penalty(logits, prev, penalty)
            if temperature == 0:
                return jnp.argmax(logits)
            logits = ex.top_p_filter(logits / temperature, top_p)
            return jax.random.categorical(key, logits)
        return pick

    _ok = all(lj.attempt(_f, jnp.zeros(4), *_a).ok for _f, _a in (
        (ex.top_p_filter, (0.9,)), (ex.apply_repetition_penalty, (jnp.array([0]), 1.3))))
    if _ok:
        _runs = [("greedy", _pick(0, 1.0, 1.0)), ("T=0.8, top-p 0.9", _pick(0.8, 0.9, 1.0)),
                 ("greedy + repetition penalty 1.3", _pick(0, 1.0, 1.3))]
        _rows = []
        for _name, _fn in _runs:
            _ids = I.generate_with(t135, c135, story_ids, 80, jax.random.key(3), _fn)
            _txt = html.escape(tok.decode(_ids)).replace("\n", " ⏎ ")
            _rows.append(f"**{_name}:** …{html.escape(story_prompt[-20:])}<em>{_txt}</em>")
        _out = mo.md("\n\n".join(_rows))
    else:
        _out = mo.callout(mo.md("**Your turn:** implement `top_p_filter` and `apply_repetition_penalty` "
                                "to generate these samples."), kind="info")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Speculative decoding

    Decode is memory-bound, so checking **several** tokens in one forward pass costs about the same as
    checking one. Speculative decoding exploits that:

    1. A small **draft** model (SmolLM2-135M) proposes $k$ tokens, one at a time — cheap.
    2. The big **target** model (360M) scores the current token plus all $k$ drafts in **one** pass,
       giving its own choice at each of the $k+1$ positions.
    3. Accept drafts left to right while they match the target's choice; at the first mismatch, take the
       target's token instead. If all $k$ match, the target's last prediction is a free bonus token.

    With greedy acceptance the output is the target's own greedy output — the draft only changes speed.
    You write step 3: `speculative_accept`.
    """)
    return


@app.cell
def _(ex, jax, lj, mo):
    _target = jax.nn.one_hot(jax.numpy.array([3, 7, 2, 9]), 10) * 5.0
    _a = lj.attempt(ex.speculative_accept, jax.numpy.array([3, 7, 5]), _target)
    _items = [mo.md("**`speculative_accept(drafts=[3, 7, 5], target greedy=[3, 7, 2, 9])`** — "
                    "should accept 2 and correct to token 2:"), _a.view(mo, show_value=False)]
    if _a.ok:
        _items.append(mo.md(f"→ accepted **{int(_a.value[0])}**, next token **{int(_a.value[1])}**"))
    accept_ok = _a.ok
    mo.vstack(_items)
    return (accept_ok,)


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "cumulative product")
    return


@app.cell
def _(I, accept_ok, c135, c360, code_ids, ex, jax, mo, np, t135, t360, time):
    def _dev_rate(params, cfg):
        _p = jax.numpy.asarray(code_ids)[None]
        jax.block_until_ready(I.decode_on_device(params, cfg, _p, 64, jax.random.key(0)))
        _t = time.perf_counter()
        _o = I.decode_on_device(params, cfg, _p, 64, jax.random.key(0))
        jax.block_until_ready(_o)
        return 64 / (time.perf_counter() - _t), np.asarray(_o[0]).tolist()

    if accept_ok:
        _acc = jax.jit(ex.speculative_accept)
        _rows, _rates, _alphas, _diffs = [], [], [], []
        for _k in (1, 2, 4):
            I.speculative_generate(t360, c360, t135, c135, code_ids, 64, _k, _acc)          # compile
            _t = time.perf_counter()
            _out, _st = I.speculative_generate(t360, c360, t135, c135, code_ids, 64, _k, _acc)
            _rates.append(64 / (time.perf_counter() - _t))
            _alphas.append(_st["accepted"] / _st["drafted"])
            _base_rate, _base = _dev_rate(t360, c360)          # baseline measured right after, same conditions
            _diffs.append(next((i for i, (a, b) in enumerate(zip(_out, _base)) if a != b), None))
            _rows.append(f"| {_k} | {_alphas[-1]:.0%} | {64 / _st['rounds']:.2f} | {_rates[-1]:.0f} | "
                         f"{_base_rate:.0f} | "
                         f"{'identical' if _diffs[-1] is None else f'first differs at token {_diffs[-1]}'} |")
        _t360, _ = _dev_rate(t360, c360)
        _t135, _ = _dev_rate(t135, c135)
        _c = _t360 / _t135                                   # draft step cost relative to a target step
        _pred = {_k: (1 - _a ** (_k + 1)) / ((1 - _a) * (_k * _c + 1)) for _k, _a in zip((1, 2, 4), _alphas)}
        _ratio = max(_rates) / _t360
        _verdict = ("**slower** than" if _ratio < 0.95 else "about the **same speed** as" if _ratio < 1.1
                    else "only modestly faster than" if _ratio < 1.5 else "clearly faster than")
        _why_diff = ("" if all(_d is None for _d in _diffs) else """

    *Why "first differs at…"?* In exact arithmetic the output is identical. In bf16, scoring several tokens
    in one pass rounds slightly differently from scoring them one at a time, and where two tokens are nearly
    tied the argmax can flip — after which the texts diverge. Same lesson as Module 08: different shapes,
    different rounding.""")
        _out = mo.md(f"""
    | drafts per round $k$ | drafts accepted | tokens per round | speculative tok/s | target alone, on-device | vs target's own greedy output |
    |---|---|---|---|---|---|
    """ + "\n".join(_rows) + f"""

    At best, speculative decoding is {_verdict} the target decoding alone on this card. The reasons the
    gain is small (or negative) at this scale:

    * **The draft isn't cheap enough.** One 135M step costs **{_c:.2f}×** a 360M step here — both are
      dominated by per-layer overhead (30 vs 32 layers), not by weight bytes. The original paper's analysis
      predicts a speedup of $\\frac{{1-\\alpha^{{k+1}}}}{{(1-\\alpha)(kc+1)}}$ for acceptance rate
      $\\alpha$ and cost ratio $c$; with this run's numbers that's
      {", ".join(f"k={_k}: {_v:.2f}×" for _k, _v in _pred.items())}, *before* paying for Python between
      rounds (each round syncs with the host to read the accepted count).
    * **A base model drafting for a chat model** agrees less often than a matched pair would.

    It pays off when the target is big and truly memory-bound — a 70B model with a 1B draft has
    $c \\approx 0.02$ — which is how llama.cpp and vLLM use it.{_why_diff}
    """)
    else:
        _out = mo.md("*(Implement `speculative_accept` to run speculative decoding on the real models.)*")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Quantization: fewer bytes per weight

    Decode speed is bytes ÷ bandwidth, and the weights are most of the bytes. Store them in 8 or 4 bits
    instead of 16 and the ceiling rises 2–4×; the model also fits in less memory (a 7B model in 4-bit is
    ~4 GB — Module 19 runs exactly that).

    * **int8, per output channel:** each column of a weight matrix gets a scale $s = \max|w| / 127$, and
      $q = \text{round}(w/s)$. Rounding error is at most half a step: $s/2$.
    * **int4, grouped:** only 16 levels, so one scale per column isn't enough — a single large weight
      would make the step huge for everything else. Instead every **group** of 32–128 consecutive weights
      gets its own scale. This is the idea behind llama.cpp's `Q4_K` and GPTQ/AWQ formats.

    We quantize every matmul weight in the 32 layers and keep the embedding table (also the output head)
    in bf16, a common choice.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo, np, plt, t360):
    _w = t360["layers"]["w_gate"][0].astype(jnp.float32)          # layer 0's gate projection, [960, 2560]
    _q8 = lj.attempt(ex.quantize_int8, _w)
    _q4 = lj.attempt(ex.quantize_int4_grouped, _w, 64)
    if _q8.ok and _q4.ok:
        _e8 = np.asarray(ex.dequantize_int8(_q8.value) - _w).ravel()
        _e4 = np.asarray(ex.dequantize_int4_grouped(_q4.value) - _w).ravel()
        _fig, _ax = plt.subplots(1, 2, figsize=(9, 2.8))
        _ax[0].hist(np.asarray(_w).ravel(), bins=200, color="C0")
        _ax[0].set(title="layer-0 gate weights", yscale="log")
        _ax[1].hist(_e8, bins=200, alpha=0.7, label=f"int8 (rms {_e8.std():.1e})")
        _ax[1].hist(_e4, bins=200, alpha=0.7, label=f"int4 g64 (rms {_e4.std():.1e})")
        _ax[1].set(title="rounding error", yscale="log")
        _ax[1].legend(fontsize=8)
        _fig.tight_layout()
        _out = mo.vstack([_fig, mo.md(
            f"Weights are bell-shaped with a few large outliers (note the log scale) — the reason per-group "
            f"scales help. int4's error is ~{_e4.std() / _e8.std():.0f}× int8's.")])
    else:
        _out = mo.vstack([mo.md("**`quantize_int8` / `quantize_int4_grouped`:**"),
                          (_q8 if not _q8.ok else _q4).view(mo)])
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "int8 absmax quantization", solves=("quantize_int8", "dequantize_int8"))
    return


@app.cell
def _(I, c360, ex, jnp, lj, mo, np, t360, tok):
    from learnjax.data import detective_corpus as _detective_corpus
    from tokenizers import Tokenizer as _Tokenizer

    _d = _detective_corpus()
    _text = _Tokenizer.from_file(str(_d["tokenizer_path"])).decode(_d["val"][:40000].tolist())
    _ids = np.asarray(tok.encode(_text).ids)
    _batch = jnp.asarray(_ids[: 16 * 512].reshape(16, 512))

    def _ppl(params):
        return float(jnp.exp(jnp.mean(jnp.stack([I.nll_per_token(params, c360, _batch[i:i + 4])
                                                  for i in range(0, 16, 4)]))))

    _ok = lj.attempt(ex.quantize_int8, jnp.ones((4, 4))).ok and lj.attempt(ex.quantize_int4_grouped,
                                                                           jnp.ones((64, 4)), 32).ok
    q8_params = None
    if _ok:
        _base = _ppl(t360)
        q8_params = I.quantize_params(t360, ex.quantize_int8)
        _rows = [("bf16 (original)", _base, I.param_bytes(t360))]
        _rows.append(("int8 per-channel", _ppl(I.dequantize_params(q8_params, ex.dequantize_int8)),
                      I.param_bytes(q8_params)))
        for _g in (32, 64):
            _q4 = I.quantize_params(t360, lambda w, g=_g: ex.quantize_int4_grouped(w, g))
            _packed = I.param_bytes(_q4) - sum(_q4["layers"][k]["q"].size for k in I.QUANT_KEYS) // 2
            _rows.append((f"int4, group {_g}", _ppl(I.dequantize_params(_q4, ex.dequantize_int4_grouped)), _packed))
        _out = mo.md(
            "Perplexity on 8,192 tokens of held-out detective fiction (lower is better):\n\n"
            "| weights | perplexity | change | model size |\n|---|---|---|---|\n"
            + "\n".join(f"| {n} | {p:.2f} | {100 * (p / _base - 1):+.1f}% | {b / 1e6:.0f} MB |" for n, p, b in _rows)
            + "\n\nint8 is essentially free. int4 costs real quality on a model this small (big models "
              "tolerate it much better — they have more redundancy), and smaller groups help. The int4 sizes "
              "assume two 4-bit values packed per byte, as real formats do; this lesson stores them in int8 "
              "containers for clarity.")
    else:
        _out = mo.md("*(Implement the quantization functions to measure their effect on perplexity.)*")
    _out
    return (q8_params,)


@app.cell
def _(I, c360, code_ids, jax, jnp, mo, np, q8_params, t360, time, tok):
    def _rate(fn, *args):
        jax.block_until_ready(fn(*args))
        _t = time.perf_counter()
        _o = fn(*args)
        jax.block_until_ready(_o)
        return 64 / (time.perf_counter() - _t), _o

    if q8_params is not None:
        _p = jnp.asarray(code_ids, jnp.int32)[None]
        _r8, _o = _rate(I.decode_on_device_q, q8_params, c360, _p, 64)
        _r16, _ = _rate(lambda *a: I.decode_on_device(*a, jax.random.key(0)), t360, c360, _p, 64)
        _txt = tok.decode(np.asarray(_o[0]).tolist())[:160].replace("\n", " ⏎ ")
        _r = _r8 / _r16
        if _r < 1.05:
            _note = ("No speedup this run. Either the GPU was busy with something else during the int8 run "
                     "(re-run the cell), or XLA converted int8 → bf16 into a temporary buffer before each "
                     "matmul — reading the bytes twice and undoing the saving. Module 22 shows how to check "
                     "which, in a profile.")
        elif _r < 1.9:
            _note = ("Not the full 2× the roofline promises: some per-layer overhead doesn't shrink with the "
                     "weights, and XLA may convert int8 → bf16 into a temporary buffer before the matmul "
                     "instead of on the fly (Module 22 shows how to check).")
        else:
            _note = "If that's above 2×, the GPU was probably busy during one of the two runs — re-run the cell."
        _out = mo.md(f"""
    **int8 decoding, on-device: {_r8:.0f} tok/s** vs {_r16:.0f} in bf16 measured just now
    ({_r8 / _r16:.2f}×). `infer.forward_q` reads int8 weights and applies the per-channel scale after each
    matmul (`(x @ q) * s` — the scale factors out because it's per output column). Fewer bytes per token
    means faster decoding while memory-bound. {_note}

    First words: *{_txt}…*
    """)
    else:
        _out = mo.md("*(Implement `quantize_int8` to try int8 decoding.)*")
    _out
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Reading the weights from memory": "a",
            "Doing the matrix multiplications": "b",
            "Running the softmax over the vocabulary": "c",
        },
        label="**Check yourself:** at batch size 1, what limits how fast a model can *decode*?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right: every token needs every weight, and the GPU does only ~2 FLOPs per "
                              "weight byte read — so bandwidth, not compute, sets the speed. That's why "
                              "quantization and batching help."), kind="success"),
        "b": mo.callout(mo.md("That's the limit for *prefill* and training. During decode the math per "
                              "weight is tiny; the GPU mostly waits for memory."), kind="warn"),
        "c": mo.callout(mo.md("The softmax is a small part of the work. The weights — hundreds of MB read "
                              "per token — dominate."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(18))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Lilian Weng, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — a map of the whole field.
    * Horace He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) — the overhead / bandwidth / compute framing used in §1–§3.
    * Maarten Grootendorst, [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — int8, int4, GPTQ and GGUF, with pictures.
    * ggml-org, the [llama.cpp quantization README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) — what `Q8_0`, `Q4_K_M` and friends actually are.

    **Next:** Module 19 — local serving: run a real 7B-class coding model with llama.cpp on your GPU,
    using everything here (a KV cache, batching, 4-bit grouped weights, speculative decoding) behind an
    OpenAI-compatible endpoint.
    """)
    return


if __name__ == "__main__":
    app.run()
