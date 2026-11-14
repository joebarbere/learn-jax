import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="16 · GPT")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import json
    import math
    import time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax

    import learnjax as lj
    from learnjax.data import tiny_shakespeare

    ex = lj.load_exercises(mo.notebook_dir())
    _spec = importlib.util.spec_from_file_location("gpt_train", mo.notebook_dir() / "train.py")
    trainmod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(trainmod)  # reuse train.py's jitted scan loop and checkpoint loader
    trainmod.enable_compilation_cache()  # persistent XLA cache in the /cache volume (section 8)
    RUN_DIR = mo.notebook_dir().parent.parent / "runs" / "16-gpt"
    return RUN_DIR, ex, jax, jnp, json, lj, math, mo, np, optax, plt, tiny_shakespeare, time, trainmod


@app.cell
def _(mo):
    mo.md(r"""
    # 16 · Build and train a GPT on your GPU

    **Goal:** assemble everything from Parts 1–4 into a complete GPT — the same architecture family as
    GPT-2 — written as plain JAX functions over a parameter pytree, the style of the deepseek-v3-jax
    model you replicated. Train it on Shakespeare live in this notebook, then on 207 detective novels
    with `train.py`, and line its validation curve up against deepseek-v3-jax's.

    **You will:** count a transformer's parameters in closed form and see where they live · initialize
    it the GPT-2 way · write the block, the model, the loss, a jit-friendly batch sampler and a sampler
    for text · train a ~0.8M-parameter character model in seconds · measure tokens/second and *model
    FLOPs utilization* against the card's ~27 bf16 TFLOP/s · compare a dense GPT with the MoE model you
    trained before, at the same compute per token.

    Everything in this module is code you have already met: LayerNorm and residuals (Module 11),
    attention (Module 15), mixed precision (Module 08), AdamW + warmup-cosine + clipping (Module 05),
    the `lax.scan` training loop (Module 10), random windows with `vmap` + `dynamic_slice` (Module 14).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The architecture

    A GPT is a stack of identical **blocks** between an embedding and an output head:

    | Stage | What it computes | Shape |
    |---|---|---|
    | token embedding | `wte[tokens]` — a learned vector per vocabulary entry | (B, T) → (B, T, D) |
    | + position embedding | `wpe[:T]` — a learned vector per position 0…T−1 | (B, T, D) |
    | × n_layers blocks | `x = x + attn(LN₁(x))`, then `x = x + mlp(LN₂(x))` | (B, T, D) |
    | final LayerNorm | `LN_f(x)` | (B, T, D) |
    | output head | `x @ wteᵀ` (tied) — a score for every vocabulary entry | (B, T, V) |

    Three ideas carry the design:
    * **The residual stream.** Each block *adds* its result to `x` instead of replacing it, so information
      and gradients flow straight through 2·n_layers additions (Module 11). Blocks are small edits to a
      shared running representation.
    * **Pre-LayerNorm.** Normalize *before* attention and the MLP, never on the residual path itself.
      GPT-2 settled on this; it trains stably without careful warmup tricks.
    * **Attention mixes positions, the MLP mixes features.** Causal attention (Module 15) is the only
      place tokens exchange information; the 4×-wide GELU MLP then processes each position on its own.

    The deepseek-v3-jax model is this same skeleton with four parts swapped: RMSNorm for LayerNorm,
    RoPE for learned positions, multi-head *latent* attention for plain attention, and a mixture of
    experts for the MLP. Module 17 makes each of those swaps.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Where the parameters live

    Per block: the attention's fused Q/K/V matrix is $D\times 3D$, its output projection $D\times D$, the
    MLP's two matrices $D\times 4D$ and $4D\times D$ — that's $12D^2$ weights — plus $13D$ biases and
    LayerNorm parameters. The embeddings add $V\cdot D + T\cdot D$.

    **Your turn:** `count_params(cfg)` — the exact count, in closed form. The table below splits it up for
    three models: the character model you'll train live, the `train.py` model, and GPT-2 small.
    """)
    return


@app.cell
def _(ex):
    cfg_char = ex.GPTConfig(vocab=65, seq_len=128, d_model=128, n_layers=4, n_heads=4)
    cfg_big = ex.GPTConfig(vocab=8192, seq_len=256, d_model=384, n_layers=6, n_heads=6)
    cfg_gpt2 = ex.GPTConfig(vocab=50257, seq_len=1024, d_model=768, n_layers=12, n_heads=12)
    return cfg_big, cfg_char, cfg_gpt2


@app.cell
def _(cfg_big, cfg_char, cfg_gpt2, ex, lj, mo):
    def _parts(c):
        d = c.d_model
        return {"token embedding (V·D)": c.vocab * d, "position embedding (T·D)": c.seq_len * d,
                "blocks (L·(12D²+13D))": c.n_layers * (12 * d * d + 13 * d), "final LayerNorm (2D)": 2 * d}

    _rows = []
    for _name, _c in (("char model (live)", cfg_char), ("train.py model", cfg_big), ("GPT-2 small", cfg_gpt2)):
        _p = _parts(_c)
        _tot = sum(_p.values())
        _rows.append(f"| **{_name}** | {_tot:,} | " + " | ".join(f"{v / _tot:.0%}" for v in _p.values()) + " |")
    _yours = lj.attempt(ex.count_params, cfg_gpt2)
    mo.vstack([
        mo.md("| model | parameters | token emb. | pos. emb. | blocks | final LN |\n|---|---|---|---|---|---|\n"
              + "\n".join(_rows)),
        mo.md("**Your `count_params(GPT-2 small)`** — should print 124,439,808 (the famous \"124M\"):"),
        _yours.view(mo),
    ])
    return


@app.cell
def _(cfg_big, cfg_char, cfg_gpt2, mo):
    _rows = []
    for _name, _c in (("char model", cfg_char), ("train.py model", cfg_big), ("GPT-2 small", cfg_gpt2)):
        _head = _c.d_model * _c.vocab
        _tied = _c.vocab * _c.d_model + _c.seq_len * _c.d_model + _c.n_layers * (12 * _c.d_model**2 + 13 * _c.d_model) + 2 * _c.d_model
        _rows.append(f"| {_name} | {_tied:,} | {_tied + _head:,} | {_head:,} ({_head / (_tied + _head):.0%}) |")
    mo.md("""
    ### Weight tying

    The output head turns a $D$-vector into one score per vocabulary entry: a $D\\times V$ matrix — the
    same shape as the token embedding, transposed. **Tying** reuses `wte` for both ("a token's input
    vector is also the direction that predicts it"). It changes almost nothing for a 65-character
    vocabulary and saves a quarter of GPT-2:

    | model | tied | untied | saved by tying |
    |---|---|---|---|
    """ + "\n    ".join(_rows) + """

    Small models with big vocabularies are mostly embedding — a reason to keep the vocabulary modest
    when you can't afford many layers (Module 13's trade-off, seen from the parameter side).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Initialization

    **Your turn:** `init_gpt(key, cfg)`. Weights are normal(0, 0.02); biases zero; LayerNorm gains one.
    One GPT-2 detail matters: the two matrices that write *into* the residual stream (`w_o`, `w_proj`)
    start smaller, std $0.02/\sqrt{2L}$. Each block adds two contributions to `x`; with $2L$ independent
    additions the variances add up, so shrinking each by $\sqrt{2L}$ keeps the stream's scale roughly
    constant with depth — Module 11's activation-statistics argument, applied to residuals.
    """)
    return


@app.cell
def _(cfg_char, ex, jax, jnp, lj, mo):
    _init = lj.attempt(ex.init_gpt, jax.random.key(0), cfg_char)
    params0 = _init.value if _init.ok else None
    if _init.ok:
        _leaves = jax.tree_util.tree_flatten_with_path(params0)[0]
        _show = [(jax.tree_util.keystr(p), v) for p, v in _leaves
                 if not jax.tree_util.keystr(p).startswith("['blocks'][1]")
                 and not jax.tree_util.keystr(p).startswith("['blocks'][2]")
                 and not jax.tree_util.keystr(p).startswith("['blocks'][3]")]
        _rows = "\n".join(f"| `{k}` | {tuple(v.shape)} | {float(jnp.mean(v)):.3f} | {float(jnp.std(v)):.4f} |"
                           for k, v in _show)
        _out = mo.md("| leaf (blocks 1–3 omitted) | shape | mean | std |\n|---|---|---|---|\n" + _rows
                     + f"\n\n`w_proj`/`w_o` std ≈ 0.02/√8 = {0.02 / 8**0.5:.4f} — the scaled residual init.")
    else:
        _out = _init.view(mo)
    _out
    return (params0,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · The forward pass and the loss

    **Your turn:** `block_apply` (one pre-LN block from the provided `layer_norm`, `causal_self_attention`
    and `mlp`), `gpt_apply` (embeddings → blocks → final LayerNorm → tied head) and `loss_fn` (next-token
    cross-entropy). Two checks below catch most bugs:

    * **Causality:** change the *last* token and no earlier position's logits may move. If they do,
      the model can see the future and will "learn" by cheating.
    * **Loss at init ≈ ln V.** A fresh model knows nothing, so it should spread probability evenly over
      the vocabulary: cross-entropy $\ln 65 = 4.17$ for characters. Much higher means the init is too
      confident (Module 10 saw 3.3 vs ln 10 = 2.3 from a too-large output layer).
    """)
    return


@app.cell
def _(cfg_char, ex, jax, jnp, lj, math, mo, params0):
    if params0 is None:
        _out = mo.md("*(needs `init_gpt` first)*")
    else:
        _tok = jax.random.randint(jax.random.key(1), (2, 129), 0, 65)
        _a = lj.attempt(ex.gpt_apply, params0, _tok[:, :-1], cfg_char)
        _l = lj.attempt(ex.loss_fn, params0, _tok, cfg_char)
        _items = [mo.md("**`gpt_apply`** on a (2, 128) batch:"), _a.view(mo, show_value=False)]
        if _a.ok:
            _b = ex.gpt_apply(params0, _tok[:, :-1].at[:, -1].set((_tok[:, -2] + 7) % 65), cfg_char)
            _leak = float(jnp.abs(_a.value[:, :-1] - _b[:, :-1]).max())
            _items.append(mo.md(f"logits {tuple(_a.value.shape)} `{_a.value.dtype}` · max change at earlier "
                                f"positions after editing the last token: **{_leak:.1e}** "
                                f"({'causal ✅' if _leak < 1e-5 else 'LEAK — the model sees the future ❌'})"))
        _items += [mo.md("**`loss_fn`** at initialization:"), _l.view(mo, show_value=False)]
        if _l.ok:
            _items.append(mo.md(f"loss **{float(_l.value):.3f}** vs ln 65 = {math.log(65):.3f}"))
        _out = mo.vstack(_items)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Data: characters, and windows sampled on the GPU

    We start with **characters** as tokens — 65 of them in tiny-shakespeare — so the vocabulary is
    trivial and every result is readable. The text becomes one long array of ids; a training example
    is any window of `seq_len + 1` consecutive ids (inputs are the first `seq_len`, targets the same
    window shifted by one).

    There are no epochs: each step draws fresh random windows. **Your turn:** `get_batch(key, data, B, T)`
    with `jax.random.randint` for start positions and `jax.vmap(jax.lax.dynamic_slice)` to cut windows.
    Because it uses no Python indexing on traced values, it runs *inside* the jitted training loop, so
    the data never leaves the GPU.
    """)
    return


@app.cell
def _(jnp, np, tiny_shakespeare):
    text = tiny_shakespeare()
    chars = sorted(set(text))
    _stoi = {c: i for i, c in enumerate(chars)}

    def encode(s):
        return [_stoi[c] for c in s]

    def decode(ids):
        return "".join(chars[int(i)] for i in ids)

    _ids = np.array(encode(text), dtype=np.int32)
    _cut = int(0.9 * len(_ids))
    char_train, char_val = jnp.asarray(_ids[:_cut]), jnp.asarray(_ids[_cut:])
    return char_train, char_val, chars, decode, encode, text


@app.cell
def _(char_train, chars, decode, ex, jax, lj, mo, text):
    _b = lj.attempt(ex.get_batch, jax.random.key(0), char_train, 4, 64)
    _items = [mo.md(f"{len(text):,} characters, vocabulary of **{len(chars)}**, "
                    f"{len(char_train):,} train / the last 10% held out for validation."),
              mo.md("**Your `get_batch(key, train, 4, 64)`**, first row decoded:"), _b.view(mo, show_value=False)]
    if _b.ok:
        _items.append(mo.md(f"shape {tuple(_b.value.shape)} — row 0: `{decode(_b.value[0])!r}`"))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Train it

    The training step is deepseek-v3-jax's, scaled down:

    * **Mixed precision:** master weights stay float32; the loss casts them to bfloat16 for compute
      (LayerNorm, softmax and the loss run in float32 inside the provided pieces) — Module 08.
    * **AdamW** with linear warmup then cosine decay, and gradient clipping at global norm 1 — Module 05.
    * **One jitted `lax.scan` per chunk of steps**, sampling batches on the GPU with your `get_batch` —
      the Module 10 lesson that removed Python overhead from the loop. (`train.py`'s `make_train_chunk`
      is reused here.)

    **Your turn:** `sample(key, params, prompt, n, cfg, temperature, top_k)`. Here is the untrained model
    continuing a prompt — every character is equally likely, so it's noise:
    """)
    return


@app.cell
def _(cfg_char, decode, encode, ex, jax, lj, mo, params0):
    if params0 is None:
        _out = mo.md("*(needs `init_gpt` first)*")
    else:
        _s = lj.attempt(ex.sample, jax.random.key(0), params0, encode("ROMEO:\n"), 120, cfg_char, 1.0, None)
        _out = mo.md(f"```\n{decode(_s.value)}\n```") if _s.ok else _s.view(mo)
    _out
    return


@app.cell
def _(cfg_char, char_train, char_val, ex, jax, jnp, lj, mo, np, optax, params0, time, trainmod):
    def _run():
        steps, chunk, batch, peak = 2000, 100, 64, 2e-3
        sched = optax.warmup_cosine_decay_schedule(0.0, peak, 100, steps, peak * 0.1)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=0.1))
        train_chunk = trainmod.make_train_chunk(ex, cfg_char, opt, batch, chunk)
        evaluate = trainmod.make_eval(ex, cfg_char)
        val_batches = [ex.get_batch(jax.random.key(100 + i), char_val, batch, cfg_char.seq_len) for i in range(8)]
        params = jax.tree.map(jnp.copy, params0)  # the chunk *donates* its inputs (section 8)
        opt_state, key = opt.init(params), jax.random.key(42)
        log, chunk_times, t_start = [], [], time.perf_counter()
        for step in range(chunk, steps + 1, chunk):
            key, sub = jax.random.split(key)
            t0 = time.perf_counter()
            params, opt_state, losses = train_chunk(params, opt_state, char_train, sub)
            losses = np.asarray(losses)
            chunk_times.append(time.perf_counter() - t0)
            val = float(np.mean([evaluate(params, b) for b in val_batches]))
            log.append((step, float(losses.mean()), val))
        steady = np.median(chunk_times[1:])
        return {"params": params, "log": log, "tok_s": chunk * batch * cfg_char.seq_len / steady,
                "first_chunk_s": chunk_times[0], "total_s": time.perf_counter() - t_start,
                "tokens": steps * batch * cfg_char.seq_len}

    _ready = params0 is not None and all(_a.ok for _a in (
        lj.attempt(ex.loss_fn, params0, jnp.zeros((2, cfg_char.seq_len + 1), jnp.int32), cfg_char),
        lj.attempt(ex.get_batch, jax.random.key(0), char_train, 2, cfg_char.seq_len)))
    char_run = _run() if _ready else None
    _out = (mo.md(f"Trained {char_run['tokens'] / 1e6:.1f}M tokens in **{char_run['total_s']:.1f} s** "
                  f"(first chunk incl. compilation {char_run['first_chunk_s']:.1f} s) — "
                  f"**{char_run['tok_s']:,.0f} tokens/s** once compiled.")
            if char_run else mo.callout(mo.md("**Your turn:** finish `init_gpt`, `block_apply`, `gpt_apply`, "
                                              "`loss_fn` and `get_batch` and this cell trains your GPT."), kind="info"))
    _out
    return (char_run,)


@app.cell
def _(char_run, math, plt):
    if char_run:
        _s, _tr, _va = zip(*char_run["log"])
        _fig, _ax = plt.subplots(figsize=(7, 3.2))
        _ax.plot(_s, _tr, label="train (mean of chunk)")
        _ax.plot(_s, _va, "o-", ms=3, label="validation")
        _ax.axhline(math.log(65), color="gray", ls=":", lw=1)
        _ax.text(_s[-1], math.log(65) - 0.12, "ln 65: uniform guessing", ha="right", fontsize=8, color="gray")
        _ax.set(xlabel="step", ylabel="cross-entropy (nats/char)", title="character GPT on tiny-shakespeare")
        _ax.legend()
        _fig.tight_layout()
        _plot = _fig
    else:
        _plot = None
    _plot
    return


@app.cell
def _(cfg_char, char_run, decode, encode, ex, jax, lj, math, mo):
    if char_run:
        _s = lj.attempt(ex.sample, jax.random.key(1), char_run["params"], encode("ROMEO:\n"), 300, cfg_char, 0.8, None)
        _g = lj.attempt(ex.sample, jax.random.key(1), char_run["params"], encode("ROMEO:\n"), 120, cfg_char, 0, None)
        _final = char_run["log"][-1][2]
        _out = mo.vstack([
            mo.md(f"Validation loss **{_final:.2f}** nats/char (perplexity {math.exp(_final):.1f}: the model "
                  f"is choosing among ~{math.exp(_final):.0f} plausible next characters instead of 65)."),
            mo.md("**Sample, temperature 0.8:**"),
            mo.md(f"```\n{decode(_s.value)}\n```") if _s.ok else _s.view(mo),
            mo.md("**Greedy (temperature 0)** — always the single most likely character. It reads smoother but "
                  "tends toward bland, repetitive phrasing (and can get stuck in loops); sampling trades a little "
                  "coherence for variety:"),
            mo.md(f"```\n{decode(_g.value)}\n```") if _g.ok else _g.view(mo),
        ])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · How well are we using the GPU?

    A rule of thumb from the scaling-law papers: training costs about **6 FLOPs per parameter per token**
    — 2 for the forward pass (one multiply and one add per weight), 4 for the backward pass (gradients
    with respect to both activations and weights). Attention adds a term that grows with context length,
    $12\cdot L\cdot T\cdot D$ per token (PaLM, appendix B). Here "parameters" means the ones used in
    matrix multiplies — the blocks plus the head; looking up an embedding row costs nothing.

    $$\text{FLOPs/token} \approx 6\,(12 L D^2 + D V) + 12\,L\,T\,D \qquad
      \text{MFU} = \frac{\text{FLOPs/token}\times\text{tokens/s}}{\text{peak FLOP/s}}$$

    **Model FLOPs utilization (MFU)** compares that useful work with what the card can do. For peak we
    use the ~27 bf16 TFLOP/s you *measured* for big matmuls in Module 00, not the spec-sheet number.

    The table below adds a wider model (and `train.py`'s, once you've run it). Watch MFU climb as
    matrices grow: Module 10's
    MLP was **overhead-bound** (the GPU idled between tiny kernels — fixed by `lax.scan`); a transformer
    becomes **compute-bound** as width grows, and the remaining gap to peak is attention's softmax,
    LayerNorm and other memory-bound operations that move data instead of multiplying it.
    """)
    return


@app.cell
def _(big_tok_s, cfg_char, char_run, ex, jax, mo, np, optax, time, trainmod):
    PEAK = 27e12

    def flops_per_token(c):
        return 6 * (12 * c.n_layers * c.d_model**2 + c.d_model * c.vocab) + 12 * c.n_layers * c.seq_len * c.d_model

    def _measure(c, batch=64, steps=20):
        opt = optax.adamw(1e-4)
        p = ex.init_gpt(jax.random.key(0), c)
        st = opt.init(p)
        fn = trainmod.make_train_chunk(ex, c, opt, batch, steps)
        data = jax.random.randint(jax.random.key(1), (200_000,), 0, c.vocab)
        p, st, _l = fn(p, st, data, jax.random.key(2))  # compile
        _l.block_until_ready()
        t0 = time.perf_counter()
        p, st, _l = fn(p, st, data, jax.random.key(3))
        np.asarray(_l)
        return steps * batch * c.seq_len / (time.perf_counter() - t0)

    if char_run:
        _rows = []
        _wide = ex.GPTConfig(vocab=65, seq_len=128, d_model=384, n_layers=4, n_heads=4)
        _big = ex.GPTConfig(vocab=8192, seq_len=256, d_model=384, n_layers=6, n_heads=6)
        _cases = [("char model, D=128 (section 6)", cfg_char, char_run["tok_s"]),
                  ("char model, D=384 (measured now)", _wide, _measure(_wide))]
        if big_tok_s:
            _cases.append(("train.py model, D=384, 6 layers, T=256 (batch 32)", _big, big_tok_s))
        for _name, _c, _t in _cases:
            _f = flops_per_token(_c)
            _rows.append(f"| {_name} | {ex.count_params(_c) / 1e6:.2f}M | {_t:,.0f} | {_f * _t / 1e12:.1f} | "
                         f"**{_f * _t / PEAK:.0%}** |")
        if not big_tok_s:
            _rows.append("| train.py model (run it — section 9) | 13.89M | — | — | — |")
        _out = mo.md("| model (batch 64 × 128 unless noted) | params | tokens/s | achieved TFLOP/s | MFU |\n"
                     "|---|---|---|---|---|\n" + "\n".join(_rows))
    else:
        _out = mo.md("*(runs after the training cell above)*")
    _out
    return PEAK, flops_per_token


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · A production-shaped training loop

    Four habits from the official [JAX Training Cookbook](https://docs.jax.dev/en/latest/the-training-cookbook.html),
    all used in `train.py` and measured here on the character model:

    1. **Don't read this step's metrics in this step.** JAX dispatches work *asynchronously*: calling the
       jitted step returns immediately with arrays the GPU is still computing. `float(loss)` or `print`
       forces Python to wait for the GPU, and while Python then prepares the next step the GPU sits idle
       — a **step bubble**. Instead, dispatch step *n + 1* first and only then read step *n*'s loss.
    2. **Compute host-side scalars on the host.** Calling a `jnp`/optax function outside `jit` to log
       the learning rate launches a tiny GPU computation and waits for it, every step. Plain Python math
       (or letting the jitted step compute it) avoids the round trip.
    3. **Donate buffers you won't reuse.** `jax.jit(step, donate_argnums=(0, 1))` hands the old params and
       optimizer state to XLA, which can write the new ones *into the same memory*. Without it, both
       copies coexist at the end of every step — for a model whose params + Adam state fill half the
       card, that's the difference between fitting and not.
    4. **Cache compilations on disk.** `jax_compilation_cache_dir` stores compiled programs (here in the
       `/cache` volume), so a re-run of the same program skips XLA compilation.
    """)
    return


@app.cell
def _(cfg_char, char_run, char_train, ex, jax, jnp, mo, np, optax, params0, time, trainmod):
    if char_run:
        _opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3, b1=0.9, b2=0.95, weight_decay=0.1))

        def _step(params, opt_state, key):
            tokens = ex.get_batch(key, char_train, 64, cfg_char.seq_len)
            loss, grads = jax.value_and_grad(
                lambda p: ex.loss_fn(jax.tree.map(lambda a: a.astype(jnp.bfloat16), p), tokens, cfg_char))(params)
            updates, opt_state = _opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        _jstep = jax.jit(_step, donate_argnums=(0, 1))

        def _bench(deferred, n=150):
            p = jax.tree.map(jnp.copy, params0)
            st = _opt.init(p)
            keys = jax.random.split(jax.random.key(0), n + 5)
            for i in range(5):  # compile + warm up
                p, st, loss = _jstep(p, st, keys[i])
            float(loss)
            t0, prev, total = time.perf_counter(), None, 0.0
            for i in range(5, n + 5):
                p, st, loss = _jstep(p, st, keys[i])
                if deferred:
                    if prev is not None:
                        total += float(prev)  # the previous step's loss: already finished
                    prev = loss
                else:
                    total += float(loss)  # this step's loss: wait for the GPU right now
            float(loss)
            return n * 64 * cfg_char.seq_len / (time.perf_counter() - t0)

        _blocking, _deferred = _bench(False), _bench(True)

        # donation: the input buffers are gone after the call
        _p_in = jax.tree.map(jnp.copy, params0)
        _st_in = _opt.init(_p_in)
        _out = _jstep(_p_in, _st_in, jax.random.key(9))
        _deleted = all(x.is_deleted() for x in jax.tree.leaves(_p_in))
        jax.block_until_ready(_out)

        # learning rate for logging: optax/jnp outside jit vs plain Python
        _sched = optax.warmup_cosine_decay_schedule(0.0, 2e-3, 100, 2000, 2e-4)
        _t0 = time.perf_counter()
        for _i in range(200):
            float(_sched(_i))
        _jnp_us = (time.perf_counter() - _t0) / 200 * 1e6
        _t0 = time.perf_counter()
        for _i in range(200):
            trainmod.lr_at(_i, 2e-3, 100, 2000, 2e-4)
        _py_us = (time.perf_counter() - _t0) / 200 * 1e6
        _same = abs(float(_sched(1234)) - trainmod.lr_at(1234, 2e-3, 100, 2000, 2e-4)) < 1e-9

        _res = mo.md(f"""
        | one jitted step per Python iteration, 150 steps | tokens/s |
        |---|---|
        | `float(loss)` of the current step (bubble) | {_blocking:,.0f} |
        | read the *previous* step's loss (deferred) | **{_deferred:,.0f}** ({_deferred / _blocking - 1:+.0%}) |
        | for reference: 100 steps per `lax.scan` chunk (section 6) | {char_run['tok_s']:,.0f} |

        * **Donation:** after `step(params, opt_state, …)` the input parameter buffers report
          `is_deleted() == {_deleted}` — XLA reused their memory for the outputs. Using a donated array
          afterwards is an error, which is why the training cell passes a *copy* of `params0`.
        * **Learning rate for the log:** optax's schedule called eagerly costs **{_jnp_us:.0f} µs** per step
          (a GPU round trip); the same formula in plain Python costs **{_py_us:.1f} µs** — and agrees
          ({'✅' if _same else '❌'}). Inside the jitted step, the optimizer's own schedule is free.
        """)
    else:
        _res = mo.md("*(runs after the training cell)*")
    _res
    return


@app.cell
def _(char_run, ex, jax, jnp, mo, time):
    if char_run:
        _small = ex.GPTConfig(vocab=65, seq_len=64, d_model=64, n_layers=2, n_heads=2)  # cheap to compile
        _p_small = ex.init_gpt(jax.random.key(0), _small)

        def _compile_seconds():
            # a fresh jit object each time, so JAX's in-memory cache can't help — only the disk cache can
            _fn = jax.jit(lambda p, t: jax.grad(lambda q: ex.loss_fn(q, t, _small))(p))
            _tok = jnp.zeros((32, _small.seq_len + 1), jnp.int32)
            _t0 = time.perf_counter()
            _fn.lower(_p_small, _tok).compile()
            return time.perf_counter() - _t0

        jax.config.update("jax_enable_compilation_cache", False)
        _cold = _compile_seconds()
        jax.config.update("jax_enable_compilation_cache", True)
        _compile_seconds()  # writes the entry (or finds one from an earlier run of this notebook)
        _warm = _compile_seconds()
        _cc = mo.md(f"""
        **Compilation cache**, compiling loss + gradient of a small 2-layer GPT: **{_cold:.1f} s** from
        scratch, **{_warm:.2f} s** when the compiled program is found in `/cache/jaxcache`. Most of a cold
        compile on ROCm is XLA *autotuning* — timing candidate matrix-multiply kernels to pick the fastest.
        That's why the training cell's first chunk is slow the first time you run this notebook and fast
        afterwards, and why `train.py`'s second launch starts training in seconds instead of a minute.
        """)
    else:
        _cc = mo.md("")
    _cc
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · The long run: 207 detective novels

    `train.py` trains a bigger GPT on the **same data, tokenizer, batch (32 × 256 tokens), step count
    (12,000), optimizer and learning-rate schedule** as deepseek-v3-jax, so the two validation curves
    line up step for step. It uses *your* `exercises.py`:

    ```bash
    containers/run.sh python modules/16-gpt/train.py              # ~15 min on the RX 7600 XT
    containers/run.sh python modules/16-gpt/train.py --help       # sizes, steps, output directory
    ```

    The comparison is unusually fair. deepseek-v3-jax has 36.2M parameters but, as a mixture of experts,
    only **13.6M are active per token**; this dense GPT has ~13.9M, all active. Roughly the same compute
    per token — so the gap is mostly what MoE's extra *capacity* (plus MLA, RoPE and multi-token
    prediction) buys. Watch where the gap is large and where it closes.
    """)
    return


@app.cell
def _(RUN_DIR, json, mo):
    _metrics = RUN_DIR / "metrics.jsonl"
    big_log = [json.loads(l) for l in _metrics.read_text().splitlines()] if _metrics.exists() else []
    big_done = bool(big_log) and (RUN_DIR / "params.npz").exists()
    _tps = sorted(m["tok_per_s"] for m in big_log[1:] if "tok_per_s" in m)  # skip the window with compilation
    big_tok_s = _tps[len(_tps) // 2] if _tps else None
    _out = (mo.md(f"Found `runs/16-gpt/` — {big_log[-1]['step']:,} steps logged.") if big_log else
            mo.callout(mo.md("No long run yet. Run `containers/run.sh python modules/16-gpt/train.py` "
                             "(~15 min), then re-run this notebook to see the comparison and samples."), kind="info"))
    _out
    return big_done, big_log, big_tok_s


@app.cell
def _(big_log, big_tok_s, mo, plt):
    DEEPSEEK_VAL = {250: 5.018, 500: 4.519, 1000: 4.193, 2000: 3.964, 3000: 3.854, 4000: 3.765, 5000: 3.709,
                    6000: 3.660, 7000: 3.614, 8000: 3.584, 9000: 3.558, 10000: 3.533, 11000: 3.518, 12000: 3.507}
    if big_log:
        _vals = {m["step"]: m["val"] for m in big_log if "val" in m}
        _fig, _ax = plt.subplots(figsize=(7, 3.4))
        _ax.plot(list(_vals), list(_vals.values()), label="your dense GPT (13.9M, all active)")
        _ax.plot(list(DEEPSEEK_VAL), list(DEEPSEEK_VAL.values()), "o--", ms=3,
                 label="deepseek-v3-jax (36.2M MoE, 13.6M active)")
        _ax.set(xlabel="step (32 × 256 tokens each)", ylabel="validation loss", ylim=(3.3, 5.6),
                title="same data, tokens and optimizer")
        _ax.legend()
        _fig.tight_layout()
        _common = [s for s in DEEPSEEK_VAL if s in _vals]
        _rows = "\n".join(f"| {s:,} | {_vals[s]:.3f} | {DEEPSEEK_VAL[s]:.3f} | {_vals[s] - DEEPSEEK_VAL[s]:+.3f} |"
                          for s in _common if s in (250, 1000, 2000, 4000, 8000, 12000) or s == _common[-1])
        _out = mo.vstack([_fig, mo.md(
            "| step | your GPT | deepseek-v3-jax | difference |\n|---|---|---|---|\n" + _rows
            + f"\n\nMedian throughput **{big_tok_s or 0:,.0f} tokens/s** (deepseek-v3-jax: ~35k on this card — its "
              "simplified MoE runs all 16 experts on every token and discards 12, so its expert layers do ~3× the "
              "arithmetic they need). Fine print: deepseek-v3-jax also trains a multi-token-prediction head, and its "
              "8 validation batches are different random windows of the same held-out text, so read the gap to "
              "±0.02 or so.")])
    else:
        _out = mo.md("")
    _out
    return (DEEPSEEK_VAL,)


@app.cell
def _(DEEPSEEK_VAL, PEAK, RUN_DIR, big_done, big_log, big_tok_s, ex, flops_per_token, jax, lj, mo, trainmod):
    if big_done:
        from tokenizers import Tokenizer as _Tokenizer

        from learnjax.data import detective_corpus as _detective_corpus

        _load = lj.attempt(trainmod.load_checkpoint, ex, RUN_DIR)
        if _load.ok:
            _params, _cfg, _meta = _load.value
            _tok = _Tokenizer.from_file(str(_detective_corpus()["tokenizer_path"]))
            _samples = []
            for _i, _prompt in enumerate(("The fog came in off the", "I first heard")):
                _s = lj.attempt(ex.sample, jax.random.key(_i), _params, _tok.encode(_prompt).ids, 80, _cfg, 0.8, 50)
                _samples.append(mo.md(f"> {_tok.decode(_s.value.tolist())}") if _s.ok else _s.view(mo))
            _tps = big_tok_s or 0
            _final = [m for m in big_log if "val" in m][-1]
            _out = mo.vstack([
                mo.md(f"### Samples from your checkpoint (step {_meta['step']:,}, temperature 0.8, top-k 50)"),
                *_samples,
                mo.md(f"Final validation loss **{_final['val']:.3f}** vs deepseek-v3-jax's "
                      f"{DEEPSEEK_VAL[12000]:.3f} at step 12,000. MFU of the long run: "
                      f"**{flops_per_token(_cfg) * _tps / PEAK:.0%}** at {_tps:,.0f} tokens/s."),
            ])
        else:
            _out = mo.callout(mo.md("Finish the exercises to load and sample from the checkpoint "
                                    "(it is rebuilt with your `init_gpt` and run with your `sample`)."), kind="info")
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Causal masking: positions after the last real token can't influence its logits": "a",
            "Zero is a special padding token the model learns to ignore": "b",
            "The position embedding cancels the padding out": "c",
        },
        label="**Check yourself:** `sample` pads the context on the right with zeros. Why is that safe?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right. Position *t* attends only to positions ≤ *t*, so anything after the last "
                              "real token is invisible to it. A fixed shape also means `jit` compiles once."),
                        kind="success"),
        "b": mo.callout(mo.md("No — id 0 is an ordinary character here (`'\\n'`), and the model never trained on "
                              "padding. It's safe for a different reason: look at the attention mask."), kind="warn"),
        "c": mo.callout(mo.md("No — position embeddings are just added to each token; they can't cancel anything. "
                              "Think about which positions the last real token can attend to."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(16))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
      and [nanoGPT](https://github.com/karpathy/nanoGPT) — the same model in PyTorch, line by line.
    * Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
      (GPT-2) — pre-LN, the scaled residual init, weight tying.
    * Chowdhery et al., [PaLM](https://arxiv.org/abs/2204.02311), appendix B — how MFU is computed.
    * Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — the 6N
      rule and why non-embedding parameters are the ones that count.

    **Next:** Module 17 — modern LLM architecture: RoPE, RMSNorm, SwiGLU, grouped-query attention, MLA and
    mixture-of-experts, swapped into this GPT one at a time until it becomes Llama 3 and DeepSeek-V3.
    """)
    return


if __name__ == "__main__":
    app.run()
