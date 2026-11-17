import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="20 · Fine-tuning and LoRA")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import json
    import sys
    import time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax

    import learnjax as lj
    from learnjax import llama
    from learnjax import rosetta

    ex = lj.load_exercises(mo.notebook_dir())

    def _load(name):
        _spec = importlib.util.spec_from_file_location(f"m20_{name}", mo.notebook_dir() / f"{name}.py")
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)
        return _mod

    task = _load("task")
    LM = _load("lora_model")
    RUNS = Path("/course/runs/20-lora")
    return LM, RUNS, ex, jax, jnp, json, lj, llama, mo, np, optax, plt, rosetta, task, time


@app.cell
def _(mo):
    mo.md(r"""
    # 20 · Fine-tuning and LoRA

    **Goal:** teach a pretrained 360M-parameter model a new skill by training **0.2%** of its parameters,
    and understand why that works.

    **You will:** do the memory arithmetic that rules out full fine-tuning for big models · implement
    LoRA (`W + (α/r)·B·A`) and plug it into a real SmolLM2 · fine-tune it on a structured-output task
    and watch exact-match go from 0% to ~100% in under a minute · freeze parameters two ways (gradients
    w.r.t. adapters only, and the JAX Training Cookbook's per-path optimizer masks) · merge the adapter
    into the weights for zero-cost inference · save an adapter that's 0.5% the size of the model.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    In the BERT and GPT era (2018–19), adapting a model meant fine-tuning every weight. Adapters (Houlsby et
    al., 2019) added small trainable layers instead, and Aghajanyan et al. (2020) showed fine-tuning lives
    in a small subspace. LoRA (Hu et al., 2021) learned a low-rank update to frozen weights, and QLoRA
    (Dettmers et al., 2023) combined it with 4-bit base models to fine-tune very large models on one GPU.
    InstructGPT (Ouyang et al., 2022) showed what fine-tuning on instructions could do.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    _rows = []
    for _name, _n in [("SmolLM2-135M", 135e6), ("SmolLM2-360M (this lesson)", 362e6), ("SmolLM2-1.7B", 1.7e9),
                      ("Llama-3-8B", 8.0e9)]:
        _infer = _n * 2 / 2**30
        _full = _n * 16 / 2**30
        _lora = (_n * 2 + 0.005 * _n * 16) / 2**30
        _rows.append(f"| {_name} | {_infer:.1f} GB | {_full:.1f} GB {'✅' if _full < 12 else '❌'} | "
                     f"{_lora:.1f} GB {'✅' if _lora < 12 else '❌'} |")
    _table = "\n".join(_rows)
    mo.md(f"""
## 1 · Why not just fine-tune everything?

Module 00's arithmetic: training with Adam keeps **4 numbers per parameter** — the weight, its
gradient, and Adam's two moments. In float32 that's **16 bytes per parameter**, before activations.

| model | just run it (bf16) | full fine-tune (16 B/param) | LoRA (bf16 base, frozen + 0.5% adapters) |
|---|---|---|---|
{_table}

(✅ = fits in the ~12 GB JAX may use on your 16 GB card, before activations.) Full fine-tuning the
model in this lesson *would* fit — but at 8B parameters it needs ~120 GB, while LoRA needs ~16 GB:
the frozen weights cost 2 bytes each and only the tiny adapters carry gradients and optimizer state.
That's how people fine-tune 7–8B models on one consumer GPU (add 4-bit base weights — QLoRA — and
it fits with room to spare).
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · The low-rank hypothesis

    In Module 02 you squeezed a photo into a rank-20 approximation with 7% of the numbers and it still
    looked like the photo. LoRA bets the same thing about **the change** fine-tuning makes to a weight
    matrix: not the weights themselves (they're full-rank), but the *update* $\Delta W$.

    Write the fine-tuned weight as $W' = W + \Delta W$ and force $\Delta W$ to have rank $r$:

    $$
    \Delta W = \frac{\alpha}{r}\,A B, \qquad A \in \mathbb{R}^{d_\text{in}\times r},\;
    B \in \mathbb{R}^{r \times d_\text{out}},\; r \ll d
    $$

    For SmolLM2's query projection ($960 \times 960$): a full $\Delta W$ is **921,600** numbers;
    rank 8 is $8\cdot(960+960) =$ **15,360** — 60× fewer. The *intrinsic dimensionality* paper (below)
    measured that fine-tuning really does live in a subspace that small: a few hundred to a few thousand
    directions are enough for many tasks.

    Three design details make it work in practice:
    * **B starts at zero**, so $W' = W$ at step 0 — training starts from exactly the pretrained model.
      A is random (so gradients for B aren't zero).
    * **Never form $AB$ during training.** Compute $(xA)B$: two skinny matmuls through the rank-$r$
      bottleneck instead of a $d \times d$ product.
    * **$\alpha/r$ scaling** keeps the update's size comparable across ranks, so you don't re-tune the
      learning rate when you change $r$ (this lesson uses $\alpha = 2r$).
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _k = jax.random.split(jax.random.key(0), 3)
    _x = jax.random.normal(_k[0], (4, 960))
    _W = jax.random.normal(_k[1], (960, 960)) / 30
    _init = lj.attempt(ex.lora_init, _k[2], 960, 960, 8)
    _items = [mo.md("## 3 · Your turn: LoRA in two functions\n\n**`lora_init(key, 960, 960, r=8)`** — "
                    "A random, B zero:"), _init.view(mo, show_value=False)]
    if _init.ok:
        _A, _B = _init.value
        _items.append(mo.md(f"A {tuple(_A.shape)} std {float(_A.std()):.4f} (target {1 / 960 ** 0.5:.4f}), "
                            f"B {tuple(_B.shape)} max |B| = {float(jnp.abs(_B).max())}"))
        _app = lj.attempt(ex.lora_apply, _x, _W, _A, _B, 16.0, 8)
        _items += [mo.md("**`lora_apply`** with the fresh adapter must equal the base layer exactly:"),
                   _app.view(mo, show_value=False)]
        if _app.ok:
            _items.append(mo.md(f"max |lora_apply − x@W| = **{float(jnp.abs(_app.value - _x @ _W).max())}** "
                                "— the adapter starts as a no-op."))
    mo.vstack(_items)
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "LoRA forward", solves="lora_apply")
    return


@app.cell
def _(ex, jax, jnp, lj):
    # Are the two exercises the lesson trains with working? Everything below waits on this.
    _A = lj.attempt(ex.lora_init, jax.random.key(0), 8, 4, 2)
    _ok = _A.ok and lj.attempt(ex.lora_apply, jnp.ones((1, 8)), jnp.ones((8, 4)), *_A.value, 4.0, 2).ok
    ready = bool(_ok)
    return (ready,)


@app.cell
def _(llama, time):
    _t = time.perf_counter()
    params, cfg, tok = llama.load(llama.SMOLLM_360M)
    load_s = time.perf_counter() - _t
    return cfg, load_s, params, tok


@app.cell
def _(cfg, llama, load_s, mo, params):
    mo.md(f"""
    ## 4 · The model and the task

    **SmolLM2-360M-Instruct** ({llama.count_params(params) / 1e6:.0f}M parameters, Apache-2.0), loaded into
    `learnjax.llama` — the same RMSNorm/RoPE/GQA/SwiGLU parts you built in Module 17 — in {load_s:.1f} s:
    {cfg.n_layers} layers, d = {cfg.d_model}, {cfg.n_heads} query heads sharing {cfg.n_kv_heads} K/V heads.

    The task: read a messy shipping order and answer with **one exact line of JSON**. It's synthetic
    (generated by `task.py`, so it's unlimited, license-free, and exactly gradable) and it's the kind of
    thing fine-tuning is genuinely used for — making a model follow *your* format every time.
    """)
    return


@app.cell
def _(mo, task, tok):
    splits = task.make_splits()
    enc = {_k: task.encode_batch(tok, _v) for _k, _v in splits.items()}
    _rows = "\n".join(f"| {_s} | {splits[_s][_i]['text']} | `{splits[_s][_i]['answer']}` |"
                      for _s in ("train", "test", "heldout") for _i in (0, 1))
    mo.md(f"""
| split | input | target |
|---|---|---|
{_rows}

**{len(splits['train'])} train**, **{len(splits['test'])} test** (same 6 sentence templates, new
orders), **{len(splits['heldout'])} heldout** (2 templates the model *never* trains on — does it learn the
task, or memorize sentence shapes?). Each answer is ~{enc['train']['loss_mask'].sum(1).mean():.0f} tokens.

**Grading without generating.** Generating 200 answers token by token is slow. Instead, run each
example once with the correct answer filled in (*teacher forcing*) and check the model's top
prediction at every answer position. If every one of them is the right token, greedy decoding would
reproduce the answer exactly — so this *is* exact match, from one batched forward pass.
""")
    return enc, splits


@app.cell
def _(LM, cfg, enc, ex, mo, params, task, time):
    alpha, rank = 16.0, 8
    exact_match = LM.make_eval(cfg, ex.lora_apply, alpha, rank)
    _t = time.perf_counter()
    base_em = {_s: float(exact_match(params, {}, enc[_s], task).mean()) for _s in ("test", "heldout")}
    mo.md(f"""
## 5 · Before: the base model

Exact match: **test {base_em['test']:.0%}**, **heldout {base_em['heldout']:.0%}** ({time.perf_counter() - _t:.1f} s
for 400 examples, including compilation).
""")
    return alpha, base_em, exact_match, rank


@app.cell
def _(LM, cfg, mo, params, splits, task, tok):
    _rows = []
    for _e in splits["test"][:2]:
        _out = LM.generate_text(params, cfg, tok, _e["text"], task, n=50)
        _rows.append(f"**input:** {_e['text']}\n\n**base model:**\n```\n{_out}\n```\n**wanted:** `{_e['answer']}`\n")
    mo.md("What it actually writes (greedy):\n\n" + "\n".join(_rows) + """
It understands the request — but pretty-prints, invents its own field values ("gray blanket" for an
item), or wanders off into writing code. Close isn't exact. Prompt engineering could get partway;
fine-tuning gets it every time.
""")
    return


@app.cell
def _(LM, alpha, cfg, ex, jax, lj, llama, mo, params, rank, ready):
    TARGETS = ("wq", "wv")
    _n_base = llama.count_params(params)
    _count = lj.attempt(ex.lora_param_count, cfg, rank, TARGETS)
    adapters0 = (LM.init_adapters(jax.random.key(0), cfg, rank, TARGETS, ex.lora_init, ex.TARGET_SHAPES)
                 if ready else None)
    _items = [mo.md(f"""
## 6 · Fine-tune: rank {rank} on the query and value projections

The original LoRA paper's default: adapt `wq` and `wv` in every layer, nothing else. Trainable
parameters (**`lora_param_count`**):
"""), _count.view(mo, show_value=False)]
    if _count.ok:
        _items.append(mo.md(f"**{_count.value:,}** trainable vs **{_n_base:,}** frozen — "
                            f"**{_count.value / _n_base:.2%}** of the model. AdamW + warmup-cosine + clipping "
                            f"(Module 05), batch 16, learning rate 1e-3, 120 steps."))
    mo.vstack(_items)
    return TARGETS, adapters0


@app.cell
def _(LM, adapters0, cfg, enc, ex, exact_match, jax, mo, params, rank, ready, task, time, alpha):
    STEPS, EVAL_AT = 120, (0, 10, 20, 40, 80, 120)
    if ready:
        _opt, train_step = LM.make_trainer(cfg, ex.lora_apply, alpha, rank, lr=1e-3, steps=STEPS)
        _st = _opt.init(adapters0)
        _it = LM.batches(jax.random.key(1), enc["train"], 16)
        _ad = adapters0
        history = {"step": [], "em": [], "loss_step": [], "loss": []}
        _t0 = time.perf_counter()
        _train_s = 0.0
        for _s in range(STEPS + 1):
            if _s in EVAL_AT:
                history["step"].append(_s)
                history["em"].append(float(exact_match(params, _ad, enc["test"], task).mean()))
            if _s == STEPS:
                break
            _x, _m = next(_it)
            _ts = time.perf_counter()
            _ad, _st, _loss = train_step(_ad, _st, params, _x, _m)
            _loss.block_until_ready()
            _train_s += time.perf_counter() - _ts if _s > 0 else 0.0
            history["loss_step"].append(_s + 1)
            history["loss"].append(float(_loss))
        adapters = _ad
        timing = {"total": time.perf_counter() - _t0, "per_step": _train_s / (STEPS - 1)}
        _out = mo.md(f"Trained {STEPS} steps in **{timing['total']:.0f} s** total "
                     f"(**{timing['per_step'] * 1e3:.0f} ms/step** after compiling, evaluations included in the total).")
    else:
        train_step, adapters, history, timing = None, None, None, None
        _out = mo.callout(mo.md("Implement `lora_init` and `lora_apply` (section 3) to train."), kind="info")
    _out
    return STEPS, adapters, history, timing, train_step


@app.cell
def _(base_em, history, mo, plt):
    if history:
        _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3.2))
        _a1.semilogy(history["loss_step"], history["loss"])
        _a1.set(xlabel="step", ylabel="loss on answer tokens", title="training loss")
        _a2.plot(history["step"], [100 * e for e in history["em"]], "o-")
        _a2.axhline(100 * base_em["test"], color="gray", ls=":", label="base model")
        _a2.set(xlabel="step", ylabel="exact match, test (%)", title="exact match", ylim=(-3, 103))
        _a2.legend(loc="center right")
        _fig.tight_layout()
        _out = _fig
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(adapters, base_em, enc, exact_match, history, mo, params, task):
    if adapters is not None:
        after_em = {_s: float(exact_match(params, adapters, enc[_s], task).mean()) for _s in ("test", "heldout")}
        _curve = ", ".join(f"step {s}: {e:.0%}" for s, e in zip(history["step"], history["em"], strict=True))
        _out = mo.md(f"""
## 7 · After

| | base | after LoRA |
|---|---|---|
| test (seen templates) | {base_em['test']:.0%} | **{after_em['test']:.0%}** |
| heldout (unseen templates) | {base_em['heldout']:.0%} | **{after_em['heldout']:.0%}** |

Learning curve on test: {_curve}. The format clicks within a few dozen steps; the
heldout score says it learned *extraction*, not the six training sentence shapes.
""")
    else:
        after_em = None
        _out = mo.md("")
    _out
    return (after_em,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Two ways to freeze the base model

    **Way 1 — differentiate only what trains** (what section 6 did). The loss takes `(adapters, params)`
    and [`jax.value_and_grad`](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html#jax.value_and_grad) differentiates argument 0 only. JAX never even builds gradients for the 362M
    frozen weights; the optimizer state covers only the adapters.

    **Way 2 — one pytree, per-path optimizer rules** (the JAX *Training Cookbook*'s approach). Put
    everything in one tree `{"base": params, "lora": adapters}`, label each leaf by its *path* with
    [`jax.tree_util.tree_map_with_path`](https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.tree_map_with_path.html#jax.tree_util.tree_map_with_path), and give optax a rule per label: AdamW for `"train"`,
    `optax.set_to_zero()` for `"freeze"`. This is how you'd freeze "the embeddings and the first 10
    layers" or give the head its own learning rate — any per-parameter policy, decided by name.

    The catch, measured below: way 2 still *computes* a gradient for every frozen weight and then throws
    it away. Guess first: does that cost time, memory, or both?
    """)
    return


@app.cell
def _(LM, TARGETS, alpha, cfg, enc, ex, jax, jnp, lj, mo, optax, params, rank, ready, time, train_step):
    if ready:
        _fresh = LM.init_adapters(jax.random.key(0), cfg, rank, TARGETS, ex.lora_init, ex.TARGET_SHAPES)
        _full = {"base": params, "lora": _fresh}
        _mask = lj.attempt(ex.freeze_mask, _full, ("lora/",))
        _items = [mo.md("**`freeze_mask(tree, (\"lora/\",))`**:"), _mask.view(mo, show_value=False)]
        if _mask.ok:
            _labels = jax.tree.map(lambda m: "train" if m else "freeze", _mask.value)
            _n_train = sum(1 for _l in jax.tree.leaves(_labels) if _l == "train")
            _opt2 = optax.multi_transform({"train": optax.adamw(1e-3), "freeze": optax.set_to_zero()}, _labels)

            def _loss_full(full, tokens, mask):
                return LM.masked_loss(LM.forward(full["base"], full["lora"], tokens, cfg, ex.lora_apply, alpha, rank),
                                      tokens, mask)

            @jax.jit
            def _step2(full, st, tokens, mask):
                loss, grads = jax.value_and_grad(_loss_full)(full, tokens, mask)
                upd, st = _opt2.update(grads, st, full)
                return optax.apply_updates(full, upd), st, loss

            _x = jnp.asarray(enc["train"]["tokens"][:16])
            _m = jnp.asarray(enc["train"]["loss_mask"][:16])
            _st2 = _opt2.init(_full)
            _f, _st2, _ = _step2(_full, _st2, _x, _m)          # compile
            jax.block_until_ready(_f)
            _t = time.perf_counter()
            for _ in range(5):
                _f, _st2, _ = _step2(_f, _st2, _x, _m)
            jax.block_until_ready(_f)
            _way2 = (time.perf_counter() - _t) / 5
            _opt1, _ = LM.make_trainer(cfg, ex.lora_apply, alpha, rank)
            _st1 = _opt1.init(_fresh)
            _a, _s1, _ = train_step(_fresh, _st1, params, _x, _m)
            jax.block_until_ready(_a)
            _t = time.perf_counter()
            for _ in range(5):
                _a, _s1, _ = train_step(_a, _s1, params, _x, _m)
            jax.block_until_ready(_a)
            _way1 = (time.perf_counter() - _t) / 5
            _base_same = all(bool(jnp.array_equal(_u, _v)) for _u, _v in
                             zip(jax.tree.leaves(_f["base"]), jax.tree.leaves(params), strict=True))
            _g1 = jax.eval_shape(lambda a: jax.grad(lambda a_: _loss_full({"base": params, "lora": a_}, _x, _m))(a),
                                 _fresh)
            _g2 = jax.eval_shape(lambda f: jax.grad(_loss_full)(f, _x, _m), _full)
            _gb = lj.attempt(ex.adapter_bytes, _g1)
            _gb2 = lj.attempt(ex.adapter_bytes, _g2)
            _sb = lj.attempt(ex.adapter_bytes, _st1)
            _sb2 = lj.attempt(ex.adapter_bytes, _st2)
            _fmt = (lambda a: f"{a.value / 2**20:,.1f} MiB" if a.ok else "(implement `adapter_bytes`)")
            _items.append(mo.md(f"""
{_n_train} of {len(jax.tree.leaves(_labels))} leaves labelled `"train"` — exactly the A and B arrays.
After 6 way-2 steps the base weights are **{'bit-for-bit unchanged' if _base_same else 'CHANGED (bug!)'}**.

| | way 1: grad w.r.t. adapters | way 2: one tree + `multi_transform` |
|---|---|---|
| step time | **{_way1 * 1e3:.0f} ms** | {_way2 * 1e3:.0f} ms |
| gradient pytree (`adapter_bytes`) | **{_fmt(_gb)}** | {_fmt(_gb2)} |
| optimizer state (`adapter_bytes`) | {_fmt(_sb)} | {_fmt(_sb2)} |

Optimizer state is the same (Adam's m and v for the adapters; `set_to_zero` keeps none for the
base). Step time barely moves — the backward pass has to push gradients through every layer to reach
the adapters anyway, and the extra weight-gradient matmuls are cheap next to that. **Memory** is the
real cost: way 2 materializes a full-size gradient for the frozen model every step — about the size of
the model itself. At 8B parameters that's 16 GB you don't have. Use way 2 for per-parameter *policies*;
when the rule is simply "train the adapters", differentiate the adapters.
"""))
        _out = mo.vstack(_items)
    else:
        _out = mo.md("*(needs `lora_init` / `lora_apply`)*")
    _out
    return


@app.cell
def _(LM, adapters, alpha, cfg, enc, ex, exact_match, jnp, lj, llama, mo, params, rank, splits, task, tok):
    if adapters is not None:
        _check = lj.attempt(ex.merge_lora, jnp.ones((4, 3)), jnp.ones((4, 2)), jnp.ones((2, 3)), 4.0, 2)
        _items = [mo.md("## 9 · Merge: zero-cost inference\n\n**`merge_lora`** folds $\\frac{\\alpha}{r}AB$ "
                        "into each weight; then the *plain* `learnjax.llama` model — no adapter code at all — "
                        "runs the fine-tuned model:"), _check.view(mo, show_value=False)]
        if _check.ok:
            merged = LM.merge(params, adapters, ex.merge_lora, alpha, rank)
            _x = jnp.asarray(enc["heldout"]["tokens"][:8])
            _la = LM.forward(params, adapters, _x, cfg, ex.lora_apply, alpha, rank)
            _lm, _ = llama.forward_jit(merged, _x, cfg)
            _mask = enc["heldout"]["loss_mask"][:8] > 0
            _agree = float((jnp.argmax(_la, -1) == jnp.argmax(_lm, -1))[_mask].mean())
            _em_m = float(exact_match(merged, {}, enc["heldout"], task).mean())
            _gen = "\n".join(f"* {_e['text']}\n  → `{LM.generate_text(merged, cfg, tok, _e['text'], task)}`"
                             for _e in splits["heldout"][2:5])
            _items.append(mo.md(f"""
Adapter forward vs merged forward on 8 heldout examples: max |Δlogit| =
**{float(jnp.abs(_la - _lm).max()):.2f}** on logits as large as {float(jnp.abs(_lm).max()):.0f}, and the predicted answer tokens agree
**{_agree:.1%}** of the time. Why not exactly equal? $W$ is bfloat16 and the update $\\frac{{\\alpha}}{{r}}AB$
is tiny next to it, so rounding $W + \\Delta W$ back to bfloat16 drops part of the update (Module 08's
precision lesson). It's harmless here; for a big merged model you'd merge in float32 and keep a
float32 copy, or keep the adapter separate. Merged model exact match on heldout:
**{_em_m:.0%}**.

Generated by the merged model with `llama.generate` (unseen templates):

{_gen}
"""))
        else:
            merged = None
        _out = mo.vstack(_items)
    else:
        merged, _out = None, mo.md("")
    _out
    return (merged,)


@app.cell
def _(RUNS, adapters, enc, ex, exact_match, jnp, lj, llama, mo, np, params, task):
    if adapters is not None:
        RUNS.mkdir(parents=True, exist_ok=True)
        _path = RUNS / "adapter_qv_r8.npz"
        np.savez(_path, **{f"{_n}/{_k}": np.asarray(_v) for _n, _ab in adapters.items() for _k, _v in _ab.items()})
        _z = np.load(_path)
        _back = {}
        for _key in _z.files:
            _n, _k = _key.split("/")
            _back.setdefault(_n, {})[_k] = jnp.asarray(_z[_key])
        _em = float(exact_match(params, _back, enc["heldout"], task).mean())
        _b = lj.attempt(ex.adapter_bytes, adapters)
        _items = [mo.md(f"""
## 10 · Ship the adapter, not the model

Saved to `runs/20-lora/{_path.name}`: **{_path.stat().st_size / 2**20:.1f} MiB** on disk vs
**{llama.count_params(params) * 2 / 2**20:,.0f} MiB** for the model's bf16 weights. Reloaded, it scores **{_em:.0%}** on heldout. One base model
can serve many adapters — one per customer or task — swapped in milliseconds.

**`adapter_bytes(adapters)`**:"""), _b.view(mo, show_value=False)]
        if _b.ok:
            _items.append(mo.md(f"{_b.value:,} bytes in memory (float32 A and B)."))
        _out = mo.vstack(_items)
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(RUNS, json, mo, plt):
    _p = RUNS / "sweep.json"
    if _p.exists():
        _res = json.loads(_p.read_text())
        _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.4))
        for _t, _mk in (("q,v", "o-"), ("all", "s--")):
            _rows = [r for r in _res if r["targets"] == _t]
            _first = [next((st for st, e in zip(r["steps"], r["em_heldout"], strict=True) if e >= 0.95), None)
                      for r in _rows]
            _a1.semilogx([r["params"] for r in _rows], _first, _mk, label=f"targets: {_t}")
            for r in _rows:
                if _t == "q,v":
                    _a2.plot(r["steps"], [100 * e for e in r["em_heldout"]], "o-", label=f"r={r['r']}")
        _a1.set(xlabel="trainable parameters (log)", ylabel="steps to ≥95% heldout EM",
                title="every config reaches 100% — bigger ones get there sooner")
        _a1.legend()
        _a2.set(xlabel="step", ylabel="heldout exact match (%)", title="q,v adapters: learning speed by rank",
                ylim=(-3, 103))
        _a2.legend(fontsize=8)
        _fig.tight_layout()
        _tbl = "\n".join(f"| {r['targets']} | {r['r']} | {r['params']:,} | {r['em_test'][-1]:.0%} | "
                         f"{r['em_heldout'][-1]:.0%} | {r['seconds']} s |" for r in _res)
        _out = mo.vstack([mo.md("## 11 · How small can the adapter be? (from `train.py`)"), _fig, mo.md(f"""
| targets | rank | trainable params | test EM | heldout EM | wall time (incl. evals) |
|---|---|---|---|---|---|
{_tbl}

**Rank 1 on q and v — 102,400 parameters, 0.03% of the model — learns the task completely.** That's
the low-rank hypothesis in action: this task's update to a 960×960 matrix fits in a single direction.
Bigger adapters buy *speed* (rank 32 is there by step 10, rank 1 needs ~40) and headroom for harder
tasks (a new language, a new domain) — not accuracy on a task this narrow. (Wall times vary because
the sweep ran while other jobs shared the GPU.)
""")])
    else:
        _out = mo.callout(mo.md("""
## 11 · How small can the adapter be?

Run the rank × target sweep (~7 min), then re-run this cell:

```
containers/run.sh python modules/20-lora/train.py
```
It trains ranks 1–32 on q,v and on every projection, and writes `runs/20-lora/sweep.json`.
"""), kind="info")
    _out
    return


@app.cell
def _(LM, STEPS, TARGETS, adapters, alpha, cfg, enc, ex, exact_match, jax, jnp, mo, params, rank, task, time,
      tok, train_step):
    if adapters is None:
        merge_table = None
        _out = mo.md("")
    else:
        _t0 = time.perf_counter()
        _dates = task.make_date_splits()
        _denc = {_k: task.encode_batch(tok, _v, instruction=task.DATE_INSTRUCTION) for _k, _v in _dates.items()}
        _opt, _ = LM.make_trainer(cfg, ex.lora_apply, alpha, rank, lr=1e-3, steps=STEPS)
        _ad = LM.init_adapters(jax.random.key(2), cfg, rank, TARGETS, ex.lora_init, ex.TARGET_SHAPES)
        _st, _it = _opt.init(_ad), LM.batches(jax.random.key(3), _denc["train"], 16)
        for _s in range(STEPS):
            _x, _m = next(_it)
            _ad, _st, _ = train_step(_ad, _st, params, _x, _m)
        adapters_dates = _ad

        def _combine(la, lb):
            """la·ΔW_orders + lb·ΔW_dates as ONE adapter of rank 2r: stack the A's side by side, the B's on top."""
            return {_n: {"A": jnp.concatenate([adapters[_n]["A"], adapters_dates[_n]["A"]], axis=-1),
                         "B": jnp.concatenate([la * adapters[_n]["B"], lb * adapters_dates[_n]["B"]], axis=-2)}
                    for _n in adapters}

        _variants = [("base model", {}), ("orders adapter only", adapters), ("dates adapter only", adapters_dates),
                     ("sum: orders + dates", _combine(1.0, 1.0)), ("average: ½ orders + ½ dates", _combine(0.5, 0.5))]
        merge_table = {_name: (float(exact_match(params, _a, enc["heldout"], task).mean()),
                               float(exact_match(params, _a, _denc["test"], task).mean()))
                       for _name, _a in _variants}
        _rows = "\n".join(f"| {_k} | {_o:.0%} | {_d:.0%} |" for _k, (_o, _d) in merge_table.items())
        _ex = "\n".join(f"* `{_e['text']}` → `{_e['answer']}`" for _e in _dates["test"][:3])
        _b, _s, _a = merge_table["base model"], merge_table["sum: orders + dates"], merge_table["average: ½ orders + ½ dates"]
        _better = "beats" if sum(_a) > sum(_s) + 0.02 else "loses to" if sum(_a) < sum(_s) - 0.02 else "ties with"
        _reading = (f"Reading it: the base model already gets **{_b[1]:.0%}** of dates right, because SmolLM2-Instruct has "
                    f"seen ISO dates, while orders start at {_b[0]:.0%}. Adding both deltas at full strength keeps most of "
                    f"each skill (**{_s[0]:.0%}** / **{_s[1]:.0%}**) but not all: both adapters move the same query and "
                    f"value weights, so they **interfere**. Here the average (½ + ½) **{_better}** the sum "
                    f"(**{_a[0]:.0%}** / **{_a[1]:.0%}**): halving each delta limits how far the pair can push shared "
                    "weights. TIES-Merging (papers below) goes further: it drops each delta's small entries and resolves "
                    "sign conflicts before adding. Merging is cheap to try, and you measure it per task like this.")
        _out = mo.md(f"""
## 12 · Task arithmetic: add two adapters

A second adapter, same rank and recipe, trained for {STEPS} steps on a different job: dates to ISO format.

{_ex}

Each adapter is a weight *delta*, $\\Delta W = \\frac{{\\alpha}}{{r}}AB$. **Task arithmetic** (Ilharco et al., 2022)
treats those deltas as vectors you can add: $W + \\Delta W_\\text{{orders}} + \\Delta W_\\text{{dates}}$. Two
rank-$r$ adapters add up to one rank-$2r$ adapter, because stacking does the sum for you:

$$x\\,[A_1\\;A_2]\\begin{{bmatrix}}B_1\\\\B_2\\end{{bmatrix}} = xA_1B_1 + xA_2B_2$$

so the merged model runs through your own `lora_apply`, unchanged. Exact match on each task's test set:

| weights | orders (heldout) | dates |
|---|---|---|
{_rows}

(Dates trained in {time.perf_counter() - _t0:.0f} s, evaluations included.)

{_reading}
""")
    _out
    return (merge_table,)


@app.cell
def _(STEPS, exact_match, merge_table, mo, params, task, tok):
    if merge_table is None:
        _out = mo.md("")
    else:
        _dates = task.make_date_splits()
        _shots = _dates["train"][:16]
        _turn = "<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n<|im_start|>user\n"
        icl, _extra = {}, {}
        for _k in (0, 1, 2, 4, 8, 16):
            _inline = task.DATE_INSTRUCTION + ("Examples:\n" + "".join(f"{_e['text']} -> {_e['answer']}\n"
                                                                       for _e in _shots[:_k]) + "\n" if _k else "")
            _turns = task.DATE_INSTRUCTION + "".join(_e["text"] + _turn.format(a=_e["answer"]) + task.DATE_INSTRUCTION
                                                     for _e in _shots[:_k])
            for _fmt, _instr in (("inline", _inline), ("turns", _turns)):
                _b = task.encode_batch(tok, _dates["test"], max_len=96 + 40 * _k, instruction=_instr)
                icl[_fmt, _k] = float(exact_match(params, {}, _b, task).mean())
                _extra[_fmt, _k] = len(tok.encode(_instr).ids) - len(tok.encode(task.DATE_INSTRUCTION).ids)
        _lora = merge_table["dates adapter only"][1]
        _ks = (0, 1, 2, 4, 8, 16)
        _rows = "\n".join(f"| {_k} | {icl['inline', _k]:.0%} | {icl['turns', _k]:.0%} | {_extra['turns', _k]} |" for _k in _ks)
        _worse = [_k for _k in _ks[1:] if icl["inline", _k] < icl["inline", 0] - 0.2]
        _one = icl["turns", 1]
        _why = ((f"Look at the inline column: at k = {', '.join(map(str, _worse))} the examples made it *worse* than none. "
                 "The model reads `text -> answer` lines and starts its reply by imitating that line format instead of "
                 "answering, so exact match fails on the first token. " if _worse else "")
                + f"As chat turns, even one example gets **{_one:.0%}**: the examples look like what the model was "
                "trained on, earlier questions in this conversation answered the right way. ")
        _out = mo.md(f"""
## 13 · Or don't train at all: in-context learning

GPT-3's surprise (Brown et al., 2020) was that a big enough language model can pick up a task from a
few examples *in the prompt*, with no gradient step: **in-context learning**. Same base model, no
adapter, the date task with $k$ worked examples, given two ways: **inline** in the instruction
(`January 20th, 2002 -> 2002-01-20`, one per line), or as earlier **turns** of the chat, each example a
user message answered by the assistant.

| examples ($k$) | inline | as chat turns | extra prompt tokens (turns) |
|---|---|---|---|
{_rows}
| *LoRA adapter, {STEPS} steps (section 12)* | | *{_lora:.0%}* | *0* |

{_why}This is Module 21's lesson again (prompt format is part of the result), and Module 27's chat
templates: a chat model learns most from examples in the shape it was trained on.

The trade: in-context learning costs nothing up front but pays on **every** request (the examples are
re-read each time: {_extra["turns", 16]} extra prompt tokens at $k=16$, Module 18's prefill cost), and how
much a model learns from examples grows with its size. Fine-tuning pays once, then the prompt stays short.
Try the cheap thing first, measure, and train only when it isn't enough.
""")
    _out
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "So the adapted layer starts exactly equal to the pretrained layer": "a",
            "So the adapter has fewer parameters": "b",
            "Because zero is the best initialization for any weight": "c",
        },
        label="**Check yourself:** why does LoRA initialize B to zero (and A randomly)?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right. $AB = 0$ at step 0, so fine-tuning starts from the pretrained model's "
                              "exact behavior. A must be random, or the gradient for B (which depends on $xA$) "
                              "would be zero too and nothing would train."), kind="success"),
        "b": mo.callout(mo.md("No — the parameter count is set by the shapes ($r(d_{in}+d_{out})$), not "
                              "the values. Zero-init is about where training *starts*."), kind="warn"),
        "c": mo.callout(mo.md("No — if both A and B were zero, every gradient would be zero and nothing "
                              "would ever train (Module 11's initialization lesson). Only B is zero."),
                        kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(20))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Sebastian Raschka, [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
      — rank, alpha, targets and learning rates from many experiments.
    * Hugging Face [PEFT documentation](https://huggingface.co/docs/peft) — the library `torch_compare.py` uses.
    * JAX docs, [The Training Cookbook](https://docs.jax.dev/en/latest/the-training-cookbook.html) —
      per-parameter optimizer control with `tree_map_with_path`.
    * `torch_compare.py` (next to this notebook) — the same fine-tune with transformers + PEFT:
      `make torch-lesson N=20`.

    **Next:** Module 21 — evaluating language models: perplexity, benchmarks, and how not to fool yourself.
    """)
    return


if __name__ == "__main__":
    app.run()
