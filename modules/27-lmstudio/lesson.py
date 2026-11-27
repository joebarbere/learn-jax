import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="27 · LM Studio")


@app.cell
def _():
    import importlib.util
    import json
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt

    import learnjax as lj
    from learnjax import llama, rosetta

    HERE = mo.notebook_dir()
    ROOT = Path(HERE).parent.parent
    RUNS = ROOT / "runs" / "27-lmstudio"
    sys.path.insert(0, str(HERE))
    import lmclient
    import gguf_meta

    def load_mod(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def read_json(name):
        p = RUNS / name
        return json.loads(p.read_text()) if p.exists() else None

    ex = lj.load_exercises(HERE)
    BASE = lmclient.find_server()
    return (BASE, ROOT, RUNS, ex, gguf_meta, jax, jnp, json, lj, llama, lmclient, load_mod, mo, np, plt,
            read_json, rosetta, time)


@app.cell
def _(mo):
    mo.md(r"""
    # 27 · Your models in LM Studio: export, convert, serve

    **Goal:** take the models you *built* in this course out of JAX and into
    [LM Studio](https://lmstudio.ai) (by Element Labs) — a desktop app and headless server that runs GGUF
    models with llama.cpp — then use them the way you'd use any local model: chat, an OpenAI- or
    Anthropic-compatible API, JSON output, tool calls.

    **You will:** write the *inverse* of the Hugging Face loader you've used since Module 18 · map your
    Module 16 GPT onto GPT-2's tensor names · see why a LoRA must be merged in float32 · convert to GGUF and
    measure what each quantization costs · read a chat template out of a GGUF file · drive LM Studio's three
    APIs · run your Module 20 JSON extractor and Module 26 reasoning model through it and check they still
    score what they scored in JAX · compare the same model four ways: JAX, raw llama.cpp, LM Studio on
    Vulkan, LM Studio on ROCm · and work as a quantizer does: importance matrices, i-quants and task-scored
    quants, first on your LoRA, then on a Qwen3.5-9B you fine-tune with QLoRA.

    **Setup** (once): `make lms-build`, then `make export-models` (export → verify → GGUF → import; ~3 min),
    then `make lms-up` before opening this lesson. Every live section below says what to run if the server
    isn't up.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    The local-model ecosystem grew around llama.cpp in 2023: the ggml library, then the single-file GGUF
    model format (August 2023). Desktop apps such as LM Studio and Ollama (both 2023) wrapped those engines
    with model catalogs and OpenAI-compatible servers. The techniques you combine here come from earlier
    modules: LoRA (Hu et al., 2021) for adapting, GPTQ-style quantization (Frantar et al., 2022) for
    shrinking, and speculative decoding (Leviathan et al., 2022) for speed. QLoRA (Dettmers et al., 2023)
    made fine-tuning a 4-bit base on one GPU routine, and llama.cpp's importance matrices and i-quants (2024,
    after QuIP#, Tseng et al.) made 2–3-bit files usable.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(BASE, lmclient, mo):
    if BASE:
        _keys = sorted(lmclient.available(BASE))
        _status = mo.callout(mo.md(f"**LM Studio is up** at `{BASE}` with {len(_keys)} models: "
                                   + ", ".join(f"`{k}`" for k in _keys[:14]) + (" …" if len(_keys) > 14 else "")),
                             kind="success")
    else:
        _status = mo.callout(mo.md("**LM Studio isn't running** — the live sections will show what to run. "
                                   "Start it with `make lms-up` (and `make export-models` once), then re-run."),
                             kind="info")
    _status
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · What LM Studio is

    | | Module 19: raw llama.cpp | LM Studio |
    |---|---|---|
    | What runs the model | `llama-server` (ggml-org) | the same llama.cpp, packaged as an **engine** (LM Studio calls them *runtimes*) |
    | How you run it | one process, one model, flags | a desktop **app**, the **`lms`** CLI, and the headless daemon **`llmster`** (what `make lms-up` runs, in a container) |
    | Models | a path to a `.gguf` | a model library (`~/.lmstudio/models/publisher/repo/file.gguf`); several quants of one model share a key like `lora20-orders@q8_0` |
    | Loading | at start | load/unload on demand over HTTP or `lms load`; several models at once |
    | APIs | OpenAI-compatible | OpenAI-compatible **+** its own REST API **+** Anthropic-compatible `/v1/messages` |
    | License | MIT | free for personal and internal business use ([LM Studio app terms](https://lmstudio.ai/app-terms)); each model keeps its own license |

    There's no official container image, so `containers/Containerfile.lmstudio` installs `llmster` with LM
    Studio's own installer ([lmstudio.ai/install.sh](https://lmstudio.ai/install.sh)), vendored so the
    version is pinned (llmster 0.0.27), on Ubuntu 24.04 with Mesa's Vulkan driver. The host only publishes
    port 1234 on 127.0.0.1; lessons reach it at `http://lmstudio:1234` on the course network.
    """)
    return


@app.cell
def _(mo, plt, read_json):
    _bench = read_json("bench.json")
    if not _bench:
        _out = mo.callout(mo.md("No benchmark yet — run `make lms-bench` on an idle GPU "
                                "(writes `runs/27-lmstudio/bench.json`)."), kind="info")
    else:
        _rows = _bench["rows"]
        _fig, _ax = plt.subplots(figsize=(8.5, 4.2))
        _labels = [f"{r['engine'].split(' (')[0].replace(' 0.4.28 / llmster 0.0.27', '')}"
                   f" {r['engine'][r['engine'].find('('):].split(',')[0]}) · {r['model'].split()[-1]}"
                   + (" · no draft (§13 baseline)" if "no draft" in r["engine"] else
                      " · 135M draft" if "draft" in r["engine"] else "")
                   for r in _rows]
        _colors = ["#4c72b0" if "JAX" in r["engine"] else "#55a868" if "llama-bench" in r["engine"]
                   else "#c44e52" if "rocm" in r["engine"] else "#dd8452" for r in _rows]
        _ax.barh(range(len(_rows)), [r["gen_tok_s"] for r in _rows], color=_colors)
        _ax.set_yticks(range(len(_rows)), _labels, fontsize=7.5)
        _ax.invert_yaxis()
        _ax.set_xlabel("generation tokens / s (SmolLM2-360M-Instruct, greedy, 128 tokens)")
        _ax.set_title(f"The same model, four ways (RX 7600 XT, measured {_bench['measured']})")
        for _i, _r in enumerate(_rows):
            _ax.text(_r["gen_tok_s"] + 3, _i, f"{_r['gen_tok_s']:.0f}", va="center", fontsize=7)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Engines on this card: Vulkan and ROCm both work

    LM Studio ships llama.cpp builds per backend. On Linux with an AMD card you get two: **Vulkan** (via
    Mesa's RADV driver — no ROCm needed at all) and **ROCm/HIP**. LM Studio doesn't publish a supported-GPU
    list, and a community report described the RX 7600 XT hanging with a self-built ROCm llama.cpp, so this
    module tested both. Results on gfx1102:

    * **Both engines run natively** — no `HSA_OVERRIDE_GFX_VERSION` (only JAX's plugin needs that, Module 00).
      `lms runtime survey` lists the card under each: "AMD Radeon RX 7600 XT (RADV NAVI33) (Vulkan, Discrete)"
      and "(ROCm, Discrete)", 15.98 GiB.
    * **Vulkan generates faster here** (chart above: ~235 vs ~184 tok/s at bf16, ~320 vs ~220 at 4-bit) — the
      same pattern Module 19 measured with raw llama.cpp: decode is memory-bound and RADV's kernels do well.
    * **Pin your engine.** The image installs llama.cpp **2.73.0** for both backends and selects one at start
      (`containers/lmstudio.sh start vulkan|rocm`). Engines only change with `lms runtime update` — a
      community report describes an automatic engine update breaking GPU use, so updating should be a choice.

    **JAX is the slowest column**, and that's expected: Module 18's decode loop is a teaching implementation
    (bf16 weights re-read every token, no fused kernels). llama.cpp is a dedicated inference engine; LM Studio
    adds a few percent of serving overhead on top of the same engine (Vulkan bf16: 235 vs llama-bench's 255).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Getting a model out of JAX

    Since Module 18, `learnjax.llama.convert_hf` has turned Hugging Face `LlamaForCausalLM` tensors into our
    pytree: stack the 32 layers on a leading axis, transpose every Linear (`[out, in]` → `[in, out]` so we
    can write `x @ W`). Exporting is the inverse — un-stack, transpose back, restore the names. Once a model is
    in Hugging Face format, everything else (transformers, llama.cpp's converter, LM Studio) understands it.

    **Your turn:** `export_llama_hf(params, cfg)`. The cell below loads SmolLM2-135M in float32, exports it,
    and compares every tensor against the original download — it must be **bit-exact**.
    """)
    return


@app.cell
def _(ex, lj, llama, mo, np):
    def _round_trip():
        from safetensors.numpy import load_file
        params, cfg, _ = llama.load(llama.SMOLLM_135M, dtype="float32")
        raw = load_file(str(next(llama.download(llama.SMOLLM_135M).glob("*.safetensors"))))
        out = ex.export_llama_hf(params, cfg)
        bad = [k for k in raw if k not in out or not np.array_equal(np.asarray(out[k]), raw[k].astype(np.float32))]
        return len(raw), len(out), bad

    _a = lj.attempt(_round_trip)
    if _a.ok:
        _n, _m, _bad = _a.value
        _view = mo.callout(mo.md(f"Round trip of SmolLM2-135M: **{_n} tensors in the download, {_m} exported, "
                                 f"{len(_bad)} differ** — " + ("bit-exact. ✅" if not _bad and _n == _m else
                                 f"first mismatches: `{_bad[:3]}`")), kind="success" if not _bad else "warn")
    else:
        _view = _a.view(mo)
    _view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your Module 16 GPT, in GPT-2's clothes

    Module 16's GPT *is* GPT-2's architecture (pre-LayerNorm blocks, fused QKV, GELU with the tanh
    approximation, learned positions, tied output head), so it fits Hugging Face's `GPT2LMHeadModel` — only
    the names differ. GPT-2 uses `Conv1D` layers whose weights are stored `[in, out]` (our layout, unlike
    `nn.Linear`), and its fused `c_attn` splits q, k, v exactly as Module 16 does: **rename, don't transpose.**
    Getting this wrong wouldn't crash — it would produce a model that speaks gibberish — which is why every
    export is checked numerically (§4).

    **Your turn:** `gpt_to_gpt2_state_dict(params, n_layers)`.
    """)
    return


@app.cell
def _(ROOT, ex, lj, load_mod, mo):
    _run16 = ROOT / "runs" / "16-gpt"
    if not (_run16 / "params.npz").exists():
        _view = mo.callout(mo.md("Module 16's checkpoint isn't there yet — run "
                                 "`containers/run.sh python modules/16-gpt/train.py` first."), kind="info")
    else:
        _t16 = load_mod(ROOT / "modules" / "16-gpt" / "train.py", "m27_lesson_train16")
        _sol16 = lj.load_exercises(ROOT / "modules" / "16-gpt", solutions=True)
        _p16, _c16, _ = _t16.load_checkpoint(_sol16, _run16)
        _a = lj.attempt(ex.gpt_to_gpt2_state_dict, _p16, _c16.n_layers)
        if _a.ok:
            _sd = _a.value
            _lines = "\n".join(f"| `{k}` | `{tuple(v.shape)}` |" for k, v in list(_sd.items())[:4] +
                               [(k, v) for k, v in _sd.items() if k.startswith("transformer.h.0.")])
            _view = mo.md(f"**{len(_sd)} tensors** for {_c16.n_layers} blocks (no `lm_head` — it's tied to `wte`). "
                          f"The globals and block 0:\n\n| GPT-2 name | shape |\n|---|---|\n{_lines}")
        else:
            _view = _a.view(mo)
    _view
    return


@app.cell
def _(ROOT, jnp, llama, mo, np):
    def _merge_precision():
        a = ROOT / "runs" / "20-lora" / "adapter_qv_r8.npz"
        if not a.exists():
            return None
        z = np.load(a)
        params, _cfg, _ = llama.load(llama.SMOLLM_360M, dtype="float32")
        W = params["layers"]["wq"][0]
        delta = (16.0 / 8) * (jnp.asarray(z["wq/A"][0]) @ jnp.asarray(z["wq/B"][0]))
        target = W + delta
        rows = []
        for name, dt in (("float32", jnp.float32), ("float16", jnp.float16), ("bfloat16", jnp.bfloat16)):
            merged = (W + delta).astype(dt).astype(jnp.float32)
            kept = jnp.sum((merged - W) * delta) / jnp.sum(delta * delta)
            err = jnp.linalg.norm(merged - target) / jnp.linalg.norm(delta)
            rows.append(f"| {name} | {float(kept):.3f} | {float(err):.3f} |")
        return (f"| merged weights stored as | share of the LoRA update kept | error ÷ update size |\n|---|---|---|\n"
                + "\n".join(rows) + f"\n\n(layer 0's `wq`: |W| ≈ {float(jnp.abs(W).mean()):.4f} per entry, "
                f"|ΔW| ≈ {float(jnp.abs(delta).mean()):.5f})")

    _t = _merge_precision()
    mo.md(r"""
    ### Merge the LoRA in float32

    Your Module 20 and 26 models are a base model plus a small LoRA update $\Delta W = \tfrac{\alpha}{r}AB$.
    LM Studio has no documented way to load an adapter at run time, so we **merge**: $W' = W + \Delta W$, then
    export $W'$. The catch is the dtype of $W'$. $\Delta W$ is ~60× smaller than $W$ per entry, and bfloat16
    keeps only 8 significant bits, so rounding $W + \Delta W$ to bf16 adds an error that is a noticeable
    fraction of the update itself. `export.py` merges in **float32** and writes float32 safetensors; the GGUF
    step then stores **f16** (11 significant bits) — and bf16 too, so you can compare. Measured on Module 20's adapter:

    """ + (_t or "*(run Module 20 first to see this measured)*").replace("\n", "\n    ") + """

    Both keep the update's *direction* (the middle column), but bf16's rounding error is ~8× f16's. For this
    adapter it doesn't change a single answer (§8: 100 % either way) — a strong, narrow task — but subtler
    fine-tunes have less margin, so merge in float32 and store f16 unless you're quantizing anyway.
    """)
    return


@app.cell
def _(mo, read_json):
    _v, _e = read_json("verify.json"), read_json("export.json")
    if not _v:
        _view = mo.callout(mo.md("Run `make export-models` to export and verify (writes `verify.json`)."), kind="info")
    else:
        _rows = "\n    ".join(f"| `{k}` | {r['max_abs_logit_diff']:.1e} | {r['logit_scale']:.1f} | {r['greedy_agreement']:.0%} | "
                          f"{'✅' if r['tokenizer_ids_match'] else '❌'} |" for k, r in _v.items())
        _rt = (_e or {}).get("smollm2-360m-instruct", {})
        _view = mo.md(f"""
    ## 4 · Proof the exports are right

    Before converting anything, `verify_hf.py` loads each export with Hugging Face **transformers** (PyTorch,
    float32, in the course's PyTorch image) and runs 64 probe tokens through it, comparing against the JAX
    logits `export.py` saved:

    | export | max \\|Δ logit\\| | logit scale | same greedy token | tokenizer gives same ids |
    |---|---|---|---|---|
    {_rows}

    Differences of ~1e-4 on logits of size ~20–30 are float32 rounding between two frameworks; every next
    token agrees. The SmolLM2-360M round trip through `export_llama_hf` was bit-exact on all
    {_rt.get('tensors', '?')} tensors.
    """)
    _view
    return


@app.cell
def _(ex, lj, mo, read_json):
    _q, _e = read_json("gguf_quality.json"), read_json("export.json")
    if not _q or not _e:
        _view = mo.callout(mo.md("Run `make export-models` (its GGUF step writes `gguf_quality.json`)."), kind="info")
    else:
        _hb = _e["heldout"]
        _tpb = {"gpt16-detective": _hb["detective_tokens"] / _hb["bytes"]}
        _base = {"smollm2-360m-instruct": "bf16", "lora20-orders": "f16", "grpo26-arith": "f16", "gpt16-detective": "f32"}
        _rows = []
        for _m, _types in _q.items():
            if _m not in _base:
                continue
            _b = _types[_base[_m]]["ppl"]
            for _t, _r in _types.items():
                _d = lj.attempt(ex.bits_per_byte_delta, _b, _r["ppl"], _tpb.get(_m, _hb["smollm_tokens"] / _hb["bytes"]))
                _dtxt = f"{_d.value:+.4f}" if _d.ok else "*your turn*"
                _rows.append(f"| `{_m}` | {_t} | {_r['size_mb']:.0f} MB | {_r['ppl']:.3f} | {_dtxt} |")
        _view = mo.md("""
    ## 5 · GGUF, and what quantization costs

    `build_gguf.sh` converts each export with llama.cpp's `convert_hf_to_gguf.py` (ggml-org's official image,
    build b12375 — the one Module 19 serves with), quantizes with `llama-quantize`, and measures perplexity on
    ~47 KB of held-out detective prose with `llama-perplexity` on the GPU (Vulkan). **Your turn:**
    `bits_per_byte_delta` turns two perplexities into extra bits per byte of text (Module 21), so models with
    different tokenizers sit on one scale:

    | model | stored as | size | perplexity | Δ bits / byte vs base |
    |---|---|---|---|---|
    """ + "\n    ".join(_rows) + """

    * **Q8_0 is nearly free** (+0.001–0.003 bits/byte); **Q4_K_M costs ~1–2 % perplexity** for a 2.7× smaller
      file — the trade Module 19 took for the 14B coder.
    * **Your 14M GPT's** 384-wide rows aren't multiples of 256, which k-quants need, so `llama-quantize` falls
      back to Q5_0 for those tensors (it prints a warning per tensor) — "Q4_K_M" is really ~6.5 bits there.
    * **The custom tokenizer.** llama.cpp identifies a BPE tokenizer's pre-splitting rules by hashing how it
      tokenizes a test string, so a tokenizer *you trained* (Module 13/16) is unknown and the converter stops
      with "BPE pre-tokenizer was not recognized". Yours uses GPT-2's exact rules (byte-level, GPT-2 regex), so
      `convert_custom_bpe.py` checks `tokenizer.json` says so and tells the converter `gpt-2`.
    * bf16 vs f16 for the merged LoRAs barely moves perplexity on *general* text — the task tests in §8 and
      §10 are where the update has to survive.
    """)
    _view
    return


@app.cell
def _(ex, lj, mo):
    _cases = [("SmolLM2-360M", 3.62e8, 32 * 5 * 64 * 2 * 2), ("Qwen2.5-1.5B", 1.54e9, 28 * 2 * 128 * 2 * 2),
              ("Qwen2.5-Coder-14B", 14.8e9, 48 * 8 * 128 * 2 * 2), ("a 32B model", 32.8e9, 64 * 8 * 128 * 2 * 2)]
    _rows = []
    for _name, _n, _kv in _cases:
        _cells = []
        for _ctx in (4096, 32768):
            _a = lj.attempt(ex.quant_recommendation, _n, 16, _ctx, _kv)
            _cells.append((_a.value or "doesn't fit") if _a.ok else "*your turn*")
        _rows.append(f"| {_name} | {_kv / 1024:.0f} KiB | {_cells[0]} | {_cells[1]} |")
    mo.md("""
    **Your turn:** `quant_recommendation` — the best GGUF type that fits the 16 GB card (weights + KV cache at
    the context length + 1 GiB of runtime overhead). KV bytes/token = layers × KV heads × head dim × 2 (K and
    V) × 2 bytes:

    | model | KV cache / token | best fit at 4K context | at 32K context |
    |---|---|---|---|
    """ + "\n    ".join(_rows) + """

    Context length is a memory budget, not a free setting: the 14B coder can afford Q6_K at 4K but must drop
    to Q4_K_M at 32K.
    """)
    return


@app.cell
def _(RUNS, gguf_meta, mo):
    def _show(name):
        p = RUNS / "gguf" / f"{name}.gguf"
        if not p.exists():
            return f"| `{name}` | *(not built)* | | |"
        m = gguf_meta.read_metadata(p)
        t = m.get("tokenizer.chat_template")
        return (f"| `{name}` | {m.get('general.architecture')} | `{m.get('tokenizer.ggml.pre')}` | "
                + (f"{len(t)} chars, starts `{t[:60].replace('|', '&#124;')}…`" if t else "**none**") + " |")

    mo.md(r"""
    ## 6 · Chat templates live inside the GGUF

    A chat model only works if its prompt is formatted the way it was trained. GGUF stores the Hugging Face
    Jinja template as metadata key `tokenizer.chat_template`, and LM Studio renders it for every chat request
    (you can override it per model in the app). `gguf_meta.py` reads the metadata at the front of a GGUF
    file — no llama.cpp needed:

    | file | architecture | pre-tokenizer | chat template |
    |---|---|---|---|
    """ + "\n    ".join(_show(n) for n in ("smollm2-360m-instruct-bf16", "lora20-orders-f16", "gpt16-detective-f16")) + r"""

    Two consequences you'll see below:

    * Your **GPT has no template** (it was never a chat model), so it's used through the plain **completions**
      endpoint (§11).
    * SmolLM2's template **adds a default system prompt** ("You are a helpful AI assistant named SmolLM…")
      whenever you don't send one. Module 20 trained the LoRA *without* a system message — so through LM
      Studio's chat endpoint your extractor sees a prompt it never trained on. §8 measures what that costs.
    """)
    return


@app.cell
def _(BASE, ex, lj, lmclient, mo):
    def _apis():
        if not BASE:
            return mo.callout(mo.md("Start LM Studio (`make lms-up`) to run these calls."), kind="info")
        key = "smollm2-360m-instruct@q8_0"
        lmclient.ensure_loaded(BASE, key)
        q = "In one sentence: why does a GPU need batching to be efficient?"
        b1 = ex.openai_chat_request(key, [{"role": "user", "content": q}], max_tokens=60)
        r1 = lmclient.post(BASE, "/v1/chat/completions", b1)
        b2 = ex.lmstudio_rest_request(key, q, max_output_tokens=60)
        r2 = lmclient.post(BASE, "/api/v1/chat", b2)
        s = r2.get("stats", {})
        return mo.md(f"""
    **OpenAI-compatible** `POST /v1/chat/completions` with your request body:

    > {r1['choices'][0]['message']['content'].strip()}

    **LM Studio REST** `POST /api/v1/chat` (same question) — this one reports performance:
    {s.get('tokens_per_second', 0):.0f} tokens/s, first token after {s.get('time_to_first_token_seconds', 0) * 1000:.0f} ms,
    {s.get('total_output_tokens')} tokens:

    > {r2['output'][0]['content'].strip()}
    """)

    _a = lj.attempt(_apis)
    mo.vstack([mo.md(r"""
    ## 7 · Three APIs

    LM Studio serves three HTTP APIs on one port ([developer docs](https://lmstudio.ai/docs/developer)):

    * **OpenAI-compatible** `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`,
      `/v1/models` — so any OpenAI client works with `base_url="http://localhost:1234/v1"`.
    * **LM Studio REST** `/api/v1/chat`, `/api/v1/models`, `/api/v1/models/load`, `.../unload`,
      `.../download` — model management, and per-request stats.
    * **Anthropic-compatible** `POST /v1/messages` (§9).

    **Your turn:** `openai_chat_request`, `lmstudio_rest_request`, `anthropic_messages_request` build the JSON
    bodies; the lesson sends exactly what your functions return.
    """), _a.value if _a.ok else _a.view(mo)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    LM Studio's own Python SDK, [`lmstudio`](https://lmstudio.ai/docs/python), wraps the same server. In your
    own projects, add it with [uv](https://docs.astral.sh/uv/) (`uv add lmstudio`) or pip (`pip install lmstudio`). It isn't installed in the course image (the lesson sticks to plain HTTP), but this is
    what the call above looks like with it:

    ```python
    import lmstudio as lms

    with lms.Client("localhost:1234") as client:
        model = client.llm.model("smollm2-360m-instruct@q8_0")
        print(model.respond("In one sentence: why does a GPU need batching to be efficient?"))
    ```
    """)
    return


@app.cell
def _(BASE, ROOT, ex, json, lj, lmclient, load_mod, mo, read_json):
    def _structured():
        if not BASE:
            return mo.callout(mo.md("Start LM Studio (`make lms-up`) to evaluate the extractor."), kind="info")
        avail = lmclient.available(BASE)
        if "lora20-orders@f16" not in avail:
            return mo.callout(mo.md("Your Module 20 model isn't imported — run `make export-models`."), kind="info")
        task = load_mod(ROOT / "modules" / "20-lora" / "task.py", "m27_lesson_task20")
        held = task.make_splits(seed=0)["heldout"][:40]
        schema = {"type": "json_schema", "json_schema": {"name": "order", "strict": True, "schema": {
            "type": "object", "required": ["item", "qty", "color", "city", "day"], "additionalProperties": False,
            "properties": {"item": {"type": "string"}, "qty": {"type": "integer"}, "color": {"type": "string"},
                           "city": {"type": "string"}, "day": {"type": "string"}}}}}

        def run(key, mode):
            lmclient.ensure_loaded(BASE, key)
            ok = 0
            for e in held:
                if mode == "completions":       # the exact prompt Module 20 trained on
                    r = lmclient.post(BASE, "/v1/completions", {"model": key, "prompt": task.chatml_prompt(e["text"]),
                                                                "max_tokens": 48, "temperature": 0, "stop": ["<|im_end|>"]})
                    text = r["choices"][0]["text"]
                else:                            # chat endpoint: the GGUF's template adds a system prompt
                    body = ex.openai_chat_request(key, [{"role": "user", "content": task.INSTRUCTION + e["text"]}],
                                                  max_tokens=48, response_format=schema if mode == "chat+schema" else None)
                    text = lmclient.post(BASE, "/v1/chat/completions", body)["choices"][0]["message"]["content"]
                try:
                    ok += json.loads(text) == json.loads(e["answer"])
                except Exception:  # noqa: BLE001 - invalid JSON counts as wrong
                    pass
            return ok / len(held)

        rows = [("lora20-orders@f16", "completions (training prompt)"), ("lora20-orders@f16", "chat"),
                ("lora20-orders@f16", "chat+schema"), ("lora20-orders@bf16", "completions (training prompt)"),
                ("lora20-orders@q4_k_m", "completions (training prompt)"),
                ("smollm2-360m-instruct@q8_0", "chat+schema")]
        out = []
        for key, mode in rows:
            out.append(f"| `{key}` | {mode} | {run(key, mode.split(' ')[0]):.0%} |")
        lmclient.unload_all(BASE)
        ref = (read_json("jax_reference.json") or {}).get("lora20-orders", {})
        return mo.md(f"JAX reference (float32 merge, `llama.generate`): **{ref.get('exact_match', float('nan')):.0%}** "
                     f"on the same {ref.get('n', 40)} examples.\n\n| LM Studio model | endpoint | correct |\n|---|---|---|\n"
                     + "\n".join(out))

    _a = lj.attempt(_structured)
    mo.vstack([mo.md(r"""
    ## 8 · Your JSON extractor, served

    Module 20's LoRA turns "Rush job: three blue lamps -> Denver, Friday." into one JSON line. Here it is
    through LM Studio on the 40 **held-out** sentences (templates it never trained on), scored by parsing the
    JSON and comparing to the answer. Three ways to ask:

    * **completions** with the exact ChatML prompt Module 20 trained on;
    * **chat**: LM Studio renders the GGUF's template — which adds SmolLM2's default system prompt;
    * **chat + `json_schema`**: LM Studio's [structured output](https://lmstudio.ai/docs/developer/openai-compat/structured-output)
      constrains sampling with a grammar so the reply *must* be valid JSON matching the schema.

    The last row runs the **base** model with a schema: always-valid JSON is not the same as the right answer.
    (~1 minute live.)
    """), _a.value if _a.ok else _a.view(mo)])
    return


@app.cell
def _(BASE, ex, json, lj, lmclient, mo):
    def _tools():
        if not BASE:
            return mo.callout(mo.md("Start LM Studio (`make lms-up`) to try tool calling."), kind="info")
        key = "qwen2.5-1.5b-instruct"
        if key not in lmclient.available(BASE):
            return mo.callout(mo.md("Qwen2.5-1.5B-Instruct isn't imported — `make export-models` downloads and "
                                    "imports it (≈1 GB, Apache-2.0)."), kind="info")
        lmclient.ensure_loaded(BASE, key)
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                  "parameters": {"type": "object", "required": ["city"], "properties": {
                      "city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}}}}}]
        body = ex.openai_chat_request(key, [{"role": "user", "content": "What's the weather in Denver, in Celsius?"}],
                                      tools=tools, stream=True)
        events = lmclient.stream(BASE, "/v1/chat/completions", body)
        frags = [e["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
                 for e in events if e.get("choices") and e["choices"][0]["delta"].get("tool_calls")]
        calls = lj.attempt(ex.assemble_tool_call_stream, events)
        res = mo.md(f"{len(events)} streamed events; the arguments arrived in **{len(frags)} fragments**: "
                    f"`{frags[:8]}`…\n\nYour reassembled calls:\n\n```json\n{json.dumps(calls.value, indent=2)}\n```")
        return res if calls.ok else calls.view(mo)

    _a = lj.attempt(_tools)
    mo.vstack([mo.md(r"""
    ## 9 · Tool calling, streamed

    With `tools` in the request, a model that knows a tool-call format answers with a structured call instead
    of text, and LM Studio parses it into OpenAI's `tool_calls` field
    ([tool use docs](https://lmstudio.ai/docs/developer/openai-compat/tools)). Streamed, the call arrives as
    fragments: the id and name once, the JSON arguments a few characters at a time.

    A real-world catch found while building this module: **Qwen2.5-Coder** (Module 19's models, 1.5B and 14B,
    from Qwen's GGUF repos) answered with a JSON code block or `<tools>…</tools>` instead of the
    `<tool_call>` tags LM Studio parses — so `tool_calls` came back empty. **Qwen2.5-1.5B-Instruct** (general
    instruct, also Apache-2.0) emits proper calls. When tool calls "don't work" locally, check what the model
    actually wrote before blaming the server.

    **Your turn:** `assemble_tool_call_stream(chunks)`.
    """), _a.value if _a.ok else _a.view(mo)])
    return


@app.cell
def _(BASE, ex, lj, lmclient, mo):
    def _anthropic():
        if not BASE:
            return mo.callout(mo.md("Start LM Studio (`make lms-up`) to call the Anthropic-compatible endpoint."), kind="info")
        key = "qwen2.5-1.5b-instruct" if "qwen2.5-1.5b-instruct" in lmclient.available(BASE) else "smollm2-360m-instruct@q8_0"
        lmclient.ensure_loaded(BASE, key)
        body = ex.anthropic_messages_request(key, [{"role": "user", "content": "What does jax.jit do?"}],
                                             max_tokens=80, system="Answer in one sentence.")
        r = lmclient.post(BASE, "/v1/messages", body, headers={"x-api-key": "lmstudio", "anthropic-version": "2023-06-01"})
        lmclient.unload_all(BASE)
        return mo.md(f"`POST /v1/messages` → `{r.get('type')}` from `{r.get('model')}`, stop_reason `{r.get('stop_reason')}`, "
                     f"usage {r.get('usage')}:\n\n> {r['content'][0]['text'].strip()}")

    _a = lj.attempt(_anthropic)
    mo.vstack([mo.md(r"""
    ## 10 · The Anthropic-compatible endpoint (and Claude Code)

    LM Studio also speaks the Anthropic Messages API
    ([docs](https://lmstudio.ai/docs/developer/anthropic-compat)): `system` is a top-level field and
    `max_tokens` is required. That means **Claude Code can be pointed at a local model**:

    ```bash
    export ANTHROPIC_BASE_URL=http://localhost:1234
    export ANTHROPIC_AUTH_TOKEN=lmstudio
    claude --model qwen2.5-coder-14b-instruct
    ```

    Honest expectations: Claude Code is an agent — it lives on reliable tool calls, long context and careful
    edits. A 1–14B local model on a 16 GB card is far weaker at all three, and §9 showed the coder models'
    tool-call format doesn't even parse here. It's a fine way to experiment offline, not a replacement.
    """), _a.value if _a.ok else _a.view(mo)])
    return


@app.cell
def _(BASE, lj, lmclient, mo):
    def _gpt():
        if not BASE:
            return mo.callout(mo.md("Start LM Studio (`make lms-up`) to sample your GPT."), kind="info")
        key = "gpt16-detective@f16"
        if key not in lmclient.available(BASE):
            return mo.callout(mo.md("Your GPT isn't imported — run `make export-models`."), kind="info")
        lmclient.ensure_loaded(BASE, key)
        out = []
        for seed in (0, 1):
            r = lmclient.post(BASE, "/v1/completions", {"model": key, "prompt": "The fog came in off the",
                                                        "max_tokens": 60, "temperature": 0.8, "seed": seed})
            out.append("> The fog came in off the" + r["choices"][0]["text"])
        lmclient.unload_all(BASE)
        return mo.md("\n\n".join(out))

    _a = lj.attempt(_gpt)
    mo.vstack([mo.md(r"""
    ## 11 · The GPT you trained, in LM Studio

    Your Module 16 model — 13.9M parameters, trained 15 minutes on 207 detective novels — loads as an
    ordinary `gpt2` model with a 256-token context. No chat template, so it uses `/v1/completions`:
    """), _a.value if _a.ok else _a.view(mo)])
    return


@app.cell
def _(BASE, ROOT, lj, lmclient, load_mod, mo, read_json):
    def _grpo():
        if not BASE:
            return mo.callout(mo.md("Start LM Studio (`make lms-up`) to evaluate the reasoning model."), kind="info")
        if "grpo26-arith@f16" not in lmclient.available(BASE):
            return mo.callout(mo.md("Your Module 26 model isn't imported — run Module 26's `train.py`, then "
                                    "`make export-models`."), kind="info")
        R = load_mod(ROOT / "modules" / "26-reasoning" / "reasoning.py", "m27_lesson_r26")
        verify = lj.load_exercises(ROOT / "modules" / "26-reasoning", solutions=True).verify_answer
        test = R.make_split(1, 64)

        def acc(key):
            lmclient.ensure_loaded(BASE, key)
            ok = 0
            for p in test:
                msgs = [{"role": "system", "content": R.SYSTEM}, {"role": "user", "content": R.SHOT_Q},
                        {"role": "assistant", "content": R.SHOT_COT}, {"role": "user", "content": p["question"]}]
                r = lmclient.post(BASE, "/v1/chat/completions", {"model": key, "messages": msgs, "temperature": 0,
                                                                 "max_tokens": 64})
                ok += bool(verify(r["choices"][0]["message"]["content"], p["target"]))
            lmclient.unload_all(BASE)
            return ok / len(test)

        ref = (read_json("jax_reference.json") or {}).get("grpo26-arith", {})
        rows = [("base SmolLM2-360M-Instruct", ref.get("accuracy_base"), acc("smollm2-360m-instruct@bf16")),
                ("after GRPO (Module 26)", ref.get("accuracy_grpo"), acc("grpo26-arith@f16"))]
        return mo.md("| model | JAX (bf16 + adapter) | LM Studio (GGUF) |\n|---|---|---|\n" +
                     "\n".join(f"| {n} | {j:.1%} | {lm:.1%} |" for n, j, lm in rows if j is not None))

    _a = lj.attempt(_grpo)
    mo.vstack([mo.md(r"""
    ## 12 · Your reasoning model, served

    Module 26's GRPO adapter, merged in float32 and stored f16, on the same 64 held-out arithmetic problems
    with the same one-shot ChatML prompt (here sent as chat messages, which the GGUF template renders
    identically), greedy. JAX numbers come from `export.py`. Expect agreement within a few problems, not
    exactly: different kernels flip bf16 near-ties (Module 26 saw the same between two JAX processes).
    (~45 s live.)
    """), _a.value if _a.ok else _a.view(mo)])
    return


@app.cell
def _(mo, read_json):
    _b = read_json("bench.json")
    _rows = [r for r in (_b or {}).get("rows", []) if "draft" in r["engine"]]
    _t = ("\n    ".join(f"| {'with 135M draft' if 'no draft' not in r['engine'] else 'alone'} | {r['gen_tok_s']:.0f} |" for r in _rows)
          if _rows else "| *(run `make lms-bench`)* | |")
    mo.md(r"""
    ## 13 · Speculative decoding in LM Studio

    LM Studio can pair a target with a small **draft model** that must share its vocabulary
    ([docs](https://lmstudio.ai/docs/app/advanced/speculative-decoding)) — SmolLM2-135M and -360M do. In
    `llmster` 0.0.27 it's configured **at load time**, not per request (a per-request `draft_model` is
    rejected: "speculative decoding must be configured at load time"):

    ```bash
    lms load smollm2-360m-instruct@bf16 --identifier spec360 \
        --speculative-draft-simple --speculative-draft-model smollm2-135m
    ```

    | SmolLM2-360M-Instruct bf16, Vulkan | tokens / s |
    |---|---|
    """ + _t + r"""

    Slower, not faster — the same verdict as Module 18's JAX implementation: a 135M draft costs too large a
    fraction of a 360M target's step for speculation to pay off. It shines when the draft is ~10–50× smaller
    than the target (e.g. a 0.5B draft for a 14B model) and the text is predictable (code).
    """)
    return


@app.cell
def _(BASE, lj, lmclient, mo, np):
    def _embed():
        if not BASE:
            return mo.callout(mo.md("Start LM Studio (`make lms-up`) to try the embeddings endpoint."), kind="info")
        emb = [m["key"] for m in lmclient.models(BASE) if m.get("type") == "embedding"]
        if not emb:
            return mo.callout(mo.md(
                "No embedding model in LM Studio yet. Get the one the owner's aws-ai `local-kb` uses, then re-run:\n\n"
                "```\ncontainers/lmstudio.sh lms get Qwen/Qwen3-Embedding-0.6B-GGUF\n```"), kind="info")
        key = emb[0]
        lmclient.ensure_loaded(BASE, key)
        texts = ["The interferometer was calibrated against a bright quasar.",
                 "Phase calibration of the radio array used a strong compact source.",
                 "The kettle on the stove began to whistle."]
        r = lmclient.post(BASE, "/v1/embeddings", {"model": key, "input": texts})
        v = np.array([d["embedding"] for d in r["data"]], np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        s = v @ v.T
        lmclient.unload_all(BASE)
        return mo.md(f"`{key}` returned {v.shape[0]} vectors of **{v.shape[1]}** dimensions. Cosine similarity: "
                     f"sentences 1–2 (same meaning, different words) **{s[0, 1]:.2f}**, 1–3 (unrelated) **{s[0, 2]:.2f}**.")

    _a = lj.attempt(_embed)
    mo.vstack([mo.md(r"""
    ## 14 · Embeddings: the other endpoint

    `/v1/embeddings` turns text into a vector instead of more text: `{"model": ..., "input": [...]}` in,
    one unit-length vector per input out. Texts with similar meaning land close together, so "find the passages
    nearest this question" becomes a matrix multiply. That is the retrieval half of **RAG**, which Module 38
    builds from scratch in JAX (BM25, a bi-encoder you train, hybrid search, reranking) and compares with an
    embedding model served exactly like this one.
    """), _a.value if _a.ok else _a.view(mo)])
    return


@app.cell
def _(mo, read_json):
    quant15 = read_json("quantlab/results.json")
    _intro = mo.md(r"""
    ## 15 · The quantizer's workflow I: importance matrices, on your LoRA

    §5 rounded every weight as if all of them mattered equally. They don't. A linear layer computes
    $y = \sum_j w_j x_j$, so an error $\delta_j = w_j - \hat w_j$ in weight $j$ costs $\delta_j x_j$ in the
    output: rounding a weight that multiplies a large, frequent input hurts far more than one whose input is
    usually near zero. If the inputs are roughly uncorrelated, the expected squared output error is

    $$\mathbb{E}\Big[\big(\textstyle\sum_j \delta_j x_j\big)^2\Big] \approx \sum_j \delta_j^2\, \mathbb{E}[x_j^2].$$

    That weighting is the **importance matrix**. `llama-imatrix` runs the model over calibration text and
    records $\mathbb{E}[x_j^2]$ for every input column of every linear layer; `llama-quantize --imatrix` then
    picks each block's scale to minimise the weighted error instead of the plain one (ggml-org,
    [imatrix README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md); Iwan Kawrakow
    added it to llama.cpp in early 2024). It costs nothing at inference: the file has the same size and format.

    The same work brought the **i-quants** (IQ2_XS, IQ3_XXS, …): groups of 8 weights stored as an index into a
    fixed codebook of points on a lattice, the idea of QuIP# (Tseng et al., 2024), reaching 2–3 bits per
    weight. At those widths blind rounding is hopeless, so `llama-quantize` won't make most of them without an
    imatrix.

    **What calibration text?** `make quant-lab` (~10 min) tries two on your Module 20 LoRA (SmolLM2-360M,
    merged in float32 by `export.py`): **task**, 1,000 of its training prompts with answers, and **generic**,
    the first half of the held-out detective prose. Perplexity is measured on the *other* half (measuring it on
    the calibration text would flatter the generic imatrix), and the task score is exact match on Module 20's
    200 held-out orders through `llama-server`.
    """)
    _intro
    return (quant15,)


@app.cell
def _(mo):
    skew15 = mo.ui.slider(1, 30, step=1, value=10, label="how much larger the 4 'loud' input channels are",
                          show_value=True)
    mo.vstack([mo.md(r"""
    **Build it from scratch.** A toy layer: 64 inputs, 32 outputs, Gaussian weights. Four of the 64 input
    channels are *loud* (their values are larger than the rest), as real activations are: a few channels
    carry outliers. Quantize the weights to 3 bits (levels −3 … 3, one scale per output column) two ways.
    Both search the same grid of scales; **plain** picks the one with the smallest weight error, **importance
    weighted** the one with the smallest error weighted by $a_i = \mathbb{E}[x_i^2]$, measured on 2,048
    calibration inputs. Then measure the layer's *output* error on 2,048 fresh inputs.
    """), skew15])
    return (skew15,)


@app.cell
def _(jax, jnp, mo, skew15):
    def _quantize(w, a, m=3):
        fs = jnp.linspace(0.5, 1.0, 11)
        amax = jnp.abs(w).max(axis=0)
        s = jnp.where(amax == 0, 1.0, amax)[None, :] / m * fs[:, None]           # (11, d_out) candidate scales
        deq = jnp.clip(jnp.round(w[None] / s[:, None]), -m, m) * s[:, None]      # (11, d_in, d_out)
        err = (a[None, :, None] * (w[None] - deq) ** 2).sum(axis=1)              # importance-weighted error
        k = err.argmin(axis=0)
        return jnp.take_along_axis(deq, k[None, None, :], axis=0)[0]

    _k = jax.random.split(jax.random.key(0), 3)
    _w = jax.random.normal(_k[0], (64, 32))
    _std = jnp.ones(64).at[:4].set(float(skew15.value))
    _calib = jax.random.normal(_k[1], (2048, 64)) * _std
    _fresh = jax.random.normal(_k[2], (2048, 64)) * _std
    _a = (_calib ** 2).mean(0)
    _rows = []
    for _name, _imp in (("plain", jnp.ones(64)), ("importance weighted", _a)):
        _wq = _quantize(_w, _imp)
        _werr = float(jnp.linalg.norm(_w - _wq) / jnp.linalg.norm(_w))
        _yerr = float(jnp.linalg.norm(_fresh @ (_w - _wq)) / jnp.linalg.norm(_fresh @ _w))
        _rows.append(f"| {_name} | {_werr:.3f} | **{_yerr:.3f}** |")
    mo.md(r"""
    | 3-bit weights, scale chosen by | weight error (relative) | output error on fresh inputs (relative) |
    |---|---|---|
    """ + "\n    ".join(_rows) + r"""

    Importance weighting accepts a *larger* error in the weights overall to make a smaller one where the loud
    inputs will multiply it, and the output error falls. With equal channels (slider at 1) the two choices
    coincide. This is the whole of what an imatrix changes: the same levels, the same file size, a different
    choice of scale. llama.cpp's version searches finer and per block of 16–32 weights, not per column.
    """)
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "importance-weighted quantization")
    return


@app.cell
def _(mo, quant15):
    if not quant15:
        _out = mo.callout(mo.md("No results yet: run `make quant-lab` (after `make export-models`), ~10 minutes, "
                                "then re-run this cell."), kind="info")
    else:
        def _c(q, c, f, fmt):
            v = quant15.get(f"{q}-{c}", {}).get(f)
            return format(v, fmt) if v is not None else "–"

        _rows = []
        for _q in ("F16", "Q8_0", "Q4_K_M", "Q3_K_S", "Q2_K", "IQ3_XXS", "IQ2_XS"):
            _any = next((v for k, v in quant15.items() if k.startswith(_q + "-")), None)
            if _any is None:
                continue
            _rows.append(f"| {_q} | {_any['bpw'] or 16:.2f} | {1000 * _any['size_gb']:.0f} | "
                         + " / ".join(_c(_q, c, "exact_match", ".1%") for c in ("none", "task", "generic")) + " | "
                         + " / ".join(_c(_q, c, "ppl", ".2f") for c in ("none", "task", "generic")) + " |")
        _out = mo.md(r"""
    | quant | real bits/weight | MB | orders exact match (none / task / generic imatrix) | perplexity (none / task / generic) |
    |---|---|---|---|---|
    """ + "\n    ".join(_rows) + r"""

    * **Read the real bits per weight, not the name.** "Q3_K_S" and "Q2_K" are both **4.80 bits** and score
      the same: SmolLM2-360M's rows are 960 wide, k-quants need multiples of 256, so `llama-quantize` falls back
      to Q4_0 for 192 of the 290 tensors (it warns: "192 of 290 tensor(s) required fallback quantization").
      Only the 32 `ffn_down` matrices, 2,560 wide, get real 3-bit blocks. The i-quants fall back the same way
      (to IQ4_NL), so "IQ2_XS" is 4.56 bits with 28 tensors truly at 2 bits. §5 saw the same with your 14M GPT.
    * **The imatrix still helps where rounding happens.** At ~4.8 bits it cuts held-out perplexity by about a
      tenth (none → generic); at Q4_K_M by about 1 %.
    * **This task can't separate the quants.** Extracting an order into JSON is so easy for this model that every
      quant stays within a few of 200 examples of perfect, and those few go either way. Too small a model, too
      easy a task: §16 does it properly.
    """)
    _out
    return


@app.cell
def _(ROOT, json, read_json):
    q9_ref = json.loads((ROOT / "modules" / "27-lmstudio" / "quantlab" / "reference_9b.json").read_text())
    q9 = read_json("quantlab9b/summary.json")
    q9_src = "your run (`runs/27-lmstudio/quantlab9b/summary.json`)"
    if q9 is None:
        q9 = q9_ref
        q9_src = ("the course author's run (`quantlab/reference_9b.json`); `make quant-lab-9b` replaces it with "
                  "yours")
    return q9, q9_ref, q9_src


@app.cell
def _(mo, plt, q9, q9_src):
    _t, _b, _f = q9.get("train", {}), q9.get("eval_base_5shot", {}), q9.get("eval_ft", {})
    _fig = None
    if _t.get("loss"):
        _fig, _ax = plt.subplots(figsize=(5.2, 2.4))
        _ax.plot([s for s, _ in _t["loss"]], [v for _, v in _t["loss"]], lw=1.2)
        _ax.set_xlabel("optimizer step (16 examples)")
        _ax.set_ylabel("training loss")
        _ax.set_title("QLoRA, Qwen3.5-9B on NL-RX", fontsize=10)
        _fig.tight_layout()
    _ex = "\n    ".join(f"| {o['nl'][:60].replace('|', '&#124;')} | `{o['gt'].replace('|', '&#124;')}` | "
                        f"`{o['gen'].replace('|', '&#124;')}` |" for o in q9.get("examples", [])[:6])
    mo.vstack([mo.md(rf"""
    ## 16 · The quantizer's workflow II: a real 9B

    §15's model was too small and its task too easy to separate the quants. Now the real thing, the job
    community uploaders such as [TheBloke](https://huggingface.co/TheBloke) (2023) and today
    [bartowski](https://huggingface.co/bartowski) and [Unsloth](https://huggingface.co/unsloth) do for every
    new model, on a model you fine-tuned yourself: **Qwen3.5-9B** (Qwen team, Apache-2.0), a hybrid of
    full attention and Gated DeltaNet linear attention (Yang, Kautz & Hatamizadeh, 2024) whose 4096-wide rows
    divide by 256, so every k-quant and i-quant is real here.

    **The task:** NL-RX (Locascio et al., 2016), English to regular expressions in a dialect with `&` (and)
    and `~` (not): "lines that do not contain 'dog' followed by a number, three times or more" →
    `~((dog.*[0-9].*){{3,}})`. The 8,000 training pairs are paraphrases written by Mechanical Turk workers;
    `learnjax.data.nlrx()` fetches the chat-formatted copy from the LLM-JEPA repo (galilai-group,
    Apache-2.0). Scored by exact string match on 500 test pairs, which undercounts: an equivalent regex
    written differently is a miss.

    **The pipeline** (`make quant-lab-9b`, ~2.5 hours, optional; this section shows {q9_src}):

    1. **QLoRA** (Dettmers et al., 2023) in the PyTorch image: the base model loaded in 4-bit NF4 with
       bitsandbytes and frozen, a rank-16 LoRA on every linear layer, one epoch: {_t.get('steps', '?')} steps
       in {_t.get('minutes', '?')} min, {_t.get('peak_gib', '?')} GiB peak, {_t.get('skipped', '?')} skipped
       updates. A 9B model in bf16 is 18 GB of weights alone; in NF4 it's about 5.
    2. **Merge in float32** into the original bf16 safetensors (`merge_f32.py`), then **convert to an f32
       GGUF** (37 GB): the reference every quant is made from.
    3. **Q8_0**, then **importance matrices from the Q8_0** on the GPU (the f32 file doesn't fit in 16 GB),
       from task text and from generic prose, as in §15.
    4. **Quantize** Q4_K_M, IQ4_XS, IQ3_XXS, IQ2_XXS; **score** each through `llama-server`.

    | | exact match, 500 test pairs |
    |---|---|
    | base Qwen3.5-9B, 5 examples in the prompt | {_b.get('exact_match', float('nan')):.1%} |
    | after QLoRA (4-bit base + adapter, PyTorch) | **{_f.get('exact_match', float('nan')):.1%}** |

    The base model writes Python-flavoured regexes with explanations; one epoch teaches it the dialect and the
    format. What it still gets wrong:

    | description | reference | fine-tuned |
    |---|---|---|
    {_ex}

    Some misses are real errors; some are equivalent or arguably better than the reference ("all lines"):
    the Turk paraphrases are noisy. Exact match is a lower bound.
    """)] + ([_fig] if _fig else []))
    return


@app.cell
def _(mo, q9, q9_ref):
    _a = q9_ref["allocator_probe"]
    _u = q9.get("update_rel") or q9_ref["update_rel"]
    mo.md(rf"""
    ### Snag 1: NaN gradients that weren't numerics

    The first full training run kept skipping updates with non-finite gradients (9 by step 105), then **hung
    the GPU** with a page fault. Bitsandbytes was the first suspect (its 0.50.2 wheel has no
    ROCm 7.16 build); rebuilding it for 7.16 changed nothing, and neither did a 4B model in plain bf16 with no
    bitsandbytes at all (7 of 40 updates skipped either way). Two observations pointed elsewhere: the **loss was always finite**, and PyTorch's
    anomaly mode found **no backward op producing a NaN**, yet ~470 of 496 LoRA gradient tensors went NaN at
    once, next to garbage values like 4.7e32, mostly on the first micro-batch after `zero_grad` (fresh
    gradient memory). That is memory corruption, not arithmetic. The cause was an environment variable every
    run had set:

    | {_a['steps']}-step probe, same data and settings | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | unset |
    |---|---|---|
    | Qwen3.5-9B, 4-bit | {_a['with']['qwen3.5-9b-4bit']} skipped | **{_a['without']['qwen3.5-9b-4bit']}** |
    | Qwen3.5-4B, bf16 | {_a['with']['qwen3.5-4b-bf16']} skipped | **{_a['without']['qwen3.5-4b-bf16']}** |

    Expandable segments make PyTorch's allocator map GPU memory through virtual-memory APIs, a far less
    exercised path on ROCm than on CUDA. Without it the full run had 0 skipped updates, the same speed and
    memory, and no hang. `qlora_nlrx.py` refuses to start with it set, and keeps a cheap guard that skips an
    update whose gradient norm isn't finite. The lesson generalises: when values go bad everywhere at once
    and no single operation produced them, suspect memory, and change one setting at a time.

    ### Why merge in float32, again

    The trained update is small next to the weights it changes: $\lVert\Delta W\rVert / \lVert W\rVert$ has a
    median of **{_u['median']:.1e}** (max {_u['max']:.1e}) over the 248 adapted matrices. bf16's relative
    precision is about 3.9e-3 and f16's about 4.9e-4, so a bf16 merge would round each update to a step not
    much smaller than the update itself (§3 measured the same effect on your Module 20 LoRA). The update also
    grows with training: after a 40-step probe it was 4.8e-4, far below bf16's step, so most of it would
    simply have vanished. Merge in f32, keep an f32 GGUF as the source, and let the quantizer do the rounding,
    once.
    """)
    return


@app.cell
def _(mo, plt, q9):
    _Q = q9["quants"]
    _order = ["Q8_0", "Q4_K_M", "IQ4_XS", "IQ3_XXS", "IQ2_XXS"]

    def _cell(q, c, f, fmt):
        v = _Q.get(f"{q}-{c}", {}).get(f)
        return format(v, fmt) if v is not None else "–"

    _rows = []
    for _q in _order:
        _any = next((v for k, v in _Q.items() if k.startswith(_q + "-")), None)
        if _any is None:
            continue
        _rows.append(f"| {_q} | {_any['bpw'] or float('nan'):.2f} | {_any['size_gb']:.2f} | "
                     + " / ".join(_cell(_q, c, "exact_match", ".1%") for c in ("none", "task", "generic")) + " | "
                     + " / ".join(_cell(_q, c, "ppl", ".1f") for c in ("none", "task", "generic")) + " | "
                     + " / ".join(_cell(_q, c, "agree_with_ref", ".0%") for c in ("none", "task", "generic")) + " |")
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(8, 2.6))
    for _c, _style in (("task", "o-"), ("generic", "s--")):
        _pts = sorted((v["bpw"], v["exact_match"], v["ppl"]) for k, v in _Q.items()
                      if k.endswith("-" + _c) or k == "Q8_0-none")
        _a1.plot([p[0] for p in _pts], [100 * p[1] for p in _pts], _style, label=f"{_c} imatrix", ms=4)
        _a2.plot([p[0] for p in _pts], [p[2] for p in _pts], _style, label=f"{_c} imatrix", ms=4)
    _a1.set_xlabel("real bits per weight")
    _a1.set_ylabel("NL-RX exact match (%)")
    _a2.set_xlabel("real bits per weight")
    _a2.set_ylabel("perplexity, held-out prose")
    _a1.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(r"""
    ### The quants

    Every quant scored on the same 500 test prompts, in the exact chat format the model was trained on,
    greedy, through `llama-server` (Vulkan); perplexity on the half of the held-out prose the generic imatrix
    didn't see. The last column is how often a quant gives *exactly the same answer* as Q8_0: steadier than
    exact match, whose noise at 500 examples is about ±2.2 points.

    | quant | real bits/weight | GB | exact match (none / task / generic imatrix) | perplexity (none / task / generic) | same answer as Q8_0 |
    |---|---|---|---|---|---|
    """ + "\n    ".join(_rows) + r"""
    """), _fig])
    return


@app.cell
def _(mo, q9):
    _Q = q9["quants"]

    def _e(k):
        return _Q.get(k, {}).get("exact_match", float("nan"))

    def _p(k):
        return _Q.get(k, {}).get("ppl", float("nan"))

    _ft = q9.get("eval_ft", {}).get("exact_match", float("nan"))

    mo.md(rf"""
    What the table says:

    1. **Down to IQ4_XS (~4.7 real bits), nothing measurable is lost.** Q8_0 {_e('Q8_0-none'):.1%},
       Q4_K_M {_e('Q4_K_M-task'):.1%}, IQ4_XS {_e('IQ4_XS-task'):.1%}: all inside the noise. IQ4_XS is
       the pick: Q8_0's accuracy in 55 % of the file. (The PyTorch number, {_ft:.1%},
       ran the adapter on a 4-bit NF4 base, so a good 4.7-bit quant of the full-precision merge can match it.)
    2. **The cliff is between 3.7 and 3.0 bits.** IQ3_XXS loses a few points; IQ2_XXS falls apart.
    3. **Calibration only matters at the cliff, and there it's a trade.** At IQ2_XXS the task imatrix
       scores {_e('IQ2_XXS-task'):.1%} against the generic one's {_e('IQ2_XXS-generic'):.1%}, while the
       generic one keeps prose better (perplexity {_p('IQ2_XXS-generic'):.1f} vs {_p('IQ2_XXS-task'):.1f}).
       Each imatrix protects what it saw. At 4–5 bits the choice barely matters.
    4. **Perplexity ranks the IQ2 quants the wrong way round for this model's job.** For a fine-tune,
       measure the task.
    5. **Read the real bits per weight, not the name.** "Q4_K_M" is 5.06 bits here (Module 27's `GGUF_BITS`
       table says 4.85): the embedding and output matrices, a sizable share of a 9B with a 248K vocabulary,
       are kept at higher precision.
    """)
    return


@app.cell
def _(mo, q9_ref):
    _t = q9_ref["lmstudio_thinking"]
    mo.md(rf"""
    ### Snag 2: a layer the imatrix never sees

    The first IQ3_XXS and IQ2_XXS runs stopped with *"Missing importance matrix for tensor blk.32.attn_k.weight
    in a very low-bit quantization. The result will be garbage, so bailing out"*. Qwen3.5-9B has 32 layers,
    but its GGUF has 33 blocks: `blk.32` is a **multi-token-prediction head** (`qwen35.nextn_predict_layers =
    1`), used only to draft tokens for speculative decoding (§13). Normal inference never runs it, so
    `llama-imatrix` collected no statistics for it, and the 2–3-bit types won't guess. The fix keeps that one
    block at Q8_0, about 0.26 GB:

    ```bash
    llama-quantize --imatrix imat-task.gguf --tensor-type 'blk\.32\.=q8_0' f32.gguf out.gguf IQ2_XXS
    ```

    ### Snag 3: thinking mode in LM Studio

    Imported into LM Studio, the IQ4_XS quant got **{_t['thinking_on']}/{_t['n']}** test items right through
    the chat API against {_t['llama_server_raw_prompt']}/{_t['n']} through `llama-server` with the raw prompt.
    Qwen3.5's chat template opens a `&lt;think&gt;` block unless `enable_thinking` is false, and LM Studio passes
    `enable_thinking=true` from its own setting, ignoring a request's `chat_template_kwargs`. The fine-tune
    only ever saw the empty `&lt;think&gt;&lt;/think&gt;` format (§6: the template is part of the model), so it wrote a
    regex inside the reasoning and another after it. Per request, `"reasoning_effort": "none"` works. For every
    client, `lab9b.sh` writes the pick with `{{%- set enable_thinking = false %}}` prepended to its template
    (`gguf_new_metadata.py --chat-template`): **{_t['thinking_forced_off']}/{_t['n']}**, with 46 of 50 answers
    identical to `llama-server`'s (the rest: LM Studio's own llama.cpp build).
    """)
    return


@app.cell
def _(mo):
    budget9 = mo.ui.slider(3, 12, step=0.25, value=5.5, label="file-size budget (GB)", show_value=True)
    tol9 = mo.ui.slider(0, 0.25, step=0.01, value=0.02, label="accuracy you'll give up (absolute)", show_value=True)
    mo.vstack([mo.md("**Your turn:** `pick_quant`: the smallest quant that fits a budget and scores within a "
                     "tolerance of the best. Move the sliders."), budget9, tol9])
    return budget9, tol9


@app.cell
def _(budget9, ex, lj, mo, q9, tol9):
    _r = {k: {"size_gb": v["size_gb"], "exact_match": v["exact_match"]} for k, v in q9["quants"].items()}
    _a = lj.attempt(ex.pick_quant, _r, budget9.value, tol9.value)
    if _a.ok:
        _v = _a.value
        _out = mo.md(f"Pick: **{_v}** ({_r[_v]['size_gb']:.2f} GB, {_r[_v]['exact_match']:.1%})" if _v
                     else "Nothing fits within that tolerance: raise the budget or the tolerance.")
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(BASE, lj, lmclient, mo):
    def _live9():
        key = "learnjax/qwen3.5-9b-nlrx-iq4_xs"
        if not BASE or not any(k.startswith(key) for k in lmclient.available(BASE)):
            return mo.callout(mo.md("Your 9B isn't in LM Studio: `make quant-lab-9b` imports it when the server is "
                                    "up (`make lms-up`). Then try your own descriptions here."), kind="info")
        lmclient.ensure_loaded(BASE, key)
        asks = ["lines that start with a vowel and end with the string 'dog'",
                "lines containing the word 'cat' followed by a number", "lines with at least 3 capital letters"]
        rows = []
        for q in asks:
            r = lmclient.post(BASE, "/v1/chat/completions", {"model": key, "temperature": 0, "max_tokens": 96,
                              "messages": [{"role": "system", "content": "Convert natural language to regular expression."},
                                           {"role": "user", "content": q}]})
            rows.append(f"| {q} | `{r['choices'][0]['message']['content'].strip().replace('|', '&#124;')}` |")
        lmclient.unload_all(BASE)
        return mo.md("| you ask | your 9B, IQ4_XS in LM Studio |\n|---|---|\n" + "\n".join(rows))

    _a = lj.attempt(_live9)
    _a.value if _a.ok else _a.view(mo)
    return


@app.cell
def _(mo):
    quiz16 = mo.ui.radio(
        options={"The task imatrix: it weights the error toward the activations regex prompts produce, which is what this model is for": "a",
                 "The generic imatrix: its perplexity is lower, so it's the better model": "b",
                 "Either: an imatrix only matters for i-quants, not for accuracy": "c"},
        label="**Check yourself:** you must ship this fine-tune at IQ2_XXS. Which imatrix?")
    quiz16
    return (quiz16,)


@app.cell
def _(mo, quiz16):
    _fb = {"a": mo.callout(mo.md("Right: in the reference run, 25 % vs 8 % on the task. Perplexity on prose measures a skill this model "
                                 "isn't for. (Better still: don't ship IQ2_XXS; IQ3_XXS keeps 41 %.)"), kind="success"),
           "b": mo.callout(mo.md("Lower perplexity on *prose*, three times worse on the task it was fine-tuned for. "
                                 "Measure the task."), kind="warn"),
           "c": mo.callout(mo.md("At 4–5 bits it barely matters, but at 2–3 bits it decides the score: 8 % vs 25 % "
                                 "in the reference run."), kind="warn")}
    _fb.get(quiz16.value, mo.md(""))
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"ΔW is far smaller than W, and rounding W+ΔW to bfloat16 (8 significant bits) adds error that is a sizable fraction of ΔW": "a",
                 "LM Studio can't read float32 files": "b",
                 "Merging in bfloat16 makes the GGUF file larger": "c"},
        label="**Check yourself:** why does `export.py` merge the LoRA into the weights in float32?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {"a": mo.callout(mo.md("Right — §3 measured it: bf16 rounding error was ~12 % of the update's size vs ~1.6 % "
                                 "for f16. Merge in f32, then store f16 or quantize."), kind="success"),
           "b": mo.callout(mo.md("No — the converter reads float32 safetensors fine (that's what export.py writes)."), kind="warn"),
           "c": mo.callout(mo.md("No — file size depends on the *stored* type, not the type used while merging."), kind="warn")}
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(27))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * LM Studio docs: [lms CLI](https://lmstudio.ai/docs/cli), [headless llmster](https://lmstudio.ai/docs/developer/core/headless),
      [REST API](https://lmstudio.ai/docs/developer/rest) and
      [prompt templates](https://lmstudio.ai/docs/app/advanced/prompt-template).
    * ggml-org, [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and
      [llama.cpp](https://github.com/ggml-org/llama.cpp) (`convert_hf_to_gguf.py`, `tools/quantize`).
    * Hugging Face docs: [safetensors](https://huggingface.co/docs/safetensors) and
      [chat templates](https://huggingface.co/docs/transformers/chat_templating).
    * Hu et al., [LoRA](https://arxiv.org/abs/2106.09685) §4.2 — merging the adapter for zero-latency inference.
    * ggml-org, [llama.cpp imatrix README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)
      and the [quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
      (every type, `--tensor-type` overrides).

    **Next:** that's the course. The capstone in Module 25 strings it all together — and now you can ship the
    result to LM Studio.
    """)
    return


if __name__ == "__main__":
    app.run()
