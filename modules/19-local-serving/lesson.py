import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="19 · Local serving")


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import time
    import urllib.request
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    return Path, ex, json, lj, mo, os, plt, subprocess, sys, tempfile, time, urllib


@app.cell
def _(mo):
    mo.md(r"""
    # 19 · Local serving for coding assistants

    **Goal:** run a capable coding model on your own RX 7600 XT behind the same HTTP API the cloud
    providers use, know what fits in 16 GB and how fast it will be, and point a coding tool at it.

    **You will:** read what's inside a quantized GGUF file · work out what fits in 16 GB (weights + KV
    cache, and what happens when it doesn't) · measure prompt and generation speed on both of llama.cpp's
    GPU backends (Vulkan and ROCm/HIP) and explain the numbers from memory bandwidth · call the
    OpenAI-compatible API, with streaming · build a tiny coding agent that writes code, runs the tests, and
    fixes its own mistakes · connect real tools.

    Modules 16–18 ran models through JAX, which is ideal for *training* and for understanding. For
    *serving* a 14B model on one consumer card, a specialised inference engine wins: llama.cpp is C/C++,
    loads quantized GGUF files, and runs on almost any GPU.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The setup

    ```text
    your editor / aider / curl ──► http://127.0.0.1:8080  (host only)
                                          │
                                 ┌────────┴─────────┐   podman network "learn-jax-net"
    this notebook ──────────────►│  llama-server     │◄── http://llama:8080
                                 │  (learn-jax-llama)│
                                 └────────┬─────────┘
                                          ▼  /dev/dri (+ /dev/kfd for ROCm)
                                     RX 7600 XT
    ```

    ```bash
    containers/llamacpp.sh download                         # ~12 GB of GGUF models, once
    containers/llamacpp.sh start vulkan                     # Qwen2.5-Coder-14B Q4_K_M on Vulkan
    containers/llamacpp.sh start rocm qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
    containers/llamacpp.sh stop
    ```

    | Backend | What it needs | Image | Notes on this card |
    |---|---|---|---|
    | **Vulkan** | only `/dev/dri` + Mesa's RADV driver — no ROCm at all | 0.9 GB | flash attention unsupported here, disabled automatically |
    | **ROCm / HIP** | `/dev/kfd` + `/dev/dri`, ROCm 7.2.1 runtime in the image | 23 GB | gfx1102 is a native build target — **no** `HSA_OVERRIDE_GFX_VERSION` needed |

    Both are ggml-org's official images, pinned by digest in `containers/llamacpp.sh`. The server
    publishes **only on 127.0.0.1**: there is no API key, so anything that can reach the port can use your
    GPU. This notebook reaches it over a private podman network instead (rootless podman's
    `host.containers.internal` points at your LAN address, not loopback, so a loopback-only port isn't
    reachable that way — by design).
    """)
    return


@app.cell
def _(Path, mo):
    GGUF_DIR = Path("/cache/gguf")
    COURSE_MODELS = {
        # file: (params, layers, kv_heads, head_dim, label)
        "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf": (1.54e9, 28, 2, 128, "Coder 1.5B Q4_K_M"),
        "qwen2.5-coder-1.5b-instruct-q8_0.gguf": (1.54e9, 28, 2, 128, "Coder 1.5B Q8_0"),
        "qwen2.5-coder-14b-instruct-q4_k_m.gguf": (14.77e9, 48, 8, 128, "Coder 14B Q4_K_M"),
    }
    _rows = []
    for _f, (_p, _l, _kv, _hd, _label) in COURSE_MODELS.items():
        _path = GGUF_DIR / _f
        if _path.exists():
            _size = _path.stat().st_size
            _rows.append(f"| {_label} | {_p / 1e9:.2f}B | {_size / 2**30:.2f} GiB | **{_size * 8 / _p:.2f}** |")
        else:
            _rows.append(f"| {_label} | {_p / 1e9:.2f}B | not downloaded | — |")
    _table = "\n".join(_rows)
    mo.md(f"""
## 2 · What's in a GGUF file

A GGUF file is one self-contained model: a header of metadata (architecture, layer count, tokenizer
vocabulary and chat template) followed by every weight tensor, most of them **quantized**. Module 18
quantized weights to int8 with one scale per row; llama.cpp's formats do the same thing in small
**blocks**:

* **Q8_0** — blocks of 32 weights as int8 plus one f16 scale: (32·8 + 16)/32 = **8.5 bits/weight**.
* **Q4_K_M** — "k-quants": super-blocks of 256 weights in 4-bit groups with their own 6-bit scales and
  minimums; the *M* mix keeps a few sensitive tensors (parts of attention and the output) at higher
  precision. Effective ≈ **4.8–4.9 bits/weight**.

Divide each file's size by its parameter count and you see the effective bits per weight directly:

| Model | Params | File size | Bits / weight |
|---|---|---|---|
{_table}

Q4_K_M is the usual sweet spot: about 3.3× smaller than bf16 with a small quality loss. Below ~4 bits
quality falls off faster, especially for small models — a 1.5B model has less redundancy to lose.
""")
    return COURSE_MODELS, GGUF_DIR


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · What fits in 16 GB

    Serving memory is **weights + KV cache + overhead**. The KV cache (Module 17–18) grows with context:
    each layer stores a key and a value per KV head per token —
    $\text{bytes} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot T \cdot \text{bytes per element}$.
    Grouped-query attention is what makes long contexts affordable: Qwen2.5-Coder-14B has 40 query heads
    but caches only 8.

    When a model doesn't fit, llama.cpp can keep some layers on the CPU (`-ngl N` puts N layers on the
    GPU). It works, but every token then waits on the much slower system memory and CPU for those
    layers. Use the sliders — the calculator is built from your exercises below.
    """)
    return


@app.cell
def _(mo):
    calc_params = mo.ui.slider(start=0.5, stop=70, step=0.5, value=14.5, label="parameters (billions)",
                               show_value=True)
    calc_bits = mo.ui.dropdown(options={"Q4_K_M (4.85)": 4.85, "Q5_K_M (5.7)": 5.7, "Q8_0 (8.5)": 8.5,
                                        "f16 (16)": 16.0}, value="Q4_K_M (4.85)", label="quantization")
    calc_ctx = mo.ui.slider(steps=[2048, 4096, 8192, 16384, 32768, 65536, 131072], value=8192,
                            label="context (tokens)", show_value=True)
    mo.vstack([calc_params, calc_bits, calc_ctx])
    return calc_bits, calc_ctx, calc_params


@app.cell
def _(calc_bits, calc_ctx, calc_params, ex, lj, mo):
    # Rough shape for a modern GQA model of this size: ~48 layers and 8 KV heads × 128 at 14B,
    # scaled with size (good enough for planning; real models are listed in their config).
    _p = calc_params.value * 1e9
    _layers = max(16, round(48 * (calc_params.value / 14.5) ** 0.35))
    _w = lj.attempt(ex.gguf_weight_bytes, _p, calc_bits.value)
    _kv = lj.attempt(ex.kv_cache_bytes, _layers, 8, 128, calc_ctx.value)
    if _w.ok and _kv.ok:
        _fits = lj.attempt(ex.fits_in_vram, _w.value, _kv.value, 16)
        _ngl = lj.attempt(ex.gpu_layers, _layers, _w.value, _kv.value, 16)
        _verdict = ""
        if _fits.ok and _ngl.ok:
            _verdict = ("✅ **fits entirely on the GPU**" if _fits.value else
                        f"❌ **doesn't fit** — about **{_ngl.value} of {_layers}** layers on the GPU (`-ngl {_ngl.value}`), "
                        "the rest on the CPU, and generation slows sharply")
        _out = mo.md(f"""
| | GiB |
|---|---|
| weights | {_w.value / 2**30:.1f} |
| KV cache ({_layers} layers × 8 KV heads × 128, f16, {calc_ctx.value:,} tokens) | {_kv.value / 2**30:.1f} |
| overhead (compute buffers, runtime) | 1.0 |
| **total** | **{(_w.value + _kv.value) / 2**30 + 1:.1f}** of 16 |

{_verdict}
""")
    else:
        _out = mo.vstack([_w.view(mo), _kv.view(mo)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Rules of thumb from the calculator on a 16 GB card: a **14B model at Q4_K_M with 8–16K context** is
    the comfortable maximum; ~32B dense models don't fit at any useful quantization. Mixture-of-experts
    models change the arithmetic (Module 17): a 30B-total / 3B-active MoE at Q4 is ~18 GB — too big for
    VRAM — but because each token only touches 3B parameters, llama.cpp can keep the *experts* in system
    RAM (`--n-cpu-moe N`) and still generate at a useful speed. That's the route to bigger coding models
    on this card later.
    """)
    return


@app.cell
def _(json, os, urllib):
    def _get(url, timeout=1.5):
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())

    def find_server():
        """First llama.cpp server that answers /health, or None."""
        cands = [os.environ.get("LLAMA_URL"), "http://llama:8080", "http://127.0.0.1:8080",
                 "http://host.containers.internal:8080"]
        for base in [c for c in cands if c]:
            try:
                if _get(base + "/health").get("status") == "ok":
                    return base
            except Exception:  # noqa: BLE001 - any failure means "not here"
                continue
        return None

    def post(base, path, body, timeout=300):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    SERVER = find_server()
    MODEL_ID = None
    if SERVER:
        try:
            MODEL_ID = _get(SERVER + "/v1/models")["data"][0]["id"]
        except Exception:  # noqa: BLE001
            MODEL_ID = "local"
    return MODEL_ID, SERVER, post


@app.cell
def _(MODEL_ID, SERVER, mo):
    START_HINT = mo.callout(mo.md(
        "**The llama.cpp server isn't reachable from this notebook.** On the host:\n\n"
        "```bash\ncontainers/llamacpp.sh download     # once, ~12 GB\ncontainers/llamacpp.sh start vulkan\n```\n\n"
        "and run this notebook's container on the server's network (`NETWORK=learn-jax-net`, see the "
        "README for the make target), then re-run this cell. Everything below that needs the server "
        "will wait for it; the rest of the lesson works without it."), kind="info")
    _status = (mo.callout(mo.md(f"**Server found** at `{SERVER}` serving `{MODEL_ID}`."), kind="success")
               if SERVER else START_HINT)
    mo.vstack([mo.md("## 4 · Is the server up?"), _status])
    return (START_HINT,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · How fast? Prompt processing vs generation

    Serving has two phases with very different limits (Module 18):

    * **Prompt processing ("prefill")** — all prompt tokens go through the model at once, as big matrix
      multiplies. **Compute-bound**: hundreds (14B) to thousands (1.5B) of tokens per second.
    * **Generation ("decode")** — one token at a time; to make each token the GPU must read **every
      weight** from memory once. **Memory-bandwidth-bound**: the ceiling is roughly
      $\text{bandwidth} / \text{model bytes}$. The RX 7600 XT has a 128-bit GDDR6 bus at 18 Gbps =
      **288 GB/s**, so a 9 GB model can't generate faster than ~32 tokens/s no matter how fast the math
      units are. That's why quantization speeds up generation almost in proportion to the bytes saved.
    """)
    return


@app.cell
def _(Path, json):
    BENCH_FILE = Path("/course/runs/19-local-serving/bench.json")
    # Measured when this lesson was written (RX 7600 XT, idle GPU, llama.cpp b12375, 128 generated tokens).
    # Re-measure on your machine with:  python3 modules/19-local-serving/bench.py
    WRITTEN = {"measured": "2026-11-14", "results": [
        {"backend": "vulkan", "model": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "prompt_tokens": 512, "prompt_tok_s": 2133.8, "gen_tok_s": 183.1},
        {"backend": "vulkan", "model": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "prompt_tokens": 4096, "prompt_tok_s": 3294.0, "gen_tok_s": 171.0},
        {"backend": "vulkan", "model": "qwen2.5-coder-1.5b-instruct-q8_0.gguf", "prompt_tokens": 512, "prompt_tok_s": 2632.0, "gen_tok_s": 132.9},
        {"backend": "vulkan", "model": "qwen2.5-coder-1.5b-instruct-q8_0.gguf", "prompt_tokens": 4096, "prompt_tok_s": 3485.6, "gen_tok_s": 125.5},
        {"backend": "vulkan", "model": "qwen2.5-coder-14b-instruct-q4_k_m.gguf", "prompt_tokens": 512, "prompt_tok_s": 495.5, "gen_tok_s": 29.0},
        {"backend": "vulkan", "model": "qwen2.5-coder-14b-instruct-q4_k_m.gguf", "prompt_tokens": 4096, "prompt_tok_s": 522.2, "gen_tok_s": 26.7},
        {"backend": "rocm", "model": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "prompt_tokens": 512, "prompt_tok_s": 3738.2, "gen_tok_s": 163.4},
        {"backend": "rocm", "model": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "prompt_tokens": 4096, "prompt_tok_s": 5122.9, "gen_tok_s": 154.7},
        {"backend": "rocm", "model": "qwen2.5-coder-1.5b-instruct-q8_0.gguf", "prompt_tokens": 512, "prompt_tok_s": 5224.9, "gen_tok_s": 103.0},
        {"backend": "rocm", "model": "qwen2.5-coder-1.5b-instruct-q8_0.gguf", "prompt_tokens": 4096, "prompt_tok_s": 5327.5, "gen_tok_s": 100.7},
        {"backend": "rocm", "model": "qwen2.5-coder-14b-instruct-q4_k_m.gguf", "prompt_tokens": 512, "prompt_tok_s": 647.6, "gen_tok_s": 27.0},
        {"backend": "rocm", "model": "qwen2.5-coder-14b-instruct-q4_k_m.gguf", "prompt_tokens": 4096, "prompt_tok_s": 630.1, "gen_tok_s": 23.0}
    ]}
    bench = json.loads(BENCH_FILE.read_text()) if BENCH_FILE.exists() else WRITTEN
    return BENCH_FILE, bench


@app.cell
def _(BENCH_FILE, COURSE_MODELS, GGUF_DIR, bench, mo, plt):
    _res = bench["results"]
    if not _res:
        _out = mo.callout(mo.md("No benchmark results yet. Run `python3 modules/19-local-serving/bench.py` "
                                "on the host (~10 min; it restarts the server for each configuration)."),
                          kind="info")
    else:
        _bw = 288e9
        _rows = []
        for _r in _res:
            _f = GGUF_DIR / _r["model"]
            _bytes = _f.stat().st_size if _f.exists() else None
            _ceiling = f"{_bw / _bytes:.0f}" if _bytes else "?"
            _eff = f"{_r['gen_tok_s'] * _bytes / _bw:.0%}" if _bytes else "?"
            _rows.append(f"| {_r['backend']} | {COURSE_MODELS[_r['model']][4]} | {_r['prompt_tokens']:,} | "
                         f"{_r['prompt_tok_s']:,.0f} | {_r['gen_tok_s']:.1f} | {_ceiling} | {_eff} |")
        _src = "your `bench.json`" if BENCH_FILE.exists() else "measurements taken when this lesson was written"
        _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.4))
        _labels = sorted({(r["model"], r["backend"]) for r in _res}, key=lambda t: (COURSE_MODELS[t[0]][0], t[0], t[1]))
        for _ax, _key, _title in [(_axes[0], "prompt_tok_s", "prompt processing (tok/s)"),
                                  (_axes[1], "gen_tok_s", "generation (tok/s)")]:
            for _i, (_m, _b) in enumerate(_labels):
                _vals = [r[_key] for r in _res if r["model"] == _m and r["backend"] == _b]
                _ax.barh(_i, max(_vals), color="C0" if _b == "vulkan" else "C3")
                _ax.text(max(_vals), _i, f" {max(_vals):,.0f}", va="center", fontsize=8)
            _ax.set_yticks(range(len(_labels)), [f"{COURSE_MODELS[m][4]} · {b}" for m, b in _labels], fontsize=8)
            _ax.set_title(_title, fontsize=10)
            _ax.margins(x=0.18)
        _fig.tight_layout()
        _table = "\n".join(_rows)
        _out = mo.vstack([mo.md(f"""
From {_src} ({bench['measured']}):

| backend | model | prompt tokens | prompt tok/s | generate tok/s | bandwidth ceiling | % of ceiling |
|---|---|---|---|---|---|---|
{_table}
"""), _fig])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    What to look for in the table (idle RX 7600 XT, llama.cpp b12375):

    * **Generation is a memory-bandwidth problem.** The 14B Q4_K_M generates at ~27–29 tok/s against a
      ceiling of ~32 — about **90% of the memory bus**. The 1.5B Q4 file gets only 60–70%: at
      ~180 tok/s each token takes ~5 ms, and fixed per-token costs (kernel launches, sampling, HTTP)
      become a big share.
    * **Quantization buys generation speed.** Q8_0 of the same 1.5B model is 1.7× bigger and generates
      1.4–1.6× slower. For small models overheads blunt the effect; for big ones it's nearly proportional.
    * **The backends trade wins.** ROCm/HIP is **1.2–2× faster at prompt processing** (better matrix
      kernels — it matters for long files pasted into the context); Vulkan is **7–29% faster at
      generation** (most for the small models) and needs no ROCm install at all. For an interactive coding assistant, where you read
      tokens as they stream, either is fine; for big prompts, prefer ROCm.
    * **Longer prompts slow generation a little** (5–15% slower at 4K tokens than at 512): each new token attends over
      the whole KV cache, and Vulkan has no flash attention on this card.
    """)
    return


@app.cell
def _(SERVER, mo):
    live_len = mo.ui.slider(steps=[128, 512, 2048, 4096], value=512, label="prompt length for a live measurement",
                            show_value=True)
    live_len if SERVER else mo.md("")
    return (live_len,)


@app.cell
def _(Path, SERVER, START_HINT, live_len, mo, post):
    if not SERVER:
        _out = START_HINT
    else:
        _code = Path("/course/learnjax/llama.py").read_text()
        _ids = post(SERVER, "/tokenize", {"content": _code})["tokens"]
        _ids = (_ids * (live_len.value // len(_ids) + 1))[: live_len.value]
        _t = post(SERVER, "/completion", {"prompt": _ids, "n_predict": 64, "cache_prompt": False,
                                          "temperature": 0, "ignore_eos": True})["timings"]
        _out = mo.md(f"**Live, on the running server:** {_t['prompt_n']} prompt tokens at "
                     f"**{_t['prompt_per_second']:,.0f} tok/s**, then {_t['predicted_n']} generated at "
                     f"**{_t['predicted_per_second']:.1f} tok/s** (llama.cpp's own timings).")
    _out
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Generation reads every weight from memory for each token; prefill reuses each weight across many tokens": "a",
            "The GPU runs at a lower clock during generation": "b",
            "Generation uses the CPU while prompt processing uses the GPU": "c",
        },
        label="**Check yourself:** why is generation ~20–30× slower per token than prompt processing?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right. Prefill does a big matrix multiply per weight load (compute-bound); "
                              "decode does a matrix–*vector* multiply per weight load (bandwidth-bound)."),
                        kind="success"),
        "b": mo.callout(mo.md("No — the clock is the same. The difference is arithmetic intensity: how much "
                              "math you do per byte read from memory."), kind="warn"),
        "c": mo.callout(mo.md("No — with `-ngl 999` both phases run entirely on the GPU."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · The OpenAI-compatible API

    llama-server speaks the same `POST /v1/chat/completions` protocol as OpenAI's API (and most other
    providers), which is why almost every coding tool can use it unchanged. A request is a JSON body with
    the model name and a list of messages; with `"stream": true` the reply arrives as **Server-Sent
    Events** — lines of `data: {...}`, each carrying the next few characters — so a UI can show tokens as
    they're generated. Your exercises build the request and parse the stream.
    """)
    return


@app.cell
def _(ex, lj, mo):
    _req = lj.attempt(ex.build_chat_request,
                      [{"role": "system", "content": "You are a concise senior Python developer."},
                       {"role": "user", "content": "One-line Python to reverse a string s?"}],
                      "local", temperature=0.0, max_tokens=64)
    _sse = lj.attempt(ex.parse_sse_stream, ['data: {"choices":[{"delta":{"content":"s[::"}}]}', "",
                                            'data: {"choices":[{"delta":{"content":"-1]"}}]}', "data: [DONE]"])
    mo.vstack([mo.md("**Your `build_chat_request`:**"), _req.view(mo),
               mo.md("**Your `parse_sse_stream` on a recorded stream** — should print `s[::-1]`:"), _sse.view(mo)])
    return


@app.cell
def _(MODEL_ID, SERVER, START_HINT, ex, json, lj, mo, time, urllib):
    def stream_chat(messages, max_tokens=200):
        """POST a streaming chat request; return (text, seconds to first token, total seconds)."""
        body = ex.build_chat_request(messages, MODEL_ID, temperature=0.0, max_tokens=max_tokens, stream=True)
        req = urllib.request.Request(SERVER + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0, first, lines = time.perf_counter(), None, []
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode()
                if first is None and '"content"' in line:
                    first = time.perf_counter() - t0
                lines.append(line)
        return ex.parse_sse_stream(lines), first, time.perf_counter() - t0

    _needs = [lj.attempt(ex.build_chat_request, [{"role": "user", "content": "x"}], "m"),
              lj.attempt(ex.parse_sse_stream, [])]
    _todo = next((n for n in _needs if not n.ok), None)
    if not SERVER:
        _out = START_HINT
    elif _todo is not None:
        _out = _todo.view(mo)
    else:
        _a = lj.attempt(stream_chat, [{"role": "user", "content":
                                       "Write a Python function `is_palindrome(s)` that ignores case and "
                                       "non-alphanumerics. Code only."}])
        if _a.ok:
            _text, _first, _total = _a.value
            _out = mo.vstack([mo.md(f"**Streamed reply** — first token after **{_first:.2f} s**, "
                                    f"complete after **{_total:.1f} s**:"), mo.md(_text)])
        else:
            _out = _a.view(mo)
    _out
    return (stream_chat,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · A tiny coding agent

    "Agentic" coding tools are, at heart, a loop: ask the model for code, **run it**, and feed the
    result back. The tests are the ground truth — the model's confidence is not. Below: ask for a
    function, pull the code out of the markdown reply (`extract_code_block`), run hidden tests in a
    subprocess with a timeout, and if they fail, send the error back once.
    """)
    return


@app.cell
def _(ex, lj, mo, subprocess, sys, tempfile):
    TASK = ("Write a Python function `slugify(title: str) -> str` that lowercases, replaces every run of "
            "characters that are not a-z or 0-9 with a single hyphen, and strips hyphens from both ends. "
            "Reply with one ```python code block only.")
    TESTS = """
assert slugify("Hello, World!") == "hello-world"
assert slugify("  --Already--Slugged--  ") == "already-slugged"
assert slugify("Café au lait") == "caf-au-lait"
assert slugify("") == ""
assert slugify("A1 B2") == "a1-b2"
print("all tests passed")
"""

    def run_tests(code: str) -> tuple[bool, str]:
        """Run the model's code plus TESTS in a fresh Python process (never exec untrusted code in-process)."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code + "\n" + TESTS)
        try:
            r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=10, check=False)
        except subprocess.TimeoutExpired:
            return False, "timed out after 10 s"
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-600:]

    _ok_example = lj.attempt(ex.extract_code_block, "Sure!\n```python\ndef f():\n    return 1\n```", "python")
    mo.vstack([mo.md("**Your `extract_code_block`** on a sample reply — should be the function body only:"),
               _ok_example.view(mo)])
    return TASK, run_tests


@app.cell
def _(MODEL_ID, SERVER, START_HINT, TASK, ex, lj, mo, post, run_tests):
    def agent(max_rounds=2):
        messages = [{"role": "user", "content": TASK}]
        log = []
        for round_ in range(1, max_rounds + 1):
            reply = post(SERVER, "/v1/chat/completions",
                         ex.build_chat_request(messages, MODEL_ID, temperature=0.2, max_tokens=400))
            text = reply["choices"][0]["message"]["content"]
            code = ex.extract_code_block(text, "python") or text
            ok, output = run_tests(code)
            log.append((round_, code, ok, output))
            if ok:
                break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"The tests failed:\n```\n{output}\n```\nFix the function. "
                                                     "Reply with one ```python code block only."}]
        return log

    _needs = [lj.attempt(ex.build_chat_request, [{"role": "user", "content": "x"}], "m"),
              lj.attempt(ex.extract_code_block, "```python\nx\n```", "python")]
    _todo = next((n for n in _needs if not n.ok), None)
    if not SERVER:
        _out = START_HINT
    elif _todo is not None:
        _out = _todo.view(mo)
    else:
        _a = lj.attempt(agent)
        if _a.ok:
            _parts = []
            for _round, _code, _ok, _output in _a.value:
                _parts.append(mo.md(f"**Round {_round}: {'✅ tests pass' if _ok else '❌ tests fail'}**"))
                _parts.append(mo.md(f"```python\n{_code}\n```"))
                _parts.append(mo.md(f"```text\n{_output}\n```"))
            _out = mo.vstack(_parts)
        else:
            _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    When this lesson was written, **Qwen2.5-Coder-14B passed on the first round**, while the **1.5B
    failed both rounds** — even with the failing test fed back, it couldn't fix its regex. Try it:
    `containers/llamacpp.sh start vulkan qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` and re-run the cell.
    Model size matters most for exactly this: following a spec precisely and using feedback. The loop
    itself — run the code, show the model the result — is what agentic tools are built on; real tools let
    the model read files, run commands and see the output, under permissions you control.

    ## 8 · Pointing real tools at your server

    Anything that speaks the OpenAI API works with **base URL `http://127.0.0.1:8080/v1`**, any API key
    (the server doesn't check one unless you start it with `--api-key`), and the model name the server
    reports at `/v1/models` (the `--alias`, e.g. `qwen2.5-coder-14b-instruct-q4_k_m`). For example:

    ```bash
    # any OpenAI SDK / CLI: environment variables
    export OPENAI_BASE_URL=http://127.0.0.1:8080/v1   OPENAI_API_KEY=local

    # aider (OpenAI-compatible endpoint)
    aider --openai-api-base http://127.0.0.1:8080/v1 --openai-api-key local \
          --model openai/qwen2.5-coder-14b-instruct-q4_k_m

    # curl
    curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
         -d '{"model":"local","messages":[{"role":"user","content":"hi"}]}'
    ```

    Editor extensions (Continue, and others) and terminal agents (opencode, …) have an
    "OpenAI-compatible provider" setting that takes the same three values; their config file formats
    change between versions, so check the tool's current docs. Practical tips: give the tool a context
    window that matches the server's `-c` (default 8192 here; raise it with `CTX=16384
    containers/llamacpp.sh start vulkan`, see the calculator for the memory cost), and expect a 14B
    local model to be good at completions, explanations and small focused edits, and weaker than frontier
    models at long multi-file agentic tasks.
    """)
    return


@app.cell
def _(ex, lj, mo):
    _w = lj.attempt(ex.gguf_weight_bytes, 14.77e9, 4.85)
    _kv = lj.attempt(ex.kv_cache_bytes, 48, 8, 128, 8192)
    _items = [mo.md("## 9 · Your turn: the numbers behind the server"),
              mo.md("**`gguf_weight_bytes(14.77e9, 4.85)`** — Qwen2.5-Coder-14B at Q4_K_M, ~8.3 GiB:"), _w.view(mo),
              mo.md("**`kv_cache_bytes(48, 8, 128, 8192)`** — its f16 KV cache at 8K context, ~1.5 GiB:"), _kv.view(mo)]
    if _w.ok and _kv.ok:
        _fit = lj.attempt(ex.fits_in_vram, _w.value, _kv.value, 16)
        _ngl = lj.attempt(ex.gpu_layers, 48, _w.value, ex.kv_cache_bytes(48, 8, 128, 131072), 16)
        _items += [mo.md("**`fits_in_vram`** (should be True):"), _fit.view(mo),
                   mo.md("**`gpu_layers`** at a 128K context (KV cache alone ~24 GiB) — how many layers stay on the GPU:"),
                   _ngl.view(mo)]
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(19))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * ggml-org, [llama.cpp server README](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) — every
      flag and endpoint (`/completion`, `/v1/chat/completions`, `/tokenize`, `/metrics`).
    * ggml-org, [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
    * Iwan Kawrakow, [k-quants pull request](https://github.com/ggml-org/llama.cpp/pull/1684) — how Q4_K and friends are
      laid out, with perplexity/size trade-offs.
    * Hui et al. (Qwen Team), [Qwen2.5-Coder technical report](https://arxiv.org/abs/2409.12186) — the models used here.
    * OpenAI, [Chat Completions API reference](https://platform.openai.com/docs/api-reference/chat) — the protocol
      llama-server implements.

    **Next:** Module 20 — fine-tuning with LoRA: teach a model new behaviour by training a few small
    matrices instead of all its weights.
    """)
    return


if __name__ == "__main__":
    app.run()
