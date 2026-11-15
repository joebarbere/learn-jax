import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="22 · Profiling")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import os

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    _spec = importlib.util.spec_from_file_location("m22_ladder", mo.notebook_dir() / "ladder.py")
    L = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(L)
    L.enable_compilation_cache()  # the ladder compiles ~7 programs; cache them across runs
    g, g_note = L.load_gpt()
    return L, ex, g, g_note, jax, jnp, lj, mo, np, optax, os, plt


@app.cell
def _(g_note, mo):
    mo.md(rf"""
    # 22 · Profiling JAX on ROCm

    **Goal:** make a training step fast *on purpose*: measure, find where the time goes, change one
    thing, measure again. Then learn the tools that tell you what to change.

    **You will:** climb a measured **speed ladder** on {g_note}, from a naive eager loop to a
    compiled, mixed-precision, donated, scanned step · capture a profiler trace and read it
    (top kernels, the gaps between steps) · read compiled HLO to find fusions and matmuls · place every
    kernel on the RX 7600 XT's **roofline** · see what donation does to memory · and finally take
    apart the Triton-GEMM failure from Module 00, without crashing your GPU.

    This module follows the spirit of Karpathy's *Let's reproduce GPT-2 (124M)*, which climbs the
    same kind of ladder in PyTorch on NVIDIA GPUs. Here every rung is JAX on an AMD card, measured.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Performance engineering for deep learning grew with GPUs. The roofline model (Williams, Waterman &
    Patterson, 2009) framed compute- versus memory-bound code; Ivanov et al. (2020) showed Transformers
    spend much of their time moving data; and Google's PaLM paper (Chowdhery et al., 2022) popularized model
    FLOPs utilization (MFU). Karpathy's "Let's reproduce GPT-2" (2024) walked through the speed-up ladder
    this lesson rebuilds in JAX.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(L, mo):
    mo.md(rf"""
    ## 1 · The speed ladder

    The model is your Module 16 GPT, sized like a small GPT-2: **{L.N_LAYERS} layers, d = {L.D_MODEL},
    {L.N_HEADS} heads, {L.SEQ}-token context, GPT-2's {L.VOCAB:,}-token vocabulary**, batch {L.BATCH}.
    The data is random tokens (speed doesn't care what they say). Each rung adds one change and is
    timed on the same model, back to back:

    | Rung | Change | Why it should help |
    |---|---|---|
    | 0 | eager float32, read the loss every step | the baseline: every op is its own dispatch |
    | 1 | [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit) the whole step | one compiled program, fused elementwise ops (Module 06) |
    | 2 | bf16 compute, f32 master weights | the card's matrix units run bf16 ~3× faster (Modules 00, 08) |
    | 3 | "nice numbers": vocab 50,257 → 50,304 | GEMM kernels like dimensions divisible by 64/128 |
    | 4 | donate params & optimizer state | update in place instead of allocating new buffers |
    | 5 | deferred metrics | stop forcing a GPU→CPU sync every step (the "step bubble") |
    | 6 | [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan) over 4 steps | one dispatch runs 4 steps: less Python per step |

    Karpathy's ladder has the same skeleton (TF32 → bf16 autocast → `torch.compile` → **flash
    attention** → nice numbers → fused AdamW). Two of his rungs have no equivalent here: XLA already
    fuses the optimizer (rung 1 covers "fused AdamW"), and **JAX has no fused attention kernel on
    ROCm** (Module 15). That missing rung is what Module 23 is about.

    *The next cell runs the whole ladder (~1 min warm; several minutes the very first time, almost all
    of it compilation — the persistent cache in `/cache/jaxcache` makes later runs fast).*
    """)
    return


@app.cell
def _(L, g):
    ladder = [L.run_rung(g, _key) | {"label": _label} for _key, _label in L.RUNGS]
    return (ladder,)


@app.cell
def _(L, ex, ladder, lj, mo, plt):
    _mfu = lj.attempt(ex.mfu, ladder[-1]["tok_s"], ladder[-1]["n_params"], L.PEAK_BF16)
    _rows = []
    for _r in ladder:
        _m = lj.attempt(ex.mfu, _r["tok_s"], _r["n_params"], L.PEAK_BF16)
        _mfu_txt = f"{100 * _m.value:.1f}%" if _m.ok else "—"
        _rows.append(f"| {_r['label']} | {_r['tok_s']:,.0f} | {_r['ms_step']:.1f} | "
                     f"{_r['tok_s'] / ladder[0]['tok_s']:.1f}× | {_mfu_txt} | {_r['first_call_s']:.1f} s |")
    _fig, _ax = plt.subplots(figsize=(8, 3.2))
    _ax.barh([_r["label"] for _r in ladder][::-1], [_r["tok_s"] / 1e3 for _r in ladder][::-1], color="C0")
    for _i, _r in enumerate(ladder[::-1]):
        _ax.text(_r["tok_s"] / 1e3, _i, f" {_r['tok_s'] / 1e3:.1f}k", va="center", fontsize=8)
    _ax.set_xlabel("training throughput (thousand tokens / s)")
    _ax.set_xlim(0, max(_r["tok_s"] for _r in ladder) / 1e3 * 1.18)
    _fig.tight_layout()
    _table = "\n".join(["| Rung | tok/s | ms/step | vs eager | MFU | first call |", "|---|---|---|---|---|---|", *_rows])
    mo.vstack([
        mo.md(_table),
        _fig,
        mo.md("MFU (model FLOPs utilization) uses **your** `mfu()` from `exercises.py` against the card's "
              f"measured bf16 peak of {L.PEAK_BF16 / 1e12:.0f} TFLOP/s; 'first call' is compile + one step.")
        if _mfu.ok else _mfu.view(mo),
    ])
    return


@app.cell
def _(ladder, mo):
    _by = {r["rung"]: r for r in ladder}

    def _gain(a, b):
        return _by[b]["tok_s"] / _by[a]["tok_s"]

    _nice = _gain("bf16", "nice")
    _nice_txt = (f"**{_nice:.2f}×** — a real gain" if _nice > 1.05 else
                 f"**{_nice:.2f}×** — within noise here" if _nice > 0.95 else f"**{_nice:.2f}×** — slower")
    mo.md(rf"""
    ### Reading the ladder

    * **jit (rung 1): {_gain("eager", "jit_f32"):.1f}×.** Eager JAX dispatches every `add`, `matmul`
      and `exp` separately and can't fuse anything. Compiling the whole step is the biggest single win —
      this is why Modules 06 and 10 insisted on it.
    * **bf16 (rung 2): {_gain("jit_f32", "bf16"):.1f}×.** The matrix units (WMMA) on RDNA3 run 16-bit
      inputs much faster than float32, and every activation is half the bytes.
    * **Nice numbers (rung 3): {_nice_txt}.** On NVIDIA, Karpathy's vocab padding (50,257 → 50,304)
      gave ~4%. hipBLASLt on RDNA3 picks tiles differently, so measure rather than assume. The output
      head is the biggest matmul in this model ({_by["nice"]["vocab"]:,} columns), so if padding
      matters anywhere, it's there.
    * **Donation (rung 4): {_gain("nice", "donate"):.2f}×.** Mostly a *memory* win (Section 5), not a
      speed win: XLA reuses the old parameter buffers for the new ones.
    * **Deferred metrics (rung 5): {_gain("donate", "deferred"):.2f}×** and **scan (rung 6):
      {_gain("deferred", "scan"):.2f}×.** These remove host-side waiting. At ~{_by["deferred"]["ms_step"]:.0f} ms
      per step the GPU is the bottleneck, so the overhead they remove is a few percent — close to the
      run-to-run noise of a shared GPU (Section 2's trace shows the idle time they remove directly). On
      the tiny MLP in Module 10 the same trick was worth 9×: overhead matters in inverse proportion to
      step size.

    Total: **{_by["scan"]["tok_s"] / _by["eager"]["tok_s"]:.0f}×** from rung 0 to rung 6, with the same
    model and the same math (up to bf16 rounding).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · A profiler trace: where did the time go?

    Throughput tells you *how fast*; a **trace** tells you *why*. [`jax.profiler.trace(dir)`](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.trace.html#jax.profiler.trace) records
    every host function and every GPU kernel with start and end times. On ROCm it works out of the box:

    ```python
    with jax.profiler.trace("/course/runs/22-profiling/trace/x", create_perfetto_trace=True):
        for i in range(6):
            with jax.profiler.StepTraceAnnotation("train", step_num=i):
                params, opt_state, loss = step(params, opt_state, batch)
    ```

    It writes `*.xplane.pb` (TensorBoard/XProf's format) and `perfetto_trace.json.gz`. To look at it
    yourself, open **https://ui.perfetto.dev** in a browser and drag in
    `runs/22-profiling/trace/<name>/plugins/profile/<date>/perfetto_trace.json.gz`. Below we read the
    same file programmatically with [`jax.profiler.ProfileData`](https://docs.jax.dev/en/latest/jax.profiler.html#module-jax.profiler).

    We take three traces of rung 4: one in a subprocess to rank the kernels (next cells), then two
    in this process to compare reading the loss every step (**synced**) with reading the previous
    step's loss (**deferred**).
    """)
    return


@app.cell
def _(L, g, jax, jnp, optax):
    def _trace(name, deferred):
        cfg = L.ladder_config(g, vocab=50304)
        opt = optax.adamw(3e-4)
        fn = jax.jit(L.make_step(g, cfg, opt, True), donate_argnums=(0, 1))
        state = {"p": g.init_gpt(jax.random.key(0), cfg)}
        state["s"] = opt.init(state["p"])
        batches = L.make_batches(8)
        for b in batches[:2]:   # warm up outside the trace
            state["p"], state["s"], _v = fn(state["p"], state["s"], b)
        float(jnp.mean(_v))

        def run(annotate):
            prev = None
            for i, b in enumerate(batches[2:]):
                with annotate(i):
                    state["p"], state["s"], v = fn(state["p"], state["s"], b)
                    if not deferred:
                        float(v)
                    elif prev is not None:
                        float(prev)
                    prev = v
            jax.block_until_ready(state["p"])
        return L.parse_xplane(L.capture_trace(name, run))

    traces = {"synced": _trace("synced", False), "deferred": _trace("deferred", True)}
    return (traces,)


@app.cell
def _(L):
    roof = L.roofline_trace()   # 4 steps, in a subprocess with XLA command buffers off (see below)
    return (roof,)


@app.cell
def _(ex, ladder, lj, mo, roof):
    if not roof["ok"]:
        _out = mo.callout(mo.md(f"The trace subprocess failed: `{roof['detail']}`"), kind="warn")
    else:
        _kernels = roof["kernels"]
        _top = lj.attempt(ex.parse_top_kernels, _kernels, 8)
        if _top.ok:
            _total = sum(k["dur"] for k in _kernels)
            _ops = {}
            for _k in _kernels:
                _ops.setdefault(_k["name"], _k["hlo_op"])
            _rows = [f"| `{_n[:46]}` | `{_ops[_n][:24]}` | {_d / 1e3 / roof['n_steps']:.2f} | {_c // roof['n_steps']} | "
                     f"{100 * _d / _total:.1f}% |" for _n, _d, _c in _top.value]
            _body = "\n".join(["| kernel | HLO op | ms / step | calls / step | share |", "|---|---|---|---|---|", *_rows])
            _gemm = next((k for k in _kernels if k["name"].startswith("Cijk")), None)
            _detail = (f"\n\nOne GEMM's launch record from the trace: `{_gemm['details'].strip()}` — grid and "
                       "block (workgroup) sizes, and `group_mem`, the on-chip shared memory (LDS) each workgroup "
                       "reserves. Keep these fields in mind for Section 8.") if _gemm else ""
            _out = mo.md(f"""
    **Top kernels: {len(_kernels) // roof["n_steps"]:,} kernel launches per training step,
    {_total / 1e3 / roof["n_steps"]:.1f} ms of GPU time per step** (your `parse_top_kernels`):

    {_body.replace(chr(10), chr(10) + "    ")}

    Names starting `Cijk_…` are **hipBLASLt/Tensile GEMMs** (the name encodes the tile, e.g.
    `MT96x96x32` = a 96×96 output tile, 32-deep slices of K). Names ending `fusion` are kernels **XLA
    generated itself**, and `triton_softmax` ones are XLA fusions emitted through Triton — Triton is
    still in use for softmax, only Triton *GEMMs* are switched off (Section 8). The "HLO op" column comes
    from the trace and points at the instruction in the compiled program; Section 4 joins it to shapes.{_detail}

    *A detail:* this trace ran in a subprocess with `XLA_FLAGS=--xla_gpu_enable_command_buffer=`. By
    default XLA records a whole step into one **command buffer** (a HIP graph) and replays it with one
    launch — which hides the per-kernel HLO names (every kernel shows up as `command_buffer_1`) but saves
    launch overhead: the ladder's rung 4 ran at ~{[r for r in ladder if r['rung'] == 'donate'][0]['ms_step']:.0f} ms per step with command buffers; this
    trace shows {_total / 1e3 / roof["n_steps"]:.0f} ms of kernels per step without them.
    """)
        else:
            _out = _top.view(mo)
    _out
    return


@app.cell
def _(ex, lj, mo, np, plt, traces):
    _fig, _ax = plt.subplots(figsize=(8, 2.4))
    _summary = []
    for _row, (_name, (_all, _steps)) in enumerate(traces.items()):
        _kernels = [k for k in _all if not _steps or k["start"] >= _steps[0][0]]  # drop warm-up stragglers
        _t0 = _kernels[0]["start"]
        _spans = [((k["start"] - _t0) / 1e3, k["dur"] / 1e3) for k in _kernels]
        _ax.broken_barh(_spans, (_row - 0.35, 0.7), color=f"C{_row}")
        _g = lj.attempt(ex.step_gaps, [(k["start"], k["start"] + k["dur"]) for k in _kernels])
        if _g.ok:
            _gaps = np.array(_g.value) / 1e3
            _wall = (_kernels[-1]["start"] + _kernels[-1]["dur"] - _t0) / 1e3
            _summary.append((_name, _wall, _gaps[_gaps > 0.05].sum(), int((_gaps > 0.05).sum())))
    _ax.set_yticks([0, 1], list(traces))
    _ax.set_xlabel("ms since the first kernel (6 training steps)")
    _ax.set_title("GPU busy (colored) vs idle (white)")
    _fig.tight_layout()
    if _summary:
        _rows = [f"| {n} | {w:.0f} ms | {i:.1f} ms ({100 * i / w:.1f}%) | {c} |" for n, w, i, c in _summary]
        _txt = mo.md("\n".join([
            "**The step bubble**, measured with your `step_gaps` (gaps > 50 µs):", "",
            "| loop | GPU wall time | idle | idle gaps |", "|---|---|---|---|", *_rows, "",
            "In the synced loop the GPU finishes a step, then waits while Python reads the loss, "
            "launches the next step and XLA uploads it: a gap at every step boundary. In the deferred "
            "loop the next step is already queued, so the gaps shrink. This is the JAX Training "
            "Cookbook's advice in one picture."]))
    else:
        _txt = lj.attempt(ex.step_gaps, [(0, 1)]).view(mo)
    mo.vstack([_fig, _txt])
    return


@app.cell
def _(L, g, jax, optax):
    _cfg = L.ladder_config(g, vocab=50304)
    _opt = optax.adamw(3e-4)
    _p = jax.eval_shape(lambda: g.init_gpt(jax.random.key(0), _cfg))
    _s = jax.eval_shape(lambda: _opt.init(g.init_gpt(jax.random.key(0), _cfg)))
    _b = jax.ShapeDtypeStruct((L.BATCH, L.SEQ + 1), "int32")
    _step = L.make_step(g, _cfg, _opt, True)
    compiled = jax.jit(_step, donate_argnums=(0, 1)).lower(_p, _s, _b).compile()
    compiled_plain = jax.jit(_step).lower(_p, _s, _b).compile()
    return compiled, compiled_plain


@app.cell
def _(L, compiled, mo, roof):
    hlo_text = roof["hlo"] if roof["ok"] else compiled.as_text()
    hlo_ops = L.parse_hlo_ops(hlo_text)
    _gemms = [n for n, o in hlo_ops.items() if o["kind"] == "gemm"]
    _fusions = [n for n, o in hlo_ops.items() if o["kind"] == "fusion"]
    _example = next((ln.strip() for ln in hlo_text.splitlines()
                     if "custom_call_target" in ln and "matmul" in ln), "")
    _example = _example[:260] + (" …" if len(_example) > 260 else "")
    _cost = compiled.cost_analysis()
    _cost = _cost[0] if isinstance(_cost, list) else _cost
    _gemm_flops = sum(hlo_ops[n]["flops"] for n in _gemms)
    mo.md(f"""
    ## 3 · Reading the compiled program

    `jax.jit(step).lower(...).compile().as_text()` is the program XLA actually runs — after fusion,
    after choosing library calls. For one training step of this GPT it contains
    **{len(_gemms)} GEMM library calls** and **{len(_fusions)} fusions**. A GEMM call looks like this:

    ```
    {_example}
    ```

    `custom_call_target="__cublas$lt$matmul"` is a historical name: on ROCm it is routed to
    **hipBLASLt**, AMD's matmul library (on NVIDIA it would be cuBLASLt). Every other operation —
    LayerNorm, GELU, softmax, the AdamW update — lives inside a `fusion`: a kernel XLA generated itself.

    A trap worth knowing: `compiled.cost_analysis()` reports **{_cost.get("flops", 0) / 1e9:.1f} GFLOP** for
    this step, but the GEMM shapes alone add up to **{_gemm_flops / 1e9:.0f} GFLOP**. XLA's cost model
    can't see inside library calls, so it misses nearly all the real work. Compute FLOPs from shapes
    (as `ladder.parse_hlo_ops` does, and as the 6·N rule does) instead.

    To see every stage of compilation, set `XLA_FLAGS=--xla_dump_to=/some/dir` before JAX starts —
    Section 6 does it for a tiny function.
    """)
    return hlo_ops, hlo_text


@app.cell
def _(L, mo):
    mo.md(rf"""
    ## 4 · The roofline: is each kernel limited by math or by memory?

    A kernel doing $F$ FLOPs while moving $B$ bytes can't finish faster than
    $\max(F / \text{{peak}},\ B / \text{{bandwidth}})$. On the RX 7600 XT, peak ≈ {L.PEAK_BF16 / 1e12:.0f}
    bf16 TFLOP/s (Module 00) and bandwidth ≈ {L.BANDWIDTH / 1e9:.0f} GB/s, so a kernel needs about
    **{L.PEAK_BF16 / L.BANDWIDTH:.0f} FLOPs per byte** (the *ridge point*) before compute becomes the limit.

    Below, every GEMM and fusion of one training step is placed by its **arithmetic intensity**
    (FLOPs ÷ bytes, from the HLO shapes) and its **achieved throughput** (FLOPs ÷ its time in the trace).
    """)
    return


@app.cell
def _(L, ex, hlo_ops, lj, mo, np, plt, roof):
    _kernels = roof["kernels"] if roof["ok"] else []
    _n_steps = roof.get("n_steps", 1)
    _time = {}
    for _k in _kernels:
        _time[_k["hlo_op"]] = _time.get(_k["hlo_op"], 0.0) + _k["dur"] * 1e-6 / _n_steps
    _ai = lj.attempt(ex.arithmetic_intensity, 1.0, 1.0)
    _rt = lj.attempt(ex.roofline_time, 1.0, 1.0, L.PEAK_BF16, L.BANDWIDTH)
    if not _kernels:
        _out = mo.md("No trace to place on the roofline (see Section 2).")
    elif _ai.ok and _rt.ok:
        _pts = []
        for _name, _op in hlo_ops.items():
            if _name in _time and _time[_name] > 2e-6:
                _intensity = ex.arithmetic_intensity(_op["flops"], _op["bytes"])
                _best = ex.roofline_time(_op["flops"], _op["bytes"], L.PEAK_BF16, L.BANDWIDTH)
                _pts.append((_name, _op["kind"], _intensity, _op["flops"] / _time[_name], _time[_name], _best))
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _x = np.logspace(-5, 3.5, 300)
        _ax.loglog(_x, np.minimum(L.PEAK_BF16, _x * L.BANDWIDTH) / 1e12, "k-", lw=1.5, label="roofline")
        _tmax = max(q[4] for q in _pts)
        for _kind, _c in (("gemm", "C0"), ("fusion", "C1")):
            for _above in (False, True):
                _sel = [p for p in _pts if p[1] == _kind and (p[3] > 1.2 * min(L.PEAK_BF16, p[2] * L.BANDWIDTH)) == _above]
                if not _sel:
                    continue
                _ax.scatter([p[2] for p in _sel], [p[3] / 1e12 for p in _sel], s=[8 + 400 * p[4] / _tmax for p in _sel],
                            facecolors="none" if _above else _c, edgecolors=_c, alpha=0.7,
                            label=f"{_kind} ({len(_sel)})" + (", estimate above roof" if _above else ""))
        _n_above = sum(1 for p in _pts if p[3] > 1.2 * min(L.PEAK_BF16, p[2] * L.BANDWIDTH))
        _direct = L.time_batched_gemm()
        _ridge = L.PEAK_BF16 / L.BANDWIDTH
        _ax.axvline(_ridge, color="gray", ls=":", lw=1)
        _ax.text(_ridge * 1.1, 0.02, f"ridge ≈ {_ridge:.0f} FLOP/B", fontsize=8, color="gray")
        _ax.set_xlabel("arithmetic intensity (FLOP / byte)")
        _ax.set_ylabel("achieved TFLOP/s")
        _ax.set_title("every kernel of one training step on the RX 7600 XT roofline\n(dot size ∝ time)", fontsize=10)
        _ax.legend(fontsize=8, loc="lower right")
        _fig.tight_layout()
        _gemm_t = sum(p[4] for p in _pts if p[1] == "gemm")
        _fus_t = sum(p[4] for p in _pts if p[1] == "fusion")
        _top = sorted(_pts, key=lambda p: -p[4])[:6]
        _vocab_t = sum(p[4] for p in _pts if hlo_ops[p[0]]["out"][1] % 50304 == 0)
        _all_t = sum(p[4] for p in _pts)
        _rows = [f"| `{p[0][:26]}` | {p[1]} | {p[2]:.3g} | {p[3] / 1e12:.3g} | {1e3 * p[4]:.2f} | {100 * p[5] / p[4]:.0f}% |"
                 for p in _top]
        _out = mo.vstack([_fig, mo.md("\n".join([
            f"GEMMs take **{1e3 * _gemm_t:.1f} ms** of each step, fusions **{1e3 * _fus_t:.1f} ms**. "
            "The six slowest kernels (your `arithmetic_intensity` and `roofline_time`):", "",
            "| op | kind | FLOP/B | TFLOP/s | ms | roofline time ÷ actual time |", "|---|---|---|---|---|---|", *_rows, "",
            "The last column is how close each kernel gets to its roofline bound (100% = as fast as the "
            "hardware allows; the 'peak' is Module 00's *measured* matmul rate, so the best GEMMs can edge past it).", "",
            f"**Hollow markers ({_n_above}) are estimates, not miracles.** Points above the roof mean the estimate "
            "is off, not the hardware: some fusions read only a slice of an operand (the byte count overstates "
            "traffic), and the trace's per-op time for the small batched attention GEMMs is too short. Timed "
            f"directly in a loop, the {L.BATCH * L.N_HEADS}×{L.SEQ}×{L.D_MODEL // L.N_HEADS} @ "
            f"{L.D_MODEL // L.N_HEADS}×{L.SEQ} attention-score GEMM runs at **{_direct:.1f} TFLOP/s** — well below "
            "the big projection GEMMs, because its matrices are too small to fill the GPU. Profilers are "
            "measurements with error bars; cross-check surprising numbers with a direct timing.", "",
            f"**The vocabulary is the bottleneck.** Kernels whose output has a 50,304-wide dimension — the "
            f"output head's matmuls, the softmax over the vocabulary and its gradient — take **{100 * _vocab_t / _all_t:.0f}%** "
            "of the step. GPT-2's 50k vocabulary on a 384-wide model is lopsided: Module 13's embedding "
            "table dominated the parameter count, and here the same matrix dominates the *time*. That's why "
            "the deepseek-v3-jax model chose an 8,192-token vocabulary for its size.", "",
            "**How to read it:** left of the ridge, a kernel can't be faster than memory allows; right of it, "
            "compute is the limit. GEMMs sit right of the ridge but well below the roof — at d = 384 the "
            "matrices are small enough that tiles and launch overheads matter. Fusions sit far left: they "
            "move bytes and do little math, so the only way to speed them up is to move fewer bytes. "
            "Attention's softmax is one of them: it reads and writes a T×T score matrix per head. A fused "
            "flash-attention kernel never writes that matrix to memory at all (Module 23)."]))])
    else:
        _out = _ai.view(mo) if not _ai.ok else _rt.view(mo)
    _out
    return


@app.cell
def _(compiled, compiled_plain, jax, mo):
    _d = compiled.memory_analysis()
    _p = compiled_plain.memory_analysis()
    _mib = 2**20
    _stats = jax.devices()[0].memory_stats() or {}
    _live = jax.live_arrays()
    mo.md(f"""
    ## 5 · Memory: what donation buys

    `compiled.memory_analysis()` reports what a compiled step needs, before running it:

    | | without donation | with `donate_argnums=(0, 1)` |
    |---|---|---|
    | arguments (params, optimizer state, batch) | {_p.argument_size_in_bytes / _mib:,.0f} MiB | {_d.argument_size_in_bytes / _mib:,.0f} MiB |
    | outputs (new params & state, loss) | {_p.output_size_in_bytes / _mib:,.0f} MiB | {_d.output_size_in_bytes / _mib:,.0f} MiB |
    | **aliased** (output written into an input's buffer) | {_p.alias_size_in_bytes / _mib:,.0f} MiB | **{_d.alias_size_in_bytes / _mib:,.0f} MiB** |
    | temporaries (activations, gradients) | {_p.temp_size_in_bytes / _mib:,.0f} MiB | {_d.temp_size_in_bytes / _mib:,.0f} MiB |

    Without donation, the old and new parameters and optimizer state exist at the same time, so the
    step needs room for both — here donation saves **{_d.alias_size_in_bytes / _mib:,.0f} MiB** of
    peak memory (the temporaries shift a little because XLA schedules the buffers differently). With donation, the outputs are written over the inputs — the aliased
    column — and your Python references to the old arrays become invalid (`is_deleted()` → True,
    Module 16). For a model whose weights and Adam state dominate memory, that's the difference between
    fitting and not fitting.

    Right now this process has **{len(_live)} live arrays** ([`jax.live_arrays()`](https://docs.jax.dev/en/latest/_autosummary/jax.live_arrays.html#jax.live_arrays)) holding
    **{sum(a.nbytes for a in _live) / _mib:,.0f} MiB**, and the allocator's peak so far was
    **{_stats.get("peak_bytes_in_use", 0) / _mib:,.0f} MiB** ([`jax.devices()[0].memory_stats()`](https://docs.jax.dev/en/latest/_autosummary/jax.devices.html#jax.devices)).
    """)
    return


@app.cell
def _(L, mo):
    _files = L.xla_dump()
    _kinds = {}
    for _f in _files:
        _stage = ("before optimizations" if "before_optimizations" in _f else
                  "after optimizations" if "after_optimizations" in _f else "other")
        _kinds[_stage] = _kinds.get(_stage, 0) + 1
    _show = "\n".join(f"    {f}" for f in _files[:8])
    mo.md(f"""
    ## 6 · Dumping XLA's work

    `XLA_FLAGS=--xla_dump_to=/course/runs/22-profiling/xla_dump` makes XLA write each compilation stage
    to disk. For `gelu(x @ w).sum()` it wrote **{len(_files)} files**
    ({", ".join(f"{v} {k}" for k, v in _kinds.items())}), e.g.:

    ```
{_show}
    ```

    Compare `*before_optimizations.txt` (what JAX handed over) with `*after_optimizations.txt` (after
    fusion and library-call selection) to see exactly what the compiler did to your code.
    """)
    return


@app.cell
def _(L, mo):
    _r = L.rocprofv3_probe()
    if _r["ok"]:
        _body = f"It works in this image. Kernel statistics for a 1024² matmul:\n\n```\n{_r['detail']}\n```"
    elif _r["available"]:
        _body = (f"`rocprofv3` is installed in this image, but profiling a JAX process fails:\n\n"
                 f"```\n{_r['detail']}\n```\n\nThe ROCm SDK's profiler and the JAX plugin's runtime disagree "
                 "on the profiler-registration interface, so `rocprofv3` aborts at start-up. Nothing is lost "
                 "for this course: `jax.profiler` (Section 2) gets the same kernel timings *and* maps each "
                 "kernel to its HLO op. On a full ROCm install, the command is:")
    else:
        _body = "`rocprofv3` isn't installed in this image. On a full ROCm install, the command is:"
    mo.md(f"""
    ## 7 · AMD's own profiler: rocprofv3

    {_body}

    ```
    rocprofv3 --kernel-trace --stats --output-format csv -d out/ -o run -- python train.py
    ```

    It writes per-kernel CSVs (`run_kernel_stats.csv`: calls, total and average time per kernel) and
    can also record hardware counters — cache hit rates, occupancy — that `jax.profiler` can't.
    """)
    return


@app.cell
def _(L, mo, os):
    _on = L.triton_gemm_count(enable_triton=True)
    _off = L.triton_gemm_count(enable_triton=False)
    if _on["ok"] and _off["ok"]:
        _counts = (f"**See it for yourself (safely).** Compiling — never running — the GPT's three matmul shapes "
                   f"(XLA merges the two that share an input into one wider GEMM, so they become 2 GEMMs) "
                   f"in a subprocess with Triton GEMM **enabled** gives **{_on['triton']} Triton GEMM fusions** "
                   f"(`__triton_nested_gemm_fusion`) and {_on['library_gemms']} hipBLASLt calls; with the course's "
                   f"setting, **{_off['triton']}** Triton GEMMs and **{_off['library_gemms']}** hipBLASLt calls. "
                   "That check passes `--xla_gpu_autotune_level=0`, which stops XLA from test-running candidate "
                   "kernels while compiling. With autotuning on (the default), XLA *times* Triton and hipBLASLt "
                   "candidates on the GPU and keeps the faster one — so with Triton GEMM enabled, a bad Triton "
                   "kernel can be dispatched while the program is still compiling, before your first step.")
    else:
        _counts = f"(The compile-only check didn't complete: {_on.get('detail') or _off.get('detail')}.)"
    _risky = os.environ.get("LEARNJAX_RISKY") == "1"
    mo.md(f"""
    ## 8 · The Triton GEMM failure, explained

    In Module 00 you met `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`. Here is what happened without
    it, when the deepseek-v3-jax model first ran on this card. The ROCm runtime reported:

    ```
    HSA_STATUS_ERROR_INVALID_DISPATCH_PARAMETERS: Kernel dispatch packet parameters exceed hardware
    limits for this agent (e.g. register usage, work-group dimensions, or other dispatch constraints)
    ... grid=[65536, 1, 1], workgroup=[1024, 1, 1], group_seg_size=16512 ... kernel: input_add_reduce_fusion_9
    ```

    Read it with Section 2's launch records in mind:

    * **workgroup = 1024 threads**: the maximum. All 1024 threads (32 waves of 32) must be resident on
      *one* workgroup processor at once, so every thread's registers must fit in that processor's
      register file *simultaneously*.
    * **group_seg_size = 16,512 bytes**: the LDS (on-chip shared memory) this workgroup reserves.
    * The kernel was compiled for **gfx1100** — that's what `HSA_OVERRIDE_GFX_VERSION=11.0.0` tells the
      runtime the card is. Navi 31 (gfx1100) and Navi 32 (gfx1101) have a 50% larger vector register
      file per SIMD than Navi 33 (gfx1102, your RX 7600 XT). A kernel whose register budget was sized for
      the bigger chips can't launch at 1024 threads on the smaller one, and the dispatch is rejected.
      That also explains why `11.0.1` failed the same way, and why tiny test models — whose fusions
      were compiled with smaller workgroups — never triggered it.

    That's the most likely mechanism; the definitive check is the kernel's register count in the
    compiled code object, which needs ROCm's disassembly tools. The fix stays the same either way: send
    matmuls to hipBLASLt, whose kernels are built and tuned for each real target, including gfx1102.

    {_counts} Triton itself isn't banned: the softmax fusions in Section 2's trace (`triton_softmax_*`)
    are Triton kernels and run fine. Only Triton *GEMMs* are switched off.

    **Live reproduction:** {"enabled (LEARNJAX_RISKY=1) — see the next cell." if _risky else
    "off. It *executes* a Triton-compiled step and **can hang the GPU until the process is killed or the "
    "driver resets** — your desktop may freeze for seconds. If you really want to see it, close other GPU "
    "work and run `LEARNJAX_RISKY=1 make lesson N=22`. It runs in a subprocess with a 120 s timeout."}
    """)
    return


@app.cell
def _(L, mo, os):
    import subprocess as _sp
    import sys as _sys

    if os.environ.get("LEARNJAX_RISKY") != "1":
        _out = mo.md("")
    else:
        _code = f"""
import sys; sys.path.insert(0, {str(L.ROOT)!r})
import importlib.util, jax, jax.numpy as jnp, optax
spec = importlib.util.spec_from_file_location("ladder", {str(L.ROOT / "modules/22-profiling/ladder.py")!r})
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
g, _ = L.load_gpt(); cfg = L.ladder_config(g, vocab=50304); opt = optax.adamw(3e-4)
p = g.init_gpt(jax.random.key(0), cfg); s = opt.init(p)
step = jax.jit(L.make_step(g, cfg, opt, True))
for b in L.make_batches(3):
    p, s, v = step(p, s, b); print("step ok, loss", float(v), flush=True)
"""
        try:
            _r = _sp.run([_sys.executable, "-c", _code], capture_output=True, text=True, timeout=120,
                         env=dict(os.environ, XLA_FLAGS=""), check=False)
            _tail = "\n".join((_r.stdout + _r.stderr).strip().splitlines()[-12:])
            _out = mo.md(f"Exit code **{_r.returncode}**. Last lines:\n\n```\n{_tail}\n```\n\n"
                         "If every step printed 'ok', this model didn't hit an oversized kernel — the original "
                         "failure came from deepseek-v3-jax's shapes, and whether a given fusion is generated "
                         "depends on the model's dimensions.")
        except _sp.TimeoutExpired:
            _out = mo.md("Timed out after 120 s — the GPU hung, as warned. The subprocess was killed.")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Your turn

    `exercises.py` has the six small functions this lesson's tables and plots call — arithmetic
    intensity, roofline time, MFU, padding, kernel ranking and gap finding. Until they're implemented the
    cells above show "Your turn" notes in their place. Quick checks:
    """)
    return


@app.cell
def _(L, ex, lj, mo):
    _checks = [
        ("pad_to_multiple(50257, 128)", lj.attempt(ex.pad_to_multiple, 50257, 128), "50304"),
        ("arithmetic_intensity(2·4096³, 3·4096²·2)", lj.attempt(ex.arithmetic_intensity, 2 * 4096**3, 3 * 4096**2 * 2), "≈ 1365"),
        ("roofline_time(1e9, 6e9, peak, bw)", lj.attempt(ex.roofline_time, 1e9, 6e9, L.PEAK_BF16, L.BANDWIDTH), "≈ 0.0208 s (memory-bound)"),
        ("mfu(112_000, 13.89M, 27e12)", lj.attempt(ex.mfu, 112_000, 13_890_000, L.PEAK_BF16), "≈ 0.35 (Module 16's train.py run)"),
    ]
    _items = []
    for _label, _a, _expect in _checks:
        if _a.ok:
            _v = f"{_a.value:,}" if isinstance(_a.value, int) else f"{_a.value:.4g}"
            _items.append(mo.md(f"`{_label}` → **{_v}** (expect {_expect})"))
        else:
            _items.append(_a.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Use bf16 so the matmuls get faster": "a",
            "Move fewer bytes: fuse it with its neighbours, or never write the intermediate to memory": "b",
            "Pad its dimensions to multiples of 128": "c",
        },
        label="**Check yourself:** a kernel sits far left of the ridge point on the roofline. What speeds it up?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "b": mo.callout(mo.md("Right. Left of the ridge, time = bytes ÷ bandwidth. Doing the math faster "
                              "changes nothing; moving fewer bytes does — which is exactly what fusion and "
                              "flash attention do."), kind="success"),
        "a": mo.callout(mo.md("bf16 *does* halve the bytes, so it helps a bit — but its headline benefit is "
                              "faster matrix units, which a memory-bound kernel isn't using. The real lever is "
                              "moving fewer bytes."), kind="warn"),
        "c": mo.callout(mo.md("Padding helps GEMM tile selection (compute-bound kernels). A memory-bound kernel "
                              "is limited by bytes moved; padding only adds bytes."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(22))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Andrej Karpathy, *Let's reproduce GPT-2 (124M)* and the [build-nanogpt](https://github.com/karpathy/build-nanogpt)
      repo — the PyTorch/NVIDIA version of this module's ladder, one commit per rung.
    * JAX docs: [Profiling computation](https://docs.jax.dev/en/latest/profiling.html) and
      [The Training Cookbook](https://docs.jax.dev/en/latest/the-training-cookbook.html) (the step bubble).
    * Google, [Perfetto UI](https://ui.perfetto.dev) — open `perfetto_trace.json.gz` traces in the browser.
    * Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) —
      compute, memory bandwidth and overhead, the three regimes this module measured.
    * AMD ROCm docs: *rocprofv3* — kernel traces and hardware counters on AMD GPUs.

    **Next:** Module 23 — kernels: Pallas, HIP and fusion, and the flash-attention rung this ladder couldn't climb.
    """)
    return


if __name__ == "__main__":
    app.run()
