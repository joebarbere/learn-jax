import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="23 · Kernels")


@app.cell
def _():
    import subprocess
    import sys
    import time

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    return ex, jax, jnp, lj, mo, np, pl, plt, pltriton, subprocess, sys, time


@app.cell
def _(mo):
    mo.md(r"""
    # 23 · Kernels: Pallas, HIP and fusion

    **Goal:** see what actually runs on the GPU when JAX "does a matmul", why the compiler usually
    writes better kernels than you would, and the one famous place it can't — attention. Then write
    kernels yourself.

    **You will:** map a kernel launch onto the RX 7600 XT's hardware and use an occupancy model to
    explain the Triton crash from Module 00 · confirm XLA's fusion in compiled HLO · see attention's
    O(T²) memory wall · implement FlashAttention's streaming softmax and a tiled attention that never
    builds the score matrix · write Pallas kernels and run them on the GPU · compare with hand-written
    HIP kernels and PyTorch's fused attention (`make torch-lesson N=23`).
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    NVIDIA's CUDA (2007) let programmers write GPU kernels in a C-like language. AMD's HIP, part of ROCm
    (2016), mirrors CUDA so kernels can move between vendors. Triton (Tillet, Kung & Cox, 2019) raised
    kernel writing from threads to tiles, and FlashAttention (Dao et al., 2022; FlashAttention-2, 2023)
    showed a hand-written kernel can beat a compiler by managing memory carefully. Pallas (2023) brought
    that style of kernel writing to JAX.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo, subprocess):
    _out = subprocess.run(["rocminfo"], capture_output=True, text=True, check=False).stdout
    _gpu = _out[_out.find("AMD Radeon"):] if "AMD Radeon" in _out else ""

    def _field(name):
        for _ln in _gpu.splitlines():
            if _ln.strip().startswith(name + ":"):
                return _ln.split(":", 1)[1].split("(")[0].strip()
        return "?"

    mo.md(f"""
    ## 1 · What a kernel is on RDNA3

    A **kernel** is one function launched on the GPU over a grid of threads. On AMD hardware:

    | level | what it is | on your RX 7600 XT (from `rocminfo`) |
    |---|---|---|
    | GPU | many compute units working independently | **{_field("Compute Unit")} compute units (CUs)** |
    | CU | runs **workgroups**; has its own LDS (shared memory) | {_field("SIMDs per CU")} SIMDs per CU, up to {_field("Max Waves Per CU")} waves resident |
    | SIMD | executes **wavefronts** in lock-step | wavefront = {_field("Wavefront Size")} threads ("wave32") |
    | workgroup | threads that can share LDS and synchronize | at most {_field("Workgroup Max Size")} threads |
    | thread | one lane; its variables live in **VGPRs** (vector registers) | ≤ 256 VGPRs each |

    The hard part of writing a kernel is fitting work into these limits. Each SIMD has a fixed
    **register file** shared by every wave resident on it, so the more registers each thread uses, the
    fewer waves fit. Fewer resident waves means less latency hiding — or, if even one workgroup doesn't
    fit, the launch fails outright. That's what happened in Module 00.
    """)
    return


@app.cell
def _(mo):
    vgpr_slider = mo.ui.slider(8, 256, step=8, value=64, label="VGPRs per thread", show_value=True)
    lds_slider = mo.ui.slider(0, 64, step=4, value=16, label="LDS per workgroup (KiB)", show_value=True)
    wg_dropdown = mo.ui.dropdown(["64", "128", "256", "512", "1024"], value="256", label="workgroup size")
    arch_dropdown = mo.ui.dropdown({"gfx1102 (RX 7600 XT, 1024)": 1024, "gfx1100 (RX 7900, 1536)": 1536},
                                   value="gfx1102 (RX 7600 XT, 1024)", label="register file per SIMD lane")
    mo.vstack([mo.md("### Occupancy calculator (your `workgroup_occupancy`)"),
               mo.hstack([vgpr_slider, lds_slider]), mo.hstack([wg_dropdown, arch_dropdown])])
    return arch_dropdown, lds_slider, vgpr_slider, wg_dropdown


@app.cell
def _(arch_dropdown, ex, lds_slider, lj, mo, vgpr_slider, wg_dropdown):
    _occ = lj.attempt(ex.workgroup_occupancy, vgpr_slider.value, lds_slider.value * 1024,
                      int(wg_dropdown.value), arch_dropdown.value)
    if _occ.ok:
        _r = _occ.value
        _verdict = (f"**fits** — {_r['workgroups_per_cu']} workgroup(s) per CU, **{_r['waves_per_simd']} of 16** "
                    f"wave slots per SIMD used, limited by **{_r['limited_by']}**" if _r["fits"]
                    else f"**does not fit** — the launch would fail (limited by **{_r['limited_by']}**)")
        _view = mo.md(_verdict)
    else:
        _view = _occ.view(mo)
    _view
    return


@app.cell
def _(ex, lj, mo):
    _rows = []
    _demo = lj.attempt(ex.workgroup_occupancy, 64, 16512, 1024)
    if _demo.ok:
        for _v in (48, 64, 72, 80, 96, 104):
            _a = ex.workgroup_occupancy(_v, 16512, 1024, 1024)["fits"]
            _b = ex.workgroup_occupancy(_v, 16512, 1024, 1536)["fits"]
            _rows.append(f"| {_v} | {'✅' if _a else '❌ launch fails'} | {'✅' if _b else '❌ launch fails'} |")
        _table = "\n".join(["| VGPRs per thread | gfx1102 (your card) | gfx1100 (what the override claims) |",
                            "|---|---|---|", *_rows])
        _body = mo.md(_table)
    else:
        _body = _demo.view(mo)
    mo.vstack([mo.md(r"""
    ### The Module 00 crash, explained (most likely)

    The failing launch from deepseek-v3-jax's first training run logged
    `workgroup=[1024, 1, 1]`, `group_seg_size=16512` (16 KiB of LDS) and
    `HSA_STATUS_ERROR_INVALID_DISPATCH_PARAMETERS`. A 1024-thread workgroup is 32 waves = **16 per
    SIMD**, so every wave slot must fit in the register file at once: at most 1024 ÷ 16 = **64 VGPRs** per
    thread on gfx1102, but 1536 ÷ 16 = **96** on gfx1100. `HSA_OVERRIDE_GFX_VERSION=11.0.0` makes the
    compiler target gfx1100 — so a Triton kernel allowed to use, say, 72 registers is legal for the chip
    it was compiled for and illegal for the chip it runs on. The log doesn't record the kernel's VGPR
    count, so this is the best-fitting explanation rather than a proof; switching XLA's GEMMs to
    hipBLASLt (tuned per real architecture) made it go away.
    """), _body])
    return


@app.cell
def _(jax, jnp, lj, mo):
    def _chain(x):
        return jax.nn.gelu(x * 2.0 + 1.0) * jax.nn.sigmoid(x) - 0.5 * x

    _x = jax.random.normal(jax.random.key(0), (1 << 24,))
    _jitted = jax.jit(_chain)
    _hlo = _jitted.lower(_x).compile().as_text()
    _entry = _hlo[_hlo.find("ENTRY"):]
    _kernels = [ln.split("=")[0].replace("ROOT", "").strip().lstrip("%") for ln in _entry.splitlines()
                if " fusion(" in ln]
    _t_eager = lj.timeit(_chain, _x, iters=5)["median"]
    _t_jit = lj.timeit(_jitted, _x, iters=10)["median"]
    _bytes = 2 * _x.size * 4
    mo.md(f"""
    ## 2 · Why XLA usually wins: fusion

    Six elementwise operations on 16M floats. Without [`jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit), each op is its own kernel that reads and
    writes the whole array. With `jit`, XLA **fuses** the chain into one kernel (Module 06): compiled HLO
    launches **{len(_kernels)} kernel**: `{', '.join(_kernels)}`.

    | | time | effective bandwidth (read x + write y) |
    |---|---|---|
    | op by op (no jit) | {_t_eager * 1e3:.2f} ms | {_bytes / _t_eager / 1e9:.0f} GB/s |
    | jitted (fused) | {_t_jit * 1e3:.2f} ms | **{_bytes / _t_jit / 1e9:.0f} GB/s** of ~288 |

    A fused elementwise kernel is limited only by memory bandwidth, and XLA generates it for you. You
    can't do much better by hand — so when is hand-writing a kernel worth it?
    """)
    return


@app.cell
def _(jax, jnp, mo):
    def naive_attention(q, k, v):
        _s = q @ k.T / jnp.sqrt(q.shape[-1])
        return jax.nn.softmax(_s, axis=-1) @ v

    _rows = []
    _last = 0.0
    for _T in (1024, 4096, 16384, 65536):
        _spec = jax.ShapeDtypeStruct((4, _T, 64), jnp.float32)
        # compile only; autotuning off so XLA doesn't try to *allocate* these buffers to benchmark kernels
        _m = (jax.jit(jax.vmap(naive_attention)).lower(_spec, _spec, _spec)
              .compile(compiler_options={"xla_gpu_autotune_level": 0}).memory_analysis())
        _last = _m.temp_size_in_bytes / 2**30
        _rows.append(f"| {_T:,} | {_m.temp_size_in_bytes / 2**20:,.0f} MiB |")
    _table = "\n".join(["| sequence length T | scratch memory XLA plans (4 heads, fp32) |", "|---|---|", *_rows])
    mo.vstack([mo.md(r"""
    ## 3 · Where fusion stops: attention

    Attention is `softmax(q kᵀ/√d) v`: a matmul, a softmax **across each whole row**, another matmul. XLA
    fuses elementwise work into matmuls, but the softmax needs every score in a row before it can
    normalize, so the full (T × T) score matrix gets written to memory between the two matmuls. Memory
    grows with T² — these numbers come from compiling (not running) the program, so nothing is allocated:
    """), mo.md(_table), mo.md(r"""
    """), mo.md(f"""
    At 64K tokens XLA plans **{_last:,.0f} GiB** for one attention call of a small model (it even logs that
    rematerialization can't squeeze it into the card's memory). Long-context models are only possible
    because of a different *algorithm*: process keys in blocks and never hold a full row of scores.
    """)])
    return (naive_attention,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · The streaming softmax

    The trick (Milakov & Gimelshein 2018, then FlashAttention): you *can* compute a softmax-weighted sum
    one block at a time if you keep three running numbers per query row — the largest score so far
    $m$, the denominator $l = \sum e^{s-m}$, and the accumulated output $\mathrm{acc} = \sum e^{s-m} v$.
    When a new block raises the maximum to $m'$, the old sums were computed against the wrong max, so
    rescale them by $e^{m-m'}$ — that's all:

    $$m' = \max(m, \max_j s_j) \qquad l' = l\,e^{m-m'} + \sum_j e^{s_j - m'} \qquad
    \mathrm{acc}' = \mathrm{acc}\,e^{m-m'} + \sum_j e^{s_j - m'} v_j$$

    After the last block, the output is $\mathrm{acc}/l$ — exactly the softmax result, no approximation.

    **Your turn:** `online_softmax_update` in `exercises.py`.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np):
    _ks = jax.random.split(jax.random.key(3), 3)
    _q = jax.random.normal(_ks[0], (6, 8))
    _k = jax.random.normal(_ks[1], (32, 8))
    _v = jax.random.normal(_ks[2], (32, 4))
    _s = _q @ _k.T / jnp.sqrt(8.0)
    _state = (jnp.full((6,), -jnp.inf), jnp.zeros(6), jnp.zeros((6, 4)))
    _rows = []
    _a = lj.attempt(ex.online_softmax_update, *_state, _s[:, :8], _v[:8])
    if _a.ok:
        _state = _a.value
        _rows.append(f"| after block 1 (keys 0–7) | {float(_state[0][0]):.3f} | {float(_state[1][0]):.3f} |")
        for _b in range(1, 4):
            _state = ex.online_softmax_update(*_state, _s[:, 8 * _b: 8 * _b + 8], _v[8 * _b: 8 * _b + 8])
            _rows.append(f"| after block {_b + 1} (keys {8 * _b}–{8 * _b + 7}) | {float(_state[0][0]):.3f} | "
                         f"{float(_state[1][0]):.3f} |")
        _err = float(np.max(np.abs(_state[2] / _state[1][:, None] - jax.nn.softmax(_s, -1) @ _v)))
        _out = mo.md("\n".join(["Query row 0, streamed over 4 blocks of 8 keys:", "",
                                "| | running max m | running denominator l |", "|---|---|---|", *_rows, "",
                                f"Max difference from the ordinary softmax: **{_err:.1e}**. "
                                "Watch m jump when a block contains a bigger score — and l shrink to compensate."]))
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · FlashAttention's algorithm in pure JAX

    Now scan over key blocks with [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan), carrying `(m, l, acc)` and calling your update — the
    (T × T) score matrix never exists, only one (T × block) tile at a time.

    **Your turn:** `flash_attention(q, k, v, block, causal=False)`.
    """)
    return


@app.cell
def _(ex, jax, lj, mo, naive_attention, np):
    _ks = jax.random.split(jax.random.key(4), 3)
    _q, _k, _v = (jax.random.normal(_kk, (256, 64)) for _kk in _ks)
    _a = lj.attempt(ex.flash_attention, _q, _k, _v, 64)
    _c = lj.attempt(ex.flash_attention, _q, _k, _v, 32, True)
    flash_ok = _a.ok and _c.ok
    if flash_ok:
        _causal_ref = jax.nn.softmax(
            np.where(np.tril(np.ones((256, 256), bool)), _q @ _k.T / 8.0, -np.inf), -1) @ _v
        _out = mo.md(f"""
        | | max difference from standard attention (T = 256, d = 64) |
        |---|---|
        | `flash_attention(block=64)` | **{float(np.max(np.abs(_a.value - naive_attention(_q, _k, _v)))):.1e}** |
        | `flash_attention(block=32, causal=True)` | **{float(np.max(np.abs(_c.value - _causal_ref))):.1e}** |

        Exact up to float rounding — it's the same math, reordered.
        """)
    else:
        _out = _a.view(mo) if not _a.ok else _c.view(mo)
    _out
    return (flash_ok,)


@app.cell
def _(ex, flash_ok, jax, jnp, lj, mo, naive_attention, plt):
    flash_results = []
    if flash_ok:
        _flash = jax.vmap(lambda q, k, v: ex.flash_attention(q, k, v, 128))
        _naive = jax.vmap(naive_attention)
        for _T in (512, 1024, 2048, 4096):
            _q = jax.random.normal(jax.random.key(_T), (4, _T, 64))
            _fn, _ff = jax.jit(_naive), jax.jit(_flash)
            _mn = _fn.lower(_q, _q, _q).compile().memory_analysis().temp_size_in_bytes
            _mf = _ff.lower(_q, _q, _q).compile().memory_analysis().temp_size_in_bytes
            flash_results.append((_T, _mn / 2**20, _mf / 2**20,
                                  lj.timeit(_fn, _q, _q, _q, iters=5)["median"] * 1e3,
                                  lj.timeit(_ff, _q, _q, _q, iters=5)["median"] * 1e3))
        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
        _Ts = [r[0] for r in flash_results]
        _ax1.loglog(_Ts, [r[1] for r in flash_results], "o-", label="standard")
        _ax1.loglog(_Ts, [r[2] for r in flash_results], "s-", label="tiled (yours)")
        _ax1.set(xlabel="sequence length T", ylabel="scratch memory (MiB)", title="memory")
        _ax2.loglog(_Ts, [r[3] for r in flash_results], "o-", label="standard")
        _ax2.loglog(_Ts, [r[4] for r in flash_results], "s-", label="tiled (yours)")
        _ax2.set(xlabel="sequence length T", ylabel="ms (4 heads, d=64, fp32)", title="time")
        for _ax in (_ax1, _ax2):
            _ax.set_xticks(_Ts, [f"{t:,}" for t in _Ts])
            _ax.xaxis.set_minor_formatter(plt.NullFormatter())
            _ax.legend()
            _ax.grid(alpha=0.3, which="both")
        _fig.tight_layout()
        _out = _fig
    else:
        _out = mo.md("*(Finish `flash_attention` to see memory and time vs sequence length.)*")
    _out
    return (flash_results,)


@app.cell
def _(flash_results, mo):
    if flash_results:
        _rows = "\n".join(f"| {T:,} | {mn:.0f} | {mf:.1f} | {tn:.2f} | {tf:.2f} |"
                          for T, mn, mf, tn, tf in flash_results)
        _T, _mn, _mf, _tn, _tf = flash_results[-1]
        _out = mo.md("\n".join([
            "| T | standard MiB | tiled MiB | standard ms | tiled ms |", "|---|---|---|---|---|", _rows, "",
            f"**Memory:** at T = {_T:,} the tiled version plans {_mn / _mf:.0f}× less scratch memory, and the "
            "gap keeps growing — linear vs quadratic.",
            "",
            f"**Time:** it's {_tf / _tn:.1f}× *slower* here. Each scan step is a small (T × 128) matmul "
            "launched as its own kernel, so the GPU sits mostly idle between steps, and the running state "
            "round-trips through memory every block. Real FlashAttention runs the whole loop *inside one "
            "kernel*, with m, l and acc held in registers and the tiles in LDS — something XLA can't express "
            "and you can't write in `jnp`. That's what kernel languages are for."]))
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Pallas: write the kernel yourself

    **Pallas** is JAX's kernel language. You write the *body* of a kernel in ordinary `jnp`, operating on
    **refs** (views of one block of each input/output), and `pl.pallas_call` launches it over a **grid**:

    ```python
    def kernel(x_ref, y_ref, o_ref):          # runs once per grid step, on one block
        o_ref[...] = x_ref[...] + y_ref[...]

    pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // B,),                                     # how many program instances
        in_specs=[pl.BlockSpec((B,), lambda i: (i,))] * 2,  # which block each instance sees
        out_specs=pl.BlockSpec((B,), lambda i: (i,)),
    )(x, y)
    ```

    On GPUs Pallas lowers through **Triton**, the same compiler behind the Module 00 crash. On ROCm you
    must ask for that backend explicitly with `compiler_params=pltriton.CompilerParams()`. And with
    `interpret=True` a kernel runs as ordinary JAX on any backend — the way to write and test kernels.

    **Your turn:** `add_relu_kernel` and `row_softmax_kernel` (tested in interpret mode).
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np, pl):
    _x = jax.random.normal(jax.random.key(5), (8, 128))
    _y = jax.random.normal(jax.random.key(6), (8, 128))
    _shape = jax.ShapeDtypeStruct(_x.shape, _x.dtype)
    _a = lj.attempt(lambda: pl.pallas_call(ex.add_relu_kernel, out_shape=_shape, interpret=True)(_x, _y))
    _b = lj.attempt(lambda: pl.pallas_call(ex.row_softmax_kernel, out_shape=_shape, interpret=True)(_x * 20))
    _items = [mo.md("**Interpret mode** (plain JAX, any backend):")]
    for _name, _r, _ref in (("add_relu_kernel", _a, jnp.maximum(_x + _y, 0)),
                            ("row_softmax_kernel", _b, jax.nn.softmax(_x * 20, -1))):
        if _r.ok:
            _items.append(mo.md(f"`{_name}` matches the reference to **{float(np.max(np.abs(_r.value - _ref))):.1e}** ✅"))
        elif _r.todo:
            _items.append(mo.callout(mo.md(f"**Your turn:** implement `{_name}` in `exercises.py`."), kind="info"))
        else:
            _items.append(mo.callout(mo.md(f"```\n{_r.error}\n```"), kind="danger"))
    mo.vstack(_items)
    return


@app.cell
def _(mo, subprocess, sys):
    _probe = r"""
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton
def k(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0
x = jnp.arange(256, dtype=jnp.float32)
try:
    pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)
except Exception as e:
    print("DEFAULT:", type(e).__name__, str(e).splitlines()[0][:200])
y = pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                   compiler_params=pltriton.CompilerParams())(x)
print("TRITON_OK" if float(y[3]) == 6.0 else "TRITON_WRONG")
"""
    try:
        _r = subprocess.run([sys.executable, "-c", _probe], capture_output=True, text=True, timeout=180, check=False)
        _txt = _r.stdout + _r.stderr
    except subprocess.TimeoutExpired:
        _txt = "TIMEOUT"
    pallas_gpu_ok = "TRITON_OK" in _txt
    _default = next((ln for ln in _txt.splitlines() if ln.startswith("DEFAULT:")), "")
    if pallas_gpu_ok:
        _msg = mo.callout(mo.md(f"""
        **Pallas runs on your GPU.** Probed in a separate process with a timeout (a hang can't freeze
        this notebook). Without `compiler_params`, Pallas tries its Mosaic GPU backend and stops with:
        `{_default.removeprefix('DEFAULT: ')}` — with the Triton backend requested, the kernel compiles and
        runs on the RX 7600 XT.
        """), kind="success")
    else:
        _msg = mo.callout(mo.md(f"Pallas could not run on the GPU here (`{_txt.strip()[-300:]}`). Everything "
                                "below falls back to interpret mode; the programming model is identical."),
                          kind="warn")
    _msg
    return (pallas_gpu_ok,)


@app.cell
def _(ex, jax, jnp, lj, mo, np, pallas_gpu_ok, pl, pltriton):
    _cp = pltriton.CompilerParams()
    _X = jax.random.normal(jax.random.key(7), (8192, 1024))
    _rows = []
    if pallas_gpu_ok:
        def _pallas_softmax(x):
            return pl.pallas_call(ex.row_softmax_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                                  grid=(x.shape[0],), in_specs=[pl.BlockSpec((1, x.shape[1]), lambda i: (i, 0))],
                                  out_specs=pl.BlockSpec((1, x.shape[1]), lambda i: (i, 0)), compiler_params=_cp)(x)

        _a = lj.attempt(jax.jit(_pallas_softmax), _X)
        if _a.ok:
            _err = float(np.max(np.abs(_a.value - jax.nn.softmax(_X, -1))))
            _tp = lj.timeit(jax.jit(_pallas_softmax), _X, iters=10)["median"]
            _tx = lj.timeit(jax.jit(lambda x: jax.nn.softmax(x, -1)), _X, iters=10)["median"]
            _gb = 2 * _X.size * 4 / 1e9
            _out = mo.md(f"""
            ### Your row-softmax kernel, on the GPU

            8192 rows × 1024, one program per row (`grid=(8192,)`), max difference from [`jax.nn.softmax`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.softmax.html#jax.nn.softmax)
            **{_err:.1e}**.

            | | ms | GB/s |
            |---|---|---|
            | your Pallas kernel | {_tp * 1e3:.3f} | {_gb / _tp:.0f} |
            | XLA's `jax.nn.softmax` | {_tx * 1e3:.3f} | {_gb / _tx:.0f} |

            Both are memory-bound, so the question is only how close each gets to ~288 GB/s. A row-wise
            softmax is exactly the kind of fusion XLA already does well — a good first kernel, not a speedup.
            """)
        else:
            _out = _a.view(mo)
    else:
        _out = mo.md("*(GPU Pallas unavailable — see above.)*")
    _out
    return


@app.cell
def _(jax, jnp, lj, mo, np, pallas_gpu_ok, pl, pltriton):
    import functools

    _cp = pltriton.CompilerParams()

    def _mm_whole_k(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32).astype(o_ref.dtype)

    def _mm_chunked(a_ref, b_ref, o_ref, *, tk, nk):
        def body(i, acc):
            a = a_ref[:, pl.ds(i * tk, tk)]
            b = b_ref[pl.ds(i * tk, tk), :]
            return acc + jnp.dot(a, b, preferred_element_type=jnp.float32)
        o_ref[...] = jax.lax.fori_loop(0, nk, body, jnp.zeros(o_ref.shape, jnp.float32)).astype(o_ref.dtype)

    def pallas_matmul(a, b, kernel, tile=64):
        n = a.shape[0]
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((n, n), a.dtype), grid=(n // tile, n // tile),
                              in_specs=[pl.BlockSpec((tile, n), lambda i, j: (i, 0)),
                                        pl.BlockSpec((n, tile), lambda i, j: (0, j))],
                              out_specs=pl.BlockSpec((tile, tile), lambda i, j: (i, j)), compiler_params=_cp)(a, b)

    _items = [mo.md(r"""
    ### A tiled matmul — and the LDS limit

    Each program computes one 64 × 64 output tile from a 64-row strip of A and a 64-column strip of B.
    The first version loads both whole strips (all of K) at once:
    """)]
    if pallas_gpu_ok:
        _n = 512
        _A = jax.random.normal(jax.random.key(8), (_n, _n))
        _first = lj.attempt(lambda: jax.block_until_ready(pallas_matmul(_A, _A, _mm_whole_k)))
        _msg = next((ln for ln in (_first.error or "").splitlines() if "RESOURCE_EXHAUSTED" in ln), "it ran")
        _items.append(mo.md(f"```\n{_msg.strip()[:200]}\n```"))
        _items.append(mo.md(r"""
        The kernel asked for **131,072 bytes** of LDS — one 64 × 512 fp32 strip is 64 · 512 · 4 = 131,072
        bytes — and a workgroup can have **65,536** (the same 64 KiB limit as the occupancy model). The fix is the classic one: loop over K in chunks inside the kernel, keeping only a
        64 × 32 slice of each strip at a time and accumulating in registers.
        """))
        _rows = []
        _ratio = {}
        for _dt in (jnp.float32, jnp.bfloat16):
            _N = 2048
            _A = jax.random.normal(jax.random.key(9), (_N, _N), _dt)
            _B = jax.random.normal(jax.random.key(10), (_N, _N), _dt)
            _f = jax.jit(functools.partial(pallas_matmul, kernel=functools.partial(_mm_chunked, tk=32, nk=_N // 32)))
            _ref = jnp.matmul(_A.astype(jnp.float32), _B.astype(jnp.float32))
            _rel = float(jnp.max(jnp.abs(_f(_A, _B).astype(jnp.float32) - _ref)) / jnp.max(jnp.abs(_ref)))
            _tp = lj.timeit(_f, _A, _B, iters=5)["median"]
            _tx = lj.timeit(jax.jit(jnp.matmul), _A, _B, iters=5)["median"]
            _ratio[jnp.dtype(_dt).name] = _tx / _tp
            _rows.append(f"| {jnp.dtype(_dt).name} | {_rel:.1e} | {2 * _N**3 / _tp / 1e12:.1f} | {2 * _N**3 / _tx / 1e12:.1f} |")
        _items.append(mo.md("\n".join(["| 2048³ | max rel. error | your-style Pallas TFLOP/s | XLA (hipBLASLt) TFLOP/s |",
                                       "|---|---|---|---|", *_rows])))
        _items.append(mo.md(f"""
        This run: **{_ratio['float32']:.0%}** of the library's speed in fp32 and **{_ratio['bfloat16']:.0%}**
        in bf16. In bf16, Triton emits the card's WMMA matrix instructions just like hipBLASLt, so even a
        simple tiled kernel lands in the same range (timings move ±30% with other GPU work). fp32 has no
        matrix-unit path on RDNA3, and there the library's years of per-architecture tuning (tile shapes,
        double-buffered LDS, register blocking) show. That's why compilers call libraries for matmuls —
        and why you'd write a kernel for what a library *can't* do, like fusing a softmax between two
        matmuls.
        """))
    else:
        _items.append(mo.md("*(GPU Pallas unavailable — see above.)*"))
    mo.vstack(_items)
    return


@app.cell
def _(flash_results, mo):
    _jax = {T: (tn, tf) for T, _mn, _mf, tn, tf in flash_results}

    def _fmt(T):
        return f"{_jax[T][0]:.2f} / {_jax[T][1]:.2f}" if T in _jax else "—"

    mo.md(f"""
    ## 7 · The PyTorch side: HIP kernels and fused attention

    `torch_compare.py` (`make torch-lesson N=23`) does the same experiments in PyTorch. Measured when this
    was written on an idle GPU:

    **A hand-written HIP kernel** (fused bias + GELU, compiled at runtime with
    `torch.utils.cpp_extension.load_inline` — `hipcc` ships in the image, gfx1102 is a native target):

    | 8192 × 4096 fp32 | ms | effective GB/s |
    |---|---|---|
    | PyTorch eager (two kernels) | 2.06 | 130 |
    | your HIP kernel | 1.03 | 262 |
    | `torch.compile` | 1.03 | 261 |

    Fusion is the whole win; the compiler finds it too.

    **Attention, 4 heads, d = 64** — PyTorch's fused kernels (AOTriton, opt-in on RDNA3, fp16) vs JAX here
    (fp32; standard / your tiled, ms, measured live above):

    | T | PyTorch math (fp32) | PyTorch flash (fp16) | JAX standard / tiled (fp32) |
    |---|---|---|---|
    | 1,024 | 0.73 ms, 37 MiB | 0.11 ms, 1 MiB | {_fmt(1024)} |
    | 2,048 | 3.88 ms, 146 MiB | 0.41 ms, 1 MiB | {_fmt(2048)} |
    | 4,096 | 14.3 ms, 580 MiB | 1.64 ms, 2 MiB | {_fmt(4096)} |

    A *fused* FlashAttention kernel gets the memory savings **and** the speed. JAX on ROCm doesn't ship one
    ([`jax.nn.dot_product_attention`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html#jax.nn.dot_product_attention)'s cuDNN path is NVIDIA-only) — the natural next project is writing it
    in Pallas: your `online_softmax_update` inside a kernel whose grid runs over query blocks.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "It never materializes the T×T score matrix, so memory grows linearly — speed needs a fused kernel": "a",
            "It is faster because scan compiles to a single kernel": "b",
            "It is an approximation that trades accuracy for memory": "c",
        },
        label="**Check yourself:** what does your `lax.scan` FlashAttention buy you on this GPU?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right. Same exact result, linear memory — and on its own it's slower, because "
                              "each block is a separate small kernel. The speed comes from fusing the loop."),
                        kind="success"),
        "b": mo.callout(mo.md("No — XLA launches the scan body's matmuls as separate kernels each step; that's "
                              "why it measured slower than standard attention."), kind="warn"),
        "c": mo.callout(mo.md("No — the streaming softmax is exact; your test matched standard attention to "
                              "~1e-6."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(23))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * JAX docs: [Pallas quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html) and the
      [Pallas design note](https://docs.jax.dev/en/latest/pallas/design/design.html)
    * AMD: [RDNA3 instruction set architecture guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
      (wavefronts, VGPRs, LDS)
    * Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) —
      the streaming-softmax trick on its own
    * PyTorch docs: [Custom C++ and CUDA/HIP extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)

    **Next:** Module 24 — parallelism and sharding: the same programs across many devices.
    """)
    return


if __name__ == "__main__":
    app.run()
