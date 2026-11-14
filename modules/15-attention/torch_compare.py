import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="15 · Attention in PyTorch")


@app.cell
def _():
    import marimo as mo
    import math
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    # Opt in to PyTorch's experimental fused attention kernels on this GPU (section 3 explains).
    # Must be set before PyTorch first runs attention — it's read once.
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    return F, Path, math, mo, np, plt, subprocess, sys, time, torch


@app.cell
def _(mo):
    mo.md(r"""
    # 15 · The same attention in PyTorch (side-by-side)

    This notebook runs in the **PyTorch** image (`make torch-lesson N=15`), which has no JAX. It rebuilds
    the lesson's multi-head causal attention twice — by hand, and with PyTorch's fused
    `F.scaled_dot_product_attention` (SDPA) — checks both against numbers the JAX lesson saved, and
    looks at something JAX on this card doesn't give you: **fused attention kernels** that never store the
    $T \times T$ score matrix (experimental on this GPU — the notebook turns them on in its first cell).
    """)
    return


@app.cell
def _(mo, torch):
    _props = torch.cuda.get_device_properties(0)
    mo.md(f"""
    ## 1 · The device

    PyTorch `{torch.__version__}` (HIP `{torch.version.hip}`) on `{torch.cuda.get_device_name(0)}`, architecture
    `{_props.gcnArchName}`. As in Module 10: `device="cuda"` means the Radeon on ROCm builds, and this image
    needs no gfx override.
    """)
    return


@app.cell
def _(F, math, torch):
    def mha_manual(p, x, n_heads, causal=True):
        """Multi-head attention by hand — the lesson's einsums, in PyTorch."""
        B, T, D = x.shape
        q, k, v = ((x @ p[n]).view(B, T, n_heads, D // n_heads) for n in ("wq", "wk", "wv"))
        scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(D // n_heads)
        if causal:
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        w = scores.softmax(dim=-1)
        out = torch.einsum("bhts,bshd->bthd", w, v)
        return out.reshape(B, T, D) @ p["wo"]

    def mha_sdpa(p, x, n_heads, causal=True):
        """Same thing with the fused kernel. SDPA wants heads before time: (B, H, T, d)."""
        B, T, D = x.shape
        q, k, v = ((x @ p[n]).view(B, T, n_heads, D // n_heads).transpose(1, 2) for n in ("wq", "wk", "wv"))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return out.transpose(1, 2).reshape(B, T, D) @ p["wo"]

    return mha_manual, mha_sdpa


@app.cell
def _(Path, mha_manual, mha_sdpa, mo, np, torch):
    _f = Path("/course/runs/15-attention/mha_reference.npz")
    if _f.exists():
        _z = np.load(_f)
        _p = {n: torch.tensor(_z[n], device="cuda") for n in ("wq", "wk", "wv", "wo")}
        _x = torch.tensor(_z["x"], device="cuda")
        _H = int(_z["n_heads"])
        _rows = []
        for _name, _fn in (("by hand", mha_manual), ("SDPA", mha_sdpa)):
            for _causal, _key in ((True, "out_causal"), (False, "out_full")):
                with torch.no_grad():
                    _got = _fn(_p, _x, _H, _causal).cpu().numpy()
                _rows.append(f"| {_name} | {'causal' if _causal else 'full'} | "
                             f"{np.abs(_got - _z[_key]).max():.1e} |")
        _out = mo.md(f"""
    ## 2 · Do the numbers match JAX?

    Same weights and input (2 × 32 tokens, D = 64, 4 heads), saved by the JAX lesson:

    | PyTorch version | mask | max difference vs JAX |
    |---|---|---|
    {chr(10).join(_rows)}

    Differences around 1e-6 are float32 rounding from different kernels summing in different orders —
    the same computation.
    """)
    else:
        _out = mo.callout(mo.md(
            "## 2 · Do the numbers match JAX?\n\nRun the JAX lesson first (`make lesson N=15`, or "
            "`containers/run.sh python modules/15-attention/lesson.py`) — it saves test vectors to "
            "`runs/15-attention/mha_reference.npz` for this comparison."), kind="info")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Which fused kernel does ROCm give you?

    `F.scaled_dot_product_attention` picks among backends:

    * **flash** — FlashAttention: works tile by tile in on-chip memory, never materializing the
      $T\times T$ scores. Memory grows like $T$, not $T^2$.
    * **efficient** — "memory-efficient attention" (xFormers-style), the same idea, different kernel.
    * **math** — the plain composition of matmul, softmax, matmul (what the hand-written version does).

    On AMD, the fused kernels come from **AOTriton** (ahead-of-time compiled Triton kernels). For consumer
    RDNA3 cards like yours PyTorch marks them *experimental* and turns them off unless the environment
    variable `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is set — which this notebook does in its first cell.
    Here is a fresh Python process **without** the flag:
    """)
    return


@app.cell
def _(mo, subprocess, sys):
    _probe = '''
import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
q = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.float16)
for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                 ("math", SDPBackend.MATH)):
    try:
        with sdpa_kernel(be):
            F.scaled_dot_product_attention(q, q, q, is_causal=True)
        print(name, "available")
    except RuntimeError:
        print(name, "unavailable")
'''
    _env = {k: v for k, v in __import__("os").environ.items() if k != "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"}
    _r = subprocess.run([sys.executable, "-c", _probe], capture_output=True, text=True, env=_env, check=False)
    _lines = [ln for ln in _r.stdout.splitlines() if ln.split()[-1] in ("available", "unavailable")]
    mo.md("```\n" + ("\n".join(_lines) or _r.stderr[-500:]) + "\n```\n\n"
          "So by default only the **math** path runs on this card. With the flag (below), all three do. "
          "*Experimental* is the operative word: this course has already seen one Triton-generated kernel "
          "hang this GPU (Module 00). The flash and efficient kernels pass the checks here, but if a long "
          "run ever hangs, unset the flag first.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now each backend is forced in turn on causal attention (B = 1, H = 8, d = 64, float16 — fused kernels
    want 16-bit inputs) and timed, with the peak GPU memory of the call:
    """)
    return


@app.cell
def _(F, mo, np, plt, time, torch):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _bench(fn, *a, iters=5):
        for _ in range(2):
            fn(*a)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*a)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters, torch.cuda.max_memory_allocated() / 2**20

    _backends = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                 "math": SDPBackend.MATH}
    _Ts = [512, 1024, 2048, 4096]
    sdpa_results = {}
    for _name, _be in _backends.items():
        sdpa_results[_name] = []
        for _T in _Ts:
            _q, _k, _v = (torch.randn(1, 8, _T, 64, device="cuda", dtype=torch.float16) for _ in range(3))
            try:
                with sdpa_kernel(_be):
                    _t, _m = _bench(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
                                    _q, _k, _v)
                sdpa_results[_name].append((_T, _t, _m))
            except RuntimeError:
                sdpa_results[_name].append((_T, None, None))
            del _q, _k, _v
            torch.cuda.empty_cache()

    _lines = []
    for _name, _res in sdpa_results.items():
        _cells = " | ".join("unavailable" if _t is None else f"{_t * 1e3:.2f} ms / {_m:,.0f} MiB"
                            for _T, _t, _m in _res)
        _lines.append(f"| {_name} | {_cells} |")
    _fig, _ax = plt.subplots(figsize=(5.2, 3.0))
    for _name, _res in sdpa_results.items():
        _pts = [(_T, _m) for _T, _t, _m in _res if _t is not None]
        if _pts:
            _ax.loglog(*zip(*_pts), "o-", label=_name, base=2)
    _ax.set(xlabel="sequence length T", ylabel="peak MiB", title="peak GPU memory of one attention call")
    _ax.legend(fontsize=8)
    _avail = [n for n, r in sdpa_results.items() if r[-1][1] is not None]
    _q0 = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.float16)
    _default_t, _ = _bench(lambda q: F.scaled_dot_product_attention(q, q, q, is_causal=True), _q0)
    _closest = min((n for n in _avail), key=lambda n: abs(dict((T, t) for T, t, _ in sdpa_results[n])[1024]
                                                          - _default_t)) if _avail else None
    mo.vstack([mo.md(f"""
    | backend | {" | ".join(f"T = {T:,}" for T in _Ts)} |
    |---|{"---|" * len(_Ts)}
    {chr(10).join(_lines)}

    Available on this card: **{", ".join(_avail) or "none"}**. With no backend forced, SDPA at T = 1,024 takes
    {_default_t * 1e3:.2f} ms — closest to **{_closest}**, which is most likely what it picks by default.
    """), _fig])
    return (sdpa_results,)


@app.cell
def _(mo, np, sdpa_results):
    _math = dict((T, m) for T, t, m in sdpa_results["math"] if t is not None)
    _fused = {n: dict((T, m) for T, t, m in r if t is not None) for n, r in sdpa_results.items() if n != "math"}
    _notes = []
    for _n, _d in _fused.items():
        if 4096 in _d and 4096 in _math:
            _notes.append(f"at T = 4,096 **{_n}** peaks at {_d[4096]:,.0f} MiB vs {_math[4096]:,.0f} MiB for math "
                          f"({_math[4096] / max(_d[4096], 1e-9):.1f}×)")
    mo.md(f"""
    ## 4 · What this means

    {"; ".join(_notes) + "." if _notes else "No fused backend ran on this card, so every path stores the full score matrix."}
    The math path materializes B·H·T² intermediates — one score matrix is 8 × 4,096² × 2 bytes ≈ 256 MiB in
    float16, and it keeps several (scores, mask, softmax output), hence over a gigabyte — so its memory grows
    4× each time T doubles. A fused kernel's memory grows only with T.

    The fused kernels are also much faster at long lengths, because they never write the scores out to
    GPU memory and read them back — attention at this size is limited by memory traffic, not arithmetic.

    The JAX lesson measured `jax.nn.dot_product_attention` at about the same speed as hand-written attention
    on this card: on ROCm it lowers to plain XLA ops (its `cudnn` path is NVIDIA-only), so JAX here behaves
    like the **math** row. Module 23 looks at writing fused kernels yourself (Pallas).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Idioms, side by side

    | | JAX (the lesson) | PyTorch (this notebook) |
    |---|---|---|
    | head layout | `(B, T, H, d)` — "bthd" | SDPA wants `(B, H, T, d)` → `.transpose(1, 2)` |
    | split heads | `x.reshape(B, T, H, d)` | `x.view(B, T, H, d)` (needs contiguous memory) |
    | causal mask | `jnp.where(mask, s, -jnp.inf)` | `s.masked_fill(~mask, float("-inf"))` |
    | built-in | `jax.nn.dot_product_attention(q, k, v, is_causal=True)` | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` |
    | pick kernel | `implementation="xla"` / `"cudnn"` (NVIDIA) | `with sdpa_kernel(SDPBackend.FLASH_ATTENTION):` |
    | fused kernels on this card | none (XLA ops) | AOTriton flash/efficient, with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` |
    | no gradients | (nothing — grads only exist where you call `grad`) | `with torch.no_grad():` |
    | einsum | `jnp.einsum("bthd,bshd->bhts", q, k)` | `torch.einsum("bthd,bshd->bhts", q, k)` — identical |

    The math is identical; the main practical differences are the tensor layout SDPA expects, and that
    PyTorch on ROCm ships fused attention kernels (experimental on this card, behind a flag) while JAX's
    built-in falls back to XLA here.

    **Next:** Module 16 — a full GPT, trained in both frameworks.
    """)
    return


if __name__ == "__main__":
    app.run()
