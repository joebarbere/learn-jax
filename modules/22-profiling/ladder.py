"""Module 22 machinery: the speed ladder, profiler-trace parsing and HLO parsing.

Loaded by lesson.py with importlib (like Module 18's infer.py). The model is *your* Module 16 GPT
(modules/16-gpt/exercises.py), or the reference solution if yours isn't finished yet.
"""
from __future__ import annotations

import glob
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "runs" / "22-profiling"

PEAK_BF16 = 27e12   # Module 00's measured bf16 matmul throughput at n=4096 (TFLOP/s ×1e12)
PEAK_F32 = 8.5e12   # Module 00's measured float32 matmul throughput
BANDWIDTH = 288e9   # RX 7600 XT memory bandwidth, bytes/s (spec: 18 Gbps × 128-bit)

# The ladder's model: GPT-2's vocabulary (so "nice numbers" has something to fix) on a small body.
VOCAB, D_MODEL, N_LAYERS, N_HEADS, SEQ, BATCH = 50257, 384, 6, 6, 256, 8


def enable_compilation_cache(path: str = "/cache/jaxcache") -> None:
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def load_gpt():
    """Your Module 16 code if it works, else the reference solution. Returns (module, note)."""
    sys.path.insert(0, str(ROOT))
    from learnjax.course import load_exercises, using_solutions

    if using_solutions():
        return load_exercises(ROOT / "modules" / "16-gpt"), "the Module 16 reference GPT (LEARNJAX_SOLUTIONS=1)"
    g = load_exercises(ROOT / "modules" / "16-gpt")
    try:
        tiny = g.GPTConfig(vocab=17, seq_len=8, d_model=16, n_layers=1, n_heads=2)
        p = g.init_gpt(jax.random.key(0), tiny)
        float(g.loss_fn(p, jnp.zeros((2, 9), jnp.int32), tiny))
        return g, "your Module 16 GPT"
    except Exception:  # noqa: BLE001 - not finished yet
        return load_exercises(ROOT / "modules" / "16-gpt", solutions=True), \
            "the Module 16 reference GPT (finish `make test N=16` to profile your own)"


def ladder_config(g, vocab: int = VOCAB):
    return g.GPTConfig(vocab=vocab, seq_len=SEQ, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS)


def make_batches(n: int, vocab: int = VOCAB, seed: int = 0):
    """Random token windows: speed doesn't depend on what the tokens say."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.integers(0, vocab, (n, BATCH, SEQ + 1), dtype=np.int32))


def make_step(g, cfg, opt, bf16: bool):
    def loss(p, b):
        if bf16:  # mixed precision: f32 master weights, bf16 compute (Module 08)
            p = jax.tree.map(lambda a: a.astype(jnp.bfloat16), p)
        return g.loss_fn(p, b, cfg)

    def step(p, s, b):
        value, grads = jax.value_and_grad(loss)(p, b)
        updates, s = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s, value
    return step


def make_scan(step, donate: bool = True):
    def multi(p, s, bs):
        def body(carry, b):
            p, s, value = step(*carry, b)
            return (p, s), value
        (p, s), values = jax.lax.scan(body, (p, s), bs)
        return p, s, values
    return jax.jit(multi, donate_argnums=(0, 1) if donate else ())


def _timed_loop(step, params, state, batches, mode: str):
    """mode: 'sync' reads every loss right away; 'deferred' reads the previous step's loss."""
    t0 = time.perf_counter()
    prev = None
    for b in batches:
        params, state, value = step(params, state, b)
        if mode == "sync":
            float(jnp.mean(value))
        else:
            if prev is not None:
                float(jnp.mean(prev))
            prev = value
    jax.block_until_ready(params)
    return time.perf_counter() - t0, params, state


def run_rung(g, rung: str, n_steps: int = 24, warmup: int = 2, repeats: int = 3):
    """One rung of the ladder. Returns a dict with tok/s, ms/step and first-call (compile) seconds.

    The timed loop runs `repeats` times and the fastest is kept: on a GPU shared with other work,
    the best of a few runs is the most repeatable number.
    """
    nice = rung in ("nice", "donate", "deferred", "scan")
    bf16 = rung not in ("eager", "jit_f32")
    cfg = ladder_config(g, vocab=math.ceil(VOCAB / 128) * 128 if nice else VOCAB)
    opt = optax.adamw(3e-4)
    params = g.init_gpt(jax.random.key(0), cfg)
    state = opt.init(params)
    step = make_step(g, cfg, opt, bf16)
    if rung == "eager":
        n_steps, warmup = 3, 1          # op-by-op dispatch is slow; a few steps tell the story
    batches = make_batches(4 + n_steps if rung == "scan" else warmup + n_steps)
    if rung == "eager":
        fn, mode = step, "sync"
    elif rung in ("jit_f32", "bf16", "nice"):
        fn, mode = jax.jit(step), "sync"
    elif rung == "donate":
        fn, mode = jax.jit(step, donate_argnums=(0, 1)), "sync"
    elif rung == "deferred":
        fn, mode = jax.jit(step, donate_argnums=(0, 1)), "deferred"
    elif rung == "scan":
        k = 4
        fn, mode = make_scan(step), "deferred"
        batches = batches.reshape(-1, k, *batches.shape[1:])
        warmup = 1
    else:
        raise ValueError(rung)
    t0 = time.perf_counter()
    params, state, value = fn(params, state, batches[0])
    float(jnp.mean(value))
    first = time.perf_counter() - t0
    for b in batches[1:warmup]:
        params, state, value = fn(params, state, b)
    float(jnp.mean(value))
    times = []
    for _ in range(1 if rung == "eager" else repeats):
        dt, params, state = _timed_loop(fn, params, state, batches[warmup:], mode)
        times.append(dt)
    dt = min(times)
    n_timed = len(batches[warmup:]) * (4 if rung == "scan" else 1)
    tokens = n_timed * BATCH * SEQ
    return {"rung": rung, "tok_s": tokens / dt, "ms_step": 1e3 * dt / n_timed, "first_call_s": first,
            "spread": max(times) / min(times) - 1,
            "n_params": int(sum(x.size for x in jax.tree.leaves(params))), "vocab": cfg.vocab}


RUNGS = [
    ("eager", "0 · eager f32: no jit, read the loss every step"),
    ("jit_f32", "1 · jit the whole step (f32)"),
    ("bf16", "2 · + bf16 compute, f32 master weights"),
    ("nice", "3 · + nice numbers: vocab 50257 → 50304"),
    ("donate", "4 · + donate params & optimizer state"),
    ("deferred", "5 · + deferred metrics (no per-step sync)"),
    ("scan", "6 · + lax.scan: 4 steps per dispatch"),
]


# ------------------------------------------------------------------------------------------------
# Profiler traces
# ------------------------------------------------------------------------------------------------
def time_batched_gemm(n_iter: int = 100) -> float:
    """Achieved TFLOP/s of one attention-score GEMM ([48,256,64] @ [48,64,256], bf16), timed directly."""
    a = jax.random.normal(jax.random.key(0), (BATCH * N_HEADS, SEQ, D_MODEL // N_HEADS), jnp.bfloat16)
    b = jax.random.normal(jax.random.key(1), (BATCH * N_HEADS, D_MODEL // N_HEADS, SEQ), jnp.bfloat16)
    zero = jnp.zeros((BATCH * N_HEADS, SEQ, SEQ), jnp.bfloat16)
    f = jax.jit(lambda a, b: jax.lax.fori_loop(0, n_iter, lambda i, acc: acc + a @ b, zero))
    f(a, b).block_until_ready()
    t0 = time.perf_counter()
    f(a, b).block_until_ready()
    dt = (time.perf_counter() - t0) / n_iter
    return 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2] / dt / 1e12


def capture_trace(name: str, run_steps) -> Path:
    """Run `run_steps(annotate)` under jax.profiler.trace; return the .xplane.pb path."""
    out = RUNS / "trace" / name
    shutil.rmtree(out, ignore_errors=True)
    with jax.profiler.trace(str(out), create_perfetto_trace=True):
        run_steps(lambda i: jax.profiler.StepTraceAnnotation("train", step_num=i))
    return Path(glob.glob(str(out / "**" / "*.xplane.pb"), recursive=True)[0])


def parse_xplane(path) -> tuple[list[dict], list[tuple[float, float]]]:
    """GPU kernel events (µs) and host step spans from a profile. Uses JAX's own ProfileData reader."""
    pd = jax.profiler.ProfileData.from_file(str(path))
    kernels, steps = [], []
    for plane in pd.planes:
        if plane.name.startswith("/device:GPU"):
            for line in plane.lines:
                for e in line.events:
                    st = dict(e.stats)
                    kernels.append({"name": e.name, "start": e.start_ns / 1e3, "dur": e.duration_ns / 1e3,
                                    "hlo_op": str(st.get("hlo_op", "")), "details": str(st.get("kernel_details", ""))})
        elif plane.name == "/host:CPU":
            for line in plane.lines:
                for e in line.events:
                    if e.name.startswith("train"):
                        steps.append((e.start_ns / 1e3, e.end_ns / 1e3))
    kernels.sort(key=lambda k: k["start"])
    return kernels, sorted(steps)


# ------------------------------------------------------------------------------------------------
# Compiled HLO
# ------------------------------------------------------------------------------------------------
_DT_BYTES = {"bf16": 2, "f16": 2, "f32": 4, "s32": 4, "u32": 4, "s8": 1, "u8": 1, "pred": 1, "f64": 8, "s64": 8}
_SHAPE = re.compile(r"\b(bf16|f16|f32|f64|s8|u8|s32|u32|s64|pred)\[([0-9,]*)\]")
_DEF = re.compile(r"^\s*(?:ROOT\s+)?%([\w.\-]+)\s*=\s*(.+?)\s+([a-z][\w\-]*)\((.*)$")


def _shapes(text: str):
    out = []
    for dt, dims in _SHAPE.findall(text):
        n = 1
        for d in filter(None, dims.split(",")):
            n *= int(d)
        out.append((dt, n))
    return out


def parse_hlo_ops(hlo_text: str) -> dict[str, dict]:
    """Per GEMM call and fusion: FLOPs and bytes estimated from shapes in compiled HLO text.

    Compiled HLO names operands without shapes (`custom-call(%a, %b)`), so a first pass records every
    instruction's output shape. GEMM: output O (b·M·N), operands A (b·M·K), B (b·K·N) →
    FLOPs = 2·sqrt(A·B·O) = 2·b·M·K·N, exact for plain and batched matmuls. Fusions: ~1 FLOP per
    output element (the point is their low intensity, not the exact count). Bytes: operands + output.
    """
    defs, shapes = [], {}
    for line in hlo_text.splitlines():
        m = _DEF.match(line)
        if m:
            name, out_txt, op, rest = m.groups()
            shapes[name] = _shapes(out_txt)
            defs.append((name, op, rest))
    ops = {}
    for name, op, rest in defs:
        if op not in ("custom-call", "fusion"):
            continue
        args = rest.split(")", 1)[0]
        ins = [shapes.get(a, [None])[0] for a in re.findall(r"%([\w.\-]+)", args)]
        ins = [x for x in ins if x]
        outs = shapes[name]
        if not outs:
            continue
        if op == "custom-call":
            if "matmul" not in rest and "gemm" not in rest.lower():
                continue
            if len(ins) < 2:
                continue
            (dt_o, n_o), (dt_a, n_a), (dt_b, n_b) = outs[0], ins[0], ins[1]
            flops = 2 * math.sqrt(n_a * n_b * n_o)
            nbytes = n_a * _DT_BYTES[dt_a] + n_b * _DT_BYTES[dt_b] + n_o * _DT_BYTES[dt_o]
            ops[name] = {"kind": "gemm", "flops": flops, "bytes": nbytes, "out": outs[0]}
        else:
            n_o = sum(n for _, n in outs)
            nbytes = sum(n * _DT_BYTES[dt] for dt, n in ins + outs)
            ops[name] = {"kind": "fusion", "flops": float(n_o), "bytes": nbytes, "out": outs[0]}
    return ops


# ------------------------------------------------------------------------------------------------
# External tools, run in subprocesses with timeouts
# ------------------------------------------------------------------------------------------------
def rocprofv3_probe(timeout: int = 90) -> dict:
    """Try AMD's rocprofv3 on a tiny JAX matmul. Returns {'available', 'ok', 'detail'}."""
    exe = shutil.which("rocprofv3")
    if not exe:
        return {"available": False, "ok": False, "detail": "rocprofv3 is not on PATH in this image"}
    out = RUNS / "rocprof"
    shutil.rmtree(out, ignore_errors=True)
    script = ("import jax, jax.numpy as jnp\nf = jax.jit(lambda a: (a @ a).sum())\n"
              "a = jnp.ones((1024, 1024), jnp.bfloat16)\nfor _ in range(3): f(a).block_until_ready()\n")
    cmd = [exe, "--kernel-trace", "--stats", "--output-format", "csv", "-d", str(out), "-o", "run", "--",
           sys.executable, "-c", script]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"available": True, "ok": False, "detail": f"timed out after {timeout} s"}
    stats = glob.glob(str(out / "**" / "*kernel_stats.csv"), recursive=True)
    if r.returncode == 0 and stats:
        return {"available": True, "ok": True, "detail": Path(stats[0]).read_text()[:2000]}
    fatal = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("F") or "ERROR" in ln]
    return {"available": True, "ok": False, "detail": (fatal[0] if fatal else f"exit code {r.returncode}")[:600]}


def triton_gemm_count(enable_triton: bool, timeout: int = 180) -> dict:
    """Compile (never run) the GPT's three matmul shapes in a subprocess; count Triton vs hipBLASLt GEMMs.

    The program is x @ W_qkv, gelu(x @ W_fc) and x @ W_te.T at the ladder's shapes. With
    `--xla_gpu_autotune_level=0` XLA does not test-run candidate kernels while compiling, so this
    dispatches nothing to the GPU even with Triton GEMM enabled.
    """
    flags = "--xla_gpu_autotune_level=0" + ("" if enable_triton else " --xla_gpu_enable_triton_gemm=false")
    code = f"""
import re, jax, jax.numpy as jnp
S = jax.ShapeDtypeStruct
x = S(({BATCH * SEQ}, {D_MODEL}), jnp.bfloat16)
ws = [S(({D_MODEL}, {3 * D_MODEL}), jnp.bfloat16), S(({D_MODEL}, {4 * D_MODEL}), jnp.bfloat16),
      S(({math.ceil(VOCAB / 128) * 128}, {D_MODEL}), jnp.bfloat16)]
f = jax.jit(lambda x, a, b, c: (x @ a, jax.nn.gelu(x @ b), x @ c.T))
txt = f.lower(x, *ws).compile().as_text()
print("TRITON", len(re.findall(r'"kind":"__triton\\w*gemm', txt)), "LIBRARY", txt.count('custom_call_target="__cublas'))
"""
    env = dict(os.environ, XLA_FLAGS=flags)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout,
                           env=env, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": f"timed out after {timeout} s"}
    m = re.search(r"TRITON (\d+) LIBRARY (\d+)", r.stdout)
    if not m:
        return {"ok": False, "detail": (r.stderr.strip().splitlines() or ["no output"])[-1][:400]}
    return {"ok": True, "triton": int(m.group(1)), "library_gemms": int(m.group(2))}


def xla_dump(timeout: int = 120) -> list[str]:
    """Dump XLA's compilation stages for a tiny function; return the file names produced."""
    out = RUNS / "xla_dump"
    shutil.rmtree(out, ignore_errors=True)
    code = ("import jax, jax.numpy as jnp\n"
            "f = jax.jit(lambda x, w: jax.nn.gelu(x @ w).sum())\n"
            "x = jnp.ones((256, 384), jnp.bfloat16); w = jnp.ones((384, 1536), jnp.bfloat16)\n"
            "f(x, w).block_until_ready()\n")
    flags = os.environ.get("XLA_FLAGS", "") + f" --xla_dump_to={out}"
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout,
                   env=dict(os.environ, XLA_FLAGS=flags), check=False)
    return sorted(p.name for p in out.glob("*lambda*")) if out.exists() else []


def roofline_trace(timeout: int = 900) -> dict:
    """Trace 4 steps of the bf16 donated step in a subprocess with XLA command buffers disabled.

    By default XLA on ROCm records a whole step into one command buffer (a HIP graph), and the trace
    then attributes every kernel to `command_buffer_N`. With `--xla_gpu_enable_command_buffer=` each
    kernel keeps its own HLO op name. Returns {'ok', 'kernels', 'hlo', 'n_steps'} or {'ok': False, 'detail'}.
    """
    import json
    out = RUNS / "trace" / "roofline"
    flags = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
    try:
        r = subprocess.run([sys.executable, str(Path(__file__)), "roofline", str(out)], capture_output=True,
                           text=True, timeout=timeout, env=dict(os.environ, XLA_FLAGS=flags), check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": f"timed out after {timeout} s"}
    if r.returncode != 0 or not (out / "kernels.json").exists():
        return {"ok": False, "detail": (r.stderr.strip().splitlines() or ["no output"])[-1][:400]}
    return {"ok": True, "kernels": json.loads((out / "kernels.json").read_text()),
            "hlo": (out / "step.hlo.txt").read_text(), "n_steps": 4}


def _roofline_main(out: Path) -> None:
    import json
    enable_compilation_cache()
    g, _ = load_gpt()
    cfg = ladder_config(g, vocab=math.ceil(VOCAB / 128) * 128)
    opt = optax.adamw(3e-4)
    p = g.init_gpt(jax.random.key(0), cfg)
    s = opt.init(p)
    batches = make_batches(6)
    step = jax.jit(make_step(g, cfg, opt, True), donate_argnums=(0, 1)).lower(p, s, batches[0]).compile()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    (out / "step.hlo.txt").write_text(step.as_text())
    for b in batches[:2]:
        p, s, v = step(p, s, b)
    float(v)
    with jax.profiler.trace(str(out / "trace")):
        for b in batches[2:]:
            p, s, v = step(p, s, b)
        jax.block_until_ready(p)
    kernels, _ = parse_xplane(glob.glob(str(out / "trace" / "**" / "*.xplane.pb"), recursive=True)[0])
    (out / "kernels.json").write_text(json.dumps(kernels))


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "roofline":
    _roofline_main(Path(sys.argv[2]))
