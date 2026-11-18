# JAX on ROCm: gotchas

Everything this course tripped over running JAX (and PyTorch, llama.cpp and LM Studio) on an AMD Radeon
RX 7600 XT (Navi 33, **gfx1102**, 16 GB) under ROCm, in podman containers. Each entry is
**symptom → cause → fix**, with the module that shows it. Numbers are from this card, measured when
the module was written; "stated" means the course explains it but didn't isolate it with a measurement.

Generic JAX/XLA traps that bite on any backend are collected at the end, because on a new platform
you can't tell at first which kind you've hit.

## The short version

The course image (`containers/Containerfile`) sets three variables. Most of this page explains why.

| Variable | Value | Why |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | JAX's ROCm plugin doesn't accept gfx1102; pretend to be gfx1100 ([1](#1-jax-doesnt-see-the-gpu-no-supported-devices-found)) |
| `XLA_FLAGS` | `--xla_gpu_enable_triton_gemm=false` | Triton matmul fusions can exceed this chip's limits and hang the GPU ([3](#3-the-gpu-hangs-mid-training-invalid_dispatch_parameters)) |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Don't grab 75% of VRAM per process; notebooks share the card ([6](#6-a-second-notebook-runs-out-of-memory)) |

PyTorch's ROCm build, llama.cpp and LM Studio need **none** of these: they support gfx1102 natively.

---

## Setup and environment

### 1. JAX doesn't see the GPU: "no supported devices found"
- **Symptom:** `no supported devices found for platform ROCM`; JAX falls back to the CPU.
- **Cause:** ROCm supports gfx1102, but JAX's ROCm plugin only accepts a fixed list of architectures.
- **Fix:** `HSA_OVERRIDE_GFX_VERSION=11.0.0`. The runtime then reports **gfx1100** (RX 7900); the instruction
  sets are close enough that the code runs. (`11.0.1` failed the same way.)
- **Where:** Module 00 (shows the reported ISA), Containerfile.

### 2. The override has a cost: kernels are compiled for a bigger chip
- **Symptom:** none by itself. It's the root of [3](#3-the-gpu-hangs-mid-training-invalid_dispatch_parameters).
- **Cause (stated, most likely):** with the override, XLA and Triton target gfx1100, which has a 50% larger
  vector register file per SIMD than gfx1102 (1,536 vs 1,024 entries). A 1,024-thread workgroup can use up to
  96 VGPRs per thread on gfx1100 but only 64 on gfx1102, so a kernel that is legal for the target can be
  illegal on the real card.
- **Where:** Module 22 §8, Module 23 (occupancy model with a gfx1102/gfx1100 switch).

### 3. The GPU hangs mid-training: INVALID_DISPATCH_PARAMETERS
- **Symptom:**
  ```
  HSA_STATUS_ERROR_INVALID_DISPATCH_PARAMETERS: Kernel dispatch packet parameters exceed hardware limits
  for this agent ... grid=[65536, 1, 1], workgroup=[1024, 1, 1] ... kernel: input_add_reduce_fusion_9
  ```
  then the GPU hangs; the desktop can freeze for seconds until the process dies or the driver resets.
- **Cause:** XLA's Triton-generated matmul fusions ask for more resources than Navi 33 allows (see 2).
  First seen training the deepseek-v3-jax model on this card.
- **Fix:** `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`. Matmuls then go to hipBLASLt/rocBLAS, which are
  built and tuned per real target, gfx1102 included. Triton still runs other fusions (softmax) fine.
- **Where:** Module 22 §8 reproduces it safely (`LEARNJAX_RISKY=1 make lesson N=22`: subprocess, 120 s timeout).

### 4. Tests pass, full-size training hangs
- **Cause:** whether an oversized fusion is generated depends on the model's dimensions; small test models
  get small workgroups. A suite passing on tiny inputs says little about full-size runs.
- **Where:** Modules 00 and 22.

### 5. Setting XLA_FLAGS yourself removes the fix
- **Cause:** `containers/run.sh` forwards your host's `XLA_FLAGS` into the container, *replacing* the image's
  value (and the Triton-GEMM fix with it).
- **Fix:** append, don't replace: `XLA_FLAGS="$XLA_FLAGS --xla_dump_to=..."` (as `modules/22-profiling/ladder.py` does).

### 6. A second notebook runs out of memory
- **Cause:** by default JAX preallocates 75% of GPU memory per process.
- **Fix:** `XLA_PYTHON_CLIENT_PREALLOCATE=false` (allocate on demand). Module 00 still reports the 75% cap
  from `memory_stats()["bytes_limit"]`: it's a ceiling, no longer a reservation.

### 7. Container access to the GPU
- **Fix:** `--device /dev/kfd --device /dev/dri --security-opt label=disable --group-add keep-groups --ipc=host`
  (`containers/run.sh`), and a host user in the `render`/`video` groups (Fedora's default). ROCm stays in
  the container; nothing is installed on the host.

---

## Missing features (compared with CUDA)

### 8. No fused attention in JAX
- **Symptom:** `jax.nn.dot_product_attention(implementation="cudnn")` fails; the default falls back to plain
  XLA, at hand-written-attention speed, storing the full T×T score matrix.
- **Cause:** the fused path is cuDNN, NVIDIA-only.
- **Consequence:** with fused attention and `torch.compile`, PyTorch beats JAX on Module 16's GPT on this card.
  A Pallas flash-attention kernel is the missing rung (Modules 22, 23).

### 9. PyTorch's fast attention is off by default
- **Symptom:** only the `math` SDPA backend; forcing flash/efficient raises "No available kernel".
- **Cause:** AOTriton's fused kernels are marked experimental on consumer RDNA3.
- **Fix:** `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, set before torch first runs attention (it's read once).
  The image deliberately doesn't set it; notebooks opt in. If a long run ever hangs, unset it first.
- **Measured** (Module 23, fp32 math vs fp16 flash): T=1,024 0.73 → 0.11 ms; T=4,096 14.3 → 1.64 ms and
  580 → 2 MiB. Module 15: about 40× faster and 27× less memory at T=4,096.

### 10. rocprofv3 can't profile JAX
- **Symptom:** rocprofv3 aborts at start-up.
- **Cause (stated):** the ROCm SDK profiler and the JAX plugin disagree on the profiler-registration interface.
- **Fix:** `jax.profiler.trace(..., create_perfetto_trace=True)` works out of the box and maps each kernel to
  its HLO op. You lose hardware counters. (Module 22.)

### 11. Pallas fails until you ask for Triton
- **Symptom:** Pallas tries its Mosaic GPU backend and errors.
- **Fix:** `compiler_params=pltriton.CompilerParams()` (`from jax.experimental.pallas import triton as pltriton`).
  `interpret=True` runs anywhere (the exercises' tests use it). Module 23 probes real kernels in a
  subprocess with a timeout, so a hang can't freeze the notebook.

### 12. Pallas: RESOURCE_EXHAUSTED on LDS
- **Symptom:** the kernel asks for 131,072 bytes of LDS; the limit is 65,536 per workgroup.
- **Fix:** loop over K in chunks inside the kernel (`fori_loop` + `pl.ds`). (Module 23.)

---

## Performance traps

### 13. A cell compiles for minutes ("Very slow compile?")
- **Symptom:** XLA prints its `slow_operation_alarm` ("Very slow compile? If you want to file a bug…").
- **Causes, measured in Module 36's FFT benchmark** (an (L, 256, 16) kernel built as `A ** arange(L)` then summed):

  | version | compile at L = 1,024 |
  |---|---|
  | `**` with a broadcast exponent, arrays closed over (the original) | over 3 minutes |
  | `**`, arrays passed as arguments | 193 s |
  | `exp(k · log A)`, arrays closed over | 24 s (55 s at L = 4,096): constant folding |
  | `exp(k · log A)`, arrays passed as arguments | about 1 s |

  Two separate problems: a power with a broadcast exponent inside a reduction takes a very slow ROCm code
  path, and a large intermediate computed only from closed-over constants is constant-folded on the CPU
  at compile time.

- **Fix:** `exp(k * log(x))` for positive x, and pass arrays as jit arguments. When the alarm appears, find
  the op and fix it; lessons don't rely on a persistent cache to hide it.

### 14. Dilated 1-D convolution is the slow path
- **Symptom:** `lax.conv_general_dilated` with `rhs_dilation` takes minutes where two shifted matmuls take seconds.
- **Fix:** write it as shifted matmuls. (Module 12.)

### 15. fp32 matmuls are 3× slower than bf16
- **Cause:** RDNA3's WMMA matrix units only take 16-bit inputs.
- **Measured:** about 27 TFLOP/s bf16 vs 8.5 TFLOP/s fp32 at n = 4,096 (Module 22). A Pallas tiled matmul gets
  near hipBLASLt in bf16 but falls far behind in fp32 (Module 23).

### 16. "Nice number" padding didn't help
- Padding GPT-2's vocabulary 50,257 → 50,304 gave about 4% on NVIDIA; here it was within noise. hipBLASLt on
  RDNA3 picks tiles differently: measure rather than assume. (Module 22.)

### 17. int8 weights didn't speed up decoding
- **Symptom:** under 1.05× over bf16.
- **Cause (stated, two candidates):** a busy GPU, or XLA converting int8 → bf16 into a temporary before each
  matmul, reading the bytes twice. (Module 18.)

### 18. Most of a cold compile is autotuning
- **Cause:** XLA times candidate matmul kernels to pick the fastest.
- **Fix:** a persistent cache for long runs: `jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")`
  and `jax_persistent_cache_min_compile_time_secs=1.0` (Module 16's `train.py`; Module 25: the first run
  compiles for about a minute).
- **Measured** (Module 40): a 25-step `lax.scan` training chunk of a small MLA + MoE model took 49 s to
  compile cold and 1.8 s from the persistent cache in a fresh process. The scan length barely matters
  (after the first compile, a 100-step chunk compiled and ran in 4 s).

### 19. Measuring a huge shape OOMs during autotuning
- **Fix:** compile with autotuning off, then read the memory analysis:
  `jax.jit(f).lower(x).compile(compiler_options={"xla_gpu_autotune_level": 0}).memory_analysis()`. (Module 23.)
- Same flag to *inspect* code that might hang: autotuning test-runs candidate kernels on the GPU, so with
  Triton GEMM enabled a bad kernel can launch before your first step (Module 22).

### 20. A fused gather + projection page-faulted the GPU
- **Symptom:** a GPU page fault under `jit` + `scan` + `grad`.
- **Cause:** gathering rows of the frozen 49,152 × 576 bf16 table and projecting them in one fused kernel
  (under `jit` + `scan` + `grad`). It faulted reliably, on an idle GPU too: a ROCm code-generation bug.
- **Fix:** project the frozen table first, then gather: `(table.astype(jnp.bfloat16) @ W)[ids]`. (Module 38.)

---

### 20b. "Failed to set shared memory size" (hipError_t 98)
- **Symptom:** a jitted function that scores a corpus and takes `top_k` in one program fails at launch with
  `hipError_t(98)` "Failed to set shared memory size", depending on shape (an IVF gather + einsum + top-k; a
  sparse BCOO matmul + top-k over 235k rows). `jax.lax.optimization_barrier` between the two didn't help.
- **Fix:** split scoring and top-k into two jitted functions. (Module 38.)

### 20c. Pinned host memory works (good news)
- JAX 0.11.1 on ROCm 10.1 exposes `['device', 'pinned_host']` as the RX 7600 XT's memory kinds, and
  `jax.device_put(x, jax.sharding.SingleDeviceSharding(dev, memory_kind="pinned_host"))` works. Module 40
  keeps an Engram table in host RAM and prefetches its rows: within 1–2% of a GPU-resident table for a
  forward pass of 32 × 256 tokens. Gathers from a big f32 table under `jit` + `scan` + `grad` didn't hit
  #20's page fault there.

## Reading profiles and HLO on ROCm

### 21. Every kernel is called `command_buffer_N`
- **Cause:** XLA on ROCm records a whole step into one HIP graph.
- **Fix:** `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (empty), in a subprocess, to see kernel names; you lose
  the launch-overhead saving. (Module 22.)

### 22. `__cublas$lt$matmul` on an AMD GPU
- A historical name: on ROCm it routes to hipBLASLt. In traces, GEMMs appear as Tensile `Cijk_…` kernels.

### 23. `cost_analysis()` misses the matmuls
- XLA's cost model can't see inside library calls; compute FLOPs from the shapes. (Module 22.)

---

## Sharing the card, and measuring

### 24. Timings move ±30% when anything else uses the GPU
- **Detect:** `cat /sys/class/drm/card*/device/gpu_busy_percent`. The course's bench JSONs record it at start.
- **Fix:** benchmark on an idle GPU; report the busy percentage with the number.

### 25. Several GPU jobs at once: slowdowns, and once a NaN and a segfault
- **Observed** (2026-11-15, building this course, not reproduced): with three or four training jobs sharing the
  card, a 6,000-step BERT run went NaN and took 81 minutes; run alone, the same code and seed trained cleanly in
  187 s. A lesson run segfaulted (exit 139) under similar load and passed twice when re-run.
- **Fix:** queue GPU jobs instead of overlapping them, e.g.
  `flock /tmp/gpu.lock containers/run.sh python train.py`. Training loops should stop on the first non-finite
  loss instead of saving NaN weights (Module 32's `pretrain`).

---

## Serving: llama.cpp, LM Studio, PyTorch

### 26. llama.cpp: Vulkan or ROCm?
| backend | needs | image | on this card |
|---|---|---|---|
| Vulkan (Mesa RADV) | `/dev/dri` only | 0.9 GB | no flash attention (disabled automatically) |
| ROCm/HIP | `/dev/kfd` + `/dev/dri` | 23 GB | gfx1102 native, no override |

- The ROCm image is 23 GB to Vulkan's 0.9 GB. It's pulled on first use, and can be removed and pulled again
  exactly (pinned by digest); see the README's *Disk space* section.
- **Measured** (Module 19): ROCm is 1.2–2× faster at prompt processing; Vulkan is 7–29% faster at generation.
  A 14B Q4_K_M model generates 27–29 tok/s, about 90% of the memory bus. The course defaults to Vulkan.

### 27. LM Studio: pin the engine version
- Both engines run natively on gfx1102. The image pins llama.cpp runtime 2.73.0; `lms runtime get … --allow-incompatible`
  is needed at build time (no GPU during the build makes every engine look incompatible), and after
  `lms daemon up` you retry `lms runtime select` while it indexes engines.
- Community reports (cited, not reproduced) describe this card hanging with a self-built ROCm llama.cpp and an
  automatic engine update breaking GPU use, hence the pin.
- **Measured** (Module 27): Vulkan ~235 vs ROCm ~184 tok/s at bf16, ~320 vs ~220 at 4-bit; LM Studio's serving
  overhead is a few percent (235 vs llama-bench's 255).

### 28. PyTorch HIP extensions
- `hipcc` is in the PyTorch image and gfx1102 is native; set `PYTORCH_ROCM_ARCH=gfx1102` or torch warns and
  builds for all visible archs. `torch.compile` (Triton) works; `device="cuda"` means the Radeon.
- **Measured** (Module 23, 8192×4096 fp32): eager 2.06 ms (130 GB/s), hand-written HIP kernel 1.03 ms (262 GB/s),
  `torch.compile` 1.03 ms.

---

## Generic JAX/XLA traps (any backend)

| trap | fix | module |
|---|---|---|
| Async dispatch: naive timing measures the launch, not the work ("impossible" TFLOP/s) | `jax.block_until_ready` | 06 |
| Excess precision: under jit, `x.astype(bf16).astype(f32)` is optimized away | `lax.reduce_precision(x, 8, 7)` | 08 |
| `jax_debug_nans` ignored inside marimo | `with jax.debug_nans(True):` | 11 |
| A new cache shape recompiles (an early "68 tok/s" was this) | time with the warm-up's exact shapes | 18 |
| Python decode loop syncs every token (`int(tok)`) | `lax.fori_loop` on device | 18 |
| Closing over big arrays bakes them into the program | pass them as arguments | 10, 36 |
| Python globals are captured at trace time, silently | pass values as arguments | 06 |
| Compiling once per new shape can be slower than eager | fixed shapes; persistent cache | 06 |
| Tiny models are overhead-bound (~10 µs of math per step) | move the loop into `lax.scan` (9× in Module 10) | 10, 22 |
| Donated buffers become deleted | `donate_argnums` is a memory win; don't reuse the old reference | 22 |

Tooling notes: Flax 0.12 reads parameters as `module.kernel[...]` (not `.value`); marimo swallows
`<|endoftext|>`-style tokens (escape them); Hugging Face metadata calls can hang a notebook offline
(`HF_HUB_ETAG_TIMEOUT`, `HF_HUB_DOWNLOAD_TIMEOUT`); a multi-line value interpolated into indented
`mo.md(f"""…""")` nested deeper than the cell body renders as code blocks (`tools/check_md_indent.py` finds it).
