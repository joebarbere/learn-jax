import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="23 · Kernels (PyTorch)")


@app.cell
def _():
    import os

    # Opt in to PyTorch's experimental fused-attention kernels on RDNA3 (AOTriton). Must be set before
    # torch runs attention for the first time — so before importing torch. See Module 15.
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    # Build only for the card we have (otherwise torch warns and picks "all visible archs").
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1102")

    import shutil
    import time

    import marimo as mo
    import torch
    import torch.nn.functional as F

    dev = torch.device("cuda")  # on ROCm builds, "cuda" means "the GPU" — the AMD card here
    return F, dev, mo, os, shutil, time, torch


@app.cell
def _(mo, shutil, torch):
    mo.md(f"""
    # 23 · Kernels, the PyTorch side: HIP and fused attention

    The JAX lesson can write kernels in **Pallas**. PyTorch's equivalent escape hatch is to write the
    kernel yourself in **HIP** (AMD's CUDA-like C++) and compile it at runtime with
    `torch.utils.cpp_extension.load_inline`. This notebook checks that works on the RX 7600 XT, measures a
    hand-written fused kernel against PyTorch eager and `torch.compile`, and then times PyTorch's fused
    attention kernels for comparison with the JAX lesson's tiled attention.

    | | |
    |---|---|
    | PyTorch | `{torch.__version__}` |
    | device | `{torch.cuda.get_device_name(0)}` |
    | architecture PyTorch sees | `{torch.cuda.get_device_properties(0).gcnArchName}` (native — no override needed) |
    | `hipcc` | `{shutil.which("hipcc")}` |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · A fused bias + GELU kernel in HIP

    `F.gelu(x + b)` in eager mode launches **two** kernels: one writes `x + b` to memory, the next reads
    it back. Both are memory-bound, so the extra round trip costs real time. A fused kernel reads `x`
    once and writes the result once. Below are two versions: a straightforward one (one element per
    thread) and one that loads four floats at a time (`float4`), which is how good kernels keep the
    memory bus full. Compilation takes ~30 s the first time; the build is cached in the course volume.
    """)
    return


@app.cell
def _(os, time):
    from torch.utils.cpp_extension import load_inline

    _cuda_src = r"""
    #include <torch/extension.h>

    __device__ __forceinline__ float gelu_tanh(float v) {
      return 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }

    // one element per thread
    __global__ void bias_gelu_kernel(const float* __restrict__ x, const float* __restrict__ b,
                                     float* __restrict__ y, long n, int d) {
      long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) y[i] = gelu_tanh(x[i] + b[i % d]);
    }

    // four elements per thread with 16-byte vector loads/stores (needs d % 4 == 0)
    __global__ void bias_gelu_vec4_kernel(const float4* __restrict__ x, const float4* __restrict__ b,
                                          float4* __restrict__ y, long n4, int d4) {
      long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n4) {
        float4 v = x[i], bb = b[i % d4];
        y[i] = make_float4(gelu_tanh(v.x + bb.x), gelu_tanh(v.y + bb.y),
                           gelu_tanh(v.z + bb.z), gelu_tanh(v.w + bb.w));
      }
    }

    torch::Tensor bias_gelu(torch::Tensor x, torch::Tensor b) {
      auto y = torch::empty_like(x);
      long n = x.numel(); int t = 256;
      bias_gelu_kernel<<<(n + t - 1) / t, t>>>(x.data_ptr<float>(), b.data_ptr<float>(),
                                               y.data_ptr<float>(), n, (int)b.numel());
      return y;
    }

    torch::Tensor bias_gelu_vec4(torch::Tensor x, torch::Tensor b) {
      auto y = torch::empty_like(x);
      long n4 = x.numel() / 4; int t = 256;
      bias_gelu_vec4_kernel<<<(n4 + t - 1) / t, t>>>((const float4*)x.data_ptr<float>(),
          (const float4*)b.data_ptr<float>(), (float4*)y.data_ptr<float>(), n4, (int)b.numel() / 4);
      return y;
    }
    """
    _cpp_src = """
    torch::Tensor bias_gelu(torch::Tensor x, torch::Tensor b);
    torch::Tensor bias_gelu_vec4(torch::Tensor x, torch::Tensor b);
    """
    _build = "/cache/torch_ext/bias_gelu"
    os.makedirs(_build, exist_ok=True)
    _t = time.perf_counter()
    hip_ext = load_inline("bias_gelu_ext", cpp_sources=_cpp_src, cuda_sources=_cuda_src,
                          functions=["bias_gelu", "bias_gelu_vec4"], build_directory=_build,
                          extra_cuda_cflags=["-O3"], verbose=False)
    build_seconds = time.perf_counter() - _t
    return build_seconds, hip_ext


@app.cell
def _(F, build_seconds, dev, hip_ext, mo, time, torch):
    _x = torch.randn(8192, 4096, device=dev)
    _b = torch.randn(4096, device=dev)
    _ref = F.gelu(_x + _b, approximate="tanh")
    _err = max((hip_ext.bias_gelu(_x, _b) - _ref).abs().max().item(),
               (hip_ext.bias_gelu_vec4(_x, _b) - _ref).abs().max().item())
    _compiled = torch.compile(lambda x, b: F.gelu(x + b, approximate="tanh"))

    def _bench(f, n=30):
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            f()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n

    _min_bytes = 2 * _x.numel() * 4  # read x once, write y once — the floor for any implementation
    _rows = []
    for _name, _f in [("PyTorch eager (2 kernels)", lambda: F.gelu(_x + _b, approximate="tanh")),
                      ("your HIP kernel, 1 float/thread", lambda: hip_ext.bias_gelu(_x, _b)),
                      ("your HIP kernel, float4", lambda: hip_ext.bias_gelu_vec4(_x, _b)),
                      ("torch.compile (Triton-generated)", lambda: _compiled(_x, _b))]:
        _s = _bench(_f)
        _rows.append(f"| {_name} | {_s * 1e3:.3f} | {_min_bytes / _s / 1e9:.0f} |")
    _hip_rows = "\n    ".join(_rows)
    mo.md(f"""
    Built in **{build_seconds:.1f} s** (≈0 when cached). Max difference from PyTorch's GELU:
    **{_err:.1e}**. Input: 8192 × 4096 float32 (128 MiB).

    | implementation | ms | effective GB/s (2 × 128 MiB ÷ time) |
    |---|---|---|
    {_hip_rows}

    The card's memory bandwidth is about **288 GB/s**. "Effective" divides the *minimum* traffic by the
    time, so eager's two-kernel version looks half as fast because it really moves twice the bytes.
    When this was written on an idle GPU, all three fused versions landed at ~260 GB/s — about 90% of
    peak — and `float4` loads made no difference: this kernel is so memory-bound that even the simple
    version saturates the bus (vector loads matter more when each thread does very little, or for
    16-bit data). The lesson: **fusion** was the whole win, and `torch.compile` found it for free.
    Write kernels by hand only when the compiler can't express the fusion you need.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · PyTorch's fused attention vs the JAX lesson's tiled attention

    `F.scaled_dot_product_attention` picks a backend: **flash** and **memory-efficient** are single fused
    kernels (FlashAttention-style, via AOTriton on ROCm — experimental on RDNA3, enabled in the first cell);
    **math** is the unfused reference that builds the full score matrix. Same shapes as the JAX lesson:
    4 heads, head dim 64.
    """)
    return


@app.cell
def _(F, dev, mo, time, torch):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _measure(backend, T, dtype):
        q = torch.randn(1, 4, T, 64, device=dev, dtype=dtype)
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, q, q)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                t = time.perf_counter()
                for _ in range(10):
                    F.scaled_dot_product_attention(q, q, q)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t) / 10 * 1e3
                mib = (torch.cuda.max_memory_allocated() - base) / 2**20
            return f"{ms:.2f} ms / {mib:.0f} MiB"
        except RuntimeError as e:
            return "n/a (" + str(e).split("\n")[0][:40] + ")"

    _rows = []
    for _T in (512, 1024, 2048, 4096):
        _rows.append(f"| {_T} | {_measure(SDPBackend.MATH, _T, torch.float32)} | "
                     f"{_measure(SDPBackend.FLASH_ATTENTION, _T, torch.float16)} | "
                     f"{_measure(SDPBackend.EFFICIENT_ATTENTION, _T, torch.float16)} |")
    _sdpa_rows = "\n    ".join(_rows)
    mo.md(f"""
    | T | math (fp32) | flash (fp16) | memory-efficient (fp16) |
    |---|---|---|---|
    {_sdpa_rows}

    **JAX on the same card** (Module 23 lesson, fp32, 4 heads, measured when this was written — this image
    has no JAX): plain XLA attention 0.23 / 0.45 / 1.78 / 5.6 ms and the `lax.scan` tiled version
    0.62 / 1.35 / 4.3 / 20 ms at T = 512 / 1024 / 2048 / 4096, with compile-time scratch memory of
    512 MiB (plain) vs 24 MiB (tiled) at T = 4096.

    (The fused kernels run in fp16 — they don't support fp32 — so compare their *shape* more than their
    exact times with the fp32 columns. Notice also that XLA's unfused attention beats PyTorch's unfused
    "math" path at every length.)

    The fused kernels win on both axes because the whole loop over key blocks happens *inside one kernel*,
    with the running max, denominator and accumulator held in registers and LDS. The JAX `lax.scan`
    version has the same memory scaling but launches a separate small matmul per block, so the GPU sits
    mostly idle. Closing that gap is what a Pallas FlashAttention kernel is for (see the lesson).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaways

    * Custom HIP kernels **build and run** on the RX 7600 XT from inside the PyTorch container
      (`hipcc` included, gfx1102 native).
    * Fusion is the main win; vectorized memory access is the next. Compilers (`torch.compile`, XLA)
      usually get both for you.
    * Attention is where compilers can't fuse across the softmax — fused attention kernels are the
      hand-written exception every framework ships.

    **Back to the JAX lesson:** `make lesson N=23`.
    """)
    return


if __name__ == "__main__":
    app.run()
