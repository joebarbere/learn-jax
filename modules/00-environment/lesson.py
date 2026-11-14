import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="00 · Environment")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp

    import learnjax as lj
    from learnjax.gpu import gpu_info, matmul_tflops

    ex = lj.load_exercises(mo.notebook_dir())
    return ex, gpu_info, jax, jnp, lj, matmul_tflops, mo


@app.cell
def _(mo):
    mo.md(r"""
    # 00 · Your GPU, the container, and the three ROCm settings

    **Goal:** know exactly what is between your Python code and the RX 7600 XT, why each layer is there,
    and how fast the card really is. Every later module assumes this works.

    **You will:** inspect the device JAX sees · learn the three settings this card needs and what breaks
    without them · measure matrix-multiply throughput in float32 and bfloat16 · do the memory arithmetic
    that decides what models you can train in 16 GB.

    **How this course works:** each module is a *lesson* (this notebook — read it, run it, poke the sliders),
    plus *exercises* in `exercises.py` that the lesson calls. When a cell says **Your turn**, open
    `exercises.py`, implement the function, and the cell will update. `make test N=00` grades you;
    `make progress` shows the whole syllabus.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The stack

    When you write `jnp.dot(a, b)`, it passes through these layers:

    | Layer | What it is | Where it lives |
    |---|---|---|
    | Your code | `jax.numpy`, `jax.grad`, `jax.jit` … | this notebook |
    | **JAX** | traces Python into a graph of array operations | pip package, from the base image |
    | **XLA** | compiler: fuses operations and generates GPU kernels | inside `jaxlib` |
    | **PJRT ROCm plugin** | JAX's bridge to AMD GPUs | `jax-rocm10-plugin` |
    | **ROCm 10.1 / HIP runtime** | AMD's GPU runtime, math libraries (hipBLASLt, rocBLAS) | inside the container |
    | **amdgpu kernel driver** | the Linux driver, `/dev/kfd` + `/dev/dri` | your Fedora host kernel |
    | **RX 7600 XT** | Navi 33, *gfx1102*, 16 GB | PCIe slot |

    Only the bottom two are on the host. Everything ROCm lives in the container image, which is why
    nothing ROCm-related is installed on Fedora and why `containers/run.sh` passes `/dev/kfd` and
    `/dev/dri` through.
    """)
    return


@app.cell
def _(gpu_info, mo):
    info = gpu_info()
    mo.md(f"""
    ## 2 · What JAX sees

    | | |
    |---|---|
    | JAX version | `{info['jax']}` |
    | default backend | `{info['backend']}` |
    | devices | `{info['devices']}` |
    | device kind | `{info['device_kind']}` |
    | memory cap for JAX | `{info['memory_gb']} GiB` (75% of the card by default) |
    | ROCm agents | `{info.get('rocm_agents')}` |
    | gfx ISA reported | `{info.get('gfx_reported')}` |

    The backend should be `gpu` and the device `rocm:0` (a `RocmDevice`). Notice the reported ISA: **gfx1100**, not
    your card's real **gfx1102** — that's the first setting below at work.
    """)
    return (info,)


@app.cell
def _(info, mo):
    _env = info["env"]
    mo.md(f"""
    ## 3 · The three settings (baked into `containers/Containerfile`)

    **`HSA_OVERRIDE_GFX_VERSION={_env['HSA_OVERRIDE_GFX_VERSION']}`** — ROCm 10.1 itself supports gfx1102,
    but JAX's ROCm plugin only accepts a list of architectures, and yours isn't on it. Without this you get
    `no supported devices found for platform ROCM`. The override makes the runtime *report* gfx1100
    (RX 7900), which the plugin accepts; the instruction sets are close enough that the code runs.

    **`XLA_FLAGS={_env['XLA_FLAGS']}`** — XLA can generate its own matrix-multiply kernels with Triton.
    On this card, for full-size models, some of those kernels ask for more resources than the hardware
    allows; the launch fails with `HSA_STATUS_ERROR_INVALID_DISPATCH_PARAMETERS` and the GPU hangs.
    Tiny models don't trigger it, which is why the bug hid from the tests. With Triton GEMM off, XLA calls
    AMD's tuned libraries (hipBLASLt / rocBLAS) instead. Module 22 reproduces this on purpose and reads the
    failing kernel's launch parameters.

    **`XLA_PYTHON_CLIENT_PREALLOCATE={_env['XLA_PYTHON_CLIENT_PREALLOCATE']}`** — by default JAX grabs 75%
    of GPU memory at startup (fast, but greedy). Several notebooks share one card in this course, so
    memory is allocated on demand instead.

    *Lesson learned the hard way:* a test suite passing on tiny inputs says little about full-size runs.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "The JAX ROCm plugin doesn't list gfx1102 as supported": "a",
            "ROCm 10.1 has no gfx1102 support at all": "b",
            "It makes the card run faster": "c",
        },
        label="**Check yourself:** why does this card need `HSA_OVERRIDE_GFX_VERSION`?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _feedback = {
        "a": mo.callout(mo.md("Right. ROCm itself ships gfx1102 libraries; it's the JAX plugin that refuses "
                              "the card. The override changes what the runtime reports."), kind="success"),
        "b": mo.callout(mo.md("Not quite — ROCm 10.1 ships gfx1102 device libraries and `rocminfo` "
                              "identifies the card correctly. The gap is in JAX's plugin."), kind="warn"),
        "c": mo.callout(mo.md("No — it doesn't change performance. It only changes which architecture "
                              "the runtime reports, so JAX's plugin will accept the card."), kind="warn"),
    }
    _feedback.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · How fast is the card?

    Multiplying an $m\times k$ matrix by a $k\times n$ one takes $2mkn$ floating-point operations (each
    output is $k$ multiplies and $k$ adds). Time it, divide, and you get **TFLOP/s**. Neural networks
    are mostly matrix multiplies, so this one number predicts a lot.

    Two things to notice when you move the slider:
    * **bfloat16 is ~3× faster than float32** — the card has dedicated matrix units (WMMA) for 16-bit
      types. This is why we train in bf16 (Module 08 covers the precision trade-off).
    * **Small matrices are slow per FLOP** — launching a kernel has fixed overhead, so tiny problems
      can't keep the GPU busy. Real models batch work to stay in the big-matrix regime.
    """)
    return


@app.cell
def _(mo):
    size = mo.ui.slider(steps=[256, 512, 1024, 2048, 4096, 8192], value=4096, label="matrix size n",
                        show_value=True)
    size
    return (size,)


@app.cell
def _(jnp, matmul_tflops, mo, size):
    _n = size.value
    _f32 = matmul_tflops(_n, jnp.float32, iters=5)
    _bf16 = matmul_tflops(_n, jnp.bfloat16, iters=5)
    mo.md(f"""
    | n = {_n} | TFLOP/s |
    |---|---|
    | float32 | **{_f32:.1f}** |
    | bfloat16 | **{_bf16:.1f}** |

    For scale: the deepseek-v3-jax smoke test measured about 24 bf16 TFLOP/s on this card at n = 2048.
    """)
    return


@app.cell
def _(jax, jnp, lj, mo):
    _cpu = jax.devices("cpu")[0]
    _n = 1024
    _a = jax.device_put(jnp.ones((_n, _n), jnp.float32), _cpu)
    _t_cpu = lj.timeit(jax.jit(jnp.matmul), _a, _a, iters=3)["median"]
    _g = jnp.ones((_n, _n), jnp.float32)
    _t_gpu = lj.timeit(jax.jit(jnp.matmul), _g, _g, iters=10)["median"]
    mo.md(f"""
    ### CPU vs GPU, same code

    A {_n}×{_n} float32 matmul: **CPU {_t_cpu * 1e3:.2f} ms**, **GPU {_t_gpu * 1e3:.3f} ms**
    (~{_t_cpu / _t_gpu:.0f}× faster). The code is identical; only `jax.device_put` moved the data.
    JAX decides where to run from *where the data lives* — check any array's `.devices()`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Your turn: the arithmetic of GPUs

    Open `modules/00-environment/exercises.py`. Five small functions, all plain Python. They turn the
    measurements above into answers to the question you'll ask before every experiment:
    **will it fit, and how long will it take?**
    """)
    return


@app.cell
def _(ex, lj, mo):
    _flops = lj.attempt(ex.matmul_flops, 4096, 4096, 4096)
    _rate = lj.attempt(ex.tflops, 2 * 4096**3, 0.005)
    mo.vstack([
        mo.md("**`matmul_flops(4096, 4096, 4096)`** — should be about 137 billion:"),
        _flops.view(mo),
        mo.md("**`tflops(2·4096³, 0.005 s)`** — that's ~27.5 TFLOP/s:"),
        _rate.view(mo),
    ])
    return


@app.cell
def _(ex, lj, mo):
    _per = lj.attempt(ex.training_bytes_per_param)
    _max = lj.attempt(ex.max_trainable_params, 16)
    _items = [mo.md("**Bytes per parameter to train with Adam in float32:**"), _per.view(mo),
              mo.md("**Largest model you can train on 16 GB (half reserved for activations):**"), _max.view(mo)]
    if _max.ok:
        _items.append(mo.md(
            f"That's about **{_max.value / 1e6:,.0f}M parameters**. The deepseek-v3-jax model you trained "
            f"has 36.2M — plenty of room. A 7B-parameter model needs ~112 GB to train this way, which is "
            f"why Module 20 uses LoRA: freeze the big model, train a few small matrices."))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(0))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * JAX docs: [Quickstart](https://docs.jax.dev/en/latest/quickstart.html) and
      [GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
    * ROCm docs: [JAX on ROCm](https://rocm.docs.amd.com/) (search "JAX compatibility")
    * Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
      — compute vs memory vs overhead, the mental model behind Section 4.

    **Next:** Module 01 — arrays, shapes and broadcasting, the grammar of everything that follows.
    """)
    return


if __name__ == "__main__":
    app.run()
