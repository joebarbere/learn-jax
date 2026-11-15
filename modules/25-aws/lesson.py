import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="25 · JAX on AWS")


@app.cell
def _():
    import json
    import sys

    import marimo as mo
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    import learnjax as lj

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    sys.path.insert(0, str(mo.notebook_dir()))
    import bench_common
    import bench_single

    ex = lj.load_exercises(mo.notebook_dir())
    RUNS = bench_common.OUT
    return RUNS, bench_common, bench_single, ex, jax, jnp, json, lj, mo, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 25 · JAX on AWS, and the capstone

    **Goal:** run the code you wrote for your RX 7600 XT, unchanged, on other accelerators: NVIDIA GPUs
    through CUDA, four GPUs in one box, two machines over a network, and AWS's Trainium. Measure what
    Module 24's simulated devices couldn't show, and decide with numbers when renting beats your own card.

    **You will:** see how a paid session is planned, run and torn down safely · benchmark your card live with
    the same script the cloud runs · compare tokens/s, utilization and $/million tokens across machines · see
    real collective bandwidth and communication/compute overlap · do the arithmetic that picks an instance
    and a parallelism strategy · estimate a run's time, cost and communication before paying for it ·
    plan your capstone.

    Everything that costs money happens outside this notebook, in `infra/aws/`, and only after you read a
    `terraform plan` and approve it. This lesson reads the results those sessions leave in `runs/25-aws/`
    and works fully without them.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Renting GPUs by the hour has been possible on AWS since its GPU cluster instances in 2010, and large-
    scale training moved from CPU clusters such as DistBelief (2012) to GPU and TPU pods. Hoffmann et al.'s
    Chinchilla (2022) showed how to spend a fixed compute budget well. At the other end, Karpathy's nanochat
    (2025) trains a small chat model end to end for about a hundred dollars of rented time, the blueprint
    for this capstone.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · What the simulation couldn't show

    Module 24 ran data parallel, FSDP and tensor parallel on 8 *simulated* CPU devices. It proved
    correctness, memory per device, and which collectives XLA inserts. The JAX docs' own
    [201 section](https://docs.jax.dev/en/latest/201/index.html) draws the line clearly: collective
    bandwidth, communication/compute overlap, pipeline bubbles, precision effects and multi-host execution
    need real accelerators. Those are what the AWS sessions measure.

    | Session (`-var profile=…`) | Hardware | Teaches | ~cost |
    |---|---|---|---|
    | `gpu1` | 1 × NVIDIA L4 24 GB (g6.xlarge) | same JAX code on CUDA · $/token vs your card | ~$1 |
    | `gpu1-l40s` | 1 × NVIDIA L40S 48 GB (g6e.xlarge) | a bigger GPU | ~$2 |
    | `multihost` | 2 × g6.xlarge | one program across two machines ([`jax.distributed`](https://docs.jax.dev/en/latest/jax.distributed.html#module-jax.distributed)) | ~$2 |
    | `gpu4` | 4 × L4 (g6.12xlarge) | real DP / FSDP / TP, collective bandwidth, overlap | ~$5 |
    | `trn` | 1 × Trainium (trn1.2xlarge, 2 NeuronCores) | JAX on a third accelerator family (beta) | ~$2 |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · How a session runs, safely

    ```
    terraform plan -var profile=gpu1   → you read it: types, count, tags, no NAT, no open ports
    terraform apply                    → the only step that costs money (your decision)
    infra/aws/run_remote.sh            → ships the code via S3, runs bench_*.py over SSM, fetches JSON
    terraform destroy                  → always
    ```

    Four guards protect your wallet, from strongest to weakest:

    1. **Auto-stop.** Each instance runs `shutdown -h +120` at boot, and its shutdown behavior is
       *terminate*. A forgotten session deletes itself after two hours.
    2. **The seatbelt.** A guardrail IAM policy on your account refuses any instance type outside an allowlist,
       any region but us-east-1, NAT gateways and fleets. This stack's plan also checks the allowlist
       before anything is created.
    3. **Quotas.** AWS's own per-family vCPU limits. They're why `trn` waits for an approved increase.
    4. **Budgets.** They email you at thresholds and **stop nothing**.

    Details and per-session costs are in `infra/aws/README.md`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Your card, measured live

    `bench_single.py` is the exact script the cloud instances run. Here its sections run on your RX 7600 XT:
    the bf16 matmul peak, then Module 16's GPT training step (13.9M parameters, bf16 compute with f32
    master weights, AdamW, donated buffers). *MFU* is measured against the matmul peak rather than a
    datasheet. Vendor peaks assume ideal conditions and some quote sparsity, so a measured denominator
    compares devices more fairly. The first run compiles for about a minute; the persistent cache makes
    later runs fast.
    """)
    return


@app.cell
def _(bench_single, mo):
    live_peak = bench_single.peaks()
    live_gpt = bench_single.gpt_train(live_peak["bf16_tflops"], steps=10)
    mo.md(f"""
    | live on this machine | |
    |---|---|
    | bf16 matmul peak | **{live_peak['bf16_tflops']:.1f} TFLOP/s** (f32 {live_peak['f32_tflops']:.1f}) |
    | GPT training | **{live_gpt['tokens_per_s']:,.0f} tokens/s** · {live_gpt['step_ms']:.1f} ms/step |
    | utilization (MFU vs measured peak) | **{live_gpt['mfu_vs_measured_peak']:.0%}** |
    """)
    return live_gpt, live_peak


@app.cell
def _(RUNS, json, live_gpt, live_peak, mo):
    results = {}
    for _p in sorted(RUNS.glob("*.json")) if RUNS.exists() else []:
        try:
            results[_p.stem] = json.loads(_p.read_text())
        except ValueError:
            pass
    _local = results.setdefault("local-rx7600xt", {"single": {}})
    _local.setdefault("single", {})
    _local["single"]["peak"] = live_peak       # this run's live numbers replace the saved ones
    _local["single"]["gpt_train"] = live_gpt
    _cloud = [k for k in results if k != "local-rx7600xt"]
    _msg = (f"Found results for **{', '.join(_cloud)}** in `runs/25-aws/`." if _cloud else
            "No cloud results yet. Only your card is shown below. To add a machine, run a session: "
            "`cd infra/aws && terraform plan -var profile=gpu1 -out=s.tfplan`, read the plan, "
            "`terraform apply s.tfplan`, `./run_remote.sh`, then `terraform destroy -var profile=gpu1` "
            "(about $1 for `gpu1`).")
    mo.callout(mo.md(_msg), kind="success" if _cloud else "info")
    return (results,)


@app.cell
def _(bench_common, mo, results):
    def _f(v, fmt):
        return format(v, fmt) if isinstance(v, (int, float)) else "—"

    rows = []
    for _name, _r in results.items():
        if not _r.get("single"):        # e.g. a multi-host-only result: shown in Section 7 instead
            continue
        _s = _r["single"]
        _price = bench_common.PRICES.get(_name)
        _gpt, _dec = _s.get("gpt_train") or {}, _s.get("decode") or {}
        _tps = _gpt.get("tokens_per_s")
        rows.append({
            "machine": _name,
            "device": (_s.get("device") or {}).get("device_kind", "this card" if _name.startswith("local") else "?"),
            "bf16 TFLOP/s": (_s.get("peak") or {}).get("bf16_tflops"),
            "GPT tok/s": _tps, "MFU": _gpt.get("mfu_vs_measured_peak"),
            "decode tok/s": _dec.get("tokens_per_s"), "$/h": _price,
            "$/M train tokens": _price / (_tps * 3600) * 1e6 if (_price and _tps) else None,
        })
    _lines = ["| machine | device | bf16 TFLOP/s | GPT tok/s | MFU | decode tok/s | $/hour | $ per M train tokens |",
              "|---|---|---|---|---|---|---|---|"]
    for _row in rows:
        _lines.append(f"| {_row['machine']} | {_row['device']} | {_f(_row['bf16 TFLOP/s'], '.1f')} | "
                      f"{_f(_row['GPT tok/s'], ',.0f')} | {_f(_row['MFU'], '.0%')} | {_f(_row['decode tok/s'], ',.0f')} | "
                      f"{_f(_row['$/h'], '.3f')} | {_f(_row['$/M train tokens'], '.4f')} |")
    mo.md("## 4 · The same code, every machine\n\n" + "\n".join(_lines) + """

    *$/hour for your card is electricity only (~165 W at $0.15/kWh), not the price of the card. Cloud prices
    are on-demand us-east-1; spot is usually much cheaper.*""")
    return (rows,)


@app.cell
def _(plt, rows):
    _have = [r for r in rows if r["GPT tok/s"]]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 1.2 + 0.5 * len(_have)))
    _axes[0].barh([r["machine"] for r in _have], [r["GPT tok/s"] for r in _have], color="C0")
    _axes[0].set_title("GPT training throughput (tokens/s)")
    _axes[1].barh([r["machine"] for r in _have], [(r["MFU"] or 0) * 100 for r in _have], color="C1")
    _axes[1].set_title("utilization vs each device's measured peak (%)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Your turn: is renting worth it?

    Open `modules/25-aws/exercises.py`. The first six functions are the arithmetic you do *before*
    `terraform apply`: cost per million tokens, how many bytes an all-reduce sends, bandwidth, which
    parallelism strategy a model needs, what spot saves, and what a session costs.
    """)
    return


@app.cell
def _(ex, lj, live_gpt, mo):
    _local = lj.attempt(ex.dollars_per_million_tokens, live_gpt["tokens_per_s"], 0.025)
    _l4 = lj.attempt(ex.dollars_per_million_tokens, live_gpt["tokens_per_s"], 0.805)
    _cost = lj.attempt(ex.session_cost, "gpu4", 60)
    _items = [mo.md("**Your card's $/M training tokens (electricity only):**"), _local.view(mo),
              mo.md("**…if an L4 matched your card's speed, at $0.805/h:**"), _l4.view(mo),
              mo.md("**An hour on the 4-GPU box (`session_cost('gpu4', 60)`):**"), _cost.view(mo)]
    if _local.ok and _l4.ok:
        _items.append(mo.md(
            f"Electricity makes your card about **{_l4.value / _local.value:.0f}× cheaper per token** than renting a "
            f"GPU of the same speed. Renting wins when you need what you don't have: more memory, more "
            f"devices, or an answer today."))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mem_gb = mo.ui.slider(1, 200, value=40, label="training state (params + grads + Adam), GB", show_value=True)
    n_dev = mo.ui.dropdown(options={"1 (your card)": 1, "4 (g6.12xlarge)": 4, "8": 8}, value="4 (g6.12xlarge)",
                           label="devices")
    mo.hstack([mem_gb, n_dev])
    return mem_gb, n_dev


@app.cell
def _(ex, lj, mem_gb, mo, n_dev):
    _dev_mem = 16e9 if n_dev.value == 1 else 24e9
    _choice = lj.attempt(ex.choose_strategy, mem_gb.value * 1e9, _dev_mem, n_dev.value)
    _ar = lj.attempt(ex.allreduce_bytes, int(mem_gb.value * 1e9 / 16), n_dev.value)
    _items = [mo.md(f"**`choose_strategy`** ({_dev_mem / 1e9:.0f} GB devices, half kept for activations):"),
              _choice.view(mo)]
    if _ar.ok and n_dev.value > 1:
        _items.append(mo.md(
            f"Each data-parallel step would all-reduce **{_ar.value / 1e9:.2f} GB per device** of f32 gradients "
            f"(ring algorithm). At the ~10 GB/s a PCIe GPU box sustains, that's ~{_ar.value / 10e9 * 1e3:.0f} ms "
            f"per step spent communicating, unless it overlaps with compute. That's what Section 6 measures."))
    mo.vstack(_items)
    return


@app.cell
def _(mo, plt, results):
    _multi = [(k, v["multi"]) for k, v in results.items() if isinstance(v.get("multi"), dict)
              and "strategies" in v["multi"] and "error" not in v["multi"]["strategies"]]
    if not _multi:
        _out = mo.vstack([mo.md("## 6 · Four GPUs in one box"), mo.callout(mo.md(
            "No `gpu4` results yet. That session (about $5) runs `bench_multi.py`: Module 24's DP / FSDP / TP / "
            "FSDP+TP on 4 × L4 with a 170M-parameter model, a psum/all_gather bandwidth sweep, and a profiler "
            "trace measuring how much collective time hides behind compute. You can check the script on "
            "simulated devices for free: `containers/run.sh python modules/25-aws/bench_multi.py --cpu-sim 8 --small`."),
            kind="info")])
    else:
        _name, _m = _multi[0]
        _rows = _m["strategies"]["rows"]
        _t = ["| strategy | mesh | ms/step | tokens/s | MiB/device | all-reduce | all-gather | reduce-scatter |",
              "|---|---|---|---|---|---|---|---|"]
        _t += [f"| {r['strategy']} | {r['mesh']} | {r['step_ms']:.1f} | {r['tokens_per_s']:,.0f} | "
               f"{r['per_device_bytes'] / 2**20:,.0f} | {r['all-reduce']} | {r['all-gather']} | {r['reduce-scatter']} |"
               for r in _rows]
        _bw = (_m.get("bandwidth") or {}).get("rows") or []
        _ov = _m.get("overlap") or {}
        _items = [mo.md(f"## 6 · Four GPUs in one box ({_name})\n\n" + "\n".join(_t)),
                  mo.md(f"Collective time overlapped with compute (FSDP, profiler): "
                        f"**{(_ov.get('collective_overlapped_fraction') or 0):.0%}** "
                        f"({_ov.get('collective_ms', 0):.1f} ms of collectives over {_ov.get('steps', '?')} steps).")]
        if _bw:
            _fig, _ax = plt.subplots(figsize=(6, 3))
            _x = [r["bytes_per_device"] / 2**20 for r in _bw]
            _ax.plot(_x, [r["psum_busbw_gbs"] for r in _bw], "o-", label="psum (all-reduce)")
            _ax.plot(_x, [r["all_gather_busbw_gbs"] for r in _bw], "s-", label="all_gather")
            _ax.set(xscale="log", xlabel="MiB per device", ylabel="bus bandwidth (GB/s)", title="collectives over PCIe")
            _ax.legend()
            _items.append(_fig)
        _out = mo.vstack(_items)
    _out
    return


@app.cell
def _(mo, results):
    _mh = [(k, v["multihost"]) for k, v in results.items() if isinstance(v.get("multihost"), dict)]
    if not _mh:
        _out = mo.vstack([mo.md("## 7 · Two machines, one program"), mo.callout(mo.md(
            "No `multihost` results yet (about $2). That session starts `bench_multihost.py` on two g6.xlarge "
            "nodes: `jax.distributed.initialize` joins them, `jax.devices()` then lists *both* GPUs while "
            "`jax.local_devices()` lists one, each node loads only its half of the batch "
            "(`make_array_from_process_local_data`), and the gradient all-reduce crosses the network. Try the "
            "mechanics for free with two CPU processes: "
            "`containers/run.sh python modules/25-aws/bench_multihost.py --local-test`."), kind="info")])
    else:
        _name, _m = _mh[0]
        _dp, _top = _m.get("dp_step") or {}, _m.get("topology") or {}
        _bw = ((_m.get("bandwidth") or {}).get("rows") or [{}])[-1]
        _out = mo.md(f"""
    ## 7 · Two machines, one program ({_name})

    | | |
    |---|---|
    | processes | {_top.get('process_count')} · global devices {_top.get('global_devices')} |
    | data-parallel step | {_dp.get('step_ms', float('nan')):.1f} ms · {_dp.get('tokens_per_s', 0):,.0f} tokens/s · {_dp.get('all_reduces_in_hlo')} all-reduce(s) |
    | psum bus bandwidth between hosts | {_bw.get('psum_busbw_gbs', float('nan')):.2f} GB/s at {_bw.get('bytes_per_device', 0) / 2**20:.0f} MiB |

    Compare with Section 6's in-box numbers: the network between two `xlarge` instances is much slower than
    PCIe inside one machine. That's why big clusters use special interconnects (EFA, NVLink, TPU ICI),
    and why the mesh's slow axis should carry the least traffic (the cookbook's advice: put hosts on the
    outer, data-parallel axis).
    """)
    _out
    return


@app.cell
def _(mo, results):
    _trn = {k: v for k, v in results.items() if k.startswith("trn")}
    if not _trn:
        _out = mo.vstack([mo.md("## 8 · Trainium"), mo.callout(mo.md(
            "No `trn` results yet (about $2, and it needs the Trainium quota). The same `bench_single.py` runs on "
            "a trn1.2xlarge in AWS's Neuron JAX environment (`jax-neuronx`, **beta**: AWS's docs say some JAX "
            "functionality may not be supported). Each section records success or the exact error, so the "
            "result says which of your course's programs run on Trainium today."), kind="info")])
    else:
        _name, _r = next(iter(_trn.items()))
        _s = _r.get("single") or {}
        _lines = [f"| {k} | {'❌ ' + v['error'][:120] if isinstance(v, dict) and 'error' in v else '✅'} |"
                  for k, v in _s.items() if k != "cost"]
        _out = mo.md(f"## 8 · Trainium ({_name})\n\n| section | result |\n|---|---|\n" + "\n".join(_lines))
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Estimate before you `terraform apply`

    Before anyone spends money on a training run, they predict it on the back of an envelope. The recipe
    takes five lines, and every input is a number you've already measured:

    | quantity | estimate | from |
    |---|---|---|
    | training state | 16 bytes per parameter (f32 weights, gradients, two Adam moments) | Modules 00, 20 |
    | compute | $6 \cdot N \cdot D$ FLOPs for $N$ parameters and $D$ tokens | Module 16 |
    | time | FLOPs ÷ (devices × peak × MFU) | Section 3 |
    | communication | ring all-reduce bytes ÷ interconnect bandwidth, every step | Sections 5–6 |
    | cost | hours × \$/hour | Section 5 |

    The communication line deserves a closer look. Per step, compute grows with $N$ times the tokens on each
    device, and the all-reduce grows with $N$. **The model size cancels**, so whether data parallelism is
    limited by communication depends only on how many tokens each device processes per step, compared
    with the hardware's ratio of FLOP/s to bytes/s. The same idea runs all through Google DeepMind's
    [How To Scale Your Model](https://jax-ml.github.io/scaling-book/). Its chapters on
    [Transformer math](https://jax-ml.github.io/scaling-book/transformers) and
    [parallelizing training](https://jax-ml.github.io/scaling-book/training) work it out in full.

    **Your turn:** `estimate_run` and `dp_comm_ratio` in `exercises.py`. The problems below use them. Try
    each one on paper first, then open the answer.
    """)
    return


@app.cell
def _(live_gpt, live_peak, mo, results):
    # Inputs for the estimates: measured where a session has measured them, stated assumptions otherwise.
    _l4 = next((r["single"] for k, r in results.items() if k.startswith("g6.xlarge") and (r.get("single") or {}).get("peak")), None)
    _bw_rows = [row for r in results.values() if isinstance(r.get("multi"), dict)
                for row in ((r["multi"].get("bandwidth") or {}).get("rows") or [])]
    env = {
        "card_peak": (live_peak["bf16_tflops"], "measured live, Section 3"),
        "card_mfu": (live_gpt["mfu_vs_measured_peak"], "measured live, Section 3 (13.9M GPT)"),
        "l4_peak": ((_l4["peak"]["bf16_tflops"], "measured on gpu1") if _l4 else
                    (121.0, "**assumed**: NVIDIA's datasheet, dense bf16; `gpu1` replaces it")),
        "l4_mfu": ((_l4["gpt_train"]["mfu_vs_measured_peak"], "measured on gpu1") if _l4 and _l4.get("gpt_train") else
                   (live_gpt["mfu_vs_measured_peak"], "**assumed**: same as your card; `gpu1` replaces it")),
        "pcie_gbs": ((max(r["psum_busbw_gbs"] for r in _bw_rows), "measured on gpu4 (best psum)") if _bw_rows else
                     (10.0, "**assumed**: what a PCIe GPU box sustains; `gpu4` replaces it")),
    }
    _names = {"card_peak": "RX 7600 XT bf16 peak (TFLOP/s)", "card_mfu": "RX 7600 XT MFU",
              "l4_peak": "L4 bf16 peak (TFLOP/s)", "l4_mfu": "L4 MFU", "pcie_gbs": "all-reduce bandwidth in g6.12xlarge (GB/s)"}
    _rows = "\n".join(f"| {_names[k]} | {v:.2f} | {src} |" if v < 1 else f"| {_names[k]} | {v:.1f} | {src} |"
                      for k, (v, src) in env.items())
    mo.md(f"""
| input | value | source |
|---|---|---|
{_rows}

Each paid session you run replaces an assumption with a measurement, and every answer below updates.
That's the habit worth keeping: predict first, measure, then find out which assumption was wrong.
""")
    return (env,)


@app.cell
def _(env, ex, lj, mo):
    _cp, _cm = env["card_peak"][0], env["card_mfu"][0]
    _lp, _lm = env["l4_peak"][0], env["l4_mfu"][0]
    _bw = env["pcie_gbs"][0]
    _N, _D = 100e6, 2e9                          # a 100M-parameter capstone, 20 tokens per parameter
    _card = lj.attempt(ex.estimate_run, _N, _D, _cp, _cm, 1, 0.025)
    _gpu4 = lj.attempt(ex.estimate_run, _N, _D, _lp, _lm, 4, ex.PRICES["g6.12xlarge"])
    _ratio = lj.attempt(ex.dp_comm_ratio, _N, 16 * 256, 4, _lp, _lm, _bw)
    _items = [mo.md("### Your turn: `estimate_run` and `dp_comm_ratio`"),
              mo.md("`estimate_run` (the capstone on your card):"), _card.view(mo),
              mo.md("`dp_comm_ratio` (gpu4, 16 × 256 tokens per GPU):"), _ratio.view(mo)]
    if _card.ok and _gpu4.ok and _ratio.ok:
        _c, _g, _r = _card.value, _gpu4.value, _ratio.value
        _big = ex.dp_comm_ratio(7e9, 16 * 256, 4, _lp, _lm, _bw)
        _crit = 16 * 256 * _r                     # tokens per GPU where comm = compute
        _N17 = 1.71e9
        _st17 = 16 * _N17
        _card17 = ex.choose_strategy(_st17, 16e9, 1)
        _gpu417 = ex.choose_strategy(_st17, 24e9, 4)
        _lora17 = 2 * _N17 * 1.005               # frozen bf16 weights + ~0.5% adapters (adapter state is tiny)
        _problems = {
            "Problem 1 · Your capstone on your card": f"""
**Question.** A 100M-parameter model, trained on 2B tokens (Chinchilla's ~20 tokens per parameter), on your
RX 7600 XT. How long does it take, does the state fit, and what does it cost?

**Answer.** {_c['flops']:.1e} FLOPs at {_cp:.1f} TFLOP/s × {_cm:.0%} MFU is **{_c['hours']:.0f} hours**
(about {_c['hours'] / 24:.1f} days). The state is {_c['state_gb']:.1f} GB, which fits easily. Electricity
costs about **\\${_c['usd']:.2f}**. That's slow but cheap. The MFU you measured comes from a 13.9M model,
and bigger matmuls usually use the card better, so the real run would likely be faster.
""",
            "Problem 2 · The same run on gpu4": f"""
**Question.** The same 100M model and 2B tokens on `gpu4` (4 × L4, \\${ex.PRICES['g6.12xlarge']:.3f}/h).

**Answer.** {_g['hours']:.1f} hours and **\\${_g['usd']:.0f}**, about {_c['hours'] / _g['hours']:.0f}× faster
than your card for about {_g['usd'] / max(_c['usd'], 1e-9):.0f}× the price. That assumes perfect scaling;
Problem 3 checks that assumption. The 2-hour auto-stop from Section 2 would also kill a run this long, so
plan checkpoints and resumes, or raise the limit deliberately.
""",
            "Problem 3 · Will data parallelism stall on communication?": f"""
**Question.** On `gpu4`, each GPU processes 16 sequences of 256 tokens per step. What's the
communication-to-compute ratio? What if the model were 7B parameters instead of 100M?

**Answer.** **{_r:.2f}** at 100M and **{_big:.2f}** at 7B: the model size cancels, as promised. At
{_bw:.0f} GB/s, about {_r:.0%} as much time goes to the all-reduce as to compute. Unless the two overlap
(Section 6 measures how much does), that time adds directly to every step. Communication equals compute
at about **{_crit:,.0f} tokens per GPU per step**. Larger per-GPU batches (or gradient accumulation) push
the ratio down. A faster interconnect does the same, which is why training clusters pay for NVLink and
InfiniBand.
""",
            "Problem 4 · Fine-tune SmolLM2-1.7B": f"""
**Question.** Full fine-tuning of SmolLM2-1.7B (~1.71B parameters). Where does it fit, and what does LoRA
change?

**Answer.** The full training state is {_st17 / 1e9:.1f} GB. `choose_strategy` gives **{_card17}** on
your 16 GB card and **{_gpu417}** on `gpu4`'s 4 × 24 GB (about {_st17 / 4e9:.1f} GB per GPU). With LoRA
(Module 20), the frozen base is about {_lora17 / 1e9:.1f} GB in bf16, and the adapters plus their Adam
state are a few MB, so it fits on your card. LoRA turns a rental into a local job.
""",
        }
        _items.append(mo.accordion({k: mo.md(v) for k, v in _problems.items()}))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Data parallel: replicate everything, all-reduce the gradients": "dp",
            "FSDP: shard parameters and optimizer state across the 4 GPUs": "fsdp",
            "Rent a bigger single GPU so no strategy is needed": "big",
        },
        label="**Check yourself:** a model's training state is 30 GB; you have 4 × 24 GB GPUs. Which first?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "dp": mo.callout(mo.md("DP replicates the full 30 GB on every GPU, and 30 GB doesn't fit in 24 GB "
                               "(let alone with room for activations)."), kind="warn"),
        "fsdp": mo.callout(mo.md("Right: sharded four ways it's 7.5 GB per GPU, leaving room for activations. "
                                 "FSDP gathers each layer's weights just in time and reduce-scatters gradients."),
                           kind="success"),
        "big": mo.callout(mo.md("Sometimes the simplest answer, but a 48 GB L40S still can't hold 30 GB *plus* "
                                "activations comfortably, and you'd pay more for one device than you'd save."),
                          kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 · The capstone: your own small LLM, end to end

    You now have every piece. The capstone is to assemble them, on your RX 7600 XT, into a model that's
    yours. Think of it as Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) (MIT) in JAX,
    scaled down to one 16 GB card. nanochat runs the whole ChatGPT-style pipeline (tokenizer, pretraining,
    SFT, RL, evaluation, a KV-cache chat engine, tool use) for about \$100 on 8 × H100; every one of its
    stages is something you already built. Each step points at the module that taught it:

    1. **Data.** Pick a corpus you care about (public domain or your own writing). Clean it and split by
       document (Module 13's `prepare` pattern, deepseek-v3-jax's `prepare_data.py`).
    2. **Tokenizer.** Train byte-level BPE on it (Module 13). Choose the vocabulary size with the
       embedding-share table in mind.
    3. **Model.** Build a DeepSeek-style MoE with MLA from your Module 17 parts, or the Llama-style one.
       Size it with Module 00's memory arithmetic and Module 22's MFU, then estimate its time and cost with
       Section 9 before you start.
    4. **Train.** Adapt Module 16's `train.py`: bf16 compute, AdamW with warmup-cosine, deferred metrics,
       donation, a persistent compile cache, `.npz` checkpoints. Watch expert load (Module 17).
    5. **Evaluate.** Bits per byte on held-out text (Module 21). Compare against SmolLM2 and your Module 16
       GPT on the *same* text.
    6. **Serve.** Decode on device with a KV cache (Module 18). Export it, convert to GGUF, and run it next to
       your coding model (Modules 19 and 27).
    7. **Specialize.** LoRA-tune it for a task with a measurable metric (Module 20). This is nanochat's SFT
       stage; for a chat model, tune on instruction/response pairs with a chat template.
    8. **Reinforce (optional).** Give it a verifiable task and run GRPO (Module 26) — nanochat's `chat_rl`.
    9. **Scale (optional).** Rerun step 4 on `gpu4` with FSDP (Section 6), and compare $/token with your card.

    | nanochat stage | Where you built it |
    |---|---|
    | tokenizer (`tok_train`, `tok_eval`) | Module 13 |
    | pretraining (`base_train`) | Modules 16, 17, 22 |
    | evaluation (`base_eval`: CORE, bits per byte; `chat_eval`) | Module 21 |
    | SFT (`chat_sft`) | Module 20 |
    | RL (`chat_rl`) | Module 26 |
    | inference engine with KV cache, chat CLI | Modules 18, 19, 27 |
    | tool use (`execution.py`) | Module 27 (tool calling) |
    | AdamW + Muon optimizer | Module 05 (Muon sidebar) |

    Write down your numbers at each step. That table is your capstone report. For how far the same pipeline
    goes with more compute, read Karpathy's
    ["Beating GPT-2 for &lt;&lt;\$100: the nanochat journey"](https://github.com/karpathy/nanochat/discussions/481).
    """)
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(25))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    * JAX docs: [201: scaling](https://docs.jax.dev/en/latest/201/index.html), the source for what needs
      real hardware, and [The Training Cookbook](https://docs.jax.dev/en/latest/the-training-cookbook.html)
      on meshes, FSDP and multi-host data loading.
    * AWS Neuron docs: [JAX on Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/jax/index.html)
      (beta) for Trainium.
    * NVIDIA: [nccl-tests](https://github.com/NVIDIA/nccl-tests), whose bus-bandwidth convention the
      collective benchmark follows.
    * Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
      on overhead, bandwidth and compute, now across machines.

    **Next:** Module 26, reasoning models. Make a small model think longer, and train it with
    verifiable rewards.
    """)
    return


if __name__ == "__main__":
    app.run()
