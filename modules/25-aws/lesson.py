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
    and a parallelism strategy · plan your capstone.

    Everything that costs money happens outside this notebook, in `infra/aws/`, and only after you read a
    `terraform plan` and approve it. This lesson reads the results those sessions leave in `runs/25-aws/`
    and works fully without them.
    """)
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
    2. **The seatbelt.** Your `aws-cloud` repo's IAM policy refuses any instance type outside an allowlist,
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

    Open `modules/25-aws/exercises.py`. The six functions are the arithmetic you do *before*
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
    ## 9 · The capstone: your own small LLM, end to end

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
       Size it with Module 00's memory arithmetic and Module 22's MFU.
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
