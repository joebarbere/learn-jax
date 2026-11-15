"""Portability benchmark: the same JAX code on whatever single accelerator this machine has.

    containers/run.sh python modules/25-aws/bench_single.py            # RX 7600 XT, in the course container
    python modules/25-aws/bench_single.py                              # on AWS, via infra/aws/run_remote.sh

Sections (each recorded independently, so a failure — e.g. an op Neuron doesn't support — doesn't lose the
others): device info · bf16/f32 matmul peak · Module 16's GPT training step (bf16 compute, f32 master
weights, AdamW, donated buffers) → tokens/s and MFU · Module 18's on-device KV-cache decode of
SmolLM2-135M → tokens/s · cost per million tokens at this machine's hourly price.
"""
from __future__ import annotations

import argparse
import importlib.util
import time

from bench_common import PRICES, device_info, machine_label, matmul_peak, save, section, timed

# A GPT big enough to keep a GPU busy but small enough for every target (L4 24 GB, Trainium 32 GB/chip).
GPT = {"vocab": 8192, "seq_len": 256, "d_model": 384, "n_layers": 6, "n_heads": 6}
BATCH = 16


def peaks():
    return {"bf16_tflops": matmul_peak(4096, "bfloat16"), "f32_tflops": matmul_peak(2048, "float32")}


def gpt_train(peak_tflops: float, steps: int = 20):
    import jax
    import jax.numpy as jnp
    import optax
    from bench_common import ROOT

    from learnjax.course import load_exercises
    m16 = load_exercises(ROOT / "modules" / "16-gpt", solutions=True)
    cfg = m16.GPTConfig(**GPT)
    params = m16.init_gpt(jax.random.key(0), cfg)
    n_params = m16.count_params(cfg)
    opt = optax.adamw(3e-4, weight_decay=0.1)
    opt_state = opt.init(params)
    tokens = jax.random.randint(jax.random.key(1), (BATCH, cfg.seq_len + 1), 0, cfg.vocab)

    def loss_bf16(p, t):  # bf16 compute, f32 master weights (Module 08)
        return m16.loss_fn(jax.tree.map(lambda a: a.astype(jnp.bfloat16), p), t, cfg)

    def train_step(p, s, t):
        loss, g = jax.value_and_grad(loss_bf16)(p, t)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, loss

    step = jax.jit(train_step, donate_argnums=(0, 1))  # params + optimizer state updated in place
    t0 = time.perf_counter()
    params, opt_state, loss = step(params, opt_state, tokens)
    jax.block_until_ready(loss)
    compile_s = time.perf_counter() - t0
    for _ in range(3):
        params, opt_state, loss = step(params, opt_state, tokens)
    jax.block_until_ready(loss)
    t0 = time.perf_counter()
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state, tokens)
    jax.block_until_ready(loss)
    dt = (time.perf_counter() - t0) / steps
    tok_s = BATCH * cfg.seq_len / dt
    return {"config": GPT, "batch": BATCH, "n_params": n_params, "step_ms": dt * 1e3, "tokens_per_s": tok_s,
            "mfu_vs_measured_peak": 6 * n_params * tok_s / (peak_tflops * 1e12), "compile_s": compile_s,
            "final_loss": float(loss)}


def decode(n_new: int = 128):
    import jax
    import jax.numpy as jnp
    from bench_common import ROOT

    from learnjax import llama
    spec = importlib.util.spec_from_file_location("infer18", ROOT / "modules" / "18-inference" / "infer.py")
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)
    params, cfg, tok = llama.load(llama.SMOLLM_135M)
    prompt = jnp.asarray([tok.encode("The detective looked at the muddy footprints and said").ids], jnp.int32)
    run = lambda: infer.decode_on_device(params, cfg, prompt, n_new, jax.random.key(0))
    jax.block_until_ready(run())                      # compile
    dt = timed(lambda: run(), warmup=1, iters=5)
    return {"model": llama.SMOLLM_135M, "n_new": n_new, "batch": 1, "tokens_per_s": n_new / dt,
            "ms_per_token": dt / n_new * 1e3}


def costs(label: str, results: dict, usd_per_hour: float | None):
    price = usd_per_hour if usd_per_hour is not None else PRICES.get(label)
    if price is None:
        return {"usd_per_hour": None}
    out = {"usd_per_hour": price}
    for k in ("gpt_train", "decode"):
        tps = (results.get(k) or {}).get("tokens_per_s")
        if tps:
            out[f"{k}_usd_per_million_tokens"] = price / (tps * 3600) * 1e6
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None, help="result file name (default: EC2 instance type or local)")
    ap.add_argument("--no-decode", action="store_true", help="skip the SmolLM2 decode section (no download)")
    ap.add_argument("--usd-per-hour", type=float, default=None, help="override the price (e.g. spot)")
    args = ap.parse_args()
    label = args.label or machine_label()
    results = {}
    section(results, "device", device_info)
    pk = section(results, "peak", peaks)
    peak = pk.get("bf16_tflops") or 1.0
    section(results, "gpt_train", gpt_train, peak)
    if not args.no_decode:
        section(results, "decode", decode)
    results["cost"] = costs(label, results, args.usd_per_hour)
    save(label, {"single": results})


if __name__ == "__main__":
    main()
