"""Train Module 41's sequence mixers on the detective corpus, and run the associative-recall sweep.

    containers/run.sh python modules/41-linear-attention-hybrids/train.py --variant attention
    containers/run.sh python modules/41-linear-attention-hybrids/train.py --variant gdn
    containers/run.sh python modules/41-linear-attention-hybrids/train.py --mqar          # the recall sweep
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/41-linear-attention-hybrids/train.py ...   # reference

Uses *your* exercises.py (finish `make test N=41` first) or solutions.py with LEARNJAX_SOLUTIONS=1.

Variants (8 layers of width 384, 6 heads of 64, SwiGLU FFNs, tied 8K embeddings: 18.5M parameters each, ±0.5%):
    attention   AAAAAAAA   gated softmax attention with RoPE
    linear      LLLLLLLL   linear attention (add, never forget)
    mamba2      MMMMMMMM   + a per-step decay (Mamba-2's SSD in attention form)
    gdn         GGGGGGGG   + the delta rule (Gated DeltaNet)
    hybrid      GGGAGGGA   3 : 1, like Qwen3-Next, Qwen3.5 and Kimi Linear

Defaults: 12,000 steps of 16 × 512 tokens (98M tokens, the same as Module 40), AdamW 1e-3 warmup-cosine.

Writes to /course/runs/41-linear-attention-hybrids/<variant>/ (metrics.jsonl, params.npz, config.json),
and mqar.json for --mqar.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # the course root, for `learnjax`

from learnjax.course import load_exercises  # after sys.path setup

RUNS = HERE.parent.parent / "runs" / "41-linear-attention-hybrids"
MQAR_LAYOUTS = {"attention": ("AA", {}), "attention+conv": ("AA", {"attn_conv": True}), "linear": ("LL", {}),
                "mamba2": ("MM", {}), "deltanet": ("DD", {}), "gdn": ("GG", {}), "hybrid": ("GA", {}),
                # the same layers with a 10× slower starting decay (Mamba-2's default forgets within ~10–100 steps)
                "mamba2-slow": ("MM", {"dt_range": (1e-4, 1e-2)}), "gdn-slow": ("GG", {"dt_range": (1e-4, 1e-2)})}
SKIP = {"hybrid/256"}   # hung the RX 7600 XT (ROCM_ERROR_LAUNCH_OUT_OF_RESOURCES; docs/ROCM_GOTCHAS.md #20d)
MQAR_PAIRS = (32, 64, 128, 256)


def _models():
    spec = importlib.util.spec_from_file_location("m41_models", HERE / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_mqar(M, ex, steps: int) -> int:
    out = RUNS / "mqar.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for n in MQAR_PAIRS:
        for name, (layout, over) in MQAR_LAYOUTS.items():
            if f"{name}/{n}" in res or f"{name}/{n}" in SKIP:
                continue
            r = M.train_mqar(ex, layout, n, steps=steps, **over)
            res[f"{name}/{n}"] = {k: r[k] for k in ("test_acc", "seconds", "n_params")}
            print(f"mqar {name:9s} {n:4d} pairs: {r['test_acc']:.1%} ({r['seconds']:.0f} s)", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", default="attention", help="one of " + ", ".join(_models().LAYOUTS))
    ap.add_argument("--mqar", action="store_true", help="run the associative-recall sweep instead")
    ap.add_argument("--mqar-steps", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from learnjax.data import detective_corpus

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    M, ex = _models(), load_exercises(HERE)
    try:
        jax.eval_shape(lambda: ex.chunked_linear_attention(*(jnp.zeros((1, 64, 4)),) * 3, jnp.zeros((1, 64)), 16))
        jax.eval_shape(lambda: ex.gated_delta_scan(*(jnp.zeros((1, 8, 4)),) * 3, jnp.zeros((1, 8))))
        jax.eval_shape(lambda: ex.segsum(jnp.zeros((4,))))
    except NotImplementedError:
        print("Finish modules/41-linear-attention-hybrids/exercises.py first (`make test N=41`), or run with "
              "LEARNJAX_SOLUTIONS=1.")
        return 1
    if args.mqar:
        return run_mqar(M, ex, args.mqar_steps)

    cfg = M.variant(args.variant)
    out = RUNS / (args.variant if args.steps == 12000 else f"{args.variant}-{args.steps}")
    corpus = detective_corpus()
    params = M.init_model(jax.random.key(args.seed), 8192, cfg)
    optimizer = M.make_optimizer(args.lr, args.steps)
    opt_state = optimizer.init(params)
    train = jnp.asarray(corpus["train"].astype(np.int32))
    vals = M.val_batches(corpus["val"], args.batch, args.seq_len)
    chunk, evaluate = M.make_trainer(ex, cfg, optimizer, args.batch, args.seq_len)
    n_params = M.count_params(params)
    print(f"{out.name} ({cfg['layout']}): {n_params / 1e6:.2f}M params | {args.steps} steps × "
          f"{args.batch}×{args.seq_len} tokens", flush=True)
    extra = {"variant": args.variant, "steps": args.steps, "batch": args.batch, "seq_len": args.seq_len,
             "lr": args.lr, "params": n_params, "data": "detective_corpus v1.0"}
    key = jax.random.key(args.seed + 1)
    t0 = window = time.perf_counter()
    window_steps, seen = 0, []
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.jsonl", "w") as log:
        for step in range(args.chunk, args.steps + 1, args.chunk):
            key, sub = jax.random.split(key)
            params, opt_state, losses = chunk(params, opt_state, train, sub, args.chunk)
            window_steps += args.chunk
            seen.append(losses)
            if step % args.eval_every == 0 or step == args.steps:
                ls = np.concatenate([np.asarray(x) for x in seen])
                if not np.isfinite(ls).all():
                    raise FloatingPointError(f"non-finite loss before step {step}")
                tok_s = window_steps * args.batch * args.seq_len / (time.perf_counter() - window)
                val = float(np.mean([evaluate(params, x, y) for x, y in vals]))
                m = {"step": step, "loss": float(ls[-args.chunk:].mean()), "val": val, "tok_per_s": tok_s,
                     "elapsed_s": time.perf_counter() - t0}
                log.write(json.dumps(m) + "\n")
                log.flush()
                M.save(params, cfg, out, {**extra, "step": step})
                print(f"step {step:6d} | loss {m['loss']:.3f} | val {val:.3f} | {tok_s:,.0f} tok/s | "
                      f"{m['elapsed_s'] / 60:.1f} min", flush=True)
                window, window_steps, seen = time.perf_counter(), 0, []
    # validation loss at each position in the window: how much does each model gain from longer context?
    pos_fn = jax.jit(lambda p, x, y: -jnp.take_along_axis(
        jax.nn.log_softmax(M.forward(ex, M._bf16(p), x, cfg).astype(jnp.float32)), y[..., None], -1)[..., 0].mean(0))
    pos = np.mean([np.asarray(pos_fn(params, x, y)) for x, y in vals], 0)
    (out / "pos_loss.json").write_text(json.dumps([round(float(v), 4) for v in pos]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
