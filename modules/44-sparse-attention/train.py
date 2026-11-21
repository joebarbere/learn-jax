"""Train Module 44's race (one model, six attention patterns) on the detective corpus, and run the
attention-sink and timing experiments.

    containers/run.sh python modules/44-sparse-attention/train.py --variant full
    containers/run.sh python modules/44-sparse-attention/train.py --variant dsa
    containers/run.sh python modules/44-sparse-attention/train.py --sinks     # SmolLM2-135M, no training
    containers/run.sh python modules/44-sparse-attention/train.py --bench     # prefill timing
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/44-sparse-attention/train.py ...   # reference

Uses *your* exercises.py (finish `make test N=44` first) or solutions.py with LEARNJAX_SOLUTIONS=1.

Variants (Module 41's attention model, 18.5M parameters; dsa adds a 0.4M-parameter indexer):
    full          every past token
    swa           the last 128 tokens in every layer
    swa-sink      the last 124 + the first 4 tokens
    local-global  128-token windows in 6 layers, full attention in 2 (LLLGLLLG)
    moba          own 64-token block + the 2 best earlier blocks (≤ 192 keys)
    dsa           the 128 tokens the lightning indexer scores highest (dense for the first 500 steps)

Defaults: 6,000 steps of 4 × 2,048 tokens (49M tokens), AdamW 1e-3 warmup-cosine, about 35-50 min each.
Writes /course/runs/44-sparse-attention/<variant>/ (metrics.jsonl, params.npz, config.json, pos_loss.json),
sinks.json and bench.json.
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


def _models():
    spec = importlib.util.spec_from_file_location("m44_models", HERE / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", default="full")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--sinks", action="store_true", help="attention-sink experiment on SmolLM2-135M")
    ap.add_argument("--bench", action="store_true", help="prefill timing: dense vs banded")
    args = ap.parse_args(argv)

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    M, ex = _models(), load_exercises(HERE)
    try:
        ex.window_mask(4, 2)
        ex.topk_causal_mask(jnp.zeros((4, 4)), 2)
    except NotImplementedError:
        print("Finish modules/44-sparse-attention/exercises.py first (`make test N=44`), or run with "
              "LEARNJAX_SOLUTIONS=1.")
        return 1
    M.RUNS.mkdir(parents=True, exist_ok=True)
    if args.sinks:
        res = {"pos_loss": M.sink_experiment(ex), "attention_to_first": M.attention_to_first(ex)}
        (M.RUNS / "sinks.json").write_text(json.dumps(res))
        return 0
    if args.bench:
        (M.RUNS / "bench.json").write_text(json.dumps(M.bench_prefill(ex), indent=1))
        return 0

    from learnjax.data import detective_corpus

    corpus = detective_corpus()
    cfg = M.variant(args.variant)
    out = M.RUNS / (args.variant if args.steps == 6000 else f"{args.variant}-{args.steps}")
    params = M.init_model(jax.random.key(0), cfg)
    optimizer = M.make_optimizer(args.lr, args.steps)
    opt_state = optimizer.init(params)
    train = jnp.asarray(corpus["train"].astype(np.int32))
    vals = M.val_batches(corpus["val"], args.batch, args.seq_len)
    chunk, eval_pos = M.make_trainer(ex, cfg, optimizer, args.batch, args.seq_len)
    n_params = M.count_params(params)
    print(f"{out.name}: {n_params / 1e6:.2f}M params | {args.steps} steps × {args.batch}×{args.seq_len}", flush=True)
    key = jax.random.key(1)
    t0 = window = time.perf_counter()
    window_steps, seen = 0, []
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.jsonl", "w") as log:
        for step in range(args.chunk, args.steps + 1, args.chunk):
            key, sub = jax.random.split(key)
            steps = np.arange(step - args.chunk, step)
            sparse_on = jnp.asarray(steps >= cfg["dense_warmup"]) if cfg["sparse"] else jnp.ones(args.chunk, bool)
            params, opt_state, losses = chunk(params, opt_state, train, jax.random.split(sub, args.chunk), sparse_on)
            window_steps += args.chunk
            seen.append(losses)
            if step % args.eval_every == 0 or step == args.steps:
                ls = np.concatenate([np.asarray(x) for x in seen])
                if not np.isfinite(ls).all():
                    raise FloatingPointError(f"non-finite loss before step {step}")
                tok_s = window_steps * args.batch * args.seq_len / (time.perf_counter() - window)
                ev = [eval_pos(params, x, y) for x, y in vals]
                val = float(np.mean([np.asarray(p).mean() for p, _ in ev]))
                read = float(np.mean([np.asarray(r).mean() for _, r in ev]))
                m = {"step": step, "loss": float(ls[-args.chunk:].mean()), "val": val, "read": read,
                     "tok_per_s": tok_s, "elapsed_s": time.perf_counter() - t0}
                log.write(json.dumps(m) + "\n")
                log.flush()
                M.save(params, cfg, out, {"steps": step, "batch": args.batch, "seq_len": args.seq_len,
                                          "lr": args.lr, "params": n_params})
                print(f"step {step:6d} | loss {m['loss']:.3f} | val {val:.3f} | keys read {read:.1%} | "
                      f"{tok_s:,.0f} tok/s | {m['elapsed_s'] / 60:.1f} min", flush=True)
                window, window_steps, seen = time.perf_counter(), 0, []
    pos = np.mean([np.asarray(eval_pos(params, x, y)[0]) for x, y in vals], 0)
    (out / "pos_loss.json").write_text(json.dumps([round(float(v), 4) for v in pos]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
