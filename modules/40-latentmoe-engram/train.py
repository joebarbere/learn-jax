"""Train Module 40's variants on the deepseek-v3-jax detective corpus — the long runs behind the lesson's race.

    containers/run.sh python modules/40-latentmoe-engram/train.py --variant moe
    containers/run.sh python modules/40-latentmoe-engram/train.py --variant latent-acc
    containers/run.sh python modules/40-latentmoe-engram/train.py --variant engram --rho 0.75 --engram-layer 1
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/40-latentmoe-engram/train.py ...   # reference answers

Uses *your* exercises.py (finish `make test N=40` first) or solutions.py with LEARNJAX_SOLUTIONS=1. Defaults
match deepseek-v3-jax: its model size (6 layers of width 384, 1 shared + 16 routed experts, top-4), 8K BPE,
12,000 steps of 32 × 256 tokens, AdamW 6e-4 with warmup-cosine. Without multi-token prediction it runs faster
than that project did: about 30 minutes for `moe` and `engram`, 45 for the latent variants, on an RX 7600 XT.

Variants (all at the same total parameter count, ±2%):
    moe          the DeepSeekMoE baseline
    latent-acc   LatentMoE, α = 4: experts in width 96, 64 experts, top-16 (same bytes, more experts)
    latent-eff   LatentMoE, α = 4, top-4 (a quarter of the bytes and expert FLOPs per token)
    engram       --rho of the 16 routed experts kept (0.75 -> 12), the rest of their parameters moved into an
                 Engram table read before layer --engram-layer (0-based)

Writes to /course/runs/40-latentmoe-engram/<name>/ (or --out):
    metrics.jsonl   one line per eval: step, loss, val, tok_per_s, max_load (busiest expert / mean), elapsed_s
    params.npz      checkpoint (plain arrays keyed by pytree path, plus the routers' balancing biases)
    config.json     the model config and training settings
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

RUNS = HERE.parent.parent / "runs" / "40-latentmoe-engram"


def _models():
    spec = importlib.util.spec_from_file_location("m40_models", HERE / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_name(variant: str, rho: float, layer: int, steps: int) -> str:
    name = variant if variant != "engram" else f"engram-r{round(rho * 100)}-l{layer}"
    return name if steps == 12000 else f"{name}-{steps}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", choices=["moe", "latent-acc", "latent-eff", "engram"], default="moe")
    ap.add_argument("--alpha", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.75, help="engram: share of the routed experts kept")
    ap.add_argument("--engram-layer", type=int, default=1)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    assert args.eval_every % args.chunk == 0 and args.steps % args.chunk == 0, "--chunk must divide both"

    from tokenizers import Tokenizer

    from learnjax.data import detective_corpus

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    M, ex = _models(), load_exercises(HERE)
    cfg = M.variant(M.BASE, args.variant, alpha=args.alpha, rho=args.rho, engram_layer=args.engram_layer)
    out = args.out or RUNS / run_name(args.variant, args.rho, args.engram_layer, args.steps)
    corpus = detective_corpus()
    params = M.init_model(jax.random.key(args.seed), 8192, cfg)
    train_np = corpus["train"]
    hashes = val_hashes = None
    try:
        if "engram" in cfg:
            hasher = M.Hasher(ex, M.token_texts(Tokenizer.from_file(str(corpus["tokenizer_path"]))), cfg)
            hashes, val_hashes = jnp.asarray(hasher.corpus(train_np)), hasher.corpus(corpus["val"])
        jax.eval_shape(lambda p: M.forward(ex, p, jnp.zeros((1, 8), jnp.int32),
                                           None if hashes is None else jnp.zeros((1, 8, hashes.shape[1]), jnp.int32),
                                           M.init_biases(cfg), cfg), params)
    except NotImplementedError:
        print("Finish modules/40-latentmoe-engram/exercises.py first (`make test N=40`), or run with "
              "LEARNJAX_SOLUTIONS=1.")
        return 1

    optimizer = M.make_optimizer(params, args.lr, args.steps)
    opt_state, biases = optimizer.init(params), M.init_biases(cfg)
    train = jnp.asarray(train_np.astype(np.int32))
    vals = M.val_batches(corpus["val"], val_hashes, args.batch, args.seq_len)
    chunk, evaluate = M.make_trainer(ex, cfg, optimizer, args.batch, args.seq_len)
    rep = M.param_report(params, cfg)
    print(f"{out.name}: {rep['total']:.1f}M params ({rep['active']:.1f}M active, Engram table {rep['engram']:.1f}M) "
          f"| {args.steps} steps × {args.batch}×{args.seq_len} tokens", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    extra = {"variant": args.variant, "steps": args.steps, "batch": args.batch, "seq_len": args.seq_len,
             "lr": args.lr, "params_M": rep, "data": "detective_corpus v1.0"}
    key = jax.random.key(args.seed + 1)
    t0 = window = time.perf_counter()
    window_steps, losses_seen, loads_seen = 0, [], []
    with open(out / "metrics.jsonl", "w") as log:
        for step in range(args.chunk, args.steps + 1, args.chunk):
            key, sub = jax.random.split(key)
            params, opt_state, biases, losses, loads = chunk(params, opt_state, biases, train, hashes, sub, args.chunk)
            window_steps += args.chunk
            losses_seen.append(losses)
            loads_seen.append(loads[-1])
            if step % args.eval_every == 0 or step == args.steps:
                ls = np.concatenate([np.asarray(x) for x in losses_seen])
                if not np.isfinite(ls).all():
                    raise FloatingPointError(f"non-finite loss before step {step}")
                tok_s = window_steps * args.batch * args.seq_len / (time.perf_counter() - window)
                load = np.asarray(loads_seen[-1])
                val = float(np.mean([evaluate(params, biases, x, y, hw) for x, y, hw in vals]))
                m = {"step": step, "loss": float(ls[-args.chunk:].mean()), "val": val, "tok_per_s": tok_s,
                     "max_load": float((load.max(-1) / load.mean(-1)).max()), "elapsed_s": time.perf_counter() - t0}
                log.write(json.dumps(m) + "\n")
                log.flush()
                M.save(params, biases, cfg, out, {**extra, "step": step})
                print(f"step {step:6d} | loss {m['loss']:.3f} | val {val:.3f} | {tok_s:,.0f} tok/s | "
                      f"max load {m['max_load']:.2f}× | {m['elapsed_s'] / 60:.1f} min", flush=True)
                window, window_steps, losses_seen, loads_seen = time.perf_counter(), 0, [], []
    return 0


if __name__ == "__main__":
    sys.exit(main())
