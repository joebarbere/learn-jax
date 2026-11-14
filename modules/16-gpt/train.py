"""Train a GPT on the deepseek-v3-jax detective corpus — the long run for Module 16.

    containers/run.sh python modules/16-gpt/train.py                     # your exercises.py
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/16-gpt/train.py  # the reference solutions

It uses *your* model code from exercises.py (or solutions.py with LEARNJAX_SOLUTIONS=1), so finish
`make test N=16` first. Default: a ~14M-parameter GPT, 12,000 steps of 32 × 256 tokens — the same
tokens per step, step count, tokenizer, data and optimizer as deepseek-v3-jax, so the two validation
curves line up step for step. About 10–15 minutes on an RX 7600 XT.

Writes to /course/runs/16-gpt/ (or --out):
    metrics.jsonl   one line per chunk: step, loss, lr, elapsed_s (+ val and tok_per_s at each eval)
    params.npz      checkpoint: one array per parameter, keyed by its pytree path (no pickle)
    config.json     the GPTConfig plus training settings
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # the course root, for `learnjax`

from learnjax.course import load_exercises  # after sys.path setup

DEFAULT_OUT = HERE.parent.parent / "runs" / "16-gpt"


# ------------------------------------------------------------------------------------------------
# Checkpoints: plain .npz keyed by pytree path, so loading never executes code (unlike pickle).
# ------------------------------------------------------------------------------------------------
def _key(path) -> str:
    return jax.tree_util.keystr(path)


def save_checkpoint(params, cfg, out: Path, extra: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    flat = {_key(p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    tmp = out / "params.tmp.npz"
    np.savez(tmp, **flat)
    tmp.replace(out / "params.npz")
    (out / "config.json").write_text(json.dumps({"model": dataclasses.asdict(cfg), **(extra or {})}, indent=2))


def load_checkpoint(ex, out: Path = DEFAULT_OUT):
    """Rebuild (params, cfg) from a run directory, using `ex.init_gpt` for the pytree structure."""
    meta = json.loads((out / "config.json").read_text())
    cfg = ex.GPTConfig(**meta["model"])
    template = ex.init_gpt(jax.random.key(0), cfg)
    arrays = np.load(out / "params.npz")
    params = jax.tree_util.tree_map_with_path(lambda p, _: jnp.asarray(arrays[_key(p)]), template)
    return params, cfg, meta


# ------------------------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------------------------
def make_train_chunk(ex, cfg, optimizer, batch: int, steps: int):
    """A jitted function that runs `steps` training steps as one lax.scan (Module 10's lesson)."""

    # donate_argnums: params and optimizer state are consumed, so XLA may write the updated ones into the
    # same GPU buffers instead of holding old + new copies (the JAX Training Cookbook's advice).
    @partial(jax.jit, donate_argnums=(0, 1))
    def chunk(params, opt_state, data, key):
        def step(carry, k):
            params, opt_state = carry
            tokens = ex.get_batch(k, data, batch, cfg.seq_len)

            def loss(p):  # mixed precision: f32 master weights, bf16 compute (Module 08)
                return ex.loss_fn(jax.tree.map(lambda a: a.astype(jnp.bfloat16), p), tokens, cfg)

            value, grads = jax.value_and_grad(loss)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), value

        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), jax.random.split(key, steps))
        return params, opt_state, losses

    return chunk


def lr_at(step: int, peak: float, warmup: int, total: int, final: float) -> float:
    """The warmup-cosine schedule, computed host-side with plain Python for logging.

    The optimizer evaluates the same schedule *inside* the jitted step. Computing it here with jnp would
    launch a tiny GPU computation and wait for the result every step — a host<->device round trip.
    """
    if step < warmup:
        return peak * step / max(warmup, 1)
    frac = min((step - warmup) / max(total - warmup, 1), 1.0)
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * frac))


def enable_compilation_cache(path: str = "/cache/jaxcache") -> None:
    """Persist compiled XLA programs across runs (the /cache volume survives container restarts)."""
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def make_eval(ex, cfg):
    return jax.jit(lambda params, tokens: ex.loss_fn(params, tokens, cfg))  # f32, like deepseek-v3-jax


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--untied", action="store_true", help="separate output head instead of tied embeddings")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=50, help="steps per jitted lax.scan call (divides eval-every)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    assert args.eval_every % args.chunk == 0 and args.steps % args.chunk == 0, "--chunk must divide both"

    from learnjax.data import detective_corpus

    enable_compilation_cache()
    ex = load_exercises(HERE)
    corpus = detective_corpus()
    cfg = ex.GPTConfig(vocab=8192, seq_len=args.seq_len, d_model=args.d_model, n_layers=args.layers,
                       n_heads=args.heads, tie_embeddings=not args.untied)
    try:
        n_params = ex.count_params(cfg)
        params = ex.init_gpt(jax.random.key(args.seed), cfg)
    except NotImplementedError:
        print("Finish modules/16-gpt/exercises.py first (`make test N=16`), or run with LEARNJAX_SOLUTIONS=1.")
        return 1

    warmup = min(args.warmup, args.steps // 2)
    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup, args.steps, args.lr * 0.1)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=0.1))
    opt_state = optimizer.init(params)
    train = jnp.asarray(corpus["train"].astype(np.int32))
    val_rng = np.random.default_rng(1234)
    val_starts = val_rng.integers(0, len(corpus["val"]) - args.seq_len - 1, (8, args.batch))
    val_batches = [jnp.asarray(np.stack([corpus["val"][s:s + args.seq_len + 1] for s in row]).astype(np.int32))
                   for row in val_starts]
    chunk = make_train_chunk(ex, cfg, optimizer, args.batch, args.chunk)
    evaluate = make_eval(ex, cfg)

    print(f"{jax.default_backend()} | {n_params / 1e6:.1f}M params | {len(train) / 1e6:.1f}M train tokens | "
          f"{args.steps} steps × {args.batch}×{args.seq_len} tokens", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    extra = {"steps": args.steps, "batch": args.batch, "lr": args.lr, "data": "detective_corpus v1.0"}
    key = jax.random.key(args.seed + 1)
    t_start = time.perf_counter()
    clock = {"window_start": time.perf_counter(), "window_steps": 0, "first_chunk_s": None}

    def finish(step, losses) -> dict:
        """Turn a chunk's device-side losses into a metrics line. np.asarray is where Python waits."""
        losses = np.asarray(losses)
        now = time.perf_counter()
        if clock["first_chunk_s"] is None:
            clock["first_chunk_s"] = now - t_start  # includes compilation (or a compilation-cache hit)
        if not np.isfinite(losses).all():
            raise FloatingPointError(f"non-finite loss at step {step}")
        return {"step": step, "loss": float(losses.mean()),
                "lr": lr_at(step, args.lr, warmup, args.steps, args.lr * 0.1), "elapsed_s": now - t_start}

    with open(args.out / "metrics.jsonl", "w") as log:
        pending = None
        for step in range(args.chunk, args.steps + 1, args.chunk):
            key, sub = jax.random.split(key)
            # Dispatch is asynchronous: this returns at once with arrays that are still being computed.
            params, opt_state, losses = chunk(params, opt_state, train, sub)
            clock["window_steps"] += args.chunk
            if pending is not None:
                # Deferred logging: read the PREVIOUS chunk's losses while this one already runs, so the
                # GPU never idles while Python converts and writes metrics (no "step bubble").
                log.write(json.dumps(finish(*pending)) + "\n")
            pending = (step, losses)
            if step % args.eval_every == 0 or step == args.steps:
                # Evaluation needs finished params, so every eval_every steps we do synchronize. Throughput
                # is measured over the whole window since the last sync: with asynchronous dispatch, the
                # wall-clock time between two syncs is the only honest timer.
                m = finish(*pending)
                pending = None
                m["tok_per_s"] = (clock["window_steps"] * args.batch * args.seq_len
                                  / (time.perf_counter() - clock["window_start"]))
                m["val"] = float(np.mean([evaluate(params, b) for b in val_batches]))
                save_checkpoint(params, cfg, args.out,
                                {**extra, "step": step, "first_chunk_s": clock["first_chunk_s"]})
                log.write(json.dumps(m) + "\n")
                log.flush()
                print(f"step {step:6d} | loss {m['loss']:.3f} | val {m['val']:.3f} | {m['tok_per_s']:,.0f} tok/s | "
                      f"lr {m['lr']:.2e} | {m['elapsed_s'] / 60:.1f} min", flush=True)
                clock["window_start"], clock["window_steps"] = time.perf_counter(), 0  # exclude eval time
    print(f"first chunk (compilation + {args.chunk} steps): {clock['first_chunk_s']:.1f} s — re-runs reuse "
          "/cache/jaxcache", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
