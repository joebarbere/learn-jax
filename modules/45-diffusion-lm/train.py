"""Train Module 45's masked-diffusion LM on the detective corpus, then benchmark sampling against Module 41's AR model.

    containers/run.sh python modules/45-diffusion-lm/train.py              # ~40 min on an RX 7600 XT
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/45-diffusion-lm/train.py   # reference answers

Uses *your* exercises.py (finish `make test N=45` first) or solutions.py with LEARNJAX_SOLUTIONS=1. Same budget as
Module 41's runs: 12,000 steps of 16 × 512 tokens (98M tokens), AdamW 1e-3 warmup-cosine.

Writes /course/runs/45-diffusion-lm/mdlm/: metrics.jsonl (step, loss, val bound), params.npz, config.json, and
sampling.json: 256/64/16/4 denoising steps, both reveal orders, and block-wise sampling — the AR model's loss on
the generated text, its unigram entropy, its repeated 4-grams, the time per sample and one sample; plus the AR
model itself and real text.
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
    spec = importlib.util.spec_from_file_location("m45_models", HERE / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def val_windows(val, n, T, seed=1234):
    starts = np.random.default_rng(seed).integers(0, len(val) - T, n)
    return [val[s:s + T].astype(np.int32) for s in starts]


def sampling_bench(M, ex, params, val, out, length=256, prompt_len=32, n=8):
    ar = M.ar_model()
    if ar is None:
        print("no Module 41 attention run: skipping the sampling benchmark")
        return
    fwd, ap = ar
    prompts = jnp.asarray(np.stack([w[:prompt_len] for w in val_windows(val, n, prompt_len, seed=7)]))
    res = {}

    def record(name, fn):
        fn()                                                         # compile
        t0 = time.perf_counter()
        seqs = jax.block_until_ready(fn())
        dt = (time.perf_counter() - t0) / n
        seqs = np.asarray(seqs)
        res[name] = {"gen_nll": M.gen_nll(fwd, ap, seqs, prompt_len),
                     "entropy": M.unigram_entropy(seqs[:, prompt_len:]), "repeat_4grams": M.repeat_4grams(
                         seqs[:, prompt_len:]), "seconds_per_sample": dt, "sample": seqs[0].tolist()}
        print(name, {k: round(v, 3) for k, v in res[name].items() if k != "sample"}, flush=True)

    for steps in (256, 64, 16, 4):
        record(f"mdlm {steps} steps", lambda s=steps: M.sample(ex, params, prompts, length, s, jax.random.key(1)))
    record("mdlm 64 steps, random order",
           lambda: M.sample(ex, params, prompts, length, 64, jax.random.key(1), by_confidence=False))
    record("mdlm 256 steps, random order",
           lambda: M.sample(ex, params, prompts, length, 256, jax.random.key(1), by_confidence=False))
    record("mdlm 64 steps, blocks of 32",
           lambda: M.sample(ex, params, prompts, length, 64, jax.random.key(1), block=32))
    record("mdlm 256 steps, blocks of 32",
           lambda: M.sample(ex, params, prompts, length, 256, jax.random.key(1), block=32))
    record("autoregressive (Module 41)", lambda: M.ar_sample(fwd, ap, prompts, length, jax.random.key(1)))
    real = np.stack([w for w in val_windows(val, n, prompt_len + length, seed=7)])
    res["real text"] = {"gen_nll": M.gen_nll(fwd, ap, real, prompt_len),
                        "entropy": M.unigram_entropy(real[:, prompt_len:]),
                        "repeat_4grams": M.repeat_4grams(real[:, prompt_len:]), "seconds_per_sample": 0.0,
                        "sample": real[0].tolist()}
    (out / "sampling.json").write_text(json.dumps(res, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--bench-only", action="store_true", help="skip training; rerun the sampling benchmark")
    args = ap.parse_args(argv)

    from learnjax.data import detective_corpus

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    M, ex = _models(), load_exercises(HERE)
    try:
        ex.mask_tokens(jax.random.key(0), jnp.zeros((1, 4), jnp.int32), jnp.ones(1), 5)
        ex.reveal_schedule(4, 2)
        ex.block_order(4, 0, 2)
    except NotImplementedError:
        print("Finish modules/45-diffusion-lm/exercises.py first (`make test N=45`), or run with LEARNJAX_SOLUTIONS=1.")
        return 1
    corpus = detective_corpus()
    out = M.RUNS / ("mdlm" if args.steps == 12000 else f"mdlm-{args.steps}")
    if args.bench_only:
        sampling_bench(M, ex, M.load(out), corpus["val"], out)
        return 0
    params = M.init_model(jax.random.key(0))
    optimizer = M.make_optimizer(args.lr, args.steps)
    opt_state = optimizer.init(params)
    train = jnp.asarray(corpus["train"].astype(np.int32))
    vw = val_windows(corpus["val"], 8, args.seq_len)
    chunk = M.make_trainer(ex, optimizer, args.batch, args.seq_len)
    n_params = M.count_params(params)
    print(f"{out.name}: {n_params / 1e6:.1f}M params | {args.steps} steps × {args.batch}×{args.seq_len}", flush=True)
    key = jax.random.key(1)
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
                val = M.nll_bound(ex, params, vw, n_t=8)
                m = {"step": step, "loss": float(ls[-args.chunk:].mean()), "val": val, "tok_per_s": tok_s,
                     "elapsed_s": time.perf_counter() - t0}
                log.write(json.dumps(m) + "\n")
                log.flush()
                M.save(params, out, {"steps": step, "batch": args.batch, "seq_len": args.seq_len, "lr": args.lr,
                                     "params": n_params})
                print(f"step {step:6d} | loss {m['loss']:.3f} | val bound {val:.3f} | {tok_s:,.0f} tok/s | "
                      f"{m['elapsed_s'] / 60:.1f} min", flush=True)
                window, window_steps, seen = time.perf_counter(), 0, []
    final = M.nll_bound(ex, params, val_windows(corpus["val"], 64, args.seq_len), n_t=32)
    (out / "final.json").write_text(json.dumps({"val_bound_64x32": final}))
    print(f"final bound (64 windows × 32 masking rates): {final:.3f}", flush=True)
    sampling_bench(M, ex, params, corpus["val"], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
