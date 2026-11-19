"""Module 42's long runs: the teacher pass, then students, continued pretraining, and prune-and-heal.

    T="containers/run.sh python modules/42-small-models/train.py"
    $T --run teacher        # SmolLM2-360M's top-16 predictions over the whole corpus (do this first)
    $T --run scratch        # 31M student, cross-entropy only
    $T --run kd             # the same student, distilled from the teacher's top-16 only
    $T --run kd-mix         # ... half distillation, half cross-entropy on the true token (Hinton's recipe)
    $T --run muon           # cross-entropy, Muon for the hidden matrices
    $T --run data-best      # cross-entropy on the half of the corpus the teacher finds most predictable
    $T --run data-random    # ... and on a random half (the control)
    $T --run cpt135         # SmolLM2-135M (pretrained on 2T tokens), continued on the corpus
    $T --run pruned         # SmolLM2-360M pruned to 24 layers and half its FFN, healed by kd-mix
    $T --run pruned-ce      # ... healed with cross-entropy instead
    LEARNJAX_SOLUTIONS=1 $T --run ...                        # with the reference exercise answers

Uses *your* exercises.py (finish `make test N=42` first) or solutions.py with LEARNJAX_SOLUTIONS=1.
Students: 6,000 steps of 16 × 512 tokens (49M tokens, about 2.4 passes over the corpus). The 135M and
pruned models: 4,000 steps of 8 × 512 (16M tokens; batch 16 doesn't fit in 16 GB with a 49,152-token output). Everything is scored on the same validation windows, in nats per SmolLM2 token.

Writes /course/runs/42-small-models/<run>/ (metrics.jsonl, params.npz, config.json, sample.txt) and
teacher/ (top_ids.npy, top_logp.npy, nll.npy, baselines.json).
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

RUNS = HERE.parent.parent / "runs" / "42-small-models"
RUN_NAMES = ["teacher", "scratch", "kd", "kd-mix", "muon", "data-best", "data-random", "cpt135", "pruned", "pruned-ce"]
PROMPT = "Holmes looked at the letter for a long moment, and then"


def _models():
    spec = importlib.util.spec_from_file_location("m42_models", HERE / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sample(M, params, cfg, n=60):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(M.llama.download(M.REFERENCE) / "tokenizer.json"))
    ids = M.llama.generate(jax.tree.map(lambda a: a.astype(jnp.bfloat16), params),
                           M.llama.Config(**{**cfg.__dict__, "dtype": "bfloat16"}),
                           tok.encode(PROMPT).ids, n, jax.random.key(0), temperature=0.7)
    return PROMPT + tok.decode(list(ids))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", choices=RUN_NAMES, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None, help="default 16; 8 for the 135M and pruned models")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args(argv)

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    M, ex = _models(), load_exercises(HERE)
    try:
        jax.eval_shape(lambda: ex.kd_loss(jnp.zeros((2, 8)), jnp.zeros((2, 3), jnp.int32), jnp.zeros((2, 3))))
        jax.eval_shape(lambda: ex.newton_schulz(jnp.ones((4, 3))))
        jax.eval_shape(lambda: ex.ffn_importance(jnp.zeros((2, 3))))
        jax.eval_shape(lambda: ex.layer_importance(jnp.ones((2, 3)), jnp.ones((2, 3))))
        jax.eval_shape(lambda: ex.prune_ffn({"w_gate": jnp.zeros((1, 2, 3)), "w_up": jnp.zeros((1, 2, 3)),
                                             "w_down": jnp.zeros((1, 3, 2))}, jnp.zeros((1, 2), jnp.int32)))
    except NotImplementedError:
        print("Finish modules/42-small-models/exercises.py first (`make test N=42`), or run with LEARNJAX_SOLUTIONS=1.")
        return 1

    teacher_dir = RUNS / "teacher"
    vw = M.val_windows()
    if args.run == "teacher":
        M.teacher_pass(teacher_dir, log=lambda s: print(s, flush=True))
        base = {}
        for name, repo in (("SmolLM2-360M (teacher)", M.TEACHER), ("SmolLM2-135M", M.REFERENCE)):
            p, c, _ = M.llama.load(repo, max_seq=M.SEQ)
            base[name] = {"val": M.val_loss(p, c, vw), "params": M.llama.count_params(p)}
            print(name, base[name], flush=True)
        (teacher_dir / "baselines.json").write_text(json.dumps(base, indent=1))
        return 0

    win = M.windows(M.smol_corpus()["train"])
    n_win = len(win)
    kind, kd_w, lr, allowed = "adamw", 0.0, 1e-3, np.arange(n_win)
    top_ids = top_logp = None
    if args.run in ("kd", "kd-mix", "pruned", "data-best", "data-random") and not (teacher_dir / "nll.npy").exists():
        print("Run `--run teacher` first.")
        return 1
    if args.run in ("kd", "kd-mix", "pruned"):
        top_ids, top_logp = np.load(teacher_dir / "top_ids.npy"), np.load(teacher_dir / "top_logp.npy")
        n_win = len(top_ids)
        allowed, kd_w = np.arange(n_win), (1.0 if args.run == "kd" else 0.5)   # kd-mix, pruned: half each
    if args.run in ("data-best", "data-random"):
        nll = np.load(teacher_dir / "nll.npy")
        n_win = len(nll)
        allowed = (np.sort(np.argsort(nll)[: n_win // 2]) if args.run == "data-best"
                   else np.sort(np.random.default_rng(0).permutation(n_win)[: n_win // 2]))
    extra = {}
    big = args.run in ("cpt135", "pruned", "pruned-ce")
    args.batch = args.batch or (8 if big else 16)      # 16 × 512 × 49,152 logits OOM the 16 GB card for 135M+
    if big:
        steps = args.steps or 4000                     # 4,000 × 8 × 512 = 16M tokens
        lr = 3e-4
        if args.run == "cpt135":
            params, cfg, _ = M.llama.load(M.REFERENCE, max_seq=M.SEQ, dtype="float32")
        else:
            params, cfg, _ = M.llama.load(M.TEACHER, max_seq=M.SEQ, dtype="float32")
            calib = win[np.random.default_rng(1).choice(len(win), 64, replace=False)].astype(np.int32)
            imp, bi = M.calibration_stats(ex, params, cfg, calib)
            params, cfg, kept = M.prune(ex, params, cfg, imp, bi, keep_layers=24, keep_ffn=cfg.d_ff // 2)
            extra = {"layers_kept": kept.tolist(), "block_influence": np.round(bi, 4).tolist(),
                     "val_before_healing": M.val_loss(params, cfg, vw)}
            print(f"pruned: {M.llama.count_params(params) / 1e6:.0f}M params, val before healing "
                  f"{extra['val_before_healing']:.3f}", flush=True)
        params = jax.tree.map(lambda a: a.astype(jnp.float32), params)
        cfg = M.llama.Config(**{**cfg.__dict__, "dtype": "float32"})
    else:
        steps = args.steps or 6000
        cfg = M.STUDENT
        params = M.init_params(jax.random.key(0), cfg)
        if args.run == "muon":
            kind = "muon"
    out = RUNS / (args.run if args.steps is None else f"{args.run}-{args.steps}")
    optimizer = M.make_optimizer(params, lr, steps, kind=kind, parts=ex)
    opt_state = optimizer.init(params)
    data = (jnp.asarray(win[:n_win].astype(np.int32)),
            None if top_ids is None else jnp.asarray(top_ids), None if top_logp is None else jnp.asarray(top_logp),
            jnp.asarray(allowed))
    chunk, _ = M.make_trainer(ex, cfg, optimizer, kd_weight=kd_w, batch=args.batch)
    n_params = M.llama.count_params(params)
    print(f"{out.name}: {n_params / 1e6:.1f}M params | {steps} steps × {args.batch}×{M.SEQ} | {kind}, "
          f"kd weight {kd_w} | {len(allowed):,} windows", flush=True)
    meta = {"run": args.run, "steps": steps, "batch": args.batch, "lr": lr, "optimizer": kind, "kd_weight": kd_w,
            "params": n_params, "windows": len(allowed), "cfg": cfg.__dict__, **extra}
    key = jax.random.key(1)
    t0 = window = time.perf_counter()
    window_steps, seen = 0, []
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.jsonl", "w") as log:
        for step in range(args.chunk, steps + 1, args.chunk):
            key, sub = jax.random.split(key)
            params, opt_state, ces = chunk(params, opt_state, data, sub, args.chunk)
            window_steps += args.chunk
            seen.append(ces)
            if step % args.eval_every == 0 or step == steps:
                ls = np.concatenate([np.asarray(x) for x in seen])
                if not np.isfinite(ls).all():
                    raise FloatingPointError(f"non-finite loss before step {step}")
                tok_s = window_steps * args.batch * M.SEQ / (time.perf_counter() - window)
                val = M.val_loss(params, cfg, vw)
                m = {"step": step, "loss": float(ls[-args.chunk:].mean()), "val": val, "tok_per_s": tok_s,
                     "elapsed_s": time.perf_counter() - t0}
                log.write(json.dumps(m) + "\n")
                log.flush()
                print(f"step {step:6d} | train ce {m['loss']:.3f} | val {val:.3f} | {tok_s:,.0f} tok/s | "
                      f"{m['elapsed_s'] / 60:.1f} min", flush=True)
                window, window_steps, seen = time.perf_counter(), 0, []
    M.save(params, out, meta)
    (out / "sample.txt").write_text(sample(M, params, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
