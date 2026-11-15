"""Module 26: a longer GRPO run on SmolLM2-360M-Instruct with LoRA adapters.

    containers/run.sh python modules/26-reasoning/train.py            # uses YOUR exercises.py
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/26-reasoning/train.py
    ... train.py --steps 120 --group 8 --prompts 8 --lr 2e-4

Each step samples `group` completions for each of `prompts` arithmetic problems, scores them with your
verifier-based reward, and takes one GRPO step on the LoRA adapters (KL to the base model, beta 0.04).
Every `--eval-every` steps it measures greedy accuracy on a 64-problem *validation* split (used to pick the
best checkpoint) and the 128-problem *test* split (reported, never used for choices). ~5 s per step on an
RX 7600 XT (60 steps ≈ 6 min). Writes /course/runs/26-reasoning/metrics.jsonl, adapters.npz (best by
validation — what the lesson evaluates and Module 27 exports to LM Studio) and adapters_last.npz.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax

import learnjax as lj
from learnjax import llama

HERE = Path(__file__).resolve().parent
OUT = Path("/course/runs/26-reasoning")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"m26_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    ex = lj.load_exercises(HERE)
    R = _load("reasoning")
    params, cfg, tok = llama.load(llama.SMOLLM_360M)
    test, train, val = R.make_split(1, 128), R.make_split(2, args.steps * args.prompts), R.make_split(3, 64)

    adapters = R.init_adapters(jax.random.key(args.seed), cfg)
    opt, update = R.make_grpo(cfg, ex.clipped_pg_loss, ex.kl_penalty, lr=args.lr, beta=args.beta)
    state = {"adapters": adapters, "opt_state": opt.init(adapters)}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "metrics.jsonl", "w") as log:

        best = {"val": -1.0}

        def evaluate(step):
            val_acc, _ = R.greedy_accuracy(params, state["adapters"], cfg, tok, val, ex.verify_answer)
            acc, _ = R.greedy_accuracy(params, state["adapters"], cfg, tok, test, ex.verify_answer)
            log.write(json.dumps({"step": step, "val_acc": val_acc, "test_acc": acc}) + "\n")
            log.flush()
            if val_acc > best["val"]:                 # choose on validation, report on test
                best.update(val=val_acc, step=step)
                R.save_adapters(state["adapters"], OUT / "adapters.npz")
            print(f"step {step:4d} | val {val_acc:.3f} | test {acc:.3f} | best val so far @ step {best['step']}",
                  flush=True)

        evaluate(0)
        key = jax.random.key(args.seed + 1)
        t0 = time.perf_counter()
        for step in range(1, args.steps + 1):
            key, sub = jax.random.split(key)
            batch = train[(step - 1) * args.prompts: step * args.prompts]
            state, stats, _ = R.grpo_step(state, params, cfg, tok, batch, ex.combined_reward, ex.group_advantages,
                                          update, sub, group=args.group)
            log.write(json.dumps({"step": step, **stats, "elapsed_s": time.perf_counter() - t0}) + "\n")
            log.flush()
            print(f"step {step:4d} | reward {stats['reward']:.3f} | kl {stats['kl']:.4f} | "
                  f"len {stats['length']:.1f} | {time.perf_counter() - t0:.0f}s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                evaluate(step)
        R.save_adapters(state["adapters"], OUT / "adapters_last.npz")
        log.write(json.dumps({"best_step": best["step"], "best_val": best["val"]}) + "\n")
        print(f"saved {OUT / 'adapters.npz'} (best validation, step {best['step']}) and adapters_last.npz")


if __name__ == "__main__":
    main()
