"""Module 34: a longer AlphaZero run on Connect Four (pgx + mctx), self-play on the GPU.

    containers/run.sh python modules/34-deep-rl-alphazero/train.py                 # uses YOUR exercises.py
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/34-deep-rl-alphazero/train.py --iterations 40

Each iteration plays `--batch` games of self-play with Gumbel MuZero search (`--sims` simulations per move),
then takes `--train-steps` gradient steps on (position, MCTS policy, game outcome). Every `--eval-every`
iterations it plays 128 games against a uniformly random player and 64 games against pure MCTS with random
rollouts (64 simulations). About 10–15 s per iteration on an idle RX 7600 XT.
Writes /course/runs/34-deep-rl-alphazero/metrics.json and params.npz (the lesson loads both).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import jax

import learnjax as lj

HERE = Path(__file__).resolve().parent
OUT = Path("/course/runs/34-deep-rl-alphazero")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"m34_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    rl = _load("rl")
    ex = lj.load_exercises(HERE)
    OUT.mkdir(parents=True, exist_ok=True)

    def log(rec):
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)

    params, hist = rl.alphazero_loop(ex, jax.random.key(args.seed), args.iterations, batch=args.batch,
                                     sims=args.sims, train_steps=args.train_steps, eval_every=args.eval_every,
                                     log=log)
    rl.save_params(OUT / "params.npz", params)
    (OUT / "metrics.json").write_text(json.dumps({"args": vars(args), "history": hist}, indent=1))
    print(f"wrote {OUT / 'metrics.json'} and params.npz")


if __name__ == "__main__":
    main()
