"""Train the Othello-GPT used in Module 37's lesson (about 10 minutes on an RX 7600 XT).

    containers/run.sh python modules/37-interpretability/train.py [--steps 6000] [--games 60000]

Writes runs/37-interpretability/othello_gpt.npz (weights, no pickle) and othello_gpt.json (config,
loss curve, legal-move accuracy on held-out games). The lesson loads these if they exist.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
HERE = Path(__file__).resolve().parent
OUT = Path("/course/runs/37-interpretability")


def load_interp():
    spec = importlib.util.spec_from_file_location("interp37", HERE / "interp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--games", type=int, default=60000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args(argv)
    interp = load_interp()
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    games = interp.gen_games(jax.random.key(0), args.games + 2000)
    train = {k: v[:-2000] for k, v in games.items()}
    test = {k: v[-2000:] for k, v in games.items()}
    print(f"generated {args.games + 2000} games in {time.perf_counter() - t0:.1f}s", flush=True)

    def log(h):
        if len(h["step"]) % 10 == 0:
            print(f"step {h['step'][-1]:5d}  loss {h['loss'][-1]:.3f}", flush=True)

    params, hist, secs = interp.train_othello_gpt(
        jax.random.key(1), train, d=args.d, n_layers=args.layers, n_heads=args.heads, steps=args.steps,
        batch=args.batch, lr=args.lr, on_log=log)
    acc = interp.legal_move_accuracy(params, test, args.heads, n_eval=2000)
    meta = {"d": args.d, "n_layers": args.layers, "n_heads": args.heads, "steps": args.steps, "games": args.games,
            "train_seconds": round(secs, 1), "legal_move_accuracy": acc, "loss": hist["loss"], "loss_step": hist["step"],
            "device": str(jax.devices()[0])}
    interp.save_params(OUT / "othello_gpt.npz", params, meta)
    (OUT / "othello_gpt.json").write_text(json.dumps(meta))
    print(f"trained {args.steps} steps in {secs:.0f}s; held-out legal-move accuracy {acc:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
