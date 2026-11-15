"""Module 33: a longer class-conditional DDPM run on Fashion-MNIST (the lesson trains a short one live).

    containers/run.sh python modules/33-generative-images/train.py            # ~5 min on an idle RX 7600 XT
    containers/run.sh python modules/33-generative-images/train.py --steps 4000

Writes runs/33-generative-images/ddpm.npz (EMA weights), ddpm_metrics.json. Uses your exercises.py
loss functions, or the reference ones with LEARNJAX_SOLUTIONS=1.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import gen

from learnjax.course import load_exercises

jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    args = ap.parse_args()

    ex = load_exercises(HERE)
    try:
        _betas, abar = ex.ddpm_schedule(1000)
        ex.q_sample(jnp.zeros((1, 28, 28, 1)), jnp.zeros(1, jnp.int32), jnp.zeros((1, 28, 28, 1)), abar)
        ex.ddpm_loss(jnp.zeros(2), jnp.zeros(2))
    except NotImplementedError:
        print("exercises.py's diffusion functions aren't done yet; using the reference solutions.")
        ex = load_exercises(HERE, solutions=True)
        _betas, abar = ex.ddpm_schedule(1000)

    tx, ty, _, _ = gen.load_data()
    model = gen.UNet(nnx.Rngs(0))
    opt = nnx.Optimizer(model, optax.adam(args.lr), wrt=nnx.Param)
    step = gen.make_ddpm_step(ex.q_sample, ex.ddpm_loss, abar, 1000)
    ema = jax.tree.map(jnp.array, nnx.state(model, nnx.Param))

    @jax.jit
    def ema_update(ema, params):
        return jax.tree.map(lambda e, p: args.ema * e + (1 - args.ema) * p, ema, params)

    rng = np.random.default_rng(0)
    key = jax.random.key(1)
    hist, t0, s = [], time.perf_counter(), 0
    while s < args.steps:
        for b in gen.batches(rng, len(tx), args.batch):
            key, k = jax.random.split(key)
            loss = step(model, opt, tx[b], ty[b], k)
            ema = ema_update(ema, nnx.state(model, nnx.Param))
            if s % 250 == 0:
                hist.append({"step": s, "loss": float(loss), "seconds": time.perf_counter() - t0})
                print(f"step {s:6d}  loss {float(loss):.4f}  {time.perf_counter() - t0:6.0f}s", flush=True)
            s += 1
            if s >= args.steps:
                break
    nnx.update(model, ema)
    gen.save_model(model, gen.RUNS / "ddpm.npz")
    (gen.RUNS / "ddpm_metrics.json").write_text(json.dumps(
        {"steps": args.steps, "batch": args.batch, "seconds": time.perf_counter() - t0, "history": hist}))
    print(f"saved {gen.RUNS / 'ddpm.npz'} after {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
