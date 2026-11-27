"""Module 31b's long run: every model x 3 seeds x 3 training-set sizes, plus the timing table (~20 min).

    containers/run.sh python modules/31b-mdrnn/train.py           # uses YOUR exercises.py
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/31b-mdrnn/train.py

Each condition trains on the same number of examples in total (300k: 5 epochs of 60k, 60 of 5k, 300 of 1k),
so small-data runs aren't simply shorter. Every model is also tested with a random 8x8-pixel square blacked
out. Writes /course/runs/31b-mdrnn/results.json, which the lesson plots.
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import models as M

from learnjax.course import load_exercises

ex = load_exercises(HERE)
fns = {"sweep": ex.mdrnn_sweep, "four_direction_sweeps": ex.four_direction_sweeps,
       "cross_scan": ex.cross_scan, "cross_merge": ex.cross_merge}
OUT = HERE.parents[1] / "runs" / "31b-mdrnn"
OUT.mkdir(parents=True, exist_ok=True)
data = M.load_data()
SIZES = [(None, 5), (5000, 60), (1000, 300)]          # (n_train, epochs): 300k examples seen each
SEEDS = [0, 1, 2]
res = {"runs": [], "timing": {}, "jax": jax.__version__, "device": str(jax.devices()[0])}


def log(s):
    print(s, flush=True)


# timing: one forward+backward over a batch of 256, after compiling
x = jnp.asarray(data[0][:256])
p1 = M.init_sweep(jax.random.key(0), 4, M.H)
res["timing"]["sweep_cellwise_s"] = M.time_forward_backward(M.sweep_cellwise, (p1, x))
res["timing"]["sweep_wavefront_s"] = M.time_forward_backward(ex.mdrnn_sweep, (p1, x))
pl = M.init("linscan", jax.random.key(0))
res["timing"]["linscan_parallel_s"] = M.time_forward_backward(lambda p, x: M.forward("linscan", p, x, fns, True), (pl, x))
res["timing"]["linscan_sequential_s"] = M.time_forward_backward(lambda p, x: M.forward("linscan", p, x, fns, False), (pl, x))
log(f"timing (fwd+bwd, batch 256): {res['timing']}")

t_all = time.time()
for n_train, epochs in SIZES:
    for seed in SEEDS:
        for kind in M.KINDS:
            _, r = M.train(kind, fns, data, epochs=epochs, n_train=n_train, seed=seed)
            res["runs"].append(r)
            log(f"{kind:16s} n={r['n_train']:5d} seed {seed}: acc {r['acc']:.4f}  occluded {r['acc_occluded_4']:.4f}  "
                f"{r['params']} params  {r['s_per_epoch']:.2f}s/epoch")
            (OUT / "results.json").write_text(json.dumps(res, indent=1))
log(f"done in {(time.time() - t_all) / 60:.1f} min -> {OUT / 'results.json'}")
