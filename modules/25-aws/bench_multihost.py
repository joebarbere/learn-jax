"""Multi-host benchmark: one JAX program spanning two machines with `jax.distributed`.

On AWS (profile `multihost`: 2 × g6.xlarge, one L4 each), run_remote.sh starts this on both nodes:

    python modules/25-aws/bench_multihost.py --coordinator 10.42.1.10:12355 --num-processes 2 --process-id 0
    python modules/25-aws/bench_multihost.py --coordinator 10.42.1.10:12355 --num-processes 2 --process-id 1

Locally, two CPU processes on one machine show the same mechanics (no GPUs involved):

    containers/run.sh python modules/25-aws/bench_multihost.py --local-test

What it shows: after `jax.distributed.initialize`, `jax.devices()` lists *every* process's devices while
`jax.local_devices()` lists this one's; a mesh spans both machines; each process feeds only its own slice of
the batch (`jax.make_array_from_process_local_data`); a data-parallel step's gradient all-reduce crosses the
network; and psum bandwidth between hosts is compared with what Module 25 measures inside one machine.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coordinator", default=None, help="host:port of process 0")
    ap.add_argument("--num-processes", type=int, default=int(os.environ.get("LEARNJAX_NODES", "1")))
    ap.add_argument("--process-id", type=int, default=int(os.environ.get("LEARNJAX_RANK", "0")))
    ap.add_argument("--cpu", action="store_true", help="CPU backend with gloo collectives")
    ap.add_argument("--local-test", action="store_true", help="spawn 2 CPU processes on this machine")
    ap.add_argument("--label", default=None)
    return ap.parse_args()


ARGS = parse()

if ARGS.local_test:  # parent: launch two CPU workers talking over localhost, then exit
    procs = [subprocess.Popen([sys.executable, __file__, "--cpu", "--coordinator", "127.0.0.1:12355",
                               "--num-processes", "2", "--process-id", str(i), "--label", "local-2proc-cpu"])
             for i in range(2)]
    sys.exit(max(p.wait(timeout=600) for p in procs))

if ARGS.cpu:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax

if ARGS.cpu:
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
jax.distributed.initialize(coordinator_address=ARGS.coordinator, num_processes=ARGS.num_processes,
                           process_id=ARGS.process_id)

import numpy as np
import optax
from bench_common import device_info, machine_label, save, section
from jax.sharding import NamedSharding

V, D, F, B_PER_PROC, T = (512, 128, 512, 8, 16) if ARGS.cpu else (8192, 1024, 4096, 16, 256)
SHAPES = {"emb": (V, D), "w1": (D, F), "w2": (F, D), "head": (D, V)}
OPT = optax.adam(1e-3)


def loss_fn(p, x, y):
    h = p["emb"][x]
    h = h + jax.nn.gelu(h @ p["w1"]) @ p["w2"]
    return optax.softmax_cross_entropy_with_integer_labels(h @ p["head"], y).mean()


def train_step(p, s, x, y):
    loss, g = jax.value_and_grad(loss_fn)(p, x, y)
    upd, s = OPT.update(g, s, p)
    return optax.apply_updates(p, upd), s, loss


def topology():
    return {"process_index": jax.process_index(), "process_count": jax.process_count(),
            "global_devices": [str(d) for d in jax.devices()], "local_devices": [str(d) for d in jax.local_devices()]}


def dp_step(steps: int = 10):
    mesh = jax.make_mesh((jax.device_count(),), ("data",), axis_types=(jax.sharding.AxisType.Auto,))
    rep, data = NamedSharding(mesh, jax.P()), NamedSharding(mesh, jax.P("data"))
    def init():  # created inside jit, directly in its sharding: identical on every process, no host copy
        ks = jax.random.split(jax.random.key(0), len(SHAPES))
        return {k: jax.random.normal(kk, s) / np.sqrt(s[0]) for kk, (k, s) in zip(ks, SHAPES.items(), strict=True)}

    params = jax.jit(init, out_shardings=rep)()
    opt_state = jax.jit(OPT.init)(params)
    # Each process loads only ITS slice of the global batch, then JAX assembles one global array.
    rng = np.random.default_rng(jax.process_index())
    local = rng.integers(0, V, (B_PER_PROC, T + 1), dtype=np.int32)
    toks = jax.make_array_from_process_local_data(data, local)
    x, y = toks[:, :-1], toks[:, 1:]
    step = jax.jit(train_step)
    compiled = step.lower(params, opt_state, x, y).compile()
    hlo = compiled.as_text()
    jax.block_until_ready(compiled(params, opt_state, x, y))
    t0 = time.perf_counter()
    for _ in range(steps):
        out = compiled(params, opt_state, x, y)
    jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / steps
    n_params = int(sum(np.prod(s) for s in SHAPES.values()))
    return {"global_batch": int(x.shape[0]), "local_batch": B_PER_PROC, "n_params": n_params,
            "gradient_bytes": n_params * 4, "step_ms": dt * 1e3, "tokens_per_s": x.shape[0] * T / dt,
            "all_reduces_in_hlo": hlo.count("all-reduce("), "loss": float(out[2])}


def bandwidth():
    n = jax.device_count()
    mesh = jax.make_mesh((n,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,))
    sh = NamedSharding(mesh, jax.P("x"))
    psum = jax.jit(jax.shard_map(lambda b: jax.lax.psum(b, "x"), mesh=mesh, in_specs=jax.P("x"),
                                 out_specs=jax.P("x")))
    rows = []
    for nbytes in [2**k for k in range(16, 21 if ARGS.cpu else 27, 2)]:
        m = nbytes // 4
        a = jax.make_array_from_process_local_data(sh, np.ones((m * jax.local_device_count(),), np.float32))
        jax.block_until_ready(psum(a))
        t0 = time.perf_counter()
        for _ in range(10):
            r = psum(a)
        jax.block_until_ready(r)
        dt = (time.perf_counter() - t0) / 10
        rows.append({"bytes_per_device": nbytes, "psum_us": dt * 1e6, "psum_busbw_gbs": nbytes * 2 * (n - 1) / n / dt / 1e9})
    return {"n_devices": n, "rows": rows}


def main():
    results = {}
    section(results, "device", device_info)
    section(results, "topology", topology)
    section(results, "dp_step", dp_step)
    section(results, "bandwidth", bandwidth)
    if jax.process_index() == 0:
        save(ARGS.label or f"{machine_label()}-x{jax.process_count()}-hosts", {"multihost": results})
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
