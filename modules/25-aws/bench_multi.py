"""Multi-device benchmark: Module 24's strategies on real GPUs, plus collective bandwidth and overlap.

    python modules/25-aws/bench_multi.py                 # on g6.12xlarge (4 × L4), via run_remote.sh
    containers/run.sh python modules/25-aws/bench_multi.py --cpu-sim 8 --small   # local correctness check

Module 24 taught DP / FSDP / TP / FSDP+TP on 8 simulated CPU devices: correct results, memory per device and
which collectives XLA inserts — but not how long anything takes. Here the same config-driven strategies run
on real devices at a size where communication matters, recording step time, per-device memory, the
collectives in the compiled HLO (does reduce-scatter appear on GPU? on CPU XLA chose all-reduce), a profiler
trace with how much collective time overlaps compute, and psum / all_gather bandwidth vs message size.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import time


def _early_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-sim", type=int, default=0, help="simulate N CPU devices (local correctness check)")
    ap.add_argument("--small", action="store_true", help="tiny model (for --cpu-sim)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--steps", type=int, default=10)
    return ap.parse_args()


ARGS = _early_args()
if ARGS.cpu_sim:  # must happen before JAX initializes a backend
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    jax.config.update("jax_num_cpu_devices", ARGS.cpu_sim)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from bench_common import ROOT, device_info, machine_label, save, section
from jax.sharding import NamedSharding

sys.path.insert(0, str(ROOT / "modules" / "24-parallelism"))
from parallel import count_collectives

# Big enough that gradients (~170M params, 0.7 GB in f32) make communication visible on PCIe GPUs.
BIG = {"V": 32768, "D": 2048, "F": 8192, "B": 32, "T": 256}
SMALL = {"V": 512, "D": 128, "F": 512, "B": 32, "T": 16}
C = SMALL if ARGS.small else BIG
SHAPES = {"emb": (C["V"], C["D"]), "w1": (C["D"], C["F"]), "w2": (C["F"], C["D"]), "head": (C["D"], C["V"])}
OPT = optax.adam(1e-3)


def init_params(key):
    ks = jax.random.split(key, len(SHAPES))
    return {k: jax.random.normal(kk, s, jnp.float32) / np.sqrt(s[0])
            for kk, (k, s) in zip(ks, SHAPES.items(), strict=True)}


def loss_fn(p, x, y):  # the Module 24 model: embedding → one residual MLP block → head
    pb = jax.tree.map(lambda a: a.astype(jnp.bfloat16), p)
    h = pb["emb"][x]
    h = h + jax.nn.gelu(h @ pb["w1"]) @ pb["w2"]
    logits = (h @ pb["head"]).astype(jnp.float32)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def train_step(p, s, x, y):
    loss, g = jax.value_and_grad(loss_fn)(p, x, y)
    upd, s = OPT.update(g, s, p)
    return optax.apply_updates(p, upd), s, loss


def strategies(n: int) -> dict:
    """Module 24's config-driven strategies, sized to n devices (FSDP+TP uses an (n/2) × 2 mesh)."""
    P = jax.P
    out = {
        "DP": ((n,), ("data",), dict.fromkeys(SHAPES, P()), P("data")),
        "FSDP": ((n,), ("data",), dict.fromkeys(SHAPES, P("data", None)), P("data")),
        "TP": ((n,), ("model",), {"emb": P(None, "model"), "w1": P(None, "model"), "w2": P("model", None),
                                  "head": P(None, "model")}, P()),
    }
    if n >= 4 and n % 2 == 0:
        out["FSDP+TP"] = ((n // 2, 2), ("data", "model"),
                          {"emb": P("data", "model"), "w1": P("data", "model"), "w2": P("model", "data"),
                           "head": P("data", "model")}, P("data"))
    return out


def batch(key):
    toks = jax.random.randint(key, (C["B"], C["T"] + 1), 0, C["V"])
    return toks[:, :-1], toks[:, 1:]


def run_strategies(steps: int):
    devs = jax.devices()
    params, (x, y) = init_params(jax.random.key(0)), batch(jax.random.key(1))
    ref_p = jax.device_put(params, devs[0])
    ref, _, _ = jax.jit(train_step)(ref_p, OPT.init(ref_p), *jax.device_put((x, y), devs[0]))
    ref = jax.tree.map(np.asarray, ref)
    rows = []
    for name, (shape, axes, specs, bspec) in strategies(len(devs)).items():
        mesh = jax.make_mesh(shape, axes, devices=devs, axis_types=(jax.sharding.AxisType.Auto,) * len(shape))
        p = {k: jax.device_put(v, NamedSharding(mesh, specs[k])) for k, v in params.items()}
        xs, ys = (jax.device_put(a, NamedSharding(mesh, bspec)) for a in (x, y))
        s = jax.jit(OPT.init)(p)
        step = jax.jit(train_step)
        t0 = time.perf_counter()
        compiled = step.lower(p, s, xs, ys).compile()
        compile_s = time.perf_counter() - t0
        p1, s1, _loss = compiled(p, s, xs, ys)
        err = max(float(np.abs(np.asarray(p1[k]) - ref[k]).max()) for k in p1)
        jax.block_until_ready(compiled(p, s, xs, ys))
        t0 = time.perf_counter()
        for _ in range(steps):
            out = compiled(p, s, xs, ys)
        jax.block_until_ready(out)
        step_ms = (time.perf_counter() - t0) / steps * 1e3
        per_dev = sum(leaf.addressable_shards[0].data.nbytes for leaf in jax.tree.leaves((p1, s1)))
        rows.append({"strategy": name, "mesh": "x".join(map(str, shape)), "step_ms": step_ms,
                     "tokens_per_s": C["B"] * C["T"] / (step_ms / 1e3), "per_device_bytes": per_dev,
                     "max_err_vs_1_device": err, "compile_s": compile_s,
                     **count_collectives(compiled.as_text())})
        print(f"  {name:8s} {step_ms:8.2f} ms/step  {per_dev / 2**20:8.1f} MiB/device  err {err:.1e}", flush=True)
    return {"model": C, "n_params": int(sum(np.prod(s) for s in SHAPES.values())), "rows": rows}


def bandwidth():
    """psum and all_gather over all devices vs message size (per-device bytes).

    algbw = bytes / time; busbw = algbw × 2(n−1)/n for all-reduce (the ring algorithm's traffic per link),
    × (n−1)/n for all-gather — the convention NCCL's own benchmarks use, so numbers compare across n.
    """
    devs = jax.devices()
    n = len(devs)
    mesh = jax.make_mesh((n,), ("x",), devices=devs, axis_types=(jax.sharding.AxisType.Explicit,))
    sizes = [2**k for k in range(16, 27 if not ARGS.small else 21, 2)]  # 64 KiB … 64 MiB per device
    out = []
    for nbytes in sizes:
        m = nbytes // 4
        a = jax.device_put(jnp.ones((n * m,), jnp.float32), NamedSharding(mesh, jax.P("x")))
        psum = jax.jit(jax.shard_map(lambda b: jax.lax.psum(b, "x"), mesh=mesh, in_specs=jax.P("x"),
                                     out_specs=jax.P("x")))
        gather = jax.jit(jax.shard_map(lambda b: jax.lax.all_gather(b, "x", tiled=True), mesh=mesh,
                                       in_specs=jax.P("x"), out_specs=jax.P("x")))
        row = {"bytes_per_device": nbytes}
        for name, f, factor in (("psum", psum, 2 * (n - 1) / n), ("all_gather", gather, (n - 1) / n)):
            jax.block_until_ready(f(a))
            t0 = time.perf_counter()
            for _ in range(10):
                r = f(a)
            jax.block_until_ready(r)
            dt = (time.perf_counter() - t0) / 10
            row[f"{name}_us"] = dt * 1e6
            row[f"{name}_busbw_gbs"] = nbytes * factor / dt / 1e9
        out.append(row)
        print(f"  {nbytes / 2**20:7.2f} MiB  psum {row['psum_busbw_gbs']:6.2f} GB/s  "
              f"all_gather {row['all_gather_busbw_gbs']:6.2f} GB/s", flush=True)
    return {"n_devices": n, "rows": out}


def overlap(steps: int = 3):
    """Profile FSDP steps; how much collective time runs concurrently with compute kernels?"""
    devs = jax.devices()
    shape, axes, specs, bspec = strategies(len(devs))["FSDP"]
    mesh = jax.make_mesh(shape, axes, devices=devs, axis_types=(jax.sharding.AxisType.Auto,))
    p = {k: jax.device_put(v, NamedSharding(mesh, specs[k])) for k, v in init_params(jax.random.key(0)).items()}
    x, y = (jax.device_put(a, NamedSharding(mesh, bspec)) for a in batch(jax.random.key(1)))
    s = jax.jit(OPT.init)(p)
    step = jax.jit(train_step)
    jax.block_until_ready(step(p, s, x, y))
    tdir = tempfile.mkdtemp(prefix="learnjax-trace-")
    with jax.profiler.trace(tdir):
        for _ in range(steps):
            out = step(p, s, x, y)
        jax.block_until_ready(out)
    path = glob.glob(f"{tdir}/**/*.xplane.pb", recursive=True)[0]
    pd = jax.profiler.ProfileData.from_file(path)
    coll_words = ("all-reduce", "all_reduce", "all-gather", "all_gather", "reduce-scatter", "reduce_scatter",
                  "nccl", "rccl", "collective")
    coll, comp = [], []
    for plane in pd.planes:
        if not plane.name.startswith("/device:"):
            continue
        for line in plane.lines:
            for e in line.events:
                iv = (e.start_ns, e.start_ns + e.duration_ns)
                (coll if any(w in e.name.lower() for w in coll_words) else comp).append(iv)

    def union(ivs):
        total, end = 0, None
        for a, b in sorted(ivs):
            if end is None or a > end:
                total += b - a
                end = b
            elif b > end:
                total += b - end
                end = b
        return total

    overlapped = union(coll) + union(comp) - union(coll + comp)  # |A∩B| = |A| + |B| − |A∪B|
    c = union(coll)
    return {"trace_dir": tdir, "steps": steps, "collective_ms": c / 1e6, "compute_ms": union(comp) / 1e6,
            "collective_overlapped_fraction": overlapped / c if c else None,
            "note": "device planes only (none on CPU); collective kernels matched by name"}


def main():
    label = ARGS.label or (f"cpu-sim-{ARGS.cpu_sim}" if ARGS.cpu_sim else machine_label())
    if len(jax.devices()) < 2:
        print("bench_multi needs ≥ 2 devices (use --cpu-sim 8 locally)", flush=True)
        save(label, {"multi": {"error": f"only {len(jax.devices())} device(s)"}})
        return
    results = {}
    section(results, "device", device_info)
    section(results, "strategies", run_strategies, ARGS.steps)
    section(results, "bandwidth", bandwidth)
    section(results, "overlap", overlap)
    save(label, {"multi": results})


if __name__ == "__main__":
    main()
