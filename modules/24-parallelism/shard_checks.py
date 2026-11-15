"""Check the shard_map exercises on 8 simulated CPU devices; prints one JSON line of results.

    python modules/24-parallelism/shard_checks.py            # your exercises.py
    LEARNJAX_SOLUTIONS=1 python modules/24-parallelism/shard_checks.py

Run in its own process because the CPU device count must be set before JAX starts its backends —
test_exercises.py calls this in a subprocess so it works inside a larger pytest run.
"""
import json
import sys
import traceback
from pathlib import Path

import jax

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 8)

import numpy as np
from jax.sharding import AxisType, NamedSharding

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learnjax.course import load_exercises


def check(fn) -> str:
    try:
        fn()
        return "ok"
    except NotImplementedError:
        return "NotImplementedError"
    except Exception:  # noqa: BLE001
        return traceback.format_exc(limit=2)[-600:]


def main() -> None:
    ex = load_exercises(Path(__file__).resolve().parent)
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    results = {}
    for n, shape in [(8, (8,)), (4, (4, 2))]:
        names = ("t",) if len(shape) == 1 else ("t", "other")
        mesh = jax.make_mesh(shape, names, axis_types=(AxisType.Explicit,) * len(shape))
        x = jax.random.normal(k1, (6, 16))
        w = jax.random.normal(k2, (16, 24))
        want = np.asarray(x @ w)

        def row(mesh=mesh, x=x, w=w, want=want):
            xs = jax.device_put(x, NamedSharding(mesh, jax.P(None, "t")))
            ws = jax.device_put(w, NamedSharding(mesh, jax.P("t", None)))
            got = np.asarray(ex.row_parallel_matmul(xs, ws, mesh, "t"))
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)

        def col(mesh=mesh, x=x, w=w, want=want):
            xs = jax.device_put(x, NamedSharding(mesh, jax.P()))
            ws = jax.device_put(w, NamedSharding(mesh, jax.P(None, "t")))
            got = np.asarray(ex.column_parallel_matmul(xs, ws, mesh, "t"))
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)

        def ring(mesh=mesh, n=n):
            blocks = jax.random.normal(k3, (n * 3,))
            xs = jax.device_put(blocks, NamedSharding(mesh, jax.P("t")))
            got = np.asarray(ex.ring_allreduce(xs, mesh, "t"))
            total = np.asarray(blocks).reshape(n, 3).sum(0)
            np.testing.assert_allclose(got, np.tile(total, n), rtol=1e-5, atol=1e-5)
            hlo = jax.jit(lambda a: ex.ring_allreduce(a, mesh, "t")).lower(xs).compile().as_text()
            assert "all-reduce" not in hlo, "use ppermute, not psum"

        for name, fn in [("row_parallel_matmul", row), ("column_parallel_matmul", col), ("ring_allreduce", ring)]:
            r = check(fn)
            if results.get(name, "ok") == "ok":
                results[name] = r if r == "ok" else f"[mesh {shape}] {r}"
    print(json.dumps(results))


if __name__ == "__main__":
    main()
