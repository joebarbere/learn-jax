"""Shared helpers for Module 25's benchmarks: identify the machine, measure its matmul peak, save results.

The bench_*.py scripts run unchanged on the RX 7600 XT (ROCm, in the course container) and on AWS
(CUDA on g6/g6e, Neuron on trn1) via infra/aws/run_remote.sh. Every result is one JSON file per machine
under runs/25-aws/, merged section by section so a partial run still records what worked.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
import traceback
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))  # the course root, for `learnjax`
OUT = ROOT / "runs" / "25-aws"

# us-east-1 on-demand USD/hour, read from instances.vantage.sh on 2026-11-14. Spot is usually much lower
# but varies; pass --usd-per-hour to override. "local" is electricity only (~165 W × $0.15/kWh), not the card.
PRICES = {
    "g6.xlarge": 0.805,
    "g6e.xlarge": 1.861,
    "g6.12xlarge": 4.602,
    "trn1.2xlarge": 1.344,
    "local-rx7600xt": 0.025,
}


def instance_type(timeout: float = 0.5) -> str | None:
    """EC2 instance type from IMDSv2, or None when not on EC2 (fails fast)."""
    try:
        req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(req, timeout=timeout).read().decode()
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-type",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=timeout).read().decode()
    except Exception:  # noqa: BLE001 - not on EC2
        return None


def machine_label() -> str:
    it = instance_type()
    if it:
        return it
    return os.environ.get("LEARNJAX_MACHINE", "local-rx7600xt")


def gpu_busy_percent() -> int | None:
    """AMD GPUs expose current utilization in sysfs; a busy GPU before a benchmark means contended numbers."""
    for p in Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"):
        try:
            return int(p.read_text().strip())
        except (OSError, ValueError):
            pass
    return None


def device_info() -> dict:
    import jax
    d = jax.devices()
    return {"jax": jax.__version__, "backend": jax.default_backend(), "n_devices": len(d),
            "device_kind": getattr(d[0], "device_kind", str(d[0])), "host": socket.gethostname(),
            "python": platform.python_version(), "process_count": jax.process_count(),
            "gpu_busy_percent_at_start": gpu_busy_percent()}


def timed(fn, *args, warmup: int = 2, iters: int = 10) -> float:
    """Median seconds per call, blocking on the result (JAX dispatch is asynchronous)."""
    import jax
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def matmul_peak(n: int = 4096, dtype: str = "bfloat16", device=None) -> float:
    """Achieved TFLOP/s for an n×n @ n×n matmul on one device: the denominator for MFU here.

    Measured, not the datasheet number: vendor peaks assume ideal conditions (and some quote sparsity),
    so utilization against a measured matmul is the fairer cross-device comparison.
    """
    import jax
    import jax.numpy as jnp
    dev = device or jax.devices()[0]
    a = jax.device_put(jax.random.normal(jax.random.key(0), (n, n), dtype), dev)
    t = timed(jax.jit(jnp.matmul), a, a, warmup=2, iters=8)
    return 2 * n**3 / t / 1e12


def section(results: dict, name: str, fn, *args, **kwargs):
    """Run one benchmark section; record its result or the error, and keep going."""
    t0 = time.perf_counter()
    try:
        results[name] = fn(*args, **kwargs)
        status = "ok"
    except Exception as ex:  # noqa: BLE001 - every section is independent; record and continue
        results[name] = {"error": f"{type(ex).__name__}: {ex}"[:2000],
                         "traceback": traceback.format_exc(limit=4)[-3000:]}
        status = "FAILED"
    print(f"[{name}] {status} in {time.perf_counter() - t0:.1f}s", flush=True)
    return results[name]


def save(label: str, results: dict) -> Path:
    """Merge `results` into runs/25-aws/<label>.json (other sections already there are kept)."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(results)
    old["label"] = label
    old["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(old, indent=2, default=float))
    print(f"wrote {path}", flush=True)
    return path
