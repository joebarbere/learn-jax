"""What JAX sees on this machine, and how fast the card multiplies matrices."""
from __future__ import annotations

import os
import shutil
import subprocess

import jax
import jax.numpy as jnp

from learnjax.timing import timeit

ROCM_ENV = ("HSA_OVERRIDE_GFX_VERSION", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_PLATFORMS")


def gpu_info() -> dict:
    dev = jax.devices()[0]
    info = {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "device_kind": getattr(dev, "device_kind", "?"),
        "env": {k: os.environ.get(k, "") for k in ROCM_ENV},
    }
    try:
        stats = dev.memory_stats() or {}
        info["memory_gb"] = round(stats.get("bytes_limit", 0) / 2**30, 1)
    except Exception:  # noqa: BLE001 - CPU devices have no memory stats
        info["memory_gb"] = None
    if shutil.which("rocminfo"):
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, check=False).stdout
        info["rocm_agents"] = sorted({ln.split(":", 1)[1].strip() for ln in out.splitlines()
                                     if "Marketing Name" in ln})
        info["gfx_reported"] = sorted({ln.split(":", 1)[1].strip() for ln in out.splitlines()
                                      if ln.strip().startswith("Name:") and "gfx" in ln})
    return info


def matmul_tflops(n: int = 4096, dtype=jnp.bfloat16, iters: int = 10) -> float:
    """Achieved TFLOP/s for an n×n @ n×n matmul (2·n³ floating-point ops)."""
    ka, kb = jax.random.split(jax.random.key(0))
    a = jax.random.normal(ka, (n, n), dtype)
    b = jax.random.normal(kb, (n, n), dtype)
    t = timeit(jax.jit(jnp.matmul), a, b, iters=iters)["median"]
    return 2 * n**3 / t / 1e12
