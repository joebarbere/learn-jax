"""Module 24 reference solutions. Try the exercises first — `make test N=24` grades yours."""
import math
import re
from functools import partial

import jax

_COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute")


def _pieces(entry, mesh_shape) -> int:
    if entry is None:
        return 1
    names = entry if isinstance(entry, tuple | list) else (entry,)
    return math.prod(mesh_shape[n] for n in names)


def partition_shape(global_shape, spec, mesh_shape):
    spec = tuple(spec)
    if len(spec) > len(global_shape):
        raise ValueError(f"spec {spec} has more entries than the array has dimensions {global_shape}")
    out = []
    for i, dim in enumerate(global_shape):
        n = _pieces(spec[i], mesh_shape) if i < len(spec) else 1
        if dim % n:
            raise ValueError(f"dimension {i} of size {dim} is not divisible into {n} pieces")
        out.append(dim // n)
    return tuple(out)


def bytes_per_device(param_shapes, specs, mesh_shape, dtype_bytes=4, optimizer_states=2):
    elems = sum(math.prod(partition_shape(s, specs[k], mesh_shape)) for k, s in param_shapes.items())
    return elems * dtype_bytes * (1 + optimizer_states)


def count_collectives(hlo_text):
    counts = dict.fromkeys(_COLLECTIVES, 0)
    for m in re.finditer(r"\b(" + "|".join(_COLLECTIVES) + r")(?:-start)?\(", hlo_text):
        counts[m.group(1)] += 1
    return counts


def fsdp_specs(params, n_shards, axis="fsdp"):
    def spec(path, leaf):
        name = jax.tree_util.keystr(path).lower()
        shape = leaf.shape
        if "bias" in name or "norm" in name or len(shape) < 2:
            return jax.P()
        for i, d in enumerate(shape):
            if d % n_shards == 0:
                return jax.P(*([None] * i + [axis] + [None] * (len(shape) - i - 1)))
        return jax.P()

    return jax.tree_util.tree_map_with_path(spec, params)


def row_parallel_matmul(x, w, mesh, axis):
    @partial(jax.shard_map, mesh=mesh, in_specs=(jax.P(None, axis), jax.P(axis, None)), out_specs=jax.P())
    def f(xb, wb):
        return jax.lax.psum(xb @ wb, axis)

    return f(x, w)


def column_parallel_matmul(x, w, mesh, axis):
    @partial(jax.shard_map, mesh=mesh, in_specs=(jax.P(), jax.P(None, axis)), out_specs=jax.P())
    def f(xb, wb):
        return jax.lax.all_gather(xb @ wb, axis, axis=1, tiled=True, to="invarying")

    return f(x, w)


def ring_allreduce(x, mesh, axis):
    n = mesh.shape[axis]
    perm = [(i, (i + 1) % n) for i in range(n)]

    @partial(jax.shard_map, mesh=mesh, in_specs=jax.P(axis), out_specs=jax.P(axis))
    def f(block):
        total, passing = block, block
        for _ in range(n - 1):
            passing = jax.lax.ppermute(passing, axis, perm)
            total = total + passing
        return total

    return f(x)

