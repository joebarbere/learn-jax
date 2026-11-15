"""Module 24 exercises: sharding arithmetic, reading compiled HLO, and hand-written collectives.

Replace each `raise NotImplementedError` with your code, then run `make test N=24`.

The first four are plain Python (no devices needed). The last three use `jax.shard_map` and are tested
on 8 simulated CPU devices in a subprocess, so they work on any machine.

A PartitionSpec (`jax.P(...)`) entry per array dimension is one of:
    None              the dimension is not split (replicated across the mesh)
    "x"               split evenly across mesh axis "x"
    ("x", "y")        split across both axes (x-major), i.e. into mesh["x"] * mesh["y"] pieces
Dimensions beyond the spec's length are not split.
"""
import jax  # noqa: F401  (jax.P, jax.shard_map, jax.lax.*)
import jax.numpy as jnp  # noqa: F401  (you'll want it)


def partition_shape(global_shape: tuple[int, ...], spec, mesh_shape: dict[str, int]) -> tuple[int, ...]:
    """The shape of ONE device's block of an array with this global shape and PartitionSpec.

    `spec` is a jax.P or any sequence of entries (see the module docstring); `mesh_shape` maps axis
    names to sizes, e.g. {"x": 4, "y": 2}. Raise ValueError if a dimension isn't divisible by the number
    of pieces it's split into, or if the spec has more entries than the array has dimensions.

    partition_shape((8, 6), jax.P("x", None), {"x": 4}) == (2, 6)
    """
    raise NotImplementedError


def bytes_per_device(param_shapes: dict[str, tuple[int, ...]], specs: dict[str, object],
                     mesh_shape: dict[str, int], dtype_bytes: int = 4, optimizer_states: int = 2) -> int:
    """Memory one device holds for the parameters AND their optimizer state.

    Each parameter is stored with its spec; Adam keeps `optimizer_states` (= 2: m and v) more arrays of
    the same shape and sharding. Ignore activations. Use partition_shape.
    """
    raise NotImplementedError


def count_collectives(hlo_text: str) -> dict[str, int]:
    """Count the collective operations in compiled HLO text (from `jax.jit(f).lower(...).compile().as_text()`).

    Count calls of: all-reduce, all-gather, reduce-scatter, all-to-all, collective-permute. An op is
    written like `%all-reduce.3 = f32[8]{0} all-reduce(%x), ...` — count the *call* (`all-reduce(`), not the
    instruction name. Asynchronous forms (`all-gather-start(`) count once; their `-done(` halves don't.
    Return a dict with all five keys (zeros included).
    """
    raise NotImplementedError


def fsdp_specs(params, n_shards: int, axis: str = "fsdp"):
    """FSDP PartitionSpecs for a parameter pytree, via `jax.tree_util.tree_map_with_path`.

    Rule: a leaf whose path contains "bias" or "norm", or that has fewer than 2 dimensions, is replicated
    (jax.P()). Otherwise shard its FIRST dimension that is divisible by n_shards across `axis` (other
    dims None, spec length = ndim); if no dimension divides evenly, replicate it.
    Leaves may be arrays or jax.ShapeDtypeStruct (anything with .shape). Return a pytree of jax.P.
    """
    raise NotImplementedError


def row_parallel_matmul(x, w, mesh, axis: str):
    """x @ w where the contraction dimension is split across `axis` (Megatron's "row-parallel" layer).

    Use jax.shard_map with in_specs=(P(None, axis), P(axis, None)) and out_specs=P(): each device
    multiplies its column-block of x by its row-block of w, giving a *partial sum* of the full result;
    combine the partial sums with the right collective. x: [n, k], w: [k, m], k divisible by mesh[axis].
    Returns the full [n, m] result (replicated).
    """
    raise NotImplementedError


def column_parallel_matmul(x, w, mesh, axis: str):
    """x @ w where w's OUTPUT columns are split across `axis` ("column-parallel").

    Use jax.shard_map with in_specs=(P(), P(None, axis)) and out_specs=P(): each device computes its own
    block of output columns; then gather the blocks into the full [n, m] result on every device.
    Hint: jax.lax.all_gather(..., axis=1, tiled=True, to="invarying") makes the result the same on all
    devices, which out_specs=P() requires.
    """
    raise NotImplementedError


def ring_allreduce(x, mesh, axis: str):
    """Sum the per-device blocks of x around a ring, using only jax.lax.ppermute (no psum!).

    x is sharded P(axis) — device i holds block i. Return an array with the same sharding where EVERY
    device's block equals the sum of all blocks (what psum would give). With n devices: n - 1 steps; in
    each, pass what you received last step to your right-hand neighbour ((i + 1) % n) and add it to your
    running total. (Real ring all-reduce is reduce-scatter + all-gather for bandwidth; this simpler
    version shows the communication pattern.)
    """
    raise NotImplementedError
