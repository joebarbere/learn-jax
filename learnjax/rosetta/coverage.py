"""Which JAX/optax/Flax names the lessons use, and which of them are plumbing rather than math.

`used()` scans every module's lesson, solutions, exercises and helper files for `jnp.`, `jax.`, `lax.`, `jr.`,
`jsp.`, `optax.` and `nnx.` names, normalized to full names ("jnp.exp" -> "jax.numpy.exp"), with the display
label of the first module that uses each. tests/test_rosetta.py requires every used name to be covered by some
entry's `apis` or listed in NON_MATH below.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_PAT = re.compile(r"(?<![\w.])(jnp|jax|lax|jr|jsp|optax|nnx)\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")
_ALIAS = {"jnp": "jax.numpy", "lax": "jax.lax", "jr": "jax.random", "jsp": "jax.scipy", "jax": "jax",
          "optax": "optax", "nnx": "flax.nnx"}

# Not math: what each group is instead. Every name here is still taught in its lesson.
_GROUPS: dict[str, str] = {
    "namespaces and links": """jax.numpy jax.numpy.linalg jax.random jax.nn jax.lax jax.experimental jax.sharding
        jax.distributed jax.profiler optax.contrib optax.readthedocs.io jax.dev""",
    "dtypes, constants and array types": """jax.numpy.bfloat16 jax.numpy.float16 jax.numpy.float32 jax.numpy.int32 jax.numpy.uint16
        jax.numpy.int8 jax.numpy.dtype jax.numpy.finfo jax.numpy.ndarray jax.Array jax.numpy.pi jax.numpy.inf
        jax.ShapeDtypeStruct jax.typeof""",
    "making and copying arrays": """jax.numpy.array jax.numpy.asarray jax.numpy.zeros jax.numpy.ones jax.numpy.full
        jax.numpy.zeros_like jax.numpy.ones_like jax.numpy.full_like jax.numpy.copy""",
    "shape queries and test helpers": """jax.numpy.shape jax.numpy.size jax.numpy.allclose jax.numpy.array_equal
        jax.numpy.isfinite""",
    "devices, compilation and debugging": """jax.jit jax.device_put jax.devices jax.block_until_ready jax.make_jaxpr
        jax.debug.print jax.config jax.config.update jax.debug_nans jax.default_matmul_precision
        jax.default_backend jax.eval_shape jax.live_arrays jax.checkpoint jax.__version__ jax.device_count
        jax.local_device_count jax.local_devices jax.process_count jax.process_index""",
    "profiling and kernels": """jax.profiler.ProfileData jax.profiler.ProfileData.from_file
        jax.profiler.StepTraceAnnotation jax.profiler.trace jax.experimental.pallas""",
    "random keys (how JAX threads randomness, not a distribution)": """jax.random.key jax.random.split
        jax.random.fold_in""",
    "pytrees (containers of arrays)": """jax.tree.map jax.tree.leaves jax.tree.flatten jax.tree.unflatten
        jax.tree.structure jax.tree_util.keystr jax.tree_util.register_dataclass
        jax.tree_util.tree_flatten_with_path jax.tree_util.tree_map_with_path jax.tree_util.tree_flatten
        jax.tree_util.tree_unflatten jax.tree_util.tree_structure""",
    "Flax NNX module plumbing": """flax.nnx.List flax.nnx.Module flax.nnx.Optimizer flax.nnx.Param flax.nnx.Rngs
        flax.nnx.jit flax.nnx.merge flax.nnx.split flax.nnx.to_pure_dict flax.nnx.state flax.nnx.update""",
    "choosing which parameters train": "optax.multi_transform optax.set_to_zero optax.GradientTransformation",
    "sharding and multi-device setup (Module 24 draws the math)": """jax.P jax.sharding.PartitionSpec
        jax.sharding.AxisType.Auto jax.sharding.AxisType.Explicit jax.make_mesh jax.set_mesh jax.shard_map
        jax.reshard jax.lax.with_sharding_constraint jax.lax.pcast jax.lax.axis_index
        jax.debug.visualize_array_sharding jax.distributed.initialize jax.distributed.shutdown
        jax.NamedSharding jax.sharding.SingleDeviceSharding
        jax.make_array_from_process_local_data jax.new_ref jax.ref jax.lax.psum jax.lax.all_gather
        jax.lax.psum_scatter jax.lax.all_to_all jax.lax.ppermute""",
}
NON_MATH: dict[str, str] = {name: why for why, names in _GROUPS.items() for name in names.split()}


def used(root: Path = ROOT) -> dict[str, str]:
    """Full name -> display label of the first module that uses it, in syllabus order."""
    from learnjax import syllabus

    first: dict[str, str] = {}
    for _, slug, _, _ in syllabus.MODULES:  # already in syllabus order
        for path in sorted((root / "modules" / slug).glob("*.py")):
            if path.name == "test_exercises.py":
                continue
            for m in _PAT.finditer(path.read_text()):
                name = re.sub(r"\.html$", "", f"{_ALIAS[m.group(1)]}.{m.group(2)}")
                first.setdefault(name, syllabus.label(slug))
    return first
