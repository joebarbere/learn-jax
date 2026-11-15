import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest

HERE = Path(__file__).resolve().parent

HLO = """
HloModule jit_step
%fused (p: f32[8]) -> f32[8] { ROOT %x = f32[8]{0} add(%p, %p) }
ENTRY %main {
  %all-gather-start.1 = (f32[64]{0}, f32[512]{0}) all-gather-start(%p0), channel_id=1, dimensions={0}
  %all-gather-done.1 = f32[512]{0} all-gather-done(%all-gather-start.1)
  %all-gather.2 = f32[8,64]{1,0} all-gather(%p1), channel_id=2, dimensions={0}
  %all-reduce = f32[2]{0} all-reduce(%w), channel_id=3, to_apply=%add
  %reduce-scatter.7 = f32[8]{0} reduce-scatter(%g), channel_id=4, dimensions={0}
  %collective-permute = f32[4]{0} collective-permute(%b), source_target_pairs={{0,1},{1,0}}
  %all-reduce-done = f32[2]{0} all-reduce-done(%ar)
}
"""


def test_partition_shape(impl):
    assert impl.partition_shape((8, 6), jax.P("x", None), {"x": 4}) == (2, 6)
    assert impl.partition_shape((8, 6), jax.P(None, "y"), {"x": 4, "y": 2}) == (8, 3)
    assert impl.partition_shape((8, 6), jax.P(("x", "y")), {"x": 4, "y": 2}) == (1, 6)   # tuple entry
    assert impl.partition_shape((8, 6, 5), jax.P("x"), {"x": 4}) == (2, 6, 5)             # trailing dims
    assert impl.partition_shape((8, 6), jax.P(), {"x": 4}) == (8, 6)
    with pytest.raises(ValueError):
        impl.partition_shape((6, 6), jax.P("x"), {"x": 4})
    with pytest.raises(ValueError):
        impl.partition_shape((8,), jax.P("x", None), {"x": 4})


def test_bytes_per_device(impl):
    shapes = {"w1": (256, 1024), "w2": (1024, 256)}
    mesh = {"d": 8}
    dp = impl.bytes_per_device(shapes, {"w1": jax.P(), "w2": jax.P()}, mesh)
    fsdp = impl.bytes_per_device(shapes, {"w1": jax.P("d", None), "w2": jax.P("d", None)}, mesh)
    assert dp == 2 * 256 * 1024 * 4 * 3
    assert fsdp * 8 == dp
    assert impl.bytes_per_device(shapes, {"w1": jax.P(), "w2": jax.P()}, mesh, dtype_bytes=2, optimizer_states=0) \
        == 2 * 256 * 1024 * 2


def test_count_collectives(impl):
    c = impl.count_collectives(HLO)
    assert c == {"all-reduce": 1, "all-gather": 2, "reduce-scatter": 1, "all-to-all": 0, "collective-permute": 1}
    assert impl.count_collectives("") == dict.fromkeys(c, 0)


def test_fsdp_specs(impl):
    params = {
        "layer": {"w": jax.ShapeDtypeStruct((6, 16), "float32"), "bias": jax.ShapeDtypeStruct((16,), "float32")},
        "norm": {"scale": jax.ShapeDtypeStruct((16, 16), "float32")},
        "head": jax.ShapeDtypeStruct((5, 7), "float32"),
        "emb": jax.ShapeDtypeStruct((32, 16), "float32"),
    }
    specs = impl.fsdp_specs(params, n_shards=8, axis="fsdp")
    assert specs["layer"]["w"] == jax.P(None, "fsdp")        # first divisible dim is dim 1
    assert specs["layer"]["bias"] == jax.P()
    assert specs["norm"]["scale"] == jax.P()
    assert specs["head"] == jax.P()                          # nothing divides by 8
    assert specs["emb"] == jax.P("fsdp", None)


@pytest.fixture(scope="module")
def shard_results():
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, str(HERE / "shard_checks.py")], capture_output=True, text=True,
                       env=env, timeout=300, check=False)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    assert lines, f"shard_checks.py failed:\n{r.stderr[-2000:]}"
    return json.loads(lines[-1])


@pytest.mark.parametrize("name", ["row_parallel_matmul", "column_parallel_matmul", "ring_allreduce"])
def test_shard_map_exercises(shard_results, name):
    assert shard_results[name] == "ok", shard_results[name]
