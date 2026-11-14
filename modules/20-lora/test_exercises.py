import jax
import jax.numpy as jnp
import numpy as np
import pytest

from learnjax.llama import Config

SMOL360 = Config(vocab=49152, d_model=960, n_layers=32, n_heads=15, n_kv_heads=5, d_ff=2560)


def test_lora_init(impl):
    A, B = impl.lora_init(jax.random.key(0), 512, 384, 8)
    assert A.shape == (512, 8) and B.shape == (8, 384)
    assert A.dtype == jnp.float32 and B.dtype == jnp.float32
    assert float(jnp.abs(B).max()) == 0.0, "B must start at zero so the adapter starts as a no-op"
    assert float(A.std()) == pytest.approx(1 / np.sqrt(512), rel=0.15)
    A2, _ = impl.lora_init(jax.random.key(1), 512, 384, 8, dtype=jnp.bfloat16)
    assert A2.dtype == jnp.bfloat16 and not jnp.array_equal(A2.astype(jnp.float32), A)


def test_lora_apply(impl):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(k1, (2, 5, 16))
    W = jax.random.normal(k2, (16, 12))
    A = jax.random.normal(k3, (16, 4))
    B = jax.random.normal(k4, (4, 12))
    out = impl.lora_apply(x, W, A, B, alpha=8.0, r=4)
    np.testing.assert_allclose(out, x @ W + 2.0 * x @ A @ B, rtol=1e-4, atol=1e-4)
    # with B = 0 it is exactly the base layer
    np.testing.assert_array_equal(impl.lora_apply(x, W, A, jnp.zeros_like(B), 8.0, 4), x @ W)
    # bf16 activations stay bf16 even with f32 adapters
    xb, Wb = x.astype(jnp.bfloat16), W.astype(jnp.bfloat16)
    assert impl.lora_apply(xb, Wb, A, B, 8.0, 4).dtype == jnp.bfloat16


def test_lora_param_count(impl):
    # SmolLM2-360M, r=8 on q and v: per layer 8*(960+960) + 8*(960+320) = 25,600; × 32 layers
    assert impl.lora_param_count(SMOL360, 8, ["wq", "wv"]) == 819_200
    every = ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]
    assert impl.lora_param_count(SMOL360, 1, every) == 32 * (1920 + 1280 + 1280 + 1920 + 3520 + 3520 + 3520)
    assert impl.lora_param_count(SMOL360, 8, []) == 0


def test_merge_lora(impl):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(2), 4)
    W = jax.random.normal(k1, (16, 12))
    A = jax.random.normal(k2, (16, 4))
    B = jax.random.normal(k3, (4, 12))
    x = jax.random.normal(k4, (3, 16))
    Wm = impl.merge_lora(W, A, B, 16.0, 4)
    assert Wm.shape == W.shape and Wm.dtype == W.dtype
    np.testing.assert_allclose(x @ Wm, x @ W + 4.0 * (x @ A) @ B, rtol=1e-4, atol=1e-3)
    assert impl.merge_lora(W.astype(jnp.bfloat16), A, B, 16.0, 4).dtype == jnp.bfloat16


def test_freeze_mask(impl):
    params = {"base": {"embed": jnp.ones(3), "layers": {"wq": jnp.ones(2), "wv": jnp.ones(2)}},
              "lora": {"wq": {"A": jnp.ones(2), "B": jnp.ones(2)}}}
    m = impl.freeze_mask(params, ("lora/",))
    assert jax.tree.structure(m) == jax.tree.structure(params)
    assert m["lora"]["wq"]["A"] is True or bool(m["lora"]["wq"]["A"])
    assert not m["base"]["embed"] and not m["base"]["layers"]["wq"]
    m2 = impl.freeze_mask(params, ("layers/wv", "lora/wq/B"))
    assert m2["base"]["layers"]["wv"] and m2["lora"]["wq"]["B"]
    assert not m2["base"]["layers"]["wq"] and not m2["lora"]["wq"]["A"]
    assert not any(jax.tree.leaves(impl.freeze_mask(params, ())))


def test_adapter_bytes(impl):
    adapters = {"wq": {"A": jnp.zeros((32, 960, 8)), "B": jnp.zeros((32, 8, 960))},
                "wv": {"A": jnp.zeros((32, 960, 8), jnp.bfloat16), "B": jnp.zeros((32, 8, 320), jnp.bfloat16)}}
    expect = 4 * (32 * 960 * 8 * 2) + 2 * (32 * 960 * 8 + 32 * 8 * 320)
    assert impl.adapter_bytes(adapters) == expect
    assert impl.adapter_bytes({}) == 0
