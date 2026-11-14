import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def small_cfg(impl, **kw):
    base = {"vocab": 17, "seq_len": 8, "d_model": 16, "n_layers": 2, "n_heads": 2, "tie_embeddings": True}
    base.update(kw)
    return impl.GPTConfig(**base)


def n_leaves(params):
    return sum(x.size for x in jax.tree.leaves(params))


def test_count_params(impl):
    for kw in ({}, {"tie_embeddings": False}, {"n_layers": 3, "d_model": 32, "n_heads": 4}):
        cfg = small_cfg(impl, **kw)
        assert impl.count_params(cfg) == n_leaves(impl.init_gpt(jax.random.key(0), cfg))
    # GPT-2 small (124M) is this formula with vocab 50257, context 1024, d 768, 12 layers, tied
    gpt2 = impl.GPTConfig(vocab=50257, seq_len=1024, d_model=768, n_layers=12, n_heads=12)
    assert impl.count_params(gpt2) == 124_439_808


def test_init_gpt(impl):
    cfg = small_cfg(impl, d_model=64, n_heads=4, n_layers=4)
    p = impl.init_gpt(jax.random.key(0), cfg)
    assert set(p) == {"wte", "wpe", "blocks", "lnf_g", "lnf_b"}
    assert p["wte"].shape == (17, 64) and p["wpe"].shape == (8, 64) and len(p["blocks"]) == 4
    b = p["blocks"][0]
    assert b["w_qkv"].shape == (64, 192) and b["w_fc"].shape == (64, 256) and b["w_proj"].shape == (256, 64)
    np.testing.assert_array_equal(b["ln1_g"], 1.0)
    np.testing.assert_array_equal(b["b_qkv"], 0.0)
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(p))
    assert float(jnp.std(b["w_fc"])) == pytest.approx(0.02, rel=0.1)
    assert float(jnp.std(b["w_proj"])) == pytest.approx(0.02 / math.sqrt(8), rel=0.1)  # scaled init
    assert not jnp.allclose(p["blocks"][0]["w_qkv"], p["blocks"][1]["w_qkv"])  # separate keys
    untied = impl.init_gpt(jax.random.key(0), small_cfg(impl, tie_embeddings=False))
    assert untied["w_head"].shape == (16, 17)


def test_block_apply(impl):
    cfg = small_cfg(impl)
    p = impl.init_gpt(jax.random.key(1), cfg)["blocks"][0]
    x = jax.random.normal(jax.random.key(2), (2, 8, 16))
    y = impl.block_apply(p, x, cfg.n_heads)
    assert y.shape == x.shape
    # causal: changing the last position must not change any earlier output
    y2 = impl.block_apply(p, x.at[:, -1].add(5.0), cfg.n_heads)
    np.testing.assert_allclose(y[:, :-1], y2[:, :-1], atol=1e-5)
    # residual stream: zero output projections -> identity
    zero = dict(p, w_o=jnp.zeros_like(p["w_o"]), w_proj=jnp.zeros_like(p["w_proj"]))
    np.testing.assert_allclose(impl.block_apply(zero, x, cfg.n_heads), x, atol=1e-6)


def test_gpt_apply(impl):
    cfg = small_cfg(impl)
    p = impl.init_gpt(jax.random.key(3), cfg)
    tok = jax.random.randint(jax.random.key(4), (3, 8), 0, 17)
    logits = impl.gpt_apply(p, tok, cfg)
    assert logits.shape == (3, 8, 17) and logits.dtype == jnp.float32
    assert impl.gpt_apply(p, tok[:, :5], cfg).shape == (3, 5, 17)  # shorter than seq_len is fine
    later = impl.gpt_apply(p, tok.at[:, -1].set((tok[:, -1] + 1) % 17), cfg)
    np.testing.assert_allclose(logits[:, :-1], later[:, :-1], atol=1e-5)  # causal end to end
    # mixed precision: bf16 params still give float32 logits, close to the f32 ones
    bf = jax.tree.map(lambda a: a.astype(jnp.bfloat16), p)
    lb = impl.gpt_apply(bf, tok, cfg)
    assert lb.dtype == jnp.float32
    np.testing.assert_allclose(lb, logits, atol=0.05)
    untied_cfg = small_cfg(impl, tie_embeddings=False)
    assert impl.gpt_apply(impl.init_gpt(jax.random.key(3), untied_cfg), tok, untied_cfg).shape == (3, 8, 17)


def test_loss_fn(impl):
    cfg = small_cfg(impl)
    p = impl.init_gpt(jax.random.key(5), cfg)
    tok = jax.random.randint(jax.random.key(6), (4, 9), 0, 17)
    loss = impl.loss_fn(p, tok, cfg)
    assert loss.shape == () and loss.dtype == jnp.float32
    logp = jax.nn.log_softmax(impl.gpt_apply(p, tok[:, :-1], cfg))
    expected = -jnp.mean(jnp.take_along_axis(logp, tok[:, 1:, None], -1))
    assert float(loss) == pytest.approx(float(expected), rel=1e-5)
    assert float(loss) == pytest.approx(math.log(17), abs=0.15)  # untrained ≈ uniform guess


def test_get_batch(impl):
    data = jnp.arange(1000, dtype=jnp.int32)
    b = impl.get_batch(jax.random.key(0), data, 16, 32)
    assert b.shape == (16, 33) and b.dtype == jnp.int32
    np.testing.assert_array_equal(jnp.diff(b, axis=1), 1)  # each row is consecutive tokens
    assert int(b.min()) >= 0 and int(b.max()) <= 999
    np.testing.assert_array_equal(b, impl.get_batch(jax.random.key(0), data, 16, 32))
    edge = impl.get_batch(jax.random.key(1), jnp.arange(9, dtype=jnp.int32), 4, 8)  # only one fits
    np.testing.assert_array_equal(edge, jnp.tile(jnp.arange(9), (4, 1)))
    jitted = jax.jit(impl.get_batch, static_argnums=(2, 3))  # must work under jit
    assert jitted(jax.random.key(2), data, 4, 8).shape == (4, 9)


def test_sample(impl):
    cfg = small_cfg(impl)
    p = impl.init_gpt(jax.random.key(7), cfg)
    out = impl.sample(jax.random.key(0), p, [1, 2, 3], 8, cfg, temperature=0)
    assert out.shape == (11,) and out.dtype == jnp.int32
    np.testing.assert_array_equal(out[:3], [1, 2, 3])
    # greedy must agree with a plain recompute on the (cropped) context — even past seq_len
    forward = jax.jit(impl.gpt_apply, static_argnums=2)
    toks = [1, 2, 3]
    for _ in range(8):
        logits = forward(p, jnp.array([toks[-cfg.seq_len:]]), cfg)[0, -1]
        toks.append(int(jnp.argmax(logits)))
    np.testing.assert_array_equal(out, toks)
    a = impl.sample(jax.random.key(1), p, [4], 20, cfg, temperature=1.0, top_k=5)
    assert a.shape == (21,) and int(a.min()) >= 0 and int(a.max()) < 17
    np.testing.assert_array_equal(a, impl.sample(jax.random.key(1), p, [4], 20, cfg, temperature=1.0, top_k=5))
    np.testing.assert_array_equal(impl.sample(jax.random.key(9), p, [4], 8, cfg, top_k=1),
                                  impl.sample(jax.random.key(0), p, [4], 8, cfg, temperature=0))
