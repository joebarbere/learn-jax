import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(43)


def _rand(i, shape):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32)


def test_patchify(impl):
    x = _rand(1, (2, 12, 8, 3))
    out = np.asarray(impl.patchify(x, 4))
    assert out.shape == (2, 6, 48)
    xn = np.asarray(x)
    n = 0
    for i in range(0, 12, 4):
        for j in range(0, 8, 4):
            np.testing.assert_array_equal(out[1, n], xn[1, i:i + 4, j:j + 4].reshape(-1))
            n += 1


def test_patchify_is_a_conv(impl):
    x, w = _rand(2, (1, 8, 8, 2)), _rand(3, (4, 4, 2, 5))                # HWIO kernel
    conv = jax.lax.conv_general_dilated(x, w, (4, 4), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    np.testing.assert_allclose(np.asarray(impl.patchify(x, 4) @ w.reshape(-1, 5)),
                               np.asarray(conv.reshape(1, 4, 5)), rtol=1e-5, atol=1e-5)


def test_resize_pos_embed(impl):
    pos = _rand(4, (16, 6))
    same = impl.resize_pos_embed(pos, 4)
    np.testing.assert_allclose(np.asarray(same), np.asarray(pos), atol=1e-6)
    big = np.asarray(impl.resize_pos_embed(pos, 8))
    assert big.shape == (64, 6)
    ref = jax.image.resize(pos.reshape(4, 4, 6), (8, 8, 6), "bilinear").reshape(64, 6)
    np.testing.assert_allclose(big, np.asarray(ref), atol=1e-6)


def test_siglip_logits(impl):
    img, txt = _rand(5, (3, 7)), _rand(6, (4, 7))
    out = np.asarray(impl.siglip_logits(img, txt, jnp.log(10.0), -2.0))
    i, t = np.asarray(img), np.asarray(txt)
    i, t = i / np.linalg.norm(i, axis=1, keepdims=True), t / np.linalg.norm(t, axis=1, keepdims=True)
    np.testing.assert_allclose(out, 10 * i @ t.T - 2, rtol=1e-5, atol=1e-5)


def test_siglip_loss(impl):
    logits = _rand(7, (4, 4)) * 3
    ln = np.asarray(logits, np.float64)
    expect = -sum(np.log(1 / (1 + np.exp(-(1 if a == b else -1) * ln[a, b]))) for a in range(4) for b in range(4)) / 4
    assert float(impl.siglip_loss(logits)) == pytest.approx(expect, rel=1e-5)
    good = 20 * jnp.eye(5) - 10                                          # confident and right
    assert float(impl.siglip_loss(good)) < 1e-3


def test_pixel_shuffle(impl):
    g, d = 4, 3
    x = jnp.arange(g * g * d, dtype=jnp.float32).reshape(1, g * g, d)
    out = np.asarray(impl.pixel_shuffle(x, 2))
    assert out.shape == (1, 4, 12)
    grid = np.asarray(x).reshape(g, g, d)
    np.testing.assert_array_equal(out[0, 0], grid[0:2, 0:2].reshape(-1))
    np.testing.assert_array_equal(out[0, 1], grid[0:2, 2:4].reshape(-1))
    np.testing.assert_array_equal(out[0, 2], grid[2:4, 0:2].reshape(-1))
    assert impl.pixel_shuffle(_rand(8, (2, 196, 5)), 7).shape == (2, 4, 245)
