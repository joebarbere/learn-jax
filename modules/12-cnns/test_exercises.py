import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _naive_conv2d_valid(img, ker):
    """Reference: loops over every output position (NumPy, one image, HWIO kernel)."""
    h, w, _ = img.shape
    kh, kw, _, co = ker.shape
    out = np.zeros((h - kh + 1, w - kw + 1, co), np.float32)
    for y in range(h - kh + 1):
        for x in range(w - kw + 1):
            out[y, x] = np.tensordot(img[y:y + kh, x:x + kw], ker, axes=3)
    return out


def test_conv_output_size(impl):
    assert impl.conv_output_size(28, 3) == 26
    assert impl.conv_output_size(28, 3, padding=1) == 28
    assert impl.conv_output_size(28, 2, stride=2) == 14
    assert impl.conv_output_size(7, 3, stride=2, padding=1) == 4
    with pytest.raises(ValueError):
        impl.conv_output_size(2, 5)


def test_conv1d(impl):
    rng = np.random.default_rng(0)
    x, w = rng.normal(size=50).astype(np.float32), rng.normal(size=5).astype(np.float32)
    np.testing.assert_allclose(impl.conv1d(jnp.array(x), jnp.array(w)), np.correlate(x, w, "valid"),
                               rtol=1e-4, atol=1e-5)
    # kernel as long as the signal: a single dot product
    np.testing.assert_allclose(impl.conv1d(jnp.arange(4.0), jnp.ones(4)), [6.0])


def test_conv2d(impl):
    rng = np.random.default_rng(1)
    img = rng.normal(size=(2, 9, 8, 3)).astype(np.float32)
    ker = rng.normal(size=(3, 3, 3, 4)).astype(np.float32)
    out = impl.conv2d(jnp.array(img), jnp.array(ker), padding="VALID")
    assert out.shape == (2, 7, 6, 4)
    np.testing.assert_allclose(out[1], _naive_conv2d_valid(img[1], ker), rtol=1e-3, atol=1e-3)
    assert impl.conv2d(jnp.array(img), jnp.array(ker)).shape == (2, 9, 8, 4)  # SAME keeps H, W
    assert impl.conv2d(jnp.array(img), jnp.array(ker), stride=2).shape == (2, 5, 4, 4)


def test_max_pool2d(impl):
    x = jnp.arange(16.0).reshape(1, 4, 4, 1)
    np.testing.assert_array_equal(impl.max_pool2d(x)[0, :, :, 0], [[5.0, 7.0], [13.0, 15.0]])
    assert impl.max_pool2d(jnp.ones((2, 28, 28, 8))).shape == (2, 14, 14, 8)
    assert impl.max_pool2d(jnp.ones((1, 5, 5, 1)), 3, 1).shape == (1, 3, 3, 1)
    neg = -jnp.ones((1, 2, 2, 1)) * 7  # all-negative input: max must not be 0
    assert float(impl.max_pool2d(neg)[0, 0, 0, 0]) == -7.0


def test_count_conv_params(impl):
    assert impl.count_conv_params(3, 3, 1, 32) == 320
    assert impl.count_conv_params(3, 3, 32, 64) == 18_496
    assert impl.count_conv_params(5, 5, 3, 8, bias=False) == 600


def test_im2col(impl):
    rng = np.random.default_rng(2)
    img = rng.normal(size=(6, 7, 2)).astype(np.float32)
    ker = rng.normal(size=(3, 2, 2, 5)).astype(np.float32)
    cols = impl.im2col(jnp.array(img), 3, 2)
    assert cols.shape == (4 * 6, 3 * 2 * 2)
    out = (cols @ jnp.array(ker).reshape(-1, 5)).reshape(4, 6, 5)
    np.testing.assert_allclose(out, _naive_conv2d_valid(img, ker), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(cols[0], img[:3, :2].reshape(-1), rtol=1e-6)


def test_conv2d_matches_lax_on_gpu_dtype(impl):
    # bf16 inputs still produce the right shape and roughly the right values
    img = jax.random.normal(jax.random.key(3), (1, 8, 8, 2), jnp.bfloat16)
    ker = jax.random.normal(jax.random.key(4), (3, 3, 2, 3), jnp.bfloat16)
    out = impl.conv2d(img, ker, padding="VALID")
    ref = _naive_conv2d_valid(np.asarray(img[0], np.float32), np.asarray(ker, np.float32))
    np.testing.assert_allclose(np.asarray(out[0], np.float32), ref, rtol=0.05, atol=0.1)
