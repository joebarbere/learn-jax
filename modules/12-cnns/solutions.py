"""Module 12 reference solutions. Try the exercises first — `make test N=12` grades yours."""
import jax
import jax.numpy as jnp


def conv_output_size(n: int, k: int, stride: int = 1, padding: int = 0) -> int:
    if stride < 1 or k > n + 2 * padding:
        raise ValueError("kernel does not fit or stride < 1")
    return (n + 2 * padding - k) // stride + 1


def conv1d(x, w):
    n, k = x.shape[0], w.shape[0]
    idx = jnp.arange(n - k + 1)[:, None] + jnp.arange(k)[None, :]
    return x[idx] @ w


def conv2d(images, kernels, stride: int = 1, padding: str = "SAME"):
    return jax.lax.conv_general_dilated(images, kernels, window_strides=(stride, stride), padding=padding,
                                        dimension_numbers=("NHWC", "HWIO", "NHWC"))


def max_pool2d(x, size: int = 2, stride: int | None = None):
    stride = size if stride is None else stride
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, size, size, 1), (1, stride, stride, 1),
                                 "VALID")


def count_conv_params(kh: int, kw: int, c_in: int, c_out: int, bias: bool = True) -> int:
    return kh * kw * c_in * c_out + (c_out if bias else 0)


def im2col(image, kh: int, kw: int):
    h, w, c = image.shape
    ho, wo = h - kh + 1, w - kw + 1
    ys = jnp.arange(ho)[:, None, None, None] + jnp.arange(kh)[None, None, :, None]   # (ho, 1, kh, 1)
    xs = jnp.arange(wo)[None, :, None, None] + jnp.arange(kw)[None, None, None, :]   # (1, wo, 1, kw)
    patches = image[ys, xs]                                                          # (ho, wo, kh, kw, c)
    return patches.reshape(ho * wo, kh * kw * c)
