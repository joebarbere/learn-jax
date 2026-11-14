"""Module 12 exercises: convolutions, from a sliding dot product to im2col.

Replace each `raise NotImplementedError` with your code, then run `make test N=12`.
"Convolution" here means what every deep learning library computes: cross-correlation (the kernel is
*not* flipped). Image layout is NHWC (batch, height, width, channels); kernel layout is HWIO
(height, width, in-channels, out-channels).
"""
import jax  # noqa: F401  (you'll need these)
import jax.numpy as jnp  # noqa: F401


def conv_output_size(n: int, k: int, stride: int = 1, padding: int = 0) -> int:
    """Output length along one axis: floor((n + 2·padding - k) / stride) + 1.

    Raise ValueError if the kernel doesn't fit (k > n + 2·padding) or stride < 1.
    """
    raise NotImplementedError


def conv1d(x, w):
    """'Valid' 1-D convolution: out[i] = sum_j x[i + j] * w[j], for i in 0 .. n-k.

    x: (n,), w: (k,) with k <= n. Returns (n - k + 1,).
    No Python loop over positions: build all windows at once (fancy indexing with a broadcast
    index grid), then one matrix-vector product.
    """
    raise NotImplementedError


def conv2d(images, kernels, stride: int = 1, padding: str = "SAME"):
    """2-D convolution of a batch of images with a bank of kernels, via jax.lax.conv_general_dilated.

    images: (N, H, W, C_in); kernels: (kh, kw, C_in, C_out); padding "SAME" or "VALID".
    Returns (N, H', W', C_out). Pass window_strides=(stride, stride) and
    dimension_numbers=("NHWC", "HWIO", "NHWC").
    """
    raise NotImplementedError


def max_pool2d(x, size: int = 2, stride: int | None = None):
    """Max pooling over size×size windows of an NHWC array (VALID padding).

    stride defaults to `size` (non-overlapping windows). Use jax.lax.reduce_window with init value
    -inf and jax.lax.max; windows and strides apply to H and W only (1 for N and C).
    """
    raise NotImplementedError


def count_conv_params(kh: int, kw: int, c_in: int, c_out: int, bias: bool = True) -> int:
    """Learnable parameters in a conv layer: one kh×kw×c_in kernel per output channel, plus a bias each."""
    raise NotImplementedError


def im2col(image, kh: int, kw: int):
    """Stretch: unfold an (H, W, C) image into patches for a VALID, stride-1 convolution.

    Returns (H'·W', kh·kw·C) with H' = H-kh+1, W' = W-kw+1, rows in row-major (y, x) order and each row
    ordered (dy, dx, c) — so that `im2col(img, kh, kw) @ kernel.reshape(kh*kw*C, C_out)` reshaped to
    (H', W', C_out) equals the VALID convolution of `img` with an HWIO `kernel`. (Convolution is a matmul!)
    """
    raise NotImplementedError
