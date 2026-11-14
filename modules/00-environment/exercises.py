"""Module 00 exercises: back-of-the-envelope GPU arithmetic.

Replace each `raise NotImplementedError` with your code, then run `make test N=00`.
These are plain Python — no JAX needed yet.
"""

BYTES_PER_DTYPE = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1}


def matmul_flops(m: int, k: int, n: int) -> int:
    """Floating-point operations to multiply an (m×k) matrix by a (k×n) matrix.

    Each of the m·n outputs is a dot product of length k: k multiplies and k adds.
    """
    raise NotImplementedError


def tflops(flops: float, seconds: float) -> float:
    """Achieved throughput in TFLOP/s (10**12 floating-point operations per second)."""
    raise NotImplementedError


def array_bytes(shape: tuple[int, ...], dtype: str) -> int:
    """Memory used by a dense array of this shape and dtype (use BYTES_PER_DTYPE)."""
    raise NotImplementedError


def training_bytes_per_param(param_dtype: str = "float32") -> int:
    """Bytes per parameter to train with Adam, ignoring activations.

    You keep: the parameter, its gradient, and Adam's two moment estimates (m and v) —
    four numbers per parameter, all stored in `param_dtype`.
    """
    raise NotImplementedError


def max_trainable_params(vram_gb: float, param_dtype: str = "float32", headroom: float = 0.5) -> int:
    """Rough largest model you can train with Adam on a card with `vram_gb` GiB of memory.

    Reserve `headroom` (a fraction, e.g. 0.5) of memory for activations and the runtime; the rest
    holds training_bytes_per_param() per parameter. 1 GiB = 2**30 bytes. Round down to an int.
    """
    raise NotImplementedError
