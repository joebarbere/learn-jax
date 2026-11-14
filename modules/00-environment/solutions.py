"""Module 00 reference solutions. Try the exercises first — `make test N=00` grades yours."""

BYTES_PER_DTYPE = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1}


def matmul_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def tflops(flops: float, seconds: float) -> float:
    return flops / seconds / 1e12


def array_bytes(shape: tuple[int, ...], dtype: str) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count * BYTES_PER_DTYPE[dtype]


def training_bytes_per_param(param_dtype: str = "float32") -> int:
    return 4 * BYTES_PER_DTYPE[param_dtype]


def max_trainable_params(vram_gb: float, param_dtype: str = "float32", headroom: float = 0.5) -> int:
    usable = vram_gb * 2**30 * (1 - headroom)
    return int(usable // training_bytes_per_param(param_dtype))
