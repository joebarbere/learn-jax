import pytest


def test_matmul_flops(impl):
    assert impl.matmul_flops(2, 3, 4) == 48
    assert impl.matmul_flops(4096, 4096, 4096) == 2 * 4096**3


def test_tflops(impl):
    assert impl.tflops(2e12, 1.0) == pytest.approx(2.0)
    assert impl.tflops(2 * 4096**3, 0.005) == pytest.approx(27.487790694, rel=1e-6)


def test_array_bytes(impl):
    assert impl.array_bytes((1024, 1024), "float32") == 4 * 2**20
    assert impl.array_bytes((8192, 384), "bfloat16") == 8192 * 384 * 2
    assert impl.array_bytes((), "int8") == 1  # a scalar still holds one value


def test_training_bytes_per_param(impl):
    assert impl.training_bytes_per_param() == 16
    assert impl.training_bytes_per_param("bfloat16") == 8


def test_max_trainable_params(impl):
    # 16 GiB card, half reserved, f32 Adam: 8 GiB / 16 B = 536,870,912 params (~0.5B)
    assert impl.max_trainable_params(16) == 536_870_912
    assert impl.max_trainable_params(16, headroom=0.75) == 268_435_456
    assert isinstance(impl.max_trainable_params(15.98), int)
