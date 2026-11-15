import pytest


def test_arithmetic_intensity(impl):
    # a 4096^3 bf16 matmul: 2·4096³ FLOPs over three 4096² bf16 matrices
    flops, nbytes = 2 * 4096**3, 3 * 4096**2 * 2
    assert impl.arithmetic_intensity(flops, nbytes) == pytest.approx(4096 / 3)
    assert impl.arithmetic_intensity(10, 40) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        impl.arithmetic_intensity(1.0, 0)


def test_roofline_time(impl):
    peak, bw = 27e12, 288e9
    # compute-bound: huge intensity → flops / peak
    assert impl.roofline_time(2.7e12, 1e6, peak, bw) == pytest.approx(0.1)
    # memory-bound: elementwise op, 1 FLOP per 6 bytes → bytes / bandwidth
    assert impl.roofline_time(1e9, 6e9, peak, bw) == pytest.approx(6e9 / 288e9)
    # exactly at the ridge point both terms are equal
    ridge = peak / bw
    assert impl.roofline_time(ridge * 1e9, 1e9, peak, bw) == pytest.approx(1e9 / bw)


def test_mfu(impl):
    assert impl.mfu(1e5, 10_000_000, 6e12) == pytest.approx(1.0)
    assert impl.mfu(112_000, 13_890_000, 27e12) == pytest.approx(6 * 13_890_000 * 112_000 / 27e12)
    assert impl.mfu(0, 1_000_000, 27e12) == 0


def test_pad_to_multiple(impl):
    assert impl.pad_to_multiple(50257, 128) == 50304
    assert impl.pad_to_multiple(50257, 64) == 50304
    assert impl.pad_to_multiple(8192, 128) == 8192
    assert impl.pad_to_multiple(1, 64) == 64
    assert impl.pad_to_multiple(0, 64) == 0
    with pytest.raises(ValueError):
        impl.pad_to_multiple(10, 0)


def test_parse_top_kernels(impl):
    events = [
        {"name": "gemm", "dur": 600.0}, {"name": "fusion", "dur": 10.0},
        {"name": "gemm", "dur": 590.0}, {"name": "softmax", "dur": 40.0},
        {"name": "fusion", "dur": 30.0}, {"name": "copy", "dur": 40.0, "extra": "ignored"},
    ]
    top = impl.parse_top_kernels(events, 3)
    assert top[0] == ("gemm", pytest.approx(1190.0), 2)
    # copy, fusion and softmax all total 40: ties are broken by name
    assert [t[0] for t in top] == ["gemm", "copy", "fusion"]
    assert top[2][2] == 2
    assert impl.parse_top_kernels(events, 10)[-1][0] == "softmax"
    assert impl.parse_top_kernels([], 5) == []


def test_step_gaps(impl):
    assert impl.step_gaps([(0, 10), (15, 20), (20, 30)]) == pytest.approx([5, 0])
    # unsorted and overlapping: the (2, 8) span lies inside (0, 10), so the next gap is 12 - 10
    assert impl.step_gaps([(12, 14), (0, 10), (2, 8)]) == pytest.approx([0, 2])
    assert impl.step_gaps([(3, 4)]) == []
    assert impl.step_gaps([]) == []
