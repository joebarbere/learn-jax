"""Module 22 reference solutions. Try the exercises first — `make test N=22` grades yours."""
import math


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved


def roofline_time(flops: float, bytes_moved: float, peak_flops: float, bandwidth: float) -> float:
    return max(flops / peak_flops, bytes_moved / bandwidth)


def mfu(tokens_per_s: float, n_params: int, peak_flops: float) -> float:
    return 6 * n_params * tokens_per_s / peak_flops


def pad_to_multiple(n: int, m: int) -> int:
    if m <= 0:
        raise ValueError("m must be positive")
    return math.ceil(n / m) * m


def parse_top_kernels(trace_events: list[dict], k: int) -> list[tuple[str, float, int]]:
    totals: dict[str, list] = {}
    for e in trace_events:
        t = totals.setdefault(e["name"], [0.0, 0])
        t[0] += e["dur"]
        t[1] += 1
    ranked = sorted(((name, d, c) for name, (d, c) in totals.items()), key=lambda x: (-x[1], x[0]))
    return ranked[:k]


def step_gaps(spans: list[tuple[float, float]]) -> list[float]:
    ordered = sorted(spans)
    gaps = []
    busy_until = None
    for start, end in ordered:
        if busy_until is not None:
            gaps.append(max(0.0, start - busy_until))
            busy_until = max(busy_until, end)
        else:
            busy_until = end
    return gaps
