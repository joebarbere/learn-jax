"""Module 22 exercises: the arithmetic of a profiler.

Replace each `raise NotImplementedError` with your code, then run `make test N=22`.
These are plain Python (no JAX needed): the lesson feeds them real measurements from your GPU — a
profiler trace, compiled-HLO shapes, and the speed ladder — so once they work, every table and
roofline plot in the lesson is computed by your code.
"""


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte of memory traffic: flops / bytes_moved.

    The single number that says whether a kernel can be compute-bound. Raise ValueError if
    bytes_moved <= 0.
    """
    raise NotImplementedError


def roofline_time(flops: float, bytes_moved: float, peak_flops: float, bandwidth: float) -> float:
    """Best possible seconds for a kernel on a roofline: the slower of computing and moving data.

    max(flops / peak_flops, bytes_moved / bandwidth), with peak_flops in FLOP/s and bandwidth in
    bytes/s. (Real kernels overlap the two at best, so this is a lower bound on time.)
    """
    raise NotImplementedError


def mfu(tokens_per_s: float, n_params: int, peak_flops: float) -> float:
    """Model FLOPs utilization: the fraction of the GPU's peak a training run achieves.

    Training a transformer costs about 6·N FLOPs per token (2N forward, 4N backward; N = parameter
    count, ignoring attention's extra term). Return 6 * n_params * tokens_per_s / peak_flops
    as a fraction (0.25 means 25%).
    """
    raise NotImplementedError


def pad_to_multiple(n: int, m: int) -> int:
    """Smallest multiple of m that is >= n (e.g. GPT-2's vocab 50257 → 50304 for m=128).

    n itself if it is already a multiple. Raise ValueError if m <= 0.
    """
    raise NotImplementedError


def parse_top_kernels(trace_events: list[dict], k: int) -> list[tuple[str, float, int]]:
    """Summarize a profiler trace: which kernels took the most total time?

    trace_events: dicts with at least "name" (kernel name) and "dur" (duration, any unit).
    Return up to k tuples (name, total_dur, count), sorted by total_dur descending; break ties by
    name ascending so the result is deterministic.
    """
    raise NotImplementedError


def step_gaps(spans: list[tuple[float, float]]) -> list[float]:
    """Idle time between consecutive busy spans on one device — the "step bubble".

    spans: (start, end) pairs, e.g. kernel intervals from a trace, in any order and possibly
    overlapping. Sort by start; walk them keeping track of when the device becomes free (the max end
    seen so far); for each span after the first, the gap is max(0, start - free_at). Return the
    list of gaps (length len(spans) - 1; empty for 0 or 1 spans).
    """
    raise NotImplementedError
