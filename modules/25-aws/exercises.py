"""Module 25 exercises: the arithmetic of renting accelerators.

Replace each `raise NotImplementedError` with your code, then run `make test N=25`. Plain Python — these
are the numbers you check *before* running `terraform apply`.
"""

# The Terraform profiles in infra/aws (instance types per session) and us-east-1 on-demand $/hour
# (instances.vantage.sh, 2026-11-14).
PROFILES = {
    "gpu1": ["g6.xlarge"],
    "gpu1-l40s": ["g6e.xlarge"],
    "multihost": ["g6.xlarge", "g6.xlarge"],
    "gpu4": ["g6.12xlarge"],
    "trn": ["trn1.2xlarge"],
}
PRICES = {"g6.xlarge": 0.805, "g6e.xlarge": 1.861, "g6.12xlarge": 4.602, "trn1.2xlarge": 1.344}


def dollars_per_million_tokens(tokens_per_s: float, usd_per_hour: float) -> float:
    """Cost to process one million tokens at this throughput and hourly price.

    The single number that compares a $0.81/h L4 with a $4.60/h 4-GPU box (and your own card).
    """
    raise NotImplementedError


def allreduce_bytes(n_params: int, n_devices: int, algorithm: str = "ring", bytes_per_param: int = 4) -> float:
    """Bytes each device must *send* to all-reduce a gradient of n_params values.

    - "ring": reduce-scatter then all-gather around a ring → 2·(n−1)/n of the gradient per device.
      (Almost independent of n: why rings scale.)
    - "naive": every device sends its whole gradient to every other device → (n−1) gradients.
    With one device nothing is sent (0). Raise ValueError for an unknown algorithm.
    """
    raise NotImplementedError


def bandwidth_gbs(nbytes: float, seconds: float) -> float:
    """Achieved bandwidth in GB/s (10**9 bytes per second)."""
    raise NotImplementedError


def choose_strategy(state_bytes: float, device_mem_bytes: float, n_devices: int, headroom: float = 0.5) -> str:
    """Pick the simplest strategy that fits (Module 24's vocabulary).

    `state_bytes` is the full training state (params + grads + optimizer moments). Only `headroom` (a
    fraction) of each device's memory may hold it; the rest is for activations.
      - "DP"   if the whole state fits on one device (replicate it everywhere: least communication),
      - "FSDP" if it fits once sharded evenly across all n devices,
      - "too big" otherwise (more devices, smaller model, or offloading).
    """
    raise NotImplementedError


def spot_savings(on_demand_usd: float, spot_usd: float) -> float:
    """Fraction saved by spot over on-demand, e.g. 0.6 for 60% cheaper. Never negative."""
    raise NotImplementedError


def session_cost(profile: str, minutes: float) -> float:
    """On-demand cost of running a Terraform profile for `minutes` (all its instances; use PROFILES/PRICES).

    Raise KeyError for an unknown profile.
    """
    raise NotImplementedError
