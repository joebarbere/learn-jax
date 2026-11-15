"""Module 25 reference solutions. Try the exercises first — `make test N=25` grades yours."""

PROFILES = {
    "gpu1": ["g6.xlarge"],
    "gpu1-l40s": ["g6e.xlarge"],
    "multihost": ["g6.xlarge", "g6.xlarge"],
    "gpu4": ["g6.12xlarge"],
    "trn": ["trn1.2xlarge"],
}
PRICES = {"g6.xlarge": 0.805, "g6e.xlarge": 1.861, "g6.12xlarge": 4.602, "trn1.2xlarge": 1.344}


def dollars_per_million_tokens(tokens_per_s: float, usd_per_hour: float) -> float:
    return usd_per_hour / (tokens_per_s * 3600) * 1e6


def allreduce_bytes(n_params: int, n_devices: int, algorithm: str = "ring", bytes_per_param: int = 4) -> float:
    grad = n_params * bytes_per_param
    if algorithm == "ring":
        return 2 * (n_devices - 1) / n_devices * grad
    if algorithm == "naive":
        return (n_devices - 1) * grad
    raise ValueError(f"unknown algorithm {algorithm!r}")


def bandwidth_gbs(nbytes: float, seconds: float) -> float:
    return nbytes / seconds / 1e9


def choose_strategy(state_bytes: float, device_mem_bytes: float, n_devices: int, headroom: float = 0.5) -> str:
    budget = device_mem_bytes * headroom
    if state_bytes <= budget:
        return "DP"
    if state_bytes / n_devices <= budget:
        return "FSDP"
    return "too big"


def spot_savings(on_demand_usd: float, spot_usd: float) -> float:
    return max(0.0, 1 - spot_usd / on_demand_usd)


def session_cost(profile: str, minutes: float) -> float:
    return sum(PRICES[t] for t in PROFILES[profile]) * minutes / 60
