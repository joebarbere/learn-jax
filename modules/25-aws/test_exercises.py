import pytest


def test_dollars_per_million_tokens(impl):
    # 1M tokens at 1,000 tok/s takes 1000 s; at $3.60/h that's $1.00
    assert impl.dollars_per_million_tokens(1000, 3.6) == pytest.approx(1.0)
    assert impl.dollars_per_million_tokens(94_833, 0.805) == pytest.approx(0.002358, rel=1e-3)


def test_allreduce_bytes(impl):
    n = 1_000_000
    assert impl.allreduce_bytes(n, 1) == 0
    assert impl.allreduce_bytes(n, 2) == pytest.approx(4e6)            # 2·1/2 of 4 MB
    assert impl.allreduce_bytes(n, 4) == pytest.approx(6e6)            # 2·3/4 of 4 MB
    assert impl.allreduce_bytes(n, 8, "naive") == pytest.approx(28e6)  # 7 full gradients
    assert impl.allreduce_bytes(n, 4, bytes_per_param=2) == pytest.approx(3e6)
    # ring cost per device approaches 2× the gradient, whatever n
    assert impl.allreduce_bytes(n, 1024) < 2 * 4e6
    with pytest.raises(ValueError):
        impl.allreduce_bytes(n, 4, "tree?")


def test_bandwidth_gbs(impl):
    assert impl.bandwidth_gbs(2e9, 1.0) == pytest.approx(2.0)
    assert impl.bandwidth_gbs(64 * 2**20, 0.01) == pytest.approx(6.7108864)


def test_choose_strategy(impl):
    gb = 1e9
    assert impl.choose_strategy(2 * gb, 24 * gb, 4) == "DP"
    assert impl.choose_strategy(30 * gb, 24 * gb, 4) == "FSDP"      # 7.5 GB per device ≤ 12 GB
    assert impl.choose_strategy(60 * gb, 24 * gb, 4) == "too big"   # 15 GB per device > 12 GB
    assert impl.choose_strategy(12 * gb, 24 * gb, 1) == "DP"        # exactly at the budget fits
    assert impl.choose_strategy(13 * gb, 24 * gb, 1) == "too big"   # one device can't shard
    assert impl.choose_strategy(20 * gb, 24 * gb, 4, headroom=0.9) == "DP"


def test_spot_savings(impl):
    assert impl.spot_savings(4.602, 1.84) == pytest.approx(0.6002, rel=1e-3)
    assert impl.spot_savings(1.0, 1.0) == 0.0
    assert impl.spot_savings(1.0, 1.2) == 0.0  # spot above on-demand: no savings, not negative


def test_session_cost(impl):
    assert impl.session_cost("gpu1", 60) == pytest.approx(0.805)
    assert impl.session_cost("multihost", 90) == pytest.approx(2 * 0.805 * 1.5)
    assert impl.session_cost("gpu4", 45) == pytest.approx(4.602 * 0.75)
    with pytest.raises(KeyError):
        impl.session_cost("p5", 60)


def test_estimate_run(impl):
    r = impl.estimate_run(1e8, 2e9, peak_tflops=100.0, mfu=0.5, usd_per_hour=2.0)
    assert set(r) == {"state_gb", "flops", "hours", "usd"}
    assert r["state_gb"] == pytest.approx(1.6)
    assert r["flops"] == pytest.approx(1.2e18)
    assert r["hours"] == pytest.approx(1.2e18 / 5e13 / 3600)
    assert r["usd"] == pytest.approx(2.0 * r["hours"])
    four = impl.estimate_run(1e8, 2e9, peak_tflops=100.0, mfu=0.5, n_devices=4, usd_per_hour=2.0)
    assert four["hours"] == pytest.approx(r["hours"] / 4)           # perfect split
    assert four["state_gb"] == pytest.approx(r["state_gb"])         # state is not divided
    assert impl.estimate_run(1e6, 1e6, 1.0, 1.0)["usd"] == 0        # edge: free (your own card's default)


def test_dp_comm_ratio(impl):
    r = impl.dp_comm_ratio(1e8, 4096, 4, peak_tflops=100.0, mfu=0.5, interconnect_gbs=10.0)
    comm = 2 * 3 / 4 * 4e8 / 10e9
    compute = 6 * 1e8 * 4096 / 5e13
    assert r == pytest.approx(comm / compute)
    # both sides scale with the parameter count, so model size cancels
    assert impl.dp_comm_ratio(7e9, 4096, 4, 100.0, 0.5, 10.0) == pytest.approx(r, rel=1e-6)
    # twice the tokens per device: twice the compute, same communication
    assert impl.dp_comm_ratio(1e8, 8192, 4, 100.0, 0.5, 10.0) == pytest.approx(r / 2)
    assert impl.dp_comm_ratio(1e8, 4096, 1, 100.0, 0.5, 10.0) == 0.0  # edge: one device
