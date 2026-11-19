import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(41)


def _rand(i, shape, scale=1.0):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32) * scale


def _unit(x):
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True)


def _inputs(B=2, H=3, T=64, dk=8, dv=5):
    q, k, v = _rand(1, (B, H, T, dk)), _unit(_rand(2, (B, H, T, dk))), _rand(3, (B, H, T, dv))
    log_a = -jax.nn.softplus(_rand(4, (B, H, T))) * 0.3
    beta = jax.nn.sigmoid(_rand(5, (B, H, T)))
    return q, k, v, log_a, beta


def _loop(q, k, v, log_a, beta=None):
    """A plain NumPy loop over time: the definition."""
    q, k, v, log_a = (np.asarray(x, np.float64) for x in (q, k, v, log_a))
    lead, T = q.shape[:-2], q.shape[-2]
    q, k, v = (x.reshape(-1, T, x.shape[-1]) for x in (q, k, v))
    log_a = log_a.reshape(-1, T)
    b = None if beta is None else np.asarray(beta, np.float64).reshape(-1, T)
    out = np.zeros(q.shape[:2] + (v.shape[-1],))
    for n in range(q.shape[0]):
        S = np.zeros((q.shape[-1], v.shape[-1]))
        for t in range(T):
            S = np.exp(log_a[n, t]) * S
            if b is None:
                S = S + np.outer(k[n, t], v[n, t])
            else:
                S = S + b[n, t] * np.outer(k[n, t], v[n, t] - k[n, t] @ S)
            out[n, t] = q[n, t] @ S
    return out.reshape(*lead, T, -1)


def _close(a, b, tol=1e-4):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.abs(a - b).max() <= tol * max(np.abs(b).max(), 1.0), np.abs(a - b).max()


# ---------------------------------------------------------------- memory_costs
def test_memory_costs(impl):
    att = impl.memory_costs(8, 1, 6, 64, 4096)
    assert att == {"attn_layers": 8, "linear_layers": 0, "kv_cache_bytes": 2 * 8 * 6 * 64 * 4096 * 2,
                   "state_bytes": 0, "total_bytes": 2 * 8 * 6 * 64 * 4096 * 2}
    lin = impl.memory_costs(8, 0, 6, 64, 4096)
    assert lin["attn_layers"] == 0 and lin["kv_cache_bytes"] == 0 and lin["state_bytes"] == 8 * 6 * 64 * 64 * 4
    assert impl.memory_costs(8, 0, 6, 64, 10**6)["total_bytes"] == lin["total_bytes"]   # constant in context
    hyb = impl.memory_costs(48, 4, 2, 256, 262144, kv_bytes=1)
    assert (hyb["attn_layers"], hyb["linear_layers"]) == (12, 36)
    assert hyb["kv_cache_bytes"] == 2 * 12 * 2 * 256 * 262144
    assert hyb["total_bytes"] == hyb["kv_cache_bytes"] + hyb["state_bytes"]
    assert all(isinstance(x, int) for x in hyb.values())


# ---------------------------------------------------------------- linear attention
def test_causal_linear_attention(impl):
    q, k, v, _, _ = _inputs()
    out = impl.causal_linear_attention(q, k, v)
    _close(out, _loop(q, k, v, jnp.zeros(q.shape[:-1])))


def test_causal_linear_attention_is_causal(impl):
    q, k, v, _, _ = _inputs(T=16)
    out = impl.causal_linear_attention(q, k, v)
    out2 = impl.causal_linear_attention(q, k, v.at[..., 10:, :].set(99.0))
    _close(out[..., :10, :], out2[..., :10, :], 1e-6)


def test_linear_attention_scan_no_decay(impl):
    q, k, v, _, _ = _inputs()
    _close(impl.linear_attention_scan(q, k, v), impl.causal_linear_attention(q, k, v))


def test_linear_attention_scan_decay(impl):
    q, k, v, log_a, _ = _inputs()
    _close(impl.linear_attention_scan(q, k, v, log_a), _loop(q, k, v, log_a))


def test_linear_attention_scan_any_leading_axes(impl):
    q, k, v, log_a, _ = _inputs()
    out = impl.linear_attention_scan(q[0, 0], k[0, 0], v[0, 0], log_a[0, 0])     # no batch or head axes
    _close(out, _loop(q[0, 0], k[0, 0], v[0, 0], log_a[0, 0]))


# ---------------------------------------------------------------- segsum + chunks
def test_segsum(impl):
    L = impl.segsum(jnp.log(jnp.array([0.5, 0.25, 0.5, 1.0])))
    expect = np.array([[1, 0, 0, 0], [0.25, 1, 0, 0], [0.125, 0.5, 1, 0], [0.125, 0.5, 1, 1]])
    _close(jnp.exp(L), expect, 1e-6)
    assert np.isneginf(np.asarray(L)[0, 1])
    assert impl.segsum(jnp.zeros((2, 3, 5))).shape == (2, 3, 5, 5)


def test_segsum_mask_equals_scan(impl):
    q, k, v, log_a, _ = _inputs(T=32)
    att = jnp.einsum("...td,...sd->...ts", q, k) * jnp.exp(impl.segsum(log_a))
    _close(att @ v, _loop(q, k, v, log_a))


@pytest.mark.parametrize("chunk", [8, 16, 64])
def test_chunked_linear_attention(impl, chunk):
    q, k, v, log_a, _ = _inputs()
    _close(impl.chunked_linear_attention(q, k, v, log_a, chunk), _loop(q, k, v, log_a))


def test_chunked_linear_attention_no_decay(impl):
    q, k, v, _, _ = _inputs()
    _close(impl.chunked_linear_attention(q, k, v, jnp.zeros(q.shape[:-1]), 16), _loop(q, k, v, jnp.zeros(q.shape[:-1])))


# ---------------------------------------------------------------- delta rule
def test_gated_delta_scan(impl):
    q, k, v, log_a, beta = _inputs()
    _close(impl.gated_delta_scan(q, k, v, beta, log_a), _loop(q, k, v, log_a, beta))


def test_delta_rule_no_decay(impl):
    q, k, v, _, beta = _inputs()
    _close(impl.gated_delta_scan(q, k, v, beta), _loop(q, k, v, jnp.zeros(q.shape[:-1]), beta))


def test_delta_rule_overwrites(impl):
    """β = 1 with the same key twice: the second value *replaces* the first (linear attention would add)."""
    key = _unit(jnp.ones((1, 4)))
    k = jnp.concatenate([key, key], 0)
    v = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    out = impl.gated_delta_scan(k, k, v, jnp.ones(2))
    _close(out[1], jnp.array([0.0, 1.0]), 1e-5)
