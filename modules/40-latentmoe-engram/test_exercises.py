import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(40)


def _rand(i, shape, scale=1.0):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32) * scale


# ---------------------------------------------------------------- latent_moe_costs
def test_latent_moe_costs_standard(impl):
    c = impl.latent_moe_costs(384, 256, 16, 4, 4, "standard")
    assert c == {"width": 384, "experts": 16, "active": 4, "expert_params": 3 * 384 * 256,
                 "routed_params": 16 * 3 * 384 * 256, "active_params": 4 * 3 * 384 * 256,
                 "flops_per_token": 8 * 3 * 384 * 256, "a2a_bytes_per_token": 2 * 4 * 384 * 2}


def test_latent_moe_costs_latent(impl):
    std = impl.latent_moe_costs(4096, 2688, 128, 6, 4, "standard", bytes_per_value=1)
    acc = impl.latent_moe_costs(4096, 2688, 128, 6, 4, "acc", bytes_per_value=1)
    eff = impl.latent_moe_costs(4096, 2688, 128, 6, 4, "eff", bytes_per_value=1)
    proj = 2 * 4096 * 1024
    assert (acc["width"], acc["experts"], acc["active"]) == (1024, 512, 24)
    assert (eff["width"], eff["experts"], eff["active"]) == (1024, 512, 6)
    assert acc["expert_params"] * 4 == std["expert_params"]
    assert acc["routed_params"] == std["routed_params"] + proj           # same expert budget, plus projections
    assert acc["active_params"] == std["active_params"] + proj
    assert acc["a2a_bytes_per_token"] == std["a2a_bytes_per_token"]     # K'ℓ = Kd
    assert eff["a2a_bytes_per_token"] * 4 == std["a2a_bytes_per_token"]
    assert eff["flops_per_token"] == 2 * (6 * 3 * 1024 * 2688 + proj)
    assert all(isinstance(v, int) for v in acc.values())


# ---------------------------------------------------------------- latent_moe
def _params(d=12, l=4, m=5, E=6, h=7):
    return {"down": _rand(1, (d, l), 0.3), "up": _rand(2, (l, d), 0.3),
            "experts": {"w1": _rand(3, (E, l, m), 0.3), "w3": _rand(4, (E, l, m), 0.3),
                        "w2": _rand(5, (E, m, l), 0.3)},
            "shared": {"w1": _rand(6, (d, h), 0.3), "w3": _rand(7, (d, h), 0.3), "w2": _rand(8, (h, d), 0.3)}}


def _gates(N, E, k, i=9):
    s = jax.nn.sigmoid(_rand(i, (N, E)))
    _, idx = jax.lax.top_k(s, k)
    picked = jax.nn.one_hot(idx, E).sum(-2) > 0
    g = jnp.where(picked, s, 0.0)
    return g / g.sum(-1, keepdims=True)


def _ref_latent(p, x, g):
    p = jax.tree.map(np.asarray, p)
    x, g = np.asarray(x), np.asarray(g)
    silu = lambda a: a / (1 + np.exp(-a))
    out = np.zeros_like(x)
    for n in range(x.shape[0]):
        z = x[n] @ p["down"]
        acc = np.zeros(p["up"].shape[0])
        for e in range(g.shape[1]):
            if g[n, e] != 0:
                w = p["experts"]
                acc += g[n, e] * ((silu(z @ w["w1"][e]) * (z @ w["w3"][e])) @ w["w2"][e])
        s = p["shared"]
        out[n] = acc @ p["up"] + (silu(x[n] @ s["w1"]) * (x[n] @ s["w3"])) @ s["w2"]
    return out


def test_latent_moe(impl):
    p, x = _params(), _rand(10, (9, 12))
    g = _gates(9, 6, 2)
    out = impl.latent_moe(p, x, g)
    assert out.shape == (9, 12)
    np.testing.assert_allclose(out, _ref_latent(p, x, g), rtol=1e-4, atol=1e-5)


def test_latent_moe_identity_projections_is_a_plain_moe(impl):
    """With ℓ = d and identity projections, LatentMoE is exactly a DeepSeekMoE layer."""
    p = _params(d=6, l=6)
    p["down"], p["up"] = jnp.eye(6), jnp.eye(6)
    x, g = _rand(11, (5, 6)), _gates(5, 6, 3, i=12)
    np.testing.assert_allclose(impl.latent_moe(p, x, g), _ref_latent(p, x, g), rtol=1e-4, atol=1e-5)
    zero = impl.latent_moe(p, x, jnp.zeros((5, 6)))                    # no routed expert: shared only
    s = p["shared"]
    np.testing.assert_allclose(zero, (jax.nn.silu(x @ s["w1"]) * (x @ s["w3"])) @ s["w2"], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- compress_vocab
def test_compress_vocab(impl):
    lookup, n = impl.compress_vocab(["The", " the", "THE", "a", " A", "é", "e", "ﬁ", "fi"])
    assert np.asarray(lookup).tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3]   # case, spaces, accents, ligature (NFKC)
    assert n == 4 and np.asarray(lookup).dtype == np.int64


def test_compress_vocab_whitespace(impl):
    lookup, n = impl.compress_vocab(["\n", "  ", " ", "\t\n", "", "x"])
    # every whitespace-only token becomes " " (kept, not stripped); "" normalizes to "" -> keeps its own text
    assert np.asarray(lookup).tolist() == [0, 0, 0, 0, 1, 2] and n == 3


def test_compress_vocab_partial_utf8_never_merges(impl):
    lookup, n = impl.compress_vocab(["\ufffd", "a", "\ufffd", "A"])
    assert np.asarray(lookup).tolist() == [0, 1, 2, 1] and n == 3


# ---------------------------------------------------------------- ngram_hashes
def _ref_hash(ids, mult, primes, pad):
    B, T = ids.shape
    cols = []
    for n, heads in zip(range(2, len(primes) + 2), primes):
        for p in heads:
            col = np.zeros((B, T), np.int64)
            for b in range(B):
                for t in range(T):
                    mix = 0
                    for k in range(n):
                        tok = int(ids[b, t - k]) if t - k >= 0 else pad
                        mix ^= tok * int(mult[k])
                    col[b, t] = mix % p
            cols.append(col)
    return np.stack(cols, 2)


def test_ngram_hashes(impl):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 6000, (2, 11)).astype(np.int64)
    mult = rng.integers(0, 2**40, 3).astype(np.int64) * 2 + 1
    primes = [[1009, 1013], [1019, 1021]]
    out = impl.ngram_hashes(ids, mult, primes, 7)
    assert out.shape == (2, 11, 4) and np.asarray(out).dtype == np.int64
    np.testing.assert_array_equal(out, _ref_hash(ids, mult, primes, 7))


def test_ngram_hashes_padding_and_locality(impl):
    mult = np.array([3, 5, 7], np.int64)
    ids = np.array([[4, 9, 9, 4, 9, 9]], np.int64)
    out = np.asarray(impl.ngram_hashes(ids, mult, [[101], [103]], 0))
    assert out[0, 0, 0] == (4 * 3) % 101                                # (pad, 4): 4*3 XOR 0*5
    assert out[0, 2, 0] == out[0, 5, 0] and out[0, 2, 1] == out[0, 5, 1]  # same n-gram, same rows: (4,9,9)
    assert out[0, 1, 0] == out[0, 4, 0] and out[0, 1, 1] != out[0, 4, 1]  # same bigram (4,9), different trigram


# ---------------------------------------------------------------- engram_gate
def test_engram_gate(impl):
    h, e = _rand(20, (2, 3, 8)), _rand(21, (2, 3, 6))
    p = {"w_k": _rand(22, (6, 8)), "w_v": _rand(23, (6, 8)), "norm_h": _rand(24, (8,)) + 1,
         "norm_k": _rand(25, (8,)) + 1}
    v, alpha = impl.engram_gate(h, e, p)
    rms = lambda a, w: a / np.sqrt(np.mean(np.asarray(a) ** 2, -1, keepdims=True) + 1e-6) * w
    k = np.asarray(e @ p["w_k"])
    ref_a = 1 / (1 + np.exp(-(rms(h, p["norm_h"]) * rms(k, p["norm_k"])).sum(-1) / np.sqrt(8)))
    assert v.shape == (2, 3, 8) and alpha.shape == (2, 3)
    np.testing.assert_allclose(alpha, ref_a, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(v, ref_a[..., None] * np.asarray(e @ p["w_v"]), rtol=1e-5, atol=1e-6)


def test_engram_gate_scale_invariant(impl):
    """RMSNorm makes the gate ignore the scale of h and of the memory."""
    h, e = _rand(30, (1, 4, 8)), _rand(31, (1, 4, 6))
    p = {"w_k": _rand(32, (6, 8)), "w_v": _rand(33, (6, 8)), "norm_h": jnp.ones(8), "norm_k": jnp.ones(8)}
    a1 = impl.engram_gate(h, e, p)[1]
    a2 = impl.engram_gate(10 * h, 0.1 * e, p)[1]
    np.testing.assert_allclose(a1, a2, rtol=1e-3)
    assert ((a1 > 0) & (a1 < 1)).all()


# ---------------------------------------------------------------- causal_dwconv
def _ref_conv(x, w, dil):
    x, w = np.asarray(x), np.asarray(w)
    T = x.shape[1]
    K = w.shape[0]
    y = np.zeros_like(x)
    for t in range(T):
        for i in range(K):
            s = t - (K - 1 - i) * dil
            if s >= 0:
                y[:, t] += w[i] * x[:, s]
    return y


@pytest.mark.parametrize("dil", [1, 3])
def test_causal_dwconv(impl, dil):
    x, w = _rand(40, (2, 13, 5)), _rand(41, (4, 5))
    out = impl.causal_dwconv(x, w, dil)
    assert out.shape == (2, 13, 5)
    np.testing.assert_allclose(out, _ref_conv(x, w, dil), rtol=1e-5, atol=1e-6)


def test_causal_dwconv_is_causal_and_zero_init(impl):
    x, w = _rand(42, (1, 10, 3)), _rand(43, (4, 3))
    x2 = x.at[:, 6:].set(99.0)                                          # change the future only
    np.testing.assert_allclose(impl.causal_dwconv(x, w, 2)[:, :6], impl.causal_dwconv(x2, w, 2)[:, :6], atol=1e-6)
    np.testing.assert_allclose(impl.causal_dwconv(x, jnp.zeros((4, 3)), 3), 0.0)
