import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(49)


def _rand(i, shape, scale=1.0):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32) * scale


# ---------------------------------------------------------------- forgetting_metrics
def test_forgetting_metrics_example(impl):
    R = [[0.9, 0.0, 0.0],
         [0.6, 0.8, 0.0],
         [0.3, 0.7, 0.9]]
    m = impl.forgetting_metrics(R)
    assert m["avg"] == pytest.approx((0.3 + 0.7 + 0.9) / 3)
    assert m["bwt"] == pytest.approx(((0.3 - 0.9) + (0.7 - 0.8)) / 2)
    assert m["forgetting"] == pytest.approx(((0.9 - 0.3) + (0.8 - 0.7)) / 2)
    assert all(isinstance(v, float) for v in m.values())


def test_forgetting_metrics_peak_later_and_single(impl):
    # forgetting measures from the best earlier score, not the score right after training
    R = np.array([[0.5, 0.0, 0.0], [0.9, 0.7, 0.0], [0.6, 0.7, 0.8]])
    m = impl.forgetting_metrics(R)
    assert m["bwt"] == pytest.approx(((0.6 - 0.5) + 0.0) / 2)      # task 0 ends above where it started …
    assert m["forgetting"] == pytest.approx(((0.9 - 0.6) + 0.0) / 2)   # … but below its best
    assert impl.forgetting_metrics([[0.4]]) == {"avg": pytest.approx(0.4), "bwt": 0.0, "forgetting": 0.0}


# ---------------------------------------------------------------- ewc_penalty
def test_ewc_penalty(impl):
    params = {"w": jnp.array([1.0, 2.0, 3.0]), "b": {"c": jnp.array([[0.5]], jnp.bfloat16)}}
    anchor = {"w": jnp.array([1.0, 0.0, 1.0]), "b": {"c": jnp.array([[0.0]], jnp.bfloat16)}}
    fisher = {"w": jnp.array([10.0, 1.0, 0.5]), "b": {"c": jnp.array([[4.0]])}}
    got = impl.ewc_penalty(params, anchor, fisher)
    assert got.dtype == jnp.float32 and got.shape == ()
    assert float(got) == pytest.approx(0 + 1.0 * 4 + 0.5 * 4 + 4.0 * 0.25)
    assert float(impl.ewc_penalty(params, params, fisher)) == 0.0


# ---------------------------------------------------------------- expert_out
def test_expert_out_matches_loop(impl):
    h, A, B = _rand(0, (2, 3, 8)), _rand(1, (3, 8, 4)), _rand(2, (3, 4, 8))
    gate = jnp.array([1.0, 0.0, 1.0])
    want = sum(gate[e] * (jax.nn.silu(h @ A[e]) @ B[e]) for e in range(3))
    got = impl.expert_out(h, A, B, gate)
    assert got.shape == (2, 3, 8) and got.dtype == h.dtype
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_expert_out_off_and_new(impl):
    h, A, B = _rand(3, (1, 5, 8)), _rand(4, (2, 8, 4)), _rand(5, (2, 4, 8))
    np.testing.assert_array_equal(impl.expert_out(h, A, B, jnp.zeros(2)), 0.0)
    np.testing.assert_array_equal(impl.expert_out(h, A, jnp.zeros_like(B), jnp.ones(2)), 0.0)
    assert impl.expert_out(h.astype(jnp.bfloat16), A, B, jnp.ones(2)).dtype == jnp.bfloat16


# ---------------------------------------------------------------- token_kl
def test_token_kl_values(impl):
    t, s = _rand(6, (2, 3, 11), 2.0), _rand(7, (2, 3, 11), 2.0)
    p, q = jax.nn.softmax(t, -1), jax.nn.softmax(s, -1)
    fwd = jnp.sum(p * jnp.log(p / q), -1)
    rev = jnp.sum(q * jnp.log(q / p), -1)
    m = (p + q) / 2
    jsd = 0.5 * jnp.sum(p * jnp.log(p / m), -1) + 0.5 * jnp.sum(q * jnp.log(q / m), -1)
    for kind, want in (("forward", fwd), ("reverse", rev), ("jsd", jsd)):
        got = impl.token_kl(t, s, kind)
        assert got.shape == (2, 3) and got.dtype == jnp.float32
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)
    assert float(jnp.max(jsd)) <= np.log(2) + 1e-6


def test_token_kl_shift_invariant_and_default(impl):
    t = _rand(8, (4, 7)).astype(jnp.bfloat16)
    np.testing.assert_allclose(impl.token_kl(t, t + 3.0), 0.0, atol=1e-3)   # bf16 rounds t + 3 a little
    t2, s2 = _rand(9, (5,)), _rand(10, (5,))
    np.testing.assert_allclose(impl.token_kl(t2, s2), impl.token_kl(t2, s2, "forward"))


# ---------------------------------------------------------------- levenshtein_reward
def test_levenshtein_reward(impl):
    assert impl.levenshtein_reward("kitten", "sitting", 5) == pytest.approx(1 - 3 / 7)
    assert impl.levenshtein_reward("kitten", "sitting", 2) == 0.0
    assert impl.levenshtein_reward([1, 2, 3], [1, 2, 3], 0) == 1.0
    assert impl.levenshtein_reward([1, 2, 3], [], 3) == 0.0
    assert impl.levenshtein_reward([], [], 0) == 1.0
    assert impl.levenshtein_reward([5, 1, 2, 3], [1, 2, 3], 1) == pytest.approx(0.75)


# ---------------------------------------------------------------- select_dreams
def test_select_dreams(impl):
    scores = [0.1, 5.0, 0.3, 4.0, 4.0, 0.2, 0.0, 9.0]
    got = impl.select_dreams(scores, k=3, b=0)
    assert got == [1, 3, 7]                                      # ties at 4.0: the lower index wins
    got = impl.select_dreams(scores, k=3, b=2, seed=0)
    assert len(got) == 5 and got == sorted(set(got)) and {1, 3, 7} <= set(got)
    assert got == impl.select_dreams(scores, k=3, b=2, seed=0)
    assert impl.select_dreams(scores, k=6, b=4) == list(range(8))
    assert all(isinstance(i, int) for i in got)
