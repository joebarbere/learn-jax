import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_bradley_terry_loss(impl):
    rc, rr = jnp.array([2.0, 0.0, -1.0]), jnp.array([0.0, 0.0, 1.0])
    expected = np.mean([np.log1p(np.exp(-2.0)), np.log(2.0), np.log1p(np.exp(2.0))])
    assert float(impl.bradley_terry_loss(rc, rr)) == pytest.approx(expected, rel=1e-5)
    # numerically stable for huge margins, and differentiable
    assert np.isfinite(float(impl.bradley_terry_loss(jnp.array([-500.0]), jnp.array([500.0]))))
    g = jax.grad(lambda a: impl.bradley_terry_loss(a, jnp.zeros(1)))(jnp.zeros(1))
    assert float(g[0]) == pytest.approx(-0.5, rel=1e-5)


def test_pairwise_accuracy(impl):
    assert impl.pairwise_accuracy(np.array([1.0, 2.0, 3.0, 0.0]), np.array([0.0, 2.0, 1.0, 1.0])) == 0.5
    assert isinstance(impl.pairwise_accuracy(np.array([1.0]), np.array([0.0])), float)


def test_sequence_logprob(impl):
    key = jax.random.key(0)
    logits = jax.random.normal(key, (2, 3, 5))
    tokens = jnp.array([[1, 4, 2], [0, 3, 3]])
    mask = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    lp = jax.nn.log_softmax(logits, -1)
    expected = [float(lp[0, 0, 1] + lp[0, 1, 4] + lp[0, 2, 2]), float(lp[1, 0, 0] + lp[1, 1, 3])]
    np.testing.assert_allclose(np.asarray(impl.sequence_logprob(logits, tokens, mask)), expected, rtol=1e-5)
    # bf16 logits: still computed in float32
    out = impl.sequence_logprob(logits.astype(jnp.bfloat16), tokens, mask)
    assert out.dtype == jnp.float32


def test_implicit_reward(impl):
    np.testing.assert_allclose(np.asarray(impl.implicit_reward(jnp.array([-3.0]), jnp.array([-5.0]), 0.1)), [0.2],
                               rtol=1e-6)


def test_dpo_loss(impl):
    # at initialization (policy == reference) every margin is 0 → loss = log 2
    z = jnp.array([-10.0, -20.0])
    assert float(impl.dpo_loss(z, z - 3, z, z - 3, 0.1)) == pytest.approx(np.log(2.0), rel=1e-6)
    pc, pr = jnp.array([-4.0]), jnp.array([-9.0])
    rc, rr = jnp.array([-5.0]), jnp.array([-6.0])
    margin = 0.5 * ((-4 + 5) - (-9 + 6))
    assert float(impl.dpo_loss(pc, pr, rc, rr, 0.5)) == pytest.approx(np.log1p(np.exp(-margin)), rel=1e-5)
    # gradient pushes chosen up and rejected down
    g = jax.grad(lambda a, b: impl.dpo_loss(a, b, rc, rr, 0.5), argnums=(0, 1))(pc, pr)
    assert float(g[0][0]) < 0 < float(g[1][0])


def test_win_rate(impl):
    assert impl.win_rate([3, 1, 2, 2], [1, 2, 2, 0]) == pytest.approx((1 + 0 + 0.5 + 1) / 4)
    assert isinstance(impl.win_rate([1.0], [0.0]), float)


def test_best_of_n(impl):
    s = np.array([[0.1, 0.9, 0.5, 2.0], [1.0, 1.0, 0.0, 3.0]])
    np.testing.assert_array_equal(impl.best_of_n(s, 1), [0, 0])
    np.testing.assert_array_equal(impl.best_of_n(s, 3), [1, 0])     # ties → earliest
    np.testing.assert_array_equal(impl.best_of_n(s, 4), [3, 3])
