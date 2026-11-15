import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_q_learning_update(impl):
    Q = jnp.zeros((4, 2)).at[3].set(jnp.array([1.0, 2.0]))
    Q2 = impl.q_learning_update(Q, 0, 1, 0.5, 3, 0.0, 0.1, 0.9)
    # target = 0.5 + 0.9 * 2 = 2.3; Q[0,1] += 0.1 * 2.3
    assert float(Q2[0, 1]) == pytest.approx(0.23)
    np.testing.assert_allclose(np.asarray(Q2).ravel()[[0, 2, 3, 4, 5]], np.asarray(Q).ravel()[[0, 2, 3, 4, 5]])
    # terminal: no bootstrap
    Q3 = impl.q_learning_update(Q, 0, 0, 1.0, 3, 1.0, 0.5, 0.9)
    assert float(Q3[0, 0]) == pytest.approx(0.5)
    jax.jit(impl.q_learning_update)(Q, 0, 1, 0.5, 3, 0.0, 0.1, 0.9)


def test_epsilon_greedy(impl):
    q = jnp.array([[0.0, 5.0, 1.0], [3.0, 0.0, 0.0]])
    a = impl.epsilon_greedy(jax.random.key(0), q, 0.0)
    np.testing.assert_array_equal(np.asarray(a), [1, 0])
    single = impl.epsilon_greedy(jax.random.key(0), jnp.array([0.0, 0.0, 9.0]), 0.0)
    assert int(single) == 2 and jnp.shape(single) == ()
    qs = jnp.tile(jnp.array([[0.0, 5.0, 1.0]]), (3000, 1))
    acts = np.asarray(impl.epsilon_greedy(jax.random.key(1), qs, 1.0))
    counts = np.bincount(acts, minlength=3) / 3000
    np.testing.assert_allclose(counts, [1 / 3] * 3, atol=0.05)   # eps=1: uniform
    acts = np.asarray(impl.epsilon_greedy(jax.random.key(2), qs, 0.3))
    assert 0.7 < np.mean(acts == 1) < 0.86                         # ~0.7 + 0.3/3


def test_td_target(impl):
    r = jnp.array([1.0, 1.0])
    done = jnp.array([0.0, 1.0])
    qn = jnp.array([[0.0, 2.0], [5.0, 7.0]])
    np.testing.assert_allclose(impl.td_target(r, 0.5, done, qn), [2.0, 1.0])


def test_dqn_loss_and_gradient(impl):
    q = jnp.array([[1.0, 2.0], [0.0, 0.0]])
    a = jnp.array([1, 0])
    target = jnp.array([2.5, 3.0])
    # errors: -0.5 (quadratic: 0.125), -3 (linear: 2.5) -> mean 1.3125
    assert float(impl.dqn_loss(q, a, target)) == pytest.approx(1.3125)
    g = jax.grad(lambda t: impl.dqn_loss(q, a, t))(target)
    np.testing.assert_allclose(g, [0.0, 0.0])                     # no gradient into the target
    gq = jax.grad(lambda qq: impl.dqn_loss(qq, a, target))(q)
    np.testing.assert_allclose(gq, [[0.0, -0.25], [-0.5, 0.0]])   # clipped to -1, averaged over 2


def _empty(cap):
    return {"obs": jnp.zeros((cap, 2)), "action": jnp.zeros(cap, jnp.int32), "ptr": jnp.int32(0),
            "size": jnp.int32(0)}


def test_replay_add_wraps_and_samples(impl):
    buf = _empty(5)
    for start in (0, 3):
        n = 3
        batch = {"obs": jnp.full((n, 2), start, jnp.float32) + jnp.arange(n)[:, None],
                 "action": jnp.arange(start, start + n, dtype=jnp.int32)}
        buf = impl.replay_add(buf, batch)
    # wrote 0,1,2 then 3,4,5 -> 5 overwrites slot 0
    assert int(buf["size"]) == 5 and int(buf["ptr"]) == 1
    np.testing.assert_array_equal(np.asarray(buf["action"]), [5, 1, 2, 3, 4])
    s = impl.replay_sample(buf, jax.random.key(0), 1000)
    assert set(s) == {"obs", "action"} and s["obs"].shape == (1000, 2)
    assert set(np.unique(np.asarray(s["action"]))) == {1, 2, 3, 4, 5}
    # only valid entries are sampled from a partly full buffer
    part = impl.replay_add(_empty(10), {"obs": jnp.ones((2, 2)), "action": jnp.array([7, 8], jnp.int32)})
    s2 = impl.replay_sample(part, jax.random.key(1), 200)
    assert set(np.unique(np.asarray(s2["action"]))) == {7, 8}
    jax.jit(impl.replay_add)(buf, {"obs": jnp.zeros((2, 2)), "action": jnp.zeros(2, jnp.int32)})


def test_alphazero_loss(impl):
    logits = jnp.log(jnp.array([[0.5, 0.25, 0.25]]))
    pi = jnp.array([[1.0, 0.0, 0.0]])
    total, (ce, mse) = impl.alphazero_loss(logits, jnp.array([0.5]), pi, jnp.array([1.0]))
    assert float(ce) == pytest.approx(np.log(2), rel=1e-5)
    assert float(mse) == pytest.approx(0.25)
    assert float(total) == pytest.approx(np.log(2) + 0.25, rel=1e-5)
    # the policy loss is minimized when the network matches the MCTS distribution
    pi2 = jnp.array([[0.5, 0.25, 0.25]])
    _, (ce_match, _) = impl.alphazero_loss(logits, jnp.array([0.0]), pi2, jnp.array([0.0]))
    _, (ce_other, _) = impl.alphazero_loss(jnp.zeros((1, 3)), jnp.array([0.0]), pi2, jnp.array([0.0]))
    assert float(ce_match) < float(ce_other)


def test_puct(impl):
    s = impl.puct(jnp.array([0.0, 0.5]), jnp.array([0.9, 0.1]), 16.0, jnp.array([0.0, 7.0]), 1.5)
    np.testing.assert_allclose(s, [5.4, 0.575], rtol=1e-6)
    # with no visits yet, a higher prior wins; with many visits, value dominates
    many = impl.puct(jnp.array([-0.5, 0.5]), jnp.array([0.9, 0.1]), 1e4, jnp.array([9e3, 9e2]), 1.0)
    assert float(many[1]) > float(many[0])
