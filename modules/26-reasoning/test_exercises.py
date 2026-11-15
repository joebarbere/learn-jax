import math

import jax.numpy as jnp
import numpy as np
import pytest


def test_verify_answer(impl):
    v = impl.verify_answer
    assert v("215 + 104 = 319. <answer>191</answer>", 191)
    assert v("<answer> -42 </answer>", -42)
    assert v("<answer>+7</answer>", 7)
    assert v("so <answer>1,024</answer>", 1024)
    assert not v("<answer>190</answer>", 191)
    assert not v("the answer is 191", 191)                                   # no tag
    assert not v("<answer>191", 191)                                         # unclosed
    assert not v("<answer>190</answer><answer>191</answer>", 191)            # hedging with two tags
    assert not v("<answer>about 191</answer>", 191)
    assert not v("<answer>191.5</answer>", 191)


def test_format_reward(impl):
    f = impl.format_reward
    assert f("1 + 1 = 2. <answer>2</answer>") == 1.0
    assert f("<answer> 2 </answer>\n  ") == 1.0
    assert f("<answer>2</answer> and more text") == 0.0                    # trailing junk
    assert f("<answer>2</answer><answer>3</answer>") == 0.0
    assert f("<answer>two</answer>") == 0.0
    assert f("no tags") == 0.0
    assert f("<answer>999</answer>") == 1.0                                # format only: wrong is fine


def test_combined_reward(impl):
    c = impl.combined_reward
    assert c("x <answer>5</answer>", 5) == pytest.approx(1.2)
    assert c("x <answer>5</answer>", 5, format_bonus=0.5) == pytest.approx(1.5)
    assert c("<answer>5</answer> trailing", 5) == pytest.approx(1.0)       # correct, untidy: no bonus
    assert c("<answer>6</answer>", 5) == 0.0                               # tidy but wrong: nothing


def test_group_advantages(impl):
    r = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    adv = np.asarray(impl.group_advantages(r, 4))
    assert adv.shape == (8,) and adv.dtype == np.float32
    np.testing.assert_allclose(adv[:4], [1.0, -1.0, -1.0, 1.0], atol=1e-3)
    np.testing.assert_allclose(adv[4:], 0.0, atol=1e-6)                     # no signal in an all-equal group
    adv2 = np.asarray(impl.group_advantages([3.0, 1.0, 2.0], 3))
    np.testing.assert_allclose(adv2, np.array([1, -1, 0]) / np.std([3, 1, 2]), rtol=1e-3)
    np.testing.assert_allclose(adv2.mean(), 0.0, atol=1e-6)


def test_clipped_pg_loss(impl):
    mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    lp = jnp.log(jnp.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]]))
    adv = jnp.array([1.0, -1.0])
    # ratio 1 everywhere: loss = -mean(A) = 0
    assert float(impl.clipped_pg_loss(lp, lp, adv, mask)) == pytest.approx(0.0, abs=1e-6)
    # ratio 2 on a positive-advantage completion is clipped to 1.2; on a negative one min() keeps 2*A
    lp_new = lp + jnp.log(2.0)
    want = -(1.2 * 1.0 + 2.0 * -1.0) / 2
    assert float(impl.clipped_pg_loss(lp_new, lp, adv, mask, 0.2)) == pytest.approx(want, rel=1e-5)
    # padded positions are ignored; an all-zero mask doesn't divide by zero
    junk = lp_new.at[0, 2].set(50.0)
    assert float(impl.clipped_pg_loss(junk, lp, adv, mask, 0.2)) == pytest.approx(want, rel=1e-5)
    z = impl.clipped_pg_loss(lp, lp, adv, jnp.zeros_like(mask))
    assert np.isfinite(float(z))


def test_kl_penalty(impl):
    mask = jnp.array([[1.0, 1.0, 0.0]])
    lp = jnp.log(jnp.array([[0.5, 0.25, 0.9]]))
    assert float(impl.kl_penalty(lp, lp, mask)) == pytest.approx(0.0, abs=1e-7)
    ref = lp + jnp.array([[0.3, -0.2, 99.0]])
    d = np.array([0.3, -0.2])
    want = np.mean(np.exp(d) - d - 1)
    assert float(impl.kl_penalty(lp, ref, mask)) == pytest.approx(want, rel=1e-5)
    assert float(impl.kl_penalty(lp, ref, mask)) >= 0


def test_majority_vote(impl):
    m = impl.majority_vote
    assert m([3, 5, 3, None, 5, 3]) == 3
    assert m([7, 5, 5, 7]) == 7                                              # tie: 7 got there first
    assert m([None, 4, None]) == 4
    assert m([None, None]) is None
    assert m([]) is None


def test_pass_at_k(impl):
    p = impl.pass_at_k
    assert p(10, 0, 1) == 0.0
    assert p(10, 10, 3) == 1.0
    assert p(10, 3, 1) == pytest.approx(0.3)
    assert p(10, 3, 2) == pytest.approx(1 - math.comb(7, 2) / math.comb(10, 2))
    assert p(5, 4, 2) == 1.0                                                 # n - c < k
