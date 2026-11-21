import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(45)
M = 99


def _x(B=4, T=2000, V=50, i=0):
    return jax.random.randint(jax.random.fold_in(KEY, i), (B, T), 0, V)


# ---------------------------------------------------------------- mask_tokens
def test_mask_tokens_rates(impl):
    x = _x()
    t = jnp.array([0.1, 0.5, 0.9, 1.0])
    z = impl.mask_tokens(jax.random.key(1), x, t, M)
    assert z.shape == x.shape and z.dtype == x.dtype
    frac = np.asarray((z == M).mean(-1))
    np.testing.assert_allclose(frac[:3], [0.1, 0.5, 0.9], atol=0.04)
    assert frac[3] == 1.0
    keep = np.asarray(z != M)
    assert (np.asarray(z)[keep] == np.asarray(x)[keep]).all()          # unmasked tokens are untouched


def test_mask_tokens_random(impl):
    x = _x(B=1)
    t = jnp.array([0.5])
    a, b = impl.mask_tokens(jax.random.key(1), x, t, M), impl.mask_tokens(jax.random.key(2), x, t, M)
    assert not np.array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------- mdlm_loss
def test_mdlm_loss(impl):
    B, T, V = 3, 6, 7
    logits = jax.random.normal(jax.random.key(3), (B, T, V))
    x = jax.random.randint(jax.random.key(4), (B, T), 0, V)
    z = jnp.where(jax.random.uniform(jax.random.key(5), (B, T)) < 0.5, M, x)
    t = jnp.array([0.2, 0.5, 0.9])
    lp = np.asarray(jax.nn.log_softmax(logits))
    expect = np.mean([sum(-lp[b, i, int(x[b, i])] for i in range(T) if int(z[b, i]) == M) / (float(t[b]) * T)
                      for b in range(B)])
    assert float(impl.mdlm_loss(logits, x, z, t, M)) == pytest.approx(expect, rel=1e-5)


def test_mdlm_loss_no_masks_is_zero(impl):
    logits = jax.random.normal(jax.random.key(3), (2, 5, 7))
    x = jnp.ones((2, 5), jnp.int32)
    assert float(impl.mdlm_loss(logits, x, x, jnp.array([0.3, 0.7]), M)) == 0.0


# ---------------------------------------------------------------- unmask_step
def _setup(B=2, T=10, V=12):
    logits = jax.random.normal(jax.random.key(6), (B, T, V + 1)) * 3
    z = jnp.array([[1, M, M, 2, M, M, M, 3, M, M], [M] * 10])
    return logits, z, V


@pytest.mark.parametrize("by_conf", [True, False])
def test_unmask_step_counts(impl, by_conf):
    logits, z, V = _setup()
    M2 = V
    z = jnp.where(z == M, M2, z)
    new = impl.unmask_step(jax.random.key(7), logits, z, 3, M2, by_confidence=by_conf)
    assert np.array_equal(np.asarray((z == M2).sum(-1) - (new == M2).sum(-1)), [3, 3])
    keep = np.asarray(z != M2)
    assert (np.asarray(new)[keep] == np.asarray(z)[keep]).all()        # revealed tokens never change
    assert int((new[(z == M2) & (new != M2)] == M2).sum()) == 0


def test_unmask_step_never_samples_mask(impl):
    logits, z, V = _setup()
    M2 = V
    z = jnp.full((2, 10), M2)
    logits = logits.at[..., M2].set(100.0)                               # the model loves [MASK]; ignore it
    new = impl.unmask_step(jax.random.key(8), logits, z, 10, M2)
    assert int((new == M2).sum()) == 0


def test_unmask_step_confidence_order(impl):
    V, M2 = 5, 5
    logits = jnp.full((1, 4, V + 1), -10.0)
    logits = logits.at[0, 0, 1].set(10.0).at[0, 2, 3].set(10.0)        # positions 0 and 2 are near-certain
    z = jnp.full((1, 4), M2)
    new = impl.unmask_step(jax.random.key(9), logits, z, 2, M2, by_confidence=True)
    assert np.array_equal(np.asarray(new[0]), [1, M2, 3, M2])


def test_unmask_step_leaves_other_values(impl):
    logits, _, V = _setup()
    z = jnp.array([[-1, V, V, -1, V, V, V, -1, V, V], [V] * 10])
    new = impl.unmask_step(jax.random.key(7), logits, z, 2, V)
    assert (np.asarray(new)[np.asarray(z) == -1] == -1).all()


# ---------------------------------------------------------------- schedules
def test_reveal_schedule(impl):
    assert impl.reveal_schedule(10, 4) == [3, 3, 2, 2]
    assert impl.reveal_schedule(8, 8) == [1] * 8
    assert impl.reveal_schedule(3, 5) == [1, 1, 1, 0, 0]
    assert sum(impl.reveal_schedule(512, 64)) == 512


def test_block_order(impl):
    assert impl.block_order(10, 2, 3) == [(2, 5), (5, 8), (8, 10)]
    assert impl.block_order(8, 0, 4) == [(0, 4), (4, 8)]
    assert impl.block_order(5, 5, 4) == []
