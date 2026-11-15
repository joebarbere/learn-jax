import jax
import jax.numpy as jnp
import numpy as np


def test_masks(impl):
    c = np.asarray(impl.causal_mask(5))
    b = np.asarray(impl.bidirectional_mask(5))
    assert c.shape == b.shape == (5, 5) and c.dtype == bool and b.dtype == bool
    assert c[3, 2] and not c[2, 3] and c[4, 4] and c.sum() == 15
    assert b.all()


def test_mask_tokens_proportions(impl):
    ids = jnp.full((200, 500), 7, jnp.int32)
    inputs, is_t = impl.mask_tokens(jax.random.key(0), ids, mask_id=999, vocab_size=50, rate=0.15)
    inputs, is_t = np.asarray(inputs), np.asarray(is_t)
    assert inputs.shape == ids.shape and is_t.dtype == bool
    frac = is_t.mean()
    assert 0.14 < frac < 0.16
    picked = inputs[is_t]
    assert 0.78 < (picked == 999).mean() < 0.82
    rand = (picked != 999) & (picked != 7)
    # ~10% random (a random draw can also hit 7, so allow a little slack)
    assert 0.08 < rand.mean() < 0.11
    assert 0.09 < (picked == 7).mean() < 0.13
    assert (inputs[~is_t] == 7).all()                     # untouched elsewhere
    assert picked.max() < 1000 and picked.min() >= 0


def test_mask_tokens_jits_and_varies_with_key(impl):
    ids = jnp.arange(64, dtype=jnp.int32).reshape(4, 16) % 20
    f = jax.jit(lambda k: impl.mask_tokens(k, ids, 99, 20, 0.3))
    _, ta = f(jax.random.key(1))
    _, tb = f(jax.random.key(2))
    assert not np.array_equal(np.asarray(ta), np.asarray(tb))


def test_mlm_loss(impl):
    logits = jnp.log(jnp.array([[[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [1 / 3, 1 / 3, 1 / 3]]]))
    targets = jnp.array([[0, 2, 1]])
    only_first = jnp.array([[True, False, False]])
    assert np.isclose(float(impl.mlm_loss(logits, targets, only_first)), -np.log(0.7), atol=1e-5)
    both = jnp.array([[True, True, False]])
    assert np.isclose(float(impl.mlm_loss(logits, targets, both)), -(np.log(0.7) + np.log(0.8)) / 2, atol=1e-5)
    none = jnp.zeros((1, 3), bool)
    assert float(impl.mlm_loss(logits, targets, none)) == 0.0


def test_pools(impl):
    h = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3)
    np.testing.assert_array_equal(np.asarray(impl.cls_pool(h)), np.asarray(h[:, 0]))
    np.testing.assert_array_equal(np.asarray(impl.last_token_pool(h)), np.asarray(h[:, -1]))
    got = np.asarray(impl.last_token_pool(h, jnp.array([2, 4])))
    np.testing.assert_array_equal(got, np.stack([np.asarray(h[0, 1]), np.asarray(h[1, 3])]))


def test_contextual_similarity(impl):
    v = jnp.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    s = np.asarray(impl.contextual_similarity(v))
    assert s.shape == (4, 4)
    np.testing.assert_allclose(np.diag(s), 1.0, atol=1e-6)
    assert np.isclose(s[0, 1], 1.0, atol=1e-6) and np.isclose(s[0, 2], 0.0, atol=1e-6)
    assert np.isclose(s[0, 3], 1 / np.sqrt(2), atol=1e-6)
    np.testing.assert_allclose(s, s.T, atol=1e-6)
