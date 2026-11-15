import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_attention_entropy(impl):
    w = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0]])
    e = np.asarray(impl.attention_entropy(w))
    assert e.shape == (3,)
    np.testing.assert_allclose(e, [0.0, np.log(4), np.log(2)], atol=1e-6)
    batched = np.asarray(impl.attention_entropy(jnp.stack([w, w])))
    assert batched.shape == (2, 3)


def test_previous_token_score(impl):
    T = 6
    prev = jnp.zeros((T, T)).at[0, 0].set(1.0).at[jnp.arange(1, T), jnp.arange(T - 1)].set(1.0)
    assert impl.previous_token_score(prev) == pytest.approx(1.0)
    uniform = jnp.tril(jnp.ones((T, T))) / jnp.arange(1, T + 1)[:, None]
    expected = float(np.mean([1 / (i + 1) for i in range(1, T)]))
    assert impl.previous_token_score(uniform) == pytest.approx(expected, rel=1e-5)
    assert impl.previous_token_score(jnp.stack([prev, uniform])) == pytest.approx((1 + expected) / 2, rel=1e-5)


def _perfect_induction(tokens):
    B, T = tokens.shape
    attn = np.zeros((B, T, T), np.float32)
    for b in range(B):
        for i in range(T):
            targets = [j for j in range(1, i + 1) if tokens[b, j - 1] == tokens[b, i]]
            if targets:
                attn[b, i, targets] = 1.0 / len(targets)
            else:
                attn[b, i, 0] = 1.0
    return attn


def test_induction_score(impl):
    rng = np.random.default_rng(0)
    first = rng.permutation(50)[:10]
    toks = np.concatenate([first, first])[None, :]                        # distinct tokens, repeated once
    perfect = _perfect_induction(toks)
    assert impl.induction_score(jnp.asarray(perfect), jnp.asarray(toks)) == pytest.approx(1.0)
    T = toks.shape[1]
    prev_only = np.zeros((1, T, T), np.float32)
    prev_only[0, 0, 0] = 1
    prev_only[0, np.arange(1, T), np.arange(T - 1)] = 1                 # a previous-token head
    s = impl.induction_score(jnp.asarray(prev_only), jnp.asarray(toks))
    assert s < 0.15
    no_repeat = np.arange(12)[None, :]                                    # no query has a target
    assert impl.induction_score(jnp.asarray(_perfect_induction(no_repeat)), jnp.asarray(no_repeat)) == 0.0


def test_ablate_head(impl):
    m = jnp.ones((2, 4))
    out = impl.ablate_head(m, 1, 2)
    assert float(out[1, 2]) == 0.0 and float(out.sum()) == 7.0
    assert float(m.sum()) == 8.0                                          # original untouched


def test_linear_probe_recovers_a_linear_code(impl):
    key = jax.random.key(0)
    N, d, K = 3000, 16, 5
    acts = jax.random.normal(key, (N, d))
    # labels are a linear (argmax) function of the activations → a linear probe can get them right
    W_true = jax.random.normal(jax.random.key(1), (d, K, 3))
    labels = jnp.argmax(jnp.einsum("nd,dkc->nkc", acts, W_true), -1)
    W, b = impl.train_linear_probe(acts[:2500], labels[:2500], 3, steps=300, lr=5e-2)
    assert W.shape == (d, K, 3) and b.shape == (K, 3)
    acc = impl.probe_accuracy(W, b, acts[2500:], labels[2500:])
    assert isinstance(acc, float) and acc > 0.9
    rand = jax.random.randint(jax.random.key(2), (N, K), 0, 3)               # unpredictable labels → chance
    W2, b2 = impl.train_linear_probe(acts[:2500], rand[:2500], 3, steps=300, lr=5e-2)
    assert impl.probe_accuracy(W2, b2, acts[2500:], rand[2500:]) < 0.45


def test_to_relative_board(impl):
    ab = jnp.array([[0, 1, 2, 1], [0, 1, 2, 2]])                        # 0 empty, 1 black, 2 white
    player = jnp.array([1, 2])                                           # row 0 seen by black, row 1 by white
    out = np.asarray(impl.to_relative_board(ab, player))
    np.testing.assert_array_equal(out, [[0, 1, 2, 1], [0, 2, 1, 1]])
    np.testing.assert_array_equal(np.asarray(impl.to_relative_board(jnp.zeros((3, 64), int), jnp.array([1, 2, 1]))), 0)
