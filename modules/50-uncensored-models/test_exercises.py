import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.key(50)


def _rand(i, shape, scale=1.0):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32) * scale


# ---------------------------------------------------------------- unwanted
def test_unwanted_case_and_substrings(impl):
    phrases = ["As an AI language model", "bias"]
    assert impl.unwanted("as an ai LANGUAGE model, I can't", phrases) is True
    assert impl.unwanted("Add a bias term to each neuron.", phrases) is True
    assert impl.unwanted("The model is unbiased.", phrases) is True        # substring: "unbiased" contains "bias"
    assert impl.unwanted("Paris is the capital of France.", phrases) is False
    assert impl.unwanted("anything at all", []) is False


def test_unwanted_missing_comma(impl):
    # a missing comma makes Python join two phrases into one that never matches (Hartford's ancestor script had 4)
    phrases = ["I'm sorry" "As an AI", "September 2021"]  # noqa: ISC004  (the bug on purpose; ruff spots it)
    assert len(phrases) == 2
    assert impl.unwanted("I'm sorry, I can't do that.", phrases) is False
    assert impl.unwanted("I'm sorryAs an AI", phrases) is True


# ---------------------------------------------------------------- response_mask
def test_response_mask_left_padding(impl):
    m = np.asarray(impl.response_mask(jnp.array([5, 3]), jnp.array([2, 1]), 6))
    assert m.shape == (2, 5) and m.dtype == bool
    # row 0: real tokens at columns 1..5, response from column 3 -> targets j with j+1 >= 3
    assert m[0].tolist() == [False, False, True, True, True]
    # row 1: real tokens at columns 3..5, response from column 4
    assert m[1].tolist() == [False, False, False, True, True]


def test_response_mask_full_row_and_counts(impl):
    lengths, starts, T = jnp.array([8, 8, 4]), jnp.array([1, 7, 2]), 8
    m = np.asarray(impl.response_mask(lengths, starts, T))
    assert m.sum(1).tolist() == [7, 1, 2]                        # number of response tokens = length − start


# ---------------------------------------------------------------- refusal_score
def test_refusal_score_log_odds(impl):
    lg = _rand(1, (3, 11))
    ids = [2, 7]
    p = jax.nn.softmax(lg, -1)[:, ids].sum(-1)
    want = np.log(p + 1e-8) - np.log(1 - p + 1e-8)
    got = impl.refusal_score(lg, ids)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-5)
    assert got.shape == (3,) and got.dtype == jnp.float32


def test_refusal_score_extremes_finite_bf16(impl):
    lg = jnp.full((2, 5), -1e4).at[0, 1].set(1e4).at[1, 3].set(1e4).astype(jnp.bfloat16)
    s = np.asarray(impl.refusal_score(lg, jnp.array([1])))
    assert np.isfinite(s).all()
    assert s[0] > 15 and s[1] < -15                              # certain refusal vs certain non-refusal


# ---------------------------------------------------------------- diff_in_means
def test_diff_in_means(impl):
    a, b = _rand(2, (3, 5, 4)), _rand(3, (3, 7, 4), 2.0)
    got = impl.diff_in_means(a.astype(jnp.bfloat16), b)
    want = a.astype(jnp.bfloat16).astype(jnp.float32).mean(1) - b.mean(1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
    assert got.shape == (3, 4) and got.dtype == jnp.float32


# ---------------------------------------------------------------- ablate_dir / orthogonalize
def test_ablate_dir_removes_only_u(impl):
    x, u = _rand(4, (6, 8)), _rand(5, (8,), 3.0)
    y = impl.ablate_dir(x, u)
    uh = u / jnp.linalg.norm(u)
    np.testing.assert_allclose(np.asarray(y @ uh), 0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(y - x), np.asarray(-(x @ uh)[:, None] * uh), atol=1e-5)
    assert impl.ablate_dir(x.astype(jnp.bfloat16), u).dtype == jnp.bfloat16


def test_orthogonalize_equals_runtime_ablation(impl):
    W, u, h = _rand(6, (5, 8)), _rand(7, (8,), 0.1), _rand(8, (3, 4, 5))
    W2 = impl.orthogonalize(W, u)
    assert W2.shape == W.shape and W2.dtype == W.dtype
    np.testing.assert_allclose(np.asarray(h @ W2), np.asarray(impl.ablate_dir(h @ W, u)), atol=1e-5)
    np.testing.assert_allclose(np.asarray(W2 @ (u / jnp.linalg.norm(u))), 0, atol=1e-5)


def test_orthogonalize_idempotent(impl):
    W, u = _rand(9, (4, 6)), _rand(10, (6,))
    once = impl.orthogonalize(W, u)
    np.testing.assert_allclose(np.asarray(impl.orthogonalize(once, u)), np.asarray(once), atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_orthogonalize_keeps_dtype(impl, dtype):
    assert impl.orthogonalize(_rand(11, (3, 4)).astype(dtype), _rand(12, (4,))).dtype == dtype
