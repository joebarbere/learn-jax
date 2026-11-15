import jax
import jax.numpy as jnp
import numpy as np


def _params(key, d_in, H, k):
    k1, k2, k3 = jax.random.split(key, 3)
    return {"Wx": 0.3 * jax.random.normal(k1, (d_in, k * H)),
            "Wh": 0.3 * jax.random.normal(k2, (H, k * H)),
            "b": 0.1 * jax.random.normal(k3, (k * H,))}


def _sig(z):
    return 1 / (1 + np.exp(-z))


def test_rnn_cell(impl):
    p = _params(jax.random.key(0), 3, 4, 1)
    h = jax.random.normal(jax.random.key(1), (2, 4))
    x = jax.random.normal(jax.random.key(2), (2, 3))
    want = np.tanh(np.asarray(x) @ np.asarray(p["Wx"]) + np.asarray(h) @ np.asarray(p["Wh"]) + np.asarray(p["b"]))
    np.testing.assert_allclose(np.asarray(impl.rnn_cell(p, h, x)), want, atol=1e-5)
    # unbatched works too
    np.testing.assert_allclose(np.asarray(impl.rnn_cell(p, h[0], x[0])), want[0], atol=1e-5)


def test_lstm_cell(impl):
    H = 4
    p = {k: np.asarray(v) for k, v in _params(jax.random.key(3), 3, H, 4).items()}
    h = np.asarray(jax.random.normal(jax.random.key(4), (2, H)))
    c = np.asarray(jax.random.normal(jax.random.key(5), (2, H)))
    x = np.asarray(jax.random.normal(jax.random.key(6), (2, 3)))
    z = x @ p["Wx"] + h @ p["Wh"] + p["b"]
    i, f, g, o = _sig(z[:, :H]), _sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), _sig(z[:, 3 * H:])
    c_want = f * c + i * g
    h_new, c_new = impl.lstm_cell(p, (jnp.asarray(h), jnp.asarray(c)), jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(c_new), c_want, atol=1e-5)
    np.testing.assert_allclose(np.asarray(h_new), o * np.tanh(c_want), atol=1e-5)
    # forget gate saturated open, input gate shut: the cell state passes through unchanged
    p2 = {"Wx": np.zeros((3, 4 * H), np.float32), "Wh": np.zeros((H, 4 * H), np.float32),
          "b": np.concatenate([-30 * np.ones(H), 30 * np.ones(H), np.zeros(2 * H)]).astype(np.float32)}
    _, c_keep = impl.lstm_cell(p2, (jnp.asarray(h), jnp.asarray(c)), jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(c_keep), c, atol=1e-5)


def test_gru_cell(impl):
    H = 4
    p = {k: np.asarray(v) for k, v in _params(jax.random.key(7), 3, H, 3).items()}
    h = np.asarray(jax.random.normal(jax.random.key(8), (2, H)))
    x = np.asarray(jax.random.normal(jax.random.key(9), (2, 3)))
    gx, gh = x @ p["Wx"] + p["b"], h @ p["Wh"]
    z = _sig(gx[:, :H] + gh[:, :H])
    r = _sig(gx[:, H:2 * H] + gh[:, H:2 * H])
    n = np.tanh(gx[:, 2 * H:] + r * gh[:, 2 * H:])
    want = (1 - z) * n + z * h
    np.testing.assert_allclose(np.asarray(impl.gru_cell(p, jnp.asarray(h), jnp.asarray(x))), want, atol=1e-5)


def _ref_rnn(p, h, xs):
    out = []
    for x in xs:
        h = np.tanh(x @ p["Wx"] + h @ p["Wh"] + p["b"])
        out.append(h)
    return h, np.stack(out)


def test_run_rnn(impl):
    p = {k: np.asarray(v) for k, v in _params(jax.random.key(10), 3, 5, 1).items()}
    xs = np.asarray(jax.random.normal(jax.random.key(11), (7, 2, 3)))
    h0 = np.zeros((2, 5), np.float32)
    want_h, want_hs = _ref_rnn(p, h0, xs)
    final, hs = impl.run_rnn(impl.rnn_cell, p, jnp.asarray(h0), jnp.asarray(xs))
    assert hs.shape == (7, 2, 5)
    np.testing.assert_allclose(np.asarray(hs), want_hs, atol=1e-5)
    np.testing.assert_allclose(np.asarray(final), want_h, atol=1e-5)
    # tuple carry (LSTM): outputs are h only, final carry is (h, c)
    pl = _params(jax.random.key(12), 3, 5, 4)
    (hT, _cT), hs2 = impl.run_rnn(impl.lstm_cell, pl, (jnp.zeros((2, 5)), jnp.zeros((2, 5))), jnp.asarray(xs))
    assert hs2.shape == (7, 2, 5)
    np.testing.assert_allclose(np.asarray(hs2[-1]), np.asarray(hT), atol=1e-6)


def test_grad_norm_by_distance(impl):
    p = {"Wx": jnp.eye(3, 4), "Wh": 0.3 * jnp.eye(4), "b": jnp.zeros(4)}
    xs = 0.01 * jax.random.normal(jax.random.key(13), (20, 3))
    norms = np.asarray(impl.grad_norm_by_distance(impl.rnn_cell, p, jnp.zeros(4), xs, jnp.ones(4)))
    assert norms.shape == (20,)
    assert np.all(norms >= 0)
    # a contracting recurrence (|Wh| = 0.3): each step back multiplies the gradient by ~0.3
    assert norms[-1] > 0.5
    assert norms[0] < 1e-6
    ratio = norms[:-1] / norms[1:]
    np.testing.assert_allclose(ratio, 0.3, rtol=0.05)
    # matches finite differences for the last input
    eps = 1e-3

    def y(xs_):
        h = jnp.zeros(4)
        for x in xs_:
            h = jnp.tanh(x @ p["Wx"] + h @ p["Wh"])
        return float(jnp.sum(h))

    fd = np.array([(y(xs.at[-1, j].add(eps)) - y(xs.at[-1, j].add(-eps))) / (2 * eps) for j in range(3)])
    np.testing.assert_allclose(norms[-1], np.linalg.norm(fd), rtol=1e-2)


def test_additive_attention(impl):
    B, S, Hq, Hk, Hv, A = 2, 5, 3, 4, 6, 7
    ks = jax.random.split(jax.random.key(14), 6)
    p = {"Wq": jax.random.normal(ks[0], (Hq, A)), "Wk": jax.random.normal(ks[1], (Hk, A)),
         "v": jax.random.normal(ks[2], (A,))}
    q = jax.random.normal(ks[3], (B, Hq))
    k = jax.random.normal(ks[4], (B, S, Hk))
    v = jax.random.normal(ks[5], (B, S, Hv))
    ctx, w = impl.additive_attention(p, q, k, v)
    assert ctx.shape == (B, Hv) and w.shape == (B, S)
    pn = {kk: np.asarray(vv) for kk, vv in p.items()}
    score = np.tanh((np.asarray(q) @ pn["Wq"])[:, None, :] + np.asarray(k) @ pn["Wk"]) @ pn["v"]
    want_w = np.exp(score - score.max(1, keepdims=True))
    want_w /= want_w.sum(1, keepdims=True)
    np.testing.assert_allclose(np.asarray(w), want_w, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ctx), np.einsum("bs,bsh->bh", want_w, np.asarray(v)), atol=1e-5)
    # masking: only position 2 allowed → weight 1 there, context = that value
    mask = jnp.zeros((B, S), bool).at[:, 2].set(True)
    ctx2, w2 = impl.additive_attention(p, q, k, v, mask)
    np.testing.assert_allclose(np.asarray(w2[:, 2]), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(ctx2), np.asarray(v[:, 2]), atol=1e-5)


def test_greedy_decode(impl):
    V = 5

    # A toy decoder whose next token is (token + state) mod V, with state counting steps.
    def step_fn(state, tokens):
        nxt = (tokens + state) % V
        return jax.nn.one_hot(nxt, V) * 10.0, state + 1

    toks = impl.greedy_decode(step_fn, jnp.array([1, 2]), jnp.array([0, 3]), 4)
    assert toks.shape == (2, 4)
    assert toks.dtype == jnp.int32
    # seq 0: start 0, state 1,2,3,4 → 1, 3, 1, 0 ;  seq 1: start 3, state 2,3,4,5 → 0, 3, 2, 2
    np.testing.assert_array_equal(np.asarray(toks), [[1, 3, 1, 0], [0, 3, 2, 2]])
