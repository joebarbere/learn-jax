import jax
import jax.numpy as jnp
import numpy as np


def _p(key, F, H, scale=0.5):
    k = jax.random.split(key, 4)
    return {"wx": jax.random.normal(k[0], (F, H)) / np.sqrt(F),
            "wu": jax.random.normal(k[1], (H, H)) * scale / np.sqrt(H),
            "wl": jax.random.normal(k[2], (H, H)) * scale / np.sqrt(H),
            "b": 0.1 * jax.random.normal(k[3], (H,))}


def _loop(p, x):  # the recurrence read literally, in NumPy
    p = {k: np.asarray(v, np.float64) for k, v in p.items()}
    x = np.asarray(x, np.float64)
    B, R, C, _ = x.shape
    H = p["wu"].shape[0]
    h = np.zeros((B, R, C, H))
    for i in range(R):
        for j in range(C):
            up = h[:, i - 1, j] if i else np.zeros((B, H))
            left = h[:, i, j - 1] if j else np.zeros((B, H))
            h[:, i, j] = np.tanh(x[:, i, j] @ p["wx"] + up @ p["wu"] + left @ p["wl"] + p["b"])
    return h


def test_mdrnn_sweep_naive(impl):
    p = _p(jax.random.key(0), 3, 5)
    x = jax.random.normal(jax.random.key(1), (2, 4, 6, 3))
    h = impl.mdrnn_sweep_naive(p, x)
    assert h.shape == (2, 4, 6, 5)
    np.testing.assert_allclose(np.asarray(h), _loop(p, x), atol=1e-5)
    # the top-left cell sees only its own input: no neighbours yet
    np.testing.assert_allclose(np.asarray(h[:, 0, 0]), np.tanh(np.asarray(x[:, 0, 0] @ p["wx"] + p["b"])), atol=1e-6)
    # a 1x1 grid works
    assert impl.mdrnn_sweep_naive(p, x[:, :1, :1]).shape == (2, 1, 1, 5)


def test_skew_unskew(impl):
    x = jnp.arange(2 * 3 * 4 * 1, dtype=jnp.float32).reshape(2, 3, 4, 1) + 1
    s = impl.skew(x)
    assert s.shape == (2, 3, 6, 1)
    for i in range(3):
        for j in range(4):
            assert float(s[1, i, i + j, 0]) == float(x[1, i, j, 0])
    assert int((np.asarray(s) != 0).sum()) == x.size          # everything else is padding zeros
    np.testing.assert_array_equal(np.asarray(impl.unskew(s, 4)), np.asarray(x))
    # one row: nothing moves; one column: a diagonal
    np.testing.assert_array_equal(np.asarray(impl.skew(x[:, :1])), np.asarray(x[:, :1]))
    assert impl.skew(x[:, :, :1]).shape == (2, 3, 3, 1)


def test_mdrnn_sweep_matches_naive(impl):
    p = _p(jax.random.key(2), 3, 4)
    for shape in [(2, 5, 7, 3), (1, 6, 2, 3), (3, 1, 4, 3), (2, 4, 1, 3)]:
        x = jax.random.normal(jax.random.key(3), shape)
        np.testing.assert_allclose(np.asarray(impl.mdrnn_sweep(p, x)), _loop(p, x), atol=2e-5)
    # padding must stay zero: with a large bias, leaked tanh(b) would change later cells
    pb = dict(p, b=jnp.full((4,), 3.0))
    x = jax.random.normal(jax.random.key(4), (2, 4, 4, 3))
    np.testing.assert_allclose(np.asarray(impl.mdrnn_sweep(pb, x)), _loop(pb, x), atol=2e-5)
    # jit and grad work
    g = jax.grad(lambda p: jax.jit(impl.mdrnn_sweep)(p, x).sum())(p)
    assert all(np.isfinite(np.asarray(v)).all() for v in g.values())


def test_four_direction_sweeps(impl):
    ps = [_p(jax.random.key(10 + k), 2, 3) for k in range(4)]
    x = jax.random.normal(jax.random.key(5), (2, 4, 5, 2))
    out = impl.four_direction_sweeps(ps, x, lambda p, v: jnp.asarray(_loop(p, v), jnp.float32))
    assert out.shape == (2, 4, 5, 12)
    np.testing.assert_allclose(np.asarray(out[..., :3]), _loop(ps[0], x), atol=1e-5)
    # bottom-left corner: flip rows, sweep, flip back
    np.testing.assert_allclose(np.asarray(out[..., 3:6]), _loop(ps[1], x[:, ::-1])[:, ::-1], atol=1e-5)
    np.testing.assert_allclose(np.asarray(out[..., 6:9]), _loop(ps[2], x[:, :, ::-1])[:, :, ::-1], atol=1e-5)
    np.testing.assert_allclose(np.asarray(out[..., 9:]), _loop(ps[3], x[:, ::-1, ::-1])[:, ::-1, ::-1], atol=1e-5)
    # tied: one set of weights for all four
    tied = impl.four_direction_sweeps(ps[:1], x, lambda p, v: jnp.asarray(_loop(p, v), jnp.float32))
    np.testing.assert_allclose(np.asarray(tied[..., 3:6]), _loop(ps[0], x[:, ::-1])[:, ::-1], atol=1e-5)


def test_cross_scan_merge(impl):
    x = jnp.arange(2 * 3 * 4 * 2, dtype=jnp.float32).reshape(2, 3, 4, 2)
    s = impl.cross_scan(x)
    assert s.shape == (4, 2, 12, 2)
    np.testing.assert_array_equal(np.asarray(s[0, 0, :4, 0]), np.asarray(x[0, 0, :, 0]))      # row 0
    np.testing.assert_array_equal(np.asarray(s[1, 0, :3, 0]), np.asarray(x[0, :, 0, 0]))      # column 0
    np.testing.assert_array_equal(np.asarray(s[2]), np.asarray(s[0][:, ::-1]))
    np.testing.assert_array_equal(np.asarray(s[3]), np.asarray(s[1][:, ::-1]))
    np.testing.assert_array_equal(np.asarray(impl.cross_merge(s, 3, 4)), 4 * np.asarray(x))
    # merge puts a per-order value back where it came from
    ys = jnp.zeros((4, 1, 12, 1)).at[1, 0, 1, 0].set(1.0)       # 2nd cell of the column-major order = (1, 0)
    m = impl.cross_merge(ys, 3, 4)
    assert float(m[0, 1, 0, 0]) == 1.0 and float(m.sum()) == 1.0
