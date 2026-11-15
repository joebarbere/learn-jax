import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest


def _params():
    return {"w": jnp.array([[1.0, -2.0], [0.5, 3.0]]), "layers": [jnp.array([0.1, -0.2, 0.3]), jnp.array(2.0)]}


def _grads(i):
    k = jax.random.key(i)
    p = _params()
    leaves, tree = jax.tree.flatten(p)
    keys = jax.random.split(k, len(leaves))
    return jax.tree.unflatten(tree, [jax.random.normal(kk, leaf.shape) * (i + 1) for kk, leaf in zip(keys, leaves)])


def _close(a, b, **kw):
    jax.tree.map(lambda x, y: np.testing.assert_allclose(np.asarray(x), np.asarray(y), **kw), a, b)


def test_sgd_step(impl):
    p, g = _params(), _grads(0)
    out = impl.sgd_step(p, g, 0.1)
    _close(out, jax.tree.map(lambda a, b: a - 0.1 * b, p, g), rtol=1e-6)
    assert jax.tree.structure(out) == jax.tree.structure(p)


def test_momentum_matches_optax(impl):
    p_mine = p_ref = _params()
    vel = jax.tree.map(jnp.zeros_like, p_mine)
    opt = optax.sgd(0.05, momentum=0.9)
    state = opt.init(p_ref)
    for i in range(5):
        g = _grads(i)
        p_mine, vel = impl.momentum_step(p_mine, g, vel, 0.05, 0.9)
        upd, state = opt.update(g, state, p_ref)
        p_ref = optax.apply_updates(p_ref, upd)
    _close(p_mine, p_ref, rtol=1e-5, atol=1e-6)


def test_adam_matches_optax(impl):
    p_mine = p_ref = _params()
    zeros = jax.tree.map(jnp.zeros_like, p_mine)
    st = (zeros, zeros, 0)
    opt = optax.adam(1e-2, b1=0.9, b2=0.999, eps=1e-8)
    state = opt.init(p_ref)
    for i in range(6):
        g = _grads(i)
        p_mine, st = impl.adam_step(p_mine, g, st, 1e-2, 0.9, 0.999, 1e-8)
        upd, state = opt.update(g, state, p_ref)
        p_ref = optax.apply_updates(p_ref, upd)
    _close(p_mine, p_ref, rtol=1e-5, atol=1e-6)
    assert int(st[2]) == 6


def test_adam_first_step_is_lr_sized(impl):
    # Bias correction makes the very first Adam step ~lr * sign(g), whatever the gradient's scale.
    p = {"a": jnp.array([0.0, 0.0])}
    g = {"a": jnp.array([1000.0, -0.001])}
    zeros = jax.tree.map(jnp.zeros_like, p)
    out, _ = impl.adam_step(p, g, (zeros, zeros, 0), lr=0.1)
    np.testing.assert_allclose(np.asarray(out["a"]), [-0.1, 0.1], rtol=1e-3)


@pytest.mark.parametrize("step", [0, 1, 100, 199, 200, 201, 6000, 11999, 12000, 15000])
def test_warmup_cosine_matches_optax(impl, step):
    sched = optax.warmup_cosine_decay_schedule(0.0, 6e-4, 200, 12000, 6e-5)
    got = impl.warmup_cosine(step, 6e-4, 200, 12000, 6e-5)
    assert got == pytest.approx(float(sched(step)), rel=1e-5, abs=1e-12)


def test_clip_by_global_norm(impl):
    g = {"a": jnp.array([3.0, 0.0]), "b": [jnp.array([[4.0]])]}  # global norm 5
    big = impl.clip_by_global_norm(g, 1.0)
    norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(big)))
    assert float(norm) == pytest.approx(1.0, rel=1e-6)
    _close(big, jax.tree.map(lambda x: x / 5.0, g), rtol=1e-6)  # direction kept
    _close(impl.clip_by_global_norm(g, 10.0), g, rtol=1e-6)  # under the limit: unchanged
    zero = impl.clip_by_global_norm({"a": jnp.zeros(3)}, 1.0)
    assert not np.isnan(np.asarray(zero["a"])).any()


def test_fit_recovers_line(impl):
    x = jnp.linspace(-1, 1, 64)
    data = (x, 2.0 * x - 1.0)

    def loss(p, d):
        xs, ys = d
        return jnp.mean((p["w"] * xs + p["b"] - ys) ** 2)

    out = impl.fit(loss, {"w": jnp.array(0.0), "b": jnp.array(0.0)}, data, steps=300, lr=0.3)
    assert float(out["w"]) == pytest.approx(2.0, abs=1e-2)
    assert float(out["b"]) == pytest.approx(-1.0, abs=1e-2)


def _polar(G):
    u, _, vt = np.linalg.svd(np.asarray(G, np.float64), full_matrices=False)
    return u @ vt


@pytest.mark.parametrize("shape", [(64, 32), (32, 96), (48, 48)])
def test_newton_schulz_orthogonalizes(impl, shape):
    key = jax.random.key(sum(shape))
    # a gradient-like matrix: a few dominant directions plus noise
    k1, k2, k3 = jax.random.split(key, 3)
    low = jax.random.normal(k1, (shape[0], 4)) @ jax.random.normal(k2, (4, shape[1]))
    G = 5 * low + 0.3 * jax.random.normal(k3, shape)
    X = np.asarray(impl.newton_schulz(G), np.float64)
    assert X.shape == shape
    # In G's own singular basis, X ≈ U·f(S)·Vᵀ: read off f(s_i) and check it lands near 1 for every
    # direction that wasn't negligible to begin with (5 steps can't lift a ~0 singular value).
    u, sg, vt = np.linalg.svd(np.asarray(G, np.float64), full_matrices=False)
    f = np.diag(u.T @ X @ vt.T)
    keep = sg / sg.max() > 1e-2
    assert f[keep].min() > 0.6 and f[keep].max() < 1.25, f
    assert np.linalg.svd(X, compute_uv=False).max() < 1.25
    ref = _polar(G)
    cos = (X * ref).sum() / (np.linalg.norm(X) * np.linalg.norm(ref))
    assert cos > 0.9


def test_newton_schulz_shrinks_spread(impl):
    G = jnp.diag(jnp.array([100.0, 10.0, 1.0, 0.1])) @ jnp.eye(4, 6)
    s_before = np.linalg.svd(np.asarray(G), compute_uv=False)
    s_after = np.linalg.svd(np.asarray(impl.newton_schulz(G)), compute_uv=False)
    assert s_before.max() / s_before.min() == pytest.approx(1000.0)
    assert s_after.max() / s_after.min() < 10  # a 1000× spread collapsed


@pytest.mark.parametrize("shape,nesterov", [((24, 40), True), ((40, 24), True), ((32, 32), False)])
def test_muon_step_matches_optax(impl, shape, nesterov):
    lr, beta = 0.02, 0.95
    W = jax.random.normal(jax.random.key(1), shape) * 0.1
    opt = optax.contrib.muon(learning_rate=lr, beta=beta, nesterov=nesterov)
    p_ref, s_ref = {"w": W}, opt.init({"w": W})
    p, mom = W, (jnp.zeros_like(W), 0)
    for i in range(5):
        g = jax.random.normal(jax.random.key(10 + i), shape)
        u, s_ref = opt.update({"w": g}, s_ref, p_ref)
        p_ref = optax.apply_updates(p_ref, u)
        p, mom = impl.muon_step(p, g, mom, lr, beta, nesterov=nesterov)
    np.testing.assert_allclose(np.asarray(p), np.asarray(p_ref["w"]), atol=2e-5)
    assert int(mom[1]) == 5
