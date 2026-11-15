import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest


def test_gan_losses(impl):
    real = jnp.array([2.0, -1.0, 0.5])
    fake = jnp.array([-3.0, 0.0])
    d, g = impl.gan_losses(real, fake)
    sig = jax.nn.sigmoid
    d_ref = -jnp.mean(jnp.log(sig(real))) - jnp.mean(jnp.log(1 - sig(fake)))
    g_ref = -jnp.mean(jnp.log(sig(fake)))
    np.testing.assert_allclose(d, d_ref, rtol=1e-5)
    np.testing.assert_allclose(g, g_ref, rtol=1e-5)
    # stable for extreme logits (no inf / nan)
    d2, g2 = impl.gan_losses(jnp.array([100.0, -100.0]), jnp.array([-100.0, 100.0]))
    assert np.isfinite(float(d2)) and np.isfinite(float(g2))
    # the non-saturating G loss still has a strong gradient when D confidently rejects fakes
    grad = jax.grad(lambda f: impl.gan_losses(jnp.zeros(1), f)[1])(jnp.array([-8.0]))
    assert abs(float(grad[0])) > 0.9


def test_reparameterize(impl):
    key = jax.random.key(0)
    mu = jnp.full((20000, 2), 3.0)
    logvar = jnp.full((20000, 2), jnp.log(4.0))
    z = impl.reparameterize(key, mu, logvar)
    assert z.shape == mu.shape
    np.testing.assert_allclose(z.mean(0), [3.0, 3.0], atol=0.05)
    np.testing.assert_allclose(z.std(0), [2.0, 2.0], atol=0.05)
    # gradients flow to mu and logvar
    g = jax.grad(lambda m, lv: impl.reparameterize(key, m, lv).sum(), argnums=(0, 1))(mu[:3], logvar[:3])
    np.testing.assert_allclose(g[0], 1.0)
    assert float(jnp.abs(g[1]).sum()) > 0


def test_elbo(impl):
    key = jax.random.key(1)
    x = jax.random.uniform(key, (4, 6, 6, 1))
    logits = jax.random.normal(jax.random.key(2), x.shape)
    mu = jax.random.normal(jax.random.key(3), (4, 3))
    logvar = 0.3 * jax.random.normal(jax.random.key(4), (4, 3))
    e, recon, kl = impl.elbo(x, logits, mu, logvar)
    recon_ref = -optax.sigmoid_binary_cross_entropy(logits, x).sum(axis=(1, 2, 3)).mean()
    kl_ref = (-0.5 * (1 + logvar - mu**2 - jnp.exp(logvar)).sum(-1)).mean()
    np.testing.assert_allclose(recon, recon_ref, rtol=1e-5)
    np.testing.assert_allclose(kl, kl_ref, rtol=1e-5)
    np.testing.assert_allclose(e, recon_ref - kl_ref, rtol=1e-5)
    # KL is zero exactly at the prior
    _, _, kl0 = impl.elbo(x, logits, jnp.zeros((4, 3)), jnp.zeros((4, 3)))
    assert abs(float(kl0)) < 1e-6


def test_ddpm_schedule(impl):
    betas, abar = impl.ddpm_schedule(1000)
    assert betas.shape == (1000,) and abar.shape == (1000,)
    np.testing.assert_allclose([betas[0], betas[-1]], [1e-4, 0.02], rtol=1e-5)
    np.testing.assert_allclose(abar[0], 1 - 1e-4, rtol=1e-6)
    assert bool(jnp.all(jnp.diff(abar) < 0))
    assert float(abar[-1]) < 1e-4  # almost pure noise at the end


def test_q_sample(impl):
    _, abar = impl.ddpm_schedule(100)
    x0 = jnp.ones((3, 4, 4, 1))
    noise = -jnp.ones((3, 4, 4, 1))
    t = jnp.array([0, 50, 99])
    xt = impl.q_sample(x0, t, noise, abar)
    for i, ti in enumerate([0, 50, 99]):
        expect = jnp.sqrt(abar[ti]) - jnp.sqrt(1 - abar[ti])
        np.testing.assert_allclose(xt[i], expect, rtol=1e-5)
    # variance preserving: x0 ~ N(0,1), noise ~ N(0,1) -> x_t ~ N(0,1)
    k1, k2 = jax.random.split(jax.random.key(5))
    big = impl.q_sample(jax.random.normal(k1, (4000, 1, 1, 1)), jnp.full((4000,), 60),
                        jax.random.normal(k2, (4000, 1, 1, 1)), abar)
    assert abs(float(big.std()) - 1.0) < 0.05


def test_ddpm_loss(impl):
    assert float(impl.ddpm_loss(jnp.zeros((2, 3)), jnp.ones((2, 3)))) == pytest.approx(1.0)
    assert float(impl.ddpm_loss(jnp.full((4,), 2.0), jnp.zeros(4))) == pytest.approx(4.0)


def test_clip_loss(impl):
    # perfectly aligned pairs and a low temperature -> loss near zero
    e = jnp.eye(4)
    assert float(impl.clip_loss(e, e, 0.01)) < 1e-3
    # norms don't matter (embeddings are normalized)
    assert float(impl.clip_loss(5 * e, 0.2 * e, 0.01)) < 1e-3
    # random embeddings at temperature 1 match a reference implementation
    a = jax.random.normal(jax.random.key(6), (8, 5))
    b = jax.random.normal(jax.random.key(7), (8, 5))
    an = a / jnp.linalg.norm(a, axis=1, keepdims=True)
    bn = b / jnp.linalg.norm(b, axis=1, keepdims=True)
    logits = an @ bn.T / 0.5
    lab = jnp.arange(8)
    ref = 0.5 * (optax.softmax_cross_entropy_with_integer_labels(logits, lab).mean()
                 + optax.softmax_cross_entropy_with_integer_labels(logits.T, lab).mean())
    np.testing.assert_allclose(impl.clip_loss(a, b, 0.5), ref, rtol=1e-5)
    # swapped pairs are worse than matched ones
    assert float(impl.clip_loss(e, e[::-1], 0.1)) > float(impl.clip_loss(e, e, 0.1))
