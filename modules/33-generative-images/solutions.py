"""Module 33 reference solutions. Try the exercises first — `make test N=33` grades yours."""
import jax
import jax.numpy as jnp


def gan_losses(d_real_logits, d_fake_logits):
    d_loss = jnp.mean(jax.nn.softplus(-d_real_logits)) + jnp.mean(jax.nn.softplus(d_fake_logits))
    g_loss = jnp.mean(jax.nn.softplus(-d_fake_logits))
    return d_loss, g_loss


def reparameterize(key, mu, logvar):
    return mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape, mu.dtype)


def elbo(x, x_recon_logits, mu, logvar):
    axes = tuple(range(1, x.ndim))
    bce = jax.nn.softplus(x_recon_logits) - x * x_recon_logits
    recon = -jnp.sum(bce, axis=axes)
    kl = -0.5 * jnp.sum(1.0 + logvar - mu**2 - jnp.exp(logvar), axis=-1)
    return jnp.mean(recon - kl), jnp.mean(recon), jnp.mean(kl)


def ddpm_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = jnp.linspace(beta_start, beta_end, T, dtype=jnp.float32)
    return betas, jnp.cumprod(1.0 - betas)


def q_sample(x0, t, noise, alphas_cumprod):
    abar = alphas_cumprod[t].reshape((-1,) + (1,) * (x0.ndim - 1))
    return jnp.sqrt(abar) * x0 + jnp.sqrt(1.0 - abar) * noise


def ddpm_loss(eps_pred, eps):
    return jnp.mean((eps_pred - eps) ** 2)


def clip_loss(img_emb, txt_emb, temperature=0.07):
    img = img_emb / jnp.linalg.norm(img_emb, axis=-1, keepdims=True)
    txt = txt_emb / jnp.linalg.norm(txt_emb, axis=-1, keepdims=True)
    logits = img @ txt.T / temperature
    labels = jnp.arange(logits.shape[0])
    i2t = -jnp.mean(jax.nn.log_softmax(logits, axis=1)[labels, labels])
    t2i = -jnp.mean(jax.nn.log_softmax(logits, axis=0)[labels, labels])
    return 0.5 * (i2t + t2i)
