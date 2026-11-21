"""Module 43 reference solutions. Same signatures as exercises.py."""
import jax
import jax.numpy as jnp


def patchify(images, p: int):
    B, H, W, C = images.shape
    x = images.reshape(B, H // p, p, W // p, p, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, (H // p) * (W // p), p * p * C)


def resize_pos_embed(pos, new_grid: int):
    N, d = pos.shape
    g = round(N ** 0.5)
    return jax.image.resize(pos.reshape(g, g, d), (new_grid, new_grid, d), "bilinear").reshape(-1, d)


def siglip_logits(img, txt, log_t, b):
    img = img / jnp.linalg.norm(img, axis=-1, keepdims=True)
    txt = txt / jnp.linalg.norm(txt, axis=-1, keepdims=True)
    return jnp.exp(log_t) * img @ txt.T + b


def siglip_loss(logits):
    n = logits.shape[0]
    z = 2 * jnp.eye(n) - 1
    return -jnp.sum(jax.nn.log_sigmoid(z * logits)) / n


def pixel_shuffle(x, r: int):
    B, N, d = x.shape
    g = round(N ** 0.5)
    x = x.reshape(B, g // r, r, g // r, r, d).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, (g // r) ** 2, r * r * d)
