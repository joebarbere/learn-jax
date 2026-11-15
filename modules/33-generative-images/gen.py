"""Module 33 helpers: four small generative models on Fashion-MNIST, in Flax NNX (as in Module 12).

The lesson passes in the loss functions (yours from exercises.py, or the reference ones), so the
training below runs on the learner's code. `train.py` reuses the diffusion parts for a longer run.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from learnjax.data import FASHION_CLASSES, fashion_mnist

RUNS = Path("/course/runs/33-generative-images")


# --------------------------------------------------------------------------- data

def load_data():
    """Fashion-MNIST as float32 NHWC in [-1, 1] (GAN/diffusion/CLIP) plus labels."""
    d = fashion_mnist()
    tx = (d["train_images"].astype(np.float32) / 127.5 - 1.0)[..., None]
    vx = (d["test_images"].astype(np.float32) / 127.5 - 1.0)[..., None]
    return tx, d["train_labels"], vx, d["test_labels"]


def batches(rng, n, batch):
    perm = rng.permutation(n)
    for i in range(0, n - batch + 1, batch):
        yield perm[i:i + batch]


def grid(images, nrow=8):
    """(N, 28, 28, 1) in [-1, 1] -> one (nrow*28, ncol*28) image in [0, 1] for imshow."""
    x = np.clip((np.asarray(images)[..., 0] + 1) / 2, 0, 1)
    n = x.shape[0]
    ncol = math.ceil(n / nrow)
    out = np.ones((ncol * 30, nrow * 30))
    for i in range(n):
        r, c = divmod(i, nrow)
        out[r * 30 + 1:r * 30 + 29, c * 30 + 1:c * 30 + 29] = x[i]
    return out


# --------------------------------------------------------------------------- a quick judge

class Classifier(nnx.Module):
    """A small CNN (Module 12's design) used to *judge* generated samples by class."""

    def __init__(self, rngs: nnx.Rngs):
        self.c1 = nnx.Conv(1, 32, (3, 3), rngs=rngs)
        self.c2 = nnx.Conv(32, 64, (3, 3), rngs=rngs)
        self.f1 = nnx.Linear(7 * 7 * 64, 128, rngs=rngs)
        self.f2 = nnx.Linear(128, 10, rngs=rngs)

    def __call__(self, x):
        pool = lambda h: jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
        x = pool(nnx.relu(self.c1(x)))
        x = pool(nnx.relu(self.c2(x)))
        return self.f2(nnx.relu(self.f1(x.reshape(x.shape[0], -1))))


def train_classifier(tx, ty, vx, vy, epochs=2, batch=256, seed=0):
    m = Classifier(nnx.Rngs(seed))
    opt = nnx.Optimizer(m, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def step(m, opt, x, y):
        loss, g = nnx.value_and_grad(lambda m: optax.softmax_cross_entropy_with_integer_labels(m(x), y).mean())(m)
        opt.update(m, g)
        return loss

    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for b in batches(rng, len(tx), batch):
            step(m, opt, tx[b], ty[b])
    acc = float(np.mean(np.concatenate([np.asarray(predict(m, vx[i:i + 2000])) for i in range(0, len(vx), 2000)]) == vy))
    return m, acc


@nnx.jit
def predict(m, x):
    return m(x).argmax(-1)


# --------------------------------------------------------------------------- (a) DCGAN

class Generator(nnx.Module):
    def __init__(self, z_dim: int, rngs: nnx.Rngs):
        self.fc = nnx.Linear(z_dim, 7 * 7 * 128, rngs=rngs)
        self.n0 = nnx.GroupNorm(128, num_groups=8, rngs=rngs)
        self.up1 = nnx.ConvTranspose(128, 64, (4, 4), strides=(2, 2), rngs=rngs)   # 7 -> 14
        self.n1 = nnx.GroupNorm(64, num_groups=8, rngs=rngs)
        self.up2 = nnx.ConvTranspose(64, 1, (4, 4), strides=(2, 2), rngs=rngs)     # 14 -> 28

    def __call__(self, z):
        h = nnx.relu(self.n0(self.fc(z).reshape(z.shape[0], 7, 7, 128)))
        h = nnx.relu(self.n1(self.up1(h)))
        return jnp.tanh(self.up2(h))


class Discriminator(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.c1 = nnx.Conv(1, 64, (4, 4), strides=(2, 2), rngs=rngs)    # 28 -> 14
        self.c2 = nnx.Conv(64, 128, (4, 4), strides=(2, 2), rngs=rngs)  # 14 -> 7
        self.fc = nnx.Linear(7 * 7 * 128, 1, rngs=rngs)

    def __call__(self, x):
        h = nnx.leaky_relu(self.c1(x), 0.2)
        h = nnx.leaky_relu(self.c2(h), 0.2)
        return self.fc(h.reshape(x.shape[0], -1))[:, 0]


def train_gan(tx, gan_losses, steps=3000, batch=128, z_dim=64, seed=0, lr=2e-4):
    G, D = Generator(z_dim, nnx.Rngs(seed)), Discriminator(nnx.Rngs(seed + 1))
    oG = nnx.Optimizer(G, optax.adam(lr, b1=0.5), wrt=nnx.Param)
    oD = nnx.Optimizer(D, optax.adam(lr, b1=0.5), wrt=nnx.Param)

    @nnx.jit
    def step(G, D, oG, oD, x, z):
        def d_loss(D):
            return gan_losses(D(x), D(jax.lax.stop_gradient(G(z))))[0]

        ld, gd = nnx.value_and_grad(d_loss)(D)
        oD.update(D, gd)

        def g_loss(G):
            return gan_losses(D(x), D(G(z)))[1]

        lg, gg = nnx.value_and_grad(g_loss)(G)
        oG.update(G, gg)
        return ld, lg

    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    hist, t0, s = [], time.perf_counter(), 0
    while s < steps:
        for b in batches(rng, len(tx), batch):
            key, k = jax.random.split(key)
            ld, lg = step(G, D, oG, oD, tx[b], jax.random.normal(k, (batch, z_dim)))
            if s % 100 == 0:
                hist.append((s, float(ld), float(lg)))
            s += 1
            if s >= steps:
                break
    jax.block_until_ready(ld)
    return G, D, hist, time.perf_counter() - t0


# --------------------------------------------------------------------------- (b) VAE

class VAE(nnx.Module):
    def __init__(self, latent: int, rngs: nnx.Rngs):
        self.e1 = nnx.Conv(1, 32, (3, 3), strides=(2, 2), rngs=rngs)     # 28 -> 14
        self.e2 = nnx.Conv(32, 64, (3, 3), strides=(2, 2), rngs=rngs)    # 14 -> 7
        self.e3 = nnx.Linear(7 * 7 * 64, 256, rngs=rngs)
        self.head = nnx.Linear(256, 2 * latent, rngs=rngs)
        self.d1 = nnx.Linear(latent, 256, rngs=rngs)
        self.d2 = nnx.Linear(256, 7 * 7 * 64, rngs=rngs)
        self.u1 = nnx.ConvTranspose(64, 32, (3, 3), strides=(2, 2), rngs=rngs)
        self.u2 = nnx.ConvTranspose(32, 1, (3, 3), strides=(2, 2), rngs=rngs)
        self.latent = latent

    def encode(self, x):
        h = nnx.relu(self.e2(nnx.relu(self.e1(x))))
        h = nnx.relu(self.e3(h.reshape(x.shape[0], -1)))
        mu, logvar = jnp.split(self.head(h), 2, axis=-1)
        return mu, logvar

    def decode(self, z):
        h = nnx.relu(self.d2(nnx.relu(self.d1(z)))).reshape(z.shape[0], 7, 7, 64)
        return self.u2(nnx.relu(self.u1(h)))  # logits


def train_vae(tx, elbo, reparameterize, latent=2, steps=3000, batch=128, seed=0):
    """VAE pixels live in [0, 1] (a Bernoulli decoder), so rescale from [-1, 1]."""
    m = VAE(latent, nnx.Rngs(seed))
    opt = nnx.Optimizer(m, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def step(m, opt, x, key):
        def loss(m):
            mu, logvar = m.encode(x)
            z = reparameterize(key, mu, logvar)
            e, recon, kl = elbo(x, m.decode(z), mu, logvar)
            return -e, (recon, kl)

        (l, (recon, kl)), g = nnx.value_and_grad(loss, has_aux=True)(m)
        opt.update(m, g)
        return l, recon, kl

    x01 = (tx + 1) / 2
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    hist, t0, s = [], time.perf_counter(), 0
    while s < steps:
        for b in batches(rng, len(x01), batch):
            key, k = jax.random.split(key)
            l, recon, kl = step(m, opt, x01[b], k)
            if s % 100 == 0:
                hist.append((s, -float(l), float(recon), float(kl)))
            s += 1
            if s >= steps:
                break
    jax.block_until_ready(l)
    return m, hist, time.perf_counter() - t0


# --------------------------------------------------------------------------- (c) DDPM

def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half) / half)
    ang = t.astype(jnp.float32)[:, None] * freqs[None]
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


class ResBlock(nnx.Module):
    def __init__(self, cin, cout, emb, rngs):
        self.n1 = nnx.GroupNorm(cin, num_groups=8, rngs=rngs)
        self.c1 = nnx.Conv(cin, cout, (3, 3), rngs=rngs)
        self.e = nnx.Linear(emb, cout, rngs=rngs)
        self.n2 = nnx.GroupNorm(cout, num_groups=8, rngs=rngs)
        self.c2 = nnx.Conv(cout, cout, (3, 3), rngs=rngs)
        self.skip = nnx.Conv(cin, cout, (1, 1), rngs=rngs) if cin != cout else None

    def __call__(self, x, emb):
        h = self.c1(nnx.silu(self.n1(x)))
        h = h + self.e(emb)[:, None, None, :]
        h = self.c2(nnx.silu(self.n2(h)))
        return h + (self.skip(x) if self.skip is not None else x)


class UNet(nnx.Module):
    """A small U-Net: 28 -> 14 -> 7 and back, with skip connections, time and (optional) class input.

    Class 10 means "no class" — used for classifier-free guidance.
    """

    def __init__(self, rngs: nnx.Rngs, ch=32, emb=128):
        self.t1 = nnx.Linear(64, emb, rngs=rngs)
        self.t2 = nnx.Linear(emb, emb, rngs=rngs)
        self.cls = nnx.Embed(11, emb, rngs=rngs)
        self.inp = nnx.Conv(1, ch, (3, 3), rngs=rngs)
        self.d1 = ResBlock(ch, ch, emb, rngs)
        self.down1 = nnx.Conv(ch, 2 * ch, (3, 3), strides=(2, 2), rngs=rngs)
        self.d2 = ResBlock(2 * ch, 2 * ch, emb, rngs)
        self.down2 = nnx.Conv(2 * ch, 4 * ch, (3, 3), strides=(2, 2), rngs=rngs)
        self.mid = ResBlock(4 * ch, 4 * ch, emb, rngs)
        self.up2 = nnx.ConvTranspose(4 * ch, 2 * ch, (4, 4), strides=(2, 2), rngs=rngs)
        self.u2 = ResBlock(4 * ch, 2 * ch, emb, rngs)
        self.up1 = nnx.ConvTranspose(2 * ch, ch, (4, 4), strides=(2, 2), rngs=rngs)
        self.u1 = ResBlock(2 * ch, ch, emb, rngs)
        self.out_n = nnx.GroupNorm(ch, num_groups=8, rngs=rngs)
        self.out = nnx.Conv(ch, 1, (3, 3), rngs=rngs)

    def __call__(self, x, t, y):
        emb = self.t2(nnx.silu(self.t1(timestep_embedding(t)))) + self.cls(y)
        h1 = self.d1(self.inp(x), emb)                    # 28
        h2 = self.d2(self.down1(h1), emb)                 # 14
        h = self.mid(self.down2(h2), emb)                 # 7
        h = self.u2(jnp.concatenate([self.up2(h), h2], -1), emb)
        h = self.u1(jnp.concatenate([self.up1(h), h1], -1), emb)
        return self.out(nnx.silu(self.out_n(h)))


def make_ddpm_step(q_sample, ddpm_loss, abar, T, p_uncond=0.1):
    @nnx.jit
    def step(m, opt, x0, y, key):
        kt, kn, kd = jax.random.split(key, 3)
        t = jax.random.randint(kt, (x0.shape[0],), 0, T)
        eps = jax.random.normal(kn, x0.shape)
        y = jnp.where(jax.random.uniform(kd, y.shape) < p_uncond, 10, y)   # drop the label sometimes (CFG)

        def loss(m):
            return ddpm_loss(m(q_sample(x0, t, eps, abar), t, y), eps)

        l, g = nnx.value_and_grad(loss)(m)
        opt.update(m, g)
        return l

    return step


def train_ddpm(tx, ty, q_sample, ddpm_loss, abar, T, steps=1500, batch=128, seed=0, lr=2e-4,
               model=None, log_every=100, on_log=None):
    m = model if model is not None else UNet(nnx.Rngs(seed))
    opt = nnx.Optimizer(m, optax.adam(lr), wrt=nnx.Param)
    step = make_ddpm_step(q_sample, ddpm_loss, abar, T)
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed + 100)
    hist, t0, s = [], time.perf_counter(), 0
    while s < steps:
        for b in batches(rng, len(tx), batch):
            key, k = jax.random.split(key)
            l = step(m, opt, tx[b], ty[b], k)
            if s % log_every == 0:
                hist.append((s, float(l)))
                if on_log:
                    on_log(s, float(l), time.perf_counter() - t0)
            s += 1
            if s >= steps:
                break
    jax.block_until_ready(l)
    return m, hist, time.perf_counter() - t0


def ddim_sampler(m, abar, T, n_steps=50):
    """Deterministic DDIM (Song et al., 2020) with classifier-free guidance, as one jitted scan."""
    graphdef, state = nnx.split(m)
    ts = jnp.linspace(T - 1, 0, n_steps).round().astype(jnp.int32)
    t_prev = jnp.concatenate([ts[1:], jnp.array([-1])])

    @jax.jit
    def sample(state, key, y, w):
        model = nnx.merge(graphdef, state)
        x = jax.random.normal(key, (y.shape[0], 28, 28, 1))

        def body(x, tt):
            t, tp = tt
            tb = jnp.full((y.shape[0],), t)
            eps_c = model(x, tb, y)
            eps_u = model(x, tb, jnp.full_like(y, 10))
            eps = (1 + w) * eps_c - w * eps_u
            a, ap = abar[t], jnp.where(tp >= 0, abar[jnp.maximum(tp, 0)], 1.0)
            x0 = jnp.clip((x - jnp.sqrt(1 - a) * eps) / jnp.sqrt(a), -1, 1)
            eps = (x - jnp.sqrt(a) * x0) / jnp.sqrt(1 - a)   # re-derive the noise from the clipped x0
            return jnp.sqrt(ap) * x0 + jnp.sqrt(1 - ap) * eps, None

        x, _ = jax.lax.scan(body, x, (ts, t_prev))
        return x

    return lambda key, y, w=0.0: sample(state, key, y, jnp.float32(w))


def ancestral_sampler(m, betas, abar, T):
    """Ho et al.'s original sampler: all T steps, adding fresh noise at each one (unconditional)."""
    graphdef, state = nnx.split(m)
    alphas = 1.0 - betas

    @jax.jit
    def sample(state, key, n):
        model = nnx.merge(graphdef, state)
        k0, key = jax.random.split(key)
        x = jax.random.normal(k0, (n.shape[0], 28, 28, 1))

        def body(carry, t):
            x, key = carry
            key, k = jax.random.split(key)
            eps = model(x, jnp.full((n.shape[0],), t), jnp.full((n.shape[0],), 10))
            mean = (x - betas[t] / jnp.sqrt(1 - abar[t]) * eps) / jnp.sqrt(alphas[t])
            noise = jnp.where(t > 0, jax.random.normal(k, x.shape), 0.0)
            return (mean + jnp.sqrt(betas[t]) * noise, key), None

        (x, _), _ = jax.lax.scan(body, (x, key), jnp.arange(T - 1, -1, -1))
        return x

    return lambda key, n: sample(state, key, jnp.zeros((n,)))


def save_model(m, path):
    leaves = jax.tree.leaves(nnx.state(m))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, *[np.asarray(x) for x in leaves])


def load_model(m, path):
    graphdef, state = nnx.split(m)
    data = np.load(path)
    leaves = [jnp.asarray(data[f"arr_{i}"]) for i in range(len(data.files))]
    state = jax.tree.unflatten(jax.tree.structure(state), leaves)
    return nnx.merge(graphdef, state)


# --------------------------------------------------------------------------- (d) mini-CLIP

TRAIN_TEMPLATES = ["a photo of a {}", "a picture of a {}", "an image of the {}", "a {}",
                   "a small {}", "a grayscale photo of a {}"]
UNSEEN_TEMPLATE = "the {} in this snapshot"


def class_words(i):
    return FASHION_CLASSES[i].lower().replace("/", " ").replace("-", " ")


def build_vocab():
    words = set()
    for tpl in TRAIN_TEMPLATES + [UNSEEN_TEMPLATE]:
        for i in range(10):
            words.update(tpl.format(class_words(i)).split())
    return {w: i + 1 for i, w in enumerate(sorted(words))}   # 0 = padding


def tokenize(texts, vocab, length=8):
    out = np.zeros((len(texts), length), np.int32)
    for r, t in enumerate(texts):
        ids = [vocab[w] for w in t.split()][:length]
        out[r, :len(ids)] = ids
    return out


class ImageEncoder(nnx.Module):
    def __init__(self, dim, rngs):
        self.c1 = nnx.Conv(1, 32, (3, 3), strides=(2, 2), rngs=rngs)
        self.c2 = nnx.Conv(32, 64, (3, 3), strides=(2, 2), rngs=rngs)
        self.f = nnx.Linear(7 * 7 * 64, dim, rngs=rngs)

    def __call__(self, x):
        h = nnx.relu(self.c2(nnx.relu(self.c1(x))))
        return self.f(h.reshape(x.shape[0], -1))


class TextEncoder(nnx.Module):
    """Mean of word embeddings -> MLP. (Real CLIP uses a Transformer; captions here are 2-6 words.)"""

    def __init__(self, vocab_size, dim, rngs):
        self.emb = nnx.Embed(vocab_size + 1, 64, rngs=rngs)
        self.f1 = nnx.Linear(64, 128, rngs=rngs)
        self.f2 = nnx.Linear(128, dim, rngs=rngs)

    def __call__(self, ids):
        mask = (ids > 0)[..., None].astype(jnp.float32)
        h = (self.emb(ids) * mask).sum(1) / jnp.maximum(mask.sum(1), 1.0)
        return self.f2(nnx.relu(self.f1(h)))


def train_clip(tx, ty, clip_loss, steps=1500, batch=256, dim=64, temperature=0.1, seed=0):
    vocab = build_vocab()
    caps = {(t, c): tokenize([tpl.format(class_words(c))], vocab)[0]
            for t, tpl in enumerate(TRAIN_TEMPLATES) for c in range(10)}
    cap_table = np.stack([caps[(t, c)] for t in range(len(TRAIN_TEMPLATES)) for c in range(10)])
    img_enc, txt_enc = ImageEncoder(dim, nnx.Rngs(seed)), TextEncoder(len(vocab), dim, nnx.Rngs(seed + 1))
    o_i = nnx.Optimizer(img_enc, optax.adam(1e-3), wrt=nnx.Param)
    o_t = nnx.Optimizer(txt_enc, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def step(img_enc, txt_enc, o_i, o_t, x, ids):
        def loss(img_enc, txt_enc):
            return clip_loss(img_enc(x), txt_enc(ids), temperature)

        l, (gi, gt) = nnx.value_and_grad(loss, argnums=(0, 1))(img_enc, txt_enc)
        o_i.update(img_enc, gi)
        o_t.update(txt_enc, gt)
        return l

    rng = np.random.default_rng(seed)
    hist, t0, s = [], time.perf_counter(), 0
    while s < steps:
        for b in batches(rng, len(tx), batch):
            tpl = rng.integers(0, len(TRAIN_TEMPLATES), len(b))
            ids = cap_table[tpl * 10 + ty[b]]
            l = step(img_enc, txt_enc, o_i, o_t, tx[b], ids)
            if s % 100 == 0:
                hist.append((s, float(l)))
            s += 1
            if s >= steps:
                break
    jax.block_until_ready(l)
    return img_enc, txt_enc, vocab, hist, time.perf_counter() - t0


@nnx.jit
def _embed_images(img_enc, x):
    e = img_enc(x)
    return e / jnp.linalg.norm(e, axis=-1, keepdims=True)


def zero_shot(img_enc, txt_enc, vocab, vx, vy, template="a photo of a {}"):
    """Classify test images by which class *caption* their embedding is closest to."""
    ids = jnp.asarray(tokenize([template.format(class_words(c)) for c in range(10)], vocab))
    t = txt_enc(ids)
    t = t / jnp.linalg.norm(t, axis=-1, keepdims=True)
    preds = np.concatenate([np.asarray((_embed_images(img_enc, vx[i:i + 2000]) @ t.T).argmax(-1))
                            for i in range(0, len(vx), 2000)])
    return float(np.mean(preds == vy)), preds
