import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="33 · Generative images")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import time

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import rosetta

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")  # warm reruns skip compilation
    ex = lj.load_exercises(mo.notebook_dir())
    ref = lj.load_exercises(mo.notebook_dir(), solutions=True)
    _spec = importlib.util.spec_from_file_location("gen33", mo.notebook_dir() / "gen.py")
    gen = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(gen)
    return ex, gen, jax, jnp, lj, mo, np, plt, ref, rosetta, time


@app.cell
def _(mo):
    mo.md(r"""
    # 33 · GANs, VAEs, diffusion and CLIP (2014–2022)

    **Part 10, Milestones rebuilt in JAX.** Take Modules 11–12 (training craft, CNNs) and 04
    (probability) first.

    **Goal:** rebuild, in eight years of history, the ideas behind every image generator you've heard of,
    on Fashion-MNIST and your RX 7600 XT, and see what each family is good and bad at.

    **You will:** train a GAN and watch two networks play a game (and sometimes cheat) · train a VAE and
    walk around its 2-D "map" of clothing · noise images step by step and train a diffusion model to undo
    it, with classifier-free guidance · teach images and captions a shared space (a mini-CLIP) and use it
    to classify clothes it was never given a classifier for.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from.** In 2014 Ian Goodfellow and colleagues at the Université de Montréal
    proposed generative adversarial networks: a generator and a discriminator trained against each other.
    A year earlier Diederik Kingma and Max Welling had introduced the variational autoencoder, which learns
    a smooth latent space by maximizing a lower bound on the data likelihood. GANs dominated image
    synthesis for years, until diffusion models — first proposed by Jascha Sohl-Dickstein and colleagues in
    2015, then made practical by Jonathan Ho, Ajay Jain and Pieter Abbeel's DDPM in 2020 — overtook them.
    In 2021 OpenAI's CLIP tied images to text, and in 2022 latent diffusion (Rombach et al.) became Stable
    Diffusion. Timeline: Module 00b.
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The story

    **Before deep learning.** "Generative model" used to mean something you wrote down by hand: a mixture
    of Gaussians, a hidden Markov model, a Boltzmann machine. You could sample from them, but they were far
    too simple to draw a convincing photograph.

    **2013 — the VAE.** Kingma and Welling's *auto-encoding variational Bayes* trains an encoder that maps
    an image to a *distribution* over a small latent code, and a decoder that maps codes back to images.
    The **reparameterization trick** lets gradients flow through the random sampling, and the loss is the
    **ELBO**: reconstruct well, but keep the codes close to a standard normal. VAEs train stably and give
    you a smooth, navigable latent space, but their samples are blurry.

    **2014 — the GAN.** Goodfellow's idea (famously sketched after an argument with friends, then coded the
    same night) was to skip writing down a likelihood at all: train a *discriminator* to tell real from
    fake, and a *generator* to fool it. When it works, samples are sharp. When it doesn't, training
    oscillates or **mode-collapses** (the generator finds a few outputs that fool the discriminator and
    stops producing anything else). **DCGAN** (Radford, Metz & Chintala, 2015) found the architecture rules
    that made GANs train reliably on images; StyleGAN (2018–2020) produced photorealistic faces.

    **2015 → 2020 — diffusion.** Sohl-Dickstein et al. borrowed from non-equilibrium thermodynamics:
    destroy data slowly with noise, then learn to reverse each tiny step. It worked but stayed obscure
    until Ho, Jain & Abbeel's **DDPM** (2020) showed a simple recipe — predict the added noise with a
    U-Net, a plain mean-squared-error loss — that matched GANs on image quality while training as stably as
    a classifier. Faster samplers (DDIM, 2020) and **classifier-free guidance** (Ho & Salimans, 2021–22)
    followed.

    **2021 — CLIP and DALL·E.** OpenAI trained an image encoder and a text encoder so that matching
    (image, caption) pairs land close together: **CLIP**, from 400 million pairs scraped from the web.
    Without ever being trained as a classifier, it can classify by comparing an image to captions like
    "a photo of a dog". The same year DALL·E generated images from text.

    **2022 — latent diffusion.** Rombach et al. ran diffusion in a VAE's compressed latent space instead
    of on pixels, steered by a text encoder: that became **Stable Diffusion**, small enough for a consumer
    GPU. Today's image and video models are mostly descendants of these three threads: diffusion (or its
    cousin, flow matching), a latent autoencoder, and a CLIP-like text-image space.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, ref):
    _k = jax.random.key(0)
    _abar10 = ref.ddpm_schedule(10)[1]
    _probes = {
        "gan_losses": (jnp.array([1.0, -1.0]), jnp.array([0.5, -0.5])),
        "reparameterize": (_k, jnp.zeros((2, 2)), jnp.zeros((2, 2))),
        "elbo": (jnp.full((2, 4, 4, 1), 0.5), jnp.zeros((2, 4, 4, 1)), jnp.zeros((2, 2)), jnp.zeros((2, 2))),
        "ddpm_schedule": (10,),
        "q_sample": (jnp.ones((1, 4, 4, 1)), jnp.zeros(1, jnp.int32), jnp.zeros((1, 4, 4, 1)), _abar10),
        "ddpm_loss": (jnp.zeros(2), jnp.ones(2)),
        "clip_loss": (jnp.eye(3), jnp.eye(3), 0.1),
    }
    attempts = {n: lj.attempt(getattr(ex, n), *a) for n, a in _probes.items()}
    F = {n: (getattr(ex, n) if a.ok else getattr(ref, n)) for n, a in attempts.items()}
    _rows = "\n".join(f"| `{n}` | {'✅ yours' if a.ok else ('✏️ your turn — using the reference for now' if a.todo else '❌ error — using the reference')} |"
                      for n, a in attempts.items())
    mo.md(f"""
    ## 2 · Your turn: the seven loss functions

    Open `modules/33-generative-images/exercises.py`. Everything below trains with **your** functions as
    soon as they work (`make test N=33` checks them); until then it quietly uses the reference ones.

    | function | status |
    |---|---|
    {_rows.replace(chr(10), chr(10) + '    ')}
    """)
    return F, attempts


@app.cell
def _(gen, mo, time):
    _t0 = time.perf_counter()
    tx, ty, vx, vy = gen.load_data()
    judge, judge_acc = gen.train_classifier(tx, ty, vx, vy, epochs=2)
    mo.md(f"""
    **Data:** 60,000 Fashion-MNIST training images, scaled to [−1, 1]. To *measure* what the generators
    make, we also train a small CNN **judge** (Module 12's design): {judge_acc:.1%} test accuracy, in
    {time.perf_counter() - _t0:.0f} s. A good generator's samples should be recognizable to it, and should
    cover all ten classes.
    """)
    return judge, judge_acc, tx, ty, vx, vy


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · GAN (2014): two networks playing a game

    The discriminator $D$ outputs a logit, "how real does this look". With $\sigma$ the sigmoid:

    $$\mathcal L_D = -\mathbb E_x \log\sigma(D(x)) - \mathbb E_z \log(1-\sigma(D(G(z))))$$

    Goodfellow's paper noticed that the generator's "minimax" loss $\log(1-\sigma(D(G(z))))$ has almost no
    gradient early on, when $D$ easily rejects every fake. The **non-saturating** fix — maximize
    $\log\sigma(D(G(z)))$ instead — is what everyone uses, and what your `gan_losses` implements.

    Our generator follows DCGAN's rules: a dense layer to 7×7, then two strided transposed convolutions
    up to 28×28, ReLU inside, tanh at the end; the discriminator is strided convolutions with leaky ReLU and
    no pooling. (DCGAN used batch normalization; we use group normalization, which behaves the same in
    training and sampling. Adam with β₁ = 0.5 is straight from the DCGAN paper.)
    """)
    return


@app.cell
def _(mo, rosetta):
    mo.vstack([
        rosetta.tabs(mo, "transposed convolution"),
        rosetta.tabs(mo, "GroupNorm"),
    ])
    return


@app.cell
def _(F, gen, tx):
    G, _D, gan_hist, gan_secs = gen.train_gan(tx, F["gan_losses"], steps=2000)
    return G, gan_hist, gan_secs


@app.cell
def _(G, gan_hist, gan_secs, gen, jax, judge, mo, np, plt):
    _z = jax.random.normal(jax.random.key(42), (1000, 64))
    gan_samples = np.asarray(G(_z))
    _pred = np.asarray(gen.predict(judge, gan_samples))
    gan_hist_cls = np.bincount(_pred, minlength=10)
    _p = gan_hist_cls / gan_hist_cls.sum()
    gan_coverage = float(-(_p[_p > 0] * np.log(_p[_p > 0])).sum() / np.log(10))
    gan_missing = [gen.class_words(i) for i in range(10) if _p[i] < 0.02]
    _fig, _ax = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
    _h = np.array(gan_hist)
    _ax[0].plot(_h[:, 0], _h[:, 1], label="discriminator")
    _ax[0].plot(_h[:, 0], _h[:, 2], label="generator")
    _ax[0].set(xlabel="step", ylabel="loss", title="the game (no loss goes to zero)")
    _ax[0].legend()
    _ax[1].imshow(gen.grid(gan_samples[:48], 8), cmap="gray")
    _ax[1].set_title("GAN samples")
    _ax[1].axis("off")
    _ax[2].bar(range(10), gan_hist_cls)
    _ax[2].set_xticks(range(10), [gen.class_words(i)[:7] for i in range(10)], rotation=60, fontsize=8)
    _ax[2].set_title("what the judge thinks they are")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"""
    2,000 steps in **{gan_secs:.0f} s**. Neither loss heads to zero: if the discriminator won outright the
    generator would get no useful gradient, and vice versa — a healthy GAN hovers. Be honest about the
    samples: after this little training they're rough, mostly upper-body garments. **Class coverage**
    (normalized entropy of the judge's histogram over 1,000 samples) is **{gan_coverage:.2f}** (1.0 = all ten
    classes equally, as in the real data), and the judge finds almost no
    **{', '.join(gan_missing) if gan_missing else '—'}** (under 2% each). That's **partial mode collapse**,
    the GAN's classic failure: the generator learns the outputs that fool the discriminator most easily and
    neglects the rest. Longer training, other seeds and learning rates move these numbers a lot — GANs are
    famously sensitive to all three, which is a big part of why diffusion replaced them.
    """)])
    return gan_coverage, gan_missing, gan_samples


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · VAE (2013): a smooth map of clothing

    The encoder outputs a mean $\mu$ and log-variance for each image; we sample
    $z = \mu + \sigma\,\epsilon$ (your `reparameterize`) and decode. The **ELBO** (your `elbo`) is

    $$\underbrace{\mathbb E_{q(z|x)}[\log p(x|z)]}_{\text{reconstruct well}} -
      \underbrace{\mathrm{KL}(q(z|x)\,\|\,\mathcal N(0,I))}_{\text{stay near the prior}}$$

    a lower bound on $\log p(x)$. We use a **2-D latent** on purpose: too small for sharp images, but you
    can *see* the whole space.
    """)
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "VAE reparameterization and Gaussian KL", solves=("reparameterize", "elbo"))
    return


@app.cell
def _(F, gen, tx):
    vae, vae_hist, vae_secs = gen.train_vae(tx, F["elbo"], F["reparameterize"], latent=2, steps=2000)
    return vae, vae_hist, vae_secs


@app.cell
def _(gen, jax, jnp, mo, np, plt, vae, vae_hist, vae_secs, vx, vy):
    _mu, _ = vae.encode((vx[:4000] + 1) / 2)
    _mu = np.asarray(_mu)
    _g = np.linspace(-2.5, 2.5, 10)
    _zz = jnp.array([[a, b] for b in _g[::-1] for a in _g])
    _dec = np.asarray(jax.nn.sigmoid(vae.decode(_zz))) * 2 - 1
    _h = np.array(vae_hist)
    _fig, _ax = plt.subplots(1, 3, figsize=(13, 4))
    _ax[0].plot(_h[:, 0], _h[:, 2], label="reconstruction log-lik")
    _ax[0].plot(_h[:, 0], -_h[:, 3], label="−KL")
    _ax[0].plot(_h[:, 0], _h[:, 1], "k", lw=2, label="ELBO")
    _ax[0].set(xlabel="step", title="ELBO = recon − KL (nats / image)")
    _ax[0].legend(fontsize=8)
    _sc = _ax[1].scatter(_mu[:, 0], _mu[:, 1], c=vy[:4000], cmap="tab10", s=3)
    _ax[1].set(title="test images in the 2-D latent space", xlim=(-3, 3), ylim=(-3, 3))
    _handles, _ = _sc.legend_elements()
    _ax[1].legend(_handles, [gen.class_words(i)[:10] for i in range(10)], fontsize=6, loc="upper right", ncol=2)
    _ax[2].imshow(gen.grid(_dec, 10), cmap="gray")
    _ax[2].set_title("decoding a grid of latent points")
    _ax[2].axis("off")
    _fig.tight_layout()
    vae_elbo = float(_h[-1, 1])
    mo.vstack([_fig, mo.md(f"""
    2,000 steps in **{vae_secs:.0f} s**; final ELBO **{vae_elbo:.0f} nats per image**
    (reconstruction {_h[-1, 2]:.0f}, KL {_h[-1, 3]:.1f}). Similar garments cluster (trousers, bags, the
    three kinds of shoes), and walking across the map morphs one into another — the "smooth latent space"
    that made VAEs useful long after GANs out-drew them, and the reason Stable Diffusion runs in a VAE's
    latent space. The blur is the price: the decoder hedges between all the images a code could be.
    """)])
    return (vae_elbo,)


@app.cell
def _(gen, jax, jnp, mo, np, plt, vae, vx):
    _a, _ = vae.encode((vx[0:1] + 1) / 2)
    _b, _ = vae.encode((vx[1:2] + 1) / 2)
    _zs = jnp.concatenate([_a + (_b - _a) * s for s in np.linspace(0, 1, 8)])
    _imgs = np.asarray(jax.nn.sigmoid(vae.decode(_zs))) * 2 - 1
    _fig, _ax = plt.subplots(figsize=(9, 1.6))
    _ax.imshow(gen.grid(_imgs, 8), cmap="gray")
    _ax.axis("off")
    _ax.set_title("interpolating between two test images in latent space")
    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Diffusion (2015 → 2020): learning to un-noise

    **Forward process.** Add a little Gaussian noise $T = 1000$ times with variances $\beta_t$ (your
    `ddpm_schedule`, linear from $10^{-4}$ to $0.02$). Because Gaussians compose, you can jump straight
    to any step (your `q_sample`):

    $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad
      \bar\alpha_t = \prod_{s\le t}(1-\beta_s)$$

    **Reverse process.** Train a U-Net $\epsilon_\theta(x_t, t)$ to predict the noise that was added —
    a plain mean-squared error (your `ddpm_loss`). To generate, start from pure noise and remove a bit of
    predicted noise at a time. Move the slider:
    """)
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "diffusion forward noising", solves=("ddpm_schedule", "q_sample"))
    return


@app.cell
def _(mo):
    t_slider = mo.ui.slider(steps=[0, 50, 100, 200, 300, 500, 700, 850, 999], value=300,
                            label="timestep t", show_value=True)
    t_slider
    return (t_slider,)


@app.cell
def _(F):
    betas, abar = F["ddpm_schedule"](1000)
    return abar, betas


@app.cell
def _(F, abar, gen, jax, jnp, mo, np, plt, t_slider, vx):
    _x0 = jnp.asarray(vx[:8])
    _t = jnp.full((8,), t_slider.value)
    _xt = F["q_sample"](_x0, _t, jax.random.normal(jax.random.key(3), _x0.shape), abar)
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 2.2), gridspec_kw={"width_ratios": [3, 1]})
    _ax[0].imshow(gen.grid(np.asarray(_xt), 8), cmap="gray")
    _ax[0].axis("off")
    _ax[0].set_title(f"x_t at t = {t_slider.value}")
    _ax[1].plot(np.asarray(abar), label=r"$\bar\alpha_t$ (signal²)")
    _ax[1].axvline(t_slider.value, color="k", lw=0.8)
    _ax[1].set(xlabel="t", ylim=(0, 1.02))
    _ax[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"At t = {t_slider.value}, **{float(abar[t_slider.value]):.1%}** of the "
                           f"variance is still the image.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Our network is a small **U-Net** (≈ 0.86M parameters): down 28 → 14 → 7 and back up with skip
    connections, each block told the timestep (sinusoidal embedding, like a Transformer's positions) and,
    optionally, the class. During training we hide the class 10% of the time; at sampling time that lets
    us use **classifier-free guidance**: $\hat\epsilon = (1+w)\,\epsilon(x,t,y) - w\,\epsilon(x,t,\varnothing)$
    — push away from "any garment" toward "this garment". For speed we sample with **DDIM** (Song et al.,
    2020): 50 deterministic steps instead of 1,000, the whole loop compiled as one [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan) (Module 08).

    A short live run first — a few epochs' worth of steps:
    """)
    return


@app.cell
def _(F, abar, gen, tx, ty):
    unet, ddpm_hist, ddpm_secs = gen.train_ddpm(tx, ty, F["q_sample"], F["ddpm_loss"], abar, 1000, steps=1200)
    return ddpm_hist, ddpm_secs, unet


@app.cell
def _(abar, ddpm_hist, ddpm_secs, gen, jax, jnp, judge, mo, np, plt, unet):
    _sampler = gen.ddim_sampler(unet, abar, 1000, 50)
    _y = jnp.repeat(jnp.arange(10), 6)
    _x = np.asarray(_sampler(jax.random.key(7), _y, 2.0))
    live_cond_acc = float(np.mean(np.asarray(gen.predict(judge, _x)) == np.asarray(_y)))
    _h = np.array(ddpm_hist)
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    _ax[0].plot(_h[:, 0], _h[:, 1])
    _ax[0].set(xlabel="step", ylabel="noise MSE", title="DDPM loss", yscale="log")
    _ax[1].imshow(gen.grid(_x, 6), cmap="gray")
    _ax[1].set_title("live model, rows = classes, guidance w = 2")
    _ax[1].axis("off")
    _fig.tight_layout()
    ddpm_final_loss = float(np.mean(_h[-3:, 1]))
    mo.vstack([_fig, mo.md(f"""
    1,200 steps in **{ddpm_secs:.0f} s**, final loss ≈ **{ddpm_final_loss:.3f}**. The judge recognizes
    **{live_cond_acc:.0%}** of these as the class we asked for. They're rough after so little training;
    the next cell uses the long run from `train.py` if you've made one.
    """)])
    return ddpm_final_loss, live_cond_acc


@app.cell
def _(mo):
    w_slider = mo.ui.slider(steps=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0], value=2.0, label="guidance w", show_value=True)
    w_slider
    return (w_slider,)


@app.cell
def _(abar, gen, jax, jnp, judge, mo, np, plt, w_slider):
    _path = gen.RUNS / "ddpm.npz"
    if not _path.exists():
        long_result = None
        _out = mo.callout(mo.md(
            "**No long run yet.** Train one (about 5 minutes on an idle RX 7600 XT, using your loss "
            "functions):\n\n`containers/run.sh python modules/33-generative-images/train.py`\n\nthen re-run "
            "this cell for sharper samples and the guidance slider."), kind="info")
    else:
        _m = gen.load_model(gen.UNet(gen.nnx.Rngs(0)), _path)
        _sampler = gen.ddim_sampler(_m, abar, 1000, 50)
        _y = jnp.repeat(jnp.arange(10), 8)
        _x = np.asarray(_sampler(jax.random.key(11), _y, w_slider.value))
        _acc = float(np.mean(np.asarray(gen.predict(judge, _x)) == np.asarray(_y)))
        _xu = np.asarray(_sampler(jax.random.key(12), jnp.full((1000,), 10), 0.0))
        _pred = np.bincount(np.asarray(gen.predict(judge, _xu)), minlength=10)
        _p = _pred / _pred.sum()
        _cov = float(-(_p[_p > 0] * np.log(_p[_p > 0])).sum() / np.log(10))
        long_result = {"model": _m, "acc": _acc, "coverage": _cov}
        _fig, _ax = plt.subplots(figsize=(7, 6))
        _ax.imshow(gen.grid(_x, 8), cmap="gray")
        _ax.set_title(f"long run (EMA weights), rows = classes, w = {w_slider.value}")
        _ax.axis("off")
        _out = mo.vstack([_fig, mo.md(f"""
        The judge recognizes **{_acc:.0%}** as the requested class at w = {w_slider.value}. Unconditional
        samples (no class, w = 0) cover the classes with entropy **{_cov:.2f}**. Raise the guidance and
        the garments get more prototypical and recognizable but less varied; at w = 0 they're diverse but
        some are ambiguous — the quality/diversity knob every text-to-image app exposes.
        """)])
    _out
    return (long_result,)


@app.cell
def _(abar, betas, gen, jax, jnp, long_result, mo, time, unet):
    _m = long_result["model"] if long_result else unet
    _ddim = gen.ddim_sampler(_m, abar, 1000, 50)
    _anc = gen.ancestral_sampler(_m, betas, abar, 1000)
    _y = jnp.full((16,), 10)
    jax.block_until_ready(_ddim(jax.random.key(0), _y, 0.0))
    jax.block_until_ready(_anc(jax.random.key(0), 16))
    _t = time.perf_counter()
    jax.block_until_ready(_ddim(jax.random.key(1), _y, 0.0))
    _t_ddim = time.perf_counter() - _t
    _t = time.perf_counter()
    jax.block_until_ready(_anc(jax.random.key(1), 16))
    _t_anc = time.perf_counter() - _t
    mo.md(f"""
    **Sampling speed** (16 images, after compilation): DDPM's original ancestral sampler takes 1,000
    network calls (×1 here, unconditional) — **{_t_anc:.2f} s**; DDIM's 50 deterministic steps (×2 for
    guidance) — **{_t_ddim:.2f} s**. Diffusion's weakness is exactly this: a GAN or VAE samples in *one*
    forward pass. Much of the research since 2020 (DDIM, distillation, consistency models, flow matching)
    is about taking fewer steps.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · CLIP (2021): images and words in one space

    Two encoders — a small CNN for images, a bag-of-words network for captions — map into the same
    64-dimensional space. For a batch of $B$ matching pairs, CLIP's loss (your `clip_loss`) is a
    $B$-way classification in both directions: each image should pick out *its* caption among all $B$, and
    each caption its image (symmetric **InfoNCE**, with embeddings L2-normalized and a temperature).

    Our captions are templated from the class names — "a photo of a sneaker", "a small bag", … — six
    templates in training. Real CLIP used 400 million naturally occurring captions, which is where its
    open-ended zero-shot ability comes from; ours only ever sees ten garment names, so think of it as the
    *mechanism*, not the magic.
    """)
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "contrastive (CLIP / InfoNCE) loss", solves="clip_loss")
    return


@app.cell
def _(F, gen, tx, ty):
    clip_img, clip_txt, clip_vocab, clip_hist, clip_secs = gen.train_clip(tx, ty, F["clip_loss"], steps=800)
    return clip_hist, clip_img, clip_secs, clip_txt, clip_vocab


@app.cell
def _(clip_hist, clip_img, clip_secs, clip_txt, clip_vocab, gen, jnp, judge_acc, mo, np, plt, vx, vy):
    clip_acc, _pred = gen.zero_shot(clip_img, clip_txt, clip_vocab, vx, vy)
    clip_acc_unseen, _ = gen.zero_shot(clip_img, clip_txt, clip_vocab, vx, vy, gen.UNSEEN_TEMPLATE)
    _ids = jnp.asarray(gen.tokenize([f"a photo of a {gen.class_words(c)}" for c in range(10)], clip_vocab))
    _t = np.asarray(clip_txt(_ids))
    _t = _t / np.linalg.norm(_t, axis=1, keepdims=True)
    _e = np.asarray(gen._embed_images(clip_img, vx[:5000]))
    _means = np.stack([_e[vy[:5000] == c].mean(0) for c in range(10)])
    _means /= np.linalg.norm(_means, axis=1, keepdims=True)
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.3]})
    _h = np.array(clip_hist)
    _ax[0].plot(_h[:, 0], _h[:, 1])
    _ax[0].axhline(np.log(256), ls="--", c="gray", lw=0.8)
    _ax[0].set(xlabel="step", ylabel="InfoNCE", title="loss (dashed: chance, ln 256)")
    _im = _ax[1].imshow(_means @ _t.T, cmap="viridis")
    _names = [gen.class_words(c)[:9] for c in range(10)]
    _ax[1].set_xticks(range(10), _names, rotation=60, fontsize=8)
    _ax[1].set_yticks(range(10), _names, fontsize=8)
    _ax[1].set(xlabel="caption 'a photo of a …'", ylabel="images of …", title="cosine similarity")
    _fig.colorbar(_im, ax=_ax[1], fraction=0.046)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"""
    800 steps in **{clip_secs:.0f} s**. **Zero-shot** accuracy — classify each test image by the closest
    caption "a photo of a …", with no classifier head anywhere: **{clip_acc:.1%}**. With a caption phrasing
    it never saw ("the … in this snapshot", whose extra words are untrained): **{clip_acc_unseen:.1%}**.
    The supervised judge CNN scores {judge_acc:.1%}. The bright diagonal is the shared space working;
    the bright off-diagonal patches are the garments people confuse too (shirt / T-shirt / pullover / coat).

    Why doesn't the loss fall near zero? A batch of 256 holds about 26 images of each class, and their
    captions are interchangeable — the loss counts every *other* "a photo of a sneaker" as a wrong answer.
    Real CLIP has the same "false negative" problem, just much rarer, because real captions are unique.
    """)])
    return clip_acc, clip_acc_unseen


@app.cell
def _(clip_acc, gan_coverage, gan_missing, live_cond_acc, long_result, mo, vae_elbo):
    _ddpm = (f"{long_result['acc']:.0%} recognizable as the requested class (long run)" if long_result
             else f"{live_cond_acc:.0%} recognizable as requested (short live run)")
    mo.md(f"""
    ## 7 · Four families, side by side

    | | GAN (2014) | VAE (2013) | Diffusion (2015/2020) | CLIP (2021) |
    |---|---|---|---|---|
    | learns | to fool a discriminator | to reconstruct through a noisy bottleneck | to remove noise | which caption goes with which image |
    | what you saw | rough samples after 2,000 steps; coverage {gan_coverage:.2f}, few {', '.join(gan_missing) or '—'} | a smooth 2-D map, blurry, ELBO {vae_elbo:.0f} nats | {_ddpm} | {clip_acc:.0%} zero-shot accuracy |
    | training | unstable; sensitive to seeds and LR | stable | stable (plain MSE) | stable; needs big batches |
    | sampling | one forward pass | one forward pass | 50–1,000 passes | not a generator |
    | lives on in | real-time synthesis, super-resolution, the "adversarial" idea in many losses | the latent space of Stable Diffusion and video models | almost every image and video generator | text conditioning, search, zero-shot classification |

    The modern recipe uses three of the four at once: a **VAE** to compress images into latents, a
    **diffusion** model (or flow matching) to generate in that latent space, steered by a **CLIP-like** text
    encoder.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "It always produces sharper images than a GAN": "a",
            "It needs many network evaluations to draw one sample": "b",
            "It can't be conditioned on a class or text": "c",
        },
        label="**Check yourself:** what is diffusion's main practical *weakness* compared with a GAN?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "b": mo.callout(mo.md("Right — 50 to 1,000 network calls per sample versus one. Most diffusion "
                              "research since 2020 is about cutting that number."), kind="success"),
        "a": mo.callout(mo.md("That's a strength (and not always true), not a weakness."), kind="warn"),
        "c": mo.callout(mo.md("It can — you just did it with class labels and classifier-free guidance."),
                        kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(33))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    * Ian Goodfellow, [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)
      — the inventor's own guide, including why the non-saturating loss.
    * Lilian Weng, [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
      — the clearest derivation of DDPM, DDIM and guidance.
    * Song, Meng & Ermon, [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) and
      Ho & Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — the two tricks
      used in Section 5.
    * Jay Alammar, [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
      — how the VAE, U-Net and text encoder fit together.
    * OpenAI, [CLIP: Connecting text and images](https://openai.com/index/clip/) — the announcement, with
      the zero-shot results that made it famous.

    **Next:** Module 34 — deep reinforcement learning, from DQN on Atari to AlphaZero.
    """)
    return


if __name__ == "__main__":
    app.run()
