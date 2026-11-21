import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="43 · From pixels to words")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import json

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import rosetta
    from learnjax.data import FASHION_CLASSES, fashion_mnist

    ex = lj.load_exercises(mo.notebook_dir())
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")  # re-runs skip XLA compilation
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    _spec = importlib.util.spec_from_file_location("m43_models", mo.notebook_dir() / "models.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    RUNS = M.RUNS
    fm = fashion_mnist()

    def read(name):
        try:
            return json.loads((RUNS / name).read_text())
        except OSError:
            return None
    return FASHION_CLASSES, M, RUNS, ex, fm, jax, jnp, lj, mo, np, plt, read, rosetta


@app.cell
def _(mo):
    mo.md(r"""
    # 43 · From pixels to words: vision transformers and vision-language models (2020–2025)

    **Take first:** Module 12 (CNNs), Module 15 (attention), Module 33 (CLIP on Fashion-MNIST) and
    Module 18 (`learnjax/llama.py` and SmolLM2).

    **Goal:** a language model reads tokens. To make it read an image, turn the image into tokens too:
    cut it into patches, run a **vision transformer** over them, and map its outputs into the language
    model's embedding space, where they sit in the prompt like words. That's how nearly every open
    vision-language model works: LLaVA, PaliGemma, SmolVLM, Qwen-VL.

    **You will:** cut images into patch tokens · race a small ViT against a CNN and see why ViTs need data
    · write SigLIP's loss and load Google's SigLIP into JAX, matching Hugging Face's numbers · teach frozen
    SmolLM2-135M to caption COCO photos by training only a small projector · trade image tokens for width
    with pixel shuffle and measure what it costs.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    Convolutional networks owned vision from AlexNet (2012) on. In 2020 Dosovitskiy et al. showed that a
    plain Transformer, fed an image as a sequence of 16 × 16 patches, matched them, **if** pretrained on
    hundreds of millions of images: "An Image is Worth 16x16 Words". **CLIP** (2021) trained a ViT and a
    text encoder together on 400M image-caption pairs, so images and words shared one embedding space;
    **SigLIP** (2023) replaced its batch-wide softmax with a per-pair sigmoid. Flamingo (2022) wired a
    vision encoder into a frozen language model with cross-attention. **LLaVA** (2023) found something
    much simpler worked: project the image tokens and put them in the prompt. By 2025 the recipe ran from
    256M-parameter **SmolVLM** to the largest open models, with **dynamic resolution** (Qwen2.5-VL) deciding
    how many tokens an image deserves.

    *Timeline: Module 00b.*
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Images as tokens

    A 224 × 224 photo has 50,176 pixels: far too many tokens. A ViT cuts it into **patches** of
    $p \times p$ pixels (16 × 16 for SigLIP, so 14 × 14 = 196 patches), flattens each into a vector of
    $p^2 C$ numbers, and multiplies by one learned matrix. That is exactly a $p \times p$ convolution with
    stride $p$: the ViT's only convolution. Then a learned **position embedding** per patch, and a
    Transformer encoder with no causal mask: every patch sees every other.

    **Your turn:** `patchify(images, p)`.
    """)
    return


@app.cell
def _(FASHION_CLASSES, ex, fm, jnp, lj, mo, np, plt):
    _img = jnp.asarray(fm["train_images"][:1] / 255.0, jnp.float32)[..., None]
    _a = lj.attempt(ex.patchify, _img, 7)
    if not _a.ok:
        _out = _a.view(mo)
    else:
        _p = np.asarray(_a.value[0])
        _fig, _axs = plt.subplots(1, 2, figsize=(7, 2.6), gridspec_kw={"width_ratios": [1, 2.2]})
        _axs[0].imshow(np.asarray(_img[0, ..., 0]), cmap="gray_r")
        for _t in (7, 14, 21):
            _axs[0].axhline(_t - 0.5, color="C1", lw=1)
            _axs[0].axvline(_t - 0.5, color="C1", lw=1)
        _axs[0].set_title(f"one {FASHION_CLASSES[int(fm['train_labels'][0])]}, 4 × 4 patches of 7 × 7", fontsize=8)
        _axs[1].imshow(_p, cmap="gray_r", aspect="auto")
        _axs[1].set_xlabel("49 pixels per patch")
        _axs[1].set_ylabel("token")
        _axs[1].set_title(f"your patchify: {_p.shape[0]} tokens × {_p.shape[1]} numbers", fontsize=8)
        for _ax in _axs:
            _ax.set_xticks([])
        _axs[0].set_yticks([])
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · A ViT against a CNN: built-in assumptions vs data

    A CNN builds in two assumptions about images: nearby pixels matter together (small kernels), and a
    pattern means the same thing anywhere (shared weights). A ViT assumes neither; attention could relate
    any patch to any other from the first layer, and it must *learn* that neighbours matter. That costs
    data. `train.py --classify` trains a small ViT (patches of 4 × 4, so 49 tokens; 4 layers of width 128)
    and a CNN of about the same size (three 3 × 3 conv layers) on 1,000, 6,000 and all 60,000 Fashion-MNIST
    images, the same 3,000 steps each, and scores both on the 10,000 test images.
    """)
    return


@app.cell
def _(mo, plt, read):
    _r = read("classify.json")
    if _r is None:
        _out = mo.md("*Run `train.py --classify` (about 10 minutes) to fill this in.*")
    else:
        _ns = sorted({int(k.split("/")[1]) for k in _r})
        _fig, _ax = plt.subplots(figsize=(5.5, 3))
        for _kind, _lab in (("cnn", "CNN"), ("vit", "ViT")):
            _ax.plot(_ns, [_r[f"{_kind}/{n}"]["test_acc"] for n in _ns], "o-",
                     label=f"{_lab} ({_r[f'{_kind}/{_ns[0]}']['params'] / 1e6:.2f}M params)")
        _ax.set_xscale("log")
        _ax.set_xlabel("training images")
        _ax.set_ylabel("test accuracy")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _gap = {n: _r[f"cnn/{n}"]["test_acc"] - _r[f"vit/{n}"]["test_acc"] for n in _ns}
        _rows = "\n".join(f"| {n:,} | {_r[f'cnn/{n}']['test_acc']:.1%} | {_r[f'vit/{n}']['test_acc']:.1%} | "
                          f"{_gap[n] * 100:+.1f} |" for n in _ns)
        _shrinks = _gap[_ns[-1]] < _gap[_ns[0]]
        _out = mo.vstack([_fig, mo.md(f"""
    | training images | CNN | ViT | CNN lead (points) |
    |---|---|---|---|
    {_rows}

    {("The CNN's lead shrinks as data grows" if _gap[_ns[0]] - _gap[_ns[-1]] > 0.03 else
      "The CNN's lead shrinks only slightly as data grows") if _shrinks else
     "The CNN keeps its lead at every size"}, with
    the ViT {"ahead" if _gap[_ns[-1]] < 0 else "still behind"} at 60,000 images; the two have about the same
    number of parameters ({_r[f"cnn/{_ns[0]}"]["params"] / 1e3:.0f}K and {_r[f"vit/{_ns[0]}"]["params"] / 1e3:.0f}K).
    Dosovitskiy et al. saw the same picture at scale: ViTs trailed ResNets when trained on ImageNet's 1.3M
    images and passed them only after pretraining on 300M. That is why every vision-language model starts
    from a ViT someone else pretrained on hundreds of millions of image-text pairs.
    """)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · SigLIP: images and captions in one space

    CLIP trains an image encoder and a text encoder so that a photo and its caption land close together
    (Module 33 did it on Fashion-MNIST). For a batch of $n$ pairs it scores all $n^2$ combinations and
    applies a softmax across each row: "which of these $n$ captions is mine?" That needs every score in the
    batch before any loss can be computed, which is awkward when the batch is split across many GPUs.

    SigLIP asks $n^2$ independent yes/no questions instead: "is this caption this image's?" Each score
    goes through a sigmoid on its own. With a learned temperature $t$ and bias $b$:

    $$\text{logit}_{ij} = t\,\hat x_i \cdot \hat y_j + b, \qquad
    \mathcal L = -\frac1n \sum_{i,j} \log \sigma\big(z_{ij}\,\text{logit}_{ij}\big),\quad z_{ij} = \pm 1.$$

    **Your turn:** `siglip_logits(img, txt, log_t, b)` and `siglip_loss(logits)`.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _k = jax.random.split(jax.random.key(3), 2)
    _x = jax.random.normal(_k[0], (8, 16))
    _a = lj.attempt(ex.siglip_logits, _x, _x + 0.1 * jax.random.normal(_k[1], (8, 16)), jnp.log(10.0), -5.0)
    _b = lj.attempt(ex.siglip_loss, _a.value) if _a.ok else _a
    siglip_ready = _a.ok and _b.ok
    if not siglip_ready:
        _out = (_a if not _a.ok else _b).view(mo)
    else:
        _shuf = ex.siglip_loss(_a.value[:, ::-1])
        _out = mo.md(f"Eight pairs where each caption is its image plus a little noise: loss "
                     f"**{float(_b.value):.3f}**; the same captions in reverse order (all wrong): "
                     f"**{float(_shuf):.3f}**. Diagonal logits ≈ {float(jnp.diag(_a.value).mean()):.1f}, "
                     f"off-diagonal ≈ {float((_a.value.sum() - jnp.trace(_a.value)) / 56):.1f}.")
    _out
    return (siglip_ready,)


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "sigmoid image-text loss (SigLIP)", solves="siglip_loss")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Google's SigLIP, in your JAX

    `models.py` loads **SigLIP-B/16-224** (Google, Apache-2.0: an 86M-parameter ViT and a text encoder,
    trained on billions of image-text pairs) from Hugging Face into plain JAX functions. Its first layer is
    your `patchify`: the checkpoint's 16 × 16 convolution kernel, reshaped to a (768, 768) matrix. The rest
    is a 12-layer pre-norm Transformer (like Module 17's, with LayerNorm and GELU) and an **attention
    pooling** head: one learned query attends over the 196 patch tokens to make the image embedding.

    `train.py --siglip` checks it three ways: Hugging Face's own example (the two-cats photo, which happens
    to be in COCO's validation set), retrieval on 500 held-out COCO photos with 5 captions each, and your
    loss on real pairs.
    """)
    return


@app.cell
def _(mo, read):
    _r = read("siglip.json")
    if _r is None:
        _out = mo.md("*Run `train.py --siglip` (downloads SigLIP, ~810 MB, and COCO val2017, ~1.07 GB) to see "
                     "the checks.*")
    else:
        _c = _r["cats"]
        _p = _c["a photo of 2 cats"]
        _out = mo.md(f"""
    * **Hugging Face's example:** probability that the photo is "a photo of 2 cats" =
      **{_p:.1%}** (Hugging Face's documentation prints 31.9%) {"✅" if abs(_p - 0.319) < 0.005 else "❌"};
      "a photo of 2 dogs": {_c["a photo of 2 dogs"]:.4%}. With a sigmoid the two needn't sum to 1: each is
      its own question, and the model isn't very sure the caption fits exactly.
    * **Retrieval, 500 COCO photos × 5 captions:** the best-scoring caption belongs to the right photo
      **{_r["image_to_text_r1"]:.1%}** of the time (chance: 0.2%); the best photo for a caption is right
      **{_r["text_to_image_r1"]:.1%}** of the time. No training on COCO.
    * **Your loss on 32 real pairs:** {_r["loss_matched"]:.2f} matched, {_r["loss_shuffled"]:.2f} with the
      captions shuffled. The learned scale is $t = e^{{{_r["log_t"]:.2f}}} \\approx {2.718281828 ** _r["log_t"]:.0f}$
      and the bias $b = {_r["b"]:.1f}$: strongly negative, because almost every pair in a batch is a non-match.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Changing resolution

    SigLIP learned 196 position embeddings for a 14 × 14 grid. Feed it a 448 × 448 image and there are
    28 × 28 patches and no embeddings for them. The standard fix treats the embeddings as a small image
    and resizes it. PaliGemma and SmolVLM run SigLIP at higher resolutions than it was trained at this way,
    usually with some fine-tuning after.

    **Your turn:** `resize_pos_embed(pos, new_grid)`.
    """)
    return


@app.cell
def _(M, RUNS, ex, jnp, lj, mo, np, plt):
    _a = lj.attempt(ex.resize_pos_embed, jnp.arange(16.0).reshape(16, 1), 8)
    if not _a.ok:
        _out = _a.view(mo)
    elif not (RUNS / "siglip.json").exists():
        _out = mo.md(f"Your resize turns a 4 × 4 ramp into an 8 × 8 one: corners "
                     f"{float(_a.value[0, 0]):.2f} and {float(_a.value[-1, 0]):.2f}. (Run `train.py --siglip` to see "
                     f"SigLIP's own embeddings here.)")
    else:
        from safetensors.numpy import load_file as _lf

        from learnjax.llama import download as _dl

        _pos = _lf(str(_dl(M.SIGLIP) / "model.safetensors"))["vision_model.embeddings.position_embedding.weight"]
        _big = np.asarray(ex.resize_pos_embed(jnp.asarray(_pos), 28))
        _unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
        _fig, _axs = plt.subplots(1, 4, figsize=(9, 2.4))
        for _ax, (_g, _P, _cell) in zip(_axs, ((14, _pos, 3 * 14 + 3), (14, _pos, 7 * 14 + 10),
                                               (28, _big, 6 * 28 + 6), (28, _big, 14 * 28 + 20))):
            _sim = (_unit(_P) @ _unit(_P)[_cell]).reshape(_g, _g)
            _ax.imshow(_sim, cmap="viridis")
            _ax.set_title(f"{_g}×{_g}: patch ({_cell // _g}, {_cell % _g})", fontsize=8)
            _ax.set_xticks([])
            _ax.set_yticks([])
        _fig.tight_layout()
        _out = mo.vstack([_fig, mo.md(
            "Cosine similarity of one patch's position embedding to all the others: nobody told SigLIP the "
            "patches form a grid, but it learned embeddings where neighbours are similar (left two). Your "
            "resize to 28 × 28 keeps the same picture at twice the resolution (right two).")])
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "bilinear resize of a grid (position-embedding interpolation)", solves="resize_pos_embed")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · LLaVA's recipe: a projector into a frozen language model

    SigLIP's 196 output tokens live in SigLIP's 768-dimensional space; SmolLM2-135M reads 576-dimensional
    embeddings. LLaVA-1.5's bridge is a 2-layer MLP, the **projector**:

    $$\text{image tokens} = W_2\,\mathrm{GELU}(W_1 \cdot \text{SigLIP tokens} + b_1) + b_2.$$

    Put those tokens at the start of the prompt, then the text "A photo of", then the caption, and train
    on the usual next-token loss over the caption only. In LLaVA's first stage, everything but the projector
    is **frozen**: SigLIP keeps seeing as it learned to, SmolLM2 keeps writing as it learned to, and 1M new
    parameters learn to translate between them. (LLaVA's second stage then fine-tunes the language model on
    visual instructions.)

    `train.py --vlm` does this on COCO: 4,500 photos × 5 captions for training (2,000 steps of 32), 500
    photos held out. SigLIP's tokens for all 5,000 photos are computed once and cached. The baseline is
    SmolLM2 with no image at all: how well can it guess a caption from "A photo of" alone?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · How many tokens is an image worth?

    196 image tokens is more than most captions, and every one of them costs the language model attention
    and KV cache (Module 44). SmolVLM's answer is **pixel shuffle**: merge each $r \times r$ neighbourhood
    of tokens into one token $r^2$ times wider. The projector's first layer then mixes the neighbourhood.
    The information is still there, just packed into fewer, wider tokens. SmolVLM cuts 1,024 patches to 64
    this way ($r = 4$).

    Our grid is 14 × 14, so $r$ can be 2 (49 tokens) or 7 (4 tokens). The race below trains one projector
    per setting.

    **Your turn:** `pixel_shuffle(x, r)`.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo, np):
    _x = jnp.arange(16.0).reshape(1, 16, 1)
    _a = lj.attempt(ex.pixel_shuffle, _x, 2)
    if not _a.ok:
        _out = _a.view(mo)
    else:
        _out = mo.md(f"A 4 × 4 grid numbered 0–15 row by row, shuffled with r = 2: "
                     f"`{np.asarray(_a.value[0]).astype(int).tolist()}`: four tokens, each holding one 2 × 2 corner "
                     f"(expected `[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]`).")
    _out
    return


@app.cell
def _(M, mo, plt, read):
    _r = read("vlm.json")
    if _r is None or "text-only" not in _r:
        _out = mo.md("*Run `train.py --vlm` (about 40 minutes) to train the captioners.*")
    else:
        _names = [n for n in M.VLM_VARIANTS if n in _r]
        _base = _r["text-only"]["test_loss"]
        _shuffle = {n: r for n, r in M.VLM_VARIANTS.items() if r is not None}
        _pp = {n: (r * r * 768) * 576 + 576 + 576 * 576 + 576 for n, r in _shuffle.items()}   # projector size
        _tr = {n: float(sum(_r[n]["train_curve"][-5:]) / 5) for n in _shuffle if _r[n]["train_curve"]}
        _rows = "\n".join(f"| {n} | {_r[n]['image_tokens']} | "
                          f"{(f'{_pp[n] / 1e6:.1f}M' if n in _pp else '—')} | "
                          f"{(f'{_tr[n]:.2f}' if n in _tr else '—')} | **{_r[n]['test_loss']:.3f}** | "
                          f"{_base - _r[n]['test_loss']:+.3f} |" for n in _names)
        _fig, _ax = plt.subplots(figsize=(6, 3))
        for _n in _names[1:]:
            _c = _r[_n]["train_curve"]
            _ax.plot([50 * (i + 1) for i in range(len(_c))], _c, label=_n)
        _ax.axhline(_base, color="gray", ls="--", label="text-only (test)")
        _ax.set_xlabel("step")
        _ax.set_ylabel("caption loss (train)")
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _v = {n: _r[n]["test_loss"] for n in _names}
        _best = min(_v, key=_v.get)
        _out = mo.vstack([_fig, mo.md(f"""
    | image input | tokens | projector parameters | train loss (last 250 steps) | test caption loss | vs no image |
    |---|---|---|---|---|---|
    {_rows}

    * Seeing the photo cuts the caption loss from {_base:.2f} to {_v[_best]:.2f} nats per token
      (best: {_best}), with only the projector trained.
    * Pixel shuffle with r = 2 costs {_v["49 tokens (shuffle 2)"] - _v["196 tokens"]:+.3f} against all 196
      tokens, for a quarter of the tokens. Squeezing to 4 tokens costs
      {_v["4 tokens (shuffle 7)"] - _v["196 tokens"]:+.3f}, and still keeps
      {(_base - _v["4 tokens (shuffle 7)"]) / (_base - _v["196 tokens"]):.0%} of the 196-token gain.
    * **Read the 4-token row with care.** Pixel shuffle widens each token r² times, so the projector's first
      layer grows with it: {_pp["4 tokens (shuffle 7)"] / 1e6:.0f}M parameters against
      {_pp["196 tokens"] / 1e6:.1f}M, trained on only 22,500 captions. Its train loss
      ({_tr.get("4 tokens (shuffle 7)", float("nan")):.2f}) is far below its test loss: it memorizes. So this
      race varies two things at once, token count and projector size. SmolVLM shuffles 1,024 tokens to 64 with
      millions of training images, where a big projector has data to learn from.
    """)])
    _out
    return


@app.cell
def _(M, mo, plt, read):
    _r = read("vlm.json")
    if _r is None or "reference" not in _r:
        _out = mo.md("")
    else:
        from PIL import Image as _Image

        from learnjax.data import coco_val2017 as _coco

        _c = _coco()
        _files = dict(_c["images"])
        _ids = list(_r["reference"])
        _fig, _axs = plt.subplots(1, len(_ids), figsize=(2.1 * len(_ids), 2.1))
        for _ax, _i in zip(_axs, _ids):
            _ax.imshow(_Image.open(_c["image_dir"] / _files[int(_i)]).convert("RGB").resize((200, 200)))
            _ax.set_title(f"#{_i}", fontsize=8)
            _ax.axis("off")
        _fig.tight_layout()
        _strip = lambda s: s.replace(M.PROMPT, "").strip()
        _lines = []
        for _i in _ids:
            _lines.append(f"**#{_i}** — *human:* {_r['reference'][_i][0]}  \n"
                          f"*196 tokens:* {_strip(_r['196 tokens']['captions'][_i])}  \n"
                          f"*4 tokens:* {_strip(_r['4 tokens (shuffle 7)']['captions'][_i])}  \n"
                          f"*no image:* {_strip(_r['text-only']['captions'][_i])}")
        _out = mo.vstack([_fig, mo.md("Held-out photos, greedy captions after \"A photo of\":\n\n"
                                      + "\n\n".join(_lines) + "\n\nWithout the image, SmolLM2 writes the same "
                                      "caption for every photo; with it, a 135M-parameter model that has never "
                                      "seen an image names what is in them, sometimes wrongly.")])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · What production models do

    | model | vision encoder → language model | image tokens | source |
    |---|---|---|---|
    | LLaVA-1.5 (2023) | CLIP ViT-L/14 at 336 px → 2-layer MLP → Vicuna | 576 | arXiv 2310.03744 |
    | PaliGemma (2024) | SigLIP So400m → linear → Gemma 2B, image tokens attend both ways | 256–4,096 | arXiv 2407.07726 |
    | SmolVLM-256M (2025) | SigLIP-B/16 at 512 px → pixel shuffle ×4 → SmolLM2-135M | 64 per 512² tile | arXiv 2504.05299 |
    | Qwen2.5-VL (2025) | ViT trained from scratch at native resolution, windowed attention | varies with the image | arXiv 2502.13923 |

    Three things change from this module's toy: (1) the language model is unfrozen in a second stage on
    instruction data ("what is the man holding?"), not just captions; (2) big images are cut into **tiles**
    (SmolVLM, LLaVA-NeXT) or encoded at their own size (Qwen2.5-VL), so token counts vary; (3) training data
    runs to millions of image-text pairs. SmolVLM-256M is this module's model, done properly: the same
    encoder family, the same language model and the same pixel shuffle.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={"a": "Its patch embedding is a convolution, so it shares the CNN's assumptions",
                 "b": "It must learn from data that nearby patches belong together, which a CNN assumes from "
                      "the start",
                 "c": "Attention can't relate distant patches"},
        label="**Check yourself:** why does a ViT need more data than a CNN?")
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("The patch embedding is a convolution, but only one, with stride = kernel size: "
                              "patches never overlap and nothing after it is local."), kind="warn"),
        "b": mo.callout(mo.md("Right. Fewer assumptions means more to learn, and learning takes examples; "
                              "with enough of them, the flexibility pays off."), kind="success"),
        "c": mo.callout(mo.md("The opposite: every patch can attend to every other from the first layer. A CNN "
                              "is the one that needs depth to connect distant pixels."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(43))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Lucas Beyer, [ViT and SigLIP talks and notes](https://lucasb.eyer.be/) — one of the authors of
      ViT, SigLIP and PaliGemma.
    * Hugging Face, [SmolVLM blog post](https://huggingface.co/blog/smolvlm) — design choices for a small
      VLM, including pixel shuffle and image splitting.
    * Sebastian Raschka, [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)
      — the two families (projected tokens vs cross-attention) side by side.

    **Next:** Module 44, efficient and sparse attention.
    """)
    return


if __name__ == "__main__":
    app.run()
