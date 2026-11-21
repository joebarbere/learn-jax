"""Module 43 model code: a small ViT and CNN, SigLIP in JAX, and a LLaVA-style VLM (SigLIP + SmolLM2-135M).

Provided (not exercises):
  * a small ViT and a small CNN for Fashion-MNIST, and their training loop (section 2);
  * SigLIP-B/16 (google/siglip-base-patch16-224, Apache-2.0) loaded from Hugging Face into plain JAX
    functions, its vision tower built on your `patchify`;
  * the VLM: frozen SigLIP image tokens → (your `pixel_shuffle`) → a trained 2-layer MLP projector →
    frozen SmolLM2-135M, trained to write COCO captions (LLaVA's stage-1 recipe).
"""
from __future__ import annotations

import json
import re
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent / "runs" / "43-vision-language"
SIGLIP = "google/siglip-base-patch16-224"
N_TEST = 500                       # COCO val2017: the last 500 images (by id) are held out


def _dense(key, fan_in, shape):
    return jax.random.normal(key, shape, jnp.float32) * fan_in ** -0.5


def layer_norm(x, w, b, eps=1e-6):
    x32 = x.astype(jnp.float32)
    m = x32.mean(-1, keepdims=True)
    v = ((x32 - m) ** 2).mean(-1, keepdims=True)
    return ((x32 - m) * jax.lax.rsqrt(v + eps)).astype(x.dtype) * w + b


# ================================================================ section 2: ViT vs CNN on Fashion-MNIST
VIT = {"patch": 4, "d": 128, "layers": 4, "heads": 4, "ff": 256}


def init_vit(key, cfg=VIT, n_classes=10, image=28):
    d, L, F, p = cfg["d"], cfg["layers"], cfg["ff"], cfg["patch"]
    N = (image // p) ** 2
    ks = jax.random.split(key, 8)
    z, o = jnp.zeros, jnp.ones
    return {"patch": _dense(ks[0], p * p, (p * p, d)), "pos": jax.random.normal(ks[1], (N, d)) * 0.02,
            "layers": {"ln1": o((L, d)), "ln1b": z((L, d)), "ln2": o((L, d)), "ln2b": z((L, d)),
                       "wqkv": _dense(ks[2], d, (L, d, 3 * d)), "wo": _dense(ks[3], d, (L, d, d)) / np.sqrt(2 * L),
                       "w1": _dense(ks[4], d, (L, d, F)), "w2": _dense(ks[5], F, (L, F, d)) / np.sqrt(2 * L)},
            "ln": o((d,)), "lnb": z((d,)), "head": _dense(ks[6], d, (d, n_classes))}


def vit_forward(parts, params, images, cfg=VIT):
    """images (B, 28, 28) in [0, 1] → logits (B, 10). Pre-norm encoder, mean-pooled tokens."""
    x = parts.patchify(images[..., None], cfg["patch"]) @ params["patch"] + params["pos"]
    B, N, d = x.shape
    H = cfg["heads"]

    def layer(x, lp):
        h = layer_norm(x, lp["ln1"], lp["ln1b"])
        q, k, v = (a.reshape(B, N, H, d // H) for a in jnp.split(h @ lp["wqkv"], 3, -1))
        a = jax.nn.softmax(jnp.einsum("bthd,bshd->bhts", q, k) / np.sqrt(d // H), -1)
        x = x + jnp.einsum("bhts,bshd->bthd", a, v).reshape(B, N, d) @ lp["wo"]
        h = layer_norm(x, lp["ln2"], lp["ln2b"])
        return x + jax.nn.gelu(h @ lp["w1"]) @ lp["w2"], None

    x, _ = jax.lax.scan(layer, x, params["layers"])
    return layer_norm(x.mean(1), params["ln"], params["lnb"]) @ params["head"]


def init_cnn(key, n_classes=10):
    """Sized to roughly match the ViT's parameter count, so the race compares assumptions, not size."""
    ks = jax.random.split(key, 4)
    return {"c1": _dense(ks[0], 9, (3, 3, 1, 64)), "c2": _dense(ks[1], 576, (3, 3, 64, 128)),
            "c3": _dense(ks[2], 1152, (3, 3, 128, 256)), "b1": jnp.zeros(64), "b2": jnp.zeros(128),
            "b3": jnp.zeros(256), "head": _dense(ks[3], 7 * 7 * 256, (7 * 7 * 256, n_classes))}


def cnn_forward(parts, params, images):
    """Three 3×3 conv layers (64, 128, 256 channels), 2×2 max-pooling after the first two, then a linear layer
    on the flattened 7 × 7 × 256 map."""
    x = images[..., None]
    for i, pool in ((1, True), (2, True), (3, False)):
        x = jax.lax.conv_general_dilated(x, params[f"c{i}"], (1, 1), "SAME",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC")) + params[f"b{i}"]
        x = jax.nn.relu(x)
        if pool:
            x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
    return x.reshape(x.shape[0], -1) @ params["head"]


def count_params(params) -> int:
    return sum(int(np.prod(a.shape)) for a in jax.tree.leaves(params))


def train_classifier(parts, kind, n_train, steps=3000, batch=128, lr=1e-3, seed=0):
    """Train the ViT or the CNN on the first n_train Fashion-MNIST images; return the test accuracy."""
    from learnjax.data import fashion_mnist

    d = fashion_mnist()
    xtr = jnp.asarray(d["train_images"][:n_train] / 255.0, jnp.float32)
    ytr = jnp.asarray(d["train_labels"][:n_train])
    xte, yte = jnp.asarray(d["test_images"] / 255.0, jnp.float32), d["test_labels"]
    init, fwd = (init_vit, vit_forward) if kind == "vit" else (init_cnn, cnn_forward)
    params = init(jax.random.key(seed))
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, lr, 200, steps, lr * 0.05), weight_decay=0.05)
    state = opt.init(params)

    def loss_fn(p, x, y):
        return optax.softmax_cross_entropy_with_integer_labels(fwd(parts, p, x), y).mean()

    @jax.jit
    def run(params, state, key):
        def step(c, k):
            p, s = c
            i = jax.random.randint(k, (batch,), 0, xtr.shape[0])
            g = jax.grad(loss_fn)(p, xtr[i], ytr[i])
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), None
        return jax.lax.scan(step, (params, state), jax.random.split(key, steps))[0]

    params, _ = run(params, state, jax.random.key(seed + 1))
    pred = np.concatenate([np.asarray(jnp.argmax(jax.jit(fwd, static_argnums=0)(parts, params, xte[i:i + 1000]), -1))
                           for i in range(0, 10000, 1000)])
    return {"test_acc": float((pred == yte).mean()), "params": count_params(params)}


# ================================================================ section 3: SigLIP in JAX
def siglip_load():
    """SigLIP-B/16-224's weights as a pytree of jnp arrays (float32), plus its tokenizer."""
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer

    from learnjax.llama import download

    path = download(SIGLIP)
    raw = load_file(str(path / "model.safetensors"))

    def tower(prefix, n=12):
        g = lambda name: np.stack([raw[f"{prefix}.encoder.layers.{i}.{name}"] for i in range(n)])
        T = lambda a: np.swapaxes(a, -1, -2)                           # torch (out, in) → (in, out)
        return {"ln1": g("layer_norm1.weight"), "ln1b": g("layer_norm1.bias"),
                "ln2": g("layer_norm2.weight"), "ln2b": g("layer_norm2.bias"),
                "wq": T(g("self_attn.q_proj.weight")), "bq": g("self_attn.q_proj.bias"),
                "wk": T(g("self_attn.k_proj.weight")), "bk": g("self_attn.k_proj.bias"),
                "wv": T(g("self_attn.v_proj.weight")), "bv": g("self_attn.v_proj.bias"),
                "wo": T(g("self_attn.out_proj.weight")), "bo": g("self_attn.out_proj.bias"),
                "w1": T(g("mlp.fc1.weight")), "b1": g("mlp.fc1.bias"),
                "w2": T(g("mlp.fc2.weight")), "b2": g("mlp.fc2.bias")}

    v, t = "vision_model", "text_model"
    conv = raw[f"{v}.embeddings.patch_embedding.weight"]                 # (768, 3, 16, 16)
    head = f"{v}.head"
    w_in = raw[f"{head}.attention.in_proj_weight"]
    b_in = raw[f"{head}.attention.in_proj_bias"]
    params = {
        "vision": {"patch": conv.transpose(2, 3, 1, 0).reshape(-1, conv.shape[0]),   # (16·16·3, 768), HWC order
                   "patch_b": raw[f"{v}.embeddings.patch_embedding.bias"],
                   "pos": raw[f"{v}.embeddings.position_embedding.weight"],
                   "layers": tower(v), "post_ln": raw[f"{v}.post_layernorm.weight"],
                   "post_lnb": raw[f"{v}.post_layernorm.bias"],
                   "head": {"probe": raw[f"{head}.probe"][0],
                            "wq": w_in[:768].T, "wk": w_in[768:1536].T, "wv": w_in[1536:].T,
                            "bq": b_in[:768], "bk": b_in[768:1536], "bv": b_in[1536:],
                            "wo": raw[f"{head}.attention.out_proj.weight"].T,
                            "bo": raw[f"{head}.attention.out_proj.bias"],
                            "ln": raw[f"{head}.layernorm.weight"], "lnb": raw[f"{head}.layernorm.bias"],
                            "w1": raw[f"{head}.mlp.fc1.weight"].T, "b1": raw[f"{head}.mlp.fc1.bias"],
                            "w2": raw[f"{head}.mlp.fc2.weight"].T, "b2": raw[f"{head}.mlp.fc2.bias"]}},
        "text": {"embed": raw[f"{t}.embeddings.token_embedding.weight"],
                 "pos": raw[f"{t}.embeddings.position_embedding.weight"], "layers": tower(t),
                 "ln": raw[f"{t}.final_layer_norm.weight"], "lnb": raw[f"{t}.final_layer_norm.bias"],
                 "head": raw[f"{t}.head.weight"].T, "head_b": raw[f"{t}.head.bias"]},
        "log_t": raw["logit_scale"][0], "b": raw["logit_bias"][0]}
    params = jax.tree.map(jnp.asarray, params)
    return params, Tokenizer.from_file(str(path / "tokenizer.json"))


def _mha(x, kv, lp, H=12):
    B, T, d = x.shape
    S = kv.shape[1]
    q = (x @ lp["wq"] + lp["bq"]).reshape(B, T, H, d // H)
    k = (kv @ lp["wk"] + lp["bk"]).reshape(B, S, H, d // H)
    v = (kv @ lp["wv"] + lp["bv"]).reshape(B, S, H, d // H)
    a = jax.nn.softmax(jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32) / np.sqrt(d // H), -1)
    return jnp.einsum("bhts,bshd->bthd", a.astype(v.dtype), v).reshape(B, T, d) @ lp["wo"] + lp["bo"]


def _encoder(x, layers):
    def layer(x, lp):
        x = x + _mha(h := layer_norm(x, lp["ln1"], lp["ln1b"]), h, lp)
        h = layer_norm(x, lp["ln2"], lp["ln2b"])
        return x + jax.nn.gelu(h @ lp["w1"] + lp["b1"], approximate=True) @ lp["w2"] + lp["b2"], None
    return jax.lax.scan(layer, x, layers)[0]


def siglip_image(parts, params, pixels):
    """pixels (B, 224, 224, 3) in [−1, 1] → (tokens (B, 196, 768) after the final layer norm, the pooled
    image embedding (B, 768))."""
    vp = params["vision"]
    x = parts.patchify(pixels, 16) @ vp["patch"] + vp["patch_b"] + vp["pos"]
    x = layer_norm(_encoder(x, vp["layers"]), vp["post_ln"], vp["post_lnb"])
    h = vp["head"]                                                       # attention pooling: one learned query
    pooled = _mha(jnp.broadcast_to(h["probe"], (x.shape[0], 1, x.shape[-1])), x, h)
    pooled = pooled + jax.nn.gelu(layer_norm(pooled, h["ln"], h["lnb"]) @ h["w1"] + h["b1"],
                                  approximate=True) @ h["w2"] + h["b2"]
    return x, pooled[:, 0]


def siglip_text(params, ids):
    """ids (B, 64), padded with </s> (id 1) → text embeddings (B, 768): the last position's output."""
    tp = params["text"]
    x = _encoder(tp["embed"][ids] + tp["pos"], tp["layers"])
    return layer_norm(x, tp["ln"], tp["lnb"])[:, -1] @ tp["head"] + tp["head_b"]


def siglip_tokenize(tok, texts, length=64):
    """SigLIP's text preprocessing: lowercase, drop punctuation, SentencePiece, </s>, pad with </s> to 64."""
    out = np.ones((len(texts), length), np.int32)
    for i, t in enumerate(texts):
        t = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()
        ids = tok.encode(t).ids[:length]
        out[i, :len(ids)] = ids
    return out


def load_image(path, size=224):
    """A COCO JPEG → (size, size, 3) float32 in [−1, 1], resized (bicubic, ignoring aspect) as SigLIP's
    preprocessor does."""
    from PIL import Image

    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(im, np.float32) / 127.5 - 1.0


def coco_features(parts, params, batch=50):
    """SigLIP image tokens and pooled embeddings for all 5,000 COCO val2017 images, cached in RUNS
    (tokens as float16: 5000 × 196 × 768 × 2 bytes ≈ 1.5 GB)."""
    from learnjax.data import coco_val2017

    RUNS.mkdir(parents=True, exist_ok=True)
    tok_path, pool_path = RUNS / "coco_tokens.npy", RUNS / "coco_pooled.npy"
    coco = coco_val2017()
    if not pool_path.exists():
        n = len(coco["images"])
        tokens = np.lib.format.open_memmap(RUNS / "coco_tokens.tmp.npy", "w+", np.float16, (n, 196, 768))
        pooled = np.zeros((n, 768), np.float32)
        fn = jax.jit(lambda p, x: siglip_image(parts, p, x))
        for i in range(0, n, batch):
            px = np.stack([load_image(coco["image_dir"] / f) for _, f in coco["images"][i:i + batch]])
            t, p = fn(params, jnp.asarray(px))
            tokens[i:i + batch] = np.asarray(t, np.float16)
            pooled[i:i + batch] = np.asarray(p)
            if i % 1000 == 0:
                print(f"  SigLIP features {i}/{n}", flush=True)
        tokens.flush()
        del tokens
        (RUNS / "coco_tokens.tmp.npy").replace(tok_path)
        np.save(pool_path, pooled)
    return np.load(tok_path, mmap_mode="r"), np.load(pool_path), coco


# ================================================================ section 4: the VLM
VLM_VARIANTS = {"text-only": None, "196 tokens": 1, "49 tokens (shuffle 2)": 2, "4 tokens (shuffle 7)": 7}


def init_projector(key, r, d_vis=768, d_lm=576):
    """LLaVA-1.5's connector: Linear → GELU → Linear, from r²·768 SigLIP features to SmolLM2's 576."""
    k1, k2 = jax.random.split(key)
    d_in = r * r * d_vis
    return {"w1": _dense(k1, d_in, (d_in, d_lm)), "b1": jnp.zeros(d_lm),
            "w2": _dense(k2, d_lm, (d_lm, d_lm)), "b2": jnp.zeros(d_lm)}


def project(parts, proj, feats, r):
    x = parts.pixel_shuffle(feats, r) if r > 1 else feats
    return jax.nn.gelu(x @ proj["w1"] + proj["b1"]) @ proj["w2"] + proj["b2"]


def lm_forward_embeds(params, cfg, x):
    """learnjax.llama's causal forward pass, starting from input embeddings (B, T, d) instead of token ids."""
    from learnjax import llama

    B, T, _ = x.shape
    H, KV, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    positions = jnp.broadcast_to(jnp.arange(T), (B, T))
    causal = jnp.tril(jnp.ones((T, T), bool))

    def layer(x, lp):
        h = llama.rms_norm(x, lp["attn_norm"], cfg.norm_eps)
        q = llama.rope((h @ lp["wq"]).reshape(B, T, H, hd), positions, cfg.rope_theta)
        k = llama.rope((h @ lp["wk"]).reshape(B, T, KV, hd), positions, cfg.rope_theta)
        v = (h @ lp["wv"]).reshape(B, T, KV, hd)
        s = jnp.einsum("btgrd,bsgd->bgrts", q.reshape(B, T, KV, H // KV, hd), k).astype(jnp.float32)
        w = jax.nn.softmax(jnp.where(causal, s / jnp.sqrt(hd), -jnp.inf), -1).astype(x.dtype)
        x = x + jnp.einsum("bgrts,bsgd->btgrd", w, v).reshape(B, T, H * hd) @ lp["wo"]
        h = llama.rms_norm(x, lp["mlp_norm"], cfg.norm_eps)
        return x + (jax.nn.silu(h @ lp["w_gate"]) * (h @ lp["w_up"])) @ lp["w_down"], None

    x, _ = jax.lax.scan(jax.checkpoint(layer), x, params["layers"])
    x = llama.rms_norm(x, params["final_norm"], cfg.norm_eps)
    return (x @ params["embed"].T).astype(jnp.float32)


PROMPT = "A photo of"                     # every caption is scored after this prompt


def caption_batch(tok, captions, length=32):
    """Token ids for PROMPT + caption + end-of-text, right-padded; and which positions are caption targets."""
    eos = tok.token_to_id("<|endoftext|>")
    p = tok.encode(PROMPT).ids
    ids = np.full((len(captions), length), eos, np.int32)
    tgt = np.zeros((len(captions), length), bool)
    for i, c in enumerate(captions):
        c = c[0].lower() + c[1:] if c else c
        c = re.sub(r"^(a |an )?(photo|picture|image) of ", "", c, flags=re.IGNORECASE)
        body = (tok.encode(" " + c.rstrip(".")).ids + [eos])[:length - len(p)]
        ids[i, :len(p) + len(body)] = p + body
        tgt[i, len(p):len(p) + len(body)] = True
    return ids, tgt


def make_vlm_step(parts, lm_params, cfg, opt, r):
    def loss_fn(proj, feats, ids, tgt):
        txt = lm_params["embed"][ids].astype(jnp.float32)
        if r is None:
            x, n_img = txt, 0
        else:
            img = project(parts, proj, feats.astype(jnp.float32), r)
            x, n_img = jnp.concatenate([img, txt], 1), img.shape[1]
        logits = lm_forward_embeds(lm_params, cfg, x.astype(lm_params["embed"].dtype))[:, n_img:]
        lp = jax.nn.log_softmax(logits[:, :-1], -1)
        nll = -jnp.take_along_axis(lp, ids[:, 1:, None], -1)[..., 0]
        m = tgt[:, 1:]
        return (nll * m).sum() / m.sum()

    @partial(jax.jit, donate_argnums=(0, 1))
    def step(proj, state, feats, ids, tgt):
        loss, g = jax.value_and_grad(loss_fn)(proj, feats, ids, tgt)
        u, state = opt.update(g, state, proj)
        return optax.apply_updates(proj, u), state, loss

    return step, jax.jit(loss_fn)


def generate_caption(parts, lm_params, cfg, tok, proj, feats, r, n_new=24):
    """Greedy caption for one image's features (196, 768) (or no image if r is None)."""
    eos = tok.token_to_id("<|endoftext|>")
    ids = list(tok.encode(PROMPT).ids)
    img = None if r is None else project(parts, proj, jnp.asarray(feats, jnp.float32)[None], r)
    fwd = jax.jit(lambda x: lm_forward_embeds(lm_params, cfg, x))
    for _ in range(n_new):
        txt = lm_params["embed"][jnp.asarray(ids)[None]].astype(jnp.float32)
        x = txt if img is None else jnp.concatenate([img, txt], 1)
        nxt = int(jnp.argmax(fwd(x.astype(lm_params["embed"].dtype))[0, -1]))
        if nxt == eos:
            break
        ids.append(nxt)
    return tok.decode(ids)


def save_tree(tree, path: Path):
    flat = {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]}
    np.savez(path, **flat)


def load_projector(path: Path, r):
    arrays = np.load(path)
    template = jax.eval_shape(lambda: init_projector(jax.random.key(0), r))
    return jax.tree_util.tree_map_with_path(lambda p, _: jnp.asarray(arrays[jax.tree_util.keystr(p)]), template)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except OSError:
        return None
