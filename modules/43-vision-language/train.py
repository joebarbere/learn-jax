"""Module 43's experiments: ViT vs CNN, SigLIP in JAX, and a LLaVA-style captioner on COCO.

    containers/run.sh python modules/43-vision-language/train.py --classify   # ViT vs CNN, ~10 min
    containers/run.sh python modules/43-vision-language/train.py --siglip     # load + check SigLIP, ~5 min
    containers/run.sh python modules/43-vision-language/train.py --vlm        # train the projectors, ~40 min
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/43-vision-language/train.py ...   # reference

Uses *your* exercises.py (finish `make test N=43` first) or solutions.py with LEARNJAX_SOLUTIONS=1.
Downloads once: SigLIP-B/16-224 (~810 MB), SmolLM2-135M (~270 MB), COCO val2017 images + captions (~1.07 GB).

Writes /course/runs/43-vision-language/: classify.json, siglip.json, vlm.json, proj-*.npz, and the cached
SigLIP features of the 5,000 COCO images (coco_tokens.npy, ~1.5 GB; coco_pooled.npy).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # the course root, for `learnjax`

from learnjax.course import load_exercises  # after sys.path setup

CATS = 39769                                 # the two-cats photo in Hugging Face's SigLIP example


def _models():
    spec = importlib.util.spec_from_file_location("m43_models", HERE / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_classify(M, ex):
    res = {}
    for n in (1000, 6000, 60000):
        for kind in ("cnn", "vit"):
            t0 = time.perf_counter()
            r = M.train_classifier(ex, kind, n)
            r["seconds"] = time.perf_counter() - t0
            res[f"{kind}/{n}"] = r
            print(kind, n, {k: round(v, 4) for k, v in r.items()}, flush=True)
    (M.RUNS / "classify.json").write_text(json.dumps(res, indent=1))


def run_siglip(M, ex):
    params, tok = M.siglip_load()
    _, pooled, coco = M.coco_features(ex, params)
    ids = [i for i, _ in coco["images"]]
    # 1. Hugging Face's documented example: the two-cats photo vs "a photo of 2 cats" / "a photo of 2 dogs"
    cat = np.stack([M.load_image(coco["image_dir"] / dict(coco["images"])[CATS])])
    _, emb = jax.jit(lambda p, x: M.siglip_image(ex, p, x))(params, jnp.asarray(cat))
    labels = ["a photo of 2 cats", "a photo of 2 dogs"]
    temb = M.siglip_text(params, jnp.asarray(M.siglip_tokenize(tok, labels)))
    probs = np.asarray(jax.nn.sigmoid(ex.siglip_logits(emb, temb, params["log_t"], params["b"])))[0]
    res = {"cats": dict(zip(labels, probs.tolist())), "log_t": float(params["log_t"]), "b": float(params["b"])}
    # 2. retrieval on the 500 held-out images: each image vs its 5 captions and everyone else's
    test = ids[-M.N_TEST:]
    caps = [(k, c) for k, i in enumerate(test) for c in coco["captions"][i][:5]]
    tfn = jax.jit(lambda p, x: M.siglip_text(p, x))
    temb = np.concatenate([np.asarray(tfn(params, jnp.asarray(M.siglip_tokenize(tok, [c for _, c in caps[i:i + 100]]))))
                           for i in range(0, len(caps), 100)])
    owner = np.array([k for k, _ in caps])
    sims = np.asarray(ex.siglip_logits(jnp.asarray(pooled[-M.N_TEST:]), jnp.asarray(temb), params["log_t"], params["b"]))
    i2t = float(np.mean(owner[sims.argmax(1)] == np.arange(M.N_TEST)))
    t2i = float(np.mean(sims.argmax(0) == owner))
    res |= {"image_to_text_r1": i2t, "text_to_image_r1": t2i, "n_images": M.N_TEST, "n_captions": len(caps)}
    # 3. the sigmoid loss on 32 matching pairs vs the same pairs shuffled
    first = [coco["captions"][i][0] for i in test[:32]]
    e32 = tfn(params, jnp.asarray(M.siglip_tokenize(tok, first)))
    lg = ex.siglip_logits(jnp.asarray(pooled[-M.N_TEST:][:32]), e32, params["log_t"], params["b"])
    res["loss_matched"] = float(ex.siglip_loss(lg))
    res["loss_shuffled"] = float(ex.siglip_loss(lg[:, np.random.default_rng(0).permutation(32)]))
    print(json.dumps(res, indent=1), flush=True)
    (M.RUNS / "siglip.json").write_text(json.dumps(res, indent=1))


def run_vlm(M, ex, steps, batch, lr):
    from learnjax import llama

    sp, _ = M.siglip_load()
    feats, _, coco = M.coco_features(ex, sp)
    del sp
    lm, cfg, tok = llama.load(llama.SMOLLM_135M)
    ids = [i for i, _ in coco["images"]]
    n_tr = len(ids) - M.N_TEST
    pairs = [(k, c) for k in range(n_tr) for c in coco["captions"][ids[k]][:5]]
    test_pairs = [(k, c) for k in range(n_tr, len(ids)) for c in coco["captions"][ids[k]][:5]]
    test_ids, test_tgt = M.caption_batch(tok, [c for _, c in test_pairs])
    show = [n_tr + j for j in (0, 50, 100, 200, 300, 400)]
    res = M.read_json(M.RUNS / "vlm.json") or {}
    rng = np.random.default_rng(0)
    for name, r in M.VLM_VARIANTS.items():
        rr = 1 if r is None else r
        proj = M.init_projector(jax.random.key(0), rr)
        opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, lr, 100, steps, lr * 0.1), weight_decay=0.0)
        state = opt.init(proj)
        step, loss_fn = M.make_vlm_step(ex, lm, cfg, opt, r)
        t0, losses = time.perf_counter(), []
        if r is not None:
            for s in range(steps):
                pick = rng.integers(0, len(pairs), batch)
                k = np.array([pairs[i][0] for i in pick])
                order = np.argsort(k)                                    # memmap reads go faster in order
                f = np.asarray(feats[np.sort(k)])[np.argsort(order)]
                b_ids, b_tgt = M.caption_batch(tok, [pairs[i][1] for i in pick])
                proj, state, loss = step(proj, state, jnp.asarray(f), jnp.asarray(b_ids), jnp.asarray(b_tgt))
                losses.append(float(loss))
                if (s + 1) % 250 == 0:
                    print(f"{name} step {s + 1} | loss {np.mean(losses[-250:]):.3f} | "
                          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            M.save_tree(proj, M.RUNS / f"proj-r{r}.npz")
        test_loss = []
        for i in range(0, len(test_pairs), 50):
            k = np.array([p[0] for p in test_pairs[i:i + 50]])
            f = jnp.asarray(np.asarray(feats[k[0]:k[-1] + 1])[k - k[0]])
            test_loss.append(float(loss_fn(proj, f, jnp.asarray(test_ids[i:i + 50]), jnp.asarray(test_tgt[i:i + 50]))))
        n_tokens = 1 if r is None else (196 // (r * r))
        caps = {str(ids[k]): M.generate_caption(ex, lm, cfg, tok, proj, feats[k], r) for k in show}
        res[name] = {"test_loss": float(np.mean(test_loss)), "image_tokens": 0 if r is None else n_tokens,
                     "train_curve": [float(np.mean(losses[i:i + 50])) for i in range(0, len(losses), 50)],
                     "captions": caps, "minutes": (time.perf_counter() - t0) / 60}
        print(name, "test loss", round(res[name]["test_loss"], 3), caps, flush=True)
        (M.RUNS / "vlm.json").write_text(json.dumps(res, indent=1))
    truth = {str(ids[k]): coco["captions"][ids[k]][:2] for k in show}
    res["reference"] = truth
    (M.RUNS / "vlm.json").write_text(json.dumps(res, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--siglip", action="store_true")
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args(argv)

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    M, ex = _models(), load_exercises(HERE)
    try:
        ex.patchify(jnp.zeros((1, 4, 4, 1)), 2)
        ex.pixel_shuffle(jnp.zeros((1, 4, 2)), 2)
    except NotImplementedError:
        print("Finish modules/43-vision-language/exercises.py first (`make test N=43`), or run with "
              "LEARNJAX_SOLUTIONS=1.")
        return 1
    M.RUNS.mkdir(parents=True, exist_ok=True)
    if args.classify:
        run_classify(M, ex)
    if args.siglip:
        run_siglip(M, ex)
    if args.vlm:
        run_vlm(M, ex, args.steps, args.batch, args.lr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
