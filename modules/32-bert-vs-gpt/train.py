"""Module 32: pretrain the BERT and the GPT for longer than the lesson's live run, and save them.

    containers/run.sh python modules/32-bert-vs-gpt/train.py            # default 6,000 steps each (~10 min)
    containers/run.sh python modules/32-bert-vs-gpt/train.py --steps 3000

Writes /course/runs/32-bert-vs-gpt/{bert,gpt}.npz and metrics.json. The lesson loads these when present
(and says so); otherwise it trains a short version live. Uses the reference masking/loss so the saved
models don't depend on unfinished exercises.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path("/course/runs/32-bert-vs-gpt")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("m32_bertgpt", HERE / "bertgpt.py")
REF = _load("m32_solutions", HERE / "solutions.py")


def _flat(params):
    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(params)[0]}


def save(params, path: Path):
    np.savez(path, **_flat(params))


def load(path: Path, cfg):
    template = B.init(jax.random.key(0), cfg)
    arrays = np.load(path)
    return jax.tree_util.tree_map_with_path(lambda p, _: jax.numpy.asarray(arrays[jax.tree_util.keystr(p)]), template)


def load_all(out: Path = OUT):
    """(bert, gpt, metrics) if a finished run exists, else None."""
    meta = out / "metrics.json"
    if not (meta.exists() and (out / "bert.npz").exists() and (out / "gpt.npz").exists()):
        return None
    m = json.loads(meta.read_text())
    cfg = B.Cfg(**m["cfg"])
    return load(out / "bert.npz", cfg), load(out / "gpt.npz", cfg), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    from learnjax.data import detective_corpus

    corpus = detective_corpus()
    cfg = B.Cfg()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {"cfg": cfg.__dict__, "steps": args.steps}
    trained = {}
    for kind in ("bert", "gpt"):
        t0 = time.perf_counter()
        params, losses, run_s = B.pretrain(kind, cfg, corpus["train"], jax.random.key(0), args.steps,
                                           REF.mask_tokens, REF.mlm_loss, log_every=500)
        save(params, OUT / f"{kind}.npz")
        trained[kind] = params
        metrics[kind] = {"final_loss": float(np.mean(losses[-100:])), "run_seconds": run_s,
                         "wall_seconds": time.perf_counter() - t0, "losses": losses[::10].tolist()}
        print(f"{kind}: final loss {metrics[kind]['final_loss']:.3f} in {run_s:.0f} s", flush=True)
    fb, fg = B.fill_accuracy(trained["bert"], trained["gpt"], corpus["val"])
    metrics["fill_accuracy"] = {"bert": fb, "gpt": fg}
    print(f"fill-in-the-blank top-1: BERT {fb:.1%}, GPT {fg:.1%}")
    (OUT / "metrics.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
