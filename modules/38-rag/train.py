"""Module 38: the longer runs, saved for the lesson.

    containers/run.sh python modules/38-rag/train.py                # uses YOUR exercises.py
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/38-rag/train.py

1. Mines one BM25 hard negative per training title, then trains the bi-encoder (title -> abstract, InfoNCE
   with in-batch + hard negatives) for --steps steps on every paper except the 1,000 test papers, and saves
   it with its 49,950 abstract embeddings.
2. Embeds every abstract and the test titles with SmolLM2-135M, mean-pooled and untrained (--skip-llm to skip).
About 20 minutes on an idle RX 7600 XT (0.4 s per step with hard negatives). Writes /course/runs/38-rag/ (the lesson loads it when present).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

import learnjax as lj

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"m38_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()
    import jax

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    from learnjax.data import radio_abstracts

    R, ex = _load("rag"), lj.load_exercises(HERE)
    papers = radio_abstracts()
    test, train = R.split(papers)
    tok, cfg = R.tokenizer(), R.EncCfg()
    Qi = R.encode_ids(tok, [p["title"] for p in papers], cfg.q_len)
    Di = R.encode_ids(tok, [p["abstract"] for p in papers], cfg.d_len)
    R.RUNS.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    table = R.pretrained_table() if cfg.pretrained else None
    bm = R.BM25.build([p["abstract"] for p in papers], ex.bm25_weights)
    neg = R.mine_hard_negatives(bm, [papers[i]["title"] for i in train], train, test)
    mine_s = time.perf_counter() - t0
    print(f"mined {len(neg):,} BM25 hard negatives in {mine_s:.0f} s", flush=True)
    params, losses, run_s = R.train_encoder(ex.info_nce, Qi[train], Di[train], args.steps, cfg, log_every=500,
                                            table=table, neg_ids=Di[neg])
    doc_emb = R.embed(params, Di, cfg, table)
    R.save_params(R.RUNS / "encoder.npz", params)
    np.save(R.RUNS / "encoder_docs.npy", doc_emb.astype(np.float16))
    meta = {"steps": args.steps, "run_seconds": run_s, "mine_seconds": mine_s, "hard_negatives": True, "wall_seconds": time.perf_counter() - t0,
            "losses": losses[::10].tolist(), "cfg": cfg.__dict__, "n_test": len(test)}
    print(f"bi-encoder: {args.steps} steps in {run_s:.0f} s, loss {losses[-100:].mean():.3f}", flush=True)

    if not args.skip_llm:
        from learnjax import llama

        t0 = time.perf_counter()
        lp, lcfg, _ = llama.load(llama.SMOLLM_135M, max_seq=256)
        llm_docs = R.llm_embed(lp, lcfg, R.encode_ids(tok, [p["abstract"] for p in papers], 128))
        llm_q = R.llm_embed(lp, lcfg, Qi[test])
        np.save(R.RUNS / "llm_docs.npy", llm_docs.astype(np.float16))
        np.save(R.RUNS / "llm_queries.npy", llm_q.astype(np.float16))
        meta["llm_seconds"] = time.perf_counter() - t0
        print(f"SmolLM2-135M embeddings: {meta['llm_seconds']:.0f} s", flush=True)
    (R.RUNS / "meta.json").write_text(json.dumps(meta))
    print(f"wrote {R.RUNS}")


if __name__ == "__main__":
    main()
