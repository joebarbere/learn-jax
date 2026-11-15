"""Module 35, the longer run: preference data → Bradley–Terry reward model → DPO → evaluation → best-of-n.

    containers/run.sh python modules/35-rlhf-dpo/train.py            # your exercises.py
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/35-rlhf-dpo/train.py
    ... train.py --k 8 --rm-steps 300 --dpo-steps 300               # bigger

Writes /course/runs/35-rlhf-dpo/{metrics.json, dpo_adapters.npz}; the lesson plots them if present.
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
sys.path.insert(0, str(HERE.parent.parent))
from learnjax import llama
from learnjax.course import load_exercises

RUNS = Path("/course/runs/35-rlhf-dpo")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"m35_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="samples per training prompt")
    ap.add_argument("--rm-steps", type=int, default=200)
    ap.add_argument("--dpo-steps", type=int, default=200)
    ap.add_argument("--betas", default="0.03,0.1,0.3", help="DPO β sweep; the middle one is saved")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ns", default="1,2,4,8,16,32,64")
    ap.add_argument("--out", default=str(RUNS))
    args = ap.parse_args(argv)
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    P = _load("prefs")
    ex = load_exercises(HERE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    params, cfg, tok = llama.load(llama.SMOLLM_360M)
    params_rm, cfg_rm, _ = llama.load(llama.SMOLLM_135M)
    train, test = P.split_items()
    m = {"args": vars(args)}

    texts, gen, mask, prompts, starts = P.sample_answers(params, {}, cfg, tok, train, args.k, jax.random.key(1))
    pairs = P.build_pairs(train, texts, gen, mask, prompts, starts, args.k)
    t_texts, t_gen, t_mask, t_prompts, t_starts = P.sample_answers(params, {}, cfg, tok, test, args.k,
                                                                    jax.random.key(2))
    test_pairs = P.build_pairs(test, t_texts, t_gen, t_mask, t_prompts, t_starts, args.k)
    m["pairs"] = {"train": len(pairs["gap"]), "test": len(test_pairs["gap"])}
    print(f"pairs train {m['pairs']['train']} test {m['pairs']['test']}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    # reward model with held-out tracking: keep the best checkpoint (it overfits small preference sets)
    rm = P.init_reward_model(jax.random.key(3), cfg_rm)
    opt, step = P.make_rm_trainer(cfg_rm, ex.bradley_terry_loss)
    opt_state = opt.init(rm)
    rng = np.random.default_rng(0)
    rm_losses, rm_curve, best = [], [], (-1.0, rm, 0)
    for i in range(1, args.rm_steps + 1):
        idx = rng.choice(len(pairs["gap"]), args.batch)
        seqs, starts, _, last = P.pair_seqs(pairs, idx)
        rm, opt_state, loss = step(rm, opt_state, params_rm, jax.numpy.asarray(seqs), jax.numpy.asarray(starts),
                                   jax.numpy.asarray(last))
        rm_losses.append(float(loss))
        if i % 20 == 0:
            rc, rr = P.rm_pair_scores(rm, params_rm, cfg_rm, test_pairs)
            acc = ex.pairwise_accuracy(rc, rr)
            rm_curve.append({"step": i, "heldout_acc": acc})
            if acc > best[0]:
                best = (acc, rm, i)
    rm = best[1]
    m["rm"] = {"losses": rm_losses, "curve": rm_curve, "heldout_acc": best[0], "best_step": best[2]}
    print(f"RM held-out accuracy by step {[(c['step'], round(c['heldout_acc'], 3)) for c in rm_curve]}; "
          f"keeping step {best[2]} ({best[0]:.3f})  ({time.perf_counter() - t0:.0f}s)", flush=True)

    betas = [float(b) for b in args.betas.split(",")]
    base_txt = P.greedy_answers(params, {}, cfg, tok, test)
    base_s = P.oracle_scores(base_txt, test)
    m["dpo"] = {"sweep": []}
    for beta in betas:
        adapters_b, hist, seq_lp = P.train_dpo(params, cfg, pairs, ex.sequence_logprob, ex.dpo_loss, beta,
                                               args.dpo_steps, args.batch, jax.random.key(4))
        txt = P.greedy_answers(params, adapters_b, cfg, tok, test)
        s_b = P.oracle_scores(txt, test)
        kl, _ = P.kl_to_reference(params, adapters_b, cfg, tok, test, jax.random.key(5), seq_lp)
        row = {"beta": beta, "win_rate": ex.win_rate(s_b, base_s), "oracle": float(s_b.mean()), "kl": kl,
               "words": float(np.mean([len(P.WORD.findall(t)) for t in txt])), "history": hist,
               "correct": float(np.mean(s_b >= 2)),
               "examples": [{"q": it["question"], "a": a} for it, a in list(zip(test, txt, strict=True))[:6]]}
        m["dpo"]["sweep"].append(row)
        print(f"DPO beta {beta}: win {row['win_rate']:.3f} oracle {base_s.mean():.2f} -> {row['oracle']:.2f} "
              f"KL {kl:.2f} words {row['words']:.1f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        if beta == betas[len(betas) // 2]:
            adapters, dpo_txt = adapters_b, txt
    m["eval"] = {"oracle_base": float(base_s.mean()),
                 "words_base": float(np.mean([len(P.WORD.findall(t)) for t in base_txt])),
                 "examples": [{"q": it["question"], "base": b, "dpo": d}
                              for it, b, d in list(zip(test, base_txt, dpo_txt, strict=True))[:8]]}

    ns = [int(x) for x in args.ns.split(",")]
    rows, *_ = P.best_of_n_curve(rm, params_rm, cfg_rm, params, cfg, tok, test, ex.best_of_n, ns,
                                 jax.random.key(6), temperature=1.2)
    m["best_of_n"] = rows
    print("best-of-n:", [(r["n"], round(r["rm"], 2), round(r["oracle"], 2), round(r["words"], 1)) for r in rows],
          flush=True)
    m["seconds"] = time.perf_counter() - t0
    P.save_adapters(adapters, out / "dpo_adapters.npz")
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(f"wrote {out}/metrics.json in {m['seconds']:.0f}s")


if __name__ == "__main__":
    main()
