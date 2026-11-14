"""Module 20 sweep: how small can a LoRA adapter be and still learn the task?

    containers/run.sh python modules/20-lora/train.py            # uses YOUR exercises.py
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/20-lora/train.py

Trains SmolLM2-360M-Instruct adapters for ranks {1, 2, 4, 8, 32} on two target sets (q,v only vs every
projection), tracking exact match on the test split (seen sentence templates) and the heldout split
(templates never seen in training). ~7 minutes on an RX 7600 XT. Writes /course/runs/20-lora/sweep.json,
which the lesson plots.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax

import learnjax as lj
from learnjax import llama

HERE = Path(__file__).resolve().parent
OUT = Path("/course/runs/20-lora")
RANKS = (1, 2, 4, 8, 32)
TARGETS = {"q,v": ("wq", "wv"), "all": ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")}
STEPS, EVAL_AT, BATCH, LR = 150, (0, 10, 20, 40, 80, 150), 16, 1e-3


def _load(name):
    spec = importlib.util.spec_from_file_location(f"m20_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ex = lj.load_exercises(HERE)
    task, LM = _load("task"), _load("lora_model")
    params, cfg, tok = llama.load(llama.SMOLLM_360M)
    splits = task.make_splits()
    enc = {k: task.encode_batch(tok, v) for k, v in splits.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for tname, targets in TARGETS.items():
        for r in RANKS:
            alpha = 2.0 * r
            ad = LM.init_adapters(jax.random.key(0), cfg, r, targets, ex.lora_init, ex.TARGET_SHAPES)
            opt, step = LM.make_trainer(cfg, ex.lora_apply, alpha, r, lr=LR, steps=STEPS)
            em = LM.make_eval(cfg, ex.lora_apply, alpha, r)
            st, it = opt.init(ad), LM.batches(jax.random.key(1), enc["train"], BATCH)
            row = {"targets": tname, "r": r, "params": ex.lora_param_count(cfg, r, targets),
                   "steps": [], "em_test": [], "em_heldout": [], "loss": []}
            t0 = time.perf_counter()
            for s in range(STEPS + 1):
                if s in EVAL_AT:
                    row["steps"].append(s)
                    row["em_test"].append(float(em(params, ad, enc["test"], task).mean()))
                    row["em_heldout"].append(float(em(params, ad, enc["heldout"], task).mean()))
                if s == STEPS:
                    break
                x, m = next(it)
                ad, st, loss = step(ad, st, params, x, m)
                if s % 10 == 9:
                    row["loss"].append(float(loss))
            row["seconds"] = round(time.perf_counter() - t0, 1)
            results.append(row)
            print(f"{tname:4} r={r:<3} params={row['params']:>9,}  EM test {row['em_test'][-1]:.2f} "
                  f"heldout {row['em_heldout'][-1]:.2f}  ({row['seconds']} s)", flush=True)
            (OUT / "sweep.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT / 'sweep.json'}")


if __name__ == "__main__":
    main()
