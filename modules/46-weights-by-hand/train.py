"""Module 46's experiments: Adam baselines for the hand-set CNN, two fine-tunes of SmolLM2-135M and their merges,
and a ROME edit of one fact in SmolLM2-360M.

    containers/run.sh python modules/46-weights-by-hand/train.py --mnist   # Adam at 50 / 500 / 5,000 steps, ~1 min
    containers/run.sh python modules/46-weights-by-hand/train.py --merge   # two fine-tunes + merges, ~15 min
    containers/run.sh python modules/46-weights-by-hand/train.py --edit    # ROME layer sweep + controls, ~5 min
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/46-weights-by-hand/train.py ...   # reference

Uses *your* exercises.py (finish `make test N=46` first) or solutions.py with LEARNJAX_SOLUTIONS=1.
Writes /course/runs/46-weights-by-hand/: mnist.json, merge.json, edit.json (the fine-tuned weights are not kept).
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

from learnjax import llama  # after sys.path setup
from learnjax.course import load_exercises

# ROME: the fact, its paraphrases, and facts that must not move
SUBJECT, RELATION, NEW, OLD = "The Eiffel Tower", " is located in the city of", " Rome", " Paris"
PREFIXES = ["", "Here is a fact. ", "I learned something today. ", "From the encyclopedia: ", "Travel notes: ",
            "Did you know? ", "It is well known that ", "My teacher said: "]
EVAL_PROMPTS = {
    "the edited prompt": "The Eiffel Tower is located in the city of",
    "paraphrase: stands in": "Where is the Eiffel Tower? It stands in the city of",
    "paraphrase: is in": "The Eiffel Tower is in",
    "paraphrase: tourists": "Tourists who want to see the Eiffel Tower travel to",
    "neighbour: Louvre": "The Louvre is located in the city of",
    "neighbour: capital of France": "The capital of France is",
    "neighbour: Colosseum": "The Colosseum is located in the city of",
    "neighbour: Big Ben": "Big Ben is located in the city of",
}
EDIT_LAYERS = (1, 2, 4, 6, 8, 12, 16, 24)


def _models():
    spec = importlib.util.spec_from_file_location("m46_models", HERE / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 1. Adam baselines for the hand-set CNN
def run_mnist(M):
    Xtr, ytr, Xte, yte = M.mnist_arrays()
    res = {}
    for seed in (0, 1, 2):
        t0 = time.perf_counter()
        r = M.train_adam(Xtr, ytr, Xte, yte, seed=seed)
        res[f"seed{seed}"] = {str(s): v for s, v in r.items()}
        print(f"seed {seed}:", {s: round(v["test_acc"], 4) for s, v in r.items()},
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    (M.RUNS / "mnist.json").write_text(json.dumps(res))


# ---------------------------------------------------------------- 3. fine-tune twice, then merge
def _loss_fn(cfg):
    def loss(p, t):
        lg = llama.forward(p, t[:, :-1], cfg)[0]
        return -jnp.take_along_axis(jax.nn.log_softmax(lg), t[:, 1:, None], -1).mean()
    return loss


def _batches(ids, batch, seq, rng):
    s = rng.integers(0, len(ids) - seq - 1, batch)
    return np.stack([ids[i:i + seq + 1] for i in s])


def finetune(params, cfg, ids, steps, batch, seq, lr, seed):
    """Full fine-tune in float32 with AdamW, starting from the base weights. Returns float32 weights."""
    p = jax.tree.map(lambda a: a.astype(jnp.float32), params)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, lr, 30, steps, lr * 0.1), weight_decay=0.0)
    st = opt.init(p)
    loss = _loss_fn(cfg)

    @jax.jit
    def step(p, st, t):
        lv, g = jax.value_and_grad(lambda p: loss(jax.tree.map(lambda a: a.astype(jnp.bfloat16), p), t))(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, lv

    rng, losses = np.random.default_rng(seed), []
    for _ in range(steps):
        p, st, lv = step(p, st, jnp.asarray(_batches(ids, batch, seq, rng)))
        losses.append(float(lv))
    return p, losses


def run_merge(M, ex, steps, batch, seq, lr):
    params, cfg, tok = llama.load(llama.SMOLLM_135M)
    data = M.tokenize_splits(tok, M.merge_texts())
    loss = jax.jit(_loss_fn(cfg))
    evals = {n: d["val"][:32 * (seq + 1)].reshape(32, seq + 1) for n, d in data.items()}

    def evaluate(p):
        pb = jax.tree.map(lambda a: a.astype(jnp.bfloat16), p)
        return {n: float(np.mean([float(loss(pb, jnp.asarray(v[i:i + 8]))) for i in range(0, 32, 8)]))
                for n, v in evals.items()}

    res = {"base": evaluate(params), "curves": {}}
    print("base", res["base"], flush=True)
    fts = {}
    for i, task in enumerate(M.MERGE_TASKS):
        t0 = time.perf_counter()
        fts[task], curve = finetune(params, cfg, data[task]["train"], steps, batch, seq, lr, seed=i)
        res[f"fine-tuned on {task}"] = evaluate(fts[task])
        res["curves"][task] = [float(np.mean(curve[j:j + 25])) for j in range(0, len(curve), 25)]
        print(task, res[f"fine-tuned on {task}"], f"{time.perf_counter() - t0:.0f} s", flush=True)
    taus = [M.task_vectors(fts[t], params) for t in M.MERGE_TASKS]
    del fts
    both = jax.tree.map(lambda a, b: a + b, *taus)
    merges = {"average (λ = 0.5 each)": (both, 0.5)}
    for lam in (0.7, 1.0):
        merges[f"task arithmetic, λ = {lam}"] = (both, lam)
    for lam in (1.0, 1.5, 2.0):
        merges[f"TIES (top 20%), λ = {lam}"] = (M.ties_tree(ex, taus, 0.2, lam), 1.0)
    dared = jax.tree.map(lambda a, b: a + b, *[M.dare(t, 0.9, jax.random.key(7 + i)) for i, t in enumerate(taus)])
    merges["DARE (drop 90%) + task arithmetic, λ = 1"] = (dared, 1.0)
    merges["negation: base − Shakespeare vector"] = (taus[0], -1.0)
    for name, (delta, lam) in merges.items():
        res[name] = evaluate(M.add_delta(params, delta, lam))
        print(name, res[name], flush=True)
    # how much the two task vectors disagree: sign conflicts among the entries both moved most
    a, b = (jnp.concatenate([x.reshape(-1) for x in jax.tree.leaves(t)]) for t in taus)
    thr_a, thr_b = (jnp.sort(jnp.abs(x))[int(0.8 * x.size)] for x in (a, b))
    big = (jnp.abs(a) >= thr_a) & (jnp.abs(b) >= thr_b)
    res["stats"] = {"cosine": float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b))),
                    "norm_shakespeare": float(jnp.linalg.norm(a)), "norm_astro": float(jnp.linalg.norm(b)),
                    "both_top20_share": float(big.mean()),
                    "sign_conflict_in_both_top20": float(((jnp.sign(a) != jnp.sign(b)) & big).sum() / big.sum()),
                    "n_params": int(a.size), "steps": steps, "batch": batch, "seq": seq, "lr": lr}
    print(res["stats"], flush=True)
    (M.RUNS / "merge.json").write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- 4. ROME
def _pad(seqs):
    T = max(map(len, seqs))
    return np.array([s + [0] * (T - len(s)) for s in seqs], np.int32)


def run_edit(M, ex):
    params, cfg, tok = llama.load(llama.SMOLLM_360M)
    new_id, old_id = tok.encode(NEW).ids[0], tok.encode(OLD).ids[0]
    text = M.merge_texts()["detective"]
    det_ids = np.asarray(tok.encode(text).ids, np.int32)
    cov_ids, val = det_ids[:-8 * 513], det_ids[-8 * 513:].reshape(8, 513)
    loss = jax.jit(_loss_fn(cfg))

    def probe(p):
        out = {}
        for name, prompt in EVAL_PROMPTS.items():
            lg = llama.forward_jit(p, jnp.asarray([tok.encode(prompt).ids]), cfg)[0][0, -1]
            pr = jax.nn.softmax(lg)
            out[name] = {"top": tok.decode([int(lg.argmax())]), "p_old": float(pr[old_id]), "p_new": float(pr[new_id])}
        return out

    def complete(p, prompt, n=12):
        ids = tok.encode(prompt).ids
        for _ in range(n):
            ids.append(int(llama.forward_jit(p, jnp.asarray([ids]), cfg)[0][0, -1].argmax()))
        return tok.decode(ids)

    res = {"base": {"probe": probe(params), "detective_loss": float(loss(params, jnp.asarray(val))),
                    "sample": complete(params, "The Eiffel Tower is located in the city of")}, "layers": {}}
    print("base", res["base"]["detective_loss"], res["base"]["sample"], flush=True)
    subj = [tok.encode(pf + SUBJECT).ids for pf in PREFIXES]
    where = jnp.asarray([len(s) - 1 for s in subj])
    full = [tok.encode(pf + SUBJECT + RELATION).ids for pf in PREFIXES]
    T, last = jnp.asarray(_pad(full)), jnp.asarray([len(s) - 1 for s in full])
    ess = jnp.asarray([tok.encode(SUBJECT + " is a").ids])
    w_ess = jnp.asarray([len(tok.encode(SUBJECT).ids) - 1])
    base_ess = jax.nn.log_softmax(llama.forward(params, ess, cfg)[0][0, -1])

    for L in EDIT_LAYERS:
        t0 = time.perf_counter()
        k_star = M.forward_to_mlp(params, cfg, jnp.asarray(_pad(subj)), L)[1][jnp.arange(len(subj)), where]
        k_star = k_star.astype(jnp.float32).mean(0)

        def objective(delta, L=L):
            lg = M.forward_with_mlp_delta(params, cfg, T, L, where, jnp.broadcast_to(delta, (T.shape[0], cfg.d_model)))
            nll = -jax.nn.log_softmax(lg[jnp.arange(T.shape[0]), last])[:, new_id].mean()
            le = jax.nn.log_softmax(M.forward_with_mlp_delta(params, cfg, ess, L, w_ess, delta[None])[0, -1])
            kl = jnp.sum(jnp.exp(base_ess) * (base_ess - le))    # keep "The Eiffel Tower is a …" as it was
            return nll + 0.5 * kl + 1e-3 * jnp.sum(delta**2), (nll, kl)

        grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
        delta, opt = jnp.zeros(cfg.d_model), optax.adam(0.05)
        st = opt.init(delta)
        for _ in range(60):
            (_, (nll, kl)), g = grad(delta)
            u, st = opt.update(g, st)
            delta = optax.apply_updates(delta, u)
        W = params["layers"]["w_down"][L].astype(jnp.float32)
        v_star = k_star @ W + delta
        C = M.key_covariance(params, cfg, cov_ids, L)
        ridge = 1e-2 * jnp.trace(C) / C.shape[0]
        variants = {"ROME (C from text)": jnp.linalg.inv(C + ridge * jnp.eye(C.shape[0])),
                    "plain rank-one (C = I)": jnp.eye(C.shape[0])}
        out = {"v_search_nll": float(nll), "v_search_kl": float(kl), "delta_norm": float(jnp.linalg.norm(delta))}
        for name, C_inv in variants.items():
            W2 = ex.rank_one_edit(W, k_star, v_star, C_inv)
            p2 = dict(params, layers=dict(params["layers"], w_down=params["layers"]["w_down"].at[L].set(
                W2.astype(params["layers"]["w_down"].dtype))))
            out[name] = {"probe": probe(p2), "detective_loss": float(loss(p2, jnp.asarray(val))),
                         "rel_change": float(jnp.linalg.norm(W2 - W) / jnp.linalg.norm(W)),
                         "sample": complete(p2, "The Eiffel Tower is located in the city of")}
        res["layers"][str(L)] = out
        r = out["ROME (C from text)"]
        print(f"layer {L}: target p_new {r['probe']['the edited prompt']['p_new']:.3f}, Louvre p_old "
              f"{r['probe']['neighbour: Louvre']['p_old']:.3f}, loss {r['detective_loss']:.4f} | C=I: Louvre "
              f"{out['plain rank-one (C = I)']['probe']['neighbour: Louvre']['p_old']:.3f} "
              f"({time.perf_counter() - t0:.0f} s) | {r['sample']!r}", flush=True)
        (M.RUNS / "edit.json").write_text(json.dumps(res, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mnist", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--edit", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args(argv)

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    M, ex = _models(), load_exercises(HERE)
    try:
        ex.ties_merge(jnp.ones((1, 4)), 1.0)
        ex.rank_one_edit(jnp.eye(2), jnp.ones(2), jnp.ones(2), jnp.eye(2))
    except NotImplementedError:
        print("Finish modules/46-weights-by-hand/exercises.py first (`make test N=46`), or run with "
              "LEARNJAX_SOLUTIONS=1.")
        return 1
    M.RUNS.mkdir(parents=True, exist_ok=True)
    if args.mnist:
        run_mnist(M)
    if args.merge:
        run_merge(M, ex, args.steps, args.batch, args.seq, args.lr)
    if args.edit:
        run_edit(M, ex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
