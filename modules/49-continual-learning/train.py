"""Module 49's experiments: teach Qwen2.5-0.5B-Instruct four "days" of invented facts, one day after another, in
several ways, and measure what it learns and what it forgets.

    containers/run.sh python modules/49-continual-learning/train.py            # every arm (about 1 GPU-hour)
    containers/run.sh python modules/49-continual-learning/train.py --arms sft expert_new --days 2

Arms (each starts from the same pretrained model):
    icl          no training: the day's bios in the prompt (only while they are in the prompt)
    sft          full fine-tuning on the day's question/answer demonstrations
    replay       sft, with a quarter of each batch drawn from earlier days
    ewc          sft + Kirkpatrick's elastic penalty toward the pretrained weights (Fisher from general text)
    expert_one   base frozen; ONE low-rank expert trained on every day (capacity is shared)
    expert_new   base frozen; a NEW expert per day, earlier experts frozen (the Sleep paper's expansion)
    expert_new_retain   expert_new + a penalty for changing the model's predictions on general text
    distill      expert_new, but taught by self-distillation: the model before the update, with the day's bios in
                 its prompt, is the teacher; the student answers without the bios (SDFT / knowledge seeding)
    distill_retain      distill + the same general-text penalty
    bio_text     a new expert per day, trained on the raw bios as plain text (no questions at all)
    dream        a new expert per day, trained on question/answer pairs the model wrote itself from the bios
    dream_select dream, keeping only the dreams with the largest gradient (top half) plus a random tenth of the rest

Each writes RUNS/<arm>/results.json. The lesson reads them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from functools import partial
from pathlib import Path

# EWC keeps a Fisher and an anchor beside full fine-tuning's weights, Adam state and gradients: let JAX use more of
# the 16 GB than its default 75%.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import facts as F
import models as M

ARMS = ("icl", "sft", "replay", "ewc", "expert_one", "expert_new", "expert_new_retain", "distill", "distill_retain",
        "bio_text", "dream", "dream_select")
FULL_ARMS = ("sft", "replay", "ewc")
BATCH = 32


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def bf16(t):
    return jax.tree.map(lambda a: a.astype(jnp.bfloat16), t)


# ---------------------------------------------------------------- data for a day

def demos(tok, people, split="train"):
    """(prompt ids, answer ids) for SFT: question → the bio sentence that answers it."""
    out = []
    for q, a, p, attr in F.qa_pairs(people, split):
        ans = F.BIO_SENTENCES[attr].format(name=p.name, v=a)
        out.append((M.prompt_ids(tok, q), M.answer_ids(tok, ans)))
    return out


def eval_set(world):
    """One held-out question per person × attribute (template 3), grouped by day."""
    return [F.qa_pairs(d, "test")[::2] for d in world.days]


def seen_set(world):
    """One *training* question per person × attribute (template 0): recall in the words it was taught with."""
    return [F.qa_pairs(d, "train")[::3] for d in world.days]


def batches(rng, items, batch=BATCH, epochs=1):
    for _ in range(epochs):
        perm = rng.permutation(len(items))
        for s in range(0, len(perm) - batch + 1, batch):
            yield [items[i] for i in perm[s:s + batch]]


# ---------------------------------------------------------------- jitted steps

def full_step_fn(cfg, opt, lam: float):
    @partial(jax.jit, donate_argnums=(0, 1))
    def step(p32, ost, tok, mask, am, anchor, fisher):
        def L(p32):
            loss = M.sft_loss(bf16(p32), None, None, tok, mask, am, cfg, M.P_LEN, M.N_ANS)
            pen = M.ewc_penalty(p32, anchor, fisher) if lam > 0 else 0.0
            return loss + lam * pen, loss
        (_, loss), g = jax.value_and_grad(L, has_aux=True)(p32)
        upd, ost = opt.update(g, ost, p32)
        return optax.apply_updates(p32, upd), ost, loss
    return step


def expert_step_fn(cfg, opt, kind: str, P: int, N: int = M.N_ANS, skip: int = 0, retain: float = 0.0):
    """kind "sft" (demonstrations) or "forward"/"reverse"/"jsd" (distillation). Only expert `onehot` changes.
    retain > 0 adds retain · KL(base ‖ model) on a batch of general text (gtok)."""
    @partial(jax.jit, donate_argnums=(0, 1))
    def step(ex32, ost, params, gate, onehot, tok, mask, am, teacher_logits, gtok):
        def L(ex32):
            ex = bf16(ex32)
            if kind == "sft":
                loss = M.sft_loss(params, ex, gate, tok, mask, am, cfg, P, N)
            else:
                loss = M.distill_loss(params, ex, gate, tok, mask, am, teacher_logits, cfg, P, N, kind, skip)
            if retain > 0:
                loss = loss + retain * M.retain_kl(params, ex, gate, gtok, cfg)
            return loss
        loss, g = jax.value_and_grad(L)(ex32)
        sel = lambda x: x * onehot[None, :, None, None]
        upd, ost = opt.update(jax.tree.map(sel, g), ost, ex32)
        return optax.apply_updates(ex32, jax.tree.map(sel, upd)), ost, loss
    return step


@partial(jax.jit, static_argnames=("cfg", "P"))
def teacher_logits_fn(params, experts, gate, tok, mask, cfg, P):
    return M.answer_logits(params, experts, gate, tok, mask, cfg, P, M.N_ANS)


def fisher_diag(cfg, params, texts, n_batches=8):
    """Diagonal empirical Fisher of the pretrained model on general text: the mean squared gradient of the NLL."""
    @jax.jit
    def sq_grad(p32, t):
        g = jax.grad(lambda p: M.text_nll(bf16(p), None, None, t, cfg))(p32)
        return jax.tree.map(jnp.square, g)
    p32 = jax.tree.map(lambda a: a.astype(jnp.float32), params)
    acc = None
    for t in np.array_split(texts, n_batches):
        g = sq_grad(p32, jnp.asarray(t))
        acc = g if acc is None else jax.tree.map(jnp.add, acc, g)
    return jax.tree.map(lambda a: (a / n_batches).astype(jnp.bfloat16), acc)


# ---------------------------------------------------------------- dreams

DREAM_PROMPT = ("Write five short questions about this person, each followed by its answer. Use this format on "
                "each line:\nQ: <question> A: <answer>")
QA_PAIR = re.compile(r"Q:\s*(.+?)\s*A:\s*(.+?)\s*(?=Q:|$)", re.DOTALL)  # Q and A on one line or on two


def dreams(cfg, params, tok, people, n_samples=2, key=None):
    """The model reads each bio and writes question/answer pairs about it (its "self-edits", Zweiger et al. 2025).
    Returns (prompt ids, answer ids) training pairs and the raw text."""
    key = jax.random.key(1) if key is None else key
    prompts = [M.prompt_ids(tok, DREAM_PROMPT, p.bio()) for p in people for _ in range(n_samples)]
    outs = M.decode(cfg, params, prompts, P=M.P_CTX, N=160, stop=M.stop_ids(tok), temperature=0.7, key=key,
                    batch=32)
    pairs, texts = [], []
    for o in outs:
        txt = tok.decode(M.strip_stop(o, M.stop_ids(tok)))
        texts.append(txt)
        for q, a in QA_PAIR.findall(txt):
            q, a = " ".join(q.split()), " ".join(a.split())
            if q and a and len(M.prompt_ids(tok, q)) <= M.P_LEN:
                pairs.append((M.prompt_ids(tok, q), M.answer_ids(tok, a)))
    return pairs, texts


def bio_lm(tok, people):
    """Each bio as plain text to continue: an empty-ish prompt (the name) and the bio as the "answer"."""
    out = []
    for p in people:
        ids = tok.encode(p.bio()).ids + [M.stop_ids(tok)[1]]
        out.append((ids[:1], ids[1:]))                        # predict everything after the first token
    return out


def dream_scores(cfg, params, ex32, gate, pairs, batch=16):
    """Importance of each dream: the norm of its SFT-loss gradient with respect to the experts' B matrices (a new
    expert's A gets no gradient while its B is still zero). Big gradient = the model has most to learn from it."""
    def one(b32, a, tok, mask, am):
        ex = {"A": a, "B": b32}
        return M.sft_loss(params, bf16(ex), gate, tok[None], mask[None], am[None], cfg, M.P_LEN, M.N_ANS)

    @jax.jit
    def norms(b32, a, tok, mask, am):
        g = jax.vmap(jax.grad(one), in_axes=(None, None, 0, 0, 0))(b32, a, tok, mask, am)
        return jnp.sqrt(jnp.sum(g.astype(jnp.float32) ** 2, axis=(1, 2, 3, 4)))

    out = []
    for s in range(0, len(pairs), batch):
        b = pairs[s:s + batch]
        b = b + [b[-1]] * (batch - len(b))
        tok, mask, am = M.layout([x[0] for x in b], [x[1] for x in b], M.P_LEN, M.N_ANS)
        out += np.asarray(norms(ex32["B"], ex32["A"], jnp.asarray(tok), jnp.asarray(mask), jnp.asarray(am))).tolist()
    return np.asarray(out[:len(pairs)])


def select_dreams(scores, k, b, seed=0):
    """Indices of the k highest scores plus b others at random (same as the exercise's reference solution)."""
    scores = np.asarray(scores, float)
    n = len(scores)
    if k + b >= n:
        return list(range(n))
    top = np.argsort(-scores, kind="stable")[:k]
    rest = np.setdiff1d(np.arange(n), top)
    extra = np.random.default_rng(seed).choice(rest, b, replace=False) if b else np.array([], int)
    return sorted(int(i) for i in np.concatenate([top, extra]))


# ---------------------------------------------------------------- evaluation

def evaluate(cfg, params, tok, world, ev, world_items, texts, experts=None, gate=None):
    R_row = [float(M.accuracy(cfg, params, tok, qa, experts=experts, gate=gate).mean()) for qa in ev]
    S_row = [float(M.accuracy(cfg, params, tok, qa, experts=experts, gate=gate).mean())
             for qa in seen_set(world)[:len(ev)]]
    w = float(M.world_accuracy(cfg, params, tok, world_items, experts=experts, gate=gate).mean())
    nll = float(M.text_nll(params, experts, gate, jnp.asarray(texts), cfg))
    return {"acc_by_day": R_row, "seen_by_day": S_row, "world": w, "text_nll": nll}


# ---------------------------------------------------------------- arms

def run_arm(arm, cfg, params, tok, world, ev, world_items, texts, a) -> dict:
    rng = np.random.default_rng(a.seed)
    n_days = a.days
    res = {"arm": arm, "rows": [], "loss": [], "secs_train": 0.0}
    gates = np.eye(4, dtype=np.float32)

    if arm == "icl":
        for d in range(n_days):
            row = [0.0] * n_days
            for j in range(n_days):    # with day j's bios in the prompt, one bio per question
                row[j] = float(M.accuracy(cfg, params, tok, ev[j], context=True).mean()) if j == d else \
                    float(M.accuracy(cfg, params, tok, ev[j]).mean())
            res["rows"].append({"acc_by_day": row, "world": None, "text_nll": None})
            log(arm, d, row)
        return res

    if arm in FULL_ARMS:
        p32 = jax.tree.map(lambda x: x.astype(jnp.float32), params)
        anchor = fisher = None
        lam = a.ewc_lambda if arm == "ewc" else 0.0
        if arm == "ewc":
            fisher = fisher_diag(cfg, params, texts)
            anchor = params                    # the loaded bf16 weights: no second copy (16 GB is tight here)
        seen: list = []
        for d in range(n_days):
            data = demos(tok, world.days[d])
            steps = (len(data) // BATCH) * a.epochs
            opt = M.adam(a.lr_full, steps, mu_dtype=jnp.bfloat16 if arm == "ewc" else None)
            ost = opt.init(p32)
            step = full_step_fn(cfg, opt, lam)
            t0 = time.time()
            for b in batches(rng, data, epochs=a.epochs):
                if arm == "replay" and seen:
                    k = BATCH // 4
                    b = b[:BATCH - k] + [seen[i] for i in rng.choice(len(seen), k, replace=False)]
                tok_, mask, am = M.layout([x[0] for x in b], [x[1] for x in b], M.P_LEN, M.N_ANS)
                p32, ost, loss = step(p32, ost, jnp.asarray(tok_), jnp.asarray(mask), jnp.asarray(am), anchor,
                                      fisher)
                res["loss"].append(float(loss))
            res["secs_train"] += time.time() - t0
            seen += data
            del ost
            row = evaluate(cfg, bf16(p32), tok, world, ev[:n_days], world_items, texts)
            res["rows"].append(row)
            log(arm, d, f"loss={res['loss'][-1]:.3f}", row)
        res["trainable"] = int(sum(x.size for x in jax.tree.leaves(p32)))
        return res

    # expert arms: base frozen. General text for the *_retain arms: other windows than the evaluation's (seed 1)
    gtexts = M.text_batch(tok, n=512, T=128, seed=1) if arm.endswith("_retain") else None

    def general():
        if gtexts is None:
            return None
        return jnp.asarray(gtexts[rng.choice(len(gtexts), 8, replace=False)])

    ex32 = M.init_experts(jax.random.key(a.seed), cfg, n=4, r=a.rank)
    res["trainable"] = int(ex32["A"][:, 0].size + ex32["B"][:, 0].size)
    for d in range(n_days):
        e = 0 if arm == "expert_one" else d
        gate = jnp.asarray(gates[:e + 1].sum(0) if arm != "expert_one" else gates[0])
        old_gate = jnp.asarray(gates[:e].sum(0)) if e > 0 else jnp.zeros(4, jnp.float32)
        onehot = jnp.asarray(gates[e])
        people = world.days[d]
        base_arm = arm.removesuffix("_retain")
        if base_arm in ("expert_one", "expert_new"):
            data, P, kind = demos(tok, people), M.P_LEN, "sft"
        elif base_arm == "distill":
            data, P, kind = [(M.prompt_ids(tok, q), M.prompt_ids(tok, q, p.bio())) for q, _, p, _ in
                             F.qa_pairs(people, "train")], M.P_LEN, a.kl
        elif arm == "bio_text":
            data, P, kind = bio_lm(tok, people), 1, "sft"
        elif arm in ("dream", "dream_select"):
            t0 = time.time()
            data, texts_d = dreams(cfg, params, tok, people)
            res.setdefault("dream_examples", texts_d[:4])
            res.setdefault("n_dreams", []).append(len(data))
            if arm == "dream_select":
                sc = dream_scores(cfg, params, ex32, gate, data)
                keep = select_dreams(sc, len(data) // 2, len(data) // 10, seed=d)
                res.setdefault("n_kept", []).append(len(keep))
                res.setdefault("score_quantiles", []).append(np.quantile(sc, [0, .25, .5, .75, 1]).tolist())
                data = [data[i] for i in keep]
            res["secs_dream"] = res.get("secs_dream", 0.0) + time.time() - t0
            P, kind = M.P_LEN, "sft"
        epochs = a.epochs_distill if base_arm == "distill" else a.epochs
        if arm in ("bio_text", "dream", "dream_select"):    # the same number of steps as SFT on the 750 questions
            epochs = max(1, round(a.epochs * (len(people) * 15 // BATCH) / max(1, len(data) // BATCH)))
        steps = max(1, len(data) // BATCH) * epochs
        opt = M.adam(a.lr_expert, steps)
        ost = opt.init(ex32)
        N = 96 if arm == "bio_text" else M.N_ANS            # a whole bio is about 70 tokens
        retain = a.retain if arm.endswith("_retain") else 0.0
        step = expert_step_fn(cfg, opt, kind, P, N, a.skip, retain)
        t0 = time.time()
        key = jax.random.key(100 + d)
        for b in batches(rng, data, batch=min(BATCH, len(data)), epochs=epochs):
            if kind == "sft":
                tok_, mask, am = M.layout([x[0] for x in b], [x[1] for x in b], P, N)
                ex32, ost, loss = step(ex32, ost, params, gate, onehot, jnp.asarray(tok_), jnp.asarray(mask),
                                       jnp.asarray(am), None, general())
            else:
                key, sub = jax.random.split(key)
                student_p = [x[0] for x in b]
                ans = M.decode(cfg, params, student_p, experts=bf16(ex32), gate=gate, stop=M.stop_ids(tok),
                               temperature=1.0, key=sub, batch=len(b))
                ttok, tmask, _ = M.layout([x[1] for x in b], ans, M.P_CTX, M.N_ANS)
                tl = teacher_logits_fn(params, bf16(ex32), old_gate, jnp.asarray(ttok), jnp.asarray(tmask), cfg,
                                       M.P_CTX)
                stok, smask, sam = M.layout(student_p, ans, M.P_LEN, M.N_ANS)
                ex32, ost, loss = step(ex32, ost, params, gate, onehot, jnp.asarray(stok), jnp.asarray(smask),
                                       jnp.asarray(sam), tl, general())
            res["loss"].append(float(loss))
        res["secs_train"] += time.time() - t0
        row = evaluate(cfg, params, tok, world, ev[:n_days], world_items, texts, experts=bf16(ex32), gate=gate)
        res["rows"].append(row)
        log(arm, d, f"loss={res['loss'][-1]:.4f}", row)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--epochs-distill", type=int, default=4)
    ap.add_argument("--lr-full", type=float, default=1e-5)
    ap.add_argument("--lr-expert", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--ewc-lambda", type=float, default=1e5)
    ap.add_argument("--kl", default="forward", choices=("forward", "reverse", "jsd"))
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--retain", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    params, cfg, tok = M.load()
    world = F.make_world(4, 50, a.seed)
    ev = eval_set(world)
    texts = M.text_batch(tok)
    M.RUNS.mkdir(parents=True, exist_ok=True)
    base = M.read_json(M.RUNS / "base.json")
    if base is None:
        wa = M.world_accuracy(cfg, params, tok, M.WORLD_QS)
        base = {"world_known": [int(i) for i in np.flatnonzero(wa)], "world_total": len(M.WORLD_QS),
                "text_nll": float(M.text_nll(params, None, None, jnp.asarray(texts), cfg)),
                "acc_by_day": [float(M.accuracy(cfg, params, tok, qa).mean()) for qa in ev]}
        (M.RUNS / "base.json").write_text(json.dumps(base, indent=1))
    log("base", base["text_nll"], len(base["world_known"]), "/", base["world_total"], base["acc_by_day"])
    world_items = [M.WORLD_QS[i] for i in base["world_known"]]

    for arm in a.arms:
        t0 = time.time()
        res = run_arm(arm, cfg, params, tok, world, ev, world_items, texts, a)
        res["secs_total"] = time.time() - t0
        res["args"] = vars(a)
        R = np.array([r["acc_by_day"] for r in res["rows"]])
        res["metrics"] = M.forgetting_metrics(R)
        out = M.RUNS / (arm + a.tag)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(res, indent=1))
        log(arm, "done", res["metrics"], f"{res['secs_total']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
