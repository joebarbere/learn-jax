"""Module 50's long runs: Hartford's experiment at small scale. SmolLM2-360M (the base model, never instruction-
tuned) fine-tuned on WizardLM's Evol-Instruct 70k, three ways that differ only in their data.

    T="containers/run.sh python modules/50-uncensored-models/train.py"
    $T --arm full          # all rows (68,500 after the held-out ones), ~3 h on the RX 7600 XT
    $T --arm filtered      # Hartford's filtered rows (53,974), ~2.3 h
    $T --arm random        # a random 53,974 of the full rows: less data, not different data, ~2.3 h
    $T --arm full --steps 20      # a short pilot (a separate run directory, <arm>-pilot)

One epoch, Hartford's settings otherwise: 128 examples per optimizer step, lr 2e-5 with 2 warmup steps then
cosine decay, AdamW without weight decay, gradient clipping at 1.0, bf16 compute with float32 master weights,
the loss on response tokens only. Sequences are cut at 1,024 tokens (95% of rows fit; Hartford used 2,048).
A run resumes from its last saved state (every 50 steps) if stopped.

Writes /course/runs/50-uncensored-models/<arm>/: metrics.jsonl (loss per step; held-out loss every 50 steps
on rows the filter keeps and rows it removes), params.safetensors (bf16, the final model), config.json,
samples.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # the course root, for `learnjax`

from jaxprobe import io

from learnjax import data, llama, sft

RUNS = Path("/course/runs/50-uncensored-models")
BASE = "HuggingFaceTB/SmolLM2-360M"
SAMPLE_PROMPTS = ["What is the capital of Australia?", "Write a haiku about rain.",
                  "How do I kill a Python process?", "What is your favourite colour?",
                  "Explain what a bias term is in a neural network."]


def _models():
    spec = importlib.util.spec_from_file_location("m50_models", HERE / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_state(path: Path, tr: sft.Trainer, step: int):
    leaves = jax.tree.leaves((tr.params, tr.opt_state))
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, step=step, **{f"a{i}": np.asarray(x) for i, x in enumerate(leaves)})
    os.replace(tmp, path)


def load_state(path: Path, tr: sft.Trainer) -> int:
    z = np.load(path)
    leaves, tdef = jax.tree.flatten((tr.params, tr.opt_state))
    new = [jnp.asarray(z[f"a{i}"]).astype(x.dtype) for i, x in enumerate(leaves)]
    tr.params, tr.opt_state = jax.tree.unflatten(tdef, new)
    return int(z["step"])


def generate(params, cfg, tok, instruction: str, n: int = 96) -> str:
    ids = tok.encode(sft.prompt(instruction)).ids
    out = llama.generate(params, cfg, ids, n, jax.random.key(0), temperature=0.0, stop_ids=(0,))
    return tok.decode(out)


# ---------------------------------------------------------------- evaluation (Parts A and B)
QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
N_NEW = 32           # greedy tokens per answer: refusals show in the first sentence


def qwen_ids(tok, q: str) -> list[int]:
    return tok.encode(f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n"
                      "<|im_start|>assistant\n").ids


def prompt_sets(M, rows, sp, seed: int = 0) -> dict[str, list[str]]:
    """Harmful (AdvBench) and harmless (short held-out Evol-Instruct) instructions, split for finding a direction,
    choosing it, and testing; plus XSTest. None of the harmless ones is in any training arm."""
    rng = np.random.default_rng(seed)
    adv = [r["goal"] for r in data.advbench()]
    adv = [adv[i] for i in rng.permutation(len(adv))]
    short = [rows[i]["instruction"] for i in sp["val_kept"]
             if len(rows[i]["instruction"]) < 300 and "\n" not in rows[i]["instruction"]]
    short = [short[i] for i in rng.permutation(len(short))]
    return {"harmful_fit": adv[:128], "harmful_val": adv[128:160], "harmful_test": adv[160:288],
            "harmless_fit": short[:128], "harmless_val": short[128:160], "harmless_test": short[160:288]}


def first_token_scores(M, cfg, params, prompts: list[list[int]], ids, patch=None, batch: int = 32):
    """Arditi's refusal score of the first answer token for each prompt (one forward pass, no generation)."""
    from jaxprobe import batch as jb
    from jaxprobe import directions

    g = M._bound(cfg)
    f = jax.jit(lambda patch, t, m: directions.refusal_score(g(patch, params, t, m, t.shape[1] - 1)[0], ids))
    out = np.zeros(len(prompts), np.float32)
    for b in jb.padded(prompts, batch, buckets=(64, 128, 256, 512)):
        s = np.asarray(f(patch or {}, jnp.asarray(b["tokens"]), jnp.asarray(b["mask"])))
        keep = b["index"] >= 0
        out[b["index"][keep]] = s[keep]
    return out


def answer_all(M, cfg, params, tok, prompts: dict[str, list[list[int]]], stop, patch=None) -> dict:
    """Greedy answers to every prompt set, with the refusal-substring verdict."""
    res = {}
    for name, ps in prompts.items():
        texts = [tok.decode(x) for x in M.greedy(cfg, params, ps, N_NEW, stop, patch)]
        res[name] = {"answers": texts, "refused": [M.is_refusal(t) for t in texts]}
    return res


def resid_last(cfg, params, prompts: list[list[int]], batch: int = 32) -> np.ndarray:
    """Residual stream after every layer at each prompt's last token: (layers, N, d) float16."""
    import jaxprobe as jp
    from jaxprobe import batch as jb
    from jaxprobe import reduce

    f = jax.jit(lambda t, m: jp.run(lambda p, t, m: llama.trunk(p, t, cfg, mask=m)[0], params, t, m,
                                    capture={"layers.resid": reduce.last})[1]["layers.resid"])
    out = np.zeros((cfg.n_layers, len(prompts), cfg.d_model), np.float16)
    for b in jb.padded(prompts, batch, buckets=(64, 128, 256, 512)):
        x = np.asarray(f(jnp.asarray(b["tokens"]), jnp.asarray(b["mask"])).astype(jnp.float32))
        keep = b["index"] >= 0
        out[:, b["index"][keep]] = x[:, keep]
    return out


def load_arm(arm: str):
    params, cfg, tok = llama.load(BASE)
    if arm != "base":
        path = RUNS / arm / "params"
        if not path.with_suffix(".safetensors").exists():
            raise FileNotFoundError(f"{path}.safetensors missing: run --arm {arm} first")
        params = jax.tree.map(jnp.asarray, io.load(path))
    return params, cfg, tok


def run_eval(arms: list[str]) -> int:
    """Part A: each model's answers to XSTest and to harmful/harmless held-out prompts (Alpaca template), its
    first-token refusal scores, and its residual stream at the last prompt token (for the refusal direction)."""
    M = _models()
    rows = data.evol_instruct()
    sp = M.splits(rows, M.removed_mask(rows, M.hartford_phrases()))
    sets = prompt_sets(M, rows, sp)
    xs = data.xstest()
    out = RUNS / "eval"
    out.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        t0 = time.time()
        params, cfg, tok = load_arm(arm)
        enc = lambda qs, tok=tok: [tok.encode(sft.prompt(q)).ids for q in qs]
        prompts = {"xstest": enc([r["prompt"] for r in xs]), "harmful_test": enc(sets["harmful_test"]),
                   "harmless_test": enc(sets["harmless_test"])}
        res = answer_all(M, cfg, params, tok, prompts, (0,))
        ids = M.refusal_ids(tok, " ")
        for k, ps in prompts.items():
            res[k]["score"] = first_token_scores(M, cfg, params, ps, ids).tolist()
        res["xstest_meta"] = [{"id": r["id"], "type": r["type"], "label": r["label"]} for r in xs]
        (out / f"{arm}.json").write_text(json.dumps(res))
        acts = {k: resid_last(cfg, params, enc(sets[k])) for k in ("harmful_fit", "harmless_fit")}
        np.savez(out / f"{arm}_resid.npz", **acts)
        xr = np.array(res["xstest"]["refused"])
        lab = np.array([r["label"] == "safe" for r in xs])
        _log(f"{arm}: XSTest refusals safe {xr[lab].mean():.1%}, unsafe {xr[~lab].mean():.1%}; AdvBench "
             f"{np.mean(res['harmful_test']['refused']):.1%}; harmless {np.mean(res['harmless_test']['refused']):.1%}"
             f" ({time.time() - t0:.0f} s)")
    return 0


def run_qwen() -> int:
    """Part B: Arditi et al.'s refusal direction in Qwen2.5-1.5B-Instruct, removed from the weights, then
    switched back on at run time."""
    import jaxprobe as jp
    from jaxprobe import batch as jb
    from jaxprobe import directions, steer

    M = _models()
    out = RUNS / "qwen"
    out.mkdir(parents=True, exist_ok=True)
    rows = data.evol_instruct()
    sp = M.splits(rows, M.removed_mask(rows, M.hartford_phrases()))
    sets = prompt_sets(M, rows, sp)
    params, cfg, tok = llama.load(QWEN)
    params = dict(params, head=params["embed"].T)       # untie: the edit must not touch the read-out
    cfg = dataclasses.replace(cfg, tied=False)
    L = cfg.n_layers
    enc = {k: [qwen_ids(tok, q) for q in v] for k, v in sets.items()}
    stop = (tok.token_to_id("<|im_end|>"),)
    ids = M.refusal_ids(tok)
    res = {"model": QWEN, "refusal_ids": ids}

    # 1. candidate directions: difference in means after every layer, at the 5 template tokens after the request
    t0 = time.time()
    pos = [-5, -4, -3, -2, -1]
    fwd = lambda p, t, m: llama.trunk(p, t, cfg, mask=m)[0]
    args = lambda b: (jnp.asarray(b["tokens"]), jnp.asarray(b["mask"]))
    stats = {k: directions.collect(fwd, params, jb.padded(enc[k], 32, buckets=(64, 128)), at="layers.resid",
                                   positions=pos, args=args) for k in ("harmful_fit", "harmless_fit")}
    cands = directions.candidates(stats["harmful_fit"], stats["harmless_fit"])       # (L, 5, d)
    res["collect_s"] = time.time() - t0

    # 2. score every candidate (Arditi's bypass / induce / KL) and pick one
    g = M._bound(cfg)
    last = lambda patch, tm: g(patch, params, tm[0], tm[1], tm[0].shape[1] - 1)[0]
    hb = next(jb.padded(enc["harmful_val"], 32, buckets=(64,)))
    nb = next(jb.padded(enc["harmless_val"], 32, buckets=(128,)))
    H, N = (jnp.asarray(hb["tokens"]), jnp.asarray(hb["mask"])), (jnp.asarray(nb["tokens"]), jnp.asarray(nb["mask"]))
    ablate_at = ["embed", "layers.attn_out", "layers.mlp_out"]
    score = directions.arditi_score(last, H, N, ids, ablate_at=ablate_at, add_at="layers.resid")
    t0 = time.time()
    grid = directions.grid(score, cands, batch_size=4)
    res["grid_s"] = time.time() - t0
    best = directions.select(grid, n_layers=L)
    res["best"] = best
    r = cands[best["layer"], best["pos"]]
    u = steer.unit(r)
    _log(f"direction: layer {best['layer']}, position {pos[best['pos']]} (bypass {best['bypass']:.2f}, induce "
         f"{best['induce']:.2f}, KL {best['kl']:.3f}; {best['n_valid']} of {L * 5} candidates pass)")
    np.savez(out / "direction.npz", cands=np.asarray(cands, np.float32), u=np.asarray(u, np.float32),
             **{k: v for k, v in grid.items()}, positions=np.array(pos))

    # 3. bake it into the weights (in float32 on the CPU, stored back as bf16) and check against run-time ablation
    writers = {"embed": -1, "layers.wo": -1, "layers.w_down": -1}
    cpu = jax.devices("cpu")[0]
    host = jax.device_put(params, cpu)
    with jax.default_device(cpu):
        e32 = directions.abliterate(jax.tree.map(lambda a: a.astype(jnp.float32), host), jax.device_put(u, cpu),
                                    writers=writers)
        f32fwd = lambda p, t, m: llama.forward(p, t, dataclasses.replace(cfg, dtype="float32"), mask=m)[0]
        p32 = jax.tree.map(lambda a: a.astype(jnp.float32), host)
        t4, m4 = jax.device_put(N[0][:4], cpu), jax.device_put(N[1][:4], cpu)
        res["check_edit_f32"] = directions.check_edit(f32fwd, p32, e32, jax.device_put(u, cpu), t4, m4,
                                                      ablate_at=ablate_at)
        edited = jax.tree.map(lambda a: a.astype(jnp.bfloat16), e32)
        del e32, p32
    edited = jax.device_put(edited, jax.devices()[0])
    bf = lambda p, t, m: llama.forward(p, t, cfg, mask=m)[0]
    res["check_edit_bf16"] = directions.check_edit(bf, params, edited, u, N[0][:4], N[1][:4], ablate_at=ablate_at)
    # how much of u is left in the stored bf16 weights (the edit made exactly 0 in float32)
    wd = edited["layers"]["w_down"].astype(jnp.float32)
    res["bf16_residual_u_in_w_down"] = float(jnp.abs(wd @ u.astype(jnp.float32)).max() /
                                             jnp.abs(params["layers"]["w_down"].astype(jnp.float32) @ u).max())
    _log(f"edit vs run-time ablation, max logit diff: float32 {res['check_edit_f32']:.2e}, "
         f"bf16 {res['check_edit_bf16']:.3f}")

    # 4. what changed: refusal rates, first-token scores, harmless KL and loss on held-out answers
    xs = data.xstest()
    test = {"xstest": [qwen_ids(tok, r["prompt"]) for r in xs], "harmful_test": enc["harmful_test"],
            "harmless_test": enc["harmless_test"]}
    on = {"layers.resid": jp.at(best["layer"], steer.add(r, 1.0))}        # Arditi's activation addition
    models = {"base": (params, None), "edited": (edited, None), "edited+add": (edited, on), "base+add": (params, on)}
    answers = {}
    for name, (p, patch) in models.items():
        t0 = time.time()
        answers[name] = answer_all(M, cfg, p, tok, test, stop, patch)
        for k, ps in test.items():
            answers[name][k]["score"] = first_token_scores(M, cfg, p, ps, ids, patch).tolist()
        xr = np.array(answers[name]["xstest"]["refused"])
        lab = np.array([r["label"] == "safe" for r in xs])
        _log(f"{name}: XSTest refusals safe {xr[lab].mean():.1%}, unsafe {xr[~lab].mean():.1%}; AdvBench "
             f"{np.mean(answers[name]['harmful_test']['refused']):.1%}; harmless "
             f"{np.mean(answers[name]['harmless_test']['refused']):.1%} ({time.time() - t0:.0f} s)")
    answers["xstest_meta"] = [{"id": r["id"], "type": r["type"], "label": r["label"]} for r in xs]
    (out / "answers.json").write_text(json.dumps(answers))

    hl = jax.jit(lambda p, t, m: g({}, p, t, m, t.shape[1] - 1)[0])
    kls = []
    for b in jb.padded(test["harmless_test"], 32, buckets=(64, 128)):
        t, m = jnp.asarray(b["tokens"]), jnp.asarray(b["mask"])
        kl = np.asarray(steer.kl(hl(params, t, m), hl(edited, t, m)))
        kls += kl[b["index"] >= 0].tolist()
    res["kl_harmless_test"] = float(np.mean(kls))
    vk = [rows[i] for i in sp["val_kept"][:300]]
    exs = []
    for rr in vk:
        pi = qwen_ids(tok, rr["instruction"])[:900]
        ids_ = np.asarray(pi + tok.encode(rr["output"]).ids + list(stop), np.int32)[:1024]
        exs.append(sft.Example(ids_, len(pi), False))
    res["nll_evol_heldout"] = {"base": sft.eval_nll(params, cfg, exs), "edited": sft.eval_nll(edited, cfg, exs)}

    # 5. the switch: refusal score as the direction is added back with a growing coefficient
    coeffs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    res["switch"] = {"coeffs": coeffs}
    for k in ("harmful_test", "harmless_test"):
        res["switch"][k] = [float(first_token_scores(
            M, cfg, edited, test[k], ids, {"layers.resid": jp.at(best["layer"], steer.add(r, c))}).mean())
            for c in coeffs]
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    _log(f"KL on harmless prompts {res['kl_harmless_test']:.3f}; held-out answer loss base "
         f"{res['nll_evol_heldout']['base']:.4f} -> edited {res['nll_evol_heldout']['edited']:.4f}")
    _log("done")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arm", choices=["full", "filtered", "random"])
    mode.add_argument("--eval", nargs="+", metavar="ARM", choices=["base", "full", "filtered", "random"],
                      help="XSTest refusal rates and the refusal direction inside each fine-tuned model")
    mode.add_argument("--qwen", action="store_true", help="Part B: the refusal direction in Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="stop after this many steps (a pilot run)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=50)
    a = ap.parse_args(argv)
    if a.eval:
        return run_eval(a.eval)
    if a.qwen:
        return run_qwen()

    name = a.arm + ("-pilot" if a.steps else "") + (f"-{a.epochs}ep" if a.epochs != 1 else "")
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "params.safetensors").exists():
        _log(f"{out} is finished (params.safetensors exists); delete the directory to retrain")
        return 0
    M = _models()
    rows = data.evol_instruct()
    removed = M.removed_mask(rows, M.hartford_phrases())
    sp = M.splits(rows, removed)
    _log(f"arm {a.arm}: {len(sp[a.arm]):,} rows; held out {len(sp['val_kept'])} kept + "
         f"{len(sp['val_removed'])} removed")

    params, cfg, tok = llama.load(BASE)
    exs = sft.encode([rows[i] for i in sp[a.arm]], tok, a.max_len)
    val_k = sft.encode([rows[i] for i in sp["val_kept"]], tok, a.max_len)
    val_r = sft.encode([rows[i] for i in sp["val_removed"]], tok, a.max_len)
    per_epoch = len(exs) // a.batch
    total = per_epoch * a.epochs if not a.steps else a.steps
    opt, sched = sft.make_optimizer(a.lr, per_epoch * a.epochs)
    tr = sft.Trainer(params, cfg, opt)
    del params

    state = out / "state.npz"
    step = load_state(state, tr) if state.exists() else 0
    metrics = (out / "metrics.jsonl").open("a")
    (out / "config.json").write_text(json.dumps({
        "arm": a.arm, "rows": len(exs), "epochs": a.epochs, "steps": total, "batch": a.batch, "lr": a.lr,
        "max_len": a.max_len, "truncated": int(sum(e.truncated for e in exs)), "base": BASE,
        "resp_tokens": int(sum(len(e.ids) - e.start for e in exs))}, indent=1))

    eval_s = [0.0]

    def evaluate(s):
        t = time.time()
        p = tr.bf16()
        m = {"step": s, "val_kept": sft.eval_nll(p, cfg, val_k), "val_removed": sft.eval_nll(p, cfg, val_r)}
        metrics.write(json.dumps(m) + "\n")
        metrics.flush()
        eval_s[0] += time.time() - t
        _log(f"step {s:4d} | held-out loss: kept rows {m['val_kept']:.4f}, removed rows {m['val_removed']:.4f}")

    if step == 0:
        evaluate(0)
    _log(f"{total} steps from step {step}")
    t0, tok_done, eval_s[0] = time.time(), 0, 0.0
    while step < total:
        epoch, within = divmod(step, per_epoch)
        for micro in sft.steps(exs, a.batch, np.random.default_rng(1000 + epoch), skip=within):
            m = tr.step(micro)
            step += 1
            tok_done += sum(int(mb["mask"].sum()) for mb in micro)
            el = time.time() - t0
            m.update(step=step, lr=float(sched(step - 1)), tok_per_s=tok_done / (el - eval_s[0]), elapsed_s=el)
            metrics.write(json.dumps(m) + "\n")
            if step % 10 == 0 or step == 1:
                metrics.flush()
                _log(f"step {step:4d}/{total} | loss {m['loss']:.4f} | {m['tok_per_s']:,.0f} tok/s | "
                     f"{el / 60:.0f} min")
            if step % a.eval_every == 0 or step == total:
                evaluate(step)
                save_state(state, tr, step)
            if step >= total:
                break

    p = tr.bf16()
    io.save(out / "params", jax.device_get(p), metadata={"arm": a.arm, "base": BASE})
    samples = [{"q": q, "a": generate(p, cfg, tok, q)} for q in SAMPLE_PROMPTS]
    (out / "samples.json").write_text(json.dumps(samples, indent=1))
    for s in samples:
        _log(f"Q: {s['q']}\nA: {s['a'][:300]}")
    state.unlink(missing_ok=True)
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
