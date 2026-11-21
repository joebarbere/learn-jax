"""Module 47's experiments on SmolLM2-135M (and SmolLM2-360M for the last one).

    containers/run.sh python modules/47-reading-activations/train.py --lens      # tuned lens, ~17 min
    containers/run.sh python modules/47-reading-activations/train.py --sae       # six SAEs, ~45 min
    containers/run.sh python modules/47-reading-activations/train.py --dash      # feature tables, ~3 min
    containers/run.sh python modules/47-reading-activations/train.py --steer     # steering sweeps, ~5 min
    containers/run.sh python modules/47-reading-activations/train.py --ablate    # weight edit vs activation edit
    containers/run.sh python modules/47-reading-activations/train.py --map       # 135M vs 360M, ~5 min
    containers/run.sh python modules/47-reading-activations/train.py --autointerp http://host:1234/v1 MODEL
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/47-reading-activations/train.py ...   # reference

--dash needs --sae; --ablate needs --steer. Uses *your* exercises.py (finish `make test N=47` first) or
solutions.py with LEARNJAX_SOLUTIONS=1. Writes /course/runs/47-reading-activations/: lens.npz/.json,
sae_*.npz, sae.json, dash.json, steer.json, ablate.json, map.json (and autointerp.json if you run it).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # the course root, for `learnjax`

from learnjax import interp, llama  # after sys.path setup
from learnjax.course import load_exercises

SMOLLM_360M_BASE = "HuggingFaceTB/SmolLM2-360M"


def _models():
    spec = importlib.util.spec_from_file_location("m47_models", HERE / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def median_norm(M, params, cfg, data):
    f = M.acts_fn(cfg)
    w, _ = M.mixed_windows(data, 64, np.random.default_rng(123))
    return float(jnp.median(jnp.linalg.norm(f(params, jnp.asarray(w)), axis=-1)))


# ---------------------------------------------------------------- 1. tuned lens
def run_lens(M, params, cfg, data, steps: int = 1000, batch: int = 16):
    L = cfg.n_layers
    coll = jax.jit(lambda p, t: interp.forward(p, cfg, t, collect=True))

    def loss(lens, R, final):
        kls, _ = M.lens_kl(params, cfg, lens, R, final)
        return kls.sum()

    opt = optax.adam(optax.warmup_cosine_decay_schedule(0, 1e-3, 50, steps, 1e-4))
    lens = interp.init_tuned_lens(cfg)
    st = opt.init(lens)

    @jax.jit
    def step(lens, st, t):
        out = coll(params, t)
        R = out["resid"][:, :, 1:].reshape(L + 1, -1, cfg.d_model)
        final = out["logits"][:, 1:].reshape(-1, cfg.vocab)
        lv, g = jax.value_and_grad(loss)(lens, R, final)
        u, st = opt.update(g, st)
        return optax.apply_updates(lens, u), st, lv

    rng = np.random.default_rng(0)
    hist = []
    for i in range(steps):
        w, _ = M.mixed_windows(data, batch, rng)
        lens, st, lv = step(lens, st, jnp.asarray(w))
        if i % 100 == 0 or i == steps - 1:
            hist.append([i, float(lv) / L])
            _log(f"lens step {i}: mean KL over layers {float(lv) / L:.3f}")
    np.savez(M.RUNS / "lens.npz", A=np.asarray(lens["A"]), b=np.asarray(lens["b"]))

    @jax.jit
    def evaluate(lens, t):
        out = coll(params, t)
        R = out["resid"][:, :, 1:].reshape(L + 1, -1, cfg.d_model)
        final = out["logits"][:, 1:].reshape(-1, cfg.vocab)
        return M.lens_kl(params, cfg, None, R, final), M.lens_kl(params, cfg, lens, R, final)

    res = {"steps": steps, "batch_windows": batch, "history": hist, "domains": {}}
    for d, name in enumerate(M.DOMAINS):
        w = jnp.asarray(data[name]["val"][: 32 * M.SEQ].reshape(32, M.SEQ))
        acc = None
        for j in range(0, 32, 8):
            r = jax.tree.map(np.asarray, evaluate(lens, w[j:j + 8]))
            acc = r if acc is None else jax.tree.map(lambda a, b: a + b, acc, r)
        (lk, lt), (tk, tt) = jax.tree.map(lambda a: (a / 4).tolist(), acc)
        res["domains"][name] = {"logit_kl": lk, "logit_top1": lt, "tuned_kl": tk, "tuned_top1": tt}
        _log(name, "logit-lens KL by layer", [round(x, 2) for x in lk[::5]], "tuned", [round(x, 2) for x in tk[::5]])
    (M.RUNS / "lens.json").write_text(json.dumps(res))


# ---------------------------------------------------------------- 2. sparse autoencoders
def run_sae(M, params, cfg, data, tokens: float, l1s, lr: float = 2e-4, batch: int = 4096):
    d, F = cfg.d_model, M.N_LATENTS
    acts = M.acts_fn(cfg)
    mnorm = median_norm(M, params, cfg, data)
    rng = np.random.default_rng(0)
    w, _ = M.mixed_windows(data, 128, rng)
    A = acts(params, jnp.asarray(w)).reshape(-1, d)
    A = A[M.typical(A, mnorm)]
    scale = float(interp.norm_scale(A))
    mean_act = A.mean(0)
    specs = [dict(s) for s in M.SAE_SPECS if s["kind"] == "topk"]
    specs += [{"name": f"relu-l1-{chr(97 + i)}", "kind": "relu", "k": 0, "l1": l1} for i, l1 in enumerate(l1s)]
    steps = int(tokens // batch)
    sched = optax.warmup_cosine_decay_schedule(0, lr, 200, steps, lr / 10)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    saes = [interp.init_sae(jax.random.key(i), d, F) for i in range(len(specs))]
    states = [opt.init(s) for s in saes]
    dead_after = min(5_000_000, tokens // 4)                    # "dead" = no firing in this many tokens

    def make_step(spec):
        @jax.jit
        def step(sae, st, X, dead, frac):
            l1 = spec["l1"] * jnp.minimum(1.0, frac * 20)        # warm the L1 penalty up over the first 5%
            (_, m), g = jax.value_and_grad(interp.sae_loss, has_aux=True)(sae, X, spec["kind"], spec["k"], l1, dead)
            u, st = opt.update(g, st, sae)
            sae = optax.apply_updates(sae, u)
            return (interp.renorm_decoder(sae) if spec["kind"] == "topk" else sae), st, m
        return step

    stepfns = [make_step(s) for s in specs]
    last = np.zeros((len(specs), F), np.int64)
    hist = {s["name"]: [] for s in specs}
    seen, i, t0 = 0, 0, time.perf_counter()
    while i < steps:
        w, _ = M.mixed_windows(data, 1024, rng)
        buf = jnp.concatenate([acts(params, jnp.asarray(w[j:j + 256])).reshape(-1, d) for j in range(0, 1024, 256)])
        buf = buf[M.typical(buf, mnorm)] * scale
        buf = buf[jax.random.permutation(jax.random.key(i), buf.shape[0])]
        for j in range(buf.shape[0] // batch):
            if i >= steps:
                break
            X = buf[j * batch:(j + 1) * batch]
            for n, s in enumerate(specs):
                dead = jnp.asarray(seen - last[n] > dead_after)
                saes[n], states[n], m = stepfns[n](saes[n], states[n], X, dead, i / steps)
                last[n][np.asarray(m["fired"])] = seen + batch
                if i % 500 == 0 or i == steps - 1:
                    hist[s["name"]].append({"tokens": seen, "mse": float(m["mse"]), "l0": float(m["l0"]),
                                            "dead": int((seen - last[n] > dead_after).sum())})
            seen += batch
            i += 1
            if i % 500 == 0:
                _log(f"{seen / 1e6:.1f}M tokens, {time.perf_counter() - t0:.0f} s:",
                     {s["name"]: (round(hist[s["name"]][-1]["mse"], 4), round(hist[s["name"]][-1]["l0"], 1),
                                  hist[s["name"]][-1]["dead"]) for s in specs})
    # evaluate on held-out windows of each domain
    wv, dv = M.val_windows(data, per_domain=96)
    fwd = jax.jit(lambda p, t, ed: interp.next_token_loss(interp.forward(p, cfg, t, edit=ed)["logits"], t),
                  static_argnums=2)

    def ce(edit):
        return float(np.mean([fwd(params, jnp.asarray(wv[j:j + 32]), edit) for j in range(0, len(wv), 32)]))

    clean = ce(None)
    zero = ce(M.replace_edit(jnp.zeros(d), mnorm))
    mean = ce(M.replace_edit(mean_act, mnorm))
    res = {"tokens": int(seen), "scale": scale, "median_norm": mnorm, "layer": M.SAE_LAYER, "n_latents": F,
           "ce_clean": clean, "ce_zero": zero, "ce_mean": mean, "saes": {}}
    Xv = jnp.concatenate([acts(params, jnp.asarray(wv[j:j + 64])) for j in range(0, len(wv), 64)]).reshape(-1, d)
    dv_tok = np.repeat(dv, M.SEQ - 1)
    keep = np.asarray(M.typical(Xv, mnorm))
    res["outlier_frac"] = float(1 - keep.mean())
    Xv, dv_tok = Xv[keep] * scale, dv_tok[keep]
    var = {k: float(((Xv[dv_tok == k] - Xv[dv_tok == k].mean(0)) ** 2).sum()) for k in range(3)}
    var_all = float(((Xv - Xv.mean(0)) ** 2).sum())
    for n, s in enumerate(specs):
        sae = saes[n]

        @jax.jit
        def chunk(X, sae=sae, s=s):
            z = interp.sae_encode(sae, X, s["kind"], s["k"])
            return ((X - interp.sae_decode(sae, z)) ** 2).sum(-1), (z > 0).sum(-1), (z > 0).sum(0)
        err, l0, fires = [], [], 0
        for j in range(0, Xv.shape[0], 4096):
            e, l, f = chunk(Xv[j:j + 4096])
            err.append(np.asarray(e))
            l0.append(np.asarray(l))
            fires = fires + np.asarray(f)
        err, l0 = np.concatenate(err), np.concatenate(l0)
        dens = fires / Xv.shape[0]
        fv = {name: float(err[dv_tok == k].sum() / var[k]) for k, name in enumerate(M.DOMAINS)}
        spliced = ce(M.splice_edit(sae, s["kind"], s["k"], scale, mnorm))
        r = {**s, "l0": float(l0.mean()), "fvu": float(err.sum() / var_all),
             "fvu_by_domain": fv, "ce_spliced": spliced,
             "ce_recovered_zero": (zero - spliced) / (zero - clean), "ce_recovered_mean": (mean - spliced) / (mean - clean),
             "dead_train": int((seen - last[n] > dead_after).sum()), "dead_after_tokens": int(dead_after),
             "never_fired_eval": int((dens == 0).sum()), "dense_eval": int((dens > 0.1).sum()), "history": hist[s["name"]]}
        res["saes"][s["name"]] = r
        M.save_sae(M.RUNS / f"sae_{s['name']}.npz", sae, {"scale": scale, "median_norm": mnorm, **s})
        _log(s["name"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                         if k not in ("history", "fvu_by_domain")})
    (M.RUNS / "sae.json").write_text(json.dumps(res))


# ---------------------------------------------------------------- 3. feature tables (main SAE)
def run_dash(M, params, cfg, data, tok, n_show: int = 8, ctx: int = 12):
    sae, meta = M.load_sae(M.RUNS / f"sae_{M.MAIN_SAE}.npz")
    scale, mnorm, k = meta["scale"], meta["median_norm"], meta["k"]
    acts = M.acts_fn(cfg)
    wv, dv = M.val_windows(data)
    F, T = M.N_LATENTS, M.SEQ - 1
    tok_fire = np.zeros((3, F))
    n_tok = np.zeros(3)
    win_fire = np.zeros((len(wv), F), bool)
    top_v = np.full((F, n_show), -1.0, np.float32)
    top_i = np.zeros((F, n_show), np.int64)

    @jax.jit
    def codes(t):
        x = acts(params, t)
        keep = M.typical(x, mnorm)
        z = interp.sae_encode(sae, x.reshape(-1, x.shape[-1]) * scale, "topk", k).reshape(x.shape[0], T, F)
        return jnp.where(keep[..., None], z, 0.0), keep

    for j in range(0, len(wv), 32):
        z, keep = codes(jnp.asarray(wv[j:j + 32]))
        zf = z.reshape(-1, F)
        fired = np.asarray(z > 0)
        win_fire[j:j + 32] = fired.any(1)
        for d in range(3):
            m = dv[j:j + 32] == d
            tok_fire[d] += fired[m].sum((0, 1))
            n_tok[d] += np.asarray(keep)[m].sum()
        v, ix = jax.lax.top_k(zf.T, n_show)
        v, ix = np.asarray(v), np.asarray(ix) + j * T
        allv, alli = np.concatenate([top_v, v], 1), np.concatenate([top_i, ix], 1)
        o = np.argsort(-allv, 1)[:, :n_show]
        top_v, top_i = np.take_along_axis(allv, o, 1), np.take_along_axis(alli, o, 1)
    rate = tok_fire / n_tok[:, None]                                 # per-domain token firing rate (3, F)
    density = tok_fire.sum(0) / n_tok.sum()
    # module 46's readout on window-level "did it fire anywhere" features: fit on even windows, test on odd
    tr, te = np.arange(len(wv)) % 2 == 0, np.arange(len(wv)) % 2 == 1
    W, b, _ = M.domain_readout(win_fire[tr], dv[tr])
    pred = np.asarray((jnp.asarray(win_fire[te], jnp.float32) @ W + b).argmax(1))
    acc = {name: float((pred[dv[te] == d] == d).mean()) for d, name in enumerate(M.DOMAINS)}
    # which latents to show: the most domain-specific per domain, the most common, and a random sample
    alive = density > 0
    show = {}
    for d, name in enumerate(M.DOMAINS):
        others = np.delete(rate, d, 0).max(0)
        spec = np.where(alive & (rate[d] > 0.005), (rate[d] + 1e-5) / (others + 1e-5), 0)
        show[name] = np.argsort(-spec)[:6].tolist()
    show["common"] = np.argsort(-density)[:6].tolist()
    show["random"] = np.random.default_rng(0).choice(np.flatnonzero(alive & (density < 0.1)), 6, replace=False).tolist()
    # logit effects: decoder direction through the final norm's gain and the unembedding
    eff = (sae["W_dec"] * params["final_norm"].astype(jnp.float32)) @ params["embed"].astype(jnp.float32).T
    flat = wv[:, 1:].reshape(-1)
    feats = {}

    def piece(i):
        return tok.decode([int(i)])

    for ids in show.values():
        for f in ids:
            if f in feats:
                continue
            contexts = []
            for v, gi in zip(top_v[f], top_i[f]):
                if v <= 0:
                    continue
                w_, p = divmod(int(gi), T)
                z, _ = codes(jnp.asarray(wv[w_:w_ + 1]))
                lo = max(0, p - ctx)
                contexts.append({"domain": M.DOMAINS[int(dv[w_])], "tokens": [piece(t) for t in flat[w_ * T + lo:w_ * T + min(p + 4, T)]],
                                 "acts": np.asarray(z[0, lo:p + 4, f]).round(2).tolist(), "peak": p - lo})
            e = np.asarray(eff[f])
            feats[int(f)] = {"density": float(density[f]), "rate": rate[:, f].tolist(), "contexts": contexts,
                             "promotes": [piece(t) for t in np.argsort(-e)[:8]],
                             "suppresses": [piece(t) for t in np.argsort(e)[:8]]}
    hist_density = np.histogram(np.log10(density[alive]), bins=30)
    res = {"n_tokens": n_tok.tolist(), "n_windows": len(wv), "readout_acc": acc,
           "readout_overall": float((pred == dv[te]).mean()), "readout_n_test": int(te.sum()),
           "alive": int(alive.sum()), "show": show, "features": feats,
           "log10_density_hist": [hist_density[0].tolist(), hist_density[1].tolist()],
           "domain_specific_counts": {name: int(((rate[d] > 0.005) & (rate[d] > 10 * np.delete(rate, d, 0).max(0))).sum())
                                      for d, name in enumerate(M.DOMAINS)}}
    (M.RUNS / "dash.json").write_text(json.dumps(res))
    _log("readout", acc, "alive", int(alive.sum()), "domain-specific", res["domain_specific_counts"])
    np.save(M.RUNS / "dash_rate.npy", rate.astype(np.float32))


# ---------------------------------------------------------------- 4. steering
def _norms(M, params, cfg, tok):
    coll = jax.jit(lambda p, t: interp.forward(p, cfg, t, collect=True)["resid"])
    t, last = M.pad_batch([tok.encode(p).ids for p in M.NEUTRAL])
    R = coll(params, t).astype(jnp.float32)
    n = jnp.linalg.norm(R, axis=-1)
    real = (jnp.arange(t.shape[1])[None] <= last[:, None]) & (jnp.arange(t.shape[1])[None] > 0)
    return np.asarray([float(jnp.median(n[layer][real])) for layer in range(n.shape[0])])


def caa_directions(M, ex, params, cfg, tok):
    """Mean-difference direction at every layer, read at the adjective of '<subject> was <adjective>.'"""
    coll = jax.jit(lambda p, t: interp.forward(p, cfg, t, collect=True)["resid"])

    def at_adj(adjs):
        t, last = M.pad_batch([tok.encode(f"{s} was {a}.").ids for s in M.SUBJECTS for a in adjs])
        R = coll(params, t).astype(jnp.float32)
        return R[:, jnp.arange(t.shape[0]), last - 1]
    P, N = at_adj(M.BUILD_POS), at_adj(M.BUILD_NEG)
    return jnp.stack([ex.mean_difference(P[layer], N[layer]) for layer in range(P.shape[0])])


def actadd_directions(M, params, cfg, tok):
    coll = jax.jit(lambda p, t: interp.forward(p, cfg, t, collect=True)["resid"])
    ia, ib = (tok.encode(s).ids for s in M.ACTADD)
    i = next(j for j, (x, y) in enumerate(zip(ia, ib)) if x != y)      # the " love" / " hate" position
    a, b = jnp.asarray([ia]), jnp.asarray([ib])
    d = (coll(params, a)[:, 0, i] - coll(params, b)[:, 0, i]).astype(jnp.float32)
    return d / jnp.linalg.norm(d, axis=-1, keepdims=True)


def run_steer(M, ex, params, cfg, data, tok):
    ep, en = M.word_ids(tok, M.EVAL_POS), M.word_ids(tok, M.EVAL_NEG)
    norms = _norms(M, params, cfg, tok)
    U = caa_directions(M, ex, params, cfg, tok)
    V = actadd_directions(M, params, cfg, tok)
    pt, pl = M.pad_batch([tok.encode(p).ids for p in M.NEUTRAL])
    det = jnp.asarray(M.windows(data["detective"]["val"], 32, np.random.default_rng(0)))
    logits = jax.jit(lambda p, t, ed: interp.forward(p, cfg, t, edit=ed)["logits"], static_argnums=2)

    @jax.jit
    def measure(layer, vec):
        ed = M.steer_edit(layer, vec)
        lg = interp.forward(params, cfg, pt, edit=ed)["logits"]
        s = M.sentiment_score(lg, pl, ep, en).mean()
        lgd = interp.forward(params, cfg, det, edit=ed)["logits"]
        return s, interp.kl(base_det, lgd)[:, 1:].mean()

    base_det = logits(params, det, None)
    base = float(M.sentiment_score(logits(params, pt, None), pl, ep, en).mean())
    layers = list(range(0, cfg.n_layers, 3))
    mults = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]
    sweep = {"caa": {}, "actadd": {}}
    for name, D in (("caa", U), ("actadd", V)):
        for layer in layers:
            row = []
            for m in mults:
                s, k = measure(jnp.int32(layer), D[layer] * m * norms[layer])
                row.append([float(s), float(k)])
            sweep[name][layer] = row
            _log(name, layer, [f"{m:+}: {s:+.2f}/{k:.2f}" for m, (s, k) in zip(mults, row)])
    # best layer: largest swing between +0.25 and −0.25 while the KL cost at ±0.25 stays under 0.5 nats
    i_neg, i_pos = mults.index(-0.25), mults.index(0.25)
    ok = [layer for layer in layers if max(sweep["caa"][layer][i_neg][1], sweep["caa"][layer][i_pos][1]) < 0.5]
    best = max(ok or layers, key=lambda layer: sweep["caa"][layer][i_pos][0] - sweep["caa"][layer][i_neg][0])
    # generations at the best layer: −, 0, + with the same sampling keys
    G, n_new, samples = 48, 20, 3
    prompts = [tok.encode(p).ids for p in M.NEUTRAL]
    gens = {}
    step_fn = jax.jit(lambda p, t, ed: interp.forward(p, cfg, t, edit=ed)["logits"], static_argnums=2)
    for m in (-0.25, 0.0, 0.25):
        vec = U[best] * m * norms[best]
        ed = M.steer_edit(best, vec)
        seqs = [list(p) for p in prompts for _ in range(samples)]
        key = jax.random.key(0)
        for _ in range(n_new):
            t, last = M.pad_batch(seqs, G)
            lg = step_fn(params, t, ed)[jnp.arange(len(seqs)), last]
            key, sk = jax.random.split(key)
            nxt = np.asarray(jax.random.categorical(sk, lg / 0.7))
            for s, n in zip(seqs, nxt):
                s.append(int(n))
        texts = [tok.decode(s[len(prompts[i // samples]):]) for i, s in enumerate(seqs)]
        bal = [M.lexicon_balance(x) for x in texts]
        gens[str(m)] = {"texts": texts, "pos": sum(b > 0 for b in bal), "neg": sum(b < 0 for b in bal),
                        "n": len(bal)}
        _log("generation", m, gens[str(m)]["pos"], "positive-leaning,", gens[str(m)]["neg"], "negative of", len(bal))
    cos = [float(U[i] @ V[i]) for i in range(cfg.n_layers + 1)]
    res = {"base_score": base, "layers": layers, "mults": mults, "norms": norms.tolist(), "sweep": sweep, "best": best,
           "generations": gens, "cos_caa_actadd": cos, "n_pairs": len(M.SUBJECTS) * len(M.BUILD_POS),
           "prompts": M.NEUTRAL}
    (M.RUNS / "steer.json").write_text(json.dumps(res))
    np.save(M.RUNS / "caa.npy", np.asarray(U))


# ---------------------------------------------------------------- 5. weight edit == activation edit
def run_ablate(M, ex, params, cfg, data, tok):
    st = json.loads((M.RUNS / "steer.json").read_text())
    best = st["best"]
    u = jnp.asarray(np.load(M.RUNS / "caa.npy")[best])
    ep, en = M.word_ids(tok, M.EVAL_POS), M.word_ids(tok, M.EVAL_NEG)
    pt, pl = M.pad_batch([tok.encode(p).ids for p in M.NEUTRAL])
    cpos = M.pad_batch([tok.encode(p).ids for p, _ in M.CONTEXTS])
    cneg = M.pad_batch([tok.encode(n).ids for _, n in M.CONTEXTS])
    det = jnp.asarray(M.windows(data["detective"]["val"], 32, np.random.default_rng(0)))
    # exactness check in float32: orthogonalized weights vs projecting the stream after every write
    p32 = jax.tree.map(lambda a: a.astype(jnp.float32), params)
    cfg32 = replace(cfg, dtype="float32")
    cfg_untied = replace(cfg32, tied=False)
    w_logits = jax.jit(lambda p, t: interp.forward(p, cfg_untied, t)["logits"])(M.orthogonalize(p32, ex, u), pt)
    a_logits = jax.jit(lambda p, t: interp.forward(p, cfg32, t, edit=M.project_out(u),
                                                   edit_mid=M.project_out(u))["logits"])(p32, pt)
    base_logits = jax.jit(lambda p, t: interp.forward(p, cfg32, t)["logits"])(p32, pt)
    real = (jnp.arange(pt.shape[1])[None] <= pl[:, None])[..., None]
    diff = float(jnp.abs(jnp.where(real, w_logits - a_logits, 0)).max())
    moved = float(jnp.abs(jnp.where(real, base_logits - a_logits, 0)).max())
    _log(f"float32: max |weights − activations| = {diff:.2e}; the edit itself moves logits by up to {moved:.2f}")
    # behaviour vs alpha (bf16 model, untied head)
    cfg_u = replace(cfg, tied=False)
    fwd = jax.jit(lambda p, t: interp.forward(p, cfg_u, t)["logits"])
    base_ce = None
    rows = []
    rand = jax.random.normal(jax.random.key(7), u.shape)
    rand = rand / jnp.linalg.norm(rand)
    for name, direction, alpha in [("sentiment", u, a) for a in (0.0, 0.5, 1.0, 1.5, 2.0)] + [("random", rand, 1.0),
                                                                                                 ("random", rand, 2.0)]:
        p = M.orthogonalize(params, ex, direction, alpha)
        neutral = float(M.sentiment_score(fwd(p, pt), pl, ep, en).mean())
        sp = M.sentiment_score(fwd(p, cpos[0]), cpos[1], ep, en)
        sn = M.sentiment_score(fwd(p, cneg[0]), cneg[1], ep, en)
        ce = float(interp.next_token_loss(fwd(p, det), det))
        base_ce = ce if base_ce is None else base_ce
        rows.append({"direction": name, "alpha": alpha, "neutral": neutral, "pos_ctx": float(sp.mean()),
                     "neg_ctx": float(sn.mean()), "gap": float((sp - sn).mean()), "ce": ce, "ce_delta": ce - base_ce})
        _log(rows[-1])
    res = {"layer": best, "max_abs_diff_f32": diff, "max_abs_moved_f32": moved, "rows": rows,
           "contexts": M.CONTEXTS}
    (M.RUNS / "ablate.json").write_text(json.dumps(res))


# ---------------------------------------------------------------- 6. two models, one space?
def run_map(M, ex, params, cfg, data, tok):
    p2, cfg2, _ = llama.load(SMOLLM_360M_BASE)
    rng = np.random.default_rng(0)
    wv, _ = M.mixed_windows(data, 320, rng, split="val", mix=(1 / 3, 1 / 3, 1 / 3))
    c1 = jax.jit(lambda p, t: interp.forward(p, cfg, t, collect=True)["resid"][:, :, 1:])
    c2 = jax.jit(lambda p, t: interp.forward(p, cfg2, t, collect=True)["resid"][:, :, 1:])

    def gather(c, p, rows):
        parts = []
        for j in range(0, rows, 32):
            r = np.asarray(c(p, jnp.asarray(wv[j:j + 32])).astype(jnp.float32))[:, :, ::4]
            parts.append(r.reshape(r.shape[0], -1, r.shape[-1]))
        return np.concatenate(parts, 1)
    R1, R2 = gather(c1, params, len(wv)), gather(c2, p2, len(wv))         # (L+1, N, d), every 4th token
    n1, n2 = np.linalg.norm(R1, axis=-1), np.linalg.norm(R2, axis=-1)
    keep = ((n1 < 10 * np.median(n1, 1, keepdims=True)).all(0) & (n2 < 10 * np.median(n2, 1, keepdims=True)).all(0))
    R1, R2 = R1[:, keep], R2[:, keep]
    N = R1.shape[1]
    _log("tokens kept", N)
    L1s, L2s = list(range(0, cfg.n_layers + 1, 3)), list(range(0, cfg2.n_layers + 1, 4))
    sub = np.random.default_rng(1).choice(N, 2048, replace=False)
    knn = [[ex.mutual_knn(R1[a, sub], R2[b, sub], 10) for b in L2s] for a in L1s]
    self_knn = [ex.mutual_knn(R1[a, sub], R1[a + 3 if a + 3 <= cfg.n_layers else a - 3, sub], 10) for a in L1s]
    shuffled = ex.mutual_knn(R1[15, sub], R2[16, np.random.default_rng(2).permutation(sub)], 10)
    _log("kNN row L15:", [round(x, 3) for x in knn[L1s.index(15)]], "shuffled", shuffled)
    # a linear map 135M h_15 → 360M h_16 (least squares with a little ridge), tested on held-out tokens
    la, lb = M.SAE_LAYER, 16
    perm = np.random.default_rng(3).permutation(N)
    tr, te = perm[: N - 2000], perm[N - 2000:]
    X, Y = R1[la], R2[lb]
    mx, my = X[tr].mean(0), Y[tr].mean(0)
    Xc, Yc = X[tr] - mx, Y[tr] - my
    Wm = np.linalg.solve(Xc.T @ Xc + 1e-2 * np.trace(Xc.T @ Xc) / Xc.shape[1] * np.eye(Xc.shape[1]), Xc.T @ Yc)
    pred = (X[te] - mx) @ Wm + my
    r2 = 1 - ((pred - Y[te]) ** 2).sum() / ((Y[te] - Y[te].mean(0)) ** 2).sum()
    pn = (pred - my) / np.linalg.norm(pred - my, axis=1, keepdims=True)
    yn = (Y[te] - my) / np.linalg.norm(Y[te] - my, axis=1, keepdims=True)
    top1 = float(((pn @ yn.T).argmax(1) == np.arange(len(te))).mean())
    _log(f"map R² {r2:.3f}, top-1 retrieval among {len(te)}: {top1:.3f}")
    # carry the sentiment direction across: steer 360M with mapped(u), its own CAA direction, a random one
    U1 = caa_directions(M, ex, params, cfg, tok)
    U2 = caa_directions(M, ex, p2, cfg2, tok)
    um = np.asarray(U1[la]) @ Wm
    um = jnp.asarray(um / np.linalg.norm(um))
    rnd = jax.random.normal(jax.random.key(9), (cfg2.d_model,))
    rnd = rnd / jnp.linalg.norm(rnd)
    ep, en = M.word_ids(tok, M.EVAL_POS), M.word_ids(tok, M.EVAL_NEG)
    pt, pl = M.pad_batch([tok.encode(p).ids for p in M.NEUTRAL])
    norms2 = _norms(M, p2, cfg2, tok)

    @jax.jit
    def score(vec):
        return M.sentiment_score(interp.forward(p2, cfg2, pt, edit=M.steer_edit(lb, vec))["logits"], pl, ep, en).mean()
    transfer = {"base": float(score(jnp.zeros(cfg2.d_model)))}
    for name, vec in (("mapped 135M direction", um), ("360M's own direction", U2[lb]), ("random direction", rnd)):
        transfer[name] = {str(m): float(score(vec * m * norms2[lb])) for m in (-0.25, 0.25)}
    res = {"layers_135m": L1s, "layers_360m": L2s, "knn": knn, "knn_self_neighbour_layer": self_knn,
           "knn_shuffled": shuffled, "k": 10, "n_knn": 2048, "map": {"from": la, "to": lb, "r2": float(r2),
           "top1": top1, "n_test": len(te), "n_train": len(tr)}, "cos_mapped_own": float(um @ U2[lb]),
           "transfer": transfer}
    _log(transfer, "cos", res["cos_mapped_own"])
    (M.RUNS / "map.json").write_text(json.dumps(res))


# ---------------------------------------------------------------- optional: auto-interpretation with a local LLM
def run_autointerp(M, url: str, model: str, n_detect: int = 6):
    """Explain each shown feature from its top contexts with a local model (OpenAI-compatible server, Modules 19/27),
    then score the explanation: show the model held-out contexts (half top-activating, half random) and count how
    often it says correctly which ones match (detection scoring, Paulo et al. 2024)."""
    import urllib.request

    dash = json.loads((M.RUNS / "dash.json").read_text())

    def chat(prompt):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                           "max_tokens": 120}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()

    def show(c):
        return "".join(f"<<{t}>>" if i == c["peak"] else t for i, t in enumerate(c["tokens"]))

    rng = np.random.default_rng(0)
    all_ctx = [c for f in dash["features"].values() for c in f["contexts"]]
    out = {}
    for fid, f in dash["features"].items():
        ctxs = f["contexts"]
        if len(ctxs) < 4:
            continue
        expl = chat("Each line is a text snippet; the token between << >> is where one internal feature of a "
                    "language model fires most. In at most 12 words, what does the feature detect?\n\n"
                    + "\n".join(show(c) for c in ctxs[: len(ctxs) // 2]))
        held = ctxs[len(ctxs) // 2:][: n_detect // 2]
        others = [all_ctx[i] for i in rng.choice(len(all_ctx), n_detect, replace=False) if all_ctx[i] not in ctxs]
        tests = [(c, True) for c in held] + [(c, False) for c in others[: len(held)]]
        right = 0
        for c, truth in tests:
            a = chat(f"A feature detects: {expl}\nDoes it fire on the marked token in this snippet? Answer yes or "
                     f"no.\n\n{show(c)}")
            right += a.lower().startswith("yes") == truth
        out[fid] = {"explanation": expl, "detection_acc": right / len(tests), "n": len(tests)}
        _log(fid, out[fid])
    (M.RUNS / "autointerp.json").write_text(json.dumps({"model": model, "features": out}))


def main():
    ap = argparse.ArgumentParser()
    for s in ("lens", "sae", "dash", "steer", "ablate", "map"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--autointerp", nargs=2, metavar=("URL", "MODEL"))
    ap.add_argument("--tokens", type=float, default=20e6, help="SAE training tokens")
    ap.add_argument("--l1", type=float, nargs="*", default=[2e-3, 6e-3], help="ReLU SAE L1 coefficients")
    ap.add_argument("--lens-steps", type=int, default=1000)
    args = ap.parse_args()
    M = _models()
    M.RUNS.mkdir(parents=True, exist_ok=True)
    if args.autointerp:
        run_autointerp(M, *args.autointerp)
        return
    ex = load_exercises(HERE)
    params, cfg, tok = llama.load(llama.SMOLLM_135M)
    data = M.corpus(tok)
    if args.lens:
        run_lens(M, params, cfg, data, steps=args.lens_steps)
    if args.sae:
        run_sae(M, params, cfg, data, args.tokens, args.l1)
    if args.dash:
        run_dash(M, params, cfg, data, tok)
    if args.steer:
        run_steer(M, ex, params, cfg, data, tok)
    if args.ablate:
        run_ablate(M, ex, params, cfg, data, tok)
    if args.map:
        run_map(M, ex, params, cfg, data, tok)


if __name__ == "__main__":
    main()
