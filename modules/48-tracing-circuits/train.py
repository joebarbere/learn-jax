"""Module 48's experiments on SmolLM2-135M.

    containers/run.sh python modules/48-tracing-circuits/train.py --patch          # IOI patching sweeps, ~3 min
    containers/run.sh python modules/48-tracing-circuits/train.py --transcoders    # 30 transcoders, ~45 min
    containers/run.sh python modules/48-tracing-circuits/train.py --graph          # attribution graphs, ~5 min
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/48-tracing-circuits/train.py ...   # reference

--graph needs --transcoders. Uses *your* exercises.py (finish `make test N=48` first) or solutions.py with
LEARNJAX_SOLUTIONS=1. Writes /course/runs/48-tracing-circuits/: patch.json/.npz, transcoders.npz,
transcoders.json, graph_*.json, and circuit-tracer files under graphs/ (see the lesson for viewing them).
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

import jaxprobe as jp

from learnjax import llama
from learnjax.course import load_exercises


def _models():
    spec = importlib.util.spec_from_file_location("m48_models", HERE / "models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- transcoders
def run_transcoders(M, params, cfg, tok, tokens: float, batch: int, lr: float):
    ids = M.wiki_tokens(tok, "train")
    _log(f"wikitext-103 train: {len(ids) / 1e6:.1f}M tokens")
    L, d = cfg.n_layers, cfg.d_model
    io = M.mlp_io(params, cfg)
    rng = np.random.default_rng(0)

    # per-layer scales (mean squared norm 1) and outlier thresholds, from 4 batches (~8k tokens)
    xs, ys = [], []
    for _ in range(4):
        x, y = io(params, jnp.asarray(M.windows(ids, batch, rng)))
        xs.append(x), ys.append(y)
    X, Y = jnp.concatenate(xs, 1), jnp.concatenate(ys, 1)
    med_out = jnp.median(jnp.linalg.norm(Y, axis=-1), axis=1)  # (L,)
    typ = jnp.linalg.norm(Y, axis=-1) < M.OUTLIER * med_out[:, None]
    ms = lambda A: (((A ** 2).sum(-1) * typ).sum(1) / typ.sum(1))
    scale_in, scale_out = 1 / jnp.sqrt(ms(X)), 1 / jnp.sqrt(ms(Y))
    _log("outliers left out per layer (%):", np.round(100 * (1 - np.asarray(typ.mean(1))), 2).tolist())
    del xs, ys, X, Y

    steps = int(tokens // (batch * (M.SEQ - 1)))
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 4), steps, lr / 10)
    opt = optax.adam(sched)
    tcs = M.init_transcoders(jax.random.key(0), L, d, M.N_FEAT)
    st = opt.init(tcs)
    since = jnp.zeros((L, M.N_FEAT), jnp.int32)  # tokens since each latent last fired
    DEAD = 1_000_000

    @jax.jit
    def step(tcs, st, since, t):
        x, y = io(params, t)
        mask = (jnp.linalg.norm(y, axis=-1) < M.OUTLIER * med_out[:, None]).astype(jnp.float32)
        dead = since > DEAD
        (_, m), g = jax.lax.map(
            lambda a: jax.value_and_grad(M.tc_loss, has_aux=True)(a[0], a[1] * a[3], a[2] * a[4], a[5], a[6]),
            (tcs, x, y, scale_in, scale_out, mask, dead))
        up, st = opt.update(g, st)
        tcs = optax.apply_updates(tcs, up)
        tcs = dict(tcs, W_dec=tcs["W_dec"] / jnp.linalg.norm(tcs["W_dec"], axis=-1, keepdims=True))
        since = jnp.where(m["fired"], 0, since + x.shape[1])
        return tcs, st, since, m["mse"], m["fvu"]

    hist, t0 = [], time.time()
    for i in range(steps):
        tcs, st, since, _mse, fvu = step(tcs, st, since, jnp.asarray(M.windows(ids, batch, rng)))
        if i % 125 == 0 or i == steps - 1:
            f = np.asarray(fvu)
            dead = np.asarray((since > DEAD).mean(1))
            hist.append({"step": i, "tokens": (i + 1) * batch * (M.SEQ - 1), "fvu": f.tolist(),
                         "dead": dead.tolist()})
            rate = (i + 1) * batch * (M.SEQ - 1) / (time.time() - t0)
            _log(f"step {i}/{steps}  FVU mean {f.mean():.3f} (min {f.min():.3f} L{f.argmin()}, max {f.max():.3f} "
                 f"L{f.argmax()})  dead {dead.mean():.1%}  {rate:,.0f} tok/s")
    meta = {"scale_in": np.asarray(scale_in).tolist(), "scale_out": np.asarray(scale_out).tolist(), "k": M.K,
            "n": M.N_FEAT, "tokens": steps * batch * (M.SEQ - 1), "seconds": time.time() - t0, "lr": lr,
            "batch": batch, "median_out": np.asarray(med_out).tolist()}
    M.save_transcoders(M.RUNS / "transcoders.npz", M.for_jaxprobe(tcs), meta)
    (M.RUNS / "transcoders_hist.json").write_text(json.dumps(hist))
    _log("saved transcoders.npz")
    return tcs, meta


def eval_transcoders(M, params, cfg, tok, n_windows: int = 256):
    """FVU per layer, and next-token loss with one / all MLPs replaced, on held-out Wikipedia."""
    tcs, meta = M.load_transcoders()
    ids = M.wiki_tokens(tok, "validation", max_tokens=1_000_000)
    W = jnp.asarray(M.windows(ids, n_windows, np.random.default_rng(7)))
    io = M.mlp_io(params, cfg)
    s_in, s_out = jnp.asarray(meta["scale_in"]), jnp.asarray(meta["scale_out"])
    med = jnp.asarray(meta["median_out"])
    num = np.zeros(cfg.n_layers)
    den = np.zeros(cfg.n_layers)

    def one_layer(a):
        tc, x, y, si, so, md = a
        yh = jp.sae.decode(tc, jp.sae.encode(tc, x * si, "topk", meta["k"])) / so
        mask = jnp.linalg.norm(y, axis=-1) < M.OUTLIER * md
        mu = (y * mask[:, None]).sum(0) / mask.sum()
        return (((y - yh) ** 2).sum(-1) * mask).sum(), (((y - mu) ** 2).sum(-1) * mask).sum()

    stats = jax.jit(lambda x, y: jax.lax.map(one_layer, (tcs, x, y, s_in, s_out, med)))
    for b in range(0, n_windows, 32):
        x, y = io(params, W[b:b + 32])
        nb, db = stats(x, y)
        num += np.asarray(nb)
        den += np.asarray(db)
    fvu = num / den
    clean_f = jax.jit(M.forward(params, cfg))
    ce = lambda f: float(np.mean([float(M.next_token_loss(f(W[b:b + 32]), W[b:b + 32]))
                                  for b in range(0, n_windows, 32)]))
    clean = ce(clean_f)
    all_ = ce(jax.jit(M.replacement_forward(params, cfg, tcs, meta)))
    per = [ce(jax.jit(M.replacement_forward(params, cfg, tcs, meta, [layer]))) for layer in range(cfg.n_layers)]
    # zero-ablating one MLP, for scale
    zero = []
    for layer in range(cfg.n_layers):
        f = jax.jit(lambda t, layer=layer: jp.run(M.forward(params, cfg), t, patch={
            "layers.mlp_out": jp.at(layer, lambda m: m.at[:, 1:].set(0))})[0])
        zero.append(ce(f))
    out = {"fvu": fvu.tolist(), "ce_clean": clean, "ce_all": all_, "ce_one": per, "ce_zero_one": zero,
           "tokens": meta["tokens"], "seconds": meta["seconds"]}
    (M.RUNS / "transcoders.json").write_text(json.dumps(out, indent=1))
    _log(f"FVU per layer: {np.round(fvu, 3).tolist()}")
    _log(f"CE clean {clean:.3f}, all 30 MLPs replaced {all_:.3f}; one at a time: mean +{np.mean(per) - clean:.3f} "
         f"(zero-ablating one: mean +{np.mean(zero) - clean:.3f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true")
    ap.add_argument("--transcoders", action="store_true")
    ap.add_argument("--eval", action="store_true", help="evaluate saved transcoders only")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--tokens", type=float, default=15e6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    M = _models()
    M.RUNS.mkdir(parents=True, exist_ok=True)
    ex = load_exercises(HERE)
    if a.transcoders:
        params, cfg, tok = llama.load(llama.SMOLLM_135M)  # bfloat16 activations for training
        run_transcoders(M, params, cfg, tok, a.tokens, a.batch, a.lr)
    if a.transcoders or a.eval:
        params, cfg, tok = llama.load(llama.SMOLLM_135M, dtype="float32")
        eval_transcoders(M, params, cfg, tok)
    if a.patch:
        run_patch(M, ex)
    if a.graph:
        run_graph(M, ex)


def run_patch(M, ex):
    """IOI on SmolLM2-135M: residual and head patching, attribution patching, head roles, and a sufficiency curve."""
    params, cfg, tok = llama.load(llama.SMOLLM_135M, dtype="float32")
    D = M.ioi(tok, 96)
    fwd = M.forward(params, cfg)
    L, H, T = cfg.n_layers, cfg.n_heads, D["clean"].shape[1]
    ld = lambda lg: ex.logit_diff(lg, D["io"], D["s"])
    names = ["embed", "layers.resid", "layers.z", "layers.pattern", "final_scale"]
    run = jax.jit(lambda t: jp.run(fwd, t, capture=names))
    (lc, ac), (lf, af) = run(D["clean"]), run(D["flip"])
    ld_c, ld_f = float(ld(lc)), float(ld(lf))
    last_c = M.last_logits(lc)
    acc = float((jnp.take_along_axis(last_c, D["io"][:, None], 1) > jnp.take_along_axis(last_c, D["s"][:, None], 1))
                .mean())
    _log(f"logit diff clean {ld_c:.2f}, flipped {ld_f:.2f}; IO > S on {acc:.0%}")
    eff = lambda v: float(ex.patching_effect(v, ld_c, ld_f))

    # residual stream: patch clean h_layer at one position into the flipped run
    R_c = jnp.concatenate([ac["embed"][None], ac["layers.resid"]])

    @jax.jit
    def resid_patch(layer, pos):
        put = lambda x: x.at[:, pos].set(R_c[layer][:, pos])
        lg, _ = jp.run(fwd, D["flip"], patch={"embed": lambda x: jnp.where(layer == 0, put(x), x),
                                              "layers.resid": lambda x: jnp.where(jp.index() + 1 == layer, put(x), x)})
        return ld(lg)

    t0 = time.time()
    resid = np.array([[eff(resid_patch(layer, p)) for p in range(T)] for layer in range(L + 1)])
    t_resid = time.time() - t0

    # heads: patch one head's clean output (all positions)
    @jax.jit
    def head_patch(layer, head):
        lg, _ = jp.run(fwd, D["flip"], patch={"layers.z": lambda z: jnp.where(
            jp.index() == layer, z.at[:, :, head].set(ac["layers.z"][layer][:, :, head]), z)})
        return ld(lg)

    t0 = time.time()
    heads = np.array([[eff(head_patch(layer, h)) for h in range(H)] for layer in range(L)])
    t_heads = time.time() - t0

    # attribution patching: one forward + backward on the flipped run, then grad · (clean − flipped)
    def metric(dz, dr):
        lg, _ = jp.run(fwd, D["flip"], patch={"layers.z": lambda z: z + dz[jp.index()],
                                              "layers.resid": lambda x: x + dr[jp.index()]})
        return ld(lg)

    t0 = time.time()
    gz, gr = jax.jit(jax.grad(metric, argnums=(0, 1)))(jnp.zeros_like(af["layers.z"]),
                                                       jnp.zeros_like(af["layers.resid"]))
    jax.block_until_ready(gz)
    t_atp = time.time() - t0
    atp_heads = np.asarray(ex.attribution_patch(gz, ac["layers.z"], af["layers.z"]).sum((1, 2))) / (ld_c - ld_f)
    atp_resid = np.asarray(ex.attribution_patch(gr, ac["layers.resid"], af["layers.resid"]).sum(1)) / (ld_c - ld_f)

    # what the top heads attend to from the last position, and what they write toward IO − S
    rep = H // cfg.n_kv_heads
    b = np.arange(len(D["text"]))
    P = np.asarray(ac["layers.pattern"])  # (L, B, KV, rep, T, S)
    head_dirs = params["embed"][D["io"]] - params["embed"][D["s"]]  # (B, d): unembedding difference (tied)
    gf = params["final_norm"]
    scale = ac["final_scale"][:, -1]  # (B, 1)
    roles = []
    order = np.argsort(-np.abs(heads).ravel())[:10]
    for i in order:
        layer, h = int(i // H), int(i % H)
        w = P[layer, :, h // rep, h % rep, -1]  # (B, S)
        out = M.head_out(params, layer, h, ac["layers.z"][layer][:, -1, h], cfg)  # (B, d)
        dla = float(((out * scale * gf) * head_dirs).sum(-1).mean())
        roles.append({"layer": layer, "head": h, "patch": float(heads[layer, h]), "atp": float(atp_heads[layer, h]),
                      "attn_io": float(w[b, D["pos_io"]].mean()), "attn_s1": float(w[b, 4 - D["pos_io"]].mean()),
                      "attn_s2": float(w[:, M.POS["S2"]].mean()), "attn_first": float(w[:, 0].mean()),
                      "dla": dla})
        _log(roles[-1])

    # sufficiency: patch only the top-k heads (by |patching effect|) into the flipped run
    rank = np.argsort(-np.abs(heads).ravel())
    pos_rank = np.argsort(-heads.ravel())

    @jax.jit
    def keep_heads(mask):  # mask (L, H) bool: which heads get their clean output
        lg, _ = jp.run(fwd, D["flip"], patch={"layers.z": lambda z: jnp.where(
            mask[jp.index()][None, None, :, None], ac["layers.z"][jp.index()], z)})
        return ld(lg)

    curve = []
    for k in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, L * H]:
        m_abs = np.zeros(L * H, bool)
        m_abs[rank[:k]] = True
        m_pos = np.zeros(L * H, bool)
        m_pos[pos_rank[:k]] = True
        curve.append({"k": k, "top_abs": eff(keep_heads(jnp.asarray(m_abs.reshape(L, H)))),
                      "top_pos": eff(keep_heads(jnp.asarray(m_pos.reshape(L, H))))})
    _log("sufficiency:", curve)
    out = {"ld_clean": ld_c, "ld_flip": ld_f, "acc": acc, "n": len(D["text"]), "examples": D["text"][:4],
           "tokens": [tok.decode([int(t)]) for t in D["clean"][0]], "t_resid": t_resid, "t_heads": t_heads,
           "t_atp": t_atp, "n_resid": (L + 1) * T, "n_heads": L * H, "roles": roles, "curve": curve,
           "corr_heads": float(np.corrcoef(heads.ravel(), atp_heads.ravel())[0, 1]),
           "corr_resid": float(np.corrcoef(resid[1:].ravel(), atp_resid.ravel())[0, 1])}
    np.savez(M.RUNS / "patch.npz", resid=resid, heads=heads, atp_heads=atp_heads, atp_resid=atp_resid)
    (M.RUNS / "patch.json").write_text(json.dumps(out, indent=1))
    _log(f"resid patching {t_resid:.0f}s for {(L + 1) * T} runs; heads {t_heads:.0f}s for {L * H}; attribution "
         f"patching {t_atp:.1f}s for all; corr heads {out['corr_heads']:.3f}, resid {out['corr_resid']:.3f}")


def run_graph(M, ex, n_label: int = 40):
    """Attribution graphs for M.PROMPTS, feature labels from held-out text, interventions that test them, and
    circuit-tracer JSON files."""
    from jaxprobe import trace

    params, cfg, tok = llama.load(llama.SMOLLM_135M, dtype="float32")
    tcs, meta = M.load_transcoders()
    fwd = M.forward(params, cfg)
    dec = lambda t: tok.decode([int(t)])
    val = M.wiki_tokens(tok, "validation", max_tokens=1_000_000)
    out = {}
    for name, (text, target) in M.PROMPTS.items():
        t0 = time.time()
        g = M.attribute(params, cfg, tok, tcs, meta, text)
        t_attr = time.time() - t0
        gp = trace.prune(g, max_features=M.VIEW_FEATURES, max_edges=M.VIEW_EDGES)
        # your normalize_rows / total_influence must agree with the library's
        A = np.asarray(ex.normalize_rows(jnp.asarray(g.adj)))
        w = np.zeros(len(A), np.float32)
        w[g.slices()["logit"]] = g.logit_probs / g.logit_probs.sum()
        mine = np.asarray(ex.total_influence(jnp.asarray(A), jnp.asarray(w)))
        agree = float(np.abs(mine - gp.influence).max())
        top = gp.top_features(n_label)
        mid = [f for f in gp.top_features(gp.N) if 1 <= f["layer"] <= cfg.n_layers - 2][:n_label]
        _log(f"{name}: {g.N} active features, {len(g.adj)} nodes; 80% of influence needs "
             f"{gp.scores['n_features_threshold']} features; kept {gp.scores['n_features']} features, "
             f"{gp.scores['n_edges']} edges; replacement {gp.scores['replacement']:.2f}, completeness "
             f"{gp.scores['completeness']:.2f}; logit check {g.scores['logit_check']:.1e}; {t_attr:.0f}s; "
             f"influence vs yours {agree:.1e}")
        out[name] = {"text": text, "target": target, "graph": gp, "top": top, "mid": mid, "seconds": t_attr,
                     "agree": agree}

    # labels: top contexts on held-out Wikipedia, and direct logit effects
    feats = sorted({(f["layer"], f["index"]) for o in out.values() for f in o["top"] + o["mid"]})
    _log(f"collecting contexts for {len(feats)} features")
    best, Wv = M.feature_contexts(params, cfg, tcs, meta, feats, val)
    labels = {}
    for (layer, f) in feats:
        up, down = M.logit_effects(params, cfg, tcs, layer, f)
        ctx = []
        for a, wi, p in best[(layer, f)][:5]:
            win = Wv[wi]
            ctx.append({"act": a, "left": tok.decode(win[max(0, p - 12):p].tolist()), "tok": dec(win[p]),
                        "right": tok.decode(win[p + 1:p + 4].tolist())})
        labels[f"{layer}/{f}"] = {"contexts": ctx, "promotes": [dec(t) for t in up], "suppresses": [dec(t) for t in down]}

    # interventions on the real model (dallas): ablate the graph's most influential features vs random active ones
    res = {}
    for name, o in out.items():
        g = o["graph"]
        ids = jnp.array([tok.encode(o["text"]).ids])
        tgt = tok.encode(o["target"]).ids[0]
        prob = lambda patch, ids=ids, tgt=tgt: float(jax.nn.softmax(jp.run(fwd, ids, patch=patch)[0][0, -1].astype(
            jnp.float32))[tgt])
        p0 = prob({})
        rows = []
        rng = np.random.default_rng(0)
        for n in (1, 3, 10, 30):
            top = [(f["layer"], f["pos"], f["index"]) for f in o["top"][:n]]
            acts = [f["act"] for f in o["top"][:n]]
            p_top = prob(trace.ablate_features(tcs, meta["scale_out"], top, acts))
            mid = [(f["layer"], f["pos"], f["index"]) for f in o["mid"][:n]]
            p_mid = prob(trace.ablate_features(tcs, meta["scale_out"], mid, [f["act"] for f in o["mid"][:n]]))
            p_rand = []
            for _ in range(5):
                j = rng.choice(g.N, n, replace=False)
                p_rand.append(prob(trace.ablate_features(tcs, meta["scale_out"], [tuple(int(v) for v in g.feat[i])
                                                                                  for i in j], [g.act[i] for i in j])))
            rows.append({"n": n, "p_top": p_top, "p_mid": p_mid, "p_random": float(np.mean(p_rand))})
        res[name] = {"p": p0, "ablate": rows}
        _log(name, f"P({o['target']!r}) = {p0:.3f};", rows)

    # export: circuit-tracer / Neuronpedia graph files
    gdir = M.RUNS / "graphs"
    for name, o in out.items():
        lab = {(f["layer"], f["index"]): "fires on " + ", ".join(
            dict.fromkeys(repr(c["tok"]) for c in labels[f"{f['layer']}/{f['index']}"]["contexts"][:4]))
               for f in o["top"] + o["mid"]}
        trace.write_graph(gdir, trace.to_json(o["graph"], dec, slug=f"smollm2-135m-{name}", scan="jaxprobe-smollm2-135m-tc",
                                              labels=lab, node_threshold=0.8))

    # what the lesson draws: kept nodes and their strongest edges
    summary = {}
    for name, o in out.items():
        g = o["graph"]
        keep = np.flatnonzero(g.keep & ((g.adj != 0).any(0) | (g.adj != 0).any(1)))
        nodes = [dict(g.node_info(int(i)), id=int(i), influence=float(g.influence[i])) for i in keep]
        for nd in nodes:
            if nd["kind"] == "logit":
                nd["tok"] = dec(nd["index"])
        sub = g.adj[np.ix_(keep, keep)]
        t_idx, s_idx = np.nonzero(sub)
        order = np.argsort(-np.abs(sub[t_idx, s_idx]))[:400]
        edges = [{"s": int(keep[s_idx[j]]), "t": int(keep[t_idx[j]]), "w": float(sub[t_idx[j], s_idx[j]])} for j in order]
        summary[name] = {"text": o["text"], "target": o["target"], "tokens": [dec(t) for t in g.tokens],
                         "logits": [{"tok": dec(t), "p": float(p)} for t, p in zip(g.logit_ids, g.logit_probs, strict=True)],
                         "n_active": int(g.N), "n_nodes": len(g.adj), "scores": g.scores, "seconds": o["seconds"],
                         "agree": o["agree"], "top": o["top"], "mid": o["mid"], "nodes": nodes, "edges": edges,
                         "layer_share": [float(g.influence[g.slices()["feature"]][g.feat[:, 0] == layer].sum() /
                                               max(g.influence[g.slices()["feature"]].sum(), 1e-12))
                                         for layer in range(cfg.n_layers)],
                         "intervene": res[name]}
    (M.RUNS / "graph.json").write_text(json.dumps({"graphs": summary, "labels": labels}, indent=1))
    _log("wrote graph.json and graphs/")


if __name__ == "__main__":
    main()
