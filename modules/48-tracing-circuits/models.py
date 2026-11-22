"""Module 48 model code: the indirect-object task and its patching experiments, per-layer transcoders for all 30
MLPs of SmolLM2-135M, and attribution graphs built with `jaxprobe.trace`.

The model is `learnjax.llama`, whose forward pass carries jaxprobe taps (embed, layers.z, layers.pattern,
layers.mlp_in, layers.mlp_out, layers.resid, the norm scales). Everything here reads or patches those taps.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxprobe as jp
import numpy as np
from jaxprobe import sae as S
from jaxprobe import trace

from learnjax import llama

RUNS = Path("/course/runs/48-tracing-circuits")
SEQ = 128            # transcoder training windows; position 0 (an attention sink) is left out
N_FEAT = 8192        # features per transcoder (d_ff is 1536, d_model 576)
K = 32               # TopK: active features per token per layer
VIEW_FEATURES = 300  # pruned graphs keep at most this many features and edges, so they can be drawn and browsed
VIEW_EDGES = 5000
OUTLIER = 10.0       # tokens whose MLP output norm is > 10× the layer's median are left out of training


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def forward(params, cfg):
    """tokens (B, T) → logits (B, T, V): the function jaxprobe.run / jaxprobe.trace call."""
    return lambda t: llama.forward(params, t, cfg)[0]


# ================================================================ 1. the indirect-object task
TEMPLATE = "When {A} and {B} went to the {P}, {S} gave a {O} to"
NAMES = ["Mary", "John", "Tom", "James", "Anna", "Paul", "Sarah", "David", "Lisa", "Mark", "Kate", "Peter", "Emma", "Jack", "Laura", "Mike", "Alice", "Bob", "Linda", "Steve", "Jane", "Chris", "Susan", "Frank", "Helen", "Brian", "Ruth", "Kevin", "Nancy", "Eric"]
PLACES = ["store", "park", "beach", "school", "office", "market", "garden", "station"]
OBJECTS = ["drink", "book", "ball", "gift", "letter", "phone", "key", "ticket"]
# token positions in the template (every name, place and object is one token)
POS = {"A": 1, "B": 3, "S2": 9, "END": 13}


def ioi(tok, n: int = 96, seed: int = 0):
    """n prompts with their *flipped* twins: the same names and order, but the repeated name (S2) is the other one,
    so the right answer flips. Half ABBA ("When Bob and Laura …, Laura gave … to" → Bob), half BABA.
    Returns token arrays (n, 14), the IO and S token ids, and where IO sits (position 1 or 3)."""
    one = lambda w: len(tok.encode(" " + w).ids) == 1
    names = [x for x in NAMES if one(x)]
    rng = np.random.default_rng(seed)
    clean, flip, io, s, pos_io, text = [], [], [], [], [], []
    for i in range(n):
        a, b = rng.choice(names, 2, replace=False)
        p, o = rng.choice(PLACES), rng.choice(OBJECTS)
        first, second = (a, b) if i % 2 == 0 else (b, a)
        c = TEMPLATE.format(A=first, B=second, P=p, S=b, O=o)
        clean.append(tok.encode(c).ids)
        flip.append(tok.encode(TEMPLATE.format(A=first, B=second, P=p, S=a, O=o)).ids)
        io.append(tok.encode(" " + a).ids[0])
        s.append(tok.encode(" " + b).ids[0])
        pos_io.append(POS["A"] if i % 2 == 0 else POS["B"])
        text.append(c)
    return {"clean": jnp.array(clean), "flip": jnp.array(flip), "io": jnp.array(io), "s": jnp.array(s),
            "pos_io": np.array(pos_io), "text": text}


def last_logits(logits):
    return logits[:, -1].astype(jnp.float32)


def head_out(params, layer: int, head: int, z, cfg):
    """What one head writes to the residual stream: its z (…, hd) through its slice of the output matrix."""
    hd = cfg.head_dim
    return z @ params["layers"]["wo"][layer, head * hd:(head + 1) * hd]


# ================================================================ 2. transcoder data
def wiki_tokens(tok, split: str = "train", max_tokens: int = 40_000_000):
    """WikiText-103 as SmolLM2 token ids (uint16, articles joined with blank lines), cached under RUNS."""
    from learnjax.data import wikitext103

    RUNS.mkdir(parents=True, exist_ok=True)
    f = RUNS / f"wiki_{split}.npy"
    if not f.exists():
        out, n = [], 0
        for art in wikitext103(split):
            ids = tok.encode(art + "\n\n").ids
            out.append(np.asarray(ids, np.uint16))
            n += len(ids)
            if n >= max_tokens:
                break
        np.save(f, np.concatenate(out))
    return np.load(f)


def windows(ids, n: int, rng, seq: int = SEQ):
    """n random windows (n, seq) int32."""
    start = rng.integers(0, len(ids) - seq, n)
    return np.stack([ids[s:s + seq] for s in start]).astype(np.int32)


def mlp_io(params, cfg):
    """jit: tokens (B, T) → MLP inputs and outputs of every layer at positions 1…, (L, B·(T−1), d) float32 each."""
    def f(p, t):
        _, a = jp.run(lambda t: llama.trunk(p, t, cfg)[0], t, capture=["layers.mlp_in", "layers.mlp_out"])
        L, d = cfg.n_layers, cfg.d_model
        return (a["layers.mlp_in"][:, :, 1:].reshape(L, -1, d).astype(jnp.float32),
                a["layers.mlp_out"][:, :, 1:].reshape(L, -1, d).astype(jnp.float32))
    return jax.jit(f)


# ================================================================ 3. transcoders
def init_transcoders(key, L: int, d: int, n: int = N_FEAT):
    tcs = jax.vmap(lambda k: S.init(k, d, n))(jax.random.split(key, L))
    return {k: v for k, v in tcs.items() if k != "log_threshold"}


def tc_loss(tc, x, y, mask, dead, k: int = K, k_aux: int = 512):
    """One layer's transcoder loss: x (N, d) scaled MLP input, y (N, d) scaled MLP output, mask (N,) tokens to use.
    Matmuls in bfloat16 with float32 parameters (~4× faster here than float32 matmuls; see the lesson).
    MSE + AuxK (Gao et al.): the residual error reconstructed by the k_aux largest *dead* latents, weight 1/32.
    Unlike an SAE, the encoder doesn't subtract b_dec: input and output live in different spaces. `for_jaxprobe`
    folds the difference into b_enc so jaxprobe's SAE encoder gives the same codes."""
    bf = jnp.bfloat16
    pre = (x.astype(bf) @ tc["W_enc"].astype(bf)).astype(jnp.float32) + tc["b_enc"]
    z = S.topk_relu(pre, k)
    y_hat = (z.astype(bf) @ tc["W_dec"].astype(bf)).astype(jnp.float32) + tc["b_dec"]
    w = mask / jnp.maximum(mask.sum(), 1.0)
    mse = (((y - y_hat) ** 2).sum(-1) * w).sum()
    z_aux = jnp.where(dead, S.topk_relu(jnp.where(dead, pre, -1e9), k_aux), 0.0)
    err_hat = (z_aux.astype(bf) @ tc["W_dec"].astype(bf)).astype(jnp.float32)
    aux = (((jax.lax.stop_gradient(y - y_hat) - err_hat) ** 2).sum(-1) * w).sum()
    aux = jnp.where(dead.any(), aux, 0.0) / 32
    var = (((y - (y * w[:, None]).sum(0)) ** 2).sum(-1) * w).sum()
    return mse + aux, {"mse": mse, "fvu": mse / var, "fired": ((z > 0) & (mask[:, None] > 0)).any(0)}


def for_jaxprobe(tcs):
    """jaxprobe.sae encodes with (x − b_dec) W_enc + b_enc; our transcoders use x W_enc + b_enc. Equal if
    b_enc' = b_enc + b_dec W_enc."""
    return dict(tcs, b_enc=tcs["b_enc"] + jnp.einsum("ld,ldn->ln", tcs["b_dec"], tcs["W_enc"]))


def save_transcoders(path, tcs, meta):
    np.savez(path, **{k: np.asarray(v) for k, v in tcs.items()}, meta=json.dumps(meta))


def load_transcoders(path=None):
    """(transcoders as float32 jnp arrays stacked over layers, meta) — meta has scale_in / scale_out per layer."""
    z = np.load(path or RUNS / "transcoders.npz")
    meta = json.loads(str(z["meta"]))
    return {k: jnp.asarray(z[k], jnp.float32) for k in ("W_enc", "b_enc", "W_dec", "b_dec")}, meta


def replacement_forward(params, cfg, tcs, meta, layers=None):
    """tokens → logits with each chosen MLP replaced by its transcoder (positions ≥ 1). A copy of the llama layer
    loop that feeds each transcoder the real MLP input."""
    s_in, s_out = jnp.asarray(meta["scale_in"]), jnp.asarray(meta["scale_out"])
    L = cfg.n_layers
    use = np.zeros(L, bool)
    use[list(layers) if layers is not None else list(range(L))] = True
    use = jnp.asarray(use)

    def f(tokens):
        B, T = tokens.shape
        positions = jnp.broadcast_to(jnp.arange(T), (B, T))
        x = params["embed"][tokens]

        def body(x, inp):
            lp, tc, i = inp
            x = x + llama.attn_block(cfg, x, lp, positions)[0]
            m_real = llama.mlp_block(cfg, x, lp)
            h = llama.rms_norm(x, lp["mlp_norm"], cfg.norm_eps).astype(jnp.float32)
            m_tc = (S.decode(tc, S.encode(tc, h * s_in[i], "topk", meta["k"])) / s_out[i]).astype(x.dtype)
            m_tc = m_tc.at[:, 0].set(m_real[:, 0])
            return x + jnp.where(use[i], m_tc, m_real), None

        x, _ = jax.lax.scan(body, x, (params["layers"], tcs, jnp.arange(L)))
        x = llama.rms_norm(x, params["final_norm"], cfg.norm_eps)
        head = params["embed"].T if cfg.tied else params["head"]
        return (x @ head).astype(jnp.float32)
    return f


def next_token_loss(logits, tokens):
    lp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32))
    return -jnp.take_along_axis(lp, tokens[:, 1:, None], -1)[:, 1:].mean()  # skip predicting from the sink


# ================================================================ 4. attribution graphs
PROMPTS = {
    "dallas": ("Fact: Dallas is a city in the state of", " Texas"),
    "opposite": ("The opposite of hot is", " cold"),
    "ioi": ("When Mary and John went to the store, John gave a drink to", " Mary"),
}


def attribute(params, cfg, tok, tcs, meta, text: str):
    """The attribution graph of a prompt (float32 model), with SmolLM2's sink at position 0 left out."""
    ids = jnp.array([tok.encode(text).ids])
    return trace.attribute(forward(params, cfg), ids, tcs, scale_in=meta["scale_in"],
                              scale_out=meta["scale_out"], kind="topk", k=meta["k"], skip_positions=(0,), batch=64)


def feature_contexts(params, cfg, tcs, meta, feats, ids, n_windows: int = 4096, seq: int = 64, top: int = 8,
                     seed: int = 1):
    """Top-activating windows for a list of (layer, feature) on held-out text: {(layer, f): [(acts, window_index,
    position)]}, plus the windows. One pass over n_windows × seq tokens."""
    rng = np.random.default_rng(seed)
    W = windows(ids, n_windows, rng, seq)
    io = mlp_io(params, cfg)
    s_in = np.asarray(meta["scale_in"])
    by_layer: dict = {}
    for layer, f in feats:
        by_layer.setdefault(int(layer), []).append(int(f))
    best = {(layer, f): [] for layer, fs in by_layer.items() for f in fs}
    enc = jax.jit(lambda tc, x, s: S.encode(tc, x * s, "topk", meta["k"]))
    for b0 in range(0, n_windows, 64):
        xin, _ = io(params, jnp.asarray(W[b0:b0 + 64]))
        for layer, fs in by_layer.items():
            tc = jax.tree.map(lambda a, layer=layer: a[layer], tcs)
            z = np.asarray(enc(tc, xin[layer], s_in[layer]))[:, fs].reshape(-1, seq - 1, len(fs))
            for j, f in enumerate(fs):
                a = z[:, :, j]
                for w in np.argsort(-a.max(1))[:top]:
                    best[(layer, f)].append((float(a[w].max()), b0 + int(w), int(a[w].argmax()) + 1))
    for key in best:
        best[key] = sorted(best[key], reverse=True)[:top]
    return best, W


def logit_effects(params, cfg, tcs, layer: int, f: int, top: int = 6):
    """The tokens a feature's decoder direction most promotes, read directly through the unembedding (ignoring
    every later layer — a quick label, not a measurement)."""
    v = tcs["W_dec"][layer, f]
    head = params["embed"].T if cfg.tied else params["head"]
    lg = np.asarray(v.astype(jnp.float32) @ head.astype(jnp.float32))
    return np.argsort(-lg)[:top], np.argsort(lg)[:top]
