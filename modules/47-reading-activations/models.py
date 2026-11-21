"""Module 47 model code: lenses, sparse autoencoders, feature dashboards, steering and cross-model maps, all on
SmolLM2 (135M, and 360M for the last section). The generic parts (residual capture, edits, SAE losses) live in
`learnjax/interp.py`, which Module 48 reuses; this file holds the experiment-specific pieces.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from learnjax import interp

RUNS = Path("/course/runs/47-reading-activations")
DOMAINS = ("detective", "astro", "shakespeare")
MIX = (0.70, 0.25, 0.05)          # SAE training mix by domain
SAE_LAYER = 15                    # the residual stream after block 15 of 30
SEQ = 128                         # window length; position 0 (an attention sink, norm ~100× the rest) is dropped


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# ================================================================ data
def corpus(tok, val_tokens: int = 200_000):
    """SmolLM2 token ids per domain: detective novels (the deepseek-v3-jax corpus, decoded and re-tokenized),
    radio-astronomy abstracts, and Shakespeare. Cached under RUNS as .npy. Returns {domain: {"train", "val"}}."""
    RUNS.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in DOMAINS:
        f = RUNS / f"tokens_{name}.npy"
        if not f.exists():
            np.save(f, np.asarray(tok.encode(_text(name)).ids, np.int32))
        ids = np.load(f)
        n_val = min(val_tokens, len(ids) // 10)
        out[name] = {"train": ids[:-n_val], "val": ids[-n_val:]}
    return out


def _text(name):
    from tokenizers import Tokenizer

    from learnjax.data import detective_corpus, radio_abstracts, tiny_shakespeare

    if name == "shakespeare":
        return tiny_shakespeare()
    if name == "astro":
        return "\n\n".join(a["title"] + "\n" + a["abstract"] for a in radio_abstracts())
    det = detective_corpus()
    dtok = Tokenizer.from_file(str(det["tokenizer_path"]))
    ids = det["train"][:6_000_000].astype(np.int64)
    return "".join(dtok.decode(ids[i:i + 200_000].tolist()) for i in range(0, len(ids), 200_000))


def windows(ids, n: int, rng, seq: int = SEQ):
    """n random windows of length seq from a token array: (n, seq) int32."""
    s = rng.integers(0, len(ids) - seq, n)
    return np.stack([ids[i:i + seq] for i in s])


def mixed_windows(data, n: int, rng, split: str = "train", mix=MIX):
    """n windows drawn from the three domains in proportion `mix`; also returns each window's domain index."""
    dom = rng.choice(len(DOMAINS), n, p=mix)
    w = np.zeros((n, SEQ), np.int32)
    for d, name in enumerate(DOMAINS):
        m = dom == d
        if m.any():
            w[m] = windows(data[name][split], int(m.sum()), rng)
    return w, dom


def val_windows(data, per_domain: int | None = None):
    """Non-overlapping validation windows per domain: (n, SEQ) int32 and domain index per window."""
    ws, ds = [], []
    for d, name in enumerate(DOMAINS):
        v = data[name]["val"]
        n = len(v) // SEQ if per_domain is None else min(per_domain, len(v) // SEQ)
        ws.append(v[:n * SEQ].reshape(n, SEQ))
        ds.append(np.full(n, d))
    return np.concatenate(ws), np.concatenate(ds)


# ================================================================ 1. lenses
def lens_kl(params, cfg, lens, R, final_logits):
    """Per-layer mean KL(final ‖ lens) for layers 0 … L−1 of collected residuals R (L+1, N, d); lens=None is the
    logit lens. Scans over layers so only one layer's logits are alive at a time."""
    lp_final = jax.nn.log_softmax(final_logits)

    @jax.checkpoint
    def one(h, A, b):
        if lens is None:
            lg = interp.unembed(params, cfg, h)
        else:
            hf = h.astype(jnp.float32)
            lg = interp.unembed(params, cfg, (hf + hf @ A + b).astype(h.dtype))
        lq = jax.nn.log_softmax(lg)
        kl = (jnp.exp(lp_final) * (lp_final - lq)).sum(-1).mean()
        top1 = (lg.argmax(-1) == final_logits.argmax(-1)).mean()
        return kl, top1

    L = cfg.n_layers
    A = jnp.zeros((L, 1, 1)) if lens is None else lens["A"]
    b = jnp.zeros((L, 1)) if lens is None else lens["b"]
    _, (kls, top1) = jax.lax.scan(lambda c, x: (c, one(*x)), 0.0, (R[:L], A, b))
    return kls, top1


# ================================================================ 2. sparse autoencoders
SAE_SPECS = (                     # train.py adds ReLU + L1 SAEs, one per --l1 coefficient
    {"name": "topk16", "kind": "topk", "k": 16, "l1": 0.0},
    {"name": "topk32", "kind": "topk", "k": 32, "l1": 0.0},
    {"name": "topk64", "kind": "topk", "k": 64, "l1": 0.0},
)
MAIN_SAE = "topk32"
N_LATENTS = 8192
OUTLIER = 10.0                    # tokens whose residual norm is > 10× the median are left out (attention sinks)


def acts_fn(cfg, layer: int = SAE_LAYER):
    """jit: windows (B, SEQ) → residual after `layer` at positions 1… : (B, SEQ−1, d) float32."""
    return jax.jit(lambda p, t: interp.forward(p, cfg, t, collect=True, upto=layer)["resid"][-1][:, 1:]
                   .astype(jnp.float32))


def typical(acts, median_norm: float):
    """Mask of tokens that are not massive-norm outliers."""
    return jnp.linalg.norm(acts, axis=-1) < OUTLIER * median_norm


def save_sae(path, sae, meta):
    np.savez(path, **{k: np.asarray(v) for k, v in sae.items()}, meta=json.dumps(meta))


def load_sae(path):
    z = np.load(path)
    return {k: jnp.asarray(z[k]) for k in ("W_enc", "b_enc", "W_dec", "b_dec")}, json.loads(str(z["meta"]))


def splice_edit(sae, kind: str, k: int, scale: float, median_norm: float, layer: int = SAE_LAYER):
    """An interp edit that replaces h_layer (typical tokens, positions ≥ 1) by the SAE's reconstruction."""
    def f(x):
        h = x[:, 1:].astype(jnp.float32)
        keep = typical(h, median_norm)[..., None]
        hh = interp.sae_decode(sae, interp.sae_encode(sae, h.reshape(-1, h.shape[-1]) * scale, kind, k))
        hh = hh.reshape(h.shape) / scale
        return x.at[:, 1:].set(jnp.where(keep, hh, h).astype(x.dtype))
    return interp.at_layer(layer, f)


def replace_edit(value, median_norm: float, layer: int = SAE_LAYER):
    """Replace h_layer (typical tokens, positions ≥ 1) by a constant: zeros, or the mean activation."""
    def f(x):
        h = x[:, 1:].astype(jnp.float32)
        keep = typical(h, median_norm)[..., None]
        return x.at[:, 1:].set(jnp.where(keep, jnp.broadcast_to(value, h.shape), h).astype(x.dtype))
    return interp.at_layer(layer, f)


# ================================================================ 3. steering
SUBJECTS = ["The movie", "The food", "My day", "The hotel", "The book", "The concert", "The weather", "The service",
            "The new phone", "The game", "Our trip", "The class", "The party", "The restaurant", "The show", "My job",
            "The coffee", "The park", "The album", "The flight"]
BUILD_POS = ["wonderful", "great", "fantastic", "delightful", "excellent"]
BUILD_NEG = ["terrible", "awful", "horrible", "dreadful", "miserable"]
# the behaviour metric uses *different* words, so steering can't score by copying the words it was built from
EVAL_POS = [" good", " nice", " amazing", " lovely", " perfect", " beautiful", " enjoyable", " fun"]
EVAL_NEG = [" bad", " poor", " boring", " disappointing", " worst", " gross", " ugly", " sad"]
NEUTRAL = ["I went to the restaurant last night and the food was", "Yesterday I watched a movie. Honestly, it was",
           "We stayed at the hotel for three nights. The room was", "I finished the book this morning. I thought it was",
           "The concert ended at midnight. Overall the music was", "My first week at the new job has been",
           "The weather this weekend was", "I tried the new coffee shop downtown. The coffee was",
           "Our flight landed an hour ago. The flight was", "The party at Sam's house was",
           "I bought a new phone last week. So far it is", "The museum tour this afternoon was"]
CONTEXTS = [  # (positive, negative) with the sentiment stated before the question
    ("The food was delicious and the staff were friendly. In one word, the meal was",
     "The food was cold and the staff were rude. In one word, the meal was"),
    ("The plot was gripping and the acting superb. All in all, the film was",
     "The plot was dull and the acting wooden. All in all, the film was"),
    ("The room was clean, quiet and bright. Our stay was",
     "The room was dirty, noisy and dark. Our stay was"),
    ("Everyone laughed and danced all night. The party was",
     "Nobody talked and people left early. The party was"),
    ("The band played every hit and the crowd sang along. The concert was",
     "The band was late and the sound kept cutting out. The concert was"),
    ("The battery lasts two days and the screen is sharp. The phone is",
     "The battery dies by noon and the screen cracked. The phone is"),
]
ACTADD = ("I love it.", "I hate it.")
LEXICON_POS = {"good", "great", "nice", "excellent", "wonderful", "amazing", "lovely", "perfect", "beautiful",
               "enjoyable", "fun", "happy", "delicious", "fantastic", "best", "impressed", "comfortable", "friendly",
               "love", "loved", "pleasant", "delightful", "awesome", "glad", "enjoyed", "favorite", "incredible"}
LEXICON_NEG = {"bad", "poor", "terrible", "awful", "horrible", "worst", "boring", "disappointing", "ugly", "sad",
               "gross", "dreadful", "miserable", "hate", "hated", "scared", "mess", "angry", "rude", "dirty", "broken",
               "disappointed", "annoying", "painful", "sick", "upset", "unfortunately", "nightmare"}


def word_ids(tok, words):
    ids = [tok.encode(w).ids for w in words]
    bad = [w for w, i in zip(words, ids) if len(i) != 1]
    assert not bad, f"not single tokens: {bad}"
    return jnp.asarray([i[0] for i in ids])


def pad_batch(seqs, length: int | None = None):
    """Right-pad token lists with 0 (causal attention: later padding can't affect earlier positions).
    Returns (tokens (N, T) int32, index of each sequence's last real token (N,))."""
    T = length or max(map(len, seqs))
    a = np.zeros((len(seqs), T), np.int32)
    for i, s in enumerate(seqs):
        a[i, :len(s)] = s
    return jnp.asarray(a), jnp.asarray([len(s) - 1 for s in seqs])


def sentiment_score(logits, last, pos_ids, neg_ids):
    """Per prompt: log P(next token ∈ EVAL_POS) − log P(next token ∈ EVAL_NEG), at each prompt's last token."""
    lp = jax.nn.log_softmax(logits[jnp.arange(logits.shape[0]), last])
    return jax.nn.logsumexp(lp[:, pos_ids], -1) - jax.nn.logsumexp(lp[:, neg_ids], -1)


def lexicon_balance(text: str) -> int:
    """+1 if a continuation uses more positive than negative lexicon words, −1 if more negative, else 0."""
    words = [w.strip(".,!?;:'\"()").lower() for w in text.split()]
    p, n = sum(w in LEXICON_POS for w in words), sum(w in LEXICON_NEG for w in words)
    return int(np.sign(p - n))


def steer_edit(layer, vec):
    """Add vec at h_layer, positions ≥ 1 (position 0 is the attention sink)."""
    return lambda i, x: jnp.where(i == layer, x.at[:, 1:].add(vec.astype(x.dtype)), x)


def project_out(u):
    """x ↦ x − (x·u) u for every token: the activation-side twin of ablate_direction."""
    def f(i, x):
        xf = x.astype(jnp.float32)
        return (xf - (xf @ u)[..., None] * u).astype(x.dtype)
    return f


def orthogonalize(params, ex, u, alpha: float = 1.0):
    """Bake direction u out of everything that writes to the residual stream (token embeddings, each attention
    output matrix, each MLP down-projection) with your ablate_direction. The output head is untied first so the
    unembedding is left alone. Returns new params (with params["head"])."""
    p = dict(params)
    p["head"] = params["embed"].T
    f = lambda W: ex.ablate_direction(W.astype(jnp.float32), u, alpha).astype(W.dtype)
    p["embed"] = f(params["embed"])
    p["layers"] = dict(params["layers"], wo=jax.vmap(f)(params["layers"]["wo"]),
                       w_down=jax.vmap(f)(params["layers"]["w_down"]))
    return p


# ================================================================ 4. reading features
def domain_readout(fires, dom, n_classes: int = 3, alpha: float = 1.0):
    """Module 46's naive Bayes readout, on SAE features: fires (N, F) bool per window, dom (N,) labels.
    Returns (W (F, C), b (C,)) with log-odds weights — the same formulas as 46's firing_rates/log_odds_readout."""
    onehot = jnp.asarray(dom)[:, None] == jnp.arange(n_classes)
    counts = onehot.T.astype(jnp.float32) @ jnp.asarray(fires, jnp.float32)
    n = onehot.sum(0).astype(jnp.float32)[:, None]
    rates = (counts + alpha) / (n + 2 * alpha)
    W = (jnp.log(rates) - jnp.log1p(-rates)).T
    prior = onehot.mean(0)
    b = jnp.log1p(-rates).sum(1) + jnp.log(prior)
    return W, b, rates
