"""Module 46 model code: a digit classifier set by hand, a transformer compiled from a program, merging two
fine-tunes of SmolLM2-135M, and a ROME-style edit of one fact in SmolLM2-360M.

Provided (not exercises): the CNN and its Adam baseline, the layer-2 line and corner detectors, the residual-stream
layout and counting MLP of the compiled transformer, the fine-tuning and merge evaluation, DARE, and ROME's key,
value and covariance estimates. Your exercises supply the filters, the readout, the selector, TIES and the edit.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

RUNS = Path("/course/runs/46-weights-by-hand")


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# ================================================================ 1. a digit classifier set by hand
# Toumbourou's architecture (5,994 weights): Conv 1→8 5×5, ReLU, pool 2; Conv 8→16 5×5, act, pool 2; Linear 256→10.
N_ORIENT = 8
THETAS = np.arange(N_ORIENT) * np.pi / N_ORIENT


def conv(x, w, b):
    return jax.lax.conv_general_dilated(x, w, (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")) + b


def pool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")


def relu1(z):
    """A ReLU capped at 1 (like MobileNet's ReLU6): with large weights it becomes a 0/1 'did it fire' switch."""
    return jnp.clip(z, 0.0, 1.0)


def cnn(p, x):
    """x: (B, 28, 28, 1) in [0, 1] → logits (B, 10); also returns the 256 pooled layer-2 features."""
    h = pool(jax.nn.relu(conv(x, p["w1"], p["b1"])))
    f = pool(relu1(conv(h, p["w2"], p["b2"]))).reshape(x.shape[0], -1)
    return f @ p["w3"] + p["b3"], f


def count_weights(p) -> int:
    return sum(int(np.prod(a.shape)) for a in jax.tree.leaves(p))


def layer2_detectors(scale: float = 200.0):
    """16 hand-set detectors over the 8 orientation maps (12 × 12 after pooling), 5 × 5 each:
    0–7  'long line': orientation o present along a 5-cell line at angle o (a stroke, not a speck);
    8–15 'corner': orientation o in the top-left 3 × 3 and the perpendicular one in the bottom-right 3 × 3.
    The large scale turns relu1 into a switch: the feature is ~1 when its weighted sum beats the bias."""
    w = np.zeros((5, 5, N_ORIENT, 16))
    for o, t in enumerate(THETAS):
        for s in range(-2, 3):
            w[round(2 + s * np.sin(t)), round(2 + s * np.cos(t)), o, o] = 1 / 5
        w[:3, :3, o, 8 + o] = 1 / 9
        w[2:, 2:, (o + N_ORIENT // 2) % N_ORIENT, 8 + o] = 1 / 9
    return jnp.asarray(w * scale, jnp.float32), jnp.full((16,), -0.1 * scale, jnp.float32)


def hand_layers(ex, b1: float = -0.05):
    """Layers 1 and 2 from your ridge_filter and the detectors above; the readout comes from counting."""
    w1 = jnp.stack([ex.ridge_filter(t) for t in THETAS], -1)[:, :, None, :]
    w2, b2 = layer2_detectors()
    return {"w1": w1, "b1": jnp.full((N_ORIENT,), b1, jnp.float32), "w2": w2, "b2": b2,
            "w3": jnp.zeros((256, 10)), "b3": jnp.zeros(10)}


@jax.jit
def _features(p, x):
    return cnn(p, x)[1]


def features(p, X, batch: int = 5000):
    return np.concatenate([np.asarray(_features(p, jnp.asarray(X[i:i + batch]))) for i in range(0, len(X), batch)])


def accuracy(p, X, y, batch: int = 5000) -> float:
    pred = np.concatenate([np.asarray(jax.jit(lambda p, x: cnn(p, x)[0].argmax(1))(p, jnp.asarray(X[i:i + batch])))
                           for i in range(0, len(X), batch)])
    return float((pred == y).mean())


def mnist_arrays():
    from learnjax.data import mnist

    d = mnist()
    return (d["train_images"][..., None].astype(np.float32) / 255, d["train_labels"],
            d["test_images"][..., None].astype(np.float32) / 255, d["test_labels"])


def train_adam(Xtr, ytr, Xte, yte, checkpoints=(50, 500, 5000), batch: int = 64, seed: int = 0):
    """The same CNN trained by Adam (lr 1e-3, batch 64), test accuracy at each checkpoint."""
    import optax

    k = jax.random.split(jax.random.key(seed), 3)
    p = {"w1": jax.random.normal(k[0], (5, 5, 1, 8)) * 0.2, "b1": jnp.zeros(8),
         "w2": jax.random.normal(k[1], (5, 5, 8, 16)) * 0.07, "b2": jnp.zeros(16),
         "w3": jax.random.normal(k[2], (256, 10)) * 0.06, "b3": jnp.zeros(10)}
    opt = optax.adam(1e-3)
    st = opt.init(p)

    @jax.jit
    def step(p, st, x, y):
        g = jax.grad(lambda p: optax.softmax_cross_entropy_with_integer_labels(cnn(p, x)[0], y).mean())(p)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st

    rng, out = np.random.default_rng(seed), {}
    for s in range(1, max(checkpoints) + 1):
        i = rng.integers(0, len(Xtr), batch)
        p, st = step(p, st, jnp.asarray(Xtr[i]), jnp.asarray(ytr[i]))
        if s in checkpoints:
            out[s] = {"test_acc": accuracy(p, Xte, yte), "w1": np.asarray(p["w1"]).tolist()}
    return out


# ================================================================ 2. a transformer compiled from a program
# Residual stream as named variables (Tracr's layout): [token one-hot V | position one-hot L | BOS share | count]
V_TOK, L_POS, MAX_COUNT = 6, 9, 8                 # tokens 0..4 plus BOS = 5; up to 8 tokens after BOS
BOS = V_TOK - 1
TOK = slice(0, V_TOK)
POS = slice(V_TOK, V_TOK + L_POS)
D_MODEL = V_TOK + L_POS


def embed(seq):
    x = np.zeros((len(seq), D_MODEL), np.float32)
    x[np.arange(len(seq)), seq] = 1
    x[np.arange(len(seq)), V_TOK + np.arange(len(seq))] = 1
    return jnp.asarray(x)


def attention(x, Wq, Wk, Wv):
    """One bidirectional head (an encoder, like Tracr's): softmax(x Wq (x Wk)ᵀ) x Wv; returns (out, weights)."""
    a = jax.nn.softmax((x @ Wq) @ (x @ Wk).T, -1)
    return a @ (x @ Wv), a


def reverse_head(ex, L: int):
    """Reverse = 'read the token at position L−1−i', using your position_selector on the position variables."""
    Pq, Pk = ex.position_selector(np.arange(L)[::-1])
    Wq = jnp.zeros((D_MODEL, L)).at[POS.start:POS.start + L].set(Pq)
    Wk = jnp.zeros((D_MODEL, L)).at[POS.start:POS.start + L].set(Pk)
    Wv = jnp.zeros((D_MODEL, V_TOK)).at[TOK].set(jnp.eye(V_TOK))
    return Wq, Wk, Wv


def histogram_head(scale: float = 50.0):
    """Histogram, step 1: each token attends equally to every copy of itself and to BOS; the share it gives BOS
    is 1 / (count + 1). (RASP: select(tokens, tokens, ==) or BOS; aggregate the BOS indicator.)"""
    Wq = np.zeros((D_MODEL, V_TOK))
    Wq[TOK] = scale * np.eye(V_TOK)
    Wq[TOK, BOS] += scale                                    # every query also matches BOS
    Wk = np.zeros((D_MODEL, V_TOK))
    Wk[TOK] = np.eye(V_TOK)
    Wv = np.zeros((D_MODEL, 1))
    Wv[BOS, 0] = 1
    return tuple(map(jnp.asarray, (Wq, Wk, Wv)))


def count_mlp(max_count: int = MAX_COUNT, s: float = 1000.0):
    """Histogram, step 2: a ReLU MLP that turns x = 1/(c+1) into a one-hot count c ∈ {0..max_count}.
    Each threshold t_k sits between 1/(k+1) and 1/k; step_k(x) = relu(s(t_k − x) + 1) − relu(s(t_k − x)) is a
    clean 0/1 'c ≥ k'. Then one-hot(c) = step_c − step_{c+1} (with step_0 = 1). 2·max_count hidden units."""
    ks = np.arange(1, max_count + 1)
    t = (1 / (ks + 1) + 1 / ks) / 2
    W1 = -s * np.ones((1, 2 * max_count))
    b1 = np.concatenate([s * t + 1, s * t])
    S = np.concatenate([np.eye(max_count), -np.eye(max_count)])          # hidden → steps
    M = np.zeros((max_count, max_count + 1))                              # steps → one-hot
    M[np.arange(max_count), ks] += 1
    M[np.arange(max_count), ks - 1] -= 1
    b2 = np.zeros(max_count + 1)
    b2[0] = 1
    return tuple(map(jnp.asarray, (W1, b1, S @ M, b2)))


def run_histogram(seq):
    x = embed(np.r_[BOS, seq])
    share, attn = attention(x, *histogram_head())
    W1, b1, W2, b2 = count_mlp()
    counts = jax.nn.relu(share @ W1 + b1) @ W2 + b2
    return np.asarray(counts.argmax(1))[1:], np.asarray(share)[1:, 0], np.asarray(attn)


# ================================================================ 3. merging fine-tunes (SmolLM2-135M)
MERGE_TASKS = ("shakespeare", "astro")             # fine-tune on each; "detective" is held out (forgetting)


def merge_texts():
    """Plain text for the three domains: Shakespeare (fine-tune A), radio-astronomy abstracts (fine-tune B), and
    detective novels (neither: measures what merging costs elsewhere)."""
    from tokenizers import Tokenizer

    from learnjax.data import detective_corpus, radio_abstracts, tiny_shakespeare

    sh = tiny_shakespeare()
    astro = "\n\n".join(a["abstract"] for a in radio_abstracts()[:20000])
    det = detective_corpus()
    dtok = Tokenizer.from_file(str(det["tokenizer_path"]))
    detective = dtok.decode(det["val"][:300_000].astype(np.int64).tolist())
    return {"shakespeare": sh, "astro": astro, "detective": detective}


def tokenize_splits(tok, texts, val_tokens: int = 65_536):
    """Token arrays per domain: the last val_tokens for validation, the rest for training."""
    out = {}
    for name, text in texts.items():
        ids = np.asarray(tok.encode(text).ids, np.int32)
        out[name] = {"train": ids[:-val_tokens], "val": ids[-val_tokens:]}
    return out


def task_vectors(ft, base):
    return jax.tree.map(lambda a, b: a.astype(jnp.float32) - b.astype(jnp.float32), ft, base)


def add_delta(base, delta, scale: float = 1.0):
    return jax.tree.map(lambda b, d: (b.astype(jnp.float32) + scale * d).astype(b.dtype), base, delta)


def dare(delta, p: float, key):
    """DARE (Yu et al. 2023): drop each entry of a task vector with probability p, rescale the rest by 1/(1−p)."""
    leaves, tree = jax.tree.flatten(delta)
    keys = jax.random.split(key, len(leaves))
    out = [jnp.where(jax.random.bernoulli(k, 1 - p, a.shape), a / (1 - p), 0.0) for k, a in zip(keys, leaves)]
    return jax.tree.unflatten(tree, out)


def ties_tree(ex, deltas, density: float, lam: float):
    """Your ties_merge applied tensor by tensor (as mergekit does), layer by layer for stacked weights."""
    def one(*leaves):
        a = jnp.stack([x.reshape(-1) for x in leaves])
        return ex.ties_merge(a, density, lam).reshape(leaves[0].shape)

    def leaf(*leaves):
        if leaves[0].ndim >= 3:                                            # stacked (L, ...) layer weights
            return jnp.stack([one(*[x[i] for x in leaves]) for i in range(leaves[0].shape[0])])
        return one(*leaves)

    return jax.tree.map(leaf, *deltas)


# ================================================================ 4. editing one fact (SmolLM2-360M)
def forward_to_mlp(params, cfg, tokens, layer: int):
    """Run SmolLM2 up to `layer`'s MLP. Returns (residual before that MLP, the MLP's hidden activation =
    silu(h @ w_gate) * (h @ w_up), i.e. the key ROME reads). tokens: (B, T)."""
    from learnjax import llama

    B, T = tokens.shape
    positions = jnp.broadcast_to(jnp.arange(T), (B, T))
    x = params["embed"][tokens]
    below = jax.tree.map(lambda a: a[:layer], params["layers"])
    x, _ = jax.lax.scan(lambda x, lp: (llama._layer(cfg, x, lp, positions, None, 0)[0], None), x, below)
    lp = jax.tree.map(lambda a: a[layer], params["layers"])
    # the attention half of the layer, then the MLP's input
    x_mid = _attn_half(cfg, x, lp, positions)
    h = llama.rms_norm(x_mid, lp["mlp_norm"], cfg.norm_eps)
    return x_mid, jax.nn.silu(h @ lp["w_gate"]) * (h @ lp["w_up"])


def _attn_half(cfg, x, lp, positions):
    from learnjax import llama

    # llama._layer = attention half + MLP half; run it with the MLP's output matrix zeroed to get the first half
    lp0 = dict(lp, w_down=jnp.zeros_like(lp["w_down"]))
    return llama._layer(cfg, x, lp0, positions, None, 0)[0]


def forward_with_mlp_delta(params, cfg, tokens, layer: int, where, delta):
    """Logits with `delta` added to `layer`'s MLP output at positions `where` (B,) — ROME's search for v*."""
    from learnjax import llama

    B, T = tokens.shape
    positions = jnp.broadcast_to(jnp.arange(T), (B, T))
    x_mid, hid = forward_to_mlp(params, cfg, tokens, layer)
    lp = jax.tree.map(lambda a: a[layer], params["layers"])
    mlp_out = hid @ lp["w_down"]
    mlp_out = mlp_out.at[jnp.arange(B), where].add(delta.astype(mlp_out.dtype))
    x = x_mid + mlp_out
    above = jax.tree.map(lambda a: a[layer + 1:], params["layers"])
    x, _ = jax.lax.scan(lambda x, lp: (llama._layer(cfg, x, lp, positions, None, 0)[0], None), x, above)
    x = llama.rms_norm(x, params["final_norm"], cfg.norm_eps)
    return (x @ params["embed"].T).astype(jnp.float32)


def key_covariance(params, cfg, tokens, layer: int, batch: int = 8, seq: int = 512, n_batches: int = 24):
    """C = mean of k kᵀ over ordinary text (ROME uses Wikipedia; we use detective novels): (d_ff, d_ff) float32."""
    f = jax.jit(lambda p, t: forward_to_mlp(p, cfg, t, layer)[1], static_argnums=())
    C, n = 0.0, 0
    for i in range(n_batches):
        t = tokens[i * batch * seq:(i + 1) * batch * seq].reshape(batch, seq)
        k = f(params, jnp.asarray(t)).astype(jnp.float32).reshape(-1, cfg.d_ff)
        C = C + k.T @ k
        n += k.shape[0]
    return C / n
