"""Module 37 helpers: tiny models we can open up.

* An **attention-only transformer** (no MLPs, no LayerNorm, as in Elhage et al. 2021) that exposes every
  head's attention pattern and lets you switch heads off, trained on repeated random sequences so it
  develops induction heads (Olsson et al. 2022).
* **Othello**: legal random games generated on the GPU with pgx (Koyamada et al. 2023), a small GPT trained
  on move sequences (Li et al. 2022's Othello-GPT, scaled down), and residual-stream activations for
  linear probes (Nanda, Lee & Wattenberg 2023).
* The **toy model of superposition** (Elhage et al. 2022): 5 sparse features squeezed into 2 dimensions.

Everything is plain JAX + optax so every activation is a visible array.
"""
from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ---------------------------------------------------------------- attention-only transformer


def attn_only_init(key, vocab: int, d: int, n_layers: int, n_heads: int, max_len: int) -> dict:
    ks = jax.random.split(key, 3 + 2 * n_layers)
    s = 0.02
    layers = []
    for i in range(n_layers):
        k1, k2 = ks[3 + 2 * i], ks[4 + 2 * i]
        layers.append({"w_qkv": jax.random.normal(k1, (d, 3 * d)) * s,
                       "w_o": jax.random.normal(k2, (d, d)) * s})
    return {"emb": jax.random.normal(ks[0], (vocab, d)) * s,
            "pos": jax.random.normal(ks[1], (max_len, d)) * s,
            "unemb": jax.random.normal(ks[2], (d, vocab)) * s,
            "layers": layers}


def attn_only_apply(params: dict, tokens, n_heads: int, head_mask=None):
    """tokens [B, T] → (logits [B, T, V], attn [L, B, H, T, T]). head_mask [L, H]: 0 switches a head off."""
    B, T = tokens.shape
    x = params["emb"][tokens] + params["pos"][:T]
    d = x.shape[-1]
    hd = d // n_heads
    causal = jnp.tril(jnp.ones((T, T), bool))
    attns = []
    for li, lp in enumerate(params["layers"]):
        q, k, v = jnp.split(x @ lp["w_qkv"], 3, axis=-1)
        q, k, v = (t.reshape(B, T, n_heads, hd) for t in (q, k, v))
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(hd)
        w = jax.nn.softmax(jnp.where(causal, scores, -1e9), axis=-1)
        attns.append(w)
        z = jnp.einsum("bhts,bshd->bthd", w, v)                     # per-head outputs
        if head_mask is not None:
            z = z * head_mask[li][None, None, :, None]
        x = x + z.reshape(B, T, d) @ lp["w_o"]
    return x @ params["unemb"], jnp.stack(attns)


def repeated_batch(key, batch: int, seq_len: int, vocab: int, min_rep: int = 8, max_rep: int = 28):
    """Sequences "A A noise": a random segment A of random length m ∈ [min_rep, max_rep], immediately repeated,
    then random filler up to seq_len. Returns (tokens [B, T], repeat_mask [B, T]) where repeat_mask marks the
    second copy of A. Because m varies, the distance back to the first copy varies too: a model can only
    predict the second copy by finding the earlier occurrence of the current token and reading what came
    next. That is the induction mechanism."""
    k1, k2 = jax.random.split(key)
    m = jax.random.randint(k1, (batch, 1), min_rep, max_rep + 1)
    noise = jax.random.randint(k2, (batch, seq_len), 0, vocab)
    pos = jnp.arange(seq_len)[None, :]
    src = jnp.where(pos < m, pos, jnp.where(pos < 2 * m, pos - m, pos))      # second copy reads the first
    tokens = jnp.take_along_axis(noise, src, axis=1)
    repeat = (pos >= m) & (pos < 2 * m)
    return tokens, repeat


def split_losses(logits, tokens, repeat):
    """(loss on non-repeated tokens, loss on the repeated second copy), per-token cross-entropy.

    Position t predicts token t+1; the first token of the second copy is unpredictable (its previous occurrence
    is the very first token, nothing came before it to match), so it counts as non-repeated."""
    lp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), -1)
    nll = -jnp.take_along_axis(lp, tokens[:, 1:, None], -1)[..., 0]
    tgt_rep = repeat[:, 1:] & repeat[:, :-1]                                  # target in copy 2, and not its first token
    rep = tgt_rep.astype(jnp.float32)
    return (nll * (1 - rep)).sum() / (1 - rep).sum(), (nll * rep).sum() / rep.sum()


def train_induction(key, *, n_layers: int, vocab=64, d=64, n_heads=4, seq_len=64, batch=64, steps=1500,
                    lr=1e-3, log_every=25):
    """Train an attention-only model on "A A noise" sequences. Returns (params, history dict, seconds)."""
    k_init, k_data = jax.random.split(key)
    params = attn_only_init(k_init, vocab, d, n_layers, n_heads, seq_len)
    opt = optax.adam(lr)
    state = opt.init(params)

    def loss_fn(p, toks, rep):
        lg, _ = attn_only_apply(p, toks, n_heads)
        a, b = split_losses(lg, toks, rep)
        lp = jax.nn.log_softmax(lg[:, :-1].astype(jnp.float32), -1)
        total = -jnp.take_along_axis(lp, toks[:, 1:, None], -1).mean()
        return total, (a, b)

    @jax.jit
    def chunk(p, s, k):
        def body(carry, kk):
            p, s = carry
            toks, rep = repeated_batch(kk, batch, seq_len, vocab)
            (_, (a, b)), g = jax.value_and_grad(loss_fn, has_aux=True)(p, toks, rep)
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), (a, b)
        (p, s), (a, b) = jax.lax.scan(body, (p, s), jax.random.split(k, log_every))
        return p, s, a.mean(), b.mean()

    hist = {"step": [], "other": [], "repeated": []}
    t0 = time.perf_counter()
    for i, k in enumerate(jax.random.split(k_data, steps // log_every)):
        params, state, a, b = chunk(params, state, k)
        hist["step"].append((i + 1) * log_every)
        hist["other"].append(float(a))
        hist["repeated"].append(float(b))
    return params, hist, time.perf_counter() - t0


# ---------------------------------------------------------------- Othello games with pgx

PASS, PAD = 64, 65
OTHELLO_VOCAB = 66


def gen_games(key, n_games: int, max_moves: int = 64, chunk: int = 4096):
    """Random legal Othello games, played on the GPU with pgx.

    Returns numpy arrays, one row per game, one column per move t:
      tokens   [n, M]      move played at t (0–63 = square, 64 = pass, 65 = padding after the game ended)
      rel      [n, M, 64]  board *after* move t from the next mover's view: 0 empty, 1 mine, 2 theirs
      absb     [n, M, 64]  same board in absolute colours: 0 empty, 1 black, 2 white
      legal    [n, M, 65]  legal next moves after move t (squares + pass)
      valid    [n, M]      move t exists
    """
    import pgx

    env = pgx.make("othello")
    init = jax.vmap(env.init)
    step = jax.vmap(env.step)

    @jax.jit
    def play(k):
        k0, k1 = jax.random.split(k)
        st = init(jax.random.split(k0, chunk))

        def body(st, kk):
            alive = ~(st.terminated | st.truncated)
            logits = jnp.where(st.legal_action_mask, 0.0, -jnp.inf)
            a = jax.random.categorical(kk, logits)
            a = jnp.where(alive, a, 0)
            st2 = step(st, a)
            board = st2._x.board                                    # relative to the NEXT mover: 1 mine, -1 theirs
            sign = jnp.where(st2._x.turn == 0, 1, -1)[:, None]      # turn 0 = black to move
            absb = board * sign                                      # +1 black, -1 white
            rel = jnp.where(board == 1, 1, jnp.where(board == -1, 2, 0))
            ab = jnp.where(absb == 1, 1, jnp.where(absb == -1, 2, 0))
            tok = jnp.where(alive, a, PAD)
            return st2, (tok, rel.astype(jnp.int8), ab.astype(jnp.int8), st2.legal_action_mask, alive)

        _, out = jax.lax.scan(body, st, jax.random.split(k1, max_moves))
        return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), out)

    parts = [play(k) for k in jax.random.split(key, -(-n_games // chunk))]
    tok, rel, ab, legal, valid = (np.concatenate([np.asarray(p[i]) for p in parts])[:n_games] for i in range(5))
    return {"tokens": tok.astype(np.int32), "rel": rel, "absb": ab, "legal": legal, "valid": valid}


# ---------------------------------------------------------------- a small GPT for move sequences


def gpt_init(key, vocab: int, d: int, n_layers: int, n_heads: int, max_len: int) -> dict:
    ks = jax.random.split(key, 2 + n_layers)
    s = 0.02

    def block(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        return {"ln1": jnp.ones(d), "ln2": jnp.ones(d),
                "w_qkv": jax.random.normal(k1, (d, 3 * d)) * s, "w_o": jax.random.normal(k2, (d, d)) * s,
                "w_fc": jax.random.normal(k3, (d, 4 * d)) * s, "w_proj": jax.random.normal(k4, (4 * d, d)) * s}

    return {"emb": jax.random.normal(ks[0], (vocab, d)) * s, "pos": jax.random.normal(ks[1], (max_len, d)) * s,
            "ln_f": jnp.ones(d), "blocks": [block(k) for k in ks[2:]]}


def _rms(x, g):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6) * g


def gpt_apply(params: dict, tokens, n_heads: int, edit=None):
    """tokens [B, T] → (logits [B, T, V], resid [L+1, B, T, d]).

    resid[l] is the residual stream after l blocks (resid[0] = embeddings). `edit(layer, x)` may modify the
    residual stream after a block (used for interventions)."""
    B, T = tokens.shape
    x = params["emb"][tokens] + params["pos"][:T]
    d = x.shape[-1]
    hd = d // n_heads
    causal = jnp.tril(jnp.ones((T, T), bool))
    resid = [x]
    for li, bp in enumerate(params["blocks"]):
        h = _rms(x, bp["ln1"])
        q, k, v = (t.reshape(B, T, n_heads, hd) for t in jnp.split(h @ bp["w_qkv"], 3, -1))
        sc = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(hd)
        w = jax.nn.softmax(jnp.where(causal, sc, -1e9), -1)
        x = x + jnp.einsum("bhts,bshd->bthd", w, v).reshape(B, T, d) @ bp["w_o"]
        h = _rms(x, bp["ln2"])
        x = x + jax.nn.gelu(h @ bp["w_fc"]) @ bp["w_proj"]
        if edit is not None:
            x = edit(li + 1, x)
        resid.append(x)
    logits = _rms(x, params["ln_f"]) @ params["emb"].T                    # tied unembedding
    return logits, jnp.stack(resid)


def train_othello_gpt(key, games: dict, *, d=128, n_layers=4, n_heads=4, steps=2000, batch=128, lr=3e-4,
                      log_every=50, params=None, on_log=None):
    """Next-move prediction on game transcripts. Returns (params, history, seconds)."""
    tokens = jnp.asarray(games["tokens"])
    n, T = tokens.shape
    if params is None:
        params = gpt_init(key, OTHELLO_VOCAB, d, n_layers, n_heads, T)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 10), steps, lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    def loss_fn(p, toks):
        lg, _ = gpt_apply(p, toks[:, :-1], n_heads)
        tgt = toks[:, 1:]
        mask = (tgt != PAD).astype(jnp.float32)
        nll = -jnp.take_along_axis(jax.nn.log_softmax(lg.astype(jnp.float32), -1), tgt[..., None], -1)[..., 0]
        return (nll * mask).sum() / mask.sum()

    @partial(jax.jit, donate_argnums=(0, 1))
    def chunk(p, s, k, toks_all):
        def body(carry, kk):
            p, s = carry
            idx = jax.random.randint(kk, (batch,), 0, n)
            loss, g = jax.value_and_grad(loss_fn)(p, toks_all[idx])
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), loss
        (p, s), losses = jax.lax.scan(body, (p, s), jax.random.split(k, log_every))
        return p, s, losses.mean()

    hist = {"step": [], "loss": []}
    t0 = time.perf_counter()
    for i, k in enumerate(jax.random.split(key, steps // log_every)):
        params, state, loss = chunk(params, state, k, tokens)
        hist["step"].append((i + 1) * log_every)
        hist["loss"].append(float(loss))
        if on_log:
            on_log(hist)
    return params, hist, time.perf_counter() - t0


def legal_move_accuracy(params, games: dict, n_heads: int, n_eval: int = 1000, batch: int = 250) -> float:
    """Fraction of positions where the model's top predicted next move is legal."""
    hits = total = 0
    fwd = jax.jit(lambda p, t: gpt_apply(p, t, n_heads)[0])
    for i in range(0, n_eval, batch):
        toks = jnp.asarray(games["tokens"][i:i + batch])
        lg = fwd(params, toks[:, :-1])[..., :65]
        pred = np.asarray(jnp.argmax(lg, -1))                                # [b, T-1]
        legal = games["legal"][i:i + batch, :-1]                             # legal after move t
        has_next = games["valid"][i:i + batch, 1:]                           # a next move exists
        ok = np.take_along_axis(legal, pred[..., None], -1)[..., 0]
        hits += int((ok & has_next).sum())
        total += int(has_next.sum())
    return hits / max(total, 1)


def collect_resid(params, games: dict, n_heads: int, n_games: int, batch: int = 250):
    """Residual streams for probing: acts [L+1, N, d] over valid positions, plus matching labels."""
    fwd = jax.jit(lambda p, t: gpt_apply(p, t, n_heads)[1])
    acts, rel, ab = [], [], []
    for i in range(0, n_games, batch):
        toks = jnp.asarray(games["tokens"][i:i + batch])
        r = np.asarray(fwd(params, toks).astype(jnp.float32))               # [L+1, b, T, d]
        v = games["valid"][i:i + batch]
        acts.append(r[:, v])
        rel.append(games["rel"][i:i + batch][v])
        ab.append(games["absb"][i:i + batch][v])
    return np.concatenate(acts, 1), np.concatenate(rel).astype(np.int32), np.concatenate(ab).astype(np.int32)


def save_params(path, params, meta: dict):
    flat, tree = jax.tree_util.tree_flatten(params)
    np.savez(path, *[np.asarray(a) for a in flat], meta=np.array(repr(meta)))
    return tree


def load_params(path, like):
    z = np.load(path, allow_pickle=False)
    flat = [jnp.asarray(z[f"arr_{i}"]) for i in range(len(jax.tree.leaves(like)))]
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(like), flat)


# ---------------------------------------------------------------- toy model of superposition


def train_superposition(key, n_feat=5, n_hidden=2, sparsity=0.95, steps=15000, batch=1024, lr=1e-3):
    """Elhage et al. 2022: x' = ReLU(WᵀW x + b). Returns W [n_hidden, n_feat], b, final loss."""
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (n_hidden, n_feat)) * 0.1
    b = jnp.zeros(n_feat)
    importance = jnp.ones(n_feat)
    opt = optax.adam(lr)
    params = (W, b)
    st = opt.init(params)

    def loss(p, x):
        W, b = p
        out = jax.nn.relu(x @ W.T @ W + b)
        return (importance * (x - out) ** 2).sum(-1).mean()

    @jax.jit
    def run(p, st, k):
        def body(c, kk):
            p, st = c
            ka, kb = jax.random.split(kk)
            on = jax.random.uniform(ka, (batch, n_feat)) > sparsity
            x = jnp.where(on, jax.random.uniform(kb, (batch, n_feat)), 0.0)
            l, g = jax.value_and_grad(loss)(p, x)
            u, st = opt.update(g, st, p)
            return (optax.apply_updates(p, u), st), l
        (p, st), ls = jax.lax.scan(body, (p, st), jax.random.split(k, steps))
        return p, ls[-200:].mean()

    (W, b), final = run(params, st, k2)
    return np.asarray(W), np.asarray(b), float(final)
