"""Module 31 helpers: the models the lesson trains, built from YOUR cells.

Every function takes the cell / run_rnn / attention functions as arguments, so the lesson trains with
the code you wrote in exercises.py (or the reference solutions when LEARNJAX_SOLUTIONS=1).
"""
from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp
import optax

GATES = {"rnn": 1, "gru": 3, "lstm": 4}


def init_cell(key, kind: str, d_in: int, H: int):
    k1, k2 = jax.random.split(key)
    g = GATES[kind]
    p = {"Wx": jax.random.normal(k1, (d_in, g * H)) / jnp.sqrt(d_in),
         "Wh": jax.random.normal(k2, (H, g * H)) / jnp.sqrt(H),
         "b": jnp.zeros(g * H)}
    if kind == "lstm":                      # forget-gate bias 1 (Gers et al. 2000): remember by default
        p["b"] = p["b"].at[H:2 * H].set(1.0)
    return p


def zero_carry(kind: str, batch_shape, H: int):
    z = jnp.zeros((*batch_shape, H))
    return (z, z) if kind == "lstm" else z


def cell_params(kind: str, d_in: int, H: int) -> int:
    g = GATES[kind]
    return g * (d_in * H + H * H + H)


# ---------------------------------------------------------------- character language model

def init_lm(key, kind: str, V: int, E: int, H: int):
    k1, k2, k3 = jax.random.split(key, 3)
    return {"emb": 0.1 * jax.random.normal(k1, (V, E)), "cell": init_cell(k2, kind, E, H),
            "Wo": jax.random.normal(k3, (H, V)) / jnp.sqrt(H), "bo": jnp.zeros(V)}


def lm_loss(cell, run_rnn, kind, params, x, y):
    """x, y: (B, T) int token windows (y = x shifted by one). Mean next-char cross-entropy (nats)."""
    H = params["Wo"].shape[0]
    xs = params["emb"][x.T]                                       # (T, B, E) time-major
    _, hs = run_rnn(cell, params["cell"], zero_carry(kind, (x.shape[0],), H), xs)
    logits = hs @ params["Wo"] + params["bo"]                     # (T, B, V)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y.T).mean()


def _windows(key, data, batch, T):
    starts = jax.random.randint(key, (batch,), 0, data.shape[0] - T - 1)
    win = jax.vmap(lambda s: jax.lax.dynamic_slice(data, (s,), (T + 1,)))(starts)
    return win[:, :-1], win[:, 1:]


def make_lm_trainer(cell, run_rnn, kind, lr=3e-3, batch=64, T=64, chunk=100):
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    loss = functools.partial(lm_loss, cell, run_rnn, kind)

    @jax.jit
    def train_chunk(params, opt_state, key, data):
        def step(carry, k):
            p, s = carry
            x, y = _windows(k, data, batch, T)
            l, g = jax.value_and_grad(loss)(p, x, y)
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), l
        (params, opt_state), ls = jax.lax.scan(step, (params, opt_state), jax.random.split(key, chunk))
        return params, opt_state, ls.mean()

    @jax.jit
    def evaluate(params, data, key):
        x, y = _windows(key, data, 256, T)
        return loss(params, x, y)

    return opt, train_chunk, evaluate


def train_lm(cell, run_rnn, kind, V, train_ids, val_ids, H, E=64, steps=2000, seed=0, **kw):
    opt, train_chunk, evaluate = make_lm_trainer(cell, run_rnn, kind, **kw)
    params = init_lm(jax.random.key(seed), kind, V, E, H)
    s = opt.init(params)
    key = jax.random.key(seed + 1)
    curve, t0 = [], time.perf_counter()
    for i in range(steps // 100):
        key, k1, k2 = jax.random.split(key, 3)
        params, s, l = train_chunk(params, s, k1, train_ids)
        curve.append((100 * (i + 1), float(l), float(evaluate(params, val_ids, k2))))
    val = float(jnp.mean(jnp.stack([evaluate(params, val_ids, jax.random.key(100 + j)) for j in range(4)])))
    return {"params": params, "curve": curve, "val": val, "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------- seq2seq: reverse a digit string

PAD, BOS, OFFSET, V_S2S = 0, 1, 2, 12        # tokens: 0 pad, 1 start, 2..11 the digits 0..9


def reverse_batch(key, batch, Lmin, Lmax):
    """Random digit strings of length L in [Lmin, Lmax]; source left-padded, target = reversed, right-padded."""
    k1, k2 = jax.random.split(key)
    L = jax.random.randint(k1, (batch,), Lmin, Lmax + 1)
    digits = jax.random.randint(k2, (batch, Lmax), 0, 10) + OFFSET
    pos = jnp.arange(Lmax)
    valid = pos[None, :] < L[:, None]                               # first L positions are real
    seq = jnp.where(valid, digits, PAD)
    rev_idx = jnp.clip(L[:, None] - 1 - pos[None, :], 0, Lmax - 1)
    tgt = jnp.where(valid, jnp.take_along_axis(seq, rev_idx, axis=1), PAD)
    shift = Lmax - L                                                 # left-pad the source
    src_idx = jnp.clip(pos[None, :] - shift[:, None], 0, Lmax - 1)
    src = jnp.where(pos[None, :] >= shift[:, None], jnp.take_along_axis(seq, src_idx, axis=1), PAD)
    return src, tgt, L


def init_s2s(key, H, E=32, A=64, attention=True):
    ks = jax.random.split(key, 6)
    dec_in = E + (H if attention else 0)
    out_in = H + (H if attention else 0)
    p = {"emb": 0.1 * jax.random.normal(ks[0], (V_S2S, E)), "enc": init_cell(ks[1], "lstm", E, H),
         "dec": init_cell(ks[2], "lstm", dec_in, H),
         "Wo": jax.random.normal(ks[3], (out_in, V_S2S)) / jnp.sqrt(out_in), "bo": jnp.zeros(V_S2S)}
    if attention:
        p["att"] = {"Wq": jax.random.normal(ks[4], (H, A)) / jnp.sqrt(H),
                    "Wk": jax.random.normal(ks[5], (H, A)) / jnp.sqrt(H), "v": jnp.ones(A) / jnp.sqrt(A)}
    return p


def encode(fns, params, src):
    lstm, run_rnn = fns["lstm_cell"], fns["run_rnn"]
    H = params["enc"]["Wh"].shape[0]
    xs = params["emb"][src.T]
    final, hs = run_rnn(lstm, params["enc"], zero_carry("lstm", (src.shape[0],), H), xs)
    return final, jnp.swapaxes(hs, 0, 1)                            # enc states (B, S, H)


def make_step_fn(fns, params, enc_final, enc_hs, src):
    """Decoder step closed over the encoder outputs: (state, token) -> (logits, state)."""
    lstm = fns["lstm_cell"]
    attention = "att" in params
    mask = src != PAD

    def step(state, tok):
        x = params["emb"][tok]
        weights = jnp.zeros(src.shape, jnp.float32)
        if attention:
            ctx, weights = fns["additive_attention"](params["att"], state[0], enc_hs, enc_hs, mask)
            x = jnp.concatenate([x, ctx], -1)
        state = lstm(params["dec"], state, x)
        feat = jnp.concatenate([state[0], ctx], -1) if attention else state[0]
        return feat @ params["Wo"] + params["bo"], state, weights

    return step


def s2s_teacher_forced(fns, params, src, tgt):
    """Logits (B, T, V) and attention weights (B, T, S) with the true previous token fed at each step."""
    enc_final, enc_hs = encode(fns, params, src)
    step = make_step_fn(fns, params, enc_final, enc_hs, src)
    inp = jnp.concatenate([jnp.full((tgt.shape[0], 1), BOS), tgt[:, :-1]], 1)

    def body(state, tok):
        logits, state, w = step(state, tok)
        return state, (logits, w)

    _, (logits, w) = jax.lax.scan(body, enc_final, inp.T)
    return jnp.swapaxes(logits, 0, 1), jnp.swapaxes(w, 0, 1)


def s2s_loss(fns, params, src, tgt):
    logits, _ = s2s_teacher_forced(fns, params, src, tgt)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, tgt)
    m = (tgt != PAD).astype(jnp.float32)
    return (ce * m).sum() / m.sum()


def train_s2s(fns, attention, H=128, Lmin=4, Lmax=24, steps=1500, batch=128, lr=3e-3, seed=0):
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    params = init_s2s(jax.random.key(seed), H, attention=attention)
    s = opt.init(params)

    @jax.jit
    def chunk(params, s, key):
        def body(carry, k):
            p, st = carry
            src, tgt, _ = reverse_batch(k, batch, Lmin, Lmax)
            l, g = jax.value_and_grad(lambda pp: s2s_loss(fns, pp, src, tgt))(p)
            u, st = opt.update(g, st, p)
            return (optax.apply_updates(p, u), st), l
        (params, s), ls = jax.lax.scan(body, (params, s), jax.random.split(key, 100))
        return params, s, ls.mean()

    key, curve, t0 = jax.random.key(seed + 7), [], time.perf_counter()
    for i in range(steps // 100):
        key, k = jax.random.split(key)
        params, s, l = chunk(params, s, k)
        curve.append((100 * (i + 1), float(l)))
    return {"params": params, "curve": curve, "seconds": time.perf_counter() - t0}


def s2s_accuracy_by_length(fns, params, Lmin, Lmax, n=512, seed=123):
    """Greedy-decode fresh examples with YOUR greedy_decode; per-length token and exact-match accuracy."""
    src, tgt, L = reverse_batch(jax.random.key(seed), n, Lmin, Lmax)

    @jax.jit
    def decode(params, src):
        enc_final, enc_hs = encode(fns, params, src)
        step = make_step_fn(fns, params, enc_final, enc_hs, src)

        def step_fn(state, tok):
            logits, state, _ = step(state, tok)
            return logits, state

        return fns["greedy_decode"](step_fn, enc_final, jnp.full((src.shape[0],), BOS), src.shape[1])

    pred = decode(params, src)
    valid = tgt != PAD
    tok_ok = (pred == tgt) & valid
    out = {}
    for length in range(Lmin, Lmax + 1):
        sel = L == length
        if int(sel.sum()) == 0:
            continue
        tok_acc = float(tok_ok[sel].sum() / valid[sel].sum())
        exact = float(jnp.mean(jnp.all(tok_ok[sel] | ~valid[sel], axis=1)))
        out[length] = (tok_acc, exact)
    return out
