"""Module 32 helpers: one Transformer stack, two ways to train it (BERT's masked LM, GPT's next token).

The block is Module 16's (pre-LayerNorm, GELU MLP, learned positions, tied embeddings); the only structural
difference between the two models is the attention mask passed in. Training runs `steps` updates as one
jitted `lax.scan` (Module 10) with bf16 compute and f32 master weights (Module 08).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

TEXT_VOCAB = 8192          # the detective corpus BPE (Module 16 / deepseek-v3-jax)
MASK_ID = 8192             # two extra tokens appended to the vocabulary
CLS_ID = 8193


@dataclass(frozen=True)
class Cfg:
    vocab: int = TEXT_VOCAB + 2
    seq_len: int = 128
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4


def layer_norm(x, g, b, eps=1e-5):
    x32 = x.astype(jnp.float32)
    mu = x32.mean(-1, keepdims=True)
    var = ((x32 - mu) ** 2).mean(-1, keepdims=True)
    return ((x32 - mu) * jax.lax.rsqrt(var + eps) * g + b).astype(x.dtype)


def init(key, cfg: Cfg) -> dict:
    d, n = cfg.d_model, cfg.n_layers
    keys = iter(jax.random.split(key, 3 + 4 * n))

    def normal(shape, s=0.02):
        return jax.random.normal(next(keys), shape, jnp.float32) * s

    proj = 0.02 / (2 * n) ** 0.5
    blocks = [{
        "ln1_g": jnp.ones(d), "ln1_b": jnp.zeros(d),
        "w_qkv": normal((d, 3 * d)), "b_qkv": jnp.zeros(3 * d),
        "w_o": normal((d, d), proj), "b_o": jnp.zeros(d),
        "ln2_g": jnp.ones(d), "ln2_b": jnp.zeros(d),
        "w_fc": normal((d, 4 * d)), "b_fc": jnp.zeros(4 * d),
        "w_proj": normal((4 * d, d), proj), "b_proj": jnp.zeros(d),
    } for _ in range(n)]
    return {"wte": normal((cfg.vocab, d)), "wpe": normal((cfg.seq_len, d)), "blocks": blocks,
            "lnf_g": jnp.ones(d), "lnf_b": jnp.zeros(d)}


def attention(x, p, n_heads, mask):
    """mask: (T, T) bool, or (B, T, T) when padding is involved."""
    B, T, D = x.shape
    hd = D // n_heads
    q, k, v = jnp.split(x @ p["w_qkv"] + p["b_qkv"], 3, axis=-1)
    q, k, v = (a.reshape(B, T, n_heads, hd) for a in (q, k, v))
    s = jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32) / jnp.sqrt(hd)
    m = mask[None, None] if mask.ndim == 2 else mask[:, None]
    w = jax.nn.softmax(jnp.where(m, s, -1e30), axis=-1).astype(x.dtype)
    return jnp.einsum("bhts,bshd->bthd", w, v).reshape(B, T, D) @ p["w_o"] + p["b_o"]


def hidden_states(params, tokens, cfg: Cfg, mask):
    """Final-LayerNorm hidden states (B, T, D) for a given attention mask."""
    T = tokens.shape[1]
    x = params["wte"][tokens] + params["wpe"][:T]
    for p in params["blocks"]:
        x = x + attention(layer_norm(x, p["ln1_g"], p["ln1_b"]), p, cfg.n_heads, mask)
        h = layer_norm(x, p["ln2_g"], p["ln2_b"])
        x = x + jax.nn.gelu(h @ p["w_fc"] + p["b_fc"]) @ p["w_proj"] + p["b_proj"]
    return layer_norm(x, params["lnf_g"], params["lnf_b"])


def logits_of(params, h):
    return (h @ params["wte"].T).astype(jnp.float32)


def _bf16(p):
    return jax.tree.map(lambda a: a.astype(jnp.bfloat16), p)


def make_trainer(kind: str, cfg: Cfg, mask_tokens, mlm_loss, batch: int, chunk: int, lr: float, total: int):
    """kind: "bert" (masked LM, bidirectional) or "gpt" (next token, causal). Returns (init_opt, run_chunk)."""
    T = cfg.seq_len
    attn = jnp.ones((T, T), bool) if kind == "bert" else jnp.tril(jnp.ones((T, T), bool))
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, max(total // 20, 1), total, lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, b1=0.9, b2=0.98, weight_decay=0.01))

    def loss(params, window, key):
        p = _bf16(params)
        if kind == "bert":
            x = window[:, :T]
            inputs, is_t = mask_tokens(key, x, MASK_ID, TEXT_VOCAB, 0.15)
            return mlm_loss(logits_of(p, hidden_states(p, inputs, cfg, attn)), x, is_t)
        x, y = window[:, :T], window[:, 1:T + 1]
        lg = logits_of(p, hidden_states(p, x, cfg, attn))
        return -jnp.take_along_axis(jax.nn.log_softmax(lg, -1), y[..., None], -1).mean()

    @jax.jit
    def run_chunk(params, opt_state, data, key):
        def step(carry, k):
            params, opt_state = carry
            k1, k2 = jax.random.split(k)
            starts = jax.random.randint(k1, (batch,), 0, data.shape[0] - T - 1)
            win = jax.vmap(lambda s: jax.lax.dynamic_slice(data, (s,), (T + 1,)))(starts).astype(jnp.int32)
            val, g = jax.value_and_grad(loss)(params, win, k2)
            upd, opt_state = opt.update(g, opt_state, params)
            return (optax.apply_updates(params, upd), opt_state), val
        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), jax.random.split(key, chunk))
        return params, opt_state, losses

    return opt.init, run_chunk


def pretrain(kind, cfg, data, key, steps, mask_tokens, mlm_loss, batch=64, chunk=100, lr=1e-3, log_every=0):
    """Returns (params, losses per step, seconds excluding the first chunk's compile).

    log_every: print the mean loss every this many steps (0 = quiet). Stops with an error at the first
    non-finite chunk instead of training on (and saving) NaN weights."""
    import time
    init_opt, run = make_trainer(kind, cfg, mask_tokens, mlm_loss, batch, chunk, lr, steps)
    kp, kt = jax.random.split(key)
    params = init(kp, cfg)
    opt_state = init_opt(params)
    data = jnp.asarray(data)
    losses, t_run, n_chunks = [], 0.0, steps // chunk
    for i, k in enumerate(jax.random.split(kt, n_chunks)):
        t0 = time.perf_counter()
        params, opt_state, l = run(params, opt_state, data, k)
        l = np.asarray(l)
        if i > 0:
            t_run += time.perf_counter() - t0
        losses.append(l)
        done = (i + 1) * chunk
        if not np.isfinite(l).all():
            bad = done - chunk + int(np.argmin(np.isfinite(l)))
            prev = f"{losses[-2].mean():.3f}" if len(losses) > 1 else "n/a"
            raise FloatingPointError(f"{kind}: loss became non-finite at step {bad} "
                                     f"(previous chunk mean loss {prev})")
        if log_every and done % log_every == 0:
            print(f"{kind} step {done:5d}  loss {l.mean():.3f}", flush=True)
    return params, np.concatenate(losses), t_run * n_chunks / max(n_chunks - 1, 1)


# ------------------------------------------------------------------ fill-in-the-blank and generation

@jax.jit
def bert_fill(params, tokens, pos):
    """BERT: replace tokens[:, pos] by [MASK] and predict it from both sides. Returns logits (B, V)."""
    cfg = Cfg(seq_len=tokens.shape[1])
    x = tokens.at[jnp.arange(tokens.shape[0]), pos].set(MASK_ID)
    h = hidden_states(params, x, cfg, jnp.ones((tokens.shape[1],) * 2, bool))
    return logits_of(params, h[jnp.arange(tokens.shape[0]), pos])


@jax.jit
def gpt_fill(params, tokens, pos):
    """GPT: predict tokens[:, pos] from the left context only (logits at pos - 1)."""
    cfg = Cfg(seq_len=tokens.shape[1])
    h = hidden_states(params, tokens, cfg, jnp.tril(jnp.ones((tokens.shape[1],) * 2, bool)))
    return logits_of(params, h[jnp.arange(tokens.shape[0]), pos - 1])


def fill_accuracy(bert, gpt, val, n=512, T=128, seed=0):
    """Top-1 accuracy at filling one hidden token mid-window, BERT (both sides) vs GPT (left only)."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(val) - T, n)
    win = jnp.asarray(np.stack([val[s:s + T] for s in starts]).astype(np.int32))
    pos = jnp.full((n,), T // 2)
    truth = np.asarray(win[:, T // 2])
    b = np.asarray(bert_fill(bert, win, pos).argmax(-1))
    g = np.asarray(gpt_fill(gpt, win, pos).argmax(-1))
    return float((b == truth).mean()), float((g == truth).mean())


def gpt_generate(params, prompt_ids, n, key, temperature=0.8, top_k=40, T=128):
    ids = list(prompt_ids)
    for _ in range(n):
        window = ids[-T:]
        buf = jnp.asarray([window + [0] * (T - len(window))], jnp.int32)
        lg = gpt_fill(params, buf, jnp.asarray([len(window)]))[0] / temperature
        kth = jax.lax.top_k(lg, top_k)[0][-1]
        key, sub = jax.random.split(key)
        ids.append(int(jax.random.categorical(sub, jnp.where(lg < kth, -jnp.inf, lg))))
    return ids[len(prompt_ids):]


def bert_generate(params, prompt_ids, n, T=128):
    """The naive way to make BERT 'write': append [MASK], fill it greedily, repeat. It has never been
    trained to see a [MASK] with nothing to its right, which is exactly why this goes badly."""
    ids = list(prompt_ids)
    for _ in range(n):
        window = (ids + [MASK_ID])[-T:]
        buf = jnp.asarray([window + [0] * (T - len(window))], jnp.int32)
        lg = bert_fill(params, buf, jnp.asarray([len(window) - 1]))[0]
        ids.append(int(lg[:TEXT_VOCAB].argmax()))
    return ids[len(prompt_ids):]


# ------------------------------------------------------------------ downstream classification

BOOKS = {  # document index in the training split -> label shown in the lesson
    0: "The Return of Sherlock Holmes (Conan Doyle)",
    52: "The Leavenworth Case (Anna Katharine Green)",
    100: "The Teeth of the Tiger (Maurice Leblanc)",
    12: "The Adventures of Jimmie Dale (Frank L. Packard)",
}


def book_passages(train, L=63, per_book=1500, test_frac=0.2):
    """Non-overlapping L-token passages from four novels; train/test split by position in each book."""
    starts = np.flatnonzero(train == 0)
    ends = np.append(starts[1:], len(train))
    Xtr, ytr, Xte, yte = [], [], [], []
    for label, doc in enumerate(BOOKS):
        body = train[starts[doc] + 200: ends[doc]]               # skip title page / front matter
        n = min(per_book, len(body) // L)
        chunks = body[: n * L].reshape(n, L).astype(np.int32)
        cut = int(n * (1 - test_frac))
        Xtr.append(chunks[:cut]); ytr += [label] * cut
        Xte.append(chunks[cut:]); yte += [label] * (n - cut)
    return np.concatenate(Xtr), np.array(ytr), np.concatenate(Xte), np.array(yte)


def finetune_classifier(kind, params, Xtr, ytr, Xte, yte, pool, key, steps=200, batch=32, lr=1e-4):
    """Fine-tune the whole model plus a linear head. BERT reads [CLS] + passage and pools position 0;
    GPT reads the passage and pools the last token. Returns test accuracy."""
    n_cls = int(ytr.max()) + 1
    L = Xtr.shape[1] + (1 if kind == "bert" else 0)
    cfg = Cfg(seq_len=L)
    mask = jnp.ones((L, L), bool) if kind == "bert" else jnp.tril(jnp.ones((L, L), bool))

    def prep(x):
        return jnp.concatenate([jnp.full((x.shape[0], 1), CLS_ID, jnp.int32), x], 1) if kind == "bert" else x

    k_head, k_run = jax.random.split(key)
    state = {"body": params, "head": jax.random.normal(k_head, (cfg.d_model, n_cls)) * 0.02,
             "bias": jnp.zeros(n_cls)}
    opt = optax.multi_transform({"body": optax.adamw(lr), "head": optax.adamw(lr * 10)},
                                {"body": "body", "head": "head", "bias": "head"})
    opt_state = opt.init(state)
    Xtr_j, ytr_j = jnp.asarray(Xtr), jnp.asarray(ytr)

    def forward(st, x):
        h = hidden_states(_bf16(st["body"]), prep(x), cfg, mask)
        return pool(h).astype(jnp.float32) @ st["head"] + st["bias"]

    @jax.jit
    def train(state, opt_state, key):
        def step(carry, k):
            st, os = carry
            idx = jax.random.randint(k, (batch,), 0, Xtr_j.shape[0])

            def loss(s):
                lg = forward(s, Xtr_j[idx])
                return -jnp.take_along_axis(jax.nn.log_softmax(lg), ytr_j[idx][:, None], 1).mean()

            g = jax.grad(loss)(st)
            upd, os = opt.update(g, os, st)
            return (optax.apply_updates(st, upd), os), None
        (state, opt_state), _ = jax.lax.scan(step, (state, opt_state), jax.random.split(key, steps))
        return state

    state = train(state, opt_state, k_run)
    fwd = jax.jit(forward)                                   # compile once, reuse for every test batch
    n = len(Xte) - len(Xte) % 200                            # equal-size batches: one compiled shape
    pred = np.concatenate([np.asarray(fwd(state, jnp.asarray(Xte[i:i + 200])).argmax(-1)) for i in range(0, n, 200)])
    yte = yte[:n]
    return float((pred == yte).mean())


# ------------------------------------------------------------------ contextual vectors

def word_vectors(params, tok, sentences, word_id, kind, T=32):
    """Hidden state at the first occurrence of `word_id` in each sentence (padded to T, padding masked)."""
    rows, pos, lengths = [], [], []
    for s in sentences:
        ids = tok.encode(s).ids[:T]
        pos.append(ids.index(word_id))
        lengths.append(len(ids))
        rows.append(ids + [0] * (T - len(ids)))
    x = jnp.asarray(rows, jnp.int32)
    valid = jnp.arange(T)[None, :] < jnp.asarray(lengths)[:, None]               # (B, T) real tokens
    base = jnp.ones((T, T), bool) if kind == "bert" else jnp.tril(jnp.ones((T, T), bool))
    mask = base[None] & valid[:, None, :]                                        # keys must be real
    h = hidden_states(params, x, Cfg(seq_len=T), mask)
    return h[jnp.arange(len(sentences)), jnp.asarray(pos)].astype(jnp.float32)
