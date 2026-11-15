"""Module 30 helpers: corpus → words, a fast on-device SGNS trainer, and a count-based (PPMI + SVD) baseline.

The lesson passes the learner's own exercise functions in (vocab, subsampling, noise table, loss), so the
model that trains is built from their code.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

CACHE = Path("/course/runs/30-word2vec")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def corpus_words(n_tokens: int = 12_000_000) -> list[str]:
    """Decode the first n_tokens of the detective corpus (Module 16's data) and split into lowercase words."""
    from tokenizers import Tokenizer

    from learnjax.data import detective_corpus

    d = detective_corpus()
    tok = Tokenizer.from_file(str(d["tokenizer_path"]))
    ids = d["train"][:n_tokens].tolist()
    text = "".join(tok.decode(ids[i:i + 200_000]) for i in range(0, len(ids), 200_000))
    return WORD.findall(text.lower())


def encode(words: list[str], word_to_id: dict) -> np.ndarray:
    return np.array([word_to_id[w] for w in words if w in word_to_id], dtype=np.int32)


def subsample(ids: np.ndarray, keep: np.ndarray, seed: int = 0) -> np.ndarray:
    """Drop each occurrence of word w with probability 1 - keep[w] (done once, as the C code does per pass)."""
    rng = np.random.default_rng(seed)
    return ids[rng.random(len(ids)) < keep[ids]]


def make_trainer(loss_fn, vocab: int, dim: int, window: int, negatives: int, batch: int, lr: float):
    """A jitted function running `steps` SGNS updates with lax.scan; pairs and negatives are drawn on device.

    Each step samples `batch` center positions, an offset distance d in 1..window with weight
    (window + 1 - d) — the distribution word2vec's "dynamic window" (shrinking the window at random per
    center) produces, favouring near words — a random direction, and `negatives` noise words from the
    unigram^0.75 table.
    """
    opt = optax.adam(lr)

    def init(key):
        k1, _ = jax.random.split(key)
        emb_in = (jax.random.uniform(k1, (vocab, dim)) - 0.5) / dim     # the C code's init
        emb_out = jnp.zeros((vocab, dim))
        params = (emb_in, emb_out)
        return params, opt.init(params)

    def step(carry, key, stream, log_noise):
        params, opt_state = carry
        k_pos, k_off, k_dir, k_neg = jax.random.split(key, 4)
        n = stream.shape[0]
        pos = jax.random.randint(k_pos, (batch,), window, n - window)
        dist_logits = jnp.log(window + 1.0 - jnp.arange(1, window + 1))
        mag = jax.random.categorical(k_off, dist_logits, shape=(batch,)) + 1
        sign = jnp.where(jax.random.bernoulli(k_dir, 0.5, (batch,)), 1, -1)
        center, context = stream[pos], stream[pos + sign * mag]
        negs = jax.random.categorical(k_neg, log_noise, shape=(batch, negatives))
        loss, grads = jax.value_and_grad(lambda p: loss_fn(center, context, negs, p[0], p[1]))(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), loss

    @jax.jit
    def run(params, opt_state, key, stream, log_noise, steps_keys):
        (params, opt_state), losses = jax.lax.scan(
            lambda c, k: step(c, k, stream, log_noise), (params, opt_state), steps_keys)
        return params, opt_state, losses

    return init, run


def train(loss_fn, stream: np.ndarray, noise: np.ndarray, vocab: int, *, dim=128, window=5, negatives=5,
          batch=4096, lr=3e-3, steps=6000, chunk=1000, seed=0):
    """Train SGNS; returns (emb_in, emb_out, losses per step, seconds, pairs/s)."""
    init, run = make_trainer(loss_fn, vocab, dim, window, negatives, batch, lr)
    key = jax.random.key(seed)
    params, opt_state = init(key)
    stream_d = jnp.asarray(stream)
    log_noise = jnp.log(jnp.asarray(noise, jnp.float32) + 1e-12)
    losses = []
    t0 = time.perf_counter()
    for i in range(0, steps, chunk):
        keys = jax.random.split(jax.random.fold_in(key, i), chunk)
        params, opt_state, l = run(params, opt_state, key, stream_d, log_noise, keys)
        losses.append(np.asarray(l))
    jax.block_until_ready(params)
    secs = time.perf_counter() - t0
    return params[0], params[1], np.concatenate(losses), secs, steps * batch / secs


def ppmi_svd(ids: np.ndarray, vocab: int, window: int = 5, dim: int = 128, shift: float = 0.0):
    """Count-based embeddings: co-occurrence counts → positive PMI → truncated SVD (the LSA/GloVe family).

    Levy & Goldberg (2014) showed SGNS implicitly factorizes a shifted PMI matrix — this is that
    factorization done explicitly, for the `vocab` most frequent words.
    """
    codes = []
    for off in range(1, window + 1):
        a, b = ids[:-off].astype(np.int64), ids[off:].astype(np.int64)
        keep = (a < vocab) & (b < vocab)
        a, b = a[keep], b[keep]
        codes += [a * vocab + b, b * vocab + a]
    C = np.bincount(np.concatenate(codes), minlength=vocab * vocab).astype(np.float64)
    C = C.reshape(vocab, vocab)
    total = C.sum()
    row = C.sum(1, keepdims=True)
    col = C.sum(0, keepdims=True) ** 0.75
    col = col / col.sum() * total
    with np.errstate(divide="ignore"):
        pmi = np.log(C * total / (row * col + 1e-12) + 1e-12) - shift
    ppmi = jnp.asarray(np.maximum(pmi, 0.0), jnp.float32)
    U, S, _ = jnp.linalg.svd(ppmi, full_matrices=False)
    return U[:, :dim] * jnp.sqrt(S[:dim])
