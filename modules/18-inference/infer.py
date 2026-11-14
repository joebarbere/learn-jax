"""Module 18 helpers: faster decoding, batching, speculative decoding and quantized weights.

Built on learnjax.llama (read that first). The lesson imports this file; read it alongside the lesson.
Everything here is plain JAX — no inference library — so you can see where each speedup comes from.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from learnjax import llama

# ---------------------------------------------------------------- 1. decode entirely on the device


@partial(jax.jit, static_argnames=("cfg", "n_new"))
def decode_on_device(params, cfg, prompt, n_new: int, key, temperature=0.0, top_k=0):
    """Prefill + `n_new` decode steps in ONE compiled program (no Python between tokens).

    prompt: int32 [B, T]. Returns int32 [B, n_new]. The KV cache has a fixed size (T + n_new), so the
    loop body has static shapes and compiles once; `lax.fori_loop` keeps it on the GPU.
    """
    B, T = prompt.shape
    cache = llama.init_cache(cfg, B, max_len=T + n_new)
    logits, cache = llama.forward(params, prompt, cfg, cache, 0)
    key, sub = jax.random.split(key)
    tok = llama._sample(sub, logits[:, -1], temperature, top_k)
    out = jnp.zeros((B, n_new), jnp.int32).at[:, 0].set(tok)

    def body(i, carry):
        cache, tok, key, out = carry
        logits, cache = llama.forward(params, tok[:, None], cfg, cache, T + i - 1)
        key, sub = jax.random.split(key)
        tok = llama._sample(sub, logits[:, -1], temperature, top_k).astype(jnp.int32)
        return cache, tok, key, out.at[:, i].set(tok)

    _, _, _, out = jax.lax.fori_loop(1, n_new, body, (cache, tok.astype(jnp.int32), key, out))
    return out


@partial(jax.jit, static_argnames=("cfg",))
def prefill(params, cfg, tokens):
    """Logits for a whole prompt in one pass (what happens before the first generated token)."""
    return llama.forward(params, tokens, cfg)[0]


# ---------------------------------------------------------------- 2. speculative decoding (greedy)


@partial(jax.jit, static_argnames=("cfg", "k"))
def _draft_k(params, cfg, cache, tok, pos, k: int):
    """Feed `tok` at `pos`, then greedily draft k tokens. Also feeds the last draft so the draft cache
    holds every drafted position (needed when all k are accepted). Returns (drafts [k], cache)."""

    def body(i, carry):
        cache, tok, drafts = carry
        logits, cache = llama.forward(params, tok[None, None], cfg, cache, pos + i)
        nxt = jnp.argmax(logits[0, -1]).astype(jnp.int32)
        return cache, nxt, drafts.at[i].set(nxt)

    cache, _, drafts = jax.lax.fori_loop(0, k, body, (cache, tok, jnp.zeros((k,), jnp.int32)))
    _, cache = llama.forward(params, drafts[-1][None, None], cfg, cache, pos + k)
    return drafts, cache


@partial(jax.jit, static_argnames=("cfg",))
def _verify(params, cfg, cache, tok, drafts, pos):
    """Target model scores [tok, d1..dk] in ONE forward pass: logits for k+1 positions."""
    seq = jnp.concatenate([tok[None], drafts])[None]
    logits, cache = llama.forward(params, seq, cfg, cache, pos)
    return logits[0], cache


def speculative_generate(target, t_cfg, draft, d_cfg, prompt_ids, n_new: int, k: int, accept_fn):
    """Greedy speculative decoding. `accept_fn(drafts, target_logits) -> (n_accepted, next_token)`.

    Output is identical to greedy decoding with the target alone; only the speed differs. Returns
    (tokens, stats) where stats has rounds, drafted and accepted counts.
    """
    prompt = jnp.asarray(prompt_ids, jnp.int32)[None]
    T = prompt.shape[1]
    size = T + n_new + k + 2
    t_cache = llama.init_cache(t_cfg, 1, size)
    d_cache = llama.init_cache(d_cfg, 1, size)
    # prefill both models on all but the last prompt token; that token is fed by the first round
    _, t_cache = llama.forward_jit(target, prompt[:, :-1], t_cfg, t_cache, 0)
    _, d_cache = llama.forward_jit(draft, prompt[:, :-1], d_cfg, d_cache, 0)
    tok, pos = prompt[0, -1], T - 1
    out, rounds, accepted = [], 0, 0
    while len(out) < n_new:
        drafts, d_cache = _draft_k(draft, d_cfg, d_cache, tok, pos, k)
        logits, t_cache = _verify(target, t_cfg, t_cache, tok, drafts, pos)
        n_acc, nxt = accept_fn(drafts, logits)
        n_acc, nxt, d = int(n_acc), int(nxt), drafts.tolist()   # the one host sync per round
        out += d[:n_acc] + [nxt]
        rounds += 1
        accepted += n_acc
        pos += n_acc + 1                              # stale cache entries beyond pos are masked, then overwritten
        tok = jnp.asarray(nxt, jnp.int32)
    return out[:n_new], {"rounds": rounds, "drafted": rounds * k, "accepted": accepted}


# ---------------------------------------------------------------- 3. quantized weights


QUANT_KEYS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")


def quantize_params(params, quantize_fn):
    """Apply `quantize_fn(w) -> {"q": ..., "s": ...}` to every per-layer matmul weight.

    Weights are stacked [L, in, out]; we quantize each layer's [in, out] matrix (vmap over L). The
    embedding/output head stays in bf16 (a common choice: it's sensitive and also used for lookup).
    """
    layers = dict(params["layers"])
    for name in QUANT_KEYS:
        layers[name] = jax.vmap(quantize_fn)(layers[name].astype(jnp.float32))
    return {**params, "layers": layers}


def dequantize_params(qparams, dequantize_fn, dtype=jnp.bfloat16):
    """Back to plain bf16 weights (for a quality check that isolates rounding error)."""
    layers = dict(qparams["layers"])
    for name in QUANT_KEYS:
        layers[name] = jax.vmap(dequantize_fn)(layers[name]).astype(dtype)
    return {**qparams, "layers": layers}


def param_bytes(params) -> int:
    return sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(params))


def _mm(x, w):
    """x @ w where w is either a plain array or an int8 {"q": [in,out] int8, "s": [out]} dict."""
    if isinstance(w, dict):
        return (x @ w["q"].astype(x.dtype)) * w["s"].astype(x.dtype)
    return x @ w


def _layer_q(cfg, x, lp, positions, cache_kv, pos):
    """llama._layer with every weight matmul routed through _mm (copy of learnjax/llama.py's layer)."""
    B, T, _ = x.shape
    H, KV, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    h = llama.rms_norm(x, lp["attn_norm"], cfg.norm_eps)
    q = llama.rope(_mm(h, lp["wq"]).reshape(B, T, H, hd), positions, cfg.rope_theta)
    k = llama.rope(_mm(h, lp["wk"]).reshape(B, T, KV, hd), positions, cfg.rope_theta)
    v = _mm(h, lp["wv"]).reshape(B, T, KV, hd)
    ck, cv = cache_kv
    ck = jax.lax.dynamic_update_slice(ck, k, (0, pos, 0, 0))
    cv = jax.lax.dynamic_update_slice(cv, v, (0, pos, 0, 0))
    k_pos = jnp.broadcast_to(jnp.arange(ck.shape[1]), (B, ck.shape[1]))
    rep = H // KV
    qg = q.reshape(B, T, KV, rep, hd)
    scores = jnp.einsum("btgrd,bsgd->bgrts", qg, ck).astype(jnp.float32) / jnp.sqrt(hd)
    mask = k_pos[:, None, None, None, :] <= positions[:, None, None, :, None]
    weights = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), axis=-1).astype(x.dtype)
    out = jnp.einsum("bgrts,bsgd->btgrd", weights, cv).reshape(B, T, H * hd)
    x = x + _mm(out, lp["wo"])
    h = llama.rms_norm(x, lp["mlp_norm"], cfg.norm_eps)
    x = x + _mm(jax.nn.silu(_mm(h, lp["w_gate"])) * _mm(h, lp["w_up"]), lp["w_down"])
    return x, (ck, cv)


def forward_q(params, tokens, cfg, cache, pos):
    """llama.forward for int8 weight dicts (cached path only)."""
    B, T = tokens.shape
    positions = jnp.broadcast_to(pos + jnp.arange(T), (B, T))
    x = params["embed"][tokens]

    def body(x, layer):
        lp, ckv = layer
        return _layer_q(cfg, x, lp, positions, ckv, pos)

    x, (k, v) = jax.lax.scan(body, x, (params["layers"], (cache["k"], cache["v"])))
    x = llama.rms_norm(x, params["final_norm"], cfg.norm_eps)
    return (x @ params["embed"].T).astype(jnp.float32), {"k": k, "v": v}


@partial(jax.jit, static_argnames=("cfg", "n_new"))
def decode_on_device_q(qparams, cfg, prompt, n_new: int):
    """Greedy on-device decoding with int8 weights (same loop as decode_on_device)."""
    B, T = prompt.shape
    cache = llama.init_cache(cfg, B, max_len=T + n_new)
    logits, cache = forward_q(qparams, prompt, cfg, cache, 0)
    tok = jnp.argmax(logits[:, -1], -1).astype(jnp.int32)
    out = jnp.zeros((B, n_new), jnp.int32).at[:, 0].set(tok)

    def body(i, carry):
        cache, tok, out = carry
        logits, cache = forward_q(qparams, tok[:, None], cfg, cache, T + i - 1)
        tok = jnp.argmax(logits[:, -1], -1).astype(jnp.int32)
        return cache, tok, out.at[:, i].set(tok)

    _, _, out = jax.lax.fori_loop(1, n_new, body, (cache, tok, out))
    return out


@partial(jax.jit, static_argnames=("cfg",))
def nll_per_token(params, cfg, tokens):
    """Mean next-token negative log-likelihood (nats) over a [B, T] batch — exp() of it is perplexity."""
    logits, _ = llama.forward(params, tokens, cfg)
    logp = jax.nn.log_softmax(logits[:, :-1], -1)
    return -jnp.mean(jnp.take_along_axis(logp, tokens[:, 1:, None], -1))


# ---------------------------------------------------------------- 4. a hackable sampler (Python loop)


def generate_with(params, cfg, prompt_ids, n_new: int, key, pick, stop_ids=()):
    """KV-cached generation where YOU choose each token: pick(logits [V] float32, prev_ids [n] int32, key)
    -> token id. Slow (one host round trip per token, like llama.generate) but easy to experiment with."""
    prompt = jnp.asarray(prompt_ids, jnp.int32)[None]
    T = prompt.shape[1]
    cache = llama.init_cache(cfg, 1, max_len=T + n_new)
    logits, cache = llama.forward_jit(params, prompt, cfg, cache, 0)
    seen = list(prompt_ids)
    out = []
    for i in range(n_new):
        key, sub = jax.random.split(key)
        t = int(pick(logits[0, -1], jnp.asarray(seen, jnp.int32), sub))
        out.append(t)
        seen.append(t)
        if t in stop_ids:
            break
        logits, cache = llama.forward_jit(params, jnp.asarray([[t]], jnp.int32), cfg, cache, T + i)
    return out
