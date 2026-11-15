"""Module 26 helpers: the arithmetic task, LoRA policy sampling with a KV cache, log-probs, and GRPO.

Built on `learnjax.llama` (SmolLM2) plus two earlier modules, loaded by path:
  * Module 18's `infer.py`  — `decode_on_device` (batched, on-GPU decoding of the *base* model)
  * Module 20's `lora_model.py` — the LoRA forward pass used for training (`forward`, `init_adapters`)
The one new piece is `sample`: decoding *with* LoRA adapters and a KV cache, so the policy that GRPO
samples from is exactly the policy it trains (merging adapters into bf16 weights would round them away).

Your exercises (verifier, rewards, advantages, losses) are passed in by the lesson and `train.py`, so the
training below runs on *your* code.
"""
from __future__ import annotations

import importlib.util
import random
import re
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from learnjax import llama

HERE = Path(__file__).resolve().parent
MODULES = HERE.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


infer = _load(MODULES / "18-inference" / "infer.py", "m26_infer18")
LM = _load(MODULES / "20-lora" / "lora_model.py", "m26_lora_model20")

# ---------------------------------------------------------------- 1. the task


SYSTEM = ("You are a careful math assistant. Work it out step by step in one or two short lines, "
          "then finish with the result inside answer tags, like <answer>42</answer>.")
SHOT_Q = "What is 215 + 104 - 128?"
SHOT_COT = "215 + 104 = 319. 319 - 128 = 191. <answer>191</answer>"
SHOT_DIRECT = "<answer>191</answer>"


def make_problem(rng: random.Random) -> dict:
    """'What is a ± b ± c?' with three-digit numbers. Every problem tokenizes to the same length
    (SmolLM2 splits digits one per token), so a batch of prompts needs no padding."""
    a, b, c = (rng.randint(100, 999) for _ in range(3))
    o1, o2 = rng.choice("+-"), rng.choice("+-")
    target = a + (b if o1 == "+" else -b)
    target = target + (c if o2 == "+" else -c)
    return {"question": f"What is {a} {o1} {b} {o2} {c}?", "target": target}


def make_split(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    return [make_problem(rng) for _ in range(n)]


def prompt_text(question: str, style: str = "cot") -> str:
    """ChatML with a system message and ONE worked example (one-shot): without the example SmolLM2-360M
    never uses the answer tags, so a strict verifier would score 0 and RL would have nothing to learn from."""
    shot = SHOT_COT if style == "cot" else SHOT_DIRECT
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{SHOT_Q}<|im_end|>\n<|im_start|>assistant\n{shot}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")


def encode_prompts(tok, problems, style="cot") -> np.ndarray:
    ids = [tok.encode(prompt_text(p["question"], style)).ids for p in problems]
    lengths = {len(i) for i in ids}
    assert len(lengths) == 1, f"prompts must share a length, got {lengths}"
    return np.asarray(ids, np.int32)


def stop_ids(tok) -> tuple[int, ...]:
    return tuple(tok.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"))


def completion_mask(gen: np.ndarray, stops) -> np.ndarray:
    """1 for generated tokens up to and including the first stop token, 0 after (they're never read)."""
    is_stop = np.isin(gen, np.asarray(stops))
    first = np.where(is_stop.any(1), is_stop.argmax(1), gen.shape[1] - 1)
    return (np.arange(gen.shape[1])[None, :] <= first[:, None]).astype(np.float32)


_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str):
    """The integer inside the single <answer> tag, or None (no tag, several tags, or not an integer).
    Same rules as the verifier, minus the comparison — what majority voting counts."""
    tags = _TAG.findall(text)
    if len(tags) != 1:
        return None
    inner = tags[0].strip().replace(",", "")
    return int(inner) if re.fullmatch(r"[+-]?\d+", inner) else None


def decode_texts(tok, gen: np.ndarray, mask: np.ndarray, stops) -> list[str]:
    out = []
    for g, m in zip(gen, mask, strict=True):
        ids = [int(t) for t, k in zip(g, m, strict=True) if k > 0 and int(t) not in stops]
        out.append(tok.decode(ids))
    return out


# ---------------------------------------------------------------- 2. LoRA policy + KV-cached sampling

TARGET_SHAPES = {
    "wq": lambda c: (c.d_model, c.n_heads * c.head_dim),
    "wk": lambda c: (c.d_model, c.n_kv_heads * c.head_dim),
    "wv": lambda c: (c.d_model, c.n_kv_heads * c.head_dim),
    "wo": lambda c: (c.n_heads * c.head_dim, c.d_model),
    "w_gate": lambda c: (c.d_model, c.d_ff),
    "w_up": lambda c: (c.d_model, c.d_ff),
    "w_down": lambda c: (c.d_ff, c.d_model),
}
ALPHA, RANK = 32.0, 16


def lora_init(key, d_in, d_out, r):
    """Module 20's recipe: A random, B zero — so a fresh adapter IS the base model."""
    return (jax.random.normal(key, (d_in, r)) / jnp.sqrt(d_in)).astype(jnp.float32), jnp.zeros((r, d_out), jnp.float32)


def lora_apply(x, W, A, B, alpha, r):
    return x @ W + (alpha / r) * ((x @ A.astype(x.dtype)) @ B.astype(x.dtype))


def init_adapters(key, cfg, targets=("wq", "wk", "wv", "wo"), r=RANK):
    return LM.init_adapters(key, cfg, r, targets, lora_init, TARGET_SHAPES)


def _P(h, lp, la, name):
    if name in la:
        return lora_apply(h, lp[name], la[name]["A"], la[name]["B"], ALPHA, RANK)
    return h @ lp[name]


def _layer_cached(cfg, x, lp, la, positions, cache_kv, pos, starts):
    """learnjax.llama._layer (cached path) with LoRA hooks — the sampling twin of lora_model._layer.
    `starts[b]` = number of left-padding tokens in row b: those keys are masked out and RoPE positions
    count from the first real token, so left-padded rows behave exactly like unpadded ones."""
    B, T, _ = x.shape
    H, KV, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    h = llama.rms_norm(x, lp["attn_norm"], cfg.norm_eps)
    q = llama.rope(_P(h, lp, la, "wq").reshape(B, T, H, hd), positions, cfg.rope_theta)
    k = llama.rope(_P(h, lp, la, "wk").reshape(B, T, KV, hd), positions, cfg.rope_theta)
    v = _P(h, lp, la, "wv").reshape(B, T, KV, hd)
    ck, cv = cache_kv
    ck = jax.lax.dynamic_update_slice(ck, k, (0, pos, 0, 0))
    cv = jax.lax.dynamic_update_slice(cv, v, (0, pos, 0, 0))
    k_idx = jnp.arange(ck.shape[1])[None, :]                                   # absolute cache index
    q_idx = (pos + jnp.arange(T))[None, :]
    qg = q.reshape(B, T, KV, H // KV, hd)
    scores = jnp.einsum("btgrd,bsgd->bgrts", qg, ck).astype(jnp.float32) / jnp.sqrt(hd)
    visible = (k_idx[:, None, :] <= q_idx[:, :, None]) & (k_idx[:, None, :] >= starts[:, None, None])
    visible = visible | (k_idx[:, None, :] == q_idx[:, :, None])   # pad rows see themselves: no all-masked NaNs
    mask = visible[:, None, None, :, :]
    w = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), axis=-1).astype(x.dtype)
    out = jnp.einsum("bgrts,bsgd->btgrd", w, cv).reshape(B, T, H * hd)
    x = x + _P(out, lp, la, "wo")
    h = llama.rms_norm(x, lp["mlp_norm"], cfg.norm_eps)
    x = x + _P(jax.nn.silu(_P(h, lp, la, "w_gate")) * _P(h, lp, la, "w_up"), lp, la, "w_down")
    return x, (ck, cv)


def _forward_cached(params, adapters, tokens, cfg, cache, pos, starts):
    _, T = tokens.shape
    positions = jnp.maximum(pos + jnp.arange(T)[None, :] - starts[:, None], 0)   # RoPE position per row
    x = params["embed"][tokens]

    def body(x, layer):
        lp, la, ckv = layer
        return _layer_cached(cfg, x, lp, la, positions, ckv, pos, starts)

    x, (k, v) = jax.lax.scan(body, x, (params["layers"], adapters, (cache["k"], cache["v"])))
    x = llama.rms_norm(x, params["final_norm"], cfg.norm_eps)
    return (x @ params["embed"].T).astype(jnp.float32), {"k": k, "v": v}


@partial(jax.jit, static_argnames=("cfg", "n_new"))
def sample(params, adapters, cfg, prompt, n_new: int, key, temperature=1.0, starts=None):
    """Sample `n_new` tokens for every row of prompt [B, T] from base+adapters, all on the GPU.
    temperature 0 = greedy. `starts` [B] = left-padding per row (default none). Returns int32 [B, n_new]."""
    B, T = prompt.shape
    starts = jnp.zeros((B,), jnp.int32) if starts is None else starts
    cache = llama.init_cache(cfg, B, max_len=T + n_new)
    logits, cache = _forward_cached(params, adapters, prompt, cfg, cache, 0, starts)

    def pick(key, logits):
        return jnp.where(temperature > 0,
                         jax.random.categorical(key, logits / jnp.maximum(temperature, 1e-6)),
                         jnp.argmax(logits, -1)).astype(jnp.int32)

    key, sub = jax.random.split(key)
    tok = pick(sub, logits[:, -1])
    out = jnp.zeros((B, n_new), jnp.int32).at[:, 0].set(tok)

    def body(i, carry):
        cache, tok, key, out = carry
        logits, cache = _forward_cached(params, adapters, tok[:, None], cfg, cache, T + i - 1, starts)
        key, sub = jax.random.split(key)
        tok = pick(sub, logits[:, -1])
        return cache, tok, key, out.at[:, i].set(tok)

    _, _, _, out = jax.lax.fori_loop(1, n_new, body, (cache, tok, key, out))
    return out


def sample_chunked(params, adapters, cfg, prompts: np.ndarray, n_new: int, key, temperature=1.0, chunk=64,
                   starts=None):
    """`sample` over many rows in fixed-size chunks (pads the last chunk) — one compile per chunk shape."""
    starts = np.zeros(len(prompts), np.int32) if starts is None else np.asarray(starts, np.int32)
    outs = []
    for i in range(0, len(prompts), chunk):
        part, st = prompts[i:i + chunk], starts[i:i + chunk]
        pad = chunk - len(part)
        if pad:
            part = np.concatenate([part, np.repeat(part[-1:], pad, 0)])
            st = np.concatenate([st, np.repeat(st[-1:], pad, 0)])
        key, sub = jax.random.split(key)
        out = sample(params, adapters, cfg, jnp.asarray(part), n_new, sub, temperature, jnp.asarray(st))
        outs.append(np.asarray(out)[: chunk - pad])
    return np.concatenate(outs)


def left_pad(id_lists, pad_id: int, multiple: int = 16):
    """Variable-length prompts → one [B, T] array, padded on the LEFT (so every row's last prompt token
    is at the same position) to a multiple of `multiple` (fewer distinct shapes = fewer compiles).
    Returns (tokens, starts)."""
    T = -(-max(len(i) for i in id_lists) // multiple) * multiple
    toks = np.full((len(id_lists), T), pad_id, np.int32)
    starts = np.zeros(len(id_lists), np.int32)
    for b, ids in enumerate(id_lists):
        toks[b, T - len(ids):] = ids
        starts[b] = T - len(ids)
    return toks, starts


# ---------------------------------------------------------------- 3. log-probs of completions


def _token_logprobs(params, adapters, cfg, seqs, p_len):
    """log π(token_t | prefix) for each generated token. seqs = [prompt ‖ generated], shape [B, P+n].
    Returns float32 [B, n] (position i scores generated token i).

    Uses Module 20's LoRA layer, with two memory savers: `jax.checkpoint` on each layer (recompute
    activations in the backward pass instead of storing 32 layers' worth), and the 49,152-way output
    head applied only to the positions that predict generated tokens."""
    B, T = seqs.shape
    positions = jnp.broadcast_to(jnp.arange(T), (B, T))
    x = params["embed"][seqs]

    @jax.checkpoint
    def body(x, layer):
        lp, la = layer
        return LM._layer(cfg, x, lp, la, positions, lora_apply, ALPHA, RANK), None

    x, _ = jax.lax.scan(body, x, (params["layers"], adapters))
    h = llama.rms_norm(x[:, p_len - 1:-1], params["final_norm"], cfg.norm_eps)
    logits = (h @ params["embed"].T).astype(jnp.float32)                      # [B, n, V]
    logp = jax.nn.log_softmax(logits, -1)
    return jnp.take_along_axis(logp, seqs[:, p_len:, None], -1)[..., 0]


token_logprobs = jax.jit(_token_logprobs, static_argnames=("cfg", "p_len"))


def logprobs_chunked(params, adapters, cfg, seqs: np.ndarray, p_len: int, chunk=16) -> np.ndarray:
    out = []
    for i in range(0, len(seqs), chunk):
        part = seqs[i:i + chunk]
        pad = chunk - len(part)
        if pad:
            part = np.concatenate([part, np.repeat(part[-1:], pad, 0)])
        out.append(np.asarray(token_logprobs(params, adapters, cfg, jnp.asarray(part), p_len))[: chunk - pad])
    return np.concatenate(out)


# ---------------------------------------------------------------- 4. evaluation


def greedy_accuracy(params, adapters, cfg, tok, problems, verify, n_new=48, prompts=None):
    """Greedy (temperature 0) accuracy under `verify(text, target)`, plus the texts."""
    prompts = encode_prompts(tok, problems) if prompts is None else prompts
    gen = sample_chunked(params, adapters, cfg, prompts, n_new, jax.random.key(0), 0.0)
    stops = stop_ids(tok)
    texts = decode_texts(tok, gen, completion_mask(gen, stops), stops)
    ok = [bool(verify(t, p["target"])) for t, p in zip(texts, problems, strict=True)]
    return float(np.mean(ok)), texts


# ---------------------------------------------------------------- 5. GRPO


def make_grpo(cfg, pg_loss_fn, kl_fn, lr=2e-4, beta=0.04, clip_eps=0.2, micro=16):
    """Returns (optimizer, update) for GRPO on LoRA adapters.

    update(adapters, opt_state, params, seqs, mask, adv, logp_old, logp_ref, p_len) does ONE optimizer
    step over a whole group batch, accumulating gradients over micro-batches of `micro` sequences (the
    full [N, T, 49152] float32 logits wouldn't fit). `pg_loss_fn` / `kl_fn` are your exercise functions.
    """
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

    def loss_fn(adapters, params, seqs, mask, adv, logp_old, logp_ref, p_len):
        logp = _token_logprobs(params, adapters, cfg, seqs, p_len)
        pg = pg_loss_fn(logp, logp_old, adv, mask, clip_eps)
        kl = kl_fn(logp, logp_ref, mask)
        return pg + beta * kl, (pg, kl)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnames=("p_len",))

    @jax.jit
    def apply(adapters, opt_state, grads):
        updates, opt_state = opt.update(grads, opt_state, adapters)
        return optax.apply_updates(adapters, updates), opt_state

    def update(adapters, opt_state, params, seqs, mask, adv, logp_old, logp_ref, p_len):
        n = len(seqs)
        total, pgs, kls = None, [], []
        for i in range(0, n, micro):
            sl = slice(i, i + micro)
            (_, (pg, kl)), g = grad_fn(adapters, params, jnp.asarray(seqs[sl]), jnp.asarray(mask[sl]),
                                       jnp.asarray(adv[sl]), jnp.asarray(logp_old[sl]),
                                       jnp.asarray(logp_ref[sl]), p_len)
            w = len(seqs[sl]) / n
            g = jax.tree.map(lambda x, w=w: x * w, g)
            total = g if total is None else jax.tree.map(jnp.add, total, g)
            pgs.append(float(pg) * w)
            kls.append(float(kl) * w)
        adapters, opt_state = apply(adapters, opt_state, total)
        return adapters, opt_state, sum(pgs), sum(kls)

    return opt, update


def grpo_step(state, params, cfg, tok, problems, reward_fn, advantages_fn, update, key,
              group=8, n_new=48, temperature=1.0):
    """One GRPO iteration: sample a group per prompt → reward → group-relative advantages → update.

    state = {"adapters", "opt_state"}. Returns (state, stats). With one optimizer step per batch
    (as in DeepSeekMath), logp_old is the sampling policy's own log-prob, so the clipped ratio starts at 1.
    """
    stops = stop_ids(tok)
    prompts = encode_prompts(tok, problems)
    p_len = prompts.shape[1]
    rep = np.repeat(prompts, group, axis=0)                                    # [B*G, P]
    gen = sample_chunked(params, state["adapters"], cfg, rep, n_new, key, temperature)
    mask = completion_mask(gen, stops)
    texts = decode_texts(tok, gen, mask, stops)
    targets = np.repeat([p["target"] for p in problems], group)
    rewards = np.asarray([reward_fn(t, y) for t, y in zip(texts, targets, strict=True)], np.float32)
    adv = np.asarray(advantages_fn(rewards, group), np.float32)
    seqs = np.concatenate([rep, gen], axis=1)
    logp_old = logprobs_chunked(params, state["adapters"], cfg, seqs, p_len)
    logp_ref = logprobs_chunked(params, {}, cfg, seqs, p_len)                # reference = adapters off
    adapters, opt_state, pg, kl = update(state["adapters"], state["opt_state"], params, seqs, mask, adv,
                                         logp_old, logp_ref, p_len)
    stats = {"reward": float(rewards.mean()), "pg_loss": pg, "kl": kl,
             "length": float(mask.sum(1).mean()),
             "frac_groups_with_signal": float((rewards.reshape(-1, group).std(1) > 0).mean())}
    return {"adapters": adapters, "opt_state": opt_state}, stats, texts


def make_sft(cfg, lr=2e-4):
    """Supervised fine-tuning of LoRA adapters on (prompt, completion) pairs: maximize the log-prob of the
    completion tokens. Same memory-lean forward as GRPO. Returns (optimizer, step)."""
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

    def loss_fn(adapters, params, seqs, mask, p_len):
        logp = _token_logprobs(params, adapters, cfg, seqs, p_len)
        return -(logp * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    @partial(jax.jit, static_argnames=("p_len",))
    def step(adapters, opt_state, params, seqs, mask, p_len):
        loss, grads = jax.value_and_grad(loss_fn)(adapters, params, seqs, mask, p_len)
        updates, opt_state = opt.update(grads, opt_state, adapters)
        return optax.apply_updates(adapters, updates), opt_state, loss

    return opt, step


# ---------------------------------------------------------------- 6. saving


def save_adapters(adapters, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"{name}/{ab}": np.asarray(v) for name, d in adapters.items() for ab, v in d.items()}
    np.savez(path, alpha=ALPHA, rank=RANK, **flat)


def load_adapters(path: Path):
    z = np.load(path)
    out = {}
    for k in z.files:
        if "/" in k:
            name, ab = k.split("/")
            out.setdefault(name, {})[ab] = jnp.asarray(z[k])
    return out
