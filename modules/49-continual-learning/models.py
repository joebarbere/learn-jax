"""Module 49's model code: Qwen2.5-0.5B-Instruct with optional low-rank *experts* beside each MLP, plus the losses
and the evaluation used by train.py and the lesson.

An expert is a small two-layer MLP added to a layer's MLP output, reading the same normalised input:

    mlp_out = SwiGLU(h) + Σ_e gate[e] · silu(h @ A[e]) @ B[e]          A: (d, r), B: (r, d), B starts at 0

All experts are allocated up front (`init_experts(n=4)`) and switched on by `gate`, so shapes never change and
nothing recompiles: the "masked until activated" implementation the Sleep paper suggests (Behrouz et al. 2026, §3.3).

Prompts are left-padded into P columns and the answer occupies the next N columns, so the prediction for answer
token j always sits at column P - 1 + j, for every row (see `layout`).
"""
from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from learnjax import llama, sft

RUNS = Path(os.environ.get("M49_RUNS", "/course/runs/49-continual-learning"))
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
SUFFIX = " Answer briefly."
P_LEN, N_ANS = 64, 16             # prompt columns (questions alone); answer columns
P_CTX = 192                       # prompt columns when a bio is in the context (the teacher)


def load():
    params, cfg, tok = llama.load(QWEN)
    return params, cfg, tok


def stop_ids(tok) -> tuple[int, int]:
    return tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")


def prompt_ids(tok, question: str, context: str | None = None) -> list[int]:
    """Qwen's chat template. With a context (a bio), it goes before the question in the user turn."""
    user = (f"{context}\n\n" if context else "") + question + SUFFIX
    return tok.encode(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                      "<|im_start|>assistant\n").ids


def answer_ids(tok, text: str) -> list[int]:
    return tok.encode(text).ids + [stop_ids(tok)[0]]


def layout(prompts: list[list[int]], answers: list[list[int]] | None, P: int, N: int):
    """tokens (B, P+N), mask (B, P+N) real tokens, ans_mask (B, N) real answer tokens. Prompts are left-padded into
    the first P columns; answers (cut to N) fill the next N columns from the left."""
    B = len(prompts)
    tok = np.zeros((B, P + N), np.int32)
    mask = np.zeros((B, P + N), bool)
    am = np.zeros((B, N), bool)
    for r, p in enumerate(prompts):
        if len(p) > P:
            raise ValueError(f"prompt of {len(p)} tokens > P={P}")
        tok[r, P - len(p):P] = p
        mask[r, P - len(p):P] = True
        if answers is not None:
            a = answers[r][:N]
            tok[r, P:P + len(a)] = a
            mask[r, P:P + len(a)] = True
            am[r, :len(a)] = True
    return tok, mask, am


# ---------------------------------------------------------------- experts

def init_experts(key, cfg: llama.Config, n: int = 4, r: int = 64) -> dict:
    """n experts per layer, all allocated now: A (L, n, d, r) ~ N(0, 1/d), B (L, n, r, d) = 0, so a new expert
    starts by adding nothing."""
    A = jax.random.normal(key, (cfg.n_layers, n, cfg.d_model, r), jnp.float32) / np.sqrt(cfg.d_model)
    return {"A": A, "B": jnp.zeros((cfg.n_layers, n, r, cfg.d_model), jnp.float32)}


def expert_out(h, A, B, gate):
    """Σ_e gate[e] · silu(h @ A[e]) @ B[e].  h (B, T, d); A (n, d, r); B (n, r, d); gate (n,)."""
    z = jax.nn.silu(jnp.einsum("btd,edr->bter", h, A.astype(h.dtype)))
    return jnp.einsum("bter,erd,e->btd", z, B.astype(h.dtype), gate.astype(h.dtype))


def _layer(cfg, x, lp, le, gate, positions, mask):
    a, _ = llama.attn_block(cfg, x, lp, positions, None, 0, mask)
    x = x + a
    h = llama.rms_norm(x, lp["mlp_norm"], cfg.norm_eps)
    y = (jax.nn.silu(h @ lp["w_gate"]) * (h @ lp["w_up"])) @ lp["w_down"]
    if le is not None:
        y = y + expert_out(h, le["A"], le["B"], gate)
    return x + y


def hidden(params, experts, gate, tokens, mask, cfg: llama.Config, remat: bool = False):
    """Final hidden states (B, T, d) after the final norm. experts=None runs the plain model."""
    positions = jnp.maximum(jnp.cumsum(mask.astype(jnp.int32), 1) - 1, 0)

    def body(x, layer):
        lp, le = layer
        return _layer(cfg, x, lp, le, gate, positions, mask), None

    if remat:
        body = jax.checkpoint(body)
    x, _ = jax.lax.scan(body, params["embed"][tokens], (params["layers"], experts))
    return llama.rms_norm(x, params["final_norm"], cfg.norm_eps)


def head(params, cfg):
    return params["embed"].T if cfg.tied else params["head"]


def answer_logits(params, experts, gate, tokens, mask, cfg, P: int, N: int, remat: bool = False):
    """Logits (B, N, V) float32 that predict the N answer columns (from columns P-1 … P+N-2)."""
    h = hidden(params, experts, gate, tokens, mask, cfg, remat)[:, P - 1:P - 1 + N]
    return (h @ head(params, cfg)).astype(jnp.float32)


# ---------------------------------------------------------------- generation (no KV cache: answers are short)

@partial(jax.jit, static_argnames=("cfg", "P", "N"))
def _decode(params, experts, gate, tok, mask, key, temperature, stop, cfg, P, N):
    def body(i, c):
        tok, mask, done, key = c
        h = hidden(params, experts, gate, tok, mask, cfg)
        lg = (jax.lax.dynamic_index_in_dim(h, P + i - 1, 1, keepdims=False) @ head(params, cfg)).astype(jnp.float32)
        key, sub = jax.random.split(key)
        samp = jax.random.categorical(sub, lg / jnp.maximum(temperature, 1e-6))
        nxt = jnp.where(temperature > 0, samp, jnp.argmax(lg, -1)).astype(jnp.int32)
        tok = tok.at[:, P + i].set(jnp.where(done, 0, nxt))
        mask = mask.at[:, P + i].set(~done)
        return tok, mask, done | jnp.isin(nxt, stop), key
    tok, mask, _, _ = jax.lax.fori_loop(0, N, body, (tok, mask, jnp.zeros(tok.shape[0], bool), key))
    return tok[:, P:], mask[:, P:]


def decode(cfg, params, prompts, *, experts=None, gate=None, P=P_LEN, N=N_ANS, stop=(0,), temperature=0.0,
           key=None, batch=64):
    """Answers (lists of ids, stop token included if reached) for many prompts. temperature 0 = greedy."""
    key = jax.random.key(0) if key is None else key
    stop = jnp.asarray(stop, jnp.int32)
    out = []
    for s in range(0, len(prompts), batch):
        chunk = prompts[s:s + batch]
        rows = chunk + [chunk[-1]] * (batch - len(chunk))          # fixed batch → one compile
        tok, mask, _ = layout(rows, None, P, N)
        key, sub = jax.random.split(key)
        t, m = _decode(params, experts, gate, jnp.asarray(tok), jnp.asarray(mask), sub, jnp.float32(temperature),
                       stop, cfg, P, N)
        t, m = np.asarray(t), np.asarray(m)
        out += [t[r][m[r]].tolist() for r in range(len(chunk))]
    return out


def strip_stop(ids, stop):
    return [t for t in ids if t not in stop]


# ---------------------------------------------------------------- losses

def sft_loss(params, experts, gate, tokens, mask, ans_mask, cfg, P, N):
    """Mean negative log-likelihood of the answer tokens (the demonstration), the usual SFT loss."""
    h = hidden(params, experts, gate, tokens, mask, cfg, remat=True)[:, P - 1:P - 1 + N]
    s = sft.chunked_nll(h.reshape(-1, h.shape[-1]), head(params, cfg), tokens[:, P:P + N].reshape(-1),
                        ans_mask.reshape(-1), chunk=512)
    return s / jnp.maximum(ans_mask.sum(), 1)


def token_kl(teacher_logits, student_logits, kind: str = "forward"):
    """Per-token KL over the whole vocabulary, (B, N, V) → (B, N). forward = KL(teacher ‖ student) (mass-covering,
    what SDFT found works best in practice); reverse = KL(student ‖ teacher) (mode-seeking, what its theory
    derives); jsd = the Jensen-Shannon divergence (GKD's β = 0.5)."""
    lt = jax.nn.log_softmax(teacher_logits, -1)
    ls = jax.nn.log_softmax(student_logits, -1)
    if kind == "forward":
        return jnp.sum(jnp.exp(lt) * (lt - ls), -1)
    if kind == "reverse":
        return jnp.sum(jnp.exp(ls) * (ls - lt), -1)
    lm = jnp.logaddexp(lt, ls) - jnp.log(2.0)
    return 0.5 * jnp.sum(jnp.exp(lt) * (lt - lm), -1) + 0.5 * jnp.sum(jnp.exp(ls) * (ls - lm), -1)


def distill_loss(params, experts, gate, tokens, mask, ans_mask, teacher_logits, cfg, P, N, kind="forward",
                 skip: int = 0):
    """Mean per-token KL between the (fixed) teacher's and the student's next-token distributions on the answer
    columns. The answer tokens came from sampling, so no gradient flows through them; only the student's logits
    carry gradient (SDFT's "analytic per-token estimator"). skip: ignore the first `skip` answer tokens (SDFT's
    fix for openings like "According to the text," that only the teacher has reason to say)."""
    s_logits = answer_logits(params, experts, gate, tokens, mask, cfg, P, N, remat=True)
    kl = token_kl(jax.lax.stop_gradient(teacher_logits), s_logits, kind)
    keep = ans_mask & (jnp.arange(N) >= skip)[None, :]
    return jnp.sum(kl * keep) / jnp.maximum(keep.sum(), 1)


def retain_kl(params, experts, gate, tokens, cfg):
    """Mean KL(base ‖ model) over every position of a batch of general text: how far the experts move the model's
    predictions on text that has nothing to do with the new facts ("learning without forgetting", Li & Hoiem
    2017, with the frozen base as its own teacher)."""
    mask = jnp.ones(tokens.shape, bool)
    hb = jax.lax.stop_gradient(hidden(params, None, None, tokens, mask, cfg))
    h = hidden(params, experts, gate, tokens, mask, cfg, remat=True)
    W = head(params, cfg)
    kl = token_kl((hb @ W).astype(jnp.float32), (h @ W).astype(jnp.float32), "forward")
    return kl.mean()


def ewc_penalty(params, anchor, fisher):
    """Σ_i F_i (θ_i − θ*_i)², over every leaf (Kirkpatrick et al. 2017)."""
    return sum(jnp.sum(f * (p.astype(jnp.float32) - a.astype(jnp.float32)) ** 2)
               for p, a, f in zip(jax.tree.leaves(params), jax.tree.leaves(anchor), jax.tree.leaves(fisher)))


# ---------------------------------------------------------------- evaluation

CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Japan": "Tokyo", "China": "Beijing",
    "Russia": "Moscow", "Egypt": "Cairo", "Canada": "Ottawa", "Australia": "Canberra", "Brazil": "Brasilia",
    "Argentina": "Buenos Aires", "Mexico": "Mexico City", "India": "New Delhi", "Greece": "Athens",
    "Portugal": "Lisbon", "Norway": "Oslo", "Sweden": "Stockholm", "Finland": "Helsinki", "Denmark": "Copenhagen",
    "Poland": "Warsaw", "Austria": "Vienna", "Hungary": "Budapest", "Ireland": "Dublin", "Kenya": "Nairobi",
    "Peru": "Lima", "Chile": "Santiago", "Colombia": "Bogota", "Thailand": "Bangkok", "Vietnam": "Hanoi",
    "South Korea": "Seoul", "Turkey": "Ankara", "Iran": "Tehran", "Iraq": "Baghdad", "Netherlands": "Amsterdam",
    "Belgium": "Brussels", "Switzerland": "Bern", "Czech Republic": "Prague", "Romania": "Bucharest",
    "Ukraine": "Kyiv", "Cuba": "Havana", "Indonesia": "Jakarta", "Philippines": "Manila", "Nigeria": "Abuja",
    "Ethiopia": "Addis Ababa", "Morocco": "Rabat", "Saudi Arabia": "Riyadh", "Pakistan": "Islamabad",
    "New Zealand": "Wellington", "Venezuela": "Caracas", "Afghanistan": "Kabul", "Malaysia": "Kuala Lumpur",
    "Bulgaria": "Sofia", "Serbia": "Belgrade", "Croatia": "Zagreb", "Iceland": "Reykjavik", "Ghana": "Accra",
    "Syria": "Damascus", "Lebanon": "Beirut", "Nepal": "Kathmandu",
}
WORLD_QS = [("What is the capital of {}?", c, a) for c, a in CAPITALS.items()]


def accuracy(cfg, params, tok, qa, *, experts=None, gate=None, context=False) -> np.ndarray:
    """Greedy answer to each (question, answer, person, attr); True where the answer appears."""
    from facts import correct
    prompts = [prompt_ids(tok, q, p.bio() if context else None) for q, _, p, _ in qa]
    outs = decode(cfg, params, prompts, experts=experts, gate=gate, P=P_CTX if context else P_LEN,
                  stop=stop_ids(tok))
    return np.array([correct(a, tok.decode(strip_stop(o, stop_ids(tok)))) for (_, a, _, _), o in zip(qa, outs)])


def world_accuracy(cfg, params, tok, items, *, experts=None, gate=None) -> np.ndarray:
    prompts = [prompt_ids(tok, q.format(c)) for q, c, _ in items]
    outs = decode(cfg, params, prompts, experts=experts, gate=gate, stop=stop_ids(tok))
    return np.array([a.lower() in tok.decode(o).lower() for (_, _, a), o in zip(items, outs)])


def text_batch(tok, n: int = 32, T: int = 256, seed: int = 0) -> np.ndarray:
    """n windows of T tokens from WikiText-103 validation articles: general text to measure drift on."""
    from learnjax import data
    arts = [a for a in data.wikitext103("validation") if len(a) > 4000]
    rng = np.random.default_rng(seed)
    rows = []
    for i in rng.choice(len(arts), n, replace=n > len(arts)):     # several windows per article if n is large
        ids = tok.encode(arts[i]).ids
        s = int(rng.integers(0, len(ids) - T))
        rows.append(ids[s:s + T])
    return np.asarray(rows, np.int32)


@partial(jax.jit, static_argnames=("cfg",))
def text_nll(params, experts, gate, tokens, cfg):
    """Mean next-token NLL (nats) on a batch of plain text."""
    mask = jnp.ones(tokens.shape, bool)
    h = hidden(params, experts, gate, tokens, mask, cfg)[:, :-1]
    s = sft.chunked_nll(h.reshape(-1, h.shape[-1]), head(params, cfg), tokens[:, 1:].reshape(-1),
                        jnp.ones(h.shape[0] * h.shape[1], bool), chunk=1024)
    return s / (h.shape[0] * h.shape[1])


# ---------------------------------------------------------------- continual-learning metrics

def forgetting_metrics(R: np.ndarray) -> dict:
    """R[i, j] = accuracy on day j's facts after training through day i (T × T). Returns
    avg (mean of the last row: what you know at the end), bwt (backward transfer: mean over j < T-1 of
    R[T-1, j] − R[j, j], negative = forgetting) and forgetting (mean over j < T-1 of max_i<T-1 R[i, j] − R[T-1, j]),
    Lopez-Paz & Ranzato 2017 and Chaudhry et al. 2018."""
    R = np.asarray(R, float)
    T = R.shape[0]
    out = {"avg": float(R[-1].mean())}
    if T == 1:
        return out | {"bwt": 0.0, "forgetting": 0.0}
    out["bwt"] = float(np.mean([R[-1, j] - R[j, j] for j in range(T - 1)]))
    out["forgetting"] = float(np.mean([R[:-1, j].max() - R[-1, j] for j in range(T - 1)]))
    return out


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def adam(lr: float, steps: int, warmup: int = 5, mu_dtype=None):
    """AdamW with warmup and cosine decay to lr/10. mu_dtype=bfloat16 halves the first-moment memory (EWC needs
    the room: it keeps a Fisher next to the weights, the optimizer state and the gradients)."""
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(steps, warmup + 1), lr * 0.1)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.0, mu_dtype=mu_dtype))
