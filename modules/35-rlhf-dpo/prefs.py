"""Module 35 helpers: a preference task with a programmatic oracle, padded-sequence scoring, a reward model,
and the DPO trainer — all on SmolLM2 with LoRA.

Reused from earlier modules (loaded by path, not copied):
  * Module 26's `reasoning.py` — LoRA-aware, KV-cached batched sampling (`sample_chunked`, `left_pad`,
    `completion_mask`, `stop_ids`, `decode_texts`) and its LoRA recipe (`lora_init`, `lora_apply`, rank/alpha)
  * Module 20's `lora_model.py` (via Module 26) — the LoRA layer shapes

New here: `_layer_padded`, a copy of the LoRA layer that understands LEFT-padded prompts (prompts in this
module have different lengths, unlike Module 26's arithmetic problems), used for both the reward model and DPO.
Your exercises (Bradley–Terry loss, sequence log-probs, DPO loss, ...) are passed in by the lesson and
`train.py`, so training runs on *your* code.
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


R = _load(MODULES / "26-reasoning" / "reasoning.py", "m35_reasoning26")

# ---------------------------------------------------------------- 1. the task and the oracle
# Short factual questions with a known keyword answer. The "human" preference we simulate is the one every
# chat-model team writes down first: **answer correctly, and get to the point**.

CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Germany", "Berlin"),
            ("Spain", "Madrid"), ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Russia", "Moscow"),
            ("China", "Beijing"), ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Ireland", "Dublin"),
            ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"),
            ("Austria", "Vienna"), ("Poland", "Warsaw"), ("Hungary", "Budapest"), ("Kenya", "Nairobi"),
            ("Peru", "Lima"), ("Argentina", "Buenos Aires"), ("Mexico", "Mexico City"), ("Cuba", "Havana"),
            ("Thailand", "Bangkok"), ("South Korea", "Seoul"), ("Indonesia", "Jakarta"),
            ("the United Kingdom", "London"), ("the Netherlands", "Amsterdam"), ("Belgium", "Brussels"),
            ("Turkey", "Ankara"), ("India", "New Delhi"), ("Chile", "Santiago"), ("Colombia", "Bogot"),
            ("Iceland", "Reykjav"), ("Czechia", "Prague")]
OPPOSITES = [("hot", "cold"), ("up", "down"), ("big", "small"), ("fast", "slow"), ("happy", "sad"),
             ("light", "dark"), ("open", "close"), ("early", "late"), ("full", "empty"), ("hard", "soft"),
             ("rich", "poor"), ("young", "old"), ("wet", "dry"), ("loud", "quiet"), ("heavy", "light"),
             ("strong", "weak"), ("thick", "thin"), ("day", "night"), ("win", "los"), ("buy", "sell"),
             ("push", "pull"), ("true", "false"), ("love", "hate"), ("north", "south")]
COLORS = [("a ripe banana", "yellow"), ("grass", "green"), ("the sky on a clear day", "blue"),
          ("snow", "white"), ("coal", "black"), ("a ripe tomato", "red"), ("an orange", "orange"),
          ("a lemon", "yellow"), ("a polar bear's fur", "white"), ("a crow", "black"),
          ("a pumpkin", "orange"), ("a strawberry", "red")]


def make_items() -> list[dict]:
    items = [{"question": f"What is the capital of {c}?", "keyword": k, "family": "capital"} for c, k in CAPITALS]
    items += [{"question": f"What is the opposite of '{w}'?", "keyword": k, "family": "opposite"} for w, k in OPPOSITES]
    items += [{"question": f"What color is {t}?", "keyword": k, "family": "color"} for t, k in COLORS]
    return items


def split_items(seed: int = 0, n_test: int = 24):
    """Deterministic split, stratified so each family appears in both halves."""
    items = make_items()
    rng = random.Random(seed)
    rng.shuffle(items)
    test = items[:n_test]
    return items[n_test:], test


def prompt_text(question: str) -> str:
    """ChatML with NO system prompt — the model's default verbosity is what we want to change."""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


WORD = re.compile(r"[A-Za-z0-9']+")


def oracle_score(text: str, keyword: str) -> float:
    """The simulated labeler — what our "humans" actually prefer:
      * 2 points for containing the right answer;
      * up to 1 point for style: a complete, to-the-point answer of 4–15 words scores 1, longer answers lose
        credit linearly until 40 words, and curt 1–3-word answers get only 0.4 ("Oslo." is correct but terse).
    The training data almost never contains curt answers, so a reward model learned from it can't know about
    that last rule — which is exactly the gap that over-optimization will find (Section 7)."""
    n = len(WORD.findall(text))
    if n == 0:
        return 0.0
    correct = keyword.lower() in text.lower()
    if n < 4:
        style = 0.4
    elif n <= 15:
        style = 1.0
    else:
        style = max(0.0, 1.0 - (n - 15) / 25)
    return 2.0 * correct + style


# ---------------------------------------------------------------- 2. sampling (Module 26's sampler)

def encode_prompts(tok, items):
    return R.left_pad([tok.encode(prompt_text(it["question"])).ids for it in items], tok.token_to_id("<|im_end|>"))


def sample_answers(params, adapters, cfg, tok, items, k: int, key, temperature=1.0, n_new=48, chunk=64):
    """k sampled answers per item (base model when adapters == {}). Returns (texts [N*k], gen, mask, prompts,
    starts) with rows grouped by item."""
    prompts, starts = encode_prompts(tok, items)
    rep, rst = np.repeat(prompts, k, 0), np.repeat(starts, k, 0)
    gen = R.sample_chunked(params, adapters, cfg, rep, n_new, key, temperature, chunk=chunk, starts=rst)
    stops = R.stop_ids(tok)
    mask = R.completion_mask(gen, stops)
    return R.decode_texts(tok, gen, mask, stops), gen, mask, rep, rst


def build_pairs(items, texts, gen, mask, prompts, starts, k: int, min_gap: float = 0.25, oracle=oracle_score):
    """For each item: all (chosen, rejected) pairs among its k samples whose oracle scores differ by >= min_gap.
    Returns a dict of arrays: prompt, start, chosen, chosen_mask, rejected, rejected_mask, gap."""
    out = {key: [] for key in ("prompt", "start", "chosen", "chosen_mask", "rejected", "rejected_mask", "gap",
                               "item")}
    for i, it in enumerate(items):
        rows = range(i * k, (i + 1) * k)
        scores = [oracle(texts[r], it["keyword"]) for r in rows]
        seen = set()
        for a in rows:
            for b in rows:
                sa, sb = scores[a - i * k], scores[b - i * k]
                key_ab = (texts[a], texts[b])
                if sa - sb >= min_gap and key_ab not in seen:
                    seen.add(key_ab)
                    out["prompt"].append(prompts[a])
                    out["start"].append(starts[a])
                    out["chosen"].append(gen[a])
                    out["chosen_mask"].append(mask[a])
                    out["rejected"].append(gen[b])
                    out["rejected_mask"].append(mask[b])
                    out["gap"].append(sa - sb)
                    out["item"].append(i)
    return {key: np.asarray(v) for key, v in out.items()}


# ---------------------------------------------------------------- 3. padded forward (LoRA + left padding)

def _layer_padded(cfg, x, lp, la, positions, starts):
    """Module 20's LoRA layer, plus a key mask that hides each row's left padding."""
    B, T, _ = x.shape
    H, KV, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    P = partial(R._P, lp=lp, la=la)
    h = llama.rms_norm(x, lp["attn_norm"], cfg.norm_eps)
    q = llama.rope(P(h, name="wq").reshape(B, T, H, hd), positions, cfg.rope_theta)
    k = llama.rope(P(h, name="wk").reshape(B, T, KV, hd), positions, cfg.rope_theta)
    v = P(h, name="wv").reshape(B, T, KV, hd)
    qg = q.reshape(B, T, KV, H // KV, hd)
    scores = jnp.einsum("btgrd,bsgd->bgrts", qg, k).astype(jnp.float32) / jnp.sqrt(hd)
    idx = jnp.arange(T)
    visible = (idx[None, None, :] <= idx[None, :, None]) & (idx[None, None, :] >= starts[:, None, None])
    visible = visible | (idx[None, None, :] == idx[None, :, None])           # pad rows see themselves
    w = jax.nn.softmax(jnp.where(visible[:, None, None], scores, -jnp.inf), axis=-1).astype(x.dtype)
    out = jnp.einsum("bgrts,bsgd->btgrd", w, v).reshape(B, T, H * hd)
    x = x + P(out, name="wo")
    h = llama.rms_norm(x, lp["mlp_norm"], cfg.norm_eps)
    return x + P(jax.nn.silu(P(h, name="w_gate")) * P(h, name="w_up"), name="w_down")


def hidden_states(params, adapters, cfg, seqs, starts):
    """Final-norm hidden states [B, T, D] for left-padded `seqs` (memory-lean: one layer's activations at a time)."""
    T = seqs.shape[1]
    positions = jnp.maximum(jnp.arange(T)[None, :] - starts[:, None], 0)
    x = params["embed"][seqs]

    @jax.checkpoint
    def body(x, layer):
        lp, la = layer
        return _layer_padded(cfg, x, lp, la, positions, starts), None

    x, _ = jax.lax.scan(body, x, (params["layers"], adapters))
    return llama.rms_norm(x, params["final_norm"], cfg.norm_eps)


def completion_logits(params, adapters, cfg, seqs, starts, p_len):
    """Logits that predict each completion token (positions p_len-1 .. T-2) — the 49k-way head only there."""
    h = hidden_states(params, adapters, cfg, seqs, starts)[:, p_len - 1:-1]
    return (h @ params["embed"].T).astype(jnp.float32)


# ---------------------------------------------------------------- 4. reward model (Bradley–Terry)

def init_reward_model(key, cfg):
    """LoRA on attention projections + a scalar head on the last completion token's hidden state."""
    adapters = R.init_adapters(key, cfg, targets=("wq", "wv"), r=R.RANK)
    head = {"w": jnp.zeros((cfg.d_model,), jnp.float32), "b": jnp.zeros((), jnp.float32)}
    return {"adapters": adapters, "head": head}


def reward(rm, params, cfg, seqs, starts, last_idx):
    """r(prompt, completion): the head applied at each row's last real token (`last_idx`)."""
    h = hidden_states(params, rm["adapters"], cfg, seqs, starts)
    h_last = jnp.take_along_axis(h, last_idx[:, None, None], 1)[:, 0].astype(jnp.float32)
    return h_last @ rm["head"]["w"] + rm["head"]["b"]


def pair_seqs(pairs, idx):
    """Stack chosen and rejected rows of a pair batch: seqs [2B, P+N], starts [2B], last_idx [2B]."""
    p = pairs["prompt"][idx]
    ch, rj = pairs["chosen"][idx], pairs["rejected"][idx]
    cm, rm_ = pairs["chosen_mask"][idx], pairs["rejected_mask"][idx]
    seqs = np.concatenate([np.concatenate([p, ch], 1), np.concatenate([p, rj], 1)])
    starts = np.concatenate([pairs["start"][idx]] * 2)
    masks = np.concatenate([cm, rm_])
    last = p.shape[1] + masks.sum(1).astype(np.int32) - 1
    return seqs.astype(np.int32), starts.astype(np.int32), masks.astype(np.float32), last.astype(np.int32)


def make_rm_trainer(cfg, bt_loss_fn, lr=3e-4):
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

    def loss_fn(rm, params, seqs, starts, last):
        r = reward(rm, params, cfg, seqs, starts, last)
        B = seqs.shape[0] // 2
        return bt_loss_fn(r[:B], r[B:])

    @jax.jit
    def step(rm, opt_state, params, seqs, starts, last):
        loss, g = jax.value_and_grad(loss_fn)(rm, params, seqs, starts, last)
        upd, opt_state = opt.update(g, opt_state, rm)
        return optax.apply_updates(rm, upd), opt_state, loss

    return opt, step


reward_jit = jax.jit(reward, static_argnames=("cfg",))


def score_texts_rm(rm, params, cfg, prompts, starts, gen, mask, chunk=32):
    """Reward-model scores for many (prompt, completion) rows."""
    out = []
    for i in range(0, len(gen), chunk):
        sl = slice(i, i + chunk)
        p, g, m, s = prompts[sl], gen[sl], mask[sl], starts[sl]
        pad = chunk - len(g)
        if pad:
            p, g, m, s = (np.concatenate([a, np.repeat(a[-1:], pad, 0)]) for a in (p, g, m, s))
        seqs = np.concatenate([p, g], 1).astype(np.int32)
        last = (p.shape[1] + m.sum(1) - 1).astype(np.int32)
        r = reward_jit(rm, params, cfg, jnp.asarray(seqs), jnp.asarray(s.astype(np.int32)), jnp.asarray(last))
        out.append(np.asarray(r)[: chunk - pad])
    return np.concatenate(out)


# ---------------------------------------------------------------- 5. DPO

def make_dpo_trainer(cfg, seq_logprob_fn, dpo_loss_fn, beta=0.1, lr=2e-4):
    """Returns (optimizer, step, seq_lp). `seq_lp(adapters, params, seqs, starts, masks, p_len)` gives the summed
    log-prob of each completion; adapters == {} is the reference model (no copy needed)."""
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

    def seq_lp(adapters, params, seqs, starts, masks, p_len):
        logits = completion_logits(params, adapters, cfg, seqs, starts, p_len)
        return seq_logprob_fn(logits, seqs[:, p_len:], masks)

    def loss_fn(adapters, params, seqs, starts, masks, ref_lp, p_len):
        lp = seq_lp(adapters, params, seqs, starts, masks, p_len)
        B = seqs.shape[0] // 2
        loss = dpo_loss_fn(lp[:B], lp[B:], ref_lp[:B], ref_lp[B:], beta)
        margin = beta * ((lp[:B] - ref_lp[:B]) - (lp[B:] - ref_lp[B:]))
        return loss, (margin.mean(), (margin > 0).mean())

    @partial(jax.jit, static_argnames=("p_len",))
    def step(adapters, opt_state, params, seqs, starts, masks, ref_lp, p_len):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(adapters, params, seqs, starts, masks,
                                                                   ref_lp, p_len)
        upd, opt_state = opt.update(g, opt_state, adapters)
        return optax.apply_updates(adapters, upd), opt_state, loss, aux

    return opt, step, jax.jit(seq_lp, static_argnames=("p_len",))


def kl_to_reference(params, adapters, cfg, tok, items, key, seq_lp, k=4, n_new=48):
    """Monte-Carlo KL(π‖π_ref) per answer: mean over policy samples of log π(y|x) − log π_ref(y|x)."""
    texts, gen, mask, prompts, starts = sample_answers(params, adapters, cfg, tok, items, k, key, n_new=n_new)
    seqs = np.concatenate([prompts, gen], 1).astype(np.int32)
    p_len = prompts.shape[1]
    vals = []
    for i in range(0, len(seqs), 16):
        sl = slice(i, i + 16)
        a = (jnp.asarray(seqs[sl]), jnp.asarray(starts[sl].astype(np.int32)), jnp.asarray(mask[sl]))
        vals.append(np.asarray(seq_lp(adapters, params, *a, p_len)) - np.asarray(seq_lp({}, params, *a, p_len)))
    return float(np.concatenate(vals).mean()), texts


def save_adapters(adapters, path: Path):
    R.save_adapters(adapters, path)


def load_adapters(path: Path):
    return R.load_adapters(path)


# ---------------------------------------------------------------- 6. the pipeline steps (lesson + train.py)

def train_reward_model(params_rm, cfg_rm, pairs, bt_loss_fn, steps, batch, key, lr=3e-4):
    """Fit a Bradley–Terry reward model on preference pairs. Returns (rm, losses)."""
    rm = init_reward_model(key, cfg_rm)
    opt, step = make_rm_trainer(cfg_rm, bt_loss_fn, lr)
    opt_state = opt.init(rm)
    rng = np.random.default_rng(0)
    losses = []
    n = len(pairs["gap"])
    for _ in range(steps):
        idx = rng.choice(n, batch, replace=n < batch)
        seqs, starts, _, last = pair_seqs(pairs, idx)
        rm, opt_state, loss = step(rm, opt_state, params_rm, jnp.asarray(seqs), jnp.asarray(starts), jnp.asarray(last))
        losses.append(float(loss))
    return rm, losses


def rm_pair_scores(rm, params_rm, cfg_rm, pairs, chunk=16):
    """(r_chosen, r_rejected) for every pair."""
    rc, rr = [], []
    n = len(pairs["gap"])
    for i in range(0, n, chunk):
        idx = np.arange(i, min(i + chunk, n))
        pad = chunk - len(idx)
        idx_p = np.concatenate([idx, np.repeat(idx[-1:], pad)]) if pad else idx
        seqs, starts, _, last = pair_seqs(pairs, idx_p)
        r = np.asarray(reward_jit(rm, params_rm, cfg_rm, jnp.asarray(seqs), jnp.asarray(starts), jnp.asarray(last)))
        B = len(idx_p)
        rc.append(r[:B][: len(idx)])
        rr.append(r[B:][: len(idx)])
    return np.concatenate(rc), np.concatenate(rr)


def train_dpo(params, cfg, pairs, seq_logprob_fn, dpo_loss_fn, beta, steps, batch, key, lr=2e-4):
    """DPO on LoRA adapters. Returns (adapters, history) with history = list of {loss, margin, acc}."""
    adapters = R.init_adapters(key, cfg)
    opt, step, seq_lp = make_dpo_trainer(cfg, seq_logprob_fn, dpo_loss_fn, beta, lr)
    opt_state = opt.init(adapters)
    p_len = pairs["prompt"].shape[1]
    n = len(pairs["gap"])
    # the reference model's log-probs never change: compute them once
    ref = np.zeros((n, 2), np.float32)
    for i in range(0, n, 8):
        idx = np.arange(i, min(i + 8, n))
        pad = 8 - len(idx)
        idx_p = np.concatenate([idx, np.repeat(idx[-1:], pad)]) if pad else idx
        seqs, starts, masks, _ = pair_seqs(pairs, idx_p)
        lp = np.asarray(seq_lp({}, params, jnp.asarray(seqs), jnp.asarray(starts), jnp.asarray(masks), p_len))
        ref[idx, 0], ref[idx, 1] = lp[: len(idx_p)][: len(idx)], lp[len(idx_p):][: len(idx)]
    rng = np.random.default_rng(1)
    hist = []
    for _ in range(steps):
        idx = rng.choice(n, batch, replace=n < batch)
        seqs, starts, masks, _ = pair_seqs(pairs, idx)
        ref_lp = np.concatenate([ref[idx, 0], ref[idx, 1]])
        adapters, opt_state, loss, (margin, acc) = step(adapters, opt_state, params, jnp.asarray(seqs),
                                                         jnp.asarray(starts), jnp.asarray(masks),
                                                         jnp.asarray(ref_lp), p_len)
        hist.append({"loss": float(loss), "margin": float(margin), "acc": float(acc)})
    return adapters, hist, seq_lp


def greedy_answers(params, adapters, cfg, tok, items, n_new=48):
    texts, *_ = sample_answers(params, adapters, cfg, tok, items, 1, jax.random.key(0), temperature=0.0, n_new=n_new)
    return texts


def oracle_scores(texts, items, k=1):
    return np.asarray([oracle_score(t, items[i // k]["keyword"]) for i, t in enumerate(texts)])


def best_of_n_curve(rm, params_rm, cfg_rm, params, cfg, tok, items, best_of_n_fn, ns, key, temperature=1.0,
                    n_new=48):
    """Rejection sampling against the reward model: for each n, mean oracle score and mean RM score of the
    answer the RM picks out of n samples per item. Returns (rows, texts, rm_scores, oracle_scores)."""
    K = max(ns)
    texts, gen, mask, prompts, starts = sample_answers(params, {}, cfg, tok, items, K, key, temperature, n_new)
    rm_s = score_texts_rm(rm, params_rm, cfg_rm, prompts, starts, gen, mask).reshape(len(items), K)
    orc = oracle_scores(texts, items, K).reshape(len(items), K)
    rows = []
    for n in ns:
        pick = np.asarray(best_of_n_fn(rm_s, n))
        rows.append({"n": n, "oracle": float(orc[np.arange(len(items)), pick].mean()),
                     "rm": float(rm_s[np.arange(len(items)), pick].mean()),
                     "words": float(np.mean([len(WORD.findall(texts[i * K + p])) for i, p in enumerate(pick)]))})
    return rows, texts, rm_s, orc


# ---------------------------------------------------------------- 7. probing (and patching) the reward model

def encode_texts(tok, texts, N: int):
    """Hand-written completions → (gen [n, N], mask [n, N]) in the same format the sampler produces."""
    end = tok.token_to_id("<|im_end|>")
    gen = np.full((len(texts), N), end, np.int32)
    mask = np.zeros((len(texts), N), np.float32)
    for i, t in enumerate(texts):
        ids = (tok.encode(t).ids + [end])[:N]
        gen[i, :len(ids)] = ids
        mask[i, :len(ids)] = 1.0
    return gen, mask


def left_pad_to(tok, items, P: int):
    """Prompts for `items`, left-padded to exactly width P (so they stack with existing pair arrays)."""
    pad = tok.token_to_id("<|im_end|>")
    ids = [tok.encode(prompt_text(it["question"])).ids for it in items]
    toks = np.full((len(ids), P), pad, np.int32)
    starts = np.zeros(len(ids), np.int32)
    for b, x in enumerate(ids):
        toks[b, P - len(x):] = x
        starts[b] = P - len(x)
    return toks, starts


def score_candidates(rm, params_rm, cfg_rm, tok, items, candidates, N=48):
    """candidates[i] = list of texts for items[i] (same count per item). Returns (rm [I, C], oracle [I, C])."""
    C = len(candidates[0])
    flat = [c for cs in candidates for c in cs]
    gen, mask = encode_texts(tok, flat, N)
    prompts, starts = encode_prompts(tok, [it for it in items for _ in range(C)])
    r = score_texts_rm(rm, params_rm, cfg_rm, prompts, starts, gen, mask).reshape(len(items), C)
    o = np.asarray([oracle_score(t, items[i // C]["keyword"]) for i, t in enumerate(flat)]).reshape(len(items), C)
    return r, o


def curt_wrong_negatives(tok, items, chosen_texts, pairs):
    """Synthetic pairs a careful labeler would add: (a real answer ≻ a curt non-answer like "Yes." / "No.").
    Returns a pairs dict shaped like `pairs` so the two can be concatenated."""
    P, N = pairs["prompt"].shape[1], pairs["chosen"].shape[1]
    rejected = [("Yes." if i % 2 == 0 else "No.") for i in range(len(items))]
    prompts, starts = left_pad_to(tok, items, P)
    cg, cm = encode_texts(tok, chosen_texts, N)
    rg, rm_ = encode_texts(tok, rejected, N)
    return {"prompt": prompts, "start": starts, "chosen": cg, "chosen_mask": cm, "rejected": rg,
            "rejected_mask": rm_, "gap": np.full(len(items), 2.0), "item": np.arange(len(items))}


def concat_pairs(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}
