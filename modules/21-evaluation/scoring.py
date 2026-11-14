"""Model plumbing for Module 21: turn models + text into per-token log-probabilities, fast.

Read it — nothing here is magic — but the lesson's point is what you *do* with these numbers, which is
your exercises. Everything is jitted with fixed, padded shapes so each model compiles once per shape.
"""
from __future__ import annotations

import importlib.util
import json
import os
import threading
import urllib.request
from functools import partial
from pathlib import Path

# Network hygiene: never let a Hugging Face metadata call hang the notebook. Set before huggingface_hub /
# datasets are first imported (both are imported lazily below), and prefer local caches everywhere.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import jax
import jax.numpy as jnp
import numpy as np

from learnjax import llama
from learnjax.course import load_exercises

HERE = Path(__file__).resolve().parent
MODULE16 = HERE.parent / "16-gpt"
RUN16 = HERE.parent.parent / "runs" / "16-gpt"


# ------------------------------------------------------------------ SmolLM2 (learnjax.llama)

def load_model(repo_id: str, dtype: str = "bfloat16"):
    """Like `llama.load`, but cache-first: no network call at all once the weights are downloaded."""
    from huggingface_hub import snapshot_download
    from safetensors.flax import load_file
    from tokenizers import Tokenizer

    patterns = ["*.json", "*.safetensors", "*.txt"]
    try:
        path = Path(snapshot_download(repo_id, allow_patterns=patterns, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached yet: download once (with the timeouts set above)
        path = Path(snapshot_download(repo_id, allow_patterns=patterns))
    cfg = llama.Config.from_hf(json.loads((path / "config.json").read_text()), dtype=dtype)
    raw = {}
    for f in sorted(path.glob("*.safetensors")):
        raw |= load_file(str(f))
    return llama.convert_hf(raw, cfg), cfg, Tokenizer.from_file(str(path / "tokenizer.json"))


def server_reachable(url: str = "http://llama:8080/v1/models", timeout: float = 2.0) -> bool:
    """True if Module 19's llama.cpp server answers within `timeout` s (DNS included). Never blocks longer."""
    result = []

    def probe():
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                result.append(r.status == 200)
        except Exception:  # noqa: BLE001 - unreachable, unresolvable, refused: all mean "no server"
            result.append(False)

    t = threading.Thread(target=probe, daemon=True)
    t.start()
    t.join(timeout + 0.5)
    return bool(result and result[0])

@partial(jax.jit, static_argnames=("cfg",))
def llama_token_logprobs(params, ids, cfg):
    """ids [B, T] → log p(ids[:, t+1] | ids[:, :t+1]) as float32 [B, T-1]. Only [B, T] leaves the GPU."""
    logits, _ = llama.forward(params, ids, cfg)
    logp = jax.nn.log_softmax(logits[:, :-1], axis=-1)
    return jnp.take_along_axis(logp, ids[:, 1:, None], axis=-1)[..., 0]


def window_logprob_fn(params, cfg, window: int, pad_id: int = 0):
    """A `token_logprob_fn(chunk)` for the sliding-window exercise: pads every chunk to `window`
    tokens (one compile), then keeps the first len(chunk)-1 results. Causal masking means padding
    *after* the chunk can't change its log-probs."""

    def fn(chunk):
        n = len(chunk)
        ids = np.full((1, window), pad_id, np.int32)
        ids[0, :n] = chunk
        return np.asarray(llama_token_logprobs(params, jnp.asarray(ids), cfg))[0, : n - 1]

    return fn


def score_continuations(params, cfg, contexts, continuations, length: int = 128, batch: int = 32, pad_id: int = 0):
    """Log-probs of each continuation given its context, batched with fixed shapes.

    contexts / continuations: lists of token-id lists (same length N). Sequences longer than `length`
    lose tokens from the *left* of the context. Returns (token_logprobs [N, length-1], mask [N, length-1])
    where mask marks the continuation's tokens (entry j scores ids[j+1]).
    """
    n = len(contexts)
    ids = np.full((n, length), pad_id, np.int32)
    mask = np.zeros((n, length - 1), bool)
    for i, (c, a) in enumerate(zip(contexts, continuations, strict=True)):
        seq = (list(c) + list(a))[-length:]
        k = min(len(a), len(seq) - 1)
        ids[i, : len(seq)] = seq
        mask[i, len(seq) - 1 - k: len(seq) - 1] = True
    out = np.zeros((n, length - 1), np.float32)
    for s in range(0, n, batch):
        chunk = ids[s: s + batch]
        if len(chunk) < batch:  # pad the last batch so the shape (and the compile) never changes
            chunk = np.concatenate([chunk, np.full((batch - len(chunk), length), pad_id, np.int32)])
        out[s: s + batch] = np.asarray(llama_token_logprobs(params, jnp.asarray(chunk), cfg))[: min(batch, n - s)]
    return out, mask


@partial(jax.jit, static_argnames=("cfg",))
def _last_logits(params, ids, lengths, cfg):
    logits, _ = llama.forward(params, ids, cfg)
    return jnp.take_along_axis(logits, (lengths - 1)[:, None, None], axis=1)[:, 0]


def next_token_logits(params, cfg, prompts, length: int = 256, pad_id: int = 0):
    """Logits for the token right after each prompt ([N, V] float32), right-padded to `length`."""
    n = len(prompts)
    ids = np.full((n, length), pad_id, np.int32)
    lens = np.array([len(p) for p in prompts], np.int32)
    for i, p in enumerate(prompts):
        ids[i, : len(p)] = p
    return np.asarray(_last_logits(params, jnp.asarray(ids), jnp.asarray(lens), cfg))


@partial(jax.jit, static_argnames=("cfg", "n_new"))
def _greedy(params, ids, lengths, cfg, n_new):
    rows = jnp.arange(ids.shape[0])

    def step(j, ids):
        logits, _ = llama.forward(params, ids, cfg)   # full recompute: simple, fixed shape, one compile
        cur = lengths + j
        nxt = jnp.argmax(logits[rows, cur - 1], -1).astype(ids.dtype)
        return ids.at[rows, cur].set(nxt)

    return jax.lax.fori_loop(0, n_new, step, ids)


def greedy_generate(params, cfg, prompts, n_new: int = 12, length: int = 128, pad_id: int = 0):
    """Batched greedy generation without a KV cache (Module 18 makes this fast). Returns new-token lists."""
    n = len(prompts)
    ids = np.full((n, length), pad_id, np.int32)
    lens = np.array([len(p) for p in prompts], np.int32)
    assert lens.max() + n_new <= length, "prompt + n_new must fit in `length`"
    for i, p in enumerate(prompts):
        ids[i, : len(p)] = p
    out = np.asarray(_greedy(params, jnp.asarray(ids), jnp.asarray(lens), cfg, n_new))
    return [out[i, lens[i]: lens[i] + n_new].tolist() for i in range(n)]


# ------------------------------------------------------------------ Module 16's GPT (if you trained it)

def load_gpt16():
    """(params, cfg, model_module, source) for runs/16-gpt, or None if that run doesn't exist.

    Uses your Module 16 exercises.py if it can build the model, otherwise the reference solutions —
    it's only needed here to *evaluate* the checkpoint."""
    if not (RUN16 / "params.npz").exists():
        return None
    spec = importlib.util.spec_from_file_location("m16_train_for_m21", MODULE16 / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    for solutions, source in ((False, "your Module 16 code"), (True, "Module 16 reference code")):
        m = load_exercises(MODULE16, solutions=solutions)
        try:
            params, cfg, _ = train.load_checkpoint(m, RUN16)
            m.gpt_apply(params, jnp.zeros((1, 8), jnp.int32), cfg)
            return params, cfg, m, source
        except NotImplementedError:
            continue
    return None


def gpt16_window_logprob_fn(params, cfg, m):
    window = cfg.seq_len

    @jax.jit
    def lp(params, ids):
        logp = jax.nn.log_softmax(m.gpt_apply(params, ids, cfg)[:, :-1].astype(jnp.float32), -1)
        return jnp.take_along_axis(logp, ids[:, 1:, None], -1)[..., 0]

    def fn(chunk):
        n = len(chunk)
        ids = np.zeros((1, window), np.int32)
        ids[0, :n] = chunk
        return np.asarray(lp(params, jnp.asarray(ids)))[0, : n - 1]

    return fn


# ------------------------------------------------------------------ ARC-Easy

TEMPLATES = {
    "Question/Answer": ("Question: {q}\nAnswer:", " {a}"),
    "Bare question": ("{q}", " {a}"),
    "Science quiz": ("The following is a science quiz question for students.\n\n{q}\nThe correct answer is", " {a}"),
}


ARC_CACHE = Path(os.environ.get("LEARNJAX_CACHE", Path.home() / ".cache" / "learnjax")) / "arc_easy"


def arc_split(split: str = "test") -> list[dict]:
    """All ARC-Easy items of a split as plain dicts, cached as JSON after the first download so later runs
    make no network calls. (ARC: AI2, CC BY-SA 4.0.)"""
    path = ARC_CACHE / f"{split}.json"
    if not path.exists():
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
        rows = [{"question": x["question"], "choices": x["choices"]["text"], "labels": x["choices"]["label"],
                 "answer": x["answerKey"]} for x in ds]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows))
        tmp.replace(path)
    return json.loads(path.read_text())


def arc_items(n: int = 300, seed: int = 0):
    """`n` ARC-Easy test items with exactly 4 choices: dicts with question, choices, label (index)."""
    items = [{"question": x["question"], "choices": x["choices"], "label": x["labels"].index(x["answer"])}
             for x in arc_split("test") if len(x["labels"]) == 4]
    idx = np.random.default_rng(seed).permutation(len(items))[:n]
    return [items[i] for i in sorted(idx)]


def build_mc(tok, items, template: str):
    """Contexts/continuations for every (item, choice), plus each choice's byte length."""
    ctx_t, ans_t = TEMPLATES[template]
    contexts, conts, nbytes = [], [], []
    for it in items:
        c = tok.encode(ctx_t.format(q=it["question"])).ids
        for ch in it["choices"]:
            a = ans_t.format(a=ch)
            contexts.append(c)
            conts.append(tok.encode(a).ids)
            nbytes.append(len(a.encode("utf-8")))
    return contexts, conts, np.array(nbytes)
