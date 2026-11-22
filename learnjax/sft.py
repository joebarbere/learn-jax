"""Supervised fine-tuning (SFT) of a `learnjax.llama` model on instruction/response pairs.

The recipe Stanford Alpaca and WizardLM used (and Eric Hartford reused for his uncensored WizardLM): wrap each
instruction in a fixed prompt template, train on prompt + response + end-of-text, and count the loss on the
response tokens only (the model is not taught to write prompts).

    ex = sft.encode(rows, tok, max_len=1024)                  # token ids + where each response starts
    for step, micro in enumerate(sft.steps(ex, batch=128, rng=rng)):   # 128 examples per optimizer step
        ...                                                   # each micro-batch: (B, T) tokens, mask, loss_mask

Batches are padded, not packed: a step's 128 examples are sorted by length and cut into micro-batches of at most
`tokens` tokens, each padded (on the left, see llama.trunk's mask) to one of a few bucket lengths, so there are only
a handful of compiled shapes and little padding. `Trainer` accumulates the gradient over a step's micro-batches.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import llama

# Stanford Alpaca's template for rows without an "input" (WizardLM's train_freeform.py uses the same one).
ALPACA = ("Below is an instruction that describes a task. Write a response that appropriately completes the "
          "request.\n\n### Instruction:\n{instruction}\n\n### Response:")


# micro-batch lengths: finer steps waste less padding (23% vs 31% with 5 buckets on evol-instruct), each is a compile
BUCKETS = (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024)


def prompt(instruction: str) -> str:
    return ALPACA.format(instruction=instruction)


@dataclasses.dataclass
class Example:
    ids: np.ndarray        # prompt + response + eos (int32)
    start: int             # index of the first response token
    truncated: bool


def encode(rows: Sequence[dict], tok, max_len: int = 1024, eos_id: int = 0) -> list[Example]:
    """Tokenise {"instruction", "output"} rows. The response is " " + output + eos (the space keeps the first
    word's usual token after "Response:"). Rows longer than max_len are cut at max_len (their eos is lost, as in
    Alpaca's code); a prompt longer than max_len - 16 is cut so at least 16 response tokens remain."""
    p = tok.encode_batch([prompt(r["instruction"]) for r in rows])
    o = tok.encode_batch([" " + r["output"] for r in rows])
    out = []
    for pe, oe in zip(p, o):
        pi = pe.ids[: max_len - 16]
        ids = np.asarray(pi + oe.ids + [eos_id], np.int32)
        out.append(Example(ids[:max_len], len(pi), len(ids) > max_len))
    return out


def bucket_of(n: int, buckets: Sequence[int]) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"length {n} > largest bucket {buckets[-1]}")


def micro_batches(exs: Sequence[Example], tokens: int = 8192, buckets=BUCKETS,
                  pad_id: int = 0) -> list[dict]:
    """Cut exs into padded micro-batches of at most `tokens` tokens (rows × bucket length). Shorter rows share a
    batch with similar ones (sorted by length). Each dict: tokens, mask (real tokens), loss_mask (response
    tokens, aligned with the *targets* tokens[:, 1:]), n (response tokens)."""
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].ids))
    groups, cur = [], []
    for i in order:
        b = bucket_of(len(exs[i].ids), buckets)
        if cur and (len(cur) + 1) * b > tokens:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        T = bucket_of(max(len(exs[i].ids) for i in g), buckets)
        rows = max(1, tokens // T)                         # fixed rows per bucket → one shape per bucket
        tok = np.full((rows, T), pad_id, np.int32)
        mask = np.zeros((rows, T), bool)
        lm = np.zeros((rows, T - 1), bool)
        for r, i in enumerate(g):
            e = exs[i]
            n = len(e.ids)
            tok[r, T - n:] = e.ids
            mask[r, T - n:] = True
            lm[r, T - n + e.start - 1:] = True             # target t+1 is a response token
        out.append({"tokens": tok, "mask": mask, "loss_mask": lm, "n": int(lm.sum())})
    return out


def steps(exs: Sequence[Example], batch: int = 128, rng: np.random.Generator | None = None, tokens: int = 8192,
          buckets=BUCKETS, skip: int = 0) -> Iterator[list[dict]]:
    """One epoch in a random order: for each optimizer step, the micro-batches holding its `batch` examples.
    The last partial step is dropped. skip: steps to skip (to resume a run with the same rng seed)."""
    rng = rng or np.random.default_rng(0)
    perm = rng.permutation(len(exs))
    for s in range(len(exs) // batch):
        if s < skip:
            continue
        yield micro_batches([exs[i] for i in perm[s * batch:(s + 1) * batch]], tokens, buckets)


def forward_train(params, tokens, mask, cfg: llama.Config):
    """llama.forward with every layer in jax.checkpoint (the backward pass recomputes a layer's activations rather
    than storing all of them), returning the final hidden states; the head is applied in the loss."""
    positions = jnp.maximum(jnp.cumsum(mask.astype(jnp.int32), 1) - 1, 0)
    layer = jax.checkpoint(lambda h, lp: (llama._layer(cfg, h, lp, positions, None, 0, mask)[0], None))
    h, _ = jax.lax.scan(layer, params["embed"][tokens], params["layers"])
    return llama.rms_norm(h, params["final_norm"], cfg.norm_eps)


def response_nll(params, tokens, mask, loss_mask, cfg: llama.Config):
    """Summed negative log-likelihood of the response tokens (and how many there are): −Σ log p(next token)."""
    h = forward_train(params, tokens, mask, cfg)[:, :-1]
    head = params["embed"].T if cfg.tied else params["head"]
    return chunked_nll(h.reshape(-1, h.shape[-1]), head, tokens[:, 1:].reshape(-1), loss_mask.reshape(-1)), \
        loss_mask.sum()


def chunked_nll(h, head, targets, keep, chunk: int = 2048):
    """Σ over kept rows of −log softmax(h @ head)[target], `chunk` rows at a time and recomputed in the backward
    pass: the full (rows, vocab) float32 logits would be 1.6 GB for 8 × 1024 tokens and a 49,152-token vocab."""
    n = h.shape[0]
    pad = (-n) % chunk
    h = jnp.pad(h, ((0, pad), (0, 0)))
    targets, keep = jnp.pad(targets, (0, pad)), jnp.pad(keep, (0, pad))

    @jax.checkpoint
    def one(hc, tc, kc):
        lp = jax.nn.log_softmax((hc @ head).astype(jnp.float32), -1)
        return jnp.where(kc, -jnp.take_along_axis(lp, tc[:, None], -1)[:, 0], 0.0).sum()

    parts = jax.lax.map(lambda a: one(*a), (h.reshape(-1, chunk, h.shape[1]), targets.reshape(-1, chunk),
                                            keep.reshape(-1, chunk)))
    return parts.sum()


def make_optimizer(lr: float, total_steps: int, warmup: int = 2, weight_decay: float = 0.0, clip: float = 1.0):
    """Hartford's (and WizardLM's) settings: AdamW, linear warmup then cosine decay to 0, no weight decay."""
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, max(warmup, 1), max(total_steps, warmup + 1), 0.0)
    return optax.chain(optax.clip_by_global_norm(clip), optax.adamw(sched, b1=0.9, b2=0.999,
                                                                    weight_decay=weight_decay)), sched


class Trainer:
    """Full fine-tuning with gradient accumulation. Master weights and Adam state in float32, compute in the
    model's dtype (bf16). One optimizer step = all micro-batches of one `steps()` item; the loss is the mean over
    all of the step's response tokens."""

    def __init__(self, params, cfg: llama.Config, optimizer):
        self.cfg = cfg
        self.compute = jnp.dtype(cfg.dtype)
        self.params = jax.tree.map(lambda a: a.astype(jnp.float32), params)
        self.opt = optimizer
        self.opt_state = optimizer.init(self.params)
        self._grad = jax.jit(jax.value_and_grad(self._loss, has_aux=True))
        self._acc = jax.jit(lambda a, g: jax.tree.map(jnp.add, a, g), donate_argnums=0)

    def _loss(self, params, tokens, mask, loss_mask):
        p = jax.tree.map(lambda a: a.astype(self.compute), params)
        s, n = response_nll(p, tokens, mask, loss_mask, self.cfg)
        return s, n

    @partial(jax.jit, static_argnums=0, donate_argnums=(1, 2, 3))
    def _apply(self, params, opt_state, grads, n):
        grads = jax.tree.map(lambda g: g / n, grads)
        updates, opt_state = self.opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, optax.global_norm(grads)

    def step(self, micro: Sequence[dict]) -> dict:
        tot, n, acc = 0.0, 0, None
        for mb in micro:
            (s, _), g = self._grad(self.params, jnp.asarray(mb["tokens"]), jnp.asarray(mb["mask"]),
                                   jnp.asarray(mb["loss_mask"]))
            acc = g if acc is None else self._acc(acc, g)
            tot, n = tot + s, n + mb["n"]
        self.params, self.opt_state, gn = self._apply(self.params, self.opt_state, acc, jnp.float32(max(n, 1)))
        return {"loss": float(tot) / max(n, 1), "resp_tokens": n, "grad_norm": float(gn)}

    def bf16(self):
        return jax.tree.map(lambda a: a.astype(self.compute), self.params)


def eval_nll(params, cfg: llama.Config, exs: Sequence[Example], tokens: int = 8192,
             buckets=BUCKETS) -> float:
    """Mean response-token NLL (nats) of exs under params."""
    f = jax.jit(lambda p, t, m, lm: response_nll(p, t, m, lm, cfg))
    s = n = 0.0
    for mb in micro_batches(exs, tokens, buckets):
        a, b = f(params, jnp.asarray(mb["tokens"]), jnp.asarray(mb["mask"]), jnp.asarray(mb["loss_mask"]))
        s, n = s + float(a), n + float(b)
    return s / max(n, 1)
