"""Module 26 reference solutions. Try the exercises first — `make test N=26` grades yours."""
from __future__ import annotations

import math
import re
from collections import Counter

import jax.numpy as jnp
import numpy as np

_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_INT = re.compile(r"[+-]?\d+")


def verify_answer(text: str, target: int) -> bool:
    tags = _TAG.findall(text)
    if len(tags) != 1:
        return False
    inner = tags[0].strip().replace(",", "")
    if not re.fullmatch(r"[+-]?\d+", inner):
        return False
    return int(inner) == int(target)


def format_reward(text: str) -> float:
    tags = _TAG.findall(text)
    if len(tags) != 1 or not re.fullmatch(r"\s*[+-]?\d+\s*", tags[0]):
        return 0.0
    tail = text[text.rindex("</answer>") + len("</answer>"):]
    return 1.0 if tail.strip() == "" else 0.0


def combined_reward(text: str, target: int, format_bonus: float = 0.2) -> float:
    if not verify_answer(text, target):
        return 0.0
    return 1.0 + format_bonus * format_reward(text)


def group_advantages(rewards, group_size: int, eps: float = 1e-4):
    r = np.asarray(rewards, np.float32).reshape(-1, group_size)
    adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + eps)
    return adv.reshape(-1)


def clipped_pg_loss(logp_new, logp_old, adv, mask, clip_eps: float = 0.2):
    ratio = jnp.exp(jnp.where(mask > 0, logp_new - logp_old, 0.0))
    a = adv[:, None]
    per_tok = jnp.minimum(ratio * a, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    per_seq = (per_tok * mask).sum(1) / jnp.maximum(mask.sum(1), 1.0)
    return -per_seq.mean()


def kl_penalty(logp, logp_ref, mask):
    d = jnp.where(mask > 0, logp_ref - logp, 0.0)        # padded positions may hold garbage: zero before exp
    k3 = jnp.exp(d) - d - 1.0
    per_seq = (k3 * mask).sum(1) / jnp.maximum(mask.sum(1), 1.0)
    return per_seq.mean()


def majority_vote(answers):
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None
    best = max(counts.values())
    for a in answers:                      # first answer to reach the top count wins ties
        if a is not None and counts[a] == best:
            return a
    return None


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
