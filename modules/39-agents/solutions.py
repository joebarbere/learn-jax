"""Module 39 reference solutions. Don't peek until you've tried exercises.py."""
import collections
import json
import re

import numpy as np

_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_call(text: str):
    m = _CALL.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not (isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict)):
        return None
    return {"name": obj["name"], "arguments": obj["arguments"]}


def dispatch_tool(registry: dict, call: dict, allowed=None) -> str:
    name = call["name"]
    if allowed is not None and name not in allowed:
        return f"error: tool {name!r} is not allowed here"
    if name not in registry:
        return f"error: unknown tool {name!r}"
    try:
        return str(registry[name](**call["arguments"]))
    except Exception as e:  # noqa: BLE001 - errors go back to the model as text
        return f"error: {type(e).__name__}: {e}"


def token_mask(token_strings: list[str], allowed_chars: str, terminators=()) -> np.ndarray:
    allowed = set(allowed_chars)
    out = np.zeros(len(token_strings), bool)
    for i, s in enumerate(token_strings):
        out[i] = (bool(s) and set(s) <= allowed) or any(s.startswith(t) for t in terminators)
    return out


def select_by_logprob(candidate_logprobs) -> int:
    return int(np.argmax([float(np.sum(lp)) for lp in candidate_logprobs]))


def trim_window(messages: list[dict], budget: int, count_tokens) -> list[dict]:
    head = [m for m in messages[:1] if m["role"] == "system"]
    used = sum(count_tokens(m["content"]) for m in head)
    kept = []
    for m in reversed(messages[len(head):]):
        n = count_tokens(m["content"])
        if used + n > budget:
            break
        kept.append(m)
        used += n
    return head + kept[::-1]


def recency_relevance_score(ages, relevances, decay: float = 0.995, w_recency: float = 1.0,
                            w_relevance: float = 1.0) -> np.ndarray:
    def minmax(x):
        x = np.asarray(x, np.float64)
        span = x.max() - x.min()
        return np.zeros_like(x) if span == 0 else (x - x.min()) / span

    recency = np.power(decay, np.asarray(ages, np.float64))
    return w_recency * minmax(recency) + w_relevance * minmax(relevances)


def majority_vote(answers: list):
    votes = collections.Counter(a for a in answers if a is not None)
    if not votes:
        return None
    best = max(votes.values())
    return next(a for a in answers if a is not None and votes[a] == best)
