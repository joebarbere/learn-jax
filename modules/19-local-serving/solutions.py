"""Module 19 reference solutions. Try the exercises first — `make test N=19` grades yours."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable

GiB = 2**30
ROLES = {"system", "user", "assistant", "tool"}


def gguf_weight_bytes(n_params: float, bits_per_weight: float) -> int:
    return math.ceil(n_params * bits_per_weight / 8)


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, context: int,
                   bytes_per_elem: int = 2, batch: int = 1) -> int:
    return 2 * n_layers * n_kv_heads * head_dim * context * bytes_per_elem * batch


def fits_in_vram(weight_bytes: int, kv_bytes: int, vram_gb: float, overhead_gb: float = 1.0) -> bool:
    return weight_bytes + kv_bytes + overhead_gb * GiB <= vram_gb * GiB


def gpu_layers(n_layers: int, weight_bytes: int, kv_bytes: int, vram_gb: float, overhead_gb: float = 1.0) -> int:
    budget = vram_gb * GiB - overhead_gb * GiB
    if budget <= 0:
        return 0
    per_layer = (weight_bytes + kv_bytes) / n_layers
    return int(min(n_layers, budget // per_layer))


def build_chat_request(messages: list[dict], model: str, *, temperature: float = 0.2, max_tokens: int = 512,
                       stream: bool = False, stop: list[str] | None = None) -> dict:
    if not messages:
        raise ValueError("messages must not be empty")
    for m in messages:
        if m.get("role") not in ROLES:
            raise ValueError(f"bad role: {m.get('role')!r}")
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": stream}
    if stop is not None:
        body["stop"] = stop
    return body


def parse_sse_stream(lines: Iterable[str]) -> str:
    out = []
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        choices = json.loads(payload).get("choices") or []
        if choices:
            piece = (choices[0].get("delta") or {}).get("content")
            if piece:
                out.append(piece)
    return "".join(out)


FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


def extract_code_block(markdown: str, language: str | None = None) -> str | None:
    blocks = [(info.strip().lower(), body) for info, body in FENCE.findall(markdown)]
    if not blocks:
        return None
    if language:
        for info, body in blocks:
            if info.split()[:1] == [language.lower()]:
                return body.rstrip("\n")
    return blocks[0][1].rstrip("\n")
