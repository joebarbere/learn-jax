"""Module 19 exercises: the arithmetic and plumbing of serving a local model.

Replace each `raise NotImplementedError` with your code, then run `make test N=19`.
None of these need the server running — they're the pieces the lesson's live cells are built from.
"""
from __future__ import annotations

from collections.abc import Iterable

GiB = 2**30


def gguf_weight_bytes(n_params: float, bits_per_weight: float) -> int:
    """Bytes needed to store `n_params` weights at `bits_per_weight` bits each, rounded UP to a whole byte.

    Quantized formats have fractional effective bits per weight (Q4_K_M ≈ 4.85, Q8_0 = 8.5) because each
    block of weights also stores a scale. Example: 1e9 params at 4.5 bits → 562,500,000 bytes.
    """
    raise NotImplementedError


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, context: int,
                   bytes_per_elem: int = 2, batch: int = 1) -> int:
    """Memory for the KV cache: every layer stores a key AND a value vector per KV head, per token.

    With grouped-query attention only the `n_kv_heads` heads are cached (not all query heads).
    Default 2 bytes per element = f16 cache, llama.cpp's default.
    """
    raise NotImplementedError


def fits_in_vram(weight_bytes: int, kv_bytes: int, vram_gb: float, overhead_gb: float = 1.0) -> bool:
    """True if weights + KV cache + a fixed overhead (compute buffers, runtime) fit in `vram_gb` GiB."""
    raise NotImplementedError


def gpu_layers(n_layers: int, weight_bytes: int, kv_bytes: int, vram_gb: float, overhead_gb: float = 1.0) -> int:
    """How many of the model's layers fit on the GPU (llama.cpp's `-ngl`); the rest run on the CPU.

    Assume weights and KV cache are spread evenly over the layers, and the overhead must always fit.
    Return a value in [0, n_layers]; return n_layers when everything fits.
    """
    raise NotImplementedError


def build_chat_request(messages: list[dict], model: str, *, temperature: float = 0.2, max_tokens: int = 512,
                       stream: bool = False, stop: list[str] | None = None) -> dict:
    """The JSON body for an OpenAI-compatible POST /v1/chat/completions.

    `messages` is a list of {"role": ..., "content": ...}; roles must be one of system, user, assistant,
    tool — raise ValueError otherwise (and for an empty list). Include "stop" only when `stop` is given.
    Keys: model, messages, temperature, max_tokens, stream (and stop).
    """
    raise NotImplementedError


def parse_sse_stream(lines: Iterable[str]) -> str:
    """Join the text of a streamed chat completion (Server-Sent Events).

    Each event line looks like `data: {"choices":[{"delta":{"content":"Hel"}}], ...}`. Concatenate every
    `choices[0].delta.content` (it may be missing or null — skip those). Ignore blank lines and lines that
    don't start with "data:"; stop at `data: [DONE]`.
    """
    raise NotImplementedError


def extract_code_block(markdown: str, language: str | None = None) -> str | None:
    """Return the body of the first fenced code block (```...```) in `markdown`, or None if there is none.

    If `language` is given (e.g. "python"), return the first block whose info string is that language,
    falling back to the first block of any language. Don't include the fence lines; strip the trailing
    newline.
    """
    raise NotImplementedError
