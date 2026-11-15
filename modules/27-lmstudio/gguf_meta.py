"""Read the key/value metadata at the front of a GGUF file (no llama.cpp needed).

GGUF layout (ggml-org's spec, github.com/ggml-org/ggml/blob/master/docs/gguf.md): magic "GGUF", uint32
version, uint64 tensor count, uint64 metadata count, then each metadata entry as
(string key, uint32 type, value). Strings are uint64 length + UTF-8 bytes; arrays are uint32 element type +
uint64 count + elements. Everything is little-endian. This is where LM Studio finds a model's architecture,
context length, tokenizer and — the part that matters for chat — `tokenizer.chat_template`.
"""
from __future__ import annotations

import struct
from pathlib import Path

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _read(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _string(f) -> str:
    return f.read(_read(f, "<Q")).decode("utf-8", errors="replace")


def _value(f, typ, keep_arrays: bool):
    if typ in _SCALAR:
        return _read(f, _SCALAR[typ])
    if typ == 8:
        return _string(f)
    if typ == 9:
        etype, n = _read(f, "<I"), _read(f, "<Q")
        if not keep_arrays:                      # skip big arrays (the vocabulary) quickly
            if etype in _SCALAR:
                f.seek(struct.calcsize(_SCALAR[etype]) * n, 1)
            else:
                for _ in range(n):
                    _value(f, etype, False)
            return f"<array of {n}>"
        return [_value(f, etype, True) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {typ}")


def read_metadata(path: str | Path, keep_arrays: bool = False) -> dict:
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version, _n_tensors, n_kv = _read(f, "<I"), _read(f, "<Q"), _read(f, "<Q")
        meta = {"GGUF.version": version}
        for _ in range(n_kv):
            key = _string(f)
            meta[key] = _value(f, _read(f, "<I"), keep_arrays)
        return meta
