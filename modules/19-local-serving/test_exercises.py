import json

import pytest

GiB = 2**30


def test_gguf_weight_bytes(impl):
    assert impl.gguf_weight_bytes(1e9, 4.5) == 562_500_000
    assert impl.gguf_weight_bytes(8, 4) == 4
    assert impl.gguf_weight_bytes(3, 4.85) == 2          # 1.82 bytes rounds UP
    assert impl.gguf_weight_bytes(0, 8.5) == 0


def test_kv_cache_bytes(impl):
    # Qwen2.5-Coder-14B: 48 layers, 8 KV heads × 128 dims, f16 — 196,608 bytes per token
    assert impl.kv_cache_bytes(48, 8, 128, 1) == 196_608
    assert impl.kv_cache_bytes(48, 8, 128, 8192) == 196_608 * 8192
    assert impl.kv_cache_bytes(48, 8, 128, 8192, bytes_per_elem=1) == 196_608 * 8192 // 2
    assert impl.kv_cache_bytes(2, 1, 4, 10, batch=3) == 2 * 2 * 1 * 4 * 10 * 2 * 3


def test_fits_in_vram(impl):
    assert impl.fits_in_vram(9 * GiB, 1.5 * GiB, 16)
    assert not impl.fits_in_vram(15 * GiB, 1 * GiB, 16)
    assert impl.fits_in_vram(14 * GiB, 1 * GiB, 16)          # exactly full still fits
    assert not impl.fits_in_vram(14 * GiB, 1 * GiB, 16, overhead_gb=1.5)


def test_gpu_layers(impl):
    assert impl.gpu_layers(48, 9 * GiB, 1 * GiB, 16) == 48                 # all fits
    # 20 GiB over 40 layers = 0.5 GiB/layer; 16 - 1 = 15 GiB budget → 30 layers
    assert impl.gpu_layers(40, 18 * GiB, 2 * GiB, 16) == 30
    assert impl.gpu_layers(40, 18 * GiB, 2 * GiB, 0.5) == 0                # overhead alone doesn't fit
    assert isinstance(impl.gpu_layers(40, 18 * GiB, 2 * GiB, 10.3), int)


def test_build_chat_request(impl):
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    body = impl.build_chat_request(msgs, "m", temperature=0.0, max_tokens=7, stream=True)
    assert body == {"model": "m", "messages": msgs, "temperature": 0.0, "max_tokens": 7, "stream": True}
    assert impl.build_chat_request(msgs, "m", stop=["\n\n"])["stop"] == ["\n\n"]
    json.dumps(body)  # must be JSON-serializable
    with pytest.raises(ValueError):
        impl.build_chat_request([{"role": "wizard", "content": "x"}], "m")
    with pytest.raises(ValueError):
        impl.build_chat_request([], "m")


def _event(content):
    delta = {} if content is None else {"content": content}
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]})


def test_parse_sse_stream(impl):
    lines = [_event("Hel"), "", _event("lo"), ": keep-alive comment", _event(None), _event(", world"),
             "data: [DONE]", _event("IGNORED")]
    assert impl.parse_sse_stream(lines) == "Hello, world"
    assert impl.parse_sse_stream([]) == ""
    assert impl.parse_sse_stream(["data: " + json.dumps({"choices": [{"delta": {"content": None}}]})]) == ""


def test_extract_code_block(impl):
    md = "Here you go:\n\n```bash\npip install x\n```\n\nand\n\n```python\ndef f():\n    return 1\n```\n"
    assert impl.extract_code_block(md) == "pip install x"
    assert impl.extract_code_block(md, "python") == "def f():\n    return 1"
    assert impl.extract_code_block(md, "rust") == "pip install x"           # falls back to the first block
    assert impl.extract_code_block("no code here") is None
    assert impl.extract_code_block("```\nplain\n```") == "plain"
