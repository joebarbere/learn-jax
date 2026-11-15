import json

import numpy as np
import pytest

from learnjax import llama


def _tiny_llama(tied=True):
    cfg = llama.Config(vocab=32, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=24, tied=tied,
                       dtype="float32")
    rng = np.random.default_rng(0)
    hd = cfg.head_dim
    shapes = {"self_attn.q_proj": (cfg.n_heads * hd, cfg.d_model), "self_attn.k_proj": (cfg.n_kv_heads * hd, cfg.d_model),
              "self_attn.v_proj": (cfg.n_kv_heads * hd, cfg.d_model), "self_attn.o_proj": (cfg.d_model, cfg.n_heads * hd),
              "mlp.gate_proj": (cfg.d_ff, cfg.d_model), "mlp.up_proj": (cfg.d_ff, cfg.d_model),
              "mlp.down_proj": (cfg.d_model, cfg.d_ff)}
    raw = {"model.embed_tokens.weight": rng.normal(size=(cfg.vocab, cfg.d_model)).astype(np.float32),
           "model.norm.weight": rng.normal(size=cfg.d_model).astype(np.float32)}
    for i in range(cfg.n_layers):
        for n, s in shapes.items():
            raw[f"model.layers.{i}.{n}.weight"] = rng.normal(size=s).astype(np.float32)
        raw[f"model.layers.{i}.input_layernorm.weight"] = rng.normal(size=cfg.d_model).astype(np.float32)
        raw[f"model.layers.{i}.post_attention_layernorm.weight"] = rng.normal(size=cfg.d_model).astype(np.float32)
    if not tied:
        raw["lm_head.weight"] = rng.normal(size=(cfg.vocab, cfg.d_model)).astype(np.float32)
    return raw, cfg


@pytest.mark.parametrize("tied", [True, False])
def test_export_llama_hf_round_trip(impl, tied):
    raw, cfg = _tiny_llama(tied)
    params = llama.convert_hf({k: v for k, v in raw.items()}, cfg)
    out = impl.export_llama_hf(params, cfg)
    assert set(out) == set(raw)
    for k in raw:
        np.testing.assert_array_equal(np.asarray(out[k]), raw[k], err_msg=k)
    back = llama.convert_hf(out, cfg)
    np.testing.assert_array_equal(np.asarray(back["layers"]["wq"]), np.asarray(params["layers"]["wq"]))


def _tiny_gpt(n=2, d=8, v=16, t=4):
    rng = np.random.default_rng(1)

    def r(*s):
        return rng.normal(size=s).astype(np.float32)

    blocks = [{"ln1_g": r(d), "ln1_b": r(d), "w_qkv": r(d, 3 * d), "b_qkv": r(3 * d), "w_o": r(d, d), "b_o": r(d),
               "ln2_g": r(d), "ln2_b": r(d), "w_fc": r(d, 4 * d), "b_fc": r(4 * d), "w_proj": r(4 * d, d),
               "b_proj": r(d)} for _ in range(n)]
    return {"wte": r(v, d), "wpe": r(t, d), "lnf_g": r(d), "lnf_b": r(d), "blocks": blocks}


def test_gpt_to_gpt2_state_dict(impl):
    p = _tiny_gpt()
    sd = impl.gpt_to_gpt2_state_dict(p, 2)
    assert len(sd) == 4 + 2 * 12
    assert "lm_head.weight" not in sd                          # tied to wte
    np.testing.assert_array_equal(sd["transformer.h.1.attn.c_attn.weight"], p["blocks"][1]["w_qkv"])
    assert sd["transformer.h.0.attn.c_attn.weight"].shape == (8, 24)   # Conv1D: [in, out], no transpose
    np.testing.assert_array_equal(sd["transformer.h.0.mlp.c_proj.bias"], p["blocks"][0]["b_proj"])
    np.testing.assert_array_equal(sd["transformer.ln_f.weight"], p["lnf_g"])
    np.testing.assert_array_equal(sd["transformer.wpe.weight"], p["wpe"])


def test_bits_per_byte_delta(impl):
    assert impl.bits_per_byte_delta(8.0, 8.0, 0.3) == pytest.approx(0.0)
    assert impl.bits_per_byte_delta(8.0, 16.0, 0.25) == pytest.approx(0.25)    # one extra bit/token
    assert impl.bits_per_byte_delta(16.0, 8.0, 0.5) == pytest.approx(-0.5)


def test_request_builders(impl):
    msgs = [{"role": "user", "content": "hi"}]
    b = impl.openai_chat_request("m", msgs)
    assert b == {"model": "m", "messages": msgs, "max_tokens": 256, "temperature": 0.0, "stream": False}
    rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    b = impl.openai_chat_request("m", msgs, response_format=rf, tools=[], stream=True)
    assert b["response_format"] == rf and b["tools"] == [] and b["stream"] is True
    r = impl.lmstudio_rest_request("m", "hello", max_output_tokens=10)
    assert r == {"model": "m", "input": "hello", "max_output_tokens": 10, "temperature": 0.0}
    assert impl.lmstudio_rest_request("m", "x", system_prompt="s")["system_prompt"] == "s"
    a = impl.anthropic_messages_request("m", msgs, max_tokens=5, system="be brief")
    assert a == {"model": "m", "max_tokens": 5, "messages": msgs, "temperature": 0.0, "system": "be brief"}
    assert "system" not in impl.anthropic_messages_request("m", msgs)
    assert all(m["role"] != "system" for m in a["messages"])


def test_assemble_tool_call_stream(impl):
    def tc(i, args, id_=None, name=None):
        fn = {"arguments": args}
        if name:
            fn["name"] = name
        d = {"index": i, "function": fn}
        if id_:
            d["id"] = id_
        return {"choices": [{"delta": {"tool_calls": [d]}}]}

    chunks = [{"choices": [{"delta": {"role": "assistant"}}]},
              tc(0, "", "call_a", "get_weather"), tc(0, '{"city": "Den'), tc(1, '{"x"', "call_b", "add"),
              tc(0, 'ver", "unit": "C"}'), tc(1, ": 2}"),
              {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}, {"choices": []}]
    out = impl.assemble_tool_call_stream(chunks)
    assert out == [{"id": "call_a", "name": "get_weather", "arguments": {"city": "Denver", "unit": "C"}},
                   {"id": "call_b", "name": "add", "arguments": {"x": 2}}]
    assert impl.assemble_tool_call_stream([{"choices": [{"delta": {"content": "hi"}}]}]) == []
    json.dumps(out)


def test_quant_recommendation(impl):
    gib = 2**30
    # Qwen2.5-Coder-14B (KV: 48 layers x 8 heads x 128 x 2 x 2 bytes = 196,608 B/token) on 16 GB:
    # at 4K context Q6_K fits; at 32K the KV cache (6 GiB) pushes it down to Q4_K_M
    assert impl.quant_recommendation(14.8e9, 16, 4096, 196608) == "Q6_K"
    assert impl.quant_recommendation(14.8e9, 16, 32768, 196608) == "Q4_K_M"
    # a 360M model fits at F16 easily
    assert impl.quant_recommendation(3.6e8, 16, 8192, 61440) == "F16"
    # 70B doesn't fit at all
    assert impl.quant_recommendation(70e9, 16, 4096, 327680) is None
    # overhead counts: exactly at the limit fits
    assert impl.quant_recommendation(gib * 8 / 16.0, 3, 0, 0, overhead_gb=1.0) == "F16"
