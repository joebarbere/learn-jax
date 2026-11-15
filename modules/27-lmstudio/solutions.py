"""Module 27 reference solutions. Try the exercises first — `make test N=27` grades yours."""
import json

import numpy as np

GGUF_BITS = {"F16": 16.0, "Q8_0": 8.5, "Q6_K": 6.56, "Q5_K_M": 5.69, "Q4_K_M": 4.85, "Q3_K_M": 3.91,
             "Q2_K": 2.63}

_LAYER_LINEAR = {"wq": "self_attn.q_proj", "wk": "self_attn.k_proj", "wv": "self_attn.v_proj",
                 "wo": "self_attn.o_proj", "w_gate": "mlp.gate_proj", "w_up": "mlp.up_proj",
                 "w_down": "mlp.down_proj"}
_LAYER_NORM = {"attn_norm": "input_layernorm", "mlp_norm": "post_attention_layernorm"}


def export_llama_hf(params: dict, cfg) -> dict:
    out = {"model.embed_tokens.weight": np.asarray(params["embed"]),
           "model.norm.weight": np.asarray(params["final_norm"])}
    layers = params["layers"]
    for i in range(cfg.n_layers):
        for ours, hf in _LAYER_LINEAR.items():
            out[f"model.layers.{i}.{hf}.weight"] = np.asarray(layers[ours][i]).T
        for ours, hf in _LAYER_NORM.items():
            out[f"model.layers.{i}.{hf}.weight"] = np.asarray(layers[ours][i])
    if not cfg.tied:
        out["lm_head.weight"] = np.asarray(params["head"]).T
    return out


_GPT_BLOCK = {"ln1_g": "ln_1.weight", "ln1_b": "ln_1.bias", "w_qkv": "attn.c_attn.weight",
              "b_qkv": "attn.c_attn.bias", "w_o": "attn.c_proj.weight", "b_o": "attn.c_proj.bias",
              "ln2_g": "ln_2.weight", "ln2_b": "ln_2.bias", "w_fc": "mlp.c_fc.weight", "b_fc": "mlp.c_fc.bias",
              "w_proj": "mlp.c_proj.weight", "b_proj": "mlp.c_proj.bias"}


def gpt_to_gpt2_state_dict(params: dict, n_layers: int) -> dict:
    out = {"transformer.wte.weight": np.asarray(params["wte"]),
           "transformer.wpe.weight": np.asarray(params["wpe"]),
           "transformer.ln_f.weight": np.asarray(params["lnf_g"]),
           "transformer.ln_f.bias": np.asarray(params["lnf_b"])}
    for i in range(n_layers):
        for ours, hf in _GPT_BLOCK.items():
            out[f"transformer.h.{i}.{hf}"] = np.asarray(params["blocks"][i][ours])
    return out


def bits_per_byte_delta(ppl_base: float, ppl_quant: float, tokens_per_byte: float) -> float:
    return (float(np.log2(ppl_quant)) - float(np.log2(ppl_base))) * tokens_per_byte


def openai_chat_request(model, messages, max_tokens=256, temperature=0.0, response_format=None, tools=None,
                        stream=False):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": stream}
    if response_format is not None:
        body["response_format"] = response_format
    if tools is not None:
        body["tools"] = tools
    return body


def lmstudio_rest_request(model, user_input, max_output_tokens=256, temperature=0.0, system_prompt=None):
    body = {"model": model, "input": user_input, "max_output_tokens": max_output_tokens,
            "temperature": temperature}
    if system_prompt is not None:
        body["system_prompt"] = system_prompt
    return body


def anthropic_messages_request(model, messages, max_tokens=256, system=None, temperature=0.0):
    body = {"model": model, "max_tokens": max_tokens, "messages": messages, "temperature": temperature}
    if system is not None:
        body["system"] = system
    return body


def assemble_tool_call_stream(chunks):
    calls: dict[int, dict] = {}
    for ch in chunks:
        choices = ch.get("choices") or []
        if not choices:
            continue
        for tc in (choices[0].get("delta") or {}).get("tool_calls") or []:
            c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                c["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                c["name"] = fn["name"]
            c["arguments"] += fn.get("arguments") or ""
    return [{"id": c["id"], "name": c["name"], "arguments": json.loads(c["arguments"] or "{}")}
            for _, c in sorted(calls.items())]


def quant_recommendation(n_params, vram_gb, ctx, kv_bytes_per_token, overhead_gb=1.0):
    budget = vram_gb * 2**30
    fixed = ctx * kv_bytes_per_token + overhead_gb * 2**30
    for name, bits in GGUF_BITS.items():
        if n_params * bits / 8 + fixed <= budget:
            return name
    return None
