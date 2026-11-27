"""Module 27 exercises: get the models you trained out of JAX and into LM Studio, then talk to it.

Replace each `raise NotImplementedError` with your code, then run `make test N=27`. Everything here is
graded offline — no LM Studio server needed.
"""

# Bits per weight of llama.cpp's common GGUF quant types (including their per-block scales), largest first.
# Source: llama.cpp's quantize tool (ggml-org) — what `llama-quantize --help` lists.
GGUF_BITS = {"F16": 16.0, "Q8_0": 8.5, "Q6_K": 6.56, "Q5_K_M": 5.69, "Q4_K_M": 4.85, "Q3_K_M": 3.91,
             "Q2_K": 2.63}


def export_llama_hf(params: dict, cfg) -> dict:
    """The inverse of `learnjax.llama.convert_hf`: our pytree → Hugging Face `LlamaForCausalLM` tensors.

    Return {hf_name: np.ndarray} with the exact names `convert_hf` reads:
      model.embed_tokens.weight, model.norm.weight,
      model.layers.{i}.input_layernorm.weight, model.layers.{i}.post_attention_layernorm.weight,
      model.layers.{i}.self_attn.{q,k,v,o}_proj.weight, model.layers.{i}.mlp.{gate,up,down}_proj.weight,
      and lm_head.weight only when the model is not tied (cfg.tied is False; it lives in params["head"]).
    Our layers are stacked on a leading axis [L, ...] and use x @ W with W [in, out]; Hugging Face Linear
    weights are [out, in] — so un-stack and transpose (norm weights are vectors: no transpose).
    Keep each array's dtype. Round trip must be exact: convert_hf(export_llama_hf(p, cfg), cfg) == p.
    """
    raise NotImplementedError


def gpt_to_gpt2_state_dict(params: dict, n_layers: int) -> dict:
    """Module 16's GPT pytree → Hugging Face `GPT2LMHeadModel` tensor names (np.ndarray values).

    Ours: wte [V,D], wpe [T,D], lnf_g/lnf_b, and blocks[i] with ln1_g/ln1_b, w_qkv [D,3D]/b_qkv,
    w_o [D,D]/b_o, ln2_g/ln2_b, w_fc [D,4D]/b_fc, w_proj [4D,D]/b_proj.
    GPT-2's names: transformer.wte.weight, transformer.wpe.weight, transformer.ln_f.{weight,bias},
    transformer.h.{i}.ln_1.{weight,bias}, .attn.c_attn.{weight,bias}, .attn.c_proj.{weight,bias},
    .ln_2.{weight,bias}, .mlp.c_fc.{weight,bias}, .mlp.c_proj.{weight,bias}.
    GPT-2 uses `Conv1D` layers, whose weight is stored [in, out] — the same layout as our x @ W — and its
    fused c_attn splits q, k, v exactly like Module 16 does. So: rename, don't transpose. (The output head
    is tied to wte, so there is no lm_head entry.)
    """
    raise NotImplementedError


def bits_per_byte_delta(ppl_base: float, ppl_quant: float, tokens_per_byte: float) -> float:
    """How many extra bits per byte of text a quantized model needs, from two perplexities on the same text.

    Perplexity is per *token*; log2(ppl) is bits per token; times tokens per byte gives bits per byte
    (Module 21). Return bpb(quant) - bpb(base) (positive = the quant is worse).
    """
    raise NotImplementedError


def openai_chat_request(model: str, messages: list, max_tokens: int = 256, temperature: float = 0.0,
                        response_format: dict | None = None, tools: list | None = None,
                        stream: bool = False) -> dict:
    """JSON body for POST /v1/chat/completions (OpenAI-compatible; LM Studio, llama.cpp, vLLM all accept it).

    Always include model, messages, max_tokens, temperature, stream. Include response_format and tools only
    when given (not None).
    """
    raise NotImplementedError


def lmstudio_rest_request(model: str, user_input: str, max_output_tokens: int = 256,
                          temperature: float = 0.0, system_prompt: str | None = None) -> dict:
    """JSON body for LM Studio's native POST /api/v1/chat.

    Keys: model, input (the user text), max_output_tokens, temperature, and system_prompt only when given.
    """
    raise NotImplementedError


def anthropic_messages_request(model: str, messages: list, max_tokens: int = 256,
                               system: str | None = None, temperature: float = 0.0) -> dict:
    """JSON body for POST /v1/messages (Anthropic Messages API; LM Studio implements it too).

    Differences from the OpenAI shape: the system prompt is a top-level "system" field, not a message
    (include it only when given); max_tokens is required. Keys: model, max_tokens, messages, temperature
    [, system].
    """
    raise NotImplementedError


def assemble_tool_call_stream(chunks: list) -> list:
    """Reassemble streamed OpenAI-style tool calls into complete calls.

    `chunks` are parsed streaming events (dicts). Each may carry
    chunk["choices"][0]["delta"]["tool_calls"] = [{"index": i, "id": ..., "function": {"name": ...,
    "arguments": "<a fragment of a JSON string>"}}, ...]. The id and name arrive once (usually in the
    first fragment for that index); the arguments arrive in pieces that must be concatenated in order.
    Chunks without tool_calls (role, content, finish_reason) are ignored.
    Return [{"id": str, "name": str, "arguments": dict}] ordered by index, with arguments parsed as JSON.
    """
    raise NotImplementedError


def quant_recommendation(n_params: float, vram_gb: float, ctx: int, kv_bytes_per_token: float,
                         overhead_gb: float = 1.0) -> str | None:
    """The highest-quality GGUF type from GGUF_BITS that fits in VRAM, or None if even Q2_K doesn't.

    Needed bytes = weights (n_params * bits / 8) + KV cache (ctx * kv_bytes_per_token)
                   + overhead_gb * 2**30 (runtime buffers, compute scratch).
    Fits if needed <= vram_gb * 2**30. Try types from most to fewest bits.
    """
    raise NotImplementedError


def pick_quant(results: dict, budget_gb: float, tolerance: float = 0.02) -> str | None:
    """The smallest quant that fits the budget and scores within `tolerance` of the best score (§16).

    results maps a name ("IQ4_XS-task") to {"size_gb": float, "exact_match": float in [0, 1]}.
    best = the highest exact_match over *all* entries, whether or not they fit. A name qualifies if
    size_gb <= budget_gb and exact_match >= best - tolerance (absolute, so 0.02 is two percentage points).
    Return the qualifying name with the smallest size_gb; break ties by higher exact_match, then by name
    (alphabetical). Return None if nothing qualifies (including an empty dict).
    """
    raise NotImplementedError
