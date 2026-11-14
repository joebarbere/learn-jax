"""A small, readable Llama-architecture model in pure JAX that runs real pretrained weights.

It is the reference the Part 5 modules (18 inference, 20 LoRA, 21 evaluation) build on: the same parts
you wrote in Module 17 (RMSNorm, RoPE, grouped-query attention, SwiGLU), loaded with Hugging Face
`LlamaForCausalLM` weights (default: SmolLM2, Apache-2.0), with a preallocated KV cache for generation.

    from learnjax import llama
    params, cfg, tok = llama.load("HuggingFaceTB/SmolLM2-360M-Instruct")
    ids = llama.generate(params, cfg, tok.encode("def fib(n):").ids, 64, jax.random.key(0))

Layers are stacked along a leading axis and run with `lax.scan`, so a 32-layer model compiles once per
shape instead of 32 times. Checked against Hugging Face transformers by tools/check_llama.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

SMOLLM_135M = "HuggingFaceTB/SmolLM2-135M"
SMOLLM_360M = "HuggingFaceTB/SmolLM2-360M-Instruct"


@jax.tree_util.register_static
@dataclass(frozen=True)
class Config:
    vocab: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tied: bool = True
    max_seq: int = 2048
    dtype: str = "bfloat16"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_hf(cls, hf: dict, **overrides) -> Config:
        return cls(vocab=hf["vocab_size"], d_model=hf["hidden_size"], n_layers=hf["num_hidden_layers"],
                   n_heads=hf["num_attention_heads"], n_kv_heads=hf["num_key_value_heads"],
                   d_ff=hf["intermediate_size"], rope_theta=float(hf.get("rope_theta", 10000.0)),
                   norm_eps=hf.get("rms_norm_eps", 1e-5), tied=hf.get("tie_word_embeddings", False),
                   **overrides)


# ---------------------------------------------------------------- loading

def download(repo_id: str) -> Path:
    """Cache first (no network once downloaded), then download. HF timeouts are set in the image."""
    from huggingface_hub import snapshot_download
    patterns = ["*.json", "*.safetensors", "*.txt"]
    try:
        return Path(snapshot_download(repo_id, allow_patterns=patterns, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached yet
        return Path(snapshot_download(repo_id, allow_patterns=patterns))


def load(repo_id: str = SMOLLM_360M, dtype: str = "bfloat16", max_seq: int = 2048):
    """Download (once) and load a Llama-architecture checkpoint. Returns (params, cfg, tokenizer)."""
    from safetensors.flax import load_file
    from tokenizers import Tokenizer

    path = download(repo_id)
    cfg = Config.from_hf(json.loads((path / "config.json").read_text()), dtype=dtype, max_seq=max_seq)
    raw = {}
    for f in sorted(path.glob("*.safetensors")):
        raw |= load_file(str(f))
    params = convert_hf(raw, cfg)
    return params, cfg, Tokenizer.from_file(str(path / "tokenizer.json"))


def convert_hf(raw: dict, cfg: Config) -> dict:
    """Hugging Face names/layouts → our pytree. HF Linear weights are [out, in]; we use x @ W with [in, out]."""
    dt = jnp.dtype(cfg.dtype)

    def w(name):
        return raw[name].astype(dt)

    def stack(fmt, transpose=True):
        return jnp.stack([(w(fmt.format(i)).T if transpose else w(fmt.format(i))) for i in range(cfg.n_layers)])

    p = "model.layers.{}."
    params = {
        "embed": w("model.embed_tokens.weight"),                       # [V, D]
        "final_norm": w("model.norm.weight"),
        "layers": {
            "attn_norm": stack(p + "input_layernorm.weight", transpose=False),
            "wq": stack(p + "self_attn.q_proj.weight"),               # [L, D, H*hd]
            "wk": stack(p + "self_attn.k_proj.weight"),               # [L, D, KV*hd]
            "wv": stack(p + "self_attn.v_proj.weight"),
            "wo": stack(p + "self_attn.o_proj.weight"),               # [L, H*hd, D]
            "mlp_norm": stack(p + "post_attention_layernorm.weight", transpose=False),
            "w_gate": stack(p + "mlp.gate_proj.weight"),              # [L, D, F]
            "w_up": stack(p + "mlp.up_proj.weight"),
            "w_down": stack(p + "mlp.down_proj.weight"),              # [L, F, D]
        },
    }
    if not cfg.tied:
        params["head"] = w("lm_head.weight").T
    return params


def count_params(params) -> int:
    return sum(x.size for x in jax.tree.leaves(params))


# ---------------------------------------------------------------- the model

def rms_norm(x, w, eps):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps)).astype(x.dtype) * w


def rope(x, positions, theta):
    """Rotate-half RoPE (the Hugging Face Llama convention). x: [B, T, H, hd], positions: [B, T]."""
    hd = x.shape[-1]
    freqs = 1.0 / theta ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd)
    ang = positions[..., None].astype(jnp.float32) * freqs                 # [B, T, hd/2]
    cos = jnp.cos(jnp.concatenate([ang, ang], -1))[:, :, None, :]
    sin = jnp.sin(jnp.concatenate([ang, ang], -1))[:, :, None, :]
    x32 = x.astype(jnp.float32)
    x1, x2 = jnp.split(x32, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], -1)
    return (x32 * cos + rotated * sin).astype(x.dtype)


def init_cache(cfg: Config, batch: int, max_len: int | None = None):
    """Preallocated KV cache: k, v of shape [L, B, S, KV, hd] (fixed shape → one compile for decoding)."""
    s = max_len or cfg.max_seq
    shape = (cfg.n_layers, batch, s, cfg.n_kv_heads, cfg.head_dim)
    return {"k": jnp.zeros(shape, cfg.dtype), "v": jnp.zeros(shape, cfg.dtype)}


def _layer(cfg: Config, x, lp, positions, cache_kv, pos):
    B, T, _ = x.shape
    H, KV, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    h = rms_norm(x, lp["attn_norm"], cfg.norm_eps)
    q = rope((h @ lp["wq"]).reshape(B, T, H, hd), positions, cfg.rope_theta)
    k = rope((h @ lp["wk"]).reshape(B, T, KV, hd), positions, cfg.rope_theta)
    v = (h @ lp["wv"]).reshape(B, T, KV, hd)
    if cache_kv is None:                          # full forward: keys = this chunk
        keys, vals, k_pos = k, v, positions
    else:                                         # write this chunk at `pos`, attend over the whole cache
        ck, cv = cache_kv
        ck = jax.lax.dynamic_update_slice(ck, k, (0, pos, 0, 0))
        cv = jax.lax.dynamic_update_slice(cv, v, (0, pos, 0, 0))
        cache_kv = (ck, cv)
        keys, vals = ck, cv
        k_pos = jnp.broadcast_to(jnp.arange(ck.shape[1]), (B, ck.shape[1]))
    rep = H // KV                                  # grouped-query attention: share each K/V head
    qg = q.reshape(B, T, KV, rep, hd)
    scores = jnp.einsum("btgrd,bsgd->bgrts", qg, keys).astype(jnp.float32) / jnp.sqrt(hd)
    mask = k_pos[:, None, None, None, :] <= positions[:, None, None, :, None]   # causal (and hides empty cache)
    scores = jnp.where(mask, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    out = jnp.einsum("bgrts,bsgd->btgrd", weights, vals).reshape(B, T, H * hd)
    x = x + out @ lp["wo"]
    h = rms_norm(x, lp["mlp_norm"], cfg.norm_eps)
    x = x + (jax.nn.silu(h @ lp["w_gate"]) * (h @ lp["w_up"])) @ lp["w_down"]
    return x, cache_kv


def forward(params, tokens, cfg: Config, cache=None, pos=0):
    """tokens [B, T] → (logits [B, T, V] float32, cache). With a cache, `pos` is where this chunk starts."""
    B, T = tokens.shape
    positions = jnp.broadcast_to(pos + jnp.arange(T), (B, T))
    x = params["embed"][tokens]

    def body(x, layer):
        lp, ckv = layer
        x, ckv = _layer(cfg, x, lp, positions, ckv, pos)
        return x, ckv

    if cache is None:
        x, _ = jax.lax.scan(lambda x, lp: (_layer(cfg, x, lp, positions, None, pos)[0], None), x, params["layers"])
        new_cache = None
    else:
        x, (k, v) = jax.lax.scan(body, x, (params["layers"], (cache["k"], cache["v"])))
        new_cache = {"k": k, "v": v}
    x = rms_norm(x, params["final_norm"], cfg.norm_eps)
    head = params["embed"].T if cfg.tied else params["head"]
    return (x @ head).astype(jnp.float32), new_cache


forward_jit = jax.jit(forward, static_argnames=("cfg",))


# ---------------------------------------------------------------- generation

@jax.jit
def _sample(key, logits, temperature, top_k):
    """logits [B, V]; temperature 0 → greedy. top_k=0 → no truncation."""
    greedy = jnp.argmax(logits, -1)
    kth = jnp.sort(logits, -1)[:, ::-1][jnp.arange(logits.shape[0]), jnp.maximum(top_k - 1, 0)][:, None]
    masked = jnp.where((top_k > 0) & (logits < kth), -jnp.inf, logits)
    sampled = jax.random.categorical(key, masked / jnp.maximum(temperature, 1e-6))
    return jnp.where(temperature > 0, sampled, greedy)


def generate(params, cfg: Config, prompt_ids, n_new: int, key, temperature: float = 0.7, top_k: int = 50,
             stop_ids=()):
    """Prefill the prompt once, then decode one token at a time from the KV cache. Returns new token ids."""
    prompt = jnp.asarray(prompt_ids, jnp.int32)[None, :]
    T = prompt.shape[1]
    cache = init_cache(cfg, 1, max_len=T + n_new)
    logits, cache = forward_jit(params, prompt, cfg, cache, 0)
    nxt = logits[:, -1]
    out = []
    for i in range(n_new):
        key, sub = jax.random.split(key)
        tok = _sample(sub, nxt, temperature, top_k)
        t = int(tok[0])
        out.append(t)
        if t in stop_ids:
            break
        logits, cache = forward_jit(params, tok[:, None], cfg, cache, T + i)
        nxt = logits[:, -1]
    return out


def chat_prompt(tokenizer, user: str, system: str | None = None) -> list[int]:
    """SmolLM2-Instruct chat template (ChatML): <|im_start|>role\\n...<|im_end|>."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    return tokenizer.encode("".join(parts)).ids
