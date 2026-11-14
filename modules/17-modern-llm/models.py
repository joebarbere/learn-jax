"""Two tiny reference LLMs assembled from *your* Module 17 parts (passed in as `parts`).

* Llama-3-style:     RMSNorm · RoPE · grouped-query attention · SwiGLU
* DeepSeek-V3-style: RMSNorm · RoPE · Multi-head Latent Attention · DeepSeekMoE (shared + routed experts,
                     aux-loss-free bias balancing)

`parts` is your exercises module (or the reference solutions). The MLA and MoE layers below follow
deepseek_v3.py from github.com/Vaibhaav-Tiwari/deepseek-v3-jax (MIT), simplified (no multi-token
prediction, no sequence-wise balance loss). Params are plain nested dicts, as in that repo.
"""
from functools import partial

import jax
import jax.numpy as jnp
import optax

LLAMA = {"d_model": 128, "n_layers": 4, "n_heads": 4, "n_kv_heads": 1, "head_dim": 32, "d_ff": 352,
         "rope_theta": 10000.0}
DEEPSEEK = {"d_model": 128, "n_layers": 4, "n_dense": 1, "n_heads": 4, "q_rank": 64, "kv_rank": 32,
            "d_nope": 32, "d_rope": 16, "d_v": 32, "d_ff": 352, "n_shared": 1, "n_routed": 8, "top_k": 2,
            "d_expert": 96, "rope_theta": 10000.0}


def _normal(key, shape, std=0.02):
    return jax.random.normal(key, shape) * std


def _swiglu_params(key, d, h):
    k1, k2, k3 = jax.random.split(key, 3)
    return {"w1": _normal(k1, (d, h)), "w3": _normal(k2, (d, h)), "w2": _normal(k3, (h, d))}


# ------------------------------------------------------------------------------------------- Llama
def init_llama(key, vocab, cfg):
    d, H, Hkv, D = cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    keys = jax.random.split(key, cfg["n_layers"] + 2)
    layers = []
    for k in keys[:-2]:
        ka, kf = jax.random.split(k)
        kq, kk, kv, ko = jax.random.split(ka, 4)
        layers.append({
            "attn_norm": jnp.ones(d),
            "attn": {"wq": _normal(kq, (d, H * D)), "wk": _normal(kk, (d, Hkv * D)),
                     "wv": _normal(kv, (d, Hkv * D)), "wo": _normal(ko, (H * D, d))},
            "ffn_norm": jnp.ones(d),
            "ffn": _swiglu_params(kf, d, cfg["d_ff"]),
        })
    return {"embed": _normal(keys[-2], (vocab, d)), "layers": layers, "norm_f": jnp.ones(d),
            "head": _normal(keys[-1], (d, vocab))}


def llama_forward(parts, params, tokens, cfg):
    positions = jnp.arange(tokens.shape[1])
    x = params["embed"][tokens]
    for p in params["layers"]:
        a, _ = parts.gqa_attention(p["attn"], parts.rms_norm(x, p["attn_norm"]), positions, cfg)
        x = x + a
        x = x + parts.swiglu(p["ffn"], parts.rms_norm(x, p["ffn_norm"]))
    return parts.rms_norm(x, params["norm_f"]) @ params["head"], None


# ---------------------------------------------------------------------------------------- DeepSeek
def init_deepseek(key, vocab, cfg):
    d, H = cfg["d_model"], cfg["n_heads"]
    dn, dr, dv, E = cfg["d_nope"], cfg["d_rope"], cfg["d_v"], cfg["n_routed"]
    keys = jax.random.split(key, cfg["n_layers"] + 2)
    layers = []
    for i, k in enumerate(keys[:-2]):
        ka, kf = jax.random.split(k)
        k1, k2, k3, k4, k5 = jax.random.split(ka, 5)
        layer = {
            "attn_norm": jnp.ones(d),
            "attn": {"wq_a": _normal(k1, (d, cfg["q_rank"])), "q_norm": jnp.ones(cfg["q_rank"]),
                     "wq_b": _normal(k2, (cfg["q_rank"], H * (dn + dr))),
                     "wkv_a": _normal(k3, (d, cfg["kv_rank"] + dr)), "kv_norm": jnp.ones(cfg["kv_rank"]),
                     "wkv_b": _normal(k4, (cfg["kv_rank"], H * (dn + dv))), "wo": _normal(k5, (H * dv, d))},
            "ffn_norm": jnp.ones(d),
        }
        if i < cfg["n_dense"]:
            layer["ffn"] = _swiglu_params(kf, d, cfg["d_ff"])
        else:
            kr, ks, k1, k2, k3 = jax.random.split(kf, 5)
            h = cfg["d_expert"]
            layer["moe"] = {"router": _normal(kr, (d, E)),
                            "shared": _swiglu_params(ks, d, cfg["n_shared"] * h),
                            "experts": {"w1": _normal(k1, (E, d, h)), "w3": _normal(k2, (E, d, h)),
                                        "w2": _normal(k3, (E, h, d))}}
        layers.append(layer)
    return {"embed": _normal(keys[-2], (vocab, d)), "layers": layers, "norm_f": jnp.ones(d),
            "head": _normal(keys[-1], (d, vocab))}


def mla(parts, p, x, positions, cfg):
    """Multi-head Latent Attention: K and V are rebuilt from a small cached latent c (kv_rank wide) plus
    one RoPE key kr shared by every head. Training path only (no cache) to keep it short."""
    B, T, _ = x.shape
    H, dn, dr, dv = cfg["n_heads"], cfg["d_nope"], cfg["d_rope"], cfg["d_v"]
    theta = cfg["rope_theta"]
    q = (parts.rms_norm(x @ p["wq_a"], p["q_norm"]) @ p["wq_b"]).reshape(B, T, H, dn + dr)
    q = jnp.concatenate([q[..., :dn], parts.rope_rotate_half(q[..., dn:], positions, theta)], -1)
    kv_a = x @ p["wkv_a"]
    c = parts.rms_norm(kv_a[..., :cfg["kv_rank"]], p["kv_norm"])               # cached (kv_rank)
    kr = parts.rope_rotate_half(kv_a[..., cfg["kv_rank"]:], positions, theta)  # cached (d_rope)
    kv = (c @ p["wkv_b"]).reshape(B, T, H, dn + dv)
    k = jnp.concatenate([kv[..., :dn], jnp.broadcast_to(kr[:, :, None], (B, T, H, dr))], -1)
    v = kv[..., dn:]
    scores = jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32) / jnp.sqrt(dn + dr)
    causal = jnp.tril(jnp.ones((T, T), bool))
    attn = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), -1).astype(x.dtype)
    return jnp.einsum("bhts,bshd->bthd", attn, v).reshape(B, T, H * dv) @ p["wo"]


def moe(parts, p, bias, x, cfg):
    """DeepSeekMoE with dense compute: every expert runs on every token, unpicked experts get gate 0."""
    B, T, d = x.shape
    xf = x.reshape(B * T, d)
    scores = jax.nn.sigmoid(xf.astype(jnp.float32) @ p["router"].astype(jnp.float32))
    gates = parts.moe_route(scores, bias, cfg["top_k"]).astype(x.dtype)        # (N, E)
    w = p["experts"]
    h = jax.nn.silu(jnp.einsum("nd,edh->neh", xf, w["w1"])) * jnp.einsum("nd,edh->neh", xf, w["w3"])
    routed = jnp.einsum("ne,ned->nd", gates, jnp.einsum("neh,ehd->ned", h, w["w2"]))
    out = routed + parts.swiglu(p["shared"], xf)
    load = (gates > 0).mean(0) / cfg["top_k"]                                  # (E,) share of routed slots
    return out.reshape(B, T, d), load


def deepseek_forward(parts, params, tokens, cfg, biases):
    positions = jnp.arange(tokens.shape[1])
    x = params["embed"][tokens]
    loads = []
    for i, p in enumerate(params["layers"]):
        x = x + mla(parts, p["attn"], parts.rms_norm(x, p["attn_norm"]), positions, cfg)
        h = parts.rms_norm(x, p["ffn_norm"])
        if "moe" in p:
            f, load = moe(parts, p["moe"], biases[i - cfg["n_dense"]], h, cfg)
            loads.append(load)
        else:
            f = parts.swiglu(p["ffn"], h)
        x = x + f
    return parts.rms_norm(x, params["norm_f"]) @ params["head"], jnp.stack(loads)


# ---------------------------------------------------------------------------------------- counting
def count_params(params) -> int:
    return sum(int(a.size) for a in jax.tree.leaves(params))


def active_params(params, cfg) -> int:
    """Parameters touched per token: everything except the routed experts that weren't picked."""
    total = count_params(params)
    if "n_routed" not in cfg:
        return total
    routed = sum(count_params(p["moe"]["experts"]) for p in params["layers"] if "moe" in p)
    return int(total - routed * (1 - cfg["top_k"] / cfg["n_routed"]))


# ---------------------------------------------------------------------------------------- training
def get_batch(key, data, batch, seq_len):
    """B random windows of seq_len+1 tokens: vmap over start offsets of lax.dynamic_slice."""
    starts = jax.random.randint(key, (batch,), 0, data.shape[0] - seq_len - 1)
    win = jax.vmap(lambda s: jax.lax.dynamic_slice(data, (s,), (seq_len + 1,)))(starts)
    return win[:, :-1], win[:, 1:]


def cross_entropy(logits, targets):
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    return -jnp.take_along_axis(logp, targets[..., None], -1).mean()


def make_trainer(forward, optimizer, batch=32, seq_len=128, gamma=0.0, update_bias=None):
    """Returns train_chunk(params, opt_state, biases, data, key, n_steps) running n_steps under one jit
    (lax.scan, as in Module 10), and eval_loss(params, biases, data, key). `forward(params, tokens, biases)`
    returns (logits, loads-or-None). With update_bias (your Module 17 function) and gamma > 0, DeepSeek's
    bias balancing runs after every step."""

    def loss_fn(params, biases, x, y):
        logits, loads = forward(params, x, biases)
        return cross_entropy(logits, y), loads

    def step(carry, key):
        params, opt_state, biases, data = carry
        x, y = get_batch(key, data, batch, seq_len)
        (loss, loads), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, biases, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if loads is not None and update_bias is not None:
            biases = update_bias(biases, loads, gamma)
        return (params, opt_state, biases, data), (loss, loads if loads is not None else jnp.zeros(()))

    @partial(jax.jit, static_argnames="n_steps")
    def train_chunk(params, opt_state, biases, data, key, n_steps):
        (params, opt_state, biases, _), (losses, loads) = jax.lax.scan(
            step, (params, opt_state, biases, data), jax.random.split(key, n_steps))
        return params, opt_state, biases, losses, loads

    @jax.jit
    def eval_loss(params, biases, data, key):
        def one(k):
            x, y = get_batch(k, data, batch, seq_len)
            return loss_fn(params, biases, x, y)[0]
        return jax.vmap(one)(jax.random.split(key, 8)).mean()

    return train_chunk, eval_loss
