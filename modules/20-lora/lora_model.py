"""SmolLM2 with LoRA adapters: learnjax.llama's layer, copied here with adapter hooks.

The base weights come from `learnjax.llama.load` and never change. Adapters live in their own pytree,
stacked over layers like the base weights so one `lax.scan` runs everything:

    adapters = {"wq": {"A": [L, d_in, r], "B": [L, r, d_out]}, "wv": {...}}

Your `lora_init` / `lora_apply` (exercises.py) are passed in, so the lesson trains with *your* code.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from learnjax import llama
from learnjax.llama import rms_norm, rope


def init_adapters(key, cfg, r, targets, lora_init, shapes):
    """One (A, B) pair per target per layer, stacked to [L, ...]."""
    out = {}
    for name, k in zip(targets, jax.random.split(key, len(targets)), strict=True):
        d_in, d_out = shapes[name](cfg)
        A, B = zip(*[lora_init(kk, d_in, d_out, r) for kk in jax.random.split(k, cfg.n_layers)], strict=True)
        out[name] = {"A": jnp.stack(A), "B": jnp.stack(B)}
    return out


def _proj(h, lp, la, name, lora_apply, alpha, r):
    if name in la:
        return lora_apply(h, lp[name], la[name]["A"], la[name]["B"], alpha, r)
    return h @ lp[name]


def _layer(cfg, x, lp, la, positions, lora_apply, alpha, r):
    B, T, _ = x.shape
    H, KV, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

    def P(h, name):
        return _proj(h, lp, la, name, lora_apply, alpha, r)

    h = rms_norm(x, lp["attn_norm"], cfg.norm_eps)
    q = rope(P(h, "wq").reshape(B, T, H, hd), positions, cfg.rope_theta)
    k = rope(P(h, "wk").reshape(B, T, KV, hd), positions, cfg.rope_theta)
    v = P(h, "wv").reshape(B, T, KV, hd)
    rep = H // KV
    qg = q.reshape(B, T, KV, rep, hd)
    scores = jnp.einsum("btgrd,bsgd->bgrts", qg, k).astype(jnp.float32) / jnp.sqrt(hd)
    causal = jnp.tril(jnp.ones((T, T), bool))
    scores = jnp.where(causal, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    out = jnp.einsum("bgrts,bsgd->btgrd", weights, v).reshape(B, T, H * hd)
    x = x + P(out, "wo")
    h = rms_norm(x, lp["mlp_norm"], cfg.norm_eps)
    x = x + P(jax.nn.silu(P(h, "w_gate")) * P(h, "w_up"), "w_down")
    return x


def forward(params, adapters, tokens, cfg, lora_apply, alpha, r):
    """tokens [B, T] → logits [B, T, V] (float32). adapters may be {} (= the base model)."""
    B, T = tokens.shape
    positions = jnp.broadcast_to(jnp.arange(T), (B, T))
    x = params["embed"][tokens]

    def body(x, layer):
        lp, la = layer
        return _layer(cfg, x, lp, la, positions, lora_apply, alpha, r), None

    x, _ = jax.lax.scan(body, x, (params["layers"], adapters))
    x = rms_norm(x, params["final_norm"], cfg.norm_eps)
    head = params["embed"].T if cfg.tied else params["head"]
    return (x @ head).astype(jnp.float32)


def masked_loss(logits, tokens, mask):
    """Mean cross-entropy over answer positions only (logits at t predict token t+1)."""
    targets = jnp.roll(tokens, -1, axis=1)
    logp = jax.nn.log_softmax(logits, -1)
    nll = -jnp.take_along_axis(logp, targets[..., None], -1)[..., 0]
    return (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def make_trainer(cfg, lora_apply, alpha, r, lr=1e-3, steps=300, warmup=20):
    """Returns (optimizer, train_step). train_step differentiates the *adapters only*."""
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.0))

    def loss_fn(adapters, params, tokens, mask):
        return masked_loss(forward(params, adapters, tokens, cfg, lora_apply, alpha, r), tokens, mask)

    @jax.jit
    def train_step(adapters, opt_state, params, tokens, mask):
        loss, grads = jax.value_and_grad(loss_fn)(adapters, params, tokens, mask)   # grads: adapters only
        updates, opt_state = opt.update(grads, opt_state, adapters)
        return optax.apply_updates(adapters, updates), opt_state, loss

    return opt, train_step


def make_eval(cfg, lora_apply, alpha, r):
    @jax.jit
    def predict_next(params, adapters, tokens):
        return jnp.argmax(forward(params, adapters, tokens, cfg, lora_apply, alpha, r), -1)

    def exact_match(params, adapters, batch, task, chunk=16):
        preds = [np.asarray(predict_next(params, adapters, jnp.asarray(batch["tokens"][i:i + chunk])))
                 for i in range(0, len(batch["tokens"]), chunk)]
        pred = np.concatenate(preds)
        return task.teacher_forced_exact_match(pred, batch)

    return exact_match


def merge(params, adapters, merge_lora, alpha, r):
    """Fold every adapter into its base weight (vmapped over the layer axis)."""
    layers = dict(params["layers"])
    for name, ab in adapters.items():
        layers[name] = jax.vmap(lambda W, A, B: merge_lora(W, A, B, alpha, r))(layers[name], ab["A"], ab["B"])
    return {**params, "layers": layers}


def batches(key, data, batch_size):
    """Endless shuffled minibatches (index arrays) over a dataset of N rows."""
    n = len(data["tokens"])
    while True:
        key, sub = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(sub, n))
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            yield jnp.asarray(data["tokens"][idx]), jnp.asarray(data["loss_mask"][idx])


def generate_text(params, cfg, tok, text, task, n=60):
    """Greedy generation with the plain learnjax.llama decoder (works on base or merged weights)."""
    ids = tok.encode(task.chatml_prompt(text)).ids
    out = llama.generate(params, cfg, ids, n, jax.random.key(0), temperature=0.0,
                         stop_ids=(tok.token_to_id("<|im_end|>"),))
    return tok.decode([t for t in out if t != tok.token_to_id("<|im_end|>")])
