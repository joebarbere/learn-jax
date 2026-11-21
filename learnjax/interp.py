"""Reading and editing the inside of a `learnjax.llama` model: residual-stream capture, edits at a layer, lenses,
and sparse autoencoders. Used by Module 47 (reading activations) and Module 48 (tracing a circuit).

Conventions. The residual stream after block ℓ is h_ℓ (ℓ = 1..L); h_0 is the token embedding. `forward` can
collect every h_ℓ as one stacked array (L+1, B, T, d) and can apply `edit(ℓ, x) -> x` to each h_ℓ as it is
produced, so the layers above see the edited stream. ℓ is a traced integer inside `lax.scan`, so edits use
`jnp.where(ℓ == k, …)` rather than Python `if`.

    from learnjax import interp, llama
    params, cfg, tok = llama.load(llama.SMOLLM_135M)
    out = interp.forward(params, cfg, tokens, collect=True)        # out["logits"], out["resid"]
    steer = lambda l, x: jnp.where(l == 12, x + 4.0 * u, x)         # add a direction at layer 12
    logits = interp.forward(params, cfg, tokens, edit=steer)["logits"]

The SAE here is the TopK autoencoder of Gao et al. 2024 (arXiv 2406.04093) with their AuxK loss for dead latents,
and the older ReLU + L1 autoencoder (Bricken et al. 2023) for comparison.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from learnjax import llama


# ---------------------------------------------------------------- the residual stream
def unembed(params, cfg, h):
    """What the model would predict if h were the last residual state: final RMSNorm, then the output matrix.
    This is the logit lens when h comes from a middle layer. Returns float32 logits (..., vocab)."""
    x = llama.rms_norm(h, params["final_norm"], cfg.norm_eps)
    head = params["embed"].T if cfg.tied else params["head"]
    return (x @ head).astype(jnp.float32)


def forward(params, cfg, tokens, edit=None, edit_mid=None, collect=(), upto: int | None = None):
    """Run the model on tokens (B, T), optionally editing and/or collecting the residual stream.

    edit:     fn(layer, x) -> x applied to h_0 … h_upto, the stream after each block (layer is a traced int32).
    edit_mid: fn(layer, x) -> x applied inside block `layer` (1-based) after attention, before the MLP.
    collect:  names to return, stacked over layers: "resid" = h_0 … h_upto, shape (upto+1, B, T, d);
              "mid" = the stream between attention and MLP, (upto, B, T, d); "mlp_out" = what each MLP adds.
              collect=True means ("resid",).
    upto:     stop after block `upto` (no logits are computed then). Default: all L blocks, plus "logits".
    """
    if collect is True:
        collect = ("resid",)
    B, T = tokens.shape
    positions = jnp.broadcast_to(jnp.arange(T), (B, T))
    n = cfg.n_layers if upto is None else upto
    layers = params["layers"] if upto is None else jax.tree.map(lambda a: a[:upto], params["layers"])
    x = params["embed"][tokens]
    if edit is not None:
        x = edit(jnp.int32(0), x)

    def body(x, inp):
        lp, i = inp
        x = x + llama.attn_block(cfg, x, lp, positions)[0]
        if edit_mid is not None:
            x = edit_mid(i + 1, x)
        mid = x
        m = llama.mlp_block(cfg, x, lp)
        x = x + m
        if edit is not None:
            x = edit(i + 1, x)
        keep = {"resid": x, "mid": mid, "mlp_out": m}
        return x, {k: keep[k] for k in collect}

    x_last, ys = jax.lax.scan(body, x, (layers, jnp.arange(n, dtype=jnp.int32)))
    out = dict(ys)
    if "resid" in collect:
        out["resid"] = jnp.concatenate([x[None], ys["resid"]])
    if upto is None:
        out["logits"] = unembed(params, cfg, x_last)
    return out


def at_layer(layer: int, fn):
    """An edit that applies fn(x) at one layer only and leaves the others alone."""
    return lambda i, x: jnp.where(i == layer, fn(x).astype(x.dtype), x)


def next_token_loss(logits, tokens):
    """Mean cross-entropy of predicting tokens[:, 1:] from logits[:, :-1] (nats)."""
    lp = jax.nn.log_softmax(logits[:, :-1])
    return -jnp.take_along_axis(lp, tokens[:, 1:, None], -1).mean()


def kl(p_logits, q_logits):
    """KL(p ‖ q) per position, in nats; both are logits over the last axis."""
    lp, lq = jax.nn.log_softmax(p_logits), jax.nn.log_softmax(q_logits)
    return (jnp.exp(lp) * (lp - lq)).sum(-1)


# ---------------------------------------------------------------- tuned lens (Belrose et al. 2023)
def init_tuned_lens(cfg):
    """One affine translator per layer h_0 … h_{L-1}: h ↦ h + h A_ℓ + b_ℓ, starting at the identity (A = b = 0)."""
    L, d = cfg.n_layers, cfg.d_model
    return {"A": jnp.zeros((L, d, d), jnp.float32), "b": jnp.zeros((L, d), jnp.float32)}


def tuned_lens_logits(params, cfg, lens, layer, h):
    hf = h.astype(jnp.float32)
    return unembed(params, cfg, (hf + hf @ lens["A"][layer] + lens["b"][layer]).astype(h.dtype))


# ---------------------------------------------------------------- sparse autoencoders
def init_sae(key, d: int, n_latents: int, scale: float = 1.0):
    """Decoder rows are unit vectors; the encoder starts as the decoder's transpose (Gao et al.)."""
    W_dec = jax.random.normal(key, (n_latents, d), jnp.float32)
    W_dec = W_dec / jnp.linalg.norm(W_dec, axis=1, keepdims=True) * scale
    return {"W_enc": W_dec.T, "b_enc": jnp.zeros(n_latents, jnp.float32), "W_dec": W_dec,
            "b_dec": jnp.zeros(d, jnp.float32)}


def sae_pre(sae, x):
    return (x - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]


def topk_relu(pre, k: int):
    """Keep each row's k largest values (after a ReLU), zero the rest. A threshold mask rather than a scatter of
    top_k's indices: same result (barring exact ties), 15% faster on the RX 7600 XT, and gradients flow as usual."""
    kth = jax.lax.top_k(jax.lax.stop_gradient(pre), k)[0][..., -1:]
    return jnp.where(pre >= kth, jax.nn.relu(pre), 0.0)


def sae_encode(sae, x, kind: str, k: int = 32):
    pre = sae_pre(sae, x)
    return topk_relu(pre, k) if kind == "topk" else jax.nn.relu(pre)


def sae_decode(sae, z):
    return z @ sae["W_dec"] + sae["b_dec"]


def sae_loss(sae, x, kind: str, k: int, l1: float, dead, k_aux: int = 256):
    """x: (N, d) float32 activations, already scaled to mean squared norm 1 (see `norm_scale`).
    TopK: MSE + AuxK (the residual error reconstructed by the k_aux largest *dead* latents, weight 1/32).
    ReLU: MSE + l1 · Σ z_i ‖W_dec,i‖."""
    pre = sae_pre(sae, x)
    z = topk_relu(pre, k) if kind == "topk" else jax.nn.relu(pre)
    x_hat = sae_decode(sae, z)
    mse = ((x - x_hat) ** 2).sum(-1).mean()
    aux = jnp.float32(0)
    if kind == "topk":
        dead_pre = jnp.where(dead, pre, -1e9)
        z_aux = jnp.where(dead, topk_relu(dead_pre, min(k_aux, pre.shape[-1])), 0.0)
        err_hat = z_aux @ sae["W_dec"]
        aux = ((jax.lax.stop_gradient(x - x_hat) - err_hat) ** 2).sum(-1).mean()
        aux = jnp.where(dead.any(), aux, 0.0) / 32
        total = mse + aux
    else:
        total = mse + l1 * (z * jnp.linalg.norm(sae["W_dec"], axis=1)).sum(-1).mean()
    return total, {"mse": mse, "aux": aux, "l0": (z > 0).sum(-1).mean(), "fired": (z > 0).any(0)}


def renorm_decoder(sae):
    """Keep decoder rows at unit length (TopK SAEs; it stops the encoder from shrinking codes)."""
    W = sae["W_dec"]
    return dict(sae, W_dec=W / jnp.linalg.norm(W, axis=1, keepdims=True))


def norm_scale(acts):
    """Scalar s with mean ‖s·x‖² = 1, so SAE losses and l1 coefficients don't depend on the layer's scale."""
    acts = jnp.asarray(acts, jnp.float32)
    return 1.0 / jnp.sqrt((acts**2).sum(-1).mean())


def fvu(x, x_hat):
    """Fraction of variance unexplained: Σ‖x − x̂‖² / Σ‖x − mean(x)‖²."""
    return ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
