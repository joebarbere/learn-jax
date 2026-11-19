"""Module 42 reference solutions. Same signatures as exercises.py."""
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------- budgets
def training_cost(n_params: float, n_tokens: float, tflops: float, mfu: float = 0.4) -> dict:
    flops = 6.0 * n_params * n_tokens
    return {"flops": flops, "hours": flops / (tflops * 1e12 * mfu) / 3600, "tokens_per_param": n_tokens / n_params,
            "chinchilla_tokens": 20.0 * n_params}


# ---------------------------------------------------------------- distillation
def kd_loss(student_logits, top_ids, top_logp, T: float = 1.0):
    p = jax.nn.softmax(top_logp / T, -1)                                   # teacher, renormalized over its top k
    logq = jax.nn.log_softmax(student_logits.astype(jnp.float32) / T, -1)
    logq_k = jnp.take_along_axis(logq, top_ids, -1)                        # student, full-vocabulary normalization
    logp = jnp.log(jnp.clip(p, 1e-30))
    return (T * T) * jnp.mean(jnp.sum(p * (logp - logq_k), -1))


# ---------------------------------------------------------------- pruning
def ffn_importance(acts):
    return jnp.mean(jnp.abs(acts.astype(jnp.float32)).reshape(-1, acts.shape[-1]), 0)


def prune_ffn(layers, keep):
    cols = keep[:, None, :]                                                # (L, 1, k): columns of (L, D, F)
    rows = keep[:, :, None]                                                # (L, k, 1): rows of (L, F, D)
    return {**layers, "w_gate": jnp.take_along_axis(layers["w_gate"], cols, 2),
            "w_up": jnp.take_along_axis(layers["w_up"], cols, 2),
            "w_down": jnp.take_along_axis(layers["w_down"], rows, 1)}


def layer_importance(h_in, h_out):
    a = h_in.astype(jnp.float32).reshape(-1, h_in.shape[-1])
    b = h_out.astype(jnp.float32).reshape(-1, h_out.shape[-1])
    cos = jnp.sum(a * b, -1) / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-8)
    return 1.0 - jnp.mean(cos)


# ---------------------------------------------------------------- Muon
def newton_schulz(G, steps: int = 5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(jnp.float32)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    X = X / (jnp.linalg.norm(X) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X
