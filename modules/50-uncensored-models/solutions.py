"""Reference solutions for Module 50. Try exercises.py first."""
import jax
import jax.numpy as jnp


def unwanted(text, phrases):
    low = text.lower()
    return any(p.lower() in low for p in phrases)


def response_mask(lengths, starts, T):
    lengths, starts = jnp.asarray(lengths), jnp.asarray(starts)
    first = T - lengths + starts                    # column of the first response token
    j = jnp.arange(T - 1)
    return (j[None, :] + 1) >= first[:, None]


def refusal_score(logits, token_ids):
    p = jax.nn.softmax(logits.astype(jnp.float32), -1)
    pr = p[..., jnp.asarray(token_ids)].sum(-1)
    return jnp.log(pr + 1e-8) - jnp.log(1 - pr + 1e-8)


def diff_in_means(harmful, harmless):
    return harmful.astype(jnp.float32).mean(1) - harmless.astype(jnp.float32).mean(1)


def ablate_dir(x, u):
    u = u.astype(jnp.float32)
    u = u / jnp.linalg.norm(u)
    xf = x.astype(jnp.float32)
    return (xf - (xf @ u)[..., None] * u).astype(x.dtype)


def orthogonalize(W, u):
    u = u.astype(jnp.float32)
    u = u / jnp.linalg.norm(u)
    Wf = W.astype(jnp.float32)
    return (Wf - (Wf @ u)[:, None] * u[None, :]).astype(W.dtype)
