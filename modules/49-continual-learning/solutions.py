"""Reference solutions for Module 49. Try exercises.py first."""
import jax
import jax.numpy as jnp
import numpy as np


def forgetting_metrics(R):
    R = np.asarray(R, float)
    T = R.shape[0]
    out = {"avg": float(R[-1].mean())}
    if T == 1:
        return out | {"bwt": 0.0, "forgetting": 0.0}
    out["bwt"] = float(np.mean([R[-1, j] - R[j, j] for j in range(T - 1)]))
    out["forgetting"] = float(np.mean([R[:-1, j].max() - R[-1, j] for j in range(T - 1)]))
    return out


def ewc_penalty(params, anchor, fisher):
    terms = jax.tree.map(lambda p, a, f: jnp.sum(f.astype(jnp.float32) * (p.astype(jnp.float32)
                                                                          - a.astype(jnp.float32)) ** 2),
                         params, anchor, fisher)
    return jnp.sum(jnp.stack(jax.tree.leaves(terms))).astype(jnp.float32)


def expert_out(h, A, B, gate):
    z = jax.nn.silu(jnp.einsum("btd,edr->bter", h, A.astype(h.dtype)))
    return jnp.einsum("bter,erd,e->btd", z, B.astype(h.dtype), gate.astype(h.dtype))


def token_kl(teacher_logits, student_logits, kind="forward"):
    lt = jax.nn.log_softmax(teacher_logits.astype(jnp.float32), -1)
    ls = jax.nn.log_softmax(student_logits.astype(jnp.float32), -1)
    if kind == "forward":
        return jnp.sum(jnp.exp(lt) * (lt - ls), -1)
    if kind == "reverse":
        return jnp.sum(jnp.exp(ls) * (ls - lt), -1)
    lm = jnp.logaddexp(lt, ls) - jnp.log(2.0)
    return 0.5 * jnp.sum(jnp.exp(lt) * (lt - lm), -1) + 0.5 * jnp.sum(jnp.exp(ls) * (ls - lm), -1)


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def levenshtein_reward(pred, target, z0):
    n = max(len(pred), len(target))
    if n == 0:
        return 1.0
    z = _edit_distance(list(pred), list(target))
    return float(1 - z / n) if z <= z0 else 0.0


def select_dreams(scores, k, b, seed=0):
    scores = np.asarray(scores, float)
    n = len(scores)
    if k + b >= n:
        return list(range(n))
    top = np.argsort(-scores, kind="stable")[:k]
    rest = np.setdiff1d(np.arange(n), top)
    extra = np.random.default_rng(seed).choice(rest, b, replace=False) if b else np.array([], int)
    return sorted(int(i) for i in np.concatenate([top, extra]))
