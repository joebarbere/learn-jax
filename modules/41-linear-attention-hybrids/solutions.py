"""Module 41 reference solutions. Same signatures as exercises.py."""
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------- memory
def memory_costs(n_layers: int, attn_every: int, n_heads: int, d_head: int, context: int,
                 kv_bytes: int = 2, state_bytes: int = 4) -> dict:
    n_attn = 0 if attn_every == 0 else n_layers // attn_every
    n_lin = n_layers - n_attn
    kv = 2 * n_attn * n_heads * d_head * context * kv_bytes
    state = n_lin * n_heads * d_head * d_head * state_bytes
    return {"attn_layers": n_attn, "linear_layers": n_lin, "kv_cache_bytes": kv, "state_bytes": state,
            "total_bytes": kv + state}


# ---------------------------------------------------------------- linear attention, two ways
def causal_linear_attention(q, k, v):
    T = q.shape[-2]
    mask = jnp.tril(jnp.ones((T, T), bool))
    scores = jnp.einsum("...td,...sd->...ts", q, k)
    return jnp.einsum("...ts,...se->...te", jnp.where(mask, scores, 0.0), v)


def linear_attention_scan(q, k, v, log_a=None):
    if log_a is None:
        log_a = jnp.zeros(q.shape[:-1], q.dtype)

    def step(S, inp):
        qt, kt, vt, lat = inp
        S = jnp.exp(lat)[..., None, None] * S + kt[..., :, None] * vt[..., None, :]
        return S, jnp.einsum("...d,...de->...e", qt, S)

    S0 = jnp.zeros(q.shape[:-2] + (q.shape[-1], v.shape[-1]), q.dtype)
    tmajor = lambda x: jnp.moveaxis(x, -2, 0) if x.ndim == q.ndim else jnp.moveaxis(x, -1, 0)
    _, out = jax.lax.scan(step, S0, (tmajor(q), tmajor(k), tmajor(v), tmajor(log_a)))
    return jnp.moveaxis(out, 0, -2)


# ---------------------------------------------------------------- decay: Mamba-2's chunked algorithm
def segsum(log_a):
    T = log_a.shape[-1]
    cs = jnp.cumsum(log_a, -1)
    diff = cs[..., :, None] - cs[..., None, :]                      # Σ_{k ≤ i} − Σ_{k ≤ j} = Σ_{j < k ≤ i}
    return jnp.where(jnp.tril(jnp.ones((T, T), bool)), diff, -jnp.inf)


def chunked_linear_attention(q, k, v, log_a, chunk: int):
    *lead, T, dk = q.shape
    dv = v.shape[-1]
    n = T // chunk
    qc, kc = q.reshape(*lead, n, chunk, dk), k.reshape(*lead, n, chunk, dk)
    vc, ac = v.reshape(*lead, n, chunk, dv), log_a.reshape(*lead, n, chunk)
    cs = jnp.cumsum(ac, -1)                                          # decay from the chunk start to i
    # 1. inside each chunk: masked, decayed attention — a matmul
    att = jnp.einsum("...cid,...cjd->...cij", qc, kc) * jnp.exp(segsum(ac))
    y = jnp.einsum("...cij,...cje->...cie", att, vc)
    # 2. each chunk's own contribution to the state at its end
    to_end = jnp.exp(cs[..., -1:] - cs)
    states = jnp.einsum("...cj,...cjd,...cje->...cde", to_end, kc, vc)
    # 3. pass states between chunks: S_in[c] = a_chunk[c-1] S_in[c-1] + states[c-1]
    total = cs[..., -1]

    def carry(S, inp):
        st, tot = inp
        return jnp.exp(tot)[..., None, None] * S + st, S

    S0 = jnp.zeros((*lead, dk, dv), q.dtype)
    _, s_in = jax.lax.scan(carry, S0, (jnp.moveaxis(states, -3, 0), jnp.moveaxis(total, -1, 0)))
    s_in = jnp.moveaxis(s_in, 0, -3)                                 # (..., n, dk, dv): state entering chunk c
    # 4. what each position reads from the state that came in
    y = y + jnp.einsum("...ci,...cid,...cde->...cie", jnp.exp(cs), qc, s_in)
    return y.reshape(*lead, T, dv)


# ---------------------------------------------------------------- the delta rule
def gated_delta_scan(q, k, v, beta, log_a=None):
    if log_a is None:
        log_a = jnp.zeros(beta.shape, q.dtype)

    def step(S, inp):
        qt, kt, vt, bt, lat = inp
        S = jnp.exp(lat)[..., None, None] * S                          # forget a little
        pred = jnp.einsum("...d,...de->...e", kt, S)                   # what the memory says for this key
        S = S + bt[..., None, None] * kt[..., :, None] * (vt - pred)[..., None, :]   # write the error
        return S, jnp.einsum("...d,...de->...e", qt, S)

    S0 = jnp.zeros(q.shape[:-2] + (q.shape[-1], v.shape[-1]), q.dtype)
    t = lambda x: jnp.moveaxis(x, -2 if x.ndim == q.ndim else -1, 0)
    _, out = jax.lax.scan(step, S0, (t(q), t(k), t(v), t(beta), t(log_a)))
    return jnp.moveaxis(out, 0, -2)
