"""Module 31 reference solutions. Try the exercises first — `make test N=31` grades yours."""
import jax
import jax.numpy as jnp


def rnn_cell(params, h, x):
    return jnp.tanh(x @ params["Wx"] + h @ params["Wh"] + params["b"])


def lstm_cell(params, carry, x):
    h, c = carry
    z = x @ params["Wx"] + h @ params["Wh"] + params["b"]
    zi, zf, zg, zo = jnp.split(z, 4, axis=-1)
    i, f, g, o = jax.nn.sigmoid(zi), jax.nn.sigmoid(zf), jnp.tanh(zg), jax.nn.sigmoid(zo)
    c_new = f * c + i * g
    return o * jnp.tanh(c_new), c_new


def gru_cell(params, h, x):
    gx = x @ params["Wx"] + params["b"]
    gh = h @ params["Wh"]
    xz, xr, xn = jnp.split(gx, 3, axis=-1)
    hz, hr, hn = jnp.split(gh, 3, axis=-1)
    z = jax.nn.sigmoid(xz + hz)
    r = jax.nn.sigmoid(xr + hr)
    n = jnp.tanh(xn + r * hn)
    return (1 - z) * n + z * h


def run_rnn(cell, params, carry0, xs):
    def step(carry, x):
        carry = cell(params, carry, x)
        h = carry[0] if isinstance(carry, tuple) else carry
        return carry, h

    return jax.lax.scan(step, carry0, xs)


def grad_norm_by_distance(cell, params, carry0, xs, w_out):
    def readout(xs_):
        final, _ = run_rnn(cell, params, carry0, xs_)
        h = final[0] if isinstance(final, tuple) else final
        return jnp.sum(h @ w_out)

    g = jax.grad(readout)(xs)                               # (T, ..., d_in)
    return jnp.sqrt(jnp.sum(g.reshape(g.shape[0], -1) ** 2, axis=1))


def additive_attention(params, query, keys, values, mask=None):
    q = query @ params["Wq"]                                 # (B, A)
    k = keys @ params["Wk"]                                  # (B, S, A)
    score = jnp.tanh(q[:, None, :] + k) @ params["v"]       # (B, S)
    if mask is not None:
        score = jnp.where(mask, score, -jnp.inf)
    weights = jax.nn.softmax(score, axis=-1)
    context = jnp.einsum("bs,bsh->bh", weights, values)
    return context, weights


def greedy_decode(step_fn, state, start_tokens, n_steps: int):
    def step(carry, _):
        st, tok = carry
        logits, st = step_fn(st, tok)
        nxt = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return (st, nxt), nxt

    _, toks = jax.lax.scan(step, (state, start_tokens.astype(jnp.int32)), None, length=n_steps)
    return toks.T
