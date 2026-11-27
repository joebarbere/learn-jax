"""Reference solutions for Module 31b. Read exercises.py first; peek here only after trying."""
import jax
import jax.numpy as jnp


def mdrnn_sweep_naive(p, x):
    B, R, C, _ = x.shape
    H = p["wu"].shape[0]
    rows, prev = [], jnp.zeros((B, C, H), x.dtype)
    for i in range(R):
        left, row = jnp.zeros((B, H), x.dtype), []
        for j in range(C):
            left = jnp.tanh(x[:, i, j] @ p["wx"] + prev[:, j] @ p["wu"] + left @ p["wl"] + p["b"])
            row.append(left)
        prev = jnp.stack(row, axis=1)
        rows.append(prev)
    return jnp.stack(rows, axis=1)


def skew(x):
    B, R, C, H = x.shape
    rows = jnp.arange(R)[:, None]
    cols = rows + jnp.arange(C)[None, :]
    return jnp.zeros((B, R, C + R - 1, H), x.dtype).at[:, rows, cols].set(x)


def unskew(s, C):
    R = s.shape[1]
    rows = jnp.arange(R)[:, None]
    return s[:, rows, rows + jnp.arange(C)[None, :]]


def mdrnn_sweep(p, x):
    B, R, C, _ = x.shape
    H = p["wu"].shape[0]
    z = skew(x @ p["wx"] + p["b"])                                   # (B, R, W, H)
    mask = skew(jnp.ones((1, R, C, 1), x.dtype))[0, :, :, 0]         # (R, W)

    def step(h_prev, col):
        zc, m = col
        up = jnp.concatenate([jnp.zeros_like(h_prev[:, :1]), h_prev[:, :-1]], axis=1)
        h = jnp.tanh(zc + up @ p["wu"] + h_prev @ p["wl"]) * m[None, :, None]
        return h, h

    _, cols = jax.lax.scan(step, jnp.zeros((B, R, H), x.dtype), (jnp.moveaxis(z, 2, 0), mask.T))
    return unskew(jnp.moveaxis(cols, 0, 2), C)


_FLIPS = [(), (1,), (2,), (1, 2)]


def four_direction_sweeps(params, x, sweep):
    outs = []
    for k, axes in enumerate(_FLIPS):
        p = params[k] if len(params) == 4 else params[0]
        v = jnp.flip(x, axes) if axes else x
        h = sweep(p, v)
        outs.append(jnp.flip(h, axes) if axes else h)
    return jnp.concatenate(outs, axis=-1)


def cross_scan(x):
    B, R, C, F = x.shape
    row = x.reshape(B, R * C, F)
    col = jnp.swapaxes(x, 1, 2).reshape(B, R * C, F)
    return jnp.stack([row, col, row[:, ::-1], col[:, ::-1]])


def cross_merge(ys, R, C):
    _, B, _, H = ys.shape
    row = ys[0] + ys[2][:, ::-1]
    col = ys[1] + ys[3][:, ::-1]
    return row.reshape(B, R, C, H) + jnp.swapaxes(col.reshape(B, C, R, H), 1, 2)
