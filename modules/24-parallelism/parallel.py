"""Module 24 helpers: a tiny GPT-like model, swappable sharding strategies, a GPipe pipeline and
expert parallelism, all on the simulated CPU mesh the lesson sets up.

The strategies follow the JAX Training Cookbook's idea of config-driven sharding: the model code never
changes; only a dict of PartitionSpecs (one per parameter, plus one for the batch) does.
"""
from __future__ import annotations

import math
import re
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import NamedSharding

V, D, F, B, T = 512, 128, 512, 32, 16          # vocab, model width, MLP width, batch, sequence
SHAPES = {"emb": (V, D), "w1": (D, F), "w2": (F, D), "head": (D, V)}
COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute")


def init_params(key):
    ks = jax.random.split(key, 4)
    return {k: jax.random.normal(kk, s) / math.sqrt(s[0]) for kk, (k, s) in zip(ks, SHAPES.items(), strict=True)}


def make_batch(key):
    toks = jax.random.randint(key, (B, T + 1), 0, V)
    return toks[:, :-1], toks[:, 1:]


def loss_fn(p, x, y):
    h = p["emb"][x]                                   # [B, T, D]
    h = h + jax.nn.gelu(h @ p["w1"]) @ p["w2"]        # one residual MLP block
    logits = h @ p["head"]
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


OPT = optax.adam(1e-2)


def train_step(p, s, x, y):
    loss, g = jax.value_and_grad(loss_fn)(p, x, y)
    upd, s = OPT.update(g, s, p)
    return optax.apply_updates(p, upd), s, loss


STRATEGIES = {
    # name: (mesh shape, mesh axes, param specs, batch spec)
    "DP": ((8,), ("data",), dict.fromkeys(SHAPES, jax.P()), jax.P("data")),
    "FSDP": ((8,), ("data",), {"emb": jax.P("data", None), "w1": jax.P("data", None),
                               "w2": jax.P("data", None), "head": jax.P("data", None)}, jax.P("data")),
    "TP": ((8,), ("model",), {"emb": jax.P(None, "model"), "w1": jax.P(None, "model"),
                              "w2": jax.P("model", None), "head": jax.P(None, "model")}, jax.P()),
    "FSDP+TP": ((4, 2), ("data", "model"), {"emb": jax.P("data", "model"), "w1": jax.P("data", "model"),
                                            "w2": jax.P("model", "data"), "head": jax.P("data", "model")},
                jax.P("data")),
}


def count_collectives(hlo: str) -> dict[str, int]:
    out = dict.fromkeys(COLLECTIVES, 0)
    for m in re.finditer(r"\b(" + "|".join(COLLECTIVES) + r")(?:-start)?\(", hlo):
        out[m.group(1)] += 1
    return out


def run_strategy(name, params, batch, ref, devices, config=None):
    """One Adam step under a strategy (a STRATEGIES entry, or `config` = (mesh shape, axes, param specs,
    batch spec)). Returns a row of measurements vs the single-device reference."""
    shape, axes, specs, bspec = config or STRATEGIES[name]
    mesh = jax.make_mesh(shape, axes, devices=devices, axis_types=(jax.sharding.AxisType.Auto,) * len(shape))
    p = {k: jax.device_put(v, NamedSharding(mesh, specs[k])) for k, v in params.items()}
    x, y = (jax.device_put(a, NamedSharding(mesh, bspec)) for a in batch)
    s = jax.jit(OPT.init)(p)                                    # optimizer state inherits each param's sharding
    step = jax.jit(train_step)
    t0 = time.perf_counter()
    compiled = step.lower(p, s, x, y).compile()
    compile_s = time.perf_counter() - t0
    p2, s2, loss = compiled(p, s, x, y)
    jax.block_until_ready(p2)
    err = max(float(np.abs(np.asarray(p2[k]) - np.asarray(ref[0][k])).max()) for k in p2)
    per_dev = sum(leaf.addressable_shards[0].data.nbytes for leaf in jax.tree.leaves((p2, s2)))
    total = sum(leaf.nbytes for leaf in jax.tree.leaves((p2, s2)))
    return {"strategy": name, "mesh": "×".join(map(str, shape)), "per_device_bytes": per_dev,
            "total_bytes": total, "max_err": err, "loss": float(loss), "compile_s": compile_s,
            **count_collectives(compiled.as_text())}


def reference_step(params, batch, device):
    p = jax.device_put(params, device)
    x, y = jax.device_put(batch, device)
    s = OPT.init(p)
    p2, _, loss = jax.jit(train_step)(p, s, x, y)
    return p2, float(loss)


# ---------------------------------------------------------------- pipeline parallelism (GPipe-style)

def stage_fn(w, h):
    return jnp.tanh(h @ w)


def pipeline_forward(stage_ws, microbatches, mesh, axis="stage"):
    """Run microbatches through len(mesh[axis]) stages, one stage per device, passing activations right
    with ppermute each clock tick. stage_ws: [S, D, D] sharded P(axis); microbatches: [M, b, D] replicated.
    Returns the last stage's [M, b, D] outputs as a NumPy array."""
    n_stages = mesh.shape[axis]
    n_micro = microbatches.shape[0]
    perm = [(i, (i + 1) % n_stages) for i in range(n_stages)]

    @partial(jax.shard_map, mesh=mesh, in_specs=(jax.P(axis), jax.P()), out_specs=jax.P(axis))
    def run(w, mbs):
        w = w[0]
        me = jax.lax.axis_index(axis)
        carry = jnp.zeros(mbs.shape[1:], mbs.dtype)
        outs = jnp.zeros_like(mbs)
        for t in range(n_micro + n_stages - 1):              # the GPipe clock
            feed = mbs[min(t, n_micro - 1)]
            inp = jnp.where(me == 0, feed, carry)              # stage 0 reads input; others read neighbour
            out = stage_fn(w, inp)
            k = t - (n_stages - 1)                             # microbatch finishing at the last stage
            if 0 <= k < n_micro:
                outs = outs.at[k].set(jnp.where(me == n_stages - 1, out, outs[k]))
            carry = jax.lax.ppermute(out, axis, perm)
        return outs[None]

    return np.asarray(run(stage_ws, microbatches))[-1]   # [S, M, b, D] → the last stage's slice


def pipeline_reference(stage_ws, microbatches):
    h = microbatches
    for w in stage_ws:
        h = stage_fn(w, h)
    return h


def gpipe_schedule(n_stages, n_micro):
    """(stage, microbatch, clock) for every forward step; the rest of the grid is the bubble."""
    return [(s, m, m + s) for s in range(n_stages) for m in range(n_micro)]


# ---------------------------------------------------------------- expert parallelism (MoE all_to_all)

def moe_reference(x, gate_w, expert_ws):
    """Top-1 MoE, computed densely on one device. x: [N, D], gate_w: [D, E], expert_ws: [E, D, D]."""
    choice = jnp.argmax(x @ gate_w, -1)
    all_out = jnp.einsum("nd,edk->nek", x, expert_ws)
    return jnp.take_along_axis(all_out, choice[:, None, None], 1)[:, 0]


def moe_expert_parallel(x, gate_w, expert_ws, mesh, axis="expert"):
    """Same MoE with one expert per device: route tokens with all_to_all, compute, route back.

    x [N, D] sharded P(axis) (each device owns N/E tokens); expert_ws [E, D, D] sharded P(axis).
    Capacity = tokens per device (no token is dropped), so the result equals the reference exactly."""
    n_exp = mesh.shape[axis]

    @partial(jax.shard_map, mesh=mesh, in_specs=(jax.P(axis), jax.P(), jax.P(axis)), out_specs=jax.P(axis))
    def run(xb, gw, wb):
        choice = jnp.argmax(xb @ gw, -1)                                       # [n_local]
        onehot = jax.nn.one_hot(choice, n_exp, dtype=xb.dtype)                 # [n_local, E]
        send = jnp.einsum("ne,nd->end", onehot, xb)                            # [E, n_local, D]: slot per token
        recv = jax.lax.all_to_all(send, axis, 0, 0, tiled=True)               # tokens for MY expert
        done = recv @ wb[0]                                                    # apply my expert
        back = jax.lax.all_to_all(done, axis, 0, 0, tiled=True)               # return to owners
        return jnp.einsum("ne,end->nd", onehot, back)

    return run(x, gate_w, expert_ws)
