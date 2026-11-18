"""Module 40, section 4: expert parallelism for standard MoE and LatentMoE on 8 simulated devices.

Run as a subprocess by the lesson (it needs the CPU split into 8 devices before JAX starts, while the lesson
itself uses the GPU). Prints one JSON object: for each variant, the bytes each device's all-to-alls move (read from the
lowered program) and the largest difference from a single-device reference.

The layer is top-k routing with several experts per device and a fixed **capacity** per (source, destination)
pair of devices, as real systems do — Module 24's version was top-1 with one expert per device. Tokens past a
device's capacity would be dropped; the reference check uses a capacity big enough that none are.
"""
from __future__ import annotations

import json
import re
import sys
from functools import partial

import jax

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 8)

import jax.numpy as jnp

P = 8                                            # devices = expert-parallel group size
D, M_EXP, N_TOK = 64, 32, 256                    # model width, expert hidden width, tokens in the batch
DTYPE = jnp.bfloat16                             # activations on the wire: 2 bytes per number


def init(key, d, w, n_experts):
    ks = jax.random.split(key, 6)
    n = lambda k, s: (jax.random.normal(k, s) * 0.2).astype(jnp.float32)
    p = {"router": n(ks[0], (d, n_experts)),
         "w1": n(ks[1], (n_experts, w, M_EXP)), "w3": n(ks[2], (n_experts, w, M_EXP)),
         "w2": n(ks[3], (n_experts, M_EXP, w))}
    if w != d:
        p["down"], p["up"] = n(ks[4], (d, w)), n(ks[5], (w, d))
    return p


def route(p, x, k):
    scores = jax.nn.sigmoid(x @ p["router"])                                   # the router sees the full x
    top, idx = jax.lax.top_k(scores, k)
    return idx, top / top.sum(-1, keepdims=True)


def expert(p, e, z):
    return (jax.nn.silu(z @ p["w1"][e]) * (z @ p["w3"][e])) @ p["w2"][e]


def reference(p, x, k):
    """Single device, dense: every token through each of its k experts."""
    idx, g = route(p, x, k)
    z = x @ p["down"] if "down" in p else x
    all_out = jnp.stack([expert(p, e, z) for e in range(p["w1"].shape[0])], 1)  # (N, E, w)
    y = jnp.einsum("nk,nkw->nw", g, jnp.take_along_axis(all_out, idx[..., None], 1))
    return y @ p["up"] if "up" in p else y


def expert_parallel(p, x, k, mesh, capacity):
    """Each device owns N/P tokens and E/P experts. One all-to-all sends each (token, expert) slot's vector to
    the expert's device, the experts run, a second all-to-all brings results home."""
    E = p["w1"].shape[0]
    per_dev = E // P
    expert_spec = {name: (jax.P("x") if name in ("w1", "w3", "w2") else jax.P()) for name in p}

    @partial(jax.shard_map, mesh=mesh, in_specs=(expert_spec, jax.P("x")), out_specs=jax.P("x"))
    def run(pl, xb):
        n_loc = xb.shape[0]
        idx, g = route(pl, xb, k)                                             # (n_loc, k)
        z = (xb @ pl["down"] if "down" in pl else xb).astype(DTYPE)          # LatentMoE: shrink BEFORE sending
        dest, local = (idx // per_dev).reshape(-1), (idx % per_dev).reshape(-1)  # one slot per (token, pick)
        onehot = jax.nn.one_hot(dest, P, dtype=jnp.int32)                     # (n_loc*k, P)
        pos = ((jnp.cumsum(onehot, 0) - 1) * onehot).sum(-1)                  # rank of the slot at its device
        keep = pos < capacity
        tok = jnp.repeat(jnp.arange(n_loc), k)
        send = jnp.zeros((P, capacity, z.shape[-1]), DTYPE).at[dest, pos].set(z[tok], mode="drop")
        send_e = jnp.full((P, capacity), -1, jnp.int32).at[dest, pos].set(local, mode="drop")
        recv = jax.lax.all_to_all(send, "x", 0, 0, tiled=True)                # slots for MY experts
        recv_e = jax.lax.all_to_all(send_e, "x", 0, 0, tiled=True)
        flat, fe = recv.reshape(-1, recv.shape[-1]).astype(jnp.float32), recv_e.reshape(-1)
        outs = jnp.stack([expert(pl, e, flat) for e in range(per_dev)], 1)     # (slots, per_dev, w)
        done = jnp.take_along_axis(outs, jnp.clip(fe, 0)[:, None, None], 1)[:, 0] * (fe >= 0)[:, None]
        back = jax.lax.all_to_all(done.reshape(recv.shape).astype(DTYPE), "x", 0, 0, tiled=True)
        got = back[dest, jnp.clip(pos, 0, capacity - 1)].astype(jnp.float32) * keep[:, None]
        y = (got.reshape(n_loc, k, -1) * g[..., None]).sum(1)
        return y @ pl["up"] if "up" in pl else y                              # LatentMoE: grow AFTER combining

    return run(p, x)


_SIZES = {"bf16": 2, "f16": 2, "f32": 4, "i32": 4}


def all_to_all_bytes(stablehlo: str) -> dict:
    """Bytes each device's all-to-alls move, by dtype, read from the *lowered* program (StableHLO: the program
    as written, one device's view inside shard_map). The compiled CPU program isn't used: XLA's CPU backend
    computes bf16 in f32, so it would report twice the bytes a GPU sends."""
    out = {}
    for line in stablehlo.splitlines():
        m = re.search(r'"stablehlo\.all_to_all".*->\s*tensor<([\dx]+)x(bf16|f16|f32|i32)>', line)
        if m:
            n = 1
            for s in m.group(1).split("x"):
                n *= int(s)
            out[m.group(2)] = out.get(m.group(2), 0) + n * _SIZES[m.group(2)]
    return out


def main() -> int:
    mesh = jax.make_mesh((P,), ("x",), devices=jax.devices())
    x = jax.random.normal(jax.random.key(0), (N_TOK, D))
    base_e, base_k, alpha = 16, 2, 4
    variants = {"standard": (D, base_e, base_k), "latent-eff": (D // alpha, base_e * alpha, base_k),
                "latent-acc": (D // alpha, base_e * alpha, base_k * alpha)}
    res = {}
    for name, (w, E, k) in variants.items():
        p = init(jax.random.key(1), D, w, E)
        ref = reference(p, x, k)
        put = lambda a, spec: jax.device_put(a, jax.NamedSharding(mesh, spec))
        p = {nm: put(a, jax.P("x") if nm in ("w1", "w3", "w2") else jax.P()) for nm, a in p.items()}
        xs = put(x, jax.P("x"))
        n_loc = N_TOK // P
        exact_cap = n_loc * k                                                   # no slot can ever be dropped
        real_cap = -(-int(1.25 * n_loc * k) // P)                               # capacity factor 1.25
        got = expert_parallel(p, xs, k, mesh, exact_cap)
        err = float(jnp.abs(got - ref).max() / jnp.abs(ref).max())
        hlo = jax.jit(lambda p_, x_, k_=k, c_=real_cap: expert_parallel(p_, x_, k_, mesh, c_)).lower(p, xs).as_text()
        b = all_to_all_bytes(hlo)
        res[name] = {"width": w, "experts": E, "top_k": k, "capacity": real_cap, "rel_err": err,
                     "act_bytes_per_device": b.get("bf16", 0), "id_bytes_per_device": b.get("i32", 0),
                     "n_all_to_all": hlo.count('"stablehlo.all_to_all"')}
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
