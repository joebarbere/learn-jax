"""Smoke test for the RL libraries in the image (modules 34, 37): pgx, mctx, gymnax on the GPU.

    containers/run.sh python tools/smoke_rl.py
"""
import time

import gymnax
import jax
import jax.numpy as jnp
import mctx
import pgx

print("devices", jax.devices())
env = pgx.make("connect_four")
st = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.key(0), 1024))
step = jax.jit(jax.vmap(env.step))
t = time.perf_counter()
st2 = jax.block_until_ready(step(st, jnp.argmax(st.legal_action_mask, -1)))
print("pgx connect_four x1024 ok", st2.current_player.shape, f"{(time.perf_counter() - t) * 1e3:.0f} ms incl. compile")
print("pgx othello ok", pgx.make("othello").init(jax.random.key(1)).legal_action_mask.shape)
B, A = 8, 7
root = mctx.RootFnOutput(prior_logits=jnp.zeros((B, A)), value=jnp.zeros(B), embedding=jnp.zeros((B,)))


def rec(params, rng, action, emb):
    out = mctx.RecurrentFnOutput(reward=jnp.zeros(B), discount=jnp.ones(B), prior_logits=jnp.zeros((B, A)),
                                 value=jnp.zeros(B))
    return out, emb


out = jax.jit(lambda k: mctx.gumbel_muzero_policy(None, k, root, rec, num_simulations=16))(jax.random.key(2))
print("mctx gumbel_muzero ok", out.action.shape)
genv, gp = gymnax.make("CartPole-v1")
obs, s = genv.reset(jax.random.key(3), gp)
out = genv.step(jax.random.key(4), s, 0, gp)  # gymnax 1.0 returns (obs, state, reward, done, info, ...)
r = out[2]
print("gymnax CartPole ok", obs.shape, float(r))
