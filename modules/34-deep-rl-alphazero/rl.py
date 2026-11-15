"""Module 34 helpers: a gridworld, an on-device DQN for CartPole, and AlphaZero for Connect Four.

Everything takes the learner's exercise functions (`ex`) as arguments, so the lesson runs on their code.
Pure JAX + optax; environments from gymnax (CartPole) and pgx (Connect Four); search from DeepMind's mctx.
"""
from __future__ import annotations

import functools
import itertools
import time

import jax
import jax.numpy as jnp
import mctx
import optax

# ----------------------------------------------------------------------------------------------- gridworld

GRID = 5
START, GOAL = 0, GRID * GRID - 1
PITS = jnp.array([7, 12, 17, 13])                      # a wall of pits with gaps at the edges
MOVES = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right


def grid_step(s, a):
    r, c = s // GRID, s % GRID
    nr = jnp.clip(r + MOVES[a, 0], 0, GRID - 1)
    nc = jnp.clip(c + MOVES[a, 1], 0, GRID - 1)
    s2 = nr * GRID + nc
    pit = jnp.any(s2 == PITS)
    goal = s2 == GOAL
    reward = jnp.where(goal, 1.0, jnp.where(pit, -1.0, -0.01))
    return s2, reward, (pit | goal).astype(jnp.float32)


def run_q_learning(ex, key, episodes=400, lr=0.5, gamma=0.95, eps=0.2, max_steps=40, snapshot_every=20):
    """Tabular Q-learning with the learner's q_learning_update / epsilon_greedy. Returns (Q, snapshots, returns)."""
    Q0 = jnp.zeros((GRID * GRID, 4))

    def episode(carry, k):
        Q = carry

        def step(state, kk):
            Q, s, done, ret = state
            a = ex.epsilon_greedy(kk, Q[s], eps)
            s2, r, d = grid_step(s, a)
            Qn = ex.q_learning_update(Q, s, a, r, s2, d, lr, gamma)
            Q = jnp.where(done > 0, Q, Qn)                   # freeze after the episode ends
            ret = ret + jnp.where(done > 0, 0.0, r)
            s = jnp.where(done > 0, s, s2)
            return (Q, s, jnp.maximum(done, d), ret), None

        (Q, _, _, ret), _ = jax.lax.scan(step, (Q, START, 0.0, 0.0), jax.random.split(k, max_steps))
        return Q, (Q, ret)

    Q, (Qs, rets) = jax.jit(lambda k: jax.lax.scan(episode, Q0, jax.random.split(k, episodes)))(key)
    return Q, Qs[snapshot_every - 1::snapshot_every], rets


def greedy_path(Q, max_steps=30):
    s, path = START, [START]
    for _ in range(max_steps):
        s, _, d = grid_step(s, int(jnp.argmax(Q[s])))
        path.append(int(s))
        if float(d):
            break
    return path


# ----------------------------------------------------------------------------------------------- DQN

def mlp_init(key, sizes):
    params = []
    for k, (i, o) in zip(jax.random.split(key, len(sizes) - 1), itertools.pairwise(sizes)):
        params.append({"w": jax.random.normal(k, (i, o)) * jnp.sqrt(2.0 / i), "b": jnp.zeros(o)})
    return params


def mlp_apply(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def make_dqn(ex, n_envs=8, capacity=20_000, batch=128, lr=5e-4, gamma=0.99, target_every=250,
             eps_decay_steps=10_000, warmup=1_000, use_replay=True, use_target=True, log_size=4096,
             double=False, clip=None, total_iters=None):
    """Build (init, run_chunk) for an on-device DQN on gymnax CartPole-v1.

    Each iteration steps all `n_envs` environments once and takes one gradient step. `use_replay=False`
    trains only on the newest n_envs transitions; `use_target=False` bootstraps from the online network.
    """
    import gymnax

    env, env_params = gymnax.make("CartPole-v1")
    n_actions = env.action_space(env_params).n
    sched = optax.linear_schedule(lr, lr * 0.1, total_iters) if total_iters else lr
    opt = optax.adam(sched) if clip is None else optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched))
    cap = capacity if use_replay else n_envs

    def init(key):
        k_net, k_env, key = jax.random.split(key, 3)
        params = mlp_init(k_net, [4, 64, 64, n_actions])
        obs, est = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_env, n_envs), env_params)
        buf = {"obs": jnp.zeros((cap, 4)), "action": jnp.zeros(cap, jnp.int32), "reward": jnp.zeros(cap),
               "next_obs": jnp.zeros((cap, 4)), "done": jnp.zeros(cap), "ptr": jnp.int32(0), "size": jnp.int32(0)}
        return {"params": params, "target": params, "opt": opt.init(params), "obs": obs, "env": est, "buf": buf,
                "key": key, "it": jnp.int32(0), "ep_ret": jnp.zeros(n_envs), "log": jnp.zeros(log_size),
                "n_eps": jnp.int32(0), "loss": jnp.float32(0)}

    def loss_fn(params, target_params, b):
        q = mlp_apply(params, b["obs"])
        q_next = mlp_apply(target_params if use_target else jax.lax.stop_gradient(params), b["next_obs"])
        if double:   # Double DQN (van Hasselt 2016): online net picks a', target net scores it
            pick = jnp.argmax(mlp_apply(params, b["next_obs"]), axis=-1)
            q_next = jnp.where(jax.nn.one_hot(pick, q_next.shape[-1]) > 0, q_next, -jnp.inf)
        return ex.dqn_loss(q, b["action"], ex.td_target(b["reward"], gamma, b["done"], q_next))

    def iteration(st, _):
        key, k_act, k_env, k_samp = jax.random.split(st["key"], 4)
        steps = st["it"] * n_envs
        eps = jnp.maximum(0.05, 1.0 - steps / eps_decay_steps * 0.95)
        a = ex.epsilon_greedy(k_act, mlp_apply(st["params"], st["obs"]), eps)
        out = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(k_env, n_envs), st["env"], a, env_params)
        # gymnax 1.0: (obs, state, reward, terminated, truncated, info). `obs` is already the auto-reset
        # observation when an episode ends, so the transition's next state is info["final_observation"].
        obs2, est2, r = out[0], out[1], out[2]
        terminated, truncated, info = out[3].astype(jnp.float32), out[4].astype(jnp.float32), out[5]
        done = jnp.maximum(terminated, truncated)          # episode over (for logging / resetting returns)
        buf = ex.replay_add(st["buf"], {"obs": st["obs"], "action": a.astype(jnp.int32), "reward": r,
                                        "next_obs": info["final_observation"],
                                        "done": terminated})  # only a true terminal stops bootstrapping
        b = ex.replay_sample(buf, k_samp, batch)
        loss, g = jax.value_and_grad(loss_fn)(st["params"], st["target"], b)
        upd, opt_state = opt.update(g, st["opt"], st["params"])
        new_params = optax.apply_updates(st["params"], upd)
        learn = steps >= warmup
        params = jax.tree.map(lambda n, o: jnp.where(learn, n, o), new_params, st["params"])
        opt_state = jax.tree.map(lambda n, o: jnp.where(learn, n, o), opt_state, st["opt"])
        target = jax.tree.map(lambda p, t: jnp.where(st["it"] % target_every == 0, p, t), params, st["target"])
        ep_ret = st["ep_ret"] + r
        # log finished episodes (rare: write each env's return if done)
        log, n_eps = st["log"], st["n_eps"]
        for i in range(n_envs):                          # n_envs is small and static
            log = jnp.where(done[i] > 0, log.at[n_eps % log_size].set(ep_ret[i]), log)
            n_eps = n_eps + (done[i] > 0).astype(jnp.int32)
        ep_ret = jnp.where(done > 0, 0.0, ep_ret)
        return {"params": params, "target": target, "opt": opt_state, "obs": obs2, "env": est2, "buf": buf,
                "key": key, "it": st["it"] + 1, "ep_ret": ep_ret, "log": log, "n_eps": n_eps, "loss": loss}, None

    @functools.partial(jax.jit, static_argnames=("n_iters",))
    def run_chunk(st, n_iters: int = 1000):
        st, _ = jax.lax.scan(iteration, st, None, length=n_iters)
        return st

    return init, run_chunk, n_envs


def train_dqn(ex, key, total_iters=12_000, chunk=1000, solved_at=475.0, **kw):
    init, run_chunk, n_envs = make_dqn(ex, total_iters=total_iters, **kw)
    st = init(key)
    curve, t0, solved = [], time.perf_counter(), None
    for _ in range(total_iters // chunk):
        st = run_chunk(st)
        n = int(st["n_eps"])
        last = st["log"][: min(n, st["log"].shape[0])]
        recent = float(jnp.mean(last[-20:])) if n else 0.0
        curve.append({"env_steps": int(st["it"]) * n_envs, "episodes": n, "mean_return_last20": recent})
        if solved is None and n >= 20 and recent >= solved_at:
            solved = curve[-1]
    returns = st["log"][: min(int(st["n_eps"]), st["log"].shape[0])]
    return {"curve": curve, "returns": returns, "solved": solved, "seconds": time.perf_counter() - t0,
            "params": st["params"]}


# ----------------------------------------------------------------------------------------------- AlphaZero

def _conv(x, w):
    return jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))


def az_init(key, channels=64, blocks=3, n_actions=7):
    ks = iter(jax.random.split(key, 4 + 2 * blocks))

    def cw(k, cin, cout, ksz=3):
        return jax.random.normal(k, (ksz, ksz, cin, cout)) * jnp.sqrt(2.0 / (ksz * ksz * cin))

    p = {"stem": cw(next(ks), 2, channels), "blocks": []}
    for _ in range(blocks):
        p["blocks"].append({"c1": cw(next(ks), channels, channels), "c2": cw(next(ks), channels, channels) * 0.5})
    p["pol"] = {"w": jax.random.normal(next(ks), (channels * 42, n_actions)) * 0.01, "b": jnp.zeros(n_actions)}
    p["val1"] = {"w": jax.random.normal(next(ks), (channels, 64)) * jnp.sqrt(2.0 / channels), "b": jnp.zeros(64)}
    p["val2"] = {"w": jnp.zeros((64, 1)), "b": jnp.zeros(1)}
    return p


def az_apply(p, obs):
    """obs: [B, 6, 7, 2] bool (pgx Connect Four, from the player-to-move's view) -> (logits [B,7], value [B])."""
    x = jax.nn.relu(_conv(obs.astype(jnp.float32), p["stem"]))
    for blk in p["blocks"]:
        x = jax.nn.relu(x + _conv(jax.nn.relu(_conv(x, blk["c1"])), blk["c2"]))
    logits = x.reshape(x.shape[0], -1) @ p["pol"]["w"] + p["pol"]["b"]
    v = jax.nn.relu(x.mean(axis=(1, 2)) @ p["val1"]["w"] + p["val1"]["b"])
    value = jnp.tanh(v @ p["val2"]["w"] + p["val2"]["b"])[:, 0]
    return logits, value


def make_env():
    import pgx

    return pgx.make("connect_four")


def _masked(logits, legal):
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    return jnp.where(legal, logits, jnp.finfo(logits.dtype).min)


def az_recurrent_fn(env):
    """mctx's recurrent_fn for a *known* simulator: the 'dynamics' is just the real game (pgx)."""

    def recurrent_fn(params, rng_key, action, state):
        player = state.current_player
        state = jax.vmap(env.step)(state, action)
        logits, value = az_apply(params, state.observation)
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = jnp.where(state.terminated, 0.0, -1.0)    # zero-sum: the next player's value is ours, negated
        out = mctx.RecurrentFnOutput(reward=reward, discount=discount * jnp.ones_like(value),
                                     prior_logits=_masked(logits, state.legal_action_mask), value=value)
        return out, state

    return recurrent_fn


def az_policy(env, params, key, state, num_simulations):
    logits, value = az_apply(params, state.observation)
    root = mctx.RootFnOutput(prior_logits=_masked(logits, state.legal_action_mask), value=value, embedding=state)
    return mctx.gumbel_muzero_policy(params, key, root, az_recurrent_fn(env), num_simulations=num_simulations,
                                     invalid_actions=~state.legal_action_mask)


def make_selfplay(env, batch, num_simulations, max_moves=42):
    @jax.jit
    def selfplay(params, key):
        k_init, k_play = jax.random.split(key)
        state = jax.vmap(env.init)(jax.random.split(k_init, batch))

        def move(state, k):
            out = az_policy(env, params, k, state, num_simulations)
            rec = {"obs": state.observation, "pi": out.action_weights, "player": state.current_player,
                   "valid": ~state.terminated}
            state = jax.vmap(env.step)(state, out.action)
            return state, (rec, state.rewards)

        state, (rec, rewards) = jax.lax.scan(move, state, jax.random.split(k_play, max_moves))
        total = rewards.sum(axis=0)                                     # [B, 2] final outcome per player
        z = jnp.take_along_axis(total[None].repeat(max_moves, 0), rec["player"][..., None], axis=-1)[..., 0]
        def flat(x):
            return x.reshape((-1,) + x.shape[2:])

        return {"obs": flat(rec["obs"]), "pi": flat(rec["pi"]), "z": flat(z), "valid": flat(rec["valid"])}

    return selfplay


def make_trainer(ex, lr=1e-3, batch=512):
    opt = optax.adamw(lr, weight_decay=1e-4)

    def loss_fn(params, b):
        logits, value = az_apply(params, b["obs"])
        total, parts = ex.alphazero_loss(logits, value, b["pi"], b["z"])
        return total, parts

    @functools.partial(jax.jit, static_argnames=("n_steps",))
    def train(params, opt_state, data, key, n_steps: int = 100):
        p = data["valid"].astype(jnp.float32)
        p = p / p.sum()

        def step(carry, k):
            params, opt_state = carry
            idx = jax.random.choice(k, p.shape[0], (batch,), p=p)
            b = {kk: data[kk][idx] for kk in ("obs", "pi", "z")}
            (loss, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
            upd, opt_state = opt.update(g, opt_state, params)
            return (optax.apply_updates(params, upd), opt_state), (loss,) + parts

        (params, opt_state), hist = jax.lax.scan(step, (params, opt_state), jax.random.split(key, n_steps))
        return params, opt_state, jax.tree.map(jnp.mean, hist)

    return opt, train


def _random_action(key, legal):
    return jax.random.categorical(key, jnp.where(legal, 0.0, -jnp.inf))


def rollout_value(env, state, key, max_moves=42):
    """Value of `state` for the player to move, from ONE uniformly random playout (pure-MCTS baseline)."""
    player = state.current_player

    def body(c):
        st, k, total, i = c
        k, ka = jax.random.split(k)
        st = env.step(st, _random_action(ka, st.legal_action_mask))
        return st, k, total + st.rewards, i + 1

    def cond(c):
        return (~c[0].terminated) & (c[3] < max_moves)

    _, _, total, _ = jax.lax.while_loop(cond, body, (state, key, jnp.zeros(2), 0))
    return jnp.where(state.terminated, 0.0, total[player])


def rollout_recurrent_fn(env):
    def recurrent_fn(params, rng_key, action, state):
        player = state.current_player
        state = jax.vmap(env.step)(state, action)
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), player]
        keys = jax.random.split(rng_key, state.rewards.shape[0])
        value = jax.vmap(lambda s, k: rollout_value(env, s, k))(state, keys)
        discount = jnp.where(state.terminated, 0.0, -1.0)
        uniform = jnp.where(state.legal_action_mask, 0.0, jnp.finfo(jnp.float32).min)
        return mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=uniform, value=value), state

    return recurrent_fn


def rollout_mcts_policy(env, key, state, num_simulations):
    k_root, k_search = jax.random.split(key)
    keys = jax.random.split(k_root, state.rewards.shape[0])
    value = jax.vmap(lambda s, k: rollout_value(env, s, k))(state, keys)
    uniform = jnp.where(state.legal_action_mask, 0.0, jnp.finfo(jnp.float32).min)
    root = mctx.RootFnOutput(prior_logits=uniform, value=value, embedding=state)
    return mctx.muzero_policy(None, k_search, root, rollout_recurrent_fn(env), num_simulations=num_simulations,
                              invalid_actions=~state.legal_action_mask, dirichlet_fraction=0.0, temperature=0.0)


def make_match(env, games, az_sims, opponent="random", opp_sims=64, max_moves=42):
    """AlphaZero vs an opponent; AZ moves first in half the games. Returns (win, draw, loss) rates for AZ."""

    @jax.jit
    def match(params, key):
        k_init, k_play = jax.random.split(key)
        state = jax.vmap(env.init)(jax.random.split(k_init, games))
        az_player = jnp.arange(games) % 2                       # pgx randomizes who moves first; fix AZ's seat

        def move(carry, k):
            state, total = carry
            k1, k2 = jax.random.split(k)
            az_a = az_policy(env, params, k1, state, az_sims).action
            if opponent == "random":
                opp_a = jax.vmap(_random_action)(jax.random.split(k2, games), state.legal_action_mask)
            else:
                opp_a = rollout_mcts_policy(env, k2, state, opp_sims).action
            a = jnp.where(state.current_player == az_player, az_a, opp_a)
            state = jax.vmap(env.step)(state, a)
            return (state, total + state.rewards), None

        (state, total), _ = jax.lax.scan(move, (state, jnp.zeros((games, 2))), jax.random.split(k_play, max_moves))
        r = jnp.take_along_axis(total, az_player[:, None], axis=1)[:, 0]
        return jnp.mean(r > 0), jnp.mean(r == 0), jnp.mean(r < 0)

    return match


def board_text(state):
    """ASCII board for one pgx Connect Four state (X = player 0, O = player 1)."""
    import numpy as np

    obs = np.asarray(state.observation)                         # [6,7,2] from player-to-move's view
    me, them = obs[..., 0], obs[..., 1]
    mover = int(state.current_player)
    x, o = (me, them) if mover == 0 else (them, me)
    rows = ["".join("X" if x[r, c] else "O" if o[r, c] else "." for c in range(7)) for r in range(6)]
    return "\n".join(" ".join(row) for row in rows) + "\n" + " ".join(str(c) for c in range(7))


def save_params(path, params):
    import numpy as np

    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    np.savez(path, **{jax.tree_util.keystr(k): np.asarray(v) for k, v in flat})


def load_params(path, template):
    import numpy as np

    data = np.load(path)
    flat, treedef = jax.tree_util.tree_flatten_with_path(template)
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(data[jax.tree_util.keystr(k)]) for k, _ in flat])


def alphazero_loop(ex, key, iterations, batch=256, sims=32, train_steps=200, eval_every=5, eval_games=128,
                   mcts_games=64, mcts_sims=64, log=print, params=None):
    """Self-play -> train -> evaluate, `iterations` times. Returns (params, history list of dicts)."""
    env = make_env()
    k_init, key = jax.random.split(key)
    params = az_init(k_init) if params is None else params
    opt, train = make_trainer(ex)
    opt_state = opt.init(params)
    selfplay = make_selfplay(env, batch, sims)
    vs_random = make_match(env, eval_games, 16, "random")
    vs_mcts = make_match(env, mcts_games, sims, "mcts", opp_sims=mcts_sims)
    hist = []

    def evaluate(it, params, k):
        k1, k2 = jax.random.split(k)
        t = time.perf_counter()
        wr = [float(x) for x in vs_random(params, k1)]
        wm = [float(x) for x in vs_mcts(params, k2)]
        return {"iteration": it, "vs_random": wr, "vs_mcts": wm, "eval_s": time.perf_counter() - t}

    key, k = jax.random.split(key)
    hist.append(evaluate(0, params, k))
    log(hist[-1])
    for it in range(1, iterations + 1):
        key, k_sp, k_tr, k_ev = jax.random.split(key, 4)
        t = time.perf_counter()
        data = jax.block_until_ready(selfplay(params, k_sp))
        t_sp = time.perf_counter() - t
        t = time.perf_counter()
        params, opt_state, (loss, pce, vmse) = train(params, opt_state, data, k_tr, train_steps)
        jax.block_until_ready(params)
        t_tr = time.perf_counter() - t
        n_pos = int(data["valid"].sum())
        rec = {"iteration": it, "loss": float(loss), "policy_ce": float(pce), "value_mse": float(vmse),
               "positions": n_pos, "selfplay_s": t_sp, "train_s": t_tr, "games_per_s": batch / t_sp,
               "sims_per_s": n_pos * sims / t_sp}
        if it % eval_every == 0 or it == iterations:
            rec.update(evaluate(it, params, k_ev))
        hist.append(rec)
        log(rec)
    return params, hist
