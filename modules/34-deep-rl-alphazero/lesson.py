import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="34 · Deep RL: DQN to AlphaZero")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import json
    import sys
    import time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj
    from learnjax import rosetta

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    ex = lj.load_exercises(mo.notebook_dir())

    def _load(name):
        _spec = importlib.util.spec_from_file_location(f"m34_{name}", mo.notebook_dir() / f"{name}.py")
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)
        return _mod

    rl = _load("rl")
    RUNS = Path("/course/runs/34-deep-rl-alphazero")
    return RUNS, ex, jax, jnp, json, lj, mo, np, plt, rl, rosetta, time


@app.cell
def _(mo):
    mo.md(r"""
    # 34 · Deep RL: DQN to AlphaZero (2013–2019)

    **Part 10, Milestones rebuilt in JAX.** Take Modules 05 (optimizers), 07 (`vmap`) and 08 (`scan`) first.

    **Goal:** retrace how machines learned to play — from a lookup table of values, to a neural network
    that learned Atari from pixels, to a program that taught itself Connect Four by playing against itself —
    with every environment, every search and every gradient step running on your RX 7600 XT.

    **You will:** solve a gridworld with tabular Q-learning (Watkins, 1989) · train a DQN on CartPole with
    its two famous tricks, a replay buffer and a target network, and see what breaks without them · write
    the PUCT rule that steers AlphaZero's search · run AlphaZero self-play on hundreds of Connect Four games
    at once with DeepMind's `mctx` and the `pgx` game simulators · play against your own agent.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from.** In 1959 Arthur Samuel's checkers program at IBM improved by playing itself,
    and he coined the term "machine learning". Reinforcement learning grew up in the 1980s around Richard
    Sutton and Andrew Barto's temporal-difference ideas; Chris Watkins introduced Q-learning in 1989, and
    Gerald Tesauro's TD-Gammon (1992) reached expert backgammon with a small neural network. Deep Blue beat
    Garry Kasparov in 1997 with search, not learning. In 2013 DeepMind's DQN learned Atari games from pixels;
    in 2016 AlphaGo beat Lee Sedol; in 2017 AlphaGo Zero and AlphaZero learned Go, chess and shogi from
    self-play alone; in 2019 MuZero learned the rules too. The same machinery now tunes language models
    (RLHF, GRPO). Timeline: Module 00b.
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The story: from checkers to self-play (1959–2019)

    **Learning from reward.** Reinforcement learning (RL) is learning from *consequences*: an agent acts,
    the world changes, and now and then a number — the **reward** — says how well it's doing. No one labels
    the right action; the agent must find it by trying. Sutton and Barto's
    [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed.,
    2018) is the field's textbook and free online.

    **1959, Samuel's checkers.** Arthur Samuel's program at IBM adjusted the weights of a board-evaluation
    function so that its estimate of a position agreed with its estimate a few moves later — the seed of
    *temporal-difference* learning. It was one of the first programs to improve by playing itself.

    **1989–1992, Q-learning and TD-Gammon.** Chris Watkins' Q-learning (PhD thesis 1989; convergence proof
    with Peter Dayan, 1992) learns the value of each action in each state from experience, with no model of
    the world. Gerald Tesauro's TD-Gammon (1992) combined TD learning with a small neural network and
    reached expert-level backgammon from self-play.

    **1997, Deep Blue.** IBM's Deep Blue beat world champion Garry Kasparov with brute-force search over
    hundreds of millions of positions per second and a hand-tuned evaluation — *search*, not learning. It
    showed what search could do, and how little it generalized.

    **2013–2015, DQN.** DeepMind's Deep Q-Network (Mnih et al.) learned to play dozens of Atari games from
    raw pixels with one architecture, reaching human level on many (Nature, 2015). Q-learning with a neural
    network had long been unstable; two tricks made it work — **experience replay** and a **target network**
    (Section 3).

    **2016, AlphaGo.** Go was thought a decade away. AlphaGo combined neural networks trained on human games
    and self-play with Monte Carlo tree search and beat Lee Sedol 4–1. Its "move 37" in game two, a move
    human experts first thought a mistake, became the symbol of a machine finding something new.

    **2017, AlphaGo Zero and AlphaZero.** AlphaGo Zero dropped the human games: one network, trained only by
    self-play from random play, guided by search — and it beat AlphaGo. AlphaZero applied the same algorithm,
    unchanged, to chess and shogi. **This is the algorithm you'll run in Section 6.**

    **2019, MuZero.** MuZero planned with a *learned* model of the game instead of the rules, which let the
    same method play Atari too (Section 8).

    **And then language models.** The policy-gradient methods of this lineage — PPO (2017), and later GRPO —
    became how language models are tuned from feedback: RLHF (Module 35) and reasoning RL (Module 26).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · RL in one picture, and tabular Q-learning

    An **agent** in **state** $s$ takes **action** $a$, gets **reward** $r$ and lands in state $s'$. It wants
    to maximize the **return**, the discounted sum of future rewards $r_0 + \gamma r_1 + \gamma^2 r_2 + \dots$
    ($\gamma < 1$ makes sooner rewards worth more).

    The **action-value** $Q(s, a)$ is the return you expect after taking $a$ in $s$ and acting well afterwards.
    If you knew it, acting would be easy: pick $\arg\max_a Q(s, a)$. It satisfies the **Bellman equation**

    $$Q(s, a) = \mathbb{E}\left[\, r + \gamma \max_{a'} Q(s', a') \,\right],$$

    and Q-learning turns that into an update: after each step, move $Q(s,a)$ a little toward the target
    $r + \gamma \max_{a'} Q(s', a')$. To keep discovering better actions, act **ε-greedily**: usually the best
    known action, sometimes a random one.

    Our world: a 5×5 grid. Start top-left, goal bottom-right (+1), a wall of pits in the middle (−1, game
    over) with gaps at the edges, and a small cost (−0.01) per step so shorter paths are better.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _Q = jnp.zeros((4, 2)).at[3].set(jnp.array([1.0, 2.0]))
    _u = lj.attempt(ex.q_learning_update, _Q, 0, 1, 0.5, 3, 0.0, 0.1, 0.9)
    _e = lj.attempt(ex.epsilon_greedy, jax.random.key(0), jnp.array([[0.0, 5.0, 1.0], [3.0, 0.0, 0.0]]), 0.0)
    _items = [mo.md("**Your turn:** `q_learning_update` and `epsilon_greedy` in `exercises.py`.")]
    if _u.ok:
        _items.append(mo.md(f"`q_learning_update`: Q[0,1] went 0 → **{float(_u.value[0, 1]):.3f}** "
                            "(target 0.5 + 0.9·2 = 2.3, learning rate 0.1). ✅"))
    else:
        _items.append(_u.view(mo))
    if _e.ok:
        _items.append(mo.md(f"`epsilon_greedy` with ε = 0 picks **{list(map(int, _e.value))}** (the argmaxes). ✅"))
    else:
        _items.append(_e.view(mo))
    tabular_ready = _u.ok and _e.ok
    mo.vstack(_items)
    return (tabular_ready,)


@app.cell
def _(ex, jax, lj, mo, rl, tabular_ready, time):
    grid_run = None
    if tabular_ready:
        _t = time.perf_counter()
        _r = lj.attempt(rl.run_q_learning, ex, jax.random.key(0))
        if _r.ok:
            _Q, _snaps, _rets = _r.value
            grid_run = {"Q": _Q, "snaps": _snaps, "returns": _rets, "seconds": time.perf_counter() - _t,
                        "path": rl.greedy_path(_Q)}
    episode_slider = mo.ui.slider(1, 20, value=20, label="episodes trained (×20)", show_value=True)
    mo.vstack([mo.md("400 episodes of Q-learning, compiled into one [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan):"), episode_slider]) \
        if grid_run else mo.md("*(The gridworld runs once both exercises above work.)*")
    return episode_slider, grid_run


@app.cell
def _(episode_slider, grid_run, mo, np, plt, rl):
    if grid_run is None:
        _out = mo.md("")
    else:
        _i = episode_slider.value - 1
        _V = np.asarray(grid_run["snaps"][_i].max(axis=1)).reshape(rl.GRID, rl.GRID)
        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
        _im = _ax1.imshow(_V, cmap="RdYlGn", vmin=-1, vmax=1)
        for _p in np.asarray(rl.PITS):
            _ax1.text(_p % rl.GRID, _p // rl.GRID, "pit", ha="center", va="center", fontsize=8, color="#1f2430")
        _ax1.text(4, 4, "goal", ha="center", va="center", fontsize=8, color="#1f2430")
        _ax1.text(0, 0, "start", ha="center", va="center", fontsize=8, color="#1f2430")
        if _i == 19:
            _pr = [(s // rl.GRID, s % rl.GRID) for s in grid_run["path"]]
            _ax1.plot([c for _, c in _pr], [r for r, _ in _pr], "b-o", ms=3, lw=1.5)
        _ax1.set(title=f"max_a Q(s, a) after {(_i + 1) * 20} episodes", xticks=[], yticks=[])
        plt.colorbar(_im, ax=_ax1, fraction=0.046)
        _ret = np.asarray(grid_run["returns"])
        _ax2.plot(np.convolve(_ret, np.ones(20) / 20, mode="valid"))
        _ax2.axvline((_i + 1) * 20, color="gray", ls=":")
        _ax2.set(title="episode return (20-episode average)", xlabel="episode")
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(grid_run, mo):
    mo.md(f"""
    Value spreads backwards from the goal one step per visit — the Bellman update at work. At the last
    snapshot the blue line is the greedy path: **{len(grid_run["path"]) - 1} moves**, through the gap
    around the pits (training took {grid_run["seconds"]:.1f} s, mostly compilation). The return curve stays
    below the ideal because ε = 0.2 keeps exploring.

    A table works for 25 states. Atari has more screens than atoms in the universe; CartPole's state is
    continuous. We need a **function** that generalizes across states: a neural network.
    """) if grid_run else mo.md("")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · DQN: Q-learning with a neural network (2013–2015)

    Replace the table with a network $Q_\theta(s, \cdot)$ that outputs one value per action, and regress it
    toward the target $y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$. Naively this diverges. DQN's two tricks:

    * **Experience replay.** Store transitions in a big buffer and train on random minibatches from it.
      Consecutive frames are nearly identical; sampling at random breaks those correlations and reuses each
      experience many times.
    * **A target network** $\theta^-$: a frozen copy of the network, refreshed every few hundred steps,
      used only to compute targets. Without it, every update moves the target it's chasing.

    DQN also clipped the error (a **Huber** loss: quadratic near zero, linear far away) so a single surprising
    transition can't produce a huge gradient.

    Our task is CartPole-v1 from `gymnax`, a JAX port of the classic control environments: keep a pole
    balanced on a cart for up to 500 steps (+1 per step). Everything — 8 environments stepped in parallel
    with [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap), the replay buffer
    as fixed-size arrays, the network update — runs inside one compiled `lax.scan`, so the GPU never waits on
    Python.
    """)
    return


@app.cell
def _(ex, jax, jnp, lj, mo):
    _td = lj.attempt(ex.td_target, jnp.array([1.0, 1.0]), 0.5, jnp.array([0.0, 1.0]),
                     jnp.array([[0.0, 2.0], [5.0, 7.0]]))
    _l = lj.attempt(ex.dqn_loss, jnp.array([[1.0, 2.0], [0.0, 0.0]]), jnp.array([1, 0]), jnp.array([2.5, 3.0]))
    _buf = {"obs": jnp.zeros((5, 2)), "action": jnp.zeros(5, jnp.int32), "ptr": jnp.int32(0), "size": jnp.int32(0)}
    _add = lj.attempt(ex.replay_add, _buf, {"obs": jnp.ones((3, 2)), "action": jnp.array([1, 2, 3], jnp.int32)})
    _smp = lj.attempt(ex.replay_sample, _add.value, jax.random.key(0), 4) if _add.ok else _add
    _rows = [mo.md("**Your turn:** `td_target`, `dqn_loss`, `replay_add`, `replay_sample`.")]
    for _name, _a, _msg in [
        ("td_target", _td, lambda v: f"targets {[float(x) for x in v]} (expect [2.0, 1.0])"),
        ("dqn_loss", _l, lambda v: f"Huber loss {float(v):.4f} (expect 1.3125)"),
        ("replay_add", _add, lambda v: f"ptr {int(v['ptr'])}, size {int(v['size'])} (expect 3, 3)"),
        ("replay_sample", _smp, lambda v: f"sampled actions {list(map(int, v['action']))}")]:
        _rows.append(mo.md(f"`{_name}`: {_msg(_a.value)} ✅") if _a.ok else _a.view(mo))
    dqn_ready = all(a.ok for a in (_td, _l, _add, _smp))
    mo.vstack(_rows)
    return (dqn_ready,)


@app.cell
def _(mo, rosetta):
    mo.vstack([
        rosetta.tabs(mo, "DQN target and Huber loss", solves=("td_target", "dqn_loss")),
        rosetta.tabs(mo, "sample without replacement"),
    ])
    return


@app.cell
def _():
    # The DQN settings used below: the best of a 12-setting x 3-seed sweep (see the text under the plot).
    DQN_KW = {"double": True, "clip": 10.0, "target_every": 1000, "lr": 1e-3}
    DQN_ITERS = 20_000
    return DQN_ITERS, DQN_KW


@app.cell
def _(DQN_ITERS, DQN_KW, dqn_ready, ex, jax, lj, mo, rl):
    dqn = None
    if dqn_ready:
        _r = lj.attempt(lambda: [rl.train_dqn(ex, jax.random.key(_s), total_iters=DQN_ITERS, **DQN_KW)
                                 for _s in range(3)])
        dqn = _r.value if _r.ok else None   # three seeds, same settings
        _view = None if _r.ok else _r.view(mo)
    else:
        _view = mo.md("*(DQN training runs once the four exercises above work.)*")
    _view
    return (dqn,)


@app.cell
def _(dqn, mo, np, plt):
    if dqn is None:
        _out = mo.md("")
    else:
        _fig, _ax = plt.subplots(figsize=(8, 3.2))
        for _s, _d in enumerate(dqn):
            _ret = np.asarray(_d["returns"])
            _ax.plot(np.convolve(_ret, np.ones(20) / 20, mode="valid"), lw=1.5, label=f"seed {_s}")
        _ax.axhline(475, color="gray", ls=":", label="'solved' (475)")
        _ax.set(xlabel="episode", ylabel="return, 20-episode average", title="DQN on CartPole-v1, three seeds, on the GPU")
        _ax.legend(fontsize=8, loc="upper left")
        _fig.tight_layout()
        _steps = sum(_d["curve"][-1]["env_steps"] for _d in dqn)
        _secs = sum(_d["seconds"] for _d in dqn)
        _lines = []
        for _s, _d in enumerate(dqn):
            _c, _sv = _d["curve"], _d["solved"]
            _lines.append(f"* **seed {_s}:** " + (f"solved after {_sv['env_steps']:,} steps ({_sv['episodes']} episodes)"
                          if _sv else f"not solved (best 20-episode average {max(x['mean_return_last20'] for x in _c):.0f})")
                          + f", final average {_c[-1]['mean_return_last20']:.0f}")
        _n = sum(_d["solved"] is not None for _d in dqn)
        _out = mo.vstack([_fig, mo.md(
            f"{_steps:,} environment steps in **{_secs:.0f} s** ({_steps / _secs:,.0f} steps/s, including compilation). "
            f"Same code, same settings, three random seeds; **{_n} of 3** reached 475:\n\n" + "\n".join(_lines) + "\n\n"
            "That spread is the honest picture of DQN. These settings are the best of a sweep over learning rate, "
            "target-update interval and Double DQN (12 settings × 3 seeds, 160k steps each): the best solved 2 seeds "
            "of 3, most solved 0 or 1, and even solved runs often fell back later (the curve 'collapses'). Reference "
            "implementations give CartPole DQN about 500k steps. Section 4 shows which of the two tricks matters most.\n\n"
            "**Double DQN** (van Hasselt et al., 2016) is one line in `rl.py`: the online network *picks* the next "
            "action and the target network *scores* it. Taking the max over noisy Q estimates systematically "
            "overestimates them; splitting picking from scoring removes most of that bias."
        )])
    _out
    return


@app.cell
def _(DQN_ITERS, DQN_KW, dqn_ready, ex, jax, mo, rl):
    ablation = None
    if dqn_ready:
        ablation = {}
        for _name, _kw in [("DQN (replay + target net)", DQN_KW),
                           ("no replay (train on newest batch)", {**DQN_KW, "use_replay": False}),
                           ("no target network", {**DQN_KW, "use_target": False})]:
            _r = rl.train_dqn(ex, jax.random.key(1), total_iters=DQN_ITERS, **_kw)
            _c = _r["curve"]
            ablation[_name] = {"best": max(x["mean_return_last20"] for x in _c),
                               "final": _c[-1]["mean_return_last20"], "episodes": _c[-1]["episodes"]}
        _rows = "\n".join(f"| {k} | {v['best']:.0f} | {v['final']:.0f} | {v['episodes']} |" for k, v in ablation.items())
        _md = mo.md(f"""
## 4 · What the two tricks buy (one seed each — RL results are noisy)

| variant | best 20-episode average | final 20-episode average | episodes played |
|---|---|---|---|
{_rows}

More episodes means *shorter* episodes: a failing agent drops the pole quickly. Without a target network
the network chases its own moving predictions; without replay it learns only from the last few, highly
correlated frames and forgets. Treat a single run as an anecdote — the original paper averaged many games
and seeds.
""")
    else:
        _md = mo.md("")
    _md
    return (ablation,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Search: from Deep Blue to PUCT

    Deep Blue searched a game tree with **alpha-beta** pruning and a hand-written evaluation function.
    Go's tree is far too wide for that, so AlphaGo used **Monte Carlo tree search (MCTS)**: grow the tree
    selectively, many simulated games ("simulations") from the current position, each walking down the tree
    by a score that balances *how good a move has looked* against *how little it has been tried*.
    AlphaZero's score is **PUCT**:

    $$\text{PUCT}(s, a) = Q(s, a) + c \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

    $Q$ is the average value seen below that move, $P$ the network's prior for it, $N$ visit counts. Early
    on the network's intuition ($P$) decides what to look at; as visits accumulate, the measured value ($Q$)
    takes over. The search's final visit counts are a *better* policy than the network's prior — and that
    improved policy is what the network is trained to imitate. Search improves the network; the network
    focuses the search.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo):
    _p = lj.attempt(ex.puct, jnp.array([0.0, 0.5, 0.2]), jnp.array([0.7, 0.1, 0.2]), 16.0,
                    jnp.array([0.0, 7.0, 3.0]), 1.5)
    if _p.ok:
        _v = [float(x) for x in _p.value]
        _out = mo.md(f"""
**Your turn: `puct`.** Three moves after 16 visits to the parent:

| move | Q | prior | visits | PUCT |
|---|---|---|---|---|
| a | 0.0 | 0.7 | 0 | **{_v[0]:.3f}** |
| b | 0.5 | 0.1 | 7 | **{_v[1]:.3f}** |
| c | 0.2 | 0.2 | 3 | **{_v[2]:.3f}** |

The untried move with a strong prior is explored next, even though move b has the best value so far. ✅
""")
    else:
        _out = _p.view(mo)
    _out
    return


@app.cell
def _(mo, rosetta):
    rosetta.tabs(mo, "PUCT score (AlphaZero search)", solves="puct")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · AlphaZero on Connect Four — all on the GPU

    The ingredients, from the JAX ecosystem:

    * [`pgx`](https://github.com/sotetsuk/pgx) (Koyamada et al., 2023): board games written in JAX, so a
      thousand games step in parallel with `vmap` and live entirely on the GPU.
    * [`mctx`](https://github.com/google-deepmind/mctx) (Google DeepMind): Monte Carlo tree search in JAX,
      batched over games and compiled. We use its **Gumbel MuZero** policy (Danihelka et al., 2022), which
      improves the policy reliably even with few simulations.
    * A small residual conv network (`rl.az_apply`): the 6×7 board in, **policy logits** for the 7 columns
      and a **value** in [−1, 1] out.

    One AlphaZero iteration: (1) **self-play** — a batch of games where every move is chosen by search
    guided by the current network; record each position, the search's visit distribution π and, at the end,
    the outcome z from that position's player's view; (2) **train** — fit the network's policy to π and its
    value to z. Repeat.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo):
    _a = lj.attempt(ex.alphazero_loss, jnp.log(jnp.array([[0.5, 0.25, 0.25]])), jnp.array([0.5]),
                    jnp.array([[1.0, 0.0, 0.0]]), jnp.array([1.0]))
    az_ready = _a.ok
    (mo.md(f"**Your turn: `alphazero_loss`.** total {float(_a.value[0]):.4f} = policy CE "
           f"{float(_a.value[1][0]):.4f} (ln 2) + value MSE {float(_a.value[1][1]):.4f} ✅") if _a.ok else _a.view(mo))
    return (az_ready,)


@app.cell
def _(az_ready, ex, jax, mo, rl, time):
    az_live = None
    if az_ready:
        _env = rl.make_env()
        _p0 = rl.az_init(jax.random.key(0))
        _sp = rl.make_selfplay(_env, 256, 32)
        jax.block_until_ready(_sp(_p0, jax.random.key(1)))        # compile
        _t = time.perf_counter()
        _d = jax.block_until_ready(_sp(_p0, jax.random.key(2)))
        _dt = time.perf_counter() - _t
        _n = int(_d["valid"].sum())
        _hist_lines = []
        _params, _hist = rl.alphazero_loop(ex, jax.random.key(3), 4, batch=128, sims=16, train_steps=100,
                                           eval_every=4, eval_games=128, mcts_games=64, mcts_sims=64,
                                           log=lambda r: _hist_lines.append(r))
        az_live = {"games_per_s": 256 / _dt, "sims_per_s": _n * 32 / _dt, "positions": _n, "selfplay_s": _dt,
                   "params": _params, "hist": _hist}
    (mo.md(f"""
**Self-play throughput** (256 games at once, 32 simulations per move, untrained network):
{az_live['positions']:,} positions in {az_live['selfplay_s']:.2f} s — **{az_live['games_per_s']:.0f} games/s,
{az_live['sims_per_s'] / 1e3:,.0f}k MCTS simulations/s** on the RX 7600 XT (shared GPU timings vary).

**A short live run** (4 iterations of 128 games, 16 simulations, 100 training steps each):

| | vs random player (win / draw / loss) | vs pure MCTS, 64 random-rollout simulations |
|---|---|---|
| before training | {" / ".join(f"{x:.0%}" for x in az_live['hist'][0]['vs_random'])} | {" / ".join(f"{x:.0%}" for x in az_live['hist'][0]['vs_mcts'])} |
| after 4 iterations | {" / ".join(f"{x:.0%}" for x in az_live['hist'][-1]['vs_random'])} | {" / ".join(f"{x:.0%}" for x in az_live['hist'][-1]['vs_mcts'])} |

Even an untrained network beats a random player most of the time: 16 simulations of search already find
immediate wins and block immediate losses. The real test is pure MCTS with random playouts — classic MCTS,
before neural networks — which the network must learn to beat. Four iterations is a taste; `train.py` runs
the real thing (next cell).
""") if az_live else mo.md("*(AlphaZero runs once `alphazero_loss` works.)*"))
    return (az_live,)


@app.cell
def _(RUNS, json, mo, plt):
    _m = RUNS / "metrics.json"
    if _m.exists():
        _hist = json.loads(_m.read_text())["history"]
        _ev = [h for h in _hist if "vs_mcts" in h]
        _tr = [h for h in _hist if "loss" in h]
        _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.4))
        _it = [h["iteration"] for h in _ev]
        _a1.plot(_it, [h["vs_random"][0] for h in _ev], "o-", label="wins vs random")
        _a1.plot(_it, [h["vs_mcts"][0] for h in _ev], "s-", label="wins vs pure MCTS (64 rollouts)")
        _a1.plot(_it, [h["vs_mcts"][2] for h in _ev], "x--", label="losses vs pure MCTS")
        _a1.set(xlabel="iteration", ylabel="fraction of games", ylim=(-0.02, 1.02), title="AlphaZero (train.py)")
        _a1.legend(fontsize=8)
        _a2.plot([h["iteration"] for h in _tr], [h["policy_ce"] for h in _tr], label="policy cross-entropy")
        _a2.plot([h["iteration"] for h in _tr], [h["value_mse"] for h in _tr], label="value MSE")
        _a2.set(xlabel="iteration", title="training losses")
        _a2.legend(fontsize=8)
        _fig.tight_layout()
        _last = _ev[-1]
        _gps = sum(h["games_per_s"] for h in _tr) / len(_tr)
        _out = mo.vstack([_fig, mo.md(
            f"After {_last['iteration']} iterations: **{_last['vs_mcts'][0]:.0%} wins, {_last['vs_mcts'][1]:.0%} "
            f"draws, {_last['vs_mcts'][2]:.0%} losses** against pure MCTS, {_last['vs_random'][0]:.0%} wins vs "
            f"random. Average self-play speed {_gps:.0f} games/s. The policy loss falls as the network "
            "learns to predict where its own search will go; the value loss tracks how predictable outcomes "
            "are (it can rise as games between stronger players get closer).")])
    else:
        _out = mo.callout(mo.md(
            "**No long run yet.** Train AlphaZero for ~30 iterations (5–8 minutes on an idle GPU):\n\n"
            "`containers/run.sh python modules/34-deep-rl-alphazero/train.py`\n\n"
            "then re-run this cell for the win-rate curve."), kind="info")
    _out
    return


@app.cell
def _(mo):
    play_moves = mo.ui.text(value="3", label="your moves so far (columns 0–6, space-separated; you move first)")
    mo.vstack([mo.md("## 7 · Play against it\n\nYou are the player who moves first. Type your moves; the "
                     "agent (64 simulations per move) replies after each one."), play_moves])
    return (play_moves,)


@app.cell
def _(RUNS, az_live, jax, mo, play_moves, rl):
    if az_live is None:
        _out = mo.md("")
    else:
        _env = rl.make_env()
        _params = az_live["params"]
        _src = "the 4-iteration live agent"
        if (RUNS / "params.npz").exists():
            _params = rl.load_params(RUNS / "params.npz", rl.az_init(jax.random.key(0)))
            _src = "the train.py agent"
        _batch = jax.tree.map(lambda x: x[None], _env.init(jax.random.key(7)))
        _you = int(_batch.current_player[0])                 # you move first
        _log, _msg = [], ""
        try:
            _moves = [int(m) for m in play_moves.value.split()]
        except ValueError:
            _moves, _msg = [], "Use column numbers 0–6 separated by spaces."
        for _i, _m in enumerate(_moves):
            if bool(_batch.terminated[0]):
                break
            if not (0 <= _m <= 6) or not bool(_batch.legal_action_mask[0, _m]):
                _msg = f"Move {_m} is not legal here."
                break
            _batch = jax.vmap(_env.step)(_batch, jax.numpy.array([_m]))
            if bool(_batch.terminated[0]):
                break
            _a = rl.az_policy(_env, _params, jax.random.key(100 + _i), _batch, 64).action
            _log.append(int(_a[0]))
            _batch = jax.vmap(_env.step)(_batch, _a)
        _one = jax.tree.map(lambda x: x[0], _batch)
        if bool(_one.terminated):
            _r = float(_one.rewards[_you])
            _status = "**Game over:** " + ("you won! 🎉" if _r > 0 else "draw." if _r == 0 else "the agent won.")
        else:
            _status = "Your move."
        _out = mo.md(f"Playing {_src}. Agent replies so far: {_log}\n\n```\n{rl.board_text(_one)}\n```\n\n"
                     f"{_status} {_msg}")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · MuZero, and why mctx's API looks the way it does

    AlphaZero needs the rules: its search calls the real game to see what a move does. MuZero (2019)
    learned that too, with three networks: a **representation** network turns the observation into a hidden
    state, a **dynamics** network predicts the next hidden state and reward for an action, and a
    **prediction** network gives policy and value. Search then runs entirely inside the learned model — which
    let the same algorithm play Atari, where there's no simulator to search with.

    That's why `mctx` asks for a `root` (prior, value, embedding) and a `recurrent_fn(params, key, action,
    embedding)` that returns reward, discount, prior and value for the *next* embedding: it's MuZero-shaped.
    For AlphaZero we cheat honestly — our `recurrent_fn` (`rl.az_recurrent_fn`) uses the real `pgx` game as
    the "dynamics" and the board itself as the "embedding". A two-player game is handled with
    `discount = −1`: the opponent's good is your bad.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · From games to language models — and to your rover

    * **Policy gradients.** DQN learns values; the other family learns the policy directly and nudges it
      toward actions that did better than expected. PPO (Schulman et al., 2017) is its workhorse; GRPO, which
      you implemented in Module 26, is its value-network-free cousin used for reasoning models.
    * **RLHF.** Replace the game's win/loss with a reward model trained on human preferences and you get the
      recipe that turned GPT-3 into InstructGPT and ChatGPT — Module 35.
    * **Robots.** Your own *jezero-lab* project trains PPO policies for the Open Source Rover in Gazebo — the
      same agent/environment/reward loop, with a physics simulator instead of a board game, and the hard
      parts (sim-to-real, reward design, seed variance) that games hide.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "It trains on its own self-play games, with search producing better move targets than the network alone": "a",
            "It memorizes human expert games, like the original AlphaGo": "b",
            "It searches deeper than Deep Blue with alpha-beta pruning": "c",
        },
        label="**Check yourself:** what is the core idea of AlphaZero?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right. Search improves on the network's policy; the network learns to "
                              "predict the search's result and the game's outcome; repeat — no human data."),
                        kind="success"),
        "b": mo.callout(mo.md("That was the *original* AlphaGo (2016). AlphaGo Zero and AlphaZero dropped "
                              "human games entirely."), kind="warn"),
        "c": mo.callout(mo.md("No — AlphaZero searches far *fewer* positions than Deep Blue, using MCTS "
                              "guided by a network instead of brute-force alpha-beta."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    * Sutton & Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html)
      (2nd ed., 2018) — the textbook, free online.
    * OpenAI, [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) — a practical tour of the
      algorithms, with code.
    * Koyamada et al., [pgx](https://github.com/sotetsuk/pgx) — JAX game simulators, including an AlphaZero
      example this lesson's structure follows.
    * Google DeepMind, [mctx](https://github.com/google-deepmind/mctx) — Monte Carlo tree search in JAX.

    **Next:** Module 35 — learning from preferences: RLHF and DPO.
    """)
    return


if __name__ == "__main__":
    app.run()
