"""Module 34 exercises: the moving parts of Q-learning, DQN and AlphaZero.

Replace each `raise NotImplementedError` with your code, then run `make test N=34`. The lesson runs a
gridworld, a CartPole DQN and an AlphaZero for Connect Four on top of these functions, so the cells marked
**Your turn** come alive as you finish them. Everything must be jit-friendly: no Python `if` on array
values (use `jnp.where`), no Python loops over the batch.
"""
import jax  # noqa: F401  (you'll want jax.random)
import jax.numpy as jnp  # noqa: F401


def q_learning_update(Q, s, a, r, s2, done, lr, gamma):
    """One tabular Q-learning step (Watkins 1989). Returns the updated table.

    Q: float array [n_states, n_actions]. s, a, s2: int scalars. r: float. done: 0.0 or 1.0 (float or bool).
    target = r + gamma * max_a' Q[s2, a'] — but just r when `done` (no future after a terminal state).
    Q[s, a] moves a fraction `lr` of the way toward the target; every other entry is unchanged.
    Use `Q.at[s, a].add(...)` (JAX arrays are immutable).
    """
    raise NotImplementedError


def epsilon_greedy(key, q_values, eps):
    """Pick actions: with probability `eps` a uniformly random action, otherwise argmax of `q_values`.

    q_values: [..., n_actions] (a single state [A] or a batch [B, A]). Returns int actions of shape [...].
    Each row explores independently. eps=0 must always be greedy; eps=1 always random.
    """
    raise NotImplementedError


def td_target(r, gamma, done, q_next_target):
    """DQN's bootstrap target r + gamma * (1 - done) * max_a Q_target(s', a), batched.

    r, done: [B] floats (done is 1.0 at the end of an episode). q_next_target: [B, n_actions] from the
    *target* network. Returns [B].
    """
    raise NotImplementedError


def dqn_loss(q, a, target):
    """Mean Huber loss (delta = 1) between Q(s, a) and the target — the error clipping of DQN (2015).

    q: [B, n_actions] online-network values, a: [B] int actions taken, target: [B] (treat as a constant:
    no gradient flows into it — wrap it in `jax.lax.stop_gradient`).
    Huber(e) = 0.5 e^2 if |e| <= 1 else |e| - 0.5.
    """
    raise NotImplementedError


def replay_add(buf, batch):
    """Write a batch of n transitions into a fixed-size ring buffer. Returns the new buffer.

    buf: dict of arrays with a leading capacity axis C (e.g. "obs" [C, 4], "action" [C], ...) plus int
    scalars "ptr" (next write slot) and "size" (number of valid entries, at most C).
    batch: dict with the same keys as the arrays (no "ptr"/"size"), each with leading axis n (n <= C).
    Write slots ptr, ptr+1, ..., wrapping around modulo C (oldest data is overwritten); advance ptr by n
    modulo C; size = min(size + n, C).
    """
    raise NotImplementedError


def replay_sample(buf, key, batch_size):
    """Sample `batch_size` transitions uniformly (with replacement) from the valid entries [0, size).

    Returns a dict with the same array keys as `buf` (not "ptr"/"size"), each with leading axis batch_size.
    """
    raise NotImplementedError


def alphazero_loss(policy_logits, value, target_policy, target_value):
    """AlphaZero's loss: policy cross-entropy against the MCTS visit distribution + value MSE.

    policy_logits: [B, A]; value: [B] (the network's prediction, in [-1, 1]); target_policy: [B, A]
    (non-negative, rows sum to 1 — MCTS's improved policy); target_value: [B] (the game outcome z from the
    player-to-move's point of view: +1 win, -1 loss, 0 draw).
    Returns (total, (policy_ce, value_mse)) where policy_ce = mean over B of -sum_a pi_a log softmax(logits)_a,
    value_mse = mean (value - z)^2, total = policy_ce + value_mse.
    """
    raise NotImplementedError


def puct(q, prior, n_parent, n_child, c):
    """AlphaZero's PUCT score for choosing which child to explore in MCTS (elementwise):

        q + c * prior * sqrt(n_parent) / (1 + n_child)

    q: mean value of the child so far; prior: the network's probability for that move; n_parent: visits
    to the parent; n_child: visits to the child; c: exploration constant. Unvisited moves with a high
    prior get explored first; as visits pile up, the value q takes over.
    """
    raise NotImplementedError
