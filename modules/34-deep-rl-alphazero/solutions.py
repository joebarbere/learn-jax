"""Module 34 reference solutions. Try the exercises first — `make test N=34` grades yours."""
import jax
import jax.numpy as jnp


def q_learning_update(Q, s, a, r, s2, done, lr, gamma):
    target = r + gamma * (1.0 - done) * jnp.max(Q[s2])
    return Q.at[s, a].add(lr * (target - Q[s, a]))


def epsilon_greedy(key, q_values, eps):
    k_explore, k_action = jax.random.split(key)
    greedy = jnp.argmax(q_values, axis=-1)
    random = jax.random.randint(k_action, greedy.shape, 0, q_values.shape[-1])
    explore = jax.random.uniform(k_explore, greedy.shape) < eps
    return jnp.where(explore, random, greedy)


def td_target(r, gamma, done, q_next_target):
    return r + gamma * (1.0 - done) * jnp.max(q_next_target, axis=-1)


def dqn_loss(q, a, target):
    q_a = jnp.take_along_axis(q, a[:, None], axis=-1)[:, 0]
    err = q_a - jax.lax.stop_gradient(target)
    huber = jnp.where(jnp.abs(err) <= 1.0, 0.5 * err**2, jnp.abs(err) - 0.5)
    return jnp.mean(huber)


def replay_add(buf, batch):
    cap = buf["action"].shape[0]
    n = batch["action"].shape[0]
    idx = (buf["ptr"] + jnp.arange(n)) % cap
    out = {k: buf[k].at[idx].set(batch[k]) for k in batch}
    out["ptr"] = (buf["ptr"] + n) % cap
    out["size"] = jnp.minimum(buf["size"] + n, cap)
    return out


def replay_sample(buf, key, batch_size):
    idx = jax.random.randint(key, (batch_size,), 0, jnp.maximum(buf["size"], 1))
    return {k: v[idx] for k, v in buf.items() if k not in ("ptr", "size")}


def alphazero_loss(policy_logits, value, target_policy, target_value):
    logp = jax.nn.log_softmax(policy_logits, axis=-1)
    policy_ce = -jnp.mean(jnp.sum(target_policy * logp, axis=-1))
    value_mse = jnp.mean((value - target_value) ** 2)
    return policy_ce + value_mse, (policy_ce, value_mse)


def puct(q, prior, n_parent, n_child, c):
    return q + c * prior * jnp.sqrt(n_parent) / (1.0 + n_child)
