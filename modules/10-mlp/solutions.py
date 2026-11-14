"""Module 10 reference solutions. Try the exercises first — `make test N=10` grades yours."""
import jax
import jax.numpy as jnp
import optax


def he_init(key, fan_in: int, fan_out: int):
    return jax.random.normal(key, (fan_in, fan_out), jnp.float32) * jnp.sqrt(2.0 / fan_in)


def init_mlp(key, sizes: list[int]) -> list[dict]:
    keys = jax.random.split(key, len(sizes) - 1)
    return [{"w": he_init(k, a, b), "b": jnp.zeros((b,), jnp.float32)}
            for k, a, b in zip(keys, sizes[:-1], sizes[1:], strict=True)]


def mlp_apply(params: list[dict], x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def loss_fn(params: list[dict], x, y):
    logp = jax.nn.log_softmax(mlp_apply(params, x))
    return -jnp.take_along_axis(logp, y[:, None], axis=1).mean()


def data_batches(key, n: int, batch_size: int) -> list:
    perm = jax.random.permutation(key, n)
    return [perm[i * batch_size:(i + 1) * batch_size] for i in range(n // batch_size)]


@jax.jit
def _correct(params, x, y):
    """Number of correct predictions in one chunk, compiled (one fused GPU program per chunk shape)."""
    return (mlp_apply(params, x).argmax(-1) == y).sum()


def accuracy(params: list[dict], x, y, batch_size: int = 1000) -> float:
    correct = sum(_correct(params, x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(x), batch_size))
    return int(correct) / len(x)


def train_step(params, opt_state, x, y, optimizer):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss
