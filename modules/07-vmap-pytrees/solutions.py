"""Module 07 reference solutions. Try the exercises first — `make test N=07` grades yours."""
import jax
import jax.numpy as jnp


def batched_dot(a, b):
    return jax.vmap(jnp.dot)(a, b)


def pairwise_distances(x, y):
    def dist(u, v):
        return jnp.sqrt(jnp.sum((u - v) ** 2))

    over_y = jax.vmap(dist, in_axes=(None, 0))  # one u against every v
    return jax.vmap(over_y, in_axes=(0, None))(x, y)  # every u


def per_example_grads(loss_fn, params, xs, ys):
    return jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))(params, xs, ys)


def tree_l2_norm(tree):
    return jnp.sqrt(sum(jnp.sum(leaf.astype(jnp.float32) ** 2) for leaf in jax.tree.leaves(tree)))


def tree_axpy(a, x, y):
    return jax.tree.map(lambda xi, yi: a * xi + yi, x, y)


def init_mlp(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        {"w": jax.random.normal(k, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in), "b": jnp.zeros(fan_out)}
        for k, fan_in, fan_out in zip(keys, sizes[:-1], sizes[1:], strict=True)
    ]


def mlp_forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    last = params[-1]
    return x @ last["w"] + last["b"]


def count_params(params) -> int:
    return int(sum(leaf.size for leaf in jax.tree.leaves(params)))
