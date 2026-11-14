"""Module 10 exercises: a multilayer perceptron, written by hand in pure JAX.

Replace each `raise NotImplementedError` with your code, then run `make test N=10`.

Parameters are a *pytree*: a list with one dict per layer, `{"w": (fan_in, fan_out), "b": (fan_out,)}`.
That's all a neural network is — nested containers of arrays, plus a function that uses them. When
these functions work, the lesson trains your network on Fashion-MNIST.
"""
import jax  # noqa: F401  (you will need these)
import jax.numpy as jnp  # noqa: F401
import optax  # noqa: F401


def he_init(key, fan_in: int, fan_out: int):
    """A (fan_in, fan_out) float32 weight matrix for a layer followed by ReLU ("He"/"Kaiming" init).

    Entries are drawn from a normal distribution with mean 0 and standard deviation sqrt(2 / fan_in).
    That scale keeps the size of activations roughly constant from layer to layer (Module 11 shows
    what goes wrong otherwise). Use jax.random.normal.
    """
    raise NotImplementedError


def init_mlp(key, sizes: list[int]) -> list[dict]:
    """Parameters for an MLP with layer widths `sizes`, e.g. [784, 256, 256, 10].

    Return a list with len(sizes) - 1 dicts, layer i being
        {"w": he_init(k_i, sizes[i], sizes[i+1]), "b": zeros(sizes[i+1], float32)}
    Give every layer its own key: split `key` with jax.random.split — never reuse a key.
    """
    raise NotImplementedError


def mlp_apply(params: list[dict], x):
    """Forward pass. x: (batch, sizes[0]) float array -> logits: (batch, sizes[-1]).

    Every layer computes x @ w + b. Apply jax.nn.relu after every layer *except the last*: the last
    layer's outputs are logits (unnormalized class scores), which may be negative.
    """
    raise NotImplementedError


def loss_fn(params: list[dict], x, y):
    """Mean cross-entropy between the MLP's logits for x and integer class labels y of shape (batch,).

    This is Module 04's cross-entropy: -mean(log_softmax(logits)[i, y_i]). Use jax.nn.log_softmax
    (numerically stable) rather than log(softmax(...)).
    """
    raise NotImplementedError


def data_batches(key, n: int, batch_size: int) -> list:
    """Shuffle the indices 0..n-1 with `key` and cut them into batches of `batch_size`.

    Return a list of int arrays, each of length batch_size. Drop the leftover indices that don't fill
    a whole batch (so every step sees the same batch shape and jit compiles only once).
    Use jax.random.permutation. The same key must give the same batches.
    """
    raise NotImplementedError


def accuracy(params: list[dict], x, y, batch_size: int = 1000) -> float:
    """Fraction of examples whose highest logit is the true label, as a Python float.

    Evaluate in chunks of `batch_size` rows (so 10,000 test images don't all hit the GPU at once),
    and don't drop the last, smaller chunk — every example counts.

    Speed tip: put the per-chunk work (forward pass, argmax, count) in a small function decorated with
    @jax.jit, and convert to a Python number only once at the end. Without jit, every operation is
    dispatched to the GPU separately from Python and evaluation becomes the slowest part of training.
    """
    raise NotImplementedError


def train_step(params, opt_state, x, y, optimizer):
    """One optimization step. Pure: return (new_params, new_opt_state, loss) and change nothing.

    1. loss and grads of loss_fn w.r.t. params at (x, y) — jax.value_and_grad
    2. updates, new_opt_state = optimizer.update(grads, opt_state, params)
       (params are passed so AdamW can apply weight decay)
    3. new_params = optax.apply_updates(params, updates)
    """
    raise NotImplementedError
