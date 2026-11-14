"""Module 09 reference solutions. Try the exercises first — `make test N=09` grades yours."""
import jax
import jax.numpy as jnp


def train_val_test_split(key, n: int, fractions=(0.7, 0.15, 0.15)):
    perm = jax.random.permutation(key, n)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def polynomial_features(x, degree: int):
    x = jnp.asarray(x)
    return x[:, None] ** jnp.arange(degree + 1)


def ridge_closed_form(X, y, lam: float):
    X, y = jnp.asarray(X), jnp.asarray(y)
    d = X.shape[1]
    return jnp.linalg.solve(X.T @ X + lam * jnp.eye(d), X.T @ y)


def bce_with_logits(logits, labels):
    z, y = jnp.asarray(logits), jnp.asarray(labels, dtype=jnp.float32)
    return jnp.mean(jnp.maximum(z, 0) - z * y + jnp.log1p(jnp.exp(-jnp.abs(z))))


def logistic_fit(X, y, steps: int = 500, lr: float = 0.5):
    X, y = jnp.asarray(X, jnp.float32), jnp.asarray(y, jnp.float32)

    def loss(params):
        w, b = params
        return bce_with_logits(X @ w + b, y)

    @jax.jit
    def step(params):
        grads = jax.grad(loss)(params)
        return jax.tree.map(lambda p, g: p - lr * g, params, grads)

    params = (jnp.zeros(X.shape[1]), jnp.zeros(()))
    for _ in range(steps):
        params = step(params)
    return params


def accuracy(predictions, labels):
    return float(jnp.mean(jnp.asarray(predictions) == jnp.asarray(labels)))


def confusion_matrix(y_true, y_pred, n_classes: int):
    m = jnp.zeros((n_classes, n_classes), jnp.int32)
    return m.at[jnp.asarray(y_true), jnp.asarray(y_pred)].add(1)
