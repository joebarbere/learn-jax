import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_central_difference(impl):
    assert impl.central_difference(math.sin, 0.7) == pytest.approx(math.cos(0.7), abs=1e-9)
    assert impl.central_difference(lambda x: x**3, 2.0) == pytest.approx(12.0, abs=1e-6)
    # Central, not forward: for x^2 at 1 with a big step, central is exact (2.0); forward gives 2.1.
    assert impl.central_difference(lambda x: x * x, 1.0, h=0.1) == pytest.approx(2.0, abs=1e-12)


def test_grad_check(impl):
    def f(x):
        return float(np.sum(np.sin(x) * x**2))

    def good(x):
        return np.cos(x) * x**2 + 2 * x * np.sin(x)

    def buggy(x):
        return np.cos(x) * x**2  # forgot the product rule's second term

    x = np.array([0.3, -1.2, 2.0, 0.9])
    x_before = x.copy()
    assert impl.grad_check(f, good, x) < 1e-7
    assert impl.grad_check(f, buggy, x) > 1e-2
    np.testing.assert_array_equal(x, x_before)  # must not modify the input
    assert isinstance(impl.grad_check(f, good, x), float)


def _close(a, b):
    return a == pytest.approx(float(b), rel=1e-4, abs=1e-6)


def test_value_matches_jax(impl):
    V = impl.Value
    a, b = V(2.0), V(-3.0)
    c = a * b + a**2
    d = c.tanh() + (a / b).exp() - 1.5 * b
    d.backward()

    def f(a, b):
        c = a * b + a**2
        return jnp.tanh(c) + jnp.exp(a / b) - 1.5 * b

    ga, gb = jax.grad(f, argnums=(0, 1))(2.0, -3.0)
    assert _close(d.data, f(2.0, -3.0))
    assert _close(a.grad, ga)
    assert _close(b.grad, gb)


def test_value_reuse_and_python_numbers(impl):
    V = impl.Value
    x = V(3.0)
    y = x * x + 2 * x + 1 - x / 2  # x used four times: gradients must accumulate
    y.backward()
    assert _close(y.data, 9 + 6 + 1 - 1.5)
    assert _close(x.grad, 2 * 3 + 2 - 0.5)

    z = V(0.5)
    w = 1 - z + 3 / z + (-z).exp()
    w.backward()
    assert _close(z.grad, -1 - 3 / 0.25 - math.exp(-0.5))


def test_neuron_grad(impl):
    w = np.array([0.5, -0.4, 0.2])
    x = np.array([1.0, 2.0, -1.5])
    b, y = -0.1, 0.8

    def loss(w, b):
        return (jnp.tanh(w @ x + b) - y) ** 2

    gw, gb = jax.grad(loss, argnums=(0, 1))(jnp.asarray(w), b)
    dw, db = impl.neuron_grad(w, b, x, y)
    np.testing.assert_allclose(np.asarray(dw), np.asarray(gw), rtol=1e-4, atol=1e-6)
    assert np.shape(dw) == w.shape
    assert _close(db, gb)


def test_jacobian_via_vjp(impl):
    def f(x):
        return jnp.stack([x[0] * x[1], jnp.sin(x[2]), jnp.sum(x**2), x[1] - x[0]])

    x = jnp.array([0.5, -1.0, 2.0])
    J = impl.jacobian_via_vjp(f, x)
    assert J.shape == (4, 3)
    np.testing.assert_allclose(np.asarray(J), np.asarray(jax.jacobian(f)(x)), rtol=1e-5, atol=1e-6)
