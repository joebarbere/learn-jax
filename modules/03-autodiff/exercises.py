"""Module 03 exercises: derivatives, gradient checking, and autodiff built from scratch.

Replace each `raise NotImplementedError` with your code, then run `make test N=03`.
The lesson notebook calls these functions, so its "Your turn" cells update as you go.
Use plain Python / NumPy unless a docstring says JAX.
"""
from __future__ import annotations

import math  # noqa: F401  (you will want math.tanh / math.exp in Value)
from collections.abc import Callable

import numpy as np


def central_difference(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Approximate f'(x) with the central difference (f(x + h) - f(x - h)) / (2h).

    Its error shrinks like h**2, versus h for the one-sided (f(x + h) - f(x)) / h.
    """
    raise NotImplementedError


def grad_check(f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], np.ndarray],
               x: np.ndarray, h: float = 1e-5) -> float:
    """Compare an analytic gradient against central differences; return the worst relative error.

    - `x` is a 1-D float64 array; `f(x)` returns a float; `grad_f(x)` returns an array shaped like x.
    - For each coordinate i, estimate df/dx_i by nudging only x[i] by ±h (don't modify x in place).
    - Relative error per coordinate: |a - n| / max(|a| + |n|, 1e-12), where a is analytic and n numeric.
    - Return the maximum over coordinates as a Python float. Rule of thumb: < 1e-7 is right,
      > 1e-3 means a bug.
    """
    raise NotImplementedError


class Value:
    """A scalar that remembers how it was computed, so it can backpropagate (micrograd-style).

    Each operation returns a new Value whose `_prev` holds its inputs and whose `_backward` closure
    knows how to push `out.grad` into the inputs' `.grad` (chain rule: input.grad += local_derivative
    * out.grad). Use `+=`, not `=`: a Value used twice receives gradient from both uses.

    Implement: __add__/__radd__, __mul__/__rmul__, __pow__ (number exponent only), __neg__,
    __sub__/__rsub__, __truediv__/__rtruediv__, tanh(), exp(), and backward().
    Plain numbers on either side should work: `2 * v`, `v + 1`, `1 - v`, `3 / v`.
    """

    def __init__(self, data: float, _prev: tuple = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __pow__(self, k: float):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def exp(self):
        raise NotImplementedError

    def backward(self) -> None:
        """Set self.grad = 1, then call every node's _backward in reverse topological order."""
        raise NotImplementedError


def neuron_grad(w: np.ndarray, b: float, x: np.ndarray, y: float) -> tuple[np.ndarray, float]:
    """Gradient of one neuron's squared error, derived by hand with the chain rule (no autodiff).

    Forward: z = w·x + b,  a = tanh(z),  L = (a - y)**2.
    Return (dL/dw, dL/db): an array shaped like w, and a float.
    Hint: dL/da = 2(a - y), da/dz = 1 - a**2, dz/dw = x, dz/db = 1.
    """
    raise NotImplementedError


def jacobian_via_vjp(f: Callable, x):
    """Stretch: the full Jacobian of f: R^n -> R^m at x (shape (m, n)) using JAX's reverse mode.

    `jax.vjp(f, x)` returns (y, vjp_fn); `vjp_fn(u)` gives the row-vector-Jacobian product u^T J
    (as a 1-tuple). Feed it every one-hot u (rows of jnp.eye(m)) at once with `jax.vmap`.
    """
    raise NotImplementedError
