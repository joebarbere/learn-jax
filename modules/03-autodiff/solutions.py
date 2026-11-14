"""Module 03 reference solutions. Try the exercises first — `make test N=03` grades yours."""
from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


def central_difference(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


def grad_check(f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], np.ndarray],
               x: np.ndarray, h: float = 1e-5) -> float:
    x = np.asarray(x, dtype=np.float64)
    analytic = np.asarray(grad_f(x), dtype=np.float64)
    worst = 0.0
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h
        numeric = (f(x + step) - f(x - step)) / (2 * h)
        a = analytic[i]
        rel = abs(a - numeric) / max(abs(a) + abs(numeric), 1e-12)
        worst = max(worst, float(rel))
    return worst


class Value:
    def __init__(self, data: float, _prev: tuple = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    @staticmethod
    def _wrap(other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, k: float):
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1

    def __rtruediv__(self, other):
        return self._wrap(other) * self**-1

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        order, seen = [], set()

        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for child in v._prev:
                    visit(child)
                order.append(v)

        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


def neuron_grad(w: np.ndarray, b: float, x: np.ndarray, y: float) -> tuple[np.ndarray, float]:
    w, x = np.asarray(w, dtype=np.float64), np.asarray(x, dtype=np.float64)
    a = np.tanh(w @ x + b)
    dL_dz = 2 * (a - y) * (1 - a**2)
    return dL_dz * x, float(dL_dz)


def jacobian_via_vjp(f: Callable, x):
    y, vjp_fn = jax.vjp(f, x)
    basis = jnp.eye(jnp.size(y), dtype=y.dtype)
    return jax.vmap(vjp_fn)(basis)[0]
