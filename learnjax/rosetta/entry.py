"""The Rosetta `Entry` record, the topic list, and helpers for building test inputs."""
from __future__ import annotations

import textwrap
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np


class Entry(NamedTuple):
    name: str  # what the course calls it
    topic: str  # one of TOPICS
    plain: str  # one or two plain-language sentences, intuition first
    math: str  # LaTeX for the formula as the course writes it
    paper: str  # other ways papers write it (LaTeX allowed in $...$), or ""
    shapes: str  # input -> output shapes
    call: str  # how the test calls each snippet, e.g. "softmax(x)"
    inputs: Callable[[np.random.Generator], dict[str, Any]]  # shared test inputs (numpy arrays and scalars)
    python: str
    numpy: str
    jax: str  # the idiomatic JAX/optax call
    torch: str
    jax_by_hand: str = ""  # the formula in jnp, when `jax` is a library call that hides the math
    einsum: str = ""  # index form, for contractions
    notes: tuple[str, ...] = ()  # where the versions differ: conventions, traps
    module: str = ""
    also: tuple[str, ...] = ()
    apis: tuple[str, ...] = ()  # full JAX/optax/Flax names this entry covers ("jax.nn.softmax"); see coverage.py
    check: str = "close"  # "close": outputs match; "stats": random samples, compare mean and std
    tol: float = 0.0  # 0 = default tolerance (rtol 2e-4 for "close", 0.05 absolute for "stats")


TOPICS = ["Arrays and reductions", "Elementwise math", "Linear algebra", "Calculus and autodiff",
          "Random numbers", "Control flow and transforms", "Probability and information", "Optimization",
          "Neural network layers", "Attention and transformers", "Sampling and inference",
          "Sequence and state-space models", "Generative models", "Reinforcement learning and preferences",
          "Retrieval and evaluation"]

VARIANTS = ["python", "numpy", "jax", "jax_by_hand", "torch"]
LABELS = {"python": "Python", "numpy": "NumPy", "jax": "JAX", "jax_by_hand": "JAX, by hand", "torch": "PyTorch"}


def code(s: str) -> str:
    return textwrap.dedent(s).strip("\n")


def orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))


def symmetric(rng: np.random.Generator, n: int) -> np.ndarray:
    m = rng.normal(size=(n, n))
    return m + m.T


def probs(rng: np.random.Generator, shape) -> np.ndarray:
    e = np.exp(rng.normal(size=shape))
    return e / e.sum(-1, keepdims=True)

