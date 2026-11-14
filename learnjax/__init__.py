"""Shared helpers for the learn-jax course notebooks (kept deliberately small).

Imports are lazy so `learnjax.data` also works in the PyTorch image, which has no JAX.
"""
from importlib import import_module

_EXPORTS = {
    "Attempt": "learnjax.course",
    "attempt": "learnjax.course",
    "load_exercises": "learnjax.course",
    "using_solutions": "learnjax.course",
    "timeit": "learnjax.timing",
}

__all__ = ["Attempt", "attempt", "load_exercises", "timeit", "using_solutions"]


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'learnjax' has no attribute {name!r}")
