"""Glue between a lesson notebook and the learner's exercise file.

Each module directory holds `exercises.py` (stubs you fill in) and `solutions.py` (reference answers).
Lessons call `load_exercises(here)` to import yours, or the reference ones when LEARNJAX_SOLUTIONS=1
(used by `make check` so every lesson can run end to end). Lessons wrap calls in `attempt(...)` so a
not-yet-done exercise shows a friendly "your turn" note instead of crashing the notebook.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


def using_solutions() -> bool:
    return os.environ.get("LEARNJAX_SOLUTIONS", "") == "1"


def load_exercises(here: str | Path, *, solutions: bool | None = None) -> ModuleType:
    """Import exercises.py (or solutions.py) from a module directory, freshly each call."""
    here = Path(here)
    if here.is_file():
        here = here.parent
    use_solutions = using_solutions() if solutions is None else solutions
    name = "solutions" if use_solutions else "exercises"
    path = here / f"{name}.py"
    mod_name = f"learnjax_{here.name.replace('-', '_')}_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Attempt:
    """Outcome of calling one exercise function."""

    name: str
    ok: bool
    value: Any = None
    todo: bool = False
    error: str = ""

    def view(self, mo, show_value: bool = True):
        """Render as a marimo element: the value, a 'your turn' callout, or the error."""
        if self.ok:
            return mo.md(f"`{self.name}` ✅") if not show_value else self.value
        if self.todo:
            return mo.callout(
                mo.md(f"**Your turn:** implement `{self.name}` in `exercises.py`, then re-run this cell "
                      f"(`make test N` checks it)."),
                kind="info",
            )
        return mo.callout(mo.md(f"`{self.name}` raised an error:\n\n```\n{self.error}\n```"), kind="danger")


def attempt(fn: Callable, *args, **kwargs) -> Attempt:
    name = getattr(fn, "__name__", repr(fn))
    try:
        return Attempt(name, True, fn(*args, **kwargs))
    except NotImplementedError:
        return Attempt(name, False, todo=True)
    except Exception:  # noqa: BLE001 - show any learner error in the notebook
        return Attempt(name, False, error=traceback.format_exc(limit=3))
