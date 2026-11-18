"""Rosetta: one math operation, five ways — the formula, plain Python, NumPy, JAX and PyTorch.

Each `Entry` pairs the notation a paper uses with code that computes it, so you can read a formula and find
the call, or translate a NumPy or PyTorch snippet into JAX. Every code field is a runnable snippet that defines
the function named in `call`; `run()` executes one variant on shared inputs, and the course's tests check that
all variants agree (tests/test_rosetta.py); random-sampling entries (`check="stats"`) are compared by mean and
standard deviation, since no two libraries draw the same random numbers. Entries live one file per area in
this package (`PARTS`); `coverage.py` lists the JAX names that are plumbing rather than math. `tabs(mo, name)` shows an entry in a lesson; `tools/gen_rosetta.py`
writes docs/ROSETTA.md.

Snippet conventions: plain Python uses only `math`, `random` and lists (loops, the formula read literally); NumPy uses
`np`; JAX uses `jax`, `jnp`, `optax`; PyTorch uses `torch` and `F` (torch.nn.functional). `module` is the
display label of the lesson that introduces the operation, `also` other lessons where it matters.
"""
from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np

from learnjax.rosetta.entry import LABELS, TOPICS, VARIANTS, Entry, code  # noqa: F401  (re-exported)

# One file per area; each defines ENTRIES. Order here is the order in docs/ROSETTA.md within a topic.
PARTS = ["basics", "arrays", "calculus_random", "training", "layers_attention", "milestones", "frontier"]
ENTRIES: list[Entry] = [e for p in PARTS for e in importlib.import_module(f"learnjax.rosetta.{p}").ENTRIES]
BY_NAME = {e.name: e for e in ENTRIES}


# ---------------------------------------------------------------------------------------------- running
def _namespace(variant: str) -> dict[str, Any]:
    import math
    import random

    ns: dict[str, Any] = {"math": math, "random": random}
    if variant == "numpy":
        ns["np"] = np
    elif variant in ("jax", "jax_by_hand"):
        import jax
        import jax.numpy as jnp
        import optax

        ns.update(jax=jax, jnp=jnp, optax=optax)
    elif variant == "torch":
        import torch
        import torch.nn.functional as F

        ns.update(torch=torch, F=F)
    return ns


def _convert(x: Any, variant: str) -> Any:
    if isinstance(x, dict):
        return {k: _convert(v, variant) for k, v in x.items()}
    if not isinstance(x, np.ndarray):
        return x
    if variant == "python":
        return x.tolist()
    if variant == "numpy":
        return x
    if variant in ("jax", "jax_by_hand"):
        import jax.numpy as jnp

        return jnp.asarray(x, dtype=jnp.float32 if x.dtype.kind == "f" else jnp.int32)
    import torch

    return torch.as_tensor(x, dtype=torch.float32 if x.dtype.kind == "f" else torch.int64)


def to_numpy(x: Any) -> Any:
    """Any variant's output as NumPy (dicts stay dicts)."""
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if hasattr(x, "detach"):  # torch
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def run(entry: Entry | str, variant: str, seed: int = 0) -> Any:
    """Execute one variant's snippet on the entry's test inputs; returns its output as NumPy."""
    e = BY_NAME[entry] if isinstance(entry, str) else entry
    src = getattr(e, variant)
    if not src:
        raise ValueError(f"{e.name!r} has no {variant} version")
    ns = _namespace(variant)
    exec(code(src), ns)  # noqa: S102 - the course's own registry snippets, not user input
    ns.update(_convert(e.inputs(np.random.default_rng(seed)), variant))
    return to_numpy(eval(e.call, ns))


# ---------------------------------------------------------------------------------------------- checking
def assert_close(e: Entry, a: Any, b: Any, name: str) -> None:
    """Raise AssertionError unless output `a` matches reference `b` (by value, or by mean/std for "stats")."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), name
        for k in a:
            assert_close(e, a[k], b[k], f"{name}[{k}]")
        return
    if e.check == "stats":  # random samples: same distribution, not the same numbers
        tol = e.tol or 0.05
        assert abs(np.mean(a) - np.mean(b)) < tol, f"{name}: mean {np.mean(a):.4f} vs {np.mean(b):.4f}"
        assert abs(np.std(a) - np.std(b)) < tol, f"{name}: std {np.std(a):.4f} vs {np.std(b):.4f}"
        return
    assert np.shape(a) == np.shape(b), f"{name}: shape {np.shape(a)} vs {np.shape(b)}"
    np.testing.assert_allclose(a, b, rtol=e.tol or 2e-4, atol=2e-5, err_msg=name)


def check(e: Entry, variants: list[str] | None = None) -> None:
    """Run every variant (default: all but torch) and compare it with the NumPy one."""
    ref = run(e, "numpy")
    for v in variants or ["python", "jax"] + (["jax_by_hand"] if e.jax_by_hand else []):
        assert_close(e, run(e, v), ref, f"{e.name} ({v})")


def rule_errors(e: Entry) -> list[str]:
    """Registry rules every entry follows (structure, not numbers)."""
    import re

    from learnjax import syllabus

    labels = {syllabus.label(slug) for _, slug, _, _ in syllabus.MODULES}
    uses = lambda src, lib: re.search(rf"\b{lib}\.", src)
    errs = []
    if e.topic not in TOPICS:
        errs.append(f"unknown topic {e.topic!r}")
    if e.check not in ("close", "stats"):
        errs.append(f"check must be close or stats, not {e.check!r}")
    if e.module not in labels or not set(e.also) <= labels:
        errs.append(f"module/also must be display labels: {e.module!r} {e.also!r}")
    if not (e.plain and e.math and e.shapes and e.python and e.numpy and e.jax and e.torch):
        errs.append("plain, math, shapes, python, numpy, jax and torch are all required")
    for lib in ("np", "jnp", "jax", "torch", "optax"):  # plain Python means plain Python
        if uses(e.python, lib):
            errs.append(f"python version uses {lib}")
    if uses(e.jax + e.jax_by_hand, "np") or uses(e.jax + e.jax_by_hand, "torch"):
        errs.append("JAX versions must not use NumPy or PyTorch")
    if uses(e.numpy, "jnp") or uses(e.numpy, "torch"):
        errs.append("NumPy version must use only NumPy")
    return errs


# ---------------------------------------------------------------------------------------------- showing
def imports(variant: str, src: str) -> str:
    if variant == "python":
        return "import math" if "math." in src else ""
    if variant == "numpy":
        return "import numpy as np"
    if variant in ("jax", "jax_by_hand"):
        lines = ["import jax" if "jax." in src.replace("jax.numpy", "") else "",
                 "import jax.numpy as jnp" if "jnp." in src else "", "import optax" if "optax." in src else ""]
        return "\n".join(x for x in lines if x)
    lines = ["import torch", "import torch.nn.functional as F" if "F." in src else ""]
    return "\n".join(x for x in lines if x)


def code_block(e: Entry, variant: str) -> str:
    src = code(getattr(e, variant))
    head = imports(variant, src)
    return f"```python\n{head + chr(10) * 2 if head else ''}{src}\n```"


def math_md(e: Entry) -> str:
    parts = [f"**{e.name}.** {e.plain}", f"$$\n{e.math}\n$$"]
    if e.paper:
        parts.append(f"**In papers:** {e.paper}")
    parts.append(f"**Shapes:** `{e.shapes}`")
    if e.einsum:
        parts.append(f"**Index form:** `einsum({e.einsum}, ...)`")
    if e.apis:
        parts.append("**JAX names:** " + ", ".join(f"`{x}`" for x in e.apis))
    return "\n\n".join(parts)


def notes_md(e: Entry) -> str:
    return "\n".join(f"- {n}" for n in e.notes)


def tabs(mo, name: str, solves: tuple[str, ...] | str = ()):
    """Show one entry: Math | Python | NumPy | JAX | PyTorch tabs (stacked in the book), then where they differ.

    `solves` names this lesson's exercise functions the entry gives away (e.g. "adam_step"). Then the tabs sit
    after that exercise and start collapsed, labelled as containing its solution.
    """
    e = BY_NAME[name]
    solves = (solves,) if isinstance(solves, str) else tuple(solves)
    panes = {"Math": mo.md(math_md(e))}
    for v in ("python", "numpy", "jax", "torch"):
        body = code_block(e, v)
        if v == "jax" and e.jax_by_hand:
            body += "\n\nThe formula, by hand:\n\n" + code_block(e, "jax_by_hand")
        panes[LABELS[v]] = mo.md(body)
    diff = mo.callout(mo.md("**Where they differ**\n\n" + notes_md(e)), kind="info") if e.notes else None
    title = f"Math ↔ code: {e.name}"
    spoils = f" (contains the solution to {', '.join(f'`{f}`' for f in solves)})" if solves else ""
    if os.environ.get("LEARNJAX_BOOK"):  # the book can't show tabs or accordions: stack them, labelled
        items = [mo.md(f"**{title}**{spoils}")]
        for label, pane in panes.items():
            items += ([pane] if label == "Math" else [mo.md(f"*{label}*"), pane])
        return mo.vstack(items + ([diff] if diff else []))
    body = [mo.ui.tabs(panes)] + ([diff] if diff else [])
    if solves:  # collapsed: try the exercise first
        return mo.accordion({f"{title}{spoils}": mo.vstack(body)})
    return mo.vstack([mo.md(f"**{title}**")] + body)


def placements(root=None) -> list[tuple[str, str, int, tuple[str, ...]]]:
    """Every `rosetta.tabs(mo, "name", solves=...)` call in the lessons: (module slug, name, line, solves)."""
    import ast
    from pathlib import Path

    root = Path(root) if root else Path(__file__).resolve().parents[2]
    out = []
    for path in sorted(root.glob("modules/*/lesson.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "tabs"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "rosetta"):
                kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
                s = kw.get("solves", ())
                out.append((path.parent.name, ast.literal_eval(node.args[1]), node.lineno,
                            (s,) if isinstance(s, str) else tuple(s)))
    return out
