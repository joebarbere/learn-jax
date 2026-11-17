"""Every Rosetta entry: all versions compute the same thing, the registry follows its rules, and every math API
the lessons use has an entry.

The PyTorch versions run only where torch is installed (the learn-jax-torch image); `make check` in the JAX
image skips them.
"""
import importlib.util
import re

import pytest

from learnjax import rosetta
from learnjax.rosetta import coverage

HAS_TORCH = importlib.util.find_spec("torch") is not None
NAMES = [e.name for e in rosetta.ENTRIES]


@pytest.mark.parametrize("name", NAMES)
def test_versions_agree(name):
    rosetta.check(rosetta.BY_NAME[name])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch versions run in the learn-jax-torch image")
@pytest.mark.parametrize("name", NAMES)
def test_torch_agrees(name):
    rosetta.check(rosetta.BY_NAME[name], ["torch"])


def test_registry_rules():
    assert len(set(NAMES)) == len(NAMES), "entry names must be unique"
    errs = [f"{e.name}: {msg}" for e in rosetta.ENTRIES for msg in rosetta.rule_errors(e)]
    assert not errs, "\n".join(errs)


def test_latex_katex_safe():
    # KaTeX rejects an operator name as a bare sub/superscript (`\eta_\max`) and shows the whole formula as
    # source; plain TeX accepts it, so only the rendered book would show the bug. Brace it: `\eta_{\max}`.
    bad = re.compile(r"[_^]\\(max|min|log|exp|sum|prod|arg|sin|cos|tan|det|lim|sup|inf|dim|ker|deg|gcd)\b")
    errs = [f"{e.name}.{field}: {m.group()}" for e in rosetta.ENTRIES for field in ("math", "paper")
            for m in bad.finditer(getattr(e, field))]
    assert not errs, "brace these subscripts: " + ", ".join(errs)


def test_every_math_api_has_an_entry():
    covered = {a for e in rosetta.ENTRIES for a in e.apis}
    missing = {n: m for n, m in coverage.used().items() if n not in covered and n not in coverage.NON_MATH}
    assert not missing, "math APIs with no Rosetta entry (module that first uses them):\n" + "\n".join(
        f"  {m}: {n}" for n, m in sorted(missing.items(), key=lambda kv: (kv[1], kv[0])))


def test_apis_are_real_and_unique():
    used, seen = coverage.used(), {}
    for e in rosetta.ENTRIES:
        for a in e.apis:
            assert a in used, f"{e.name}: {a} is not used in any lesson (typo, or drop it)"
            assert a not in coverage.NON_MATH, f"{e.name}: {a} is listed as non-math"
            assert a not in seen, f"{a} is claimed by both {seen[a]!r} and {e.name!r}"
            seen[a] = e.name


def test_tabs_render():
    import marimo as mo

    for e in rosetta.ENTRIES:
        assert rosetta.tabs(mo, e.name) is not None


def test_tabs_collapse_when_they_solve_an_exercise():
    import marimo as mo

    out = rosetta.tabs(mo, rosetta.ENTRIES[0].name, solves="some_exercise")
    assert "some_exercise" in out.text


def test_lesson_placements():
    """Each entry shows up in some lesson, once per lesson; tabs that give away an exercise come after it."""
    import re
    from collections import Counter

    found = rosetta.placements()
    errs = [f"{slug}:{line}: no entry named {name!r}" for slug, name, line, _ in found if name not in rosetta.BY_NAME]
    errs += [f"{slug}: {name!r} placed {n} times" for (slug, name), n in
             Counter((s, n) for s, n, _, _ in found).items() if n > 1]
    placed = {name for _, name, _, _ in found}
    errs += [f"{e.name!r} is in no lesson" for e in rosetta.ENTRIES if e.name not in placed]
    for slug, name, line, solves in found:
        src = (coverage.ROOT / "modules" / slug / "lesson.py").read_text().splitlines()
        for fn in solves:
            first = next((i + 1 for i, s in enumerate(src) if re.search(rf"\bex\.{fn}\b|['\"]{fn}['\"]", s)), None)
            if first is None or first > line:
                errs.append(f"{slug}:{line}: {name!r} solves {fn} but comes before the lesson tries {fn}")
    assert not errs, "\n".join(errs)
