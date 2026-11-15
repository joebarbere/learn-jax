"""Find the marimo dedent pitfall (docs/AUTHORING.md): a `mo.md(f\"\"\"...\"\"\")` nested inside an if/for/def in a
cell (so its text is still indented after marimo dedents the cell body) with an interpolation that starts a line. If that value is multi-line (a joined table, a list), its later lines have no indent,
the dedent fails, and the whole cell renders as code blocks. Static, so it flags *candidates*: a value
that is always one line is harmless. Fix by indenting the value, e.g. `{_rows.replace(chr(10), chr(10) + "    ")}`,
or by writing that markdown without indentation.

    python3 tools/check_md_indent.py modules/*/lesson.py      # exit 1 if anything is flagged
"""
import ast
import re
import sys

_JOIN = re.compile(r"""(["'])\\n\1\s*\.join|chr\(10\)\s*\.join""")


def multiline(src, fn, value) -> bool:
    """Is this interpolated value built by joining with a bare newline in the same cell?"""
    if isinstance(value, ast.Call) and _JOIN.search(ast.get_source_segment(src, value) or ""):
        return True
    if fn is None or not isinstance(value, ast.Name):
        return False
    for n in ast.walk(fn):
        if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == value.id for t in n.targets)
                and _JOIN.search(ast.get_source_segment(src, n.value) or "")):
            return True
    return False


def candidates(path: str):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    cell_of = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for n in ast.walk(fn):
                cell_of.setdefault(id(n), fn)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "md" and node.args):
            continue
        arg = node.args[0]
        if not isinstance(arg, ast.JoinedStr):
            continue
        literal = "".join(v.value for v in arg.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
        lines = [ln for ln in literal.split("\n") if ln.strip()]
        # marimo dedents each cell body by 4 before running it, so markdown at the cell's top level ends up
        # flush-left at runtime; only text indented deeper (inside an if/for/def) keeps an indent to break.
        if not lines or min(len(ln) - len(ln.lstrip(" ")) for ln in lines) <= 4:
            continue
        for prev, val in zip(arg.values, arg.values[1:], strict=False):
            if (isinstance(val, ast.FormattedValue) and isinstance(prev, ast.Constant)
                    and isinstance(prev.value, str) and prev.value.split("\n")[-1].strip() == ""
                    and "\n" in prev.value):
                expr = ast.get_source_segment(src, val.value) or "?"
                if "replace" not in expr and multiline(src, cell_of.get(id(node)), val.value):
                    yield node.lineno, expr


if __name__ == "__main__":
    found = 0
    for p in sys.argv[1:]:
        for line, expr in candidates(p):
            print(f"{p}:{line}: {{{expr}}} starts a line in indented markdown")
            found += 1
    print(f"{found} candidate(s)")
    sys.exit(1 if found else 0)
