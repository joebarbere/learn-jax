"""Write docs/GLOSSARY.md (the course glossary) from learnjax/glossary.py.

    python3 tools/gen_glossary.py
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Load by file path: learnjax/__init__.py is light, but this keeps the tool runnable on a bare host Python.
_spec = importlib.util.spec_from_file_location("glossary", ROOT / "learnjax" / "glossary.py")
_glossary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_glossary)
TERMS = _glossary.TERMS


def letter(term: str) -> str:
    c = term.lstrip("[*_`(")[:1].upper()
    return c if "A" <= c <= "Z" else "#"


def entry(t) -> str:
    where = f"Module {t.module}" + (f"; also {', '.join(t.also)}" if t.also else "")
    origin = f" Origin: {t.origin}." if t.origin else ""
    return f"- **{t.term}** — {t.definition} *({where}.)*{origin}"


groups: dict[str, list] = {}
for t in sorted(TERMS, key=lambda t: t.term.lower()):
    groups.setdefault(letter(t.term), []).append(t)
order = sorted(groups, key=lambda k: (k != "#", k))
anchor = {k: ("symbols" if k == "#" else k.lower()) for k in order}
title = {k: ("Numbers & symbols" if k == "#" else k) for k in order}

out = ["# Glossary", "",
       (f"{len(TERMS)} terms from the course, each with the module that teaches it and the others where it "
        "matters; read a term's module section for the full story."), "",
       " · ".join(f"[{title[k] if k != '#' else '#'}](#{anchor[k]})" for k in order), ""]
for k in order:
    out += [f'<a id="{anchor[k]}"></a>', "", f"## {title[k]}", ""]
    out += [entry(t) for t in groups[k]]
    out.append("")
(ROOT / "docs" / "GLOSSARY.md").write_text("\n".join(out))
print(f"wrote docs/GLOSSARY.md ({len(TERMS)} terms)")
