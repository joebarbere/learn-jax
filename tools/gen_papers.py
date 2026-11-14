"""Write docs/PAPERS.md (the course reading list) from learnjax/papers.py.

    containers/run.sh python tools/gen_papers.py
"""
from pathlib import Path

from learnjax.papers import PAPERS
from learnjax.syllabus import MODULES, PARTS

out = ["# Seminal papers, by module", "",
       ("The papers behind each module's ideas, with a line on why each matters. Read a module's papers "
        "after its lesson, once the ideas have clicked. Every link is checked by `tools/check_papers.py`."), ""]
part = None
for n, _slug, title, p in MODULES:
    if p != part:
        part = p
        out += [f"## Part {p} — {PARTS[p]}", ""]
    out += [f"### {n:02d} · {title}", ""]
    out += [f"- **[{x.title}]({x.url})** — {x.authors}, {x.year}. {x.why}" for x in PAPERS.get(n, [])]
    out.append("")
Path("docs/PAPERS.md").write_text("\n".join(out))
print(f"wrote docs/PAPERS.md ({sum(map(len, PAPERS.values()))} entries)")
