"""Write docs/PAPERS.md (the course reading list) from learnjax/papers.py.

    containers/run.sh python tools/gen_papers.py
"""
from pathlib import Path

from learnjax.papers import BOOKS, PAPERS, SERIES, all_companions
from learnjax.syllabus import MODULES, PARTS, label

out = ["# Seminal papers, by module", "",
       ("The papers behind each module's ideas, with a line on why each matters. Read a module's papers "
        "after its lesson, once the ideas have clicked. Every link is checked by `tools/check_papers.py`."), ""]
part = None
for n, _slug, title, p in MODULES:
    if p != part:
        part = p
        out += [f"## Part {p} — {PARTS[p]}", ""]
    out += [f"### {label(_slug)} · {title}", ""]
    out += [f"- **[{x.title}]({x.url})** — {x.authors}, {x.year}. {x.why}" for x in PAPERS.get(n, [])]
    for series, lectures in all_companions(n).items():
        name = series.rsplit(" (", 1)[0]
        out += [f"- *Companion ([{name}]({SERIES[series]})):* [{v.title}]({v.url}). {v.why}" for v in lectures]
    out.append("")
out += ["## Books", "", "For a second pass, or a different angle. Most are free to read online from their authors; each entry says so.", ""]
out += [f"- **[{b.title}]({b.url})** — {b.authors}, {b.year}. {b.why} *(Modules {b.modules}.)*" for b in BOOKS]
out.append("")
Path("docs/PAPERS.md").write_text("\n".join(out))
print(f"wrote docs/PAPERS.md ({sum(map(len, PAPERS.values()))} entries)")
