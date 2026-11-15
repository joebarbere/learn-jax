"""Write docs/LINEAGE.md (the family tree and the taxonomy) from learnjax/lineage.py.

    containers/run.sh python tools/gen_lineage.py      (or: make lineage, which checks first)
"""
from pathlib import Path

from learnjax import lineage as L
from learnjax.papers import PAPERS

paper_of = {p.url: p for ps in PAPERS.values() for p in ps}
models, blocks = L.by_id(), L.blocks_by_id()
parents = {m.id: m.parents for m in L.MODELS}


def cite(url: str) -> str:
    p = paper_of.get(url)
    return f"[{p.authors}, {p.year}]({url})" if p else f"[{L.SOURCES.get(url, 'source')}]({url})"


def ancestors(mid: str) -> set[str]:
    seen, stack = set(), list(parents[mid])
    while stack:
        x = stack.pop()
        if x not in seen:
            seen.add(x)
            stack.extend(parents[x])
    return seen


def added(mid: str) -> list[str]:
    inherited = {b for a in ancestors(mid) for b in models[a].uses}
    return [b for b in models[mid].uses if b not in inherited]


out = ["# The family tree: models and their building blocks", "",
       ("Generated from `learnjax/lineage.py` by `tools/gen_lineage.py`; module 00b (Section 12) draws it as an "
        "interactive graph. `make lineage` checks the registry's rules, the sources' arXiv titles, and every open "
        "model's Hugging Face `config.json` against the blocks listed here."), "",
       f"{len(L.MODELS)} models, {len(L.BLOCKS)} building blocks in {len(L.SLOTS)} slots.", "",
       "## The taxonomy: building blocks by slot", "",
       ("*Introduced in* is the paper the block comes from. *First model* is set when that paper's own model is "
        "in the registry. *Module* is where this course builds or discusses it."), ""]
for slot in L.SLOTS:
    out += [f"### {slot}", "", "| block | what it does | introduced in | first model | used by | module |",
            "|---|---|---|---|---|---|"]
    for b in (b for b in L.BLOCKS if b.slot == slot):
        users = sorted((m for m in L.MODELS if b.id in m.uses), key=lambda m: (m.year, m.name))
        out.append(f"| **{b.name}** | {b.what} | {cite(b.paper)} | "
                   f"{models[b.first_model].name if b.first_model else '—'} | "
                   f"{', '.join(m.name for m in users)} | {b.module or '—'} |")
    out.append("")
out += ["## The lineage: models by family", "",
        ("*New to its lineage* lists the blocks a model uses that none of its ancestors use (module 00b's "
         "`new_blocks`). That is not the same as inventing them: a block can enter a lineage long after its paper."),
        ""]
for fam in L.FAMILIES:
    out += [f"### {fam}", "", "| model | year | by | builds on | new to its lineage | why it matters | module |",
            "|---|---|---|---|---|---|---|"]
    for m in sorted((m for m in L.MODELS if m.family == fam), key=lambda m: (m.year, m.name)):
        new = ", ".join(blocks[b].name for b in added(m.id)) or "—"
        out.append(f"| [{m.name}]({m.source}) | {m.year} | {m.org} | "
                   f"{', '.join(models[p].name for p in m.parents) or '—'} | {new} | {m.note} | {m.module or '—'} |")
    out.append("")
Path("docs/LINEAGE.md").write_text("\n".join(out))
print(f"wrote docs/LINEAGE.md ({len(L.MODELS)} models, {len(L.BLOCKS)} blocks)")
