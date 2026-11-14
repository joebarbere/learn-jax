"""Insert a 'Seminal papers' cell before a lesson's Further reading cell (idempotent).

    python3 tools/add_papers_cell.py modules/03-autodiff/lesson.py
"""
import sys
from pathlib import Path

CELL = '''@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md({n}))
    return


'''

for arg in sys.argv[1:]:
    path = Path(arg)
    src = path.read_text()
    if "papers_md(" in src:
        print(f"skip (already has papers): {path}")
        continue
    n = int(path.parent.name[:2])
    idx = src.find("## Further reading")
    if idx < 0:
        print(f"NO Further reading cell: {path}")
        continue
    start = src.rfind("@app.cell", 0, idx)
    path.write_text(src[:start] + CELL.format(n=n) + src[start:])
    print(f"added papers cell (module {n}): {path}")
